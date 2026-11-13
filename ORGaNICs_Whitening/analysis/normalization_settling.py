'''
normalization_settling.py

How long does the normalization circuit of V1Dynamics_Surround (y, u, a) take to reach steady
state after a stimulus onset, and which (tau_u, tau_a) let every neuron settle within 30 ms?

Isolation: adaptation is switched off by holding the interneuron gains at exactly zero
(g_cRF = g_surround = 0 initially and tau_g = inf, so dg/dt = 0). The gain-feedback term of
Eq. dy is then identically zero and y, u, a evolve as single-layer ORGaNICs. v and mu are still
integrated by the model but do not feed back. No model code is changed: the right-hand side is
V1Dynamics_Surround._derivatives and the integrator is its own run_simulation (fixed-step RK4).

Equations integrated (docs/equations.md, with g = 0, W_yy = I):
    Eq. dy:  tau_y dy/dt = -y + beta z + W_yy floor(y) / (1 + floor(a))
    Eq. du:  tau_u du/dt = -u + (sigma/2)^2 + N (floor(y)^2 * floor(u))
    Eq. da:  tau_a da/dt = -a + floor(u)^(1/2) (1 + a)

Analytic fixed point (exact here: z >= 0, W_yy = I, N all ones so u and a are uniform, g = 0):
    u* = (sigma^2 + sum z^2) / 4,   a* = sqrt(u*) / (1 - sqrt(u*)),   y* = z / sqrt(sigma^2 + sum z^2)
    i.e. the normalized state y*^2 = z^2 / (sigma^2 + N z^2) of Eq. y_steady with z' = z.

Stimulus: one full-field grating (StimulusGenerator.oriented_drive, theta = pi/2, contrast 1,
routed to the cRF and every surround set), switched on at t = 0 after a PRE_DURATION blank.
Initial state: the blank (z = 0) steady state, y = 0, u = sigma^2/4, a = sqrt(u)/(1 - sqrt(u)).

Settling time of a population x in {y, u, a}: the first time after which
    e_x(t) = max_i |x_i(t) - x_i*| / max_i |x_i*|
stays at or below SETTLE_TOL (1 %) for the rest of the simulated window. A run "settles within
30 ms" if max(T_y, T_u, T_a) <= SETTLE_DEADLINE and its fixed point is linearly stable.

Outputs (results/normalization_settling/): normalization_settling.npz (time courses, sweep grid,
settling times, Jacobian spectra) and params.json (all parameters, git commit, seed, date).
Figures are drawn by figures/fig_norm_timecourse.py, fig_norm_discrepancy.py and fig_norm_tau_sweep.py.

Run from ORGaNICs_Whitening/:  python analysis/normalization_settling.py [--smoke]
'''

import os
import sys
import io
import json
import time
import argparse
import datetime
import contextlib
import subprocess
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.environ.setdefault('TQDM_DISABLE', '1')   # run_simulation's progress bar, once per sweep point

import numpy as np
from simulation import Frame, V1Dynamics_Surround
from tunings import V1Tunings
from stimuli import StimulusGenerator

# ---- Population and stimulus (same layout as adapt_single_stimulus.py) ----
N_RF        = 13
N_SETS      = 6
FRAME_PATH  = os.path.join(REPO_ROOT, "data/frames/N13_mercedes_K182_Frame.csv")
TARGET_COV_PATH = os.path.join(REPO_ROOT, "data/target_covs/uniform_target_covariance_low_c.csv")
TUNING_WIDTH = 0.75
STIM_THETA   = np.pi / 2
STIM_CONTRAST = 1.0
STIM_ROUTE   = 'adapt CRF and surround'    # grating covers the cRF and every surround RF

# ---- Durations (ms) ----
DT_DEFAULT      = 0.1      # V1Dynamics_Surround's default step
PRE_DURATION    = 5.0      # blank before onset (checks the initial state is a fixed point)
TRACE_DURATION  = 300.0    # post-onset window for the two time courses (Fig. 1, 2)
SWEEP_DURATION  = 60.0     # post-onset window for each sweep point
SETTLE_TOL      = 0.01     # 1 % band
SETTLE_DEADLINE = 30.0     # ms
MIN_DWELL       = 20.0     # ms a run must stay in the band before the window ends to count as settled
DT_TAU_RATIO    = 4.0      # dt = min(DT_DEFAULT, min(tau_y, tau_u, tau_a) / DT_TAU_RATIO)
ROBUST_CONTRASTS = [0.125, 0.25, 0.5, 1.0]   # contrasts for the tuned point's robustness check
SEED            = 0        # nothing here is random; set for the record

# ---- Sweep grids: half-octave steps that contain the code defaults (tau_u = 30, tau_a = 0.2)
# and the docs/equations.md (tau_u, tau_a) = (15, 0.1); tau_y stays at the code's 0.4 ms, so
# the docs' full parameter set (tau_y = 0.2) is not part of the sweep ----
TAU_U_GRID = 30.0 * 2.0 ** (np.arange(-12, 3) / 2)    # 0.47 ... 60 ms (15 values)
TAU_A_GRID = 0.2 * 2.0 ** (np.arange(-4, 9) / 2)      # 0.05 ... 3.2 ms (13 values)

OUT_DIR = os.path.join(REPO_ROOT, "results", "normalization_settling")


# ------------------------------------------------------------------------------------------
# Model set-up
# ------------------------------------------------------------------------------------------

def build_model():
    '''V1Dynamics_Surround with adaptation frozen off (g = 0, tau_g = inf).'''
    with contextlib.redirect_stdout(io.StringIO()):
        frame = Frame(FRAME_PATH)
        dyn = V1Dynamics_Surround(V1Tunings(N=N_RF), frame, dt=DT_DEFAULT, N_RF=N_RF,
                                  N_SETS=N_SETS, target_covariance_path=TARGET_COV_PATH)
    dyn.tau_g = np.inf          # Eq. dg: dg/dt = (...)/inf = 0, so g stays at its initial 0
    return dyn


def build_stimulus():
    stim_gen = StimulusGenerator(N=N_RF, N_RF=N_RF, N_SETS=N_SETS, tuning_width=TUNING_WIDTH)
    return stim_gen.oriented_drive(STIM_THETA, STIM_CONTRAST, STIM_ROUTE)


def analytic_fixed_point(dyn, z):
    '''Exact fixed point of Eqs. dy/du/da with g = 0 (see module docstring).'''
    norm_sq = dyn.sigma**2 + dyn.N_matrix[0] @ (z**2)     # row of the all-ones N: sum z^2
    u_ss = norm_sq / 4
    a_ss = np.sqrt(u_ss) / (1 - np.sqrt(u_ss))
    y_ss = z / np.sqrt(norm_sq)
    assert np.allclose(dyn.N_matrix, 1.0), "closed form assumes the all-ones normalization matrix"
    assert np.allclose(dyn.W_yy, np.eye(dyn.W_yy.shape[0])), "closed form assumes W_yy = I"
    assert np.all(z >= 0), "closed form assumes nonnegative drive (z' = z >= 0)"
    assert np.sqrt(u_ss) < 1, "a* = sqrt(u)/(1 - sqrt(u)) requires u* < 1"
    return y_ss, np.full(N_RF * N_SETS, u_ss), np.full(N_RF * N_SETS, a_ss)


def blank_state(dyn):
    '''Steady state for z = 0: y = 0, u = sigma^2/4, a = sqrt(u)/(1 - sqrt(u)); g = v = mu = 0.'''
    state = np.zeros(dyn.state_size)
    u0 = (dyn.sigma / 2) ** 2
    a0 = np.sqrt(u0) / (1 - np.sqrt(u0))
    state[dyn._off_u_cRF:dyn._off_u_cRF + 2 * N_RF] = u0      # u_cRF, u_surround
    state[dyn._off_a_cRF:dyn._off_a_cRF + 2 * N_RF] = a0      # a_cRF, a_surround
    return state


def set_taus(dyn, tau_u, tau_a):
    dyn.tau_u, dyn.tau_a = float(tau_u), float(tau_a)
    dyn.dt = min(DT_DEFAULT, min(dyn.tau_y, dyn.tau_u, dyn.tau_a) / DT_TAU_RATIO)


# ------------------------------------------------------------------------------------------
# Integration (model's own RK4 run_simulation)
# ------------------------------------------------------------------------------------------

def simulate_onset(dyn, z, duration, pre_duration=PRE_DURATION, dt=None):
    '''Blank for pre_duration, then z held for duration (ms), from the blank steady state.
    Returns t (ms, onset at 0) and per-neuron y, u, a histories, each (N_TOT, n_steps).
    u and a are expanded from the model's shared cRF/surround copies to all N_TOT neurons.'''
    if dt is not None:
        dyn.dt = dt
    n_pre = int(round(pre_duration / dyn.dt))
    n_on = int(round(duration / dyn.dt))
    stream = np.concatenate([np.zeros((z.size, n_pre)), np.tile(z[:, None], n_on)], axis=1)
    with contextlib.redirect_stdout(io.StringIO()):
        out = dyn.run_simulation(stream, initial_state=blank_state(dyn))
    y_hist, u_cRF, u_sur, a_cRF, a_sur, g_cRF, g_sur = out[:7]
    assert np.all(g_cRF == 0) and np.all(g_sur == 0), "gains moved: adaptation was not isolated"
    u_hist = np.concatenate([u_cRF] + [u_sur] * (N_SETS - 1), axis=0)
    a_hist = np.concatenate([a_cRF] + [a_sur] * (N_SETS - 1), axis=0)
    t = (np.arange(1, n_pre + n_on + 1) - n_pre) * dyn.dt     # time at the end of each step
    return t, y_hist, u_hist, a_hist


# ------------------------------------------------------------------------------------------
# Analysis
# ------------------------------------------------------------------------------------------

def linf_rel_error(x_hist, x_ss):
    '''e(t) = max_i |x_i(t) - x_i*| / max_i |x_i*|.'''
    return np.max(np.abs(x_hist - x_ss[:, None]), axis=0) / np.max(np.abs(x_ss))


def rate_rel_error(y_hist, y_ss):
    '''Relative L2 discrepancy of firing rates from the normalized state:
    || floor(y(t))^2 - y*^2 ||_2 / || y*^2 ||_2.'''
    r, r_ss = np.maximum(y_hist, 0) ** 2, y_ss ** 2
    return np.linalg.norm(r - r_ss[:, None], axis=0) / np.linalg.norm(r_ss)


def settling_time(t, err, tol=SETTLE_TOL, min_dwell=MIN_DWELL):
    '''First post-onset time after which err stays <= tol for the rest of the window; nan if the
    run is still outside at the end, or entered the band less than min_dwell ms before the end
    (too late to tell whether it stays).'''
    post = t > 0
    t, err = t[post], err[post]
    if not np.all(np.isfinite(err)) or err[-1] > tol:
        return np.nan
    outside = np.nonzero(err > tol)[0]
    T = 0.0 if outside.size == 0 else t[outside[-1] + 1]
    return T if t[-1] - T >= min_dwell else np.nan


def fast_subsystem_jacobian(dyn, z, y_ss, u_ss, a_ss, eps=1e-7):
    '''Central-difference Jacobian of (y, u_cRF, u_surround, a_cRF, a_surround) at the analytic
    fixed point, from the model's own _derivatives (g = 0, so v/mu don't enter). Returns its
    eigenvalues (1/ms) and the max residual |d state/dt| at the fixed point (should be ~0).'''
    N_TOT = N_RF * N_SETS
    state = np.zeros(dyn.state_size)
    state[:N_TOT] = y_ss
    state[dyn._off_u_cRF:dyn._off_u_cRF + 2 * N_RF] = u_ss[0]
    state[dyn._off_a_cRF:dyn._off_a_cRF + 2 * N_RF] = a_ss[0]
    idx = np.r_[0:N_TOT, dyn._off_u_cRF:dyn._off_u_cRF + 4 * N_RF]   # y, u_cRF, u_sur, a_cRF, a_sur
    f = lambda s: dyn._derivatives(s, z, 0.0)[idx]
    residual = np.max(np.abs(f(state)))
    J = np.zeros((idx.size, idx.size))
    for j, k in enumerate(idx):
        dp, dm = state.copy(), state.copy()
        dp[k] += eps
        dm[k] -= eps
        J[:, j] = (f(dp) - f(dm)) / (2 * eps)
    return np.linalg.eigvals(J), residual


def symmetric_subspace_eigs(dyn, z, y_ss, u_ss, a_ss, eps=1e-7):
    '''Eigenvalues (1/ms) of the Jacobian restricted to the invariant subspace in which u and a
    are uniform across neurons (variables: y, scalar u, scalar a). With the all-ones N every
    u row sees the same pool, so this subspace is invariant and a uniform stimulus onset stays
    in it. Its slowest eigenvalue is the mode that sets the settling time. The full Jacobian
    also has the u_i - u_j difference modes at exactly -1/tau_u, which the onset never excites.'''
    N_TOT = N_RF * N_SETS
    base = np.zeros(dyn.state_size)
    base[:N_TOT] = y_ss
    u_sl = slice(dyn._off_u_cRF, dyn._off_u_cRF + 2 * N_RF)
    a_sl = slice(dyn._off_a_cRF, dyn._off_a_cRF + 2 * N_RF)
    base[u_sl], base[a_sl] = u_ss[0], a_ss[0]

    def f(s):
        d = dyn._derivatives(s, z, 0.0)
        return np.concatenate([d[:N_TOT], [d[dyn._off_u_cRF]], [d[dyn._off_a_cRF]]])

    J = np.zeros((N_TOT + 2, N_TOT + 2))
    for j in range(N_TOT + 2):
        dp, dm = base.copy(), base.copy()
        if j < N_TOT:
            dp[j] += eps
            dm[j] -= eps
        else:
            sl = u_sl if j == N_TOT else a_sl
            dp[sl] += eps
            dm[sl] -= eps
        J[:, j] = (f(dp) - f(dm)) / (2 * eps)
    return np.linalg.eigvals(J)


def perturbation_test(dyn, z, y_ss, u_ss, a_ss, rel=0.05, duration=None, seed=SEED):
    '''Start at the fixed point with a random +-rel relative perturbation on y, u, a and check the
    state returns: returns the final max relative error over y, u, a.'''
    rng = np.random.default_rng(seed)
    N_TOT = N_RF * N_SETS
    state = np.zeros(dyn.state_size)
    state[:N_TOT] = y_ss * (1 + rel * rng.uniform(-1, 1, N_TOT))
    for off, val in [(dyn._off_u_cRF, u_ss[0]), (dyn._off_u_surround, u_ss[0]),
                     (dyn._off_a_cRF, a_ss[0]), (dyn._off_a_surround, a_ss[0])]:
        state[off:off + N_RF] = val * (1 + rel * rng.uniform(-1, 1, N_RF))
    if duration is None:
        duration = 10 * max(dyn.tau_u, dyn.tau_a, dyn.tau_y)
    n = int(round(duration / dyn.dt))
    with contextlib.redirect_stdout(io.StringIO()):
        dyn.run_simulation(np.tile(z[:, None], n), initial_state=state)
    s = dyn.unpack_state(dyn.last_state)
    u_all = np.concatenate([s['u_cRF'], s['u_surround']])
    a_all = np.concatenate([s['a_cRF'], s['a_surround']])
    return max(np.max(np.abs(s['y'] - y_ss)) / np.max(y_ss),
               np.max(np.abs(u_all - u_ss[0])) / u_ss[0],
               np.max(np.abs(a_all - a_ss[0])) / a_ss[0])


def run_point(dyn, z, fp, tau_u, tau_a, duration):
    '''Settling times (T_y, T_u, T_a), the fixed point's max Re(eigenvalue) (full Jacobian:
    stability) and the slowest eigenvalue on the symmetric subspace (settling mode).'''
    set_taus(dyn, tau_u, tau_a)
    y_ss, u_ss, a_ss = fp
    with np.errstate(over='ignore', invalid='ignore'):
        t, y, u, a = simulate_onset(dyn, z, duration)
        T = [settling_time(t, linf_rel_error(x, xs)) for x, xs in [(y, y_ss), (u, u_ss), (a, a_ss)]]
    eig, _ = fast_subsystem_jacobian(dyn, z, *fp)
    sym = symmetric_subspace_eigs(dyn, z, *fp)
    return T, np.max(eig.real), sym[np.argmax(sym.real)], dyn.dt


def trace(dyn, z, fp, tau_u, tau_a, duration=TRACE_DURATION):
    '''Full time course for one (tau_u, tau_a): population means, error traces, settling times.'''
    set_taus(dyn, tau_u, tau_a)
    y_ss, u_ss, a_ss = fp
    t, y, u, a = simulate_onset(dyn, z, duration)
    err = {k: linf_rel_error(x, xs) for k, x, xs in [('y', y, y_ss), ('u', u, u_ss), ('a', a, a_ss)]}
    rate_err = rate_rel_error(y, y_ss)
    T = {k: settling_time(t, e) for k, e in err.items()}
    eig, residual = fast_subsystem_jacobian(dyn, z, *fp)
    sym = symmetric_subspace_eigs(dyn, z, *fp)
    return dict(t=t, y_mean=y.mean(0), u_mean=u.mean(0), a_mean=a.mean(0),
                err_y=err['y'], err_u=err['u'], err_a=err['a'], rate_err=rate_err,
                T_y=T['y'], T_u=T['u'], T_a=T['a'], T_rate=settling_time(t, rate_err),
                eig=eig, sym_eig=sym, slow_sym=sym[np.argmax(sym.real)], fp_residual=residual, dt=dyn.dt, tau_u=tau_u, tau_a=tau_a)


def git_commit():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return 'unknown'


def git_dirty():
    try:
        return bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO_ROOT, text=True).strip())
    except Exception:
        return None


def main(smoke=False):
    np.random.seed(SEED)
    dyn = build_model()
    z = build_stimulus()
    fp = analytic_fixed_point(dyn, z)
    tau_y, tau_u0, tau_a0 = dyn.tau_y, dyn.tau_u, dyn.tau_a
    print(f"tau_y = {tau_y} ms (fixed), defaults tau_u = {tau_u0} ms, tau_a = {tau_a0} ms")
    print(f"sum z^2 = {np.sum(z**2):.4f}; u* = {fp[1][0]:.5f}, a* = {fp[2][0]:.5f}, "
          f"mean y* = {fp[0].mean():.5f}")

    tau_u_grid = TAU_U_GRID[::3] if smoke else TAU_U_GRID
    tau_a_grid = TAU_A_GRID[::3] if smoke else TAU_A_GRID
    sweep_duration = 40.0 if smoke else SWEEP_DURATION
    trace_duration = 60.0 if smoke else TRACE_DURATION

    # ---- 1. Original time constants ----
    t0 = time.time()
    orig = trace(dyn, z, fp, tau_u0, tau_a0, trace_duration)
    print(f"[original] T_y = {orig['T_y']:.2f}, T_u = {orig['T_u']:.2f}, T_a = {orig['T_a']:.2f} ms "
          f"(rates 1 %: {orig['T_rate']:.2f} ms); max Re(lambda) = {orig['eig'].real.max():.4g} /ms, "
          f"slowest symmetric mode = {orig['slow_sym']:.4g} /ms ({time.time() - t0:.1f} s)")

    # ---- 2. Sweep (tau_u, tau_a) ----
    T_sweep = np.full((len(tau_u_grid), len(tau_a_grid), 3), np.nan)
    max_re = np.full((len(tau_u_grid), len(tau_a_grid)), np.nan)
    dt_sweep = np.full_like(max_re, np.nan)
    slow_sym = np.full(max_re.shape, np.nan, dtype=complex)
    t0 = time.time()
    for i, tu in enumerate(tau_u_grid):
        for j, ta in enumerate(tau_a_grid):
            T_sweep[i, j], max_re[i, j], slow_sym[i, j], dt_sweep[i, j] = run_point(
                dyn, z, fp, tu, ta, sweep_duration)
    print(f"sweep: {T_sweep.shape[0] * T_sweep.shape[1]} points in {time.time() - t0:.1f} s")
    T_max = np.max(T_sweep, axis=2)                        # nan if any population unsettled
    feasible = (T_max <= SETTLE_DEADLINE) & (max_re < 0)

    # Pick the feasible point closest to the original time constants (log distance) whose 8 grid
    # neighbours are also feasible. Settling is not monotonic in (tau_u, tau_a): isolated points
    # settle fast because the onset transient excites the slowest symmetric mode with a small
    # amplitude (or the mode is oscillatory and crosses into the band at a favourable phase),
    # which is specific to this stimulus. Requiring an interior point avoids picking one of
    # those pockets. Fallbacks: any feasible point, then the fastest stable point.
    interior = np.zeros_like(feasible)
    for a in range(1, feasible.shape[0] - 1):
        for b in range(1, feasible.shape[1] - 1):
            interior[a, b] = feasible[a - 1:a + 2, b - 1:b + 2].all()
    candidates = interior if interior.any() else feasible
    if candidates.any():
        dist = np.hypot(np.log(tau_u_grid[:, None] / tau_u0), np.log(tau_a_grid[None, :] / tau_a0))
        i, j = np.unravel_index(np.argmin(np.where(candidates, dist, np.inf)), dist.shape)
    else:
        stable_T = np.where(max_re < 0, T_max, np.nan)
        if np.all(np.isnan(stable_T)):
            raise RuntimeError("no stable sweep point settles within the sweep window")
        i, j = np.unravel_index(np.nanargmin(stable_T), T_max.shape)
    tuned = trace(dyn, z, fp, tau_u_grid[i], tau_a_grid[j], trace_duration)
    print(f"[tuned] tau_u = {tau_u_grid[i]:.3g} ms, tau_a = {tau_a_grid[j]:.3g} ms: "
          f"T_y = {tuned['T_y']:.2f}, T_u = {tuned['T_u']:.2f}, T_a = {tuned['T_a']:.2f} ms")

    # ---- 3. Checks ----
    # dt convergence: settling times of the original and tuned points (and the stiffest grid
    # corner) at dt/4 vs the sweep's dt. The original point needs the long trace window to settle.
    conv = []
    for tu, ta, dur in [(tau_u0, tau_a0, trace_duration), (tau_u_grid[i], tau_a_grid[j], sweep_duration),
                        (tau_u_grid[0], tau_a_grid[0], sweep_duration)]:
        T_coarse, _, _, dt_c = run_point(dyn, z, fp, tu, ta, dur)
        set_taus(dyn, tu, ta)
        t, y, u, a = simulate_onset(dyn, z, dur, dt=dt_c / 4)
        T_fine = [settling_time(t, linf_rel_error(x, xs)) for x, xs in zip((y, u, a), fp)]
        conv.append(dict(tau_u=tu, tau_a=ta, window_ms=dur, dt=dt_c, T_dt=T_coarse, T_dt_over_4=T_fine))
        print(f"  dt check tau_u={tu:.3g}, tau_a={ta:.3g}: dt={dt_c:.4g} -> {np.round(T_coarse, 3)}, "
              f"dt/4 -> {np.round(T_fine, 3)}")
    pert = {}
    for name, (tu, ta) in [('original', (tau_u0, tau_a0)), ('tuned', (tau_u_grid[i], tau_a_grid[j]))]:
        set_taus(dyn, tu, ta)
        pert[name] = perturbation_test(dyn, z, *fp)
        print(f"  perturbation (+-5 %) {name}: final max rel. error {pert[name]:.2e}")

    # Robustness of the tuned point to stimulus contrast (the criterion was set at contrast 1).
    stim_gen = StimulusGenerator(N=N_RF, N_RF=N_RF, N_SETS=N_SETS, tuning_width=TUNING_WIDTH)
    contrast_check = []
    for c in ROBUST_CONTRASTS:
        z_c = stim_gen.oriented_drive(STIM_THETA, c, STIM_ROUTE)
        T_c, re_c, _, _ = run_point(dyn, z_c, analytic_fixed_point(dyn, z_c),
                                 tau_u_grid[i], tau_a_grid[j], sweep_duration)
        contrast_check.append(dict(contrast=c, T_y=T_c[0], T_u=T_c[1], T_a=T_c[2], max_re_eig=re_c))
        print(f"  tuned point, contrast {c}: T(y, u, a) = {np.round(T_c, 2)} ms")

    # ---- 4. Save ----
    os.makedirs(OUT_DIR, exist_ok=True)
    tag = '_smoke' if smoke else ''
    npz_path = os.path.join(OUT_DIR, f"normalization_settling{tag}.npz")
    save = dict(z=z, y_ss=fp[0], u_ss=fp[1][0], a_ss=fp[2][0],
                tau_u_grid=tau_u_grid, tau_a_grid=tau_a_grid, T_sweep=T_sweep, max_re_eig=max_re,
                dt_sweep=dt_sweep, slow_sym_eig=slow_sym, feasible=feasible, interior=interior, tuned_idx=np.array([i, j]))
    for prefix, d in [('orig', orig), ('tuned', tuned)]:
        for k, v in d.items():
            save[f"{prefix}_{k}"] = v
    np.savez_compressed(npz_path, **save)

    params = dict(
        script='analysis/normalization_settling.py', smoke=smoke,
        date=datetime.datetime.now().isoformat(timespec='seconds'),
        git_commit=git_commit(), git_dirty=git_dirty(), seed=SEED,
        model='V1Dynamics_Surround, adaptation frozen (g = 0, tau_g = inf)',
        N_RF=N_RF, N_SETS=N_SETS, frame=os.path.relpath(FRAME_PATH, REPO_ROOT),
        sigma=dyn.sigma, beta=dyn.beta, tau_y=tau_y, tau_u_default=tau_u0, tau_a_default=tau_a0,
        stimulus=dict(theta_rad=STIM_THETA, contrast=STIM_CONTRAST, route=STIM_ROUTE,
                      tuning_width=TUNING_WIDTH, sum_z_sq=float(np.sum(z**2))),
        initial_state='blank (z = 0) steady state', pre_duration_ms=PRE_DURATION,
        integrator='V1Dynamics_Surround.run_simulation (fixed-step RK4)',
        dt_rule=f"min({DT_DEFAULT}, min(tau_y, tau_u, tau_a) / {DT_TAU_RATIO}) ms",
        trace_duration_ms=trace_duration, sweep_duration_ms=sweep_duration,
        settle_tol=SETTLE_TOL, settle_deadline_ms=SETTLE_DEADLINE, min_dwell_ms=MIN_DWELL,
        settle_metric='max_i |x_i - x_i*| / max_i |x_i*| <= tol for the rest of the window '
                      '(and for at least min_dwell_ms before the window ends)',
        tau_u_grid=tau_u_grid.tolist(), tau_a_grid=tau_a_grid.tolist(),
        tuned=dict(tau_u=float(tau_u_grid[i]), tau_a=float(tau_a_grid[j]),
                   rule='feasible grid point whose 8 neighbours are feasible, nearest the '
                        'defaults in log(tau) distance'),
        results=dict(
            original=dict(T_y=orig['T_y'], T_u=orig['T_u'], T_a=orig['T_a'], T_rate=orig['T_rate'],
                          max_re_eig=float(orig['eig'].real.max()), fp_residual=orig['fp_residual'], dt=orig['dt'],
                slowest_symmetric_eig=[float(orig['slow_sym'].real), float(orig['slow_sym'].imag)]),
            tuned=dict(T_y=tuned['T_y'], T_u=tuned['T_u'], T_a=tuned['T_a'], T_rate=tuned['T_rate'],
                       max_re_eig=float(tuned['eig'].real.max()), fp_residual=tuned['fp_residual'], dt=tuned['dt'],
                slowest_symmetric_eig=[float(tuned['slow_sym'].real), float(tuned['slow_sym'].imag)]),
            n_feasible=int(feasible.sum()), n_points=int(feasible.size),
            n_unstable=int(np.sum(max_re >= 0)),
            n_interior=int(interior.sum()),
            dt_convergence=conv, perturbation_final_rel_error=pert,
            tuned_contrast_check=contrast_check),
    )
    with open(os.path.join(OUT_DIR, f"params{tag}.json"), 'w') as f:
        json.dump(params, f, indent=2, default=float)
    print(f"saved {npz_path}")
    return save, params


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true', help='coarse grid, short windows')
    main(parser.parse_args().smoke)
