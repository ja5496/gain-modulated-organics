'''
test_simulation_variants.py

Verification of simulation_variants.AdaptiveORGaNICs against (i) the equations it claims to
implement, re-implemented here independently with explicit loops, (ii) the existing model
simulation_whiten.V1Dynamics_Surround, and (iii) analytic results.

Run:  cd ORGaNICs_Whitening && python3 -m pytest tests/test_simulation_variants.py -q

Tolerances are fixed a priori and justified where they are set; they must not be loosened to make
a test pass. Tests whose docstring starts with "FINDING" assert the behaviour stated in the
module's docstrings (or the stated conventions) and are expected to fail until the module is
changed; they document a discrepancy.

Time unit: ms. Everything runs single-process.
'''

import contextlib
import io
import itertools
import os
import sys
from dataclasses import replace
from types import SimpleNamespace

os.environ.setdefault('MPLBACKEND', 'Agg')

import numpy as np
import pytest
from scipy.optimize import root

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'supplementary'))

import simulation_variants as sv          # noqa: E402
import simulation_whiten as sw            # noqa: E402

sw.tqdm = lambda x, **k: x                # no progress bars from V1Dynamics_Surround


# ----------------------------------------------------------------------------- small test model
N_RF, N_SETS, K = 3, 3, 5                 # 3 sets: the shared surround copy is applied to 2 blocks
N_TOT = N_RF * N_SETS


def frame(seed=7, n=N_RF, k=K):
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((n, k))
    return W / np.linalg.norm(W, axis=0)


W_SMALL = frame()


def small_model(options=sv.CODEBASE, theta=None, **kw):
    theta = np.full(K, 0.01) if theta is None else theta
    kw.setdefault('N_RF', N_RF)
    kw.setdefault('N_SETS', N_SETS)
    return sv.AdaptiveORGaNICs(W_SMALL, theta, options=options, **kw)


GATING_PLACEMENT = [(g, pl) for g in ('a', 'u') for pl in ('principal', 'interneuron')]
ALL_OPTIONS = [sv.EquationOptions(gating=g, gain_placement=pl, variance_signal=vs, rectify_c=rc,
                                  rectified_variance=rv, normalization=nm)
               for g, pl, vs, rc, rv, nm in itertools.product(
                   ('a', 'u'), ('principal', 'interneuron'), ('filtered', 'instant'),
                   (False, True), (False, True), (True, False))]


def spy_on(model):
    '''Records every call of model.derivatives (simulate() looks the method up once per call).'''
    calls = []
    orig = model.derivatives

    def wrapped(state, z, q_delayed, adapt, extra):
        calls.append(dict(state=state.copy(), z=np.array(z, copy=True),
                          q_delayed=None if q_delayed is None else np.array(q_delayed, copy=True),
                          extra={k: (np.array(v, copy=True) if v is not None else None) for k, v in extra.items()}))
        return orig(state, z, q_delayed, adapt, extra)

    model.derivatives = wrapped
    return calls


# ----------------------------------------------------------------------------- reference RHS
def reference_rhs(W, N, W_yy, p, opt, theta, n, NS, st, z, q_delayed=None, eta=None):
    '''
    The equations of the task statement for ONE simulation, written with explicit blocks and
    1-D arrays, independently of simulation_variants.derivatives.
      st: dict y (NT), u (NT), a (NT), g (2, K), v (2, K), f (2, K), mu (2, n) [neuron space]
      q_delayed: None (no delay) or (NT,) pool contributions that the cRF receives from the surround
      eta: None, or (NT,) white_response noise added to sqrt(y+)
    Returns dict of derivatives; 'dmu_c' is the derivative of the running mean in interneuron
    space, tau_mu dmu_c/dt = -mu_c + c with mu_c = W.T mu.
    '''
    NT = n * NS
    Kk = W.shape[1]
    y, u, a = st['y'], st['u'], st['a']
    g, v, f, mu = st['g'], st['v'], st['f'], st['mu']
    sqrt_yplus = np.maximum(y, 0.0)                      # sqrt(y+) = max(y, 0)
    if eta is not None:
        sqrt_yplus = sqrt_yplus + eta
    yplus = sqrt_yplus ** 2                              # y+ = max(y, 0)^2
    uplus = np.maximum(u, 0.0) ** 0.5                    # u+ = max(u, 0)^(1/2)
    aplus = np.maximum(a, 0.0)                           # a+ = max(a, 0)

    # c: cRF copy from block 0, shared surround copy from block 1
    src = sqrt_yplus if opt.rectify_c else y
    c = np.zeros((2, Kk))
    c[0] = W.T @ src[0:n]
    if NS >= 2:
        c[1] = W.T @ src[n:2 * n]
    mu_c = np.stack([W.T @ mu[0], W.T @ mu[1]])

    out = {}
    dv = np.zeros((2, Kk))
    df = np.zeros((2, Kk))
    fb = np.zeros((2, n))
    for i in range(2):
        if opt.gain_placement == 'principal':
            dv[i] = (-v[i] + c[i]) / p.tau_v
            fb[i] = W @ (g[i] * v[i])
            df[i] = dv[i]
            c_filtered = v[i]
        else:
            dv[i] = (-v[i] + g[i] * c[i]) / p.tau_v
            fb[i] = W @ v[i]
            df[i] = (-f[i] + c[i]) / p.tau_v
            c_filtered = f[i]
        dev = (c[i] if opt.variance_signal == 'instant' else c_filtered) - mu_c[i]
        var = np.maximum(dev, 0.0) ** 2 if opt.rectified_variance else dev ** 2
        out.setdefault('dg', np.zeros((2, Kk)))[i] = (var - theta) / p.tau_g
    out['dv'], out['df'] = dv, df
    out['dmu_c'] = (-mu_c + c) / p.tau_mu

    dy = np.zeros(NT)
    du = np.zeros(NT)
    da = np.zeros(NT)
    q = yplus * uplus ** 2
    for b in range(NS):
        sl = slice(b * n, (b + 1) * n)
        fb_b = fb[0] if b == 0 else fb[1]
        if opt.normalization:
            gate = uplus[sl] if opt.gating == 'u' else aplus[sl] / (1.0 + aplus[sl])
            rec = (W_yy @ sqrt_yplus[sl]) / (1.0 + aplus[sl])
            dy[sl] = (-y[sl] + p.beta * z[sl] + rec - gate * fb_b) / p.tau_y
        else:
            dy[sl] = (-y[sl] + z[sl] - fb_b) / p.tau_y
    if opt.normalization:
        for i in range(NT):
            pool = 0.0
            for j in range(NT):
                delayed = q_delayed is not None and i < n and j >= n      # surround -> cRF
                pool += N[i, j] * (q_delayed[j] if delayed else q[j])
            du[i] = (-u[i] + (p.sigma / 2.0) ** 2 + pool) / p.tau_u
            da[i] = (-a[i] + uplus[i] + a[i] * uplus[i]) / p.tau_a
    out['dy'], out['du'], out['da'] = dy, du, da
    return out


def random_state(rng, B, model):
    '''Random states with both signs in every variable (to exercise every rectification).'''
    state = rng.standard_normal((B, model.state_size)) * 0.5
    state[:, model.sl['u']] = rng.uniform(-0.05, 0.4, (B, model.N_TOT))
    return state


def split_state(model, row):
    s, Kk, n = model.sl, model.K, model.N_RF
    return dict(y=row[s['y']], u=row[s['u']], a=row[s['a']], g=row[s['g']].reshape(2, Kk),
                v=row[s['v']].reshape(2, Kk), f=row[s['f']].reshape(2, Kk), mu=row[s['mu']].reshape(2, n))


# ============================================================================== 1. equations
@pytest.mark.parametrize('delay', [0.0, 1.0])
def test_rhs_matches_equations_for_every_option_combination(delay):
    '''derivatives() against the independent reference for all 64 option combinations, random
    states of both signs, a non-symmetric N, a non-symmetric W_yy, non-uniform theta_t, with and
    without the surround delay. Tolerance 1e-12: pure floating-point reordering.'''
    rng = np.random.default_rng(100)
    Nmat = rng.uniform(0.0, 2.0, (N_TOT, N_TOT))
    W_yy = 0.8 * np.eye(N_RF) + 0.1 * rng.standard_normal((N_RF, N_RF))
    theta = rng.uniform(0.001, 0.02, K)
    p = sv.ModelParams(tau_y=0.37, tau_a=0.23, tau_u=29.0, tau_v=31.0, tau_g=510.0, tau_mu=1900.0,
                       sigma=0.17, beta=0.45)    # all different, so that no time constant can be swapped
    B = 3
    worst = 0.0
    for opt in ALL_OPTIONS:
        m = small_model(opt, theta=theta, params=p, N_matrix=Nmat, W_yy=W_yy, surround_delay=delay)
        state = random_state(rng, B, m)
        z = rng.standard_normal((B, N_TOT))
        qd = rng.uniform(0.0, 0.3, (B, N_TOT)) if delay > 0 else None
        d = m.derivatives(state, z, qd, True, {})
        s = m.sl
        for b in range(B):
            ref = reference_rhs(W_SMALL, Nmat, W_yy, p, opt, theta, N_RF, N_SETS, split_state(m, state[b]),
                                z[b], None if qd is None else qd[b])
            got = split_state(m, d[b])
            for name in ('dy', 'du', 'da', 'dg', 'dv'):
                err = np.abs(got[name[1:]] - ref[name]).max()
                worst = max(worst, err)
                assert err < 1e-12, f'{name} mismatch {err:.2e} for {opt}'
            if opt.gain_placement == 'interneuron' or opt.variance_signal == 'filtered':
                assert np.abs(got['f'] - ref['df']).max() < 1e-12, f'df mismatch for {opt}'
            # Running mean: the module tracks it in neuron space; its projection must obey
            # tau_mu dmu_c/dt = -mu_c + c (linearity), also with rectify_c=True.
            dmu_c = np.stack([W_SMALL.T @ got['mu'][0], W_SMALL.T @ got['mu'][1]])
            assert np.abs(dmu_c - ref['dmu_c']).max() < 1e-12, f'dmu mismatch for {opt}'
    print(f'\n[rhs] worst |module - reference| over all option combinations (delay {delay}): {worst:.1e}')


def test_rhs_frozen_adaptation_zeroes_gain_and_mean_derivatives():
    m = small_model(sv.NEW)
    rng = np.random.default_rng(1)
    d = m.derivatives(random_state(rng, 2, m), rng.standard_normal((2, N_TOT)), None, False, {})
    assert np.all(d[:, m.sl['g']] == 0) and np.all(d[:, m.sl['mu']] == 0)


def test_rhs_single_set_has_inert_surround_copy():
    '''N_SETS = 1: the surround copy is driven by zero, as in V1Dynamics_Surround.'''
    rng = np.random.default_rng(2)
    theta = rng.uniform(0.001, 0.02, K)
    p = sv.ModelParams()
    for opt in (sv.CODEBASE, sv.NEW):
        m = small_model(opt, theta=theta, N_SETS=1)
        state = random_state(rng, 2, m)
        z = rng.standard_normal((2, N_RF))
        d = m.derivatives(state, z, None, True, {})
        for b in range(2):
            ref = reference_rhs(W_SMALL, np.ones((N_RF, N_RF)), np.eye(N_RF), p, opt, theta, N_RF, 1,
                                split_state(m, state[b]), z[b])
            got = split_state(m, d[b])
            for name in ('dy', 'du', 'da', 'dg', 'dv'):
                assert np.abs(got[name[1:]] - ref[name]).max() < 1e-12, (name, opt)


def test_normalization_off_reduces_exactly_to_reduced_equations():
    '''normalization=False (+ gain_placement='interneuron') must be exactly
         tau_y dy/dt = -y + z - W v,   tau_v dv/dt = -v + g c,   du/dt = da/dt = 0,
       and nothing normalization-related may leak in: gating, beta, sigma, W_yy, N, the delay and
       the delayed pool signal must have no effect at all (bit-identical derivatives).'''
    rng = np.random.default_rng(3)
    theta = rng.uniform(0.001, 0.02, K)
    base = sv.EquationOptions(gain_placement='interneuron', variance_signal='instant', rectify_c=True,
                              normalization=False)
    m0 = small_model(base, theta=theta)
    state = random_state(rng, 4, m0)
    z = rng.standard_normal((4, N_TOT))
    d0 = m0.derivatives(state, z, None, True, {})
    s, p = m0.sl, m0.p
    # hand-written reduced equations
    for b in range(4):
        st = split_state(m0, state[b])
        y = st['y']
        c = np.stack([W_SMALL.T @ np.maximum(y[:N_RF], 0), W_SMALL.T @ np.maximum(y[N_RF:2 * N_RF], 0)])
        fb = np.concatenate([W_SMALL @ st['v'][0]] + [W_SMALL @ st['v'][1]] * (N_SETS - 1))
        assert np.abs(d0[b, s['y']] - (-y + z[b] - fb) / p.tau_y).max() < 1e-12
        assert np.abs(d0[b, s['v']].reshape(2, K) - (-st['v'] + st['g'] * c) / p.tau_v).max() < 1e-12
        assert np.all(d0[b, s['u']] == 0) and np.all(d0[b, s['a']] == 0)
    # nothing else matters
    Nmat = rng.uniform(0.0, 2.0, (N_TOT, N_TOT))
    W_yy = 0.8 * np.eye(N_RF) + 0.1 * rng.standard_normal((N_RF, N_RF))
    m1 = small_model(replace(base, gating='u'), theta=theta, N_matrix=Nmat, W_yy=W_yy, surround_delay=2.0,
                     params=sv.ModelParams(beta=0.3, sigma=0.5, tau_u=7.0, tau_a=3.0))
    d1 = m1.derivatives(state, z, rng.uniform(0, 1, (4, N_TOT)), True, {})
    assert np.array_equal(d0, d1)


def test_default_parameters_equal_codebase():
    '''ModelParams defaults against the attributes of V1Dynamics_Surround.'''
    dyn = make_old(W_SMALL, np.full(K, 0.01), N_RF, N_SETS, np.eye(N_RF))
    p = sv.ModelParams()
    for name in ('tau_y', 'tau_a', 'tau_u', 'tau_v', 'tau_g', 'tau_mu', 'sigma', 'beta'):
        assert getattr(p, name) == getattr(dyn, name), name


def test_theta_t_from_target_matches_codebase_calibration():
    rng = np.random.default_rng(4)
    A = rng.standard_normal((N_RF, N_RF))
    C_target = A @ A.T
    dyn = make_old(W_SMALL, np.ones(K), N_RF, N_SETS, np.eye(N_RF))
    ref = dyn.calibrate_theta_t(C_zz_uniform=C_target, uniform_target=True, verbose=False)
    assert np.allclose(sv.theta_t_from_target(W_SMALL, C_target), ref, rtol=1e-13, atol=0)


# ============================================================================== 2. regression
_COV_FILES = {}


def cov_file(n):
    '''V1Dynamics_Surround loads an (n, n) target covariance from disk; it is overwritten by
    theta_t below, so its content is irrelevant.'''
    if n not in _COV_FILES:
        import tempfile
        path = os.path.join(tempfile.mkdtemp(prefix='sv_tests_'), f'cov_{n}.csv')
        np.savetxt(path, np.eye(n), delimiter=',')
        _COV_FILES[n] = path
    return _COV_FILES[n]


def make_old(W, theta, n, NS, W_yy, dt=0.1, gains_nonneg=True, params=None):
    with contextlib.redirect_stdout(io.StringIO()):
        dyn = sw.V1Dynamics_Surround(SimpleNamespace(W_yy=W_yy), SimpleNamespace(W=W, K=W.shape[1]), dt=dt,
                                     N_RF=n, N_SETS=NS, target_covariance_path=cov_file(n),
                                     gains_nonneg=gains_nonneg)
    dyn.theta_t = np.array(theta, dtype=float)
    for k, v in (params or {}).items():
        setattr(dyn, k, v)
    return dyn


def run_old_and_new(W, theta, n, NS, z_pres, steps, delay, W_yy=None, params=None, gains_nonneg=True,
                    baseline_ms=30.0, dt=0.1):
    '''Runs V1Dynamics_Surround and AdaptiveORGaNICs(CODEBASE) on the same presentations, with the
    delayed surround signal held at the blank-settled value until the first delayed sample exists
    (each model settles its own baseline). Returns (old outputs, old model, new result, new model).'''
    W_yy = np.eye(n) if W_yy is None else W_yy
    params = params or {}
    Kk = W.shape[1]
    blank = np.full(n * NS, 0.03)
    dyn = make_old(W, theta, n, NS, W_yy, dt=dt, gains_nonneg=gains_nonneg, params=params)
    model = sv.AdaptiveORGaNICs(W, theta, N_RF=n, N_SETS=NS, options=sv.CODEBASE, W_yy=W_yy, dt=dt,
                                params=sv.ModelParams(**params), gains_nonneg=gains_nonneg)
    with contextlib.redirect_stdout(io.StringIO()):
        msg, q0 = 0.0, None
        if delay > 0:
            nb = int(round(baseline_ms / dt))
            tau_g = dyn.tau_g
            dyn.tau_g = np.inf                                 # as response_tools.settled_surround_msg
            dyn.run_simulation(np.repeat(blank[:, None], nb, axis=1))
            dyn.tau_g = tau_g
            msg = dyn._surround_msg(dyn.last_state)
            q0 = model.q_of(model.settle(blank, 0.0, 0.0, settle_ms=baseline_ms)['state'])
        dyn.set_surround_delay(delay)
        out = dyn.run_simulation(np.repeat(z_pres.T, steps, axis=1), surround_msg_baseline=msg)
    model.set_surround_delay(delay)
    res = model.simulate(z_pres, steps, q_baseline=q0, record_traces=True)
    return out, dyn, res, model


def compare_old_new(out, dyn, res, model, steps, tol):
    '''Element-wise comparison of every state variable over the whole run. Returns the worst error.'''
    (y_h, u_c, u_s, a_c, a_s, g_c, g_s, v_c, v_s, mu_c, mu_s, _) = out
    n, NS, Kk = model.N_RF, model.N_SETS, model.K
    tr = res['traces']
    errs = {}
    errs['y (rectified, every step)'] = np.abs(np.maximum(tr['y'][0], 0).T - y_h).max()
    errs['u cRF'] = np.abs(tr['u'][0, :, :n].T - u_c).max()
    errs['a cRF'] = np.abs(tr['a'][0, :, :n].T - a_c).max()
    for b in range(1, NS):                                   # every surround block against the shared copy
        errs[f'u surround block {b}'] = np.abs(tr['u'][0, :, b * n:(b + 1) * n].T - u_s).max()
        errs[f'a surround block {b}'] = np.abs(tr['a'][0, :, b * n:(b + 1) * n].T - a_s).max()
    errs['v cRF'] = np.abs(tr['v'][0, :, :Kk].T - v_c).max()
    errs['v surround'] = np.abs(tr['v'][0, :, Kk:].T - v_s).max()
    errs['g cRF (presentation ends)'] = np.abs(res['g_end'][0, :, :Kk].T - g_c[:, steps - 1::steps]).max()
    errs['g surround (presentation ends)'] = np.abs(res['g_end'][0, :, Kk:].T - g_s[:, steps - 1::steps]).max()
    old_final = dyn.unpack_state(dyn.last_state)
    new_final = model.unpack(res['state'])
    errs['y (raw, final)'] = np.abs(new_final['y'][0] - old_final['y']).max()
    errs['y (raw, presentation ends)'] = np.abs(np.maximum(res['y_end'][0], 0).T - y_h[:, steps - 1::steps]).max()
    errs['mu cRF (final)'] = np.abs(new_final['mu_cRF'][0] - old_final['mu_cRF']).max()
    errs['mu surround (final)'] = np.abs(new_final['mu_surround'][0] - old_final['mu_surround']).max()
    errs['mu cRF (history end)'] = np.abs(new_final['mu_cRF'][0] - mu_c[:, -1]).max()
    for name, e in errs.items():
        assert e < tol, f'{name}: {e:.2e}'
    return max(errs.values())


# Regression tolerance 1e-11 (absolute; all variables are O(1) or smaller): the two codes evaluate
# the same RK4 scheme with a different order of floating-point operations, so only round-off
# (~1e-16 per operation, accumulated over < 1e4 steps of a contracting system) may differ.
REG_TOL = 1e-11


def test_regression_full_configuration_with_delay():
    '''The configuration of supplementary/common.py (N_RF 13, 6 sets, K = 182 frame, default time
    constants, 40 ms delay, blank-settled baseline), 6 presentations of 32 ms of the biased ensemble.'''
    import common as C
    from tunings_whiten import V1Tunings
    z, _ = C.ensemble_presentations(6, True, 11, C.LARGE)
    W = C.load_frame()
    theta = sv.theta_t_from_target(W, C.target_covariance())
    out, dyn, res, model = run_old_and_new(W, theta, C.N_RF, C.N_SETS, z, 320, C.SURROUND_DELAY,
                                           W_yy=V1Tunings(N=C.N_RF).W_yy)
    worst = compare_old_new(out, dyn, res, model, 320, REG_TOL)
    print(f'\n[regression full] worst error {worst:.1e}; max g {res["g_end"].max():.2e}')


def _fast_scenario(seed=21, n_pres=30):
    '''Small model with accelerated adaptation, so that within a short run the gains become large,
    the clamp acts, responses go negative and mu is far from zero: every term is exercised.'''
    rng = np.random.default_rng(seed)
    # log-uniform targets up to 0.1: some are too large to be reached, so those gains get clamped at 0
    theta = np.exp(rng.uniform(np.log(0.002), np.log(0.1), K))
    z = rng.uniform(0.0, 0.6, (n_pres, N_TOT))            # different drive in every block
    W_yy = 0.8 * np.eye(N_RF) + 0.08 * rng.standard_normal((N_RF, N_RF))    # non-symmetric
    params = dict(tau_g=5.0, tau_mu=50.0, tau_v=5.0)
    return theta, z, W_yy, params


def test_regression_accelerated_adaptation_nonidentity_Wyy_with_delay():
    theta, z, W_yy, params = _fast_scenario()
    out, dyn, res, model = run_old_and_new(W_SMALL, theta, N_RF, N_SETS, z, 80, 1.3, W_yy=W_yy, params=params)
    worst = compare_old_new(out, dyn, res, model, 80, REG_TOL)
    g = res['g_end'][0]
    # the scenario must actually exercise the gain feedback, the clamp and negative responses
    assert g.max() > 0.5 and (g == 0).any() and res['traces']['y'].min() < -0.01
    print(f'\n[regression fast] worst error {worst:.1e}; max g {g.max():.2f}, min y {res["traces"]["y"].min():.3f}, '
          f'fraction of clamped gains {np.mean(g == 0):.2f}')


def test_regression_without_delay_and_without_clamp():
    theta, z, W_yy, params = _fast_scenario(seed=22, n_pres=6)
    theta[0] = 0.08                                          # unreachable target: this gain goes negative
    out, dyn, res, model = run_old_and_new(W_SMALL, theta, N_RF, N_SETS, z, 80, 0.0, W_yy=W_yy, params=params,
                                           gains_nonneg=False)
    worst = compare_old_new(out, dyn, res, model, 80, REG_TOL)
    assert res['g_end'].min() < 0                           # the unclamped path was exercised
    print(f'\n[regression no delay, no clamp] worst error {worst:.1e}; min g {res["g_end"].min():.3f}')


def test_regression_single_set_with_delay():
    rng = np.random.default_rng(23)
    theta = rng.uniform(0.002, 0.02, K)
    z = rng.uniform(0.0, 0.6, (10, N_RF))
    out, dyn, res, model = run_old_and_new(W_SMALL, theta, N_RF, 1, z, 80, 1.3,
                                           params=dict(tau_g=5.0, tau_mu=50.0, tau_v=5.0))
    worst = compare_old_new(out, dyn, res, model, 80, REG_TOL)
    print(f'\n[regression N_SETS=1] worst error {worst:.1e}')


def test_delay_buffer_holds_the_state_n_lag_steps_back():
    '''Independent of V1Dynamics_Surround: the delayed signal handed to derivatives() at step t is
    q(state after step t - n_lag) (the codebase convention: surround_msg_hist[t - n_lag]), and the
    supplied baseline for t < n_lag, for an n_lag that does not divide the run length.'''
    theta, z, W_yy, params = _fast_scenario(seed=24, n_pres=5)
    m = small_model(sv.CODEBASE, theta=theta, params=sv.ModelParams(**params), surround_delay=0.7)
    assert m.n_lag == 7
    calls = spy_on(m)
    q0 = np.random.default_rng(0).uniform(0, 0.1, (1, N_TOT))
    res = m.simulate(z, 11, q_baseline=q0, record_traces=True)
    tr = res['traces']
    for t in range(z.shape[0] * 11):
        got = calls[4 * t]['q_delayed'][0]
        for k in range(1, 4):                                # held over the four RK4 stages
            assert np.array_equal(calls[4 * t + k]['q_delayed'][0], got)
        if t < m.n_lag:
            expect = q0[0]
        else:
            i = t - m.n_lag
            expect = np.maximum(tr['y'][0, i], 0) ** 2 * np.maximum(tr['u'][0, i], 0)
        assert np.abs(got - expect).max() < 1e-15, t


# ============================================================================== 3. fixed points
def fast_residual(model, z, g_cRF, g_sur):
    '''RHS of the fast subsystem (y, u, a, v) as a function of those variables, gains frozen.'''
    s = model.sl
    idx = np.concatenate([np.arange(s[k].start, s[k].stop) for k in ('y', 'u', 'a', 'v')])
    base = model.zero_state(1)
    base[0, s['g']] = np.concatenate([g_cRF, g_sur])

    def fun(x):
        st = base.copy()
        st[0, idx] = x
        return model.derivatives(st, z[None], None, False, {})[0, idx]

    return fun, idx


def analytic_fast_state(model, z, g_cRF, g_sur):
    '''(y, u, a, v) of the analytic fixed point, in the order used by fast_residual.'''
    fp = model.analytic_fixed_point(z[None], g_cRF, g_sur)
    y = fp['y'][0]
    n = model.N_RF
    c = np.concatenate([W_SMALL.T @ np.maximum(y[:n], 0), W_SMALL.T @ np.maximum(y[n:2 * n], 0)])
    g = np.concatenate([g_cRF, g_sur])
    v = g * c if model.opt.gain_placement == 'interneuron' else c
    return np.concatenate([y, fp['u'][0], fp['a'][0], v]), fp


def _fixed_point_cases():
    rng = np.random.default_rng(31)
    # The closed form (block-diagonal gain operator) needs the same drive in every surround set:
    # the single surround copy of g, v is driven by surround set 1 and fed back to all of them.
    z_c, z_s = 0.04 + 0.06 * rng.random((4, N_RF)), 0.04 + 0.06 * rng.random((4, N_RF))
    z = np.concatenate([z_c] + [z_s] * (N_SETS - 1), axis=1)
    g0 = np.zeros(K)
    g1c, g1s = rng.uniform(0.0, 0.8, K), rng.uniform(0.0, 0.8, K)
    return z, [('g = 0', g0, g0), ('g > 0', g1c, g1s)]


@pytest.mark.parametrize('gating,placement', GATING_PLACEMENT)
def test_fixed_point_matches_analytic(gating, placement):
    '''CLAUDE.md Verification 1-2, 4: frozen gains, W_yy = I, all-ones N. Simulated steady state
    (600 ms = 20 tau_u from a zero state) against
       z' = (I + W D(g) W.T)^-1 z,  y = z'/sqrt(sigma^2 + sum z'^2),  u = (sigma^2 + sum z'^2)/4,
       a/(1+a) = sqrt(u),  v = D(g) W.T y  (interneuron placement) or W.T y (principal),
    to 1e-6 relative (CLAUDE.md), confirmed with root-finding on the RHS (1e-9), with u* < 1,
    z' >= 0 and all Jacobian eigenvalues in the left half-plane.'''
    opt = replace(sv.CODEBASE, gating=gating, gain_placement=placement, rectify_c=(gating == 'u'))
    m = small_model(opt)
    z, cases = _fixed_point_cases()
    for label, gc, gs in cases:
        fp = m.analytic_fixed_point(z, gc, gs)
        assert fp['z_prime'].min() >= 0, 'test set-up: the closed form needs z\' >= 0'
        assert fp['u'].max() < 1, 'validity: u* < 1'
        res = m.settle(z, gc, gs, settle_ms=600.0)
        st = m.unpack(res['state'])
        assert st['u'].max() < 1
        rel = lambda x, ref: np.abs(x - ref).max() / np.abs(ref).max()
        e = dict(y=rel(st['y'], fp['y']), u=rel(st['u'], fp['u']), a=rel(st['a'], fp['a']))
        c = np.concatenate([np.maximum(fp['y'][:, :N_RF], 0) @ W_SMALL,
                            np.maximum(fp['y'][:, N_RF:2 * N_RF], 0) @ W_SMALL], axis=1)
        v_ref = np.concatenate([gc, gs]) * c if placement == 'interneuron' else c
        v_sim = np.concatenate([st['v_cRF'], st['v_surround']], axis=1)
        e['v'] = np.abs(v_sim - v_ref).max() / np.abs(c).max()
        for name, err in e.items():
            assert err < 1e-6, f'{label} {name}: {err:.2e}'

        # root-finding on the RHS from a perturbed start, and linear stability at the root
        rng = np.random.default_rng(32)
        worst_root, worst_eig, stiff = 0.0, -np.inf, 0.0
        for i in range(z.shape[0]):
            fun, idx = fast_residual(m, z[i], gc, gs)
            x_ref, _ = analytic_fast_state(m, z[i], gc, gs)
            assert np.abs(fun(x_ref)).max() < 1e-12, 'analytic fixed point is not a zero of the RHS'
            sol = root(fun, x_ref * (1 + 0.02 * rng.standard_normal(x_ref.size)), tol=1e-12)
            assert np.abs(fun(sol.x)).max() < 1e-12, 'root-finder did not reach a zero of the RHS'
            worst_root = max(worst_root, np.abs(sol.x - x_ref).max() / np.abs(x_ref).max())
            J = np.zeros((idx.size, idx.size))
            h = 1e-7
            for j in range(idx.size):
                dx = np.zeros(idx.size)
                dx[j] = h
                J[:, j] = (fun(x_ref + dx) - fun(x_ref - dx)) / (2 * h)
            ev = np.linalg.eigvals(J)
            worst_eig = max(worst_eig, ev.real.max())
            stiff = max(stiff, np.abs(ev.real).max() / np.abs(ev.real).min())
        assert worst_root < 1e-9, f'{label}: root differs from the analytic fixed point by {worst_root:.2e}'
        assert worst_eig < 0, f'{label}: unstable fixed point, max Re(lambda) = {worst_eig:.3e} /ms'
        print(f'\n[fixed point] gating {gating}, {placement:11s}, {label}: rel. err y {e["y"]:.1e} u {e["u"]:.1e} '
              f'a {e["a"]:.1e} v {e["v"]:.1e}; root {worst_root:.1e}; max u* {fp["u"].max():.3f}; '
              f'min z\' {fp["z_prime"].min():.3f}; max Re(lambda) {worst_eig:.4f}/ms; stiffness {stiff:.0f}')


@pytest.mark.parametrize('gating', ['a', 'u'])
def test_gain_placements_are_equivalent_for_frozen_gains(gating):
    '''For frozen g, feedback W (g v) with tau_v dv/dt = -v + c and feedback W v with
    tau_v dv/dt = -v + g c are the same system (v_interneuron = g v_principal) at every time, not
    only at steady state; and the state f of the interneuron placement is the codebase v.
    Tolerance 1e-12: round-off only (RK4 commutes with the constant rescaling of v).'''
    rng = np.random.default_rng(41)
    z = rng.uniform(0.0, 0.5, (2, N_TOT))
    gc, gs = rng.uniform(0, 1.5, K), rng.uniform(0, 1.5, K)
    for rectify_c in (False, True):
        runs = {}
        for pl in ('principal', 'interneuron'):
            m = small_model(replace(sv.CODEBASE, gating=gating, gain_placement=pl, rectify_c=rectify_c))
            runs[pl] = m.settle(z, gc, gs, settle_ms=150.0, record_traces=True)
        tp, ti = runs['principal']['traces'], runs['interneuron']['traces']
        for name in ('y', 'u', 'a'):
            assert np.abs(tp[name] - ti[name]).max() < 1e-12, name
        g = np.concatenate([gc, gs])
        assert np.abs(g * tp['v'] - ti['v']).max() < 1e-12
        f_int = runs['interneuron']['state'][:, m.sl['f']]
        v_pri = runs['principal']['state'][:, m.sl['v']]
        assert np.abs(f_int - v_pri).max() < 1e-12


@pytest.mark.parametrize('placement', ['principal', 'interneuron'])
def test_normalization_off_steady_state(placement):
    '''normalization=False, frozen g, c = W.T y: the system is linear and its steady state is
    y = (I + W D(g) W.T)^-1 z for inputs of either sign. 600 ms = 20 tau_v gives exp(-20) = 2e-9
    at worst (the slowest rate is >= 1/tau_v); tolerance 1e-7 relative.'''
    rng = np.random.default_rng(51)
    z_c, z_s = rng.standard_normal((4, N_RF)) * 0.3, rng.standard_normal((4, N_RF)) * 0.3
    z = np.concatenate([z_c] + [z_s] * (N_SETS - 1), axis=1)     # same drive in every surround set
    gc, gs = rng.uniform(0, 2.0, K), rng.uniform(0, 2.0, K)
    m = small_model(sv.EquationOptions(gain_placement=placement, variance_signal='instant', normalization=False))
    res = m.settle(z, gc, gs, settle_ms=600.0)
    y = m.unpack(res['state'])['y']
    y_ref = np.linalg.solve(np.eye(N_TOT) + m.gain_operator(gc, gs), z.T).T
    err = np.abs(y - y_ref).max() / np.abs(y_ref).max()
    assert (y_ref < 0).any()
    assert err < 1e-7, err
    assert np.allclose(m.analytic_fixed_point(z, gc, gs)['y'], y_ref, rtol=0, atol=1e-14)
    assert np.all(res['state'][:, m.sl['u']] == 0) and np.all(res['state'][:, m.sl['a']] == 0)
    print(f'\n[normalization off] {placement}: rel. err of y vs (I + W D(g) W.T)^-1 z = {err:.1e}')

    # Block layout with a different drive in every set: the surround copy is driven by set 1 only,
    # so y_0 = (I + G_c)^-1 z_0, y_1 = (I + G_s)^-1 z_1 and y_b = z_b - G_s y_1 for b >= 2
    # (NOT (I + G_s)^-1 z_b: gain_operator / analytic_fixed_point do not apply to such inputs).
    z = rng.standard_normal((4, N_TOT)) * 0.3
    y = m.unpack(m.settle(z, gc, gs, settle_ms=600.0)['state'])['y']
    G_c, G_s = W_SMALL @ np.diag(gc) @ W_SMALL.T, W_SMALL @ np.diag(gs) @ W_SMALL.T
    I = np.eye(N_RF)
    y0 = np.linalg.solve(I + G_c, z[:, :N_RF].T).T
    y1 = np.linalg.solve(I + G_s, z[:, N_RF:2 * N_RF].T).T
    expect = np.concatenate([y0, y1] + [z[:, b * N_RF:(b + 1) * N_RF] - y1 @ G_s.T for b in range(2, N_SETS)], axis=1)
    err2 = np.abs(y - expect).max() / np.abs(expect).max()
    assert err2 < 1e-7, err2
    print(f'[normalization off] {placement}: different drive per set, rel. err vs shared-copy fixed point = {err2:.1e}')


# ============================================================================== 4. gains
def test_gains_are_nonnegative_at_every_step():
    '''Gains recorded after every single step (one step per "presentation") are >= 0, the clamp is
    actually reached, and the same run without the clamp does go negative.'''
    theta, z, W_yy, params = _fast_scenario(seed=92, n_pres=30)
    z_steps = np.repeat(z, 80, axis=0)                       # (2400, N_TOT), one step each
    for name, opt in (('codebase', sv.CODEBASE), ('new', sv.NEW)):
        m = small_model(opt, theta=theta, params=sv.ModelParams(**params), surround_delay=1.3)
        g = m.simulate(z_steps, 1)['g_end'][0]
        assert g.shape[0] == z_steps.shape[0]
        assert g.min() >= 0.0
        assert (g == 0).mean() > 0.01 and g.max() > 0.1
        m2 = small_model(opt, theta=theta, params=sv.ModelParams(**params), surround_delay=1.3, gains_nonneg=False)
        g2 = m2.simulate(z_steps, 1)['g_end'][0]
        assert g2.min() < 0
        print(f'\n[clamp] {name}: min g {g.min():.1e}, fraction of steps*gains at 0: {(g == 0).mean():.2f}, '
              f'max g {g.max():.2f}; without clamp min g {g2.min():.2f}')


@pytest.mark.parametrize('name,opt', [('codebase', sv.CODEBASE), ('new', sv.NEW),
                                      ('new, rectified variance', replace(sv.NEW, rectified_variance=True))])
def test_constant_input_gains_decay_to_zero_and_stay(name, opt):
    '''CLAUDE.md Verification 5. Constant input: c -> mu, so dg/dt -> -theta_t/tau_g until g = 0,
    then g stays at 0. tau_g and tau_mu are shortened to 30 ms (the property does not depend on
    them) and theta_t = 0.05, so the decay from g ~ 0.3-0.6 takes < 400 ms.
    Slope tolerance 5%: while g decays, y drifts and mu lags c by ~tau_mu dc/dt ~ 30 ms x 2e-3/ms
    x 0.3 ~ 0.02, i.e. (c - mu)^2 ~ 4e-4 << theta = 0.05 (< 1% of the drive); 5% leaves a factor 5.'''
    p = sv.ModelParams(tau_g=30.0, tau_mu=30.0)
    theta = np.full(K, 0.05)
    rng = np.random.default_rng(71)
    m = small_model(opt, theta=theta, params=p)
    z = 0.05 + 0.05 * rng.random((1, N_TOT))
    st = m.zero_state(1)
    st[:, m.sl['g']] = 0.3
    g = m.simulate(np.repeat(z, 800, axis=0), 10, state0=st)['g_end'][0]     # every 1 ms, 800 ms
    assert g.min() >= 0
    assert np.all(g[-1] == 0), f'gains not at 0 after 800 ms: {g[-1]}'
    first_zero = np.array([np.argmax(g[:, k] == 0) for k in range(2 * K)])
    for k in range(2 * K):
        assert np.all(g[first_zero[k]:, k] == 0), 'a gain left 0 again under constant input'
    assert first_zero.max() < 700, 'gains must have stayed at 0 for at least 100 ms'
    # after the initial transient (5 tau_mu), strictly non-increasing with slope -theta/tau_g while > 0
    late = g[150:]
    assert np.diff(late, axis=0).max() <= 0
    slopes = []
    for k in range(2 * K):
        i1 = min(first_zero[k] - 2, 250)
        if i1 > 170:
            slopes.append((g[i1, k] - g[150, k]) / (i1 - 150))
    slopes = np.array(slopes)
    assert slopes.size > 0
    ratio = slopes / (-theta[0] / p.tau_g)
    assert np.abs(ratio - 1).max() < 0.05, ratio
    print(f'\n[constant input] {name}: max g reached {g.max():.3f}; all gains 0 from {first_zero.max()} ms; '
          f'late slope / (-theta/tau_g) in [{ratio.min():.4f}, {ratio.max():.4f}]')


def test_rectified_variance_term_is_half_of_a_symmetric_deviation_exactly():
    '''Unit test of the dg drive: for the pair of states y and -y (mu = 0, c = W.T y), the drives
    max(c, 0)^2 summed over the pair equal the unrectified c^2 of one of them, i.e. the rectified
    variance of a symmetric signal is half its variance.'''
    rng = np.random.default_rng(81)
    theta = rng.uniform(0.001, 0.02, K)
    base = sv.EquationOptions(variance_signal='instant', rectify_c=False)
    m_full, m_rect = small_model(base, theta=theta), small_model(replace(base, rectified_variance=True), theta=theta)
    st = m_full.zero_state(2)
    y = rng.standard_normal(N_TOT)
    st[0, m_full.sl['y']], st[1, m_full.sl['y']] = y, -y
    z = np.zeros((2, N_TOT))
    tau_g = m_full.p.tau_g
    theta2 = np.concatenate([theta, theta])
    full = m_full.derivatives(st, z, None, True, {})[:, m_full.sl['g']] * tau_g + theta2
    rect = m_rect.derivatives(st, z, None, True, {})[:, m_rect.sl['g']] * tau_g + theta2
    assert np.abs(full[0] - full[1]).max() < 1e-14
    assert np.abs(rect.mean(axis=0) - 0.5 * full.mean(axis=0)).max() < 1e-14
    assert np.all(rect >= 0)


def test_rectified_variance_of_noise_is_half_the_variance_statistically():
    '''Statistical version on the integrated drive. normalization off, z = 0, white input noise:
    c is zero-mean Gaussian. With theta = 0, no clamp and tau_g = 1e9 ms the gains stay ~1e-9
    (no feedback), so g(T) tau_g / T is the time average of the drive; the two runs use the same
    noise. Tolerance: for n_eff independent samples the ratio sum max(x,0)^2 / sum x^2 has s.d.
    sqrt(3)/2 / sqrt(n_eff); c decorrelates in ~2 tau_y = 0.8 ms, so 100 traces x 60 ms give
    n_eff ~ 7500 and s.d. ~ 0.01. Accept 0.5 +- 0.05 (5 s.d.) for every interneuron.'''
    p = sv.ModelParams(tau_g=1e9)
    noise = sv.NoiseOptions('white_input', 0.05)
    base = sv.EquationOptions(variance_signal='instant', rectify_c=False, normalization=False)
    g = {}
    for rect in (False, True):
        m = small_model(replace(base, rectified_variance=rect), theta=np.zeros(K), params=p, noise=noise,
                        gains_nonneg=False)
        res = m.simulate(np.zeros((100, 1, N_TOT)), 600, rng=np.random.default_rng(82))
        g[rect] = res['g_end'][:, -1].mean(axis=0)
    ratio = g[True] / g[False]
    assert g[False].min() > 0
    assert np.abs(ratio - 0.5).max() < 0.05, ratio
    print(f'\n[rectified variance] integrated drive ratio per interneuron: {np.round(ratio, 3)}')


# ============================================================================== 5. time step
def test_rk4_is_fourth_order_on_a_smooth_trajectory():
    '''Frozen gains, all responses positive (no rectification kink is crossed), stimulus switch:
    the difference between successive halvings of dt must shrink by ~2^4. Accept an observed
    order > 3.5 between dt = 0.1, 0.05 and 0.025 ms.'''
    rng = np.random.default_rng(91)
    gc, gs = 0.3 * rng.random(K), 0.3 * rng.random(K)
    z0, z1 = 0.05 + 0.05 * rng.random((1, N_TOT)), 0.05 + 0.05 * rng.random((1, N_TOT))
    for name, opt in (('codebase', sv.CODEBASE), ('new', sv.NEW)):
        start = small_model(opt, dt=0.025).settle(z0, gc, gs, settle_ms=450.0)['state']
        tr = {}
        for dt in (0.1, 0.05, 0.025):
            t = small_model(opt, dt=dt).settle(z1, gc, gs, settle_ms=20.0, state0=start, record_traces=True)['traces']
            k = int(round(0.1 / dt))
            assert t['y'].min() > 0
            tr[dt] = np.concatenate([t[v][0, k - 1::k] for v in ('y', 'u', 'a', 'v')], axis=1)
        e1, e2 = np.abs(tr[0.1] - tr[0.05]).max(), np.abs(tr[0.05] - tr[0.025]).max()
        order = np.log2(e1 / e2)
        assert order > 3.5, f'{name}: observed order {order:.2f} (differences {e1:.1e}, {e2:.1e})'
        print(f'\n[RK4 order] {name}: |x_0.1 - x_0.05| = {e1:.1e}, |x_0.05 - x_0.025| = {e2:.1e}, order {order:.2f}')


@pytest.mark.parametrize('delay', [0.0, 4.0])
@pytest.mark.parametrize('name,opt', [('codebase', sv.CODEBASE), ('new', sv.NEW)])
def test_halving_dt_changes_an_adaptation_run_by_less_than_1e3(name, opt, delay):
    '''CLAUDE.md Verification 8, on a run with switching stimuli, adapting gains (accelerated),
    the clamp and the surround delay (the delayed signal is sampled once per step, an O(dt)
    scheme, so the convergence is not fourth order here). Tolerance: relative change of the gains
    and of the responses at presentation ends < 1e-3 between dt = 0.1 and 0.05 ms; and the change
    between 0.05 and 0.025 must be smaller still (it converges).'''
    theta, z, _, params = _fast_scenario(seed=92, n_pres=20)
    params = dict(tau_g=40.0, tau_mu=200.0)
    out = {}
    for dt in (0.1, 0.05, 0.025):
        m = small_model(opt, theta=theta, params=sv.ModelParams(**params), dt=dt, surround_delay=delay)
        out[dt] = m.simulate(z, int(round(16.0 / dt)))
    rel = lambda a, b, key: np.abs(out[a][key] - out[b][key]).max() / np.abs(out[b][key]).max()
    eg1, eg2 = rel(0.1, 0.05, 'g_end'), rel(0.05, 0.025, 'g_end')
    ey1, ey2 = rel(0.1, 0.05, 'y_end'), rel(0.05, 0.025, 'y_end')
    assert out[0.1]['g_end'].max() > 0.05
    print(f'\n[dt halving] {name}, delay {delay:g} ms: gains {eg1:.1e} (0.1 vs 0.05), {eg2:.1e} (0.05 vs 0.025); '
          f'y at presentation ends {ey1:.1e}, {ey2:.1e}; max g {out[0.1]["g_end"].max():.2f}')
    assert eg1 < 1e-3 and ey1 < 1e-3, (eg1, ey1)
    assert eg2 < eg1 and ey2 < ey1, (eg1, eg2, ey1, ey2)


# ============================================================================== 6. batches
def test_batched_run_equals_separate_runs():
    '''Per-batch theta_t (B, K), N_matrix (B, N_TOT, N_TOT) and inputs, with the delay: every row
    of a batched run must equal the corresponding single run (1e-13: round-off of BLAS batching).'''
    rng = np.random.default_rng(101)
    B = 3
    theta = rng.uniform(0.002, 0.02, (B, K))
    Nmat = rng.uniform(0.5, 1.5, (B, N_TOT, N_TOT))
    z = rng.uniform(0.0, 0.5, (B, 8, N_TOT))
    p = sv.ModelParams(tau_g=5.0, tau_mu=50.0, tau_v=5.0)
    q0 = rng.uniform(0, 0.05, (B, N_TOT))
    for opt in (sv.CODEBASE, sv.NEW):
        mb = small_model(opt, theta=theta, params=p, N_matrix=Nmat, surround_delay=1.3)
        rb = mb.simulate(z, 60, q_baseline=q0)
        assert rb['y_end'].shape == (B, 8, N_TOT) and rb['g_end'].shape == (B, 8, 2 * K)
        for b in range(B):
            m1 = small_model(opt, theta=theta[b], params=p, N_matrix=Nmat[b], surround_delay=1.3)
            r1 = m1.simulate(z[b], 60, q_baseline=q0[b:b + 1])
            assert np.abs(rb['y_end'][b] - r1['y_end'][0]).max() < 1e-13
            assert np.abs(rb['g_end'][b] - r1['g_end'][0]).max() < 1e-13
            assert np.abs(rb['state'][b] - r1['state'][0]).max() < 1e-13
        assert np.abs(rb['g_end'][0] - rb['g_end'][1]).max() > 1e-3       # the rows really differ


def test_batch_size_follows_theta_and_amplitude_arrays():
    z = np.full((1, 2, N_TOT), 0.1)
    m = small_model(sv.NEW, theta=np.full((4, K), 0.01))
    assert m.simulate(z, 5)['y_end'].shape[0] == 4
    m = small_model(sv.NEW, noise=sv.NoiseOptions('white_input', np.array([0.0, 0.01, 0.02])))
    assert m.simulate(z, 5)['y_end'].shape[0] == 3


@pytest.mark.parametrize('kind,amps', [('white_input', [0.0, 0.05]), ('white_response', [0.0, 0.05]),
                                       ('poisson_input', [np.inf, 50.0]), ('poisson_response', [np.inf, 50.0])])
def test_noise_free_row_of_an_amplitude_array_equals_the_noise_free_model_new_equations(kind, amps):
    '''Per-batch amplitude arrays: the row with zero noise (amplitude 0 for white noise, inf for
    Poisson) must reproduce the noise-free model, the other row must not. NEW equations
    (rectify_c=True). Tolerance 1e-12 (poisson_response takes sqrt(y+^2), a round-off difference).'''
    theta, z, _, params = _fast_scenario(seed=111, n_pres=6)
    p = sv.ModelParams(**params)
    ref = small_model(sv.NEW, theta=theta, params=p).simulate(z, 80)
    m = small_model(sv.NEW, theta=theta, params=p, noise=sv.NoiseOptions(kind, np.array(amps)))
    res = m.simulate(z, 80, rng=np.random.default_rng(5))
    assert res['y_end'].shape[0] == 2
    assert np.abs(res['y_end'][0] - ref['y_end'][0]).max() < 1e-12
    assert np.abs(res['g_end'][0] - ref['g_end'][0]).max() < 1e-12
    assert np.abs(res['y_end'][1] - ref['y_end'][0]).max() > 1e-4


@pytest.mark.parametrize('kind,amp', [('white_response', 0.0), ('poisson_response', np.inf)])
def test_FINDING_zero_response_noise_changes_the_codebase_equations(kind, amp):
    '''FINDING. A response-noise kind with zero noise (white amplitude 0; Poisson amplitude inf,
    documented as "no noise") must reproduce the noise-free model. With rectify_c=False (the
    codebase equations, c = W.T y) it does not: derivatives() feeds the rectified response to c
    and to the running mean whenever a response-noise kind is active, i.e. rectify_c is silently
    forced to True (simulation_variants.py lines 215-218).'''
    theta, z, _, params = _fast_scenario(seed=111, n_pres=6)
    p = sv.ModelParams(**params)
    ref = small_model(sv.CODEBASE, theta=theta, params=p).simulate(z, 80, record_traces=True)
    assert ref['traces']['y'].min() < -0.01                  # the scenario has negative responses
    m = small_model(sv.CODEBASE, theta=theta, params=p, noise=sv.NoiseOptions(kind, amp))
    res = m.simulate(z, 80, rng=np.random.default_rng(5))
    err_y = np.abs(res['y_end'] - ref['y_end']).max()
    err_g = np.abs(res['g_end'] - ref['g_end']).max()
    print(f'\n[finding: response noise forces rectify_c] {kind}: max |dy| {err_y:.2e}, max |dg| {err_g:.2e}')
    assert err_y < 1e-12 and err_g < 1e-12, (err_y, err_g)


def test_FINDING_poisson_input_noise_is_shared_across_a_batch_defined_by_theta():
    '''FINDING. The batch is documented as "independent simulations run in parallel". When the
    batch size comes from theta_t (or N_matrix) and z_pres has a single row, the Poisson input
    counts are drawn with the shape of z (1, N_TOT) and broadcast: every simulation receives the
    same noise sample (white_input draws (B, N_TOT)). Two rows with identical parameters must
    differ if their noise is independent.'''
    theta = np.full((2, K), 0.01)
    z = np.full((1, 3, N_TOT), 0.2)
    for kind, amp in (('white_input', 0.05), ('poisson_input', 50.0)):
        m = small_model(sv.NEW, theta=theta, noise=sv.NoiseOptions(kind, amp))
        y = m.simulate(z, 40, rng=np.random.default_rng(6))['y_end']
        diff = np.abs(y[0] - y[1]).max()
        print(f'\n[batch noise independence] {kind}: max |y_row0 - y_row1| = {diff:.2e}')
        assert diff > 1e-6, f'{kind}: the two simulations received identical noise'


def test_FINDING_q_of_without_extra_fails_for_white_response_noise():
    '''FINDING. q_of(state) (extra defaults to None; used by supplementary/common.blank_pool to
    build the delay baseline) raises KeyError('eta') when the model has white_response noise, so
    common.adapt() with a delay cannot be used with that noise kind.'''
    m = small_model(sv.NEW, noise=sv.NoiseOptions('white_response', 0.01))
    q = m.q_of(m.zero_state(1))
    assert q.shape == (1, N_TOT)


# ============================================================================== 7. white noise
def _chunked_states(model, z, n_chunks, steps, rng, state0=None):
    '''Runs n_chunks x steps and returns the state after every chunk (B, n_chunks, state_size).'''
    state = state0
    out = []
    for _ in range(n_chunks):
        state = model.simulate(z, steps, state0=state, adapt=False, rng=rng)['state']
        out.append(state)
    return np.stack(out, axis=1)


def test_white_input_noise_seen_by_slow_variables_is_dt_independent_and_has_the_stated_amplitude():
    '''Stated convention: the white input noise low-passed at tau_n has s.d. = amplitude, i.e.
    intensity s = amplitude sqrt(2 tau_n). With normalization off and zero gains the model is
    linear: tau_y dy/dt = -y + z + xi, tau_v dv/dt = -v + W.T y, so in continuous time
        var(v_k) = s^2 / (2 (tau_y + tau_v)) = amplitude^2 tau_n / (tau_y + tau_v)   (|w_k| = 1).
    tau_v = 4 ms to keep the run short. Statistics: 300 traces x 130 ms after a 30 ms burn-in,
    v^2 decorrelates in ~tau_v/2, so n_eff ~ 300 x 130 / 4 ~ 1e4 per interneuron, the relative
    s.e. of the s.d. is ~sqrt(1/(2 n_eff)) ~ 0.7%. Tolerance on each interneuron's s.d.: 4%
    (> 5 s.e.) against the formula, 5% between dt = 0.1 and 0.05 (independent samples, 5 s.e. x sqrt 2).'''
    amp, tau_n = 0.05, 1.0
    p = sv.ModelParams(tau_v=4.0)
    opt = sv.EquationOptions(normalization=False)
    target = amp * np.sqrt(tau_n / (p.tau_y + p.tau_v))
    sd = {}
    for dt in (0.1, 0.05):
        m = small_model(opt, params=p, dt=dt, noise=sv.NoiseOptions('white_input', amp, tau_n=tau_n))
        states = _chunked_states(m, np.zeros((300, 1, N_TOT)), 80, int(round(2.0 / dt)), np.random.default_rng(121))
        v = states[:, 15:, m.sl['v']]
        sd[dt] = v.std(axis=(0, 1))
        assert np.abs(sd[dt] / target - 1).max() < 0.04, (dt, sd[dt] / target)
    assert np.abs(sd[0.1] / sd[0.05] - 1).max() < 0.05
    print(f'\n[white input, slow variable v] s.d./analytic: dt 0.1 {np.round(sd[0.1] / target, 3)}, '
          f'dt 0.05 {np.round(sd[0.05] / target, 3)}')


@pytest.mark.parametrize('dt', [0.1, 0.05])
def test_FINDING_white_input_noise_in_recorded_y_depends_on_dt(dt):
    '''FINDING. Same linear model; y is exactly the first-order low-pass (tau_y) of z + xi, so by
    the stated convention with tau_n = tau_y the s.d. of y must equal the amplitude, at any dt.
    The noise is added as one kick per step after the RK4 update and y is recorded right after
    the kick, so the recorded variance is kick^2 / (1 - exp(-2 dt/tau_y)), i.e. inflated by
    x / (1 - exp(-x)), x = 2 dt / tau_y: +12.7% in s.d. at dt = 0.1 and +6.3% at dt = 0.05 ms
    (the time average over a step, which is what the slow variables integrate, is correct, see
    the previous test). The same inflated samples feed y_end, traces and the delay buffer (q_of).
    Statistics: 200 traces x 100 samples 1 ms apart (correlation exp(-2.5)) x 9 neurons =
    1.8e5 samples, relative s.e. of the s.d. ~0.2%. Tolerance 2% (10 s.e.).'''
    amp = 0.05
    p = sv.ModelParams()
    m = small_model(sv.EquationOptions(normalization=False), params=p, dt=dt,
                    noise=sv.NoiseOptions('white_input', amp, tau_n=p.tau_y))
    res = m.simulate(np.zeros((200, 105, N_TOT)), int(round(1.0 / dt)), adapt=False, rng=np.random.default_rng(122))
    sd = res['y_end'][:, 5:].std()
    x = 2 * dt / p.tau_y
    print(f'\n[white input, recorded y] dt {dt}: s.d./amplitude = {sd / amp:.4f} '
          f'(kick-after-step prediction {np.sqrt(x / (1 - np.exp(-x))):.4f})')
    assert abs(sd / amp - 1) < 0.02, sd / amp


def test_white_input_noise_enters_through_beta_and_scales_with_sqrt_dt():
    '''One step from a fixed state, noise on and off with everything else equal: the increment of
    y is beta * amplitude * sqrt(2 tau_n) * sqrt(dt) / tau_y * N(0, 1) (beta = 1 without
    normalization), per-batch amplitudes included. 4e4 samples per case: s.e. of the s.d. 0.35%,
    tolerance 2%.'''
    amps = np.array([0.02, 0.07])
    for norm in (True, False):
        for dt in (0.1, 0.025):
            p = sv.ModelParams()
            opt = replace(sv.NEW, normalization=norm)
            beta = p.beta if norm else 1.0
            tau_n = 3.0
            kicks = []
            for amp in amps:
                noisy = small_model(opt, dt=dt, noise=sv.NoiseOptions('white_input', np.full(4500, amp), tau_n=tau_n))
                clean = small_model(opt, dt=dt)
                z = np.full((1, 1, N_TOT), 0.1)
                y1 = noisy.simulate(z, 1, rng=np.random.default_rng(123))['y_end'][:, 0]
                y0 = clean.simulate(z, 1)['y_end'][:, 0]
                kicks.append(y1 - y0)
                expect = beta * amp * np.sqrt(2 * tau_n * dt) / p.tau_y
                assert abs(kicks[-1].std() / expect - 1) < 0.02, (norm, dt, amp, kicks[-1].std() / expect)
                assert abs(kicks[-1].mean()) < 5 * expect / np.sqrt(kicks[-1].size)


@pytest.mark.parametrize('dt', [0.1, 0.05])
def test_white_response_noise_is_an_exact_ou_process(dt):
    '''eta handed to derivatives(): stationary s.d. = amplitude from the first step, one-step
    correlation exp(-dt/tau_n), constant over the four RK4 stages, for a per-batch amplitude.
    300 traces x 9 neurons x 600 steps; n_eff ~ 300 x 9 x 600 dt / (2 tau_n) >= 4e4, s.e. of the
    s.d. <= 0.4%; tolerance 2%. Correlation: s.e. ~ sqrt((1 - rho^2) / n) ~ 3e-4, tolerance 2e-3.'''
    tau_n = 2.0
    amps = np.where(np.arange(300) < 150, 0.03, 0.08)
    m = small_model(sv.NEW, dt=dt, noise=sv.NoiseOptions('white_response', amps, tau_n=tau_n))
    calls = spy_on(m)
    m.simulate(np.full((1, 1, N_TOT), 0.1), 600, adapt=False, rng=np.random.default_rng(124))
    eta = np.stack([c['extra']['eta'] for c in calls])            # (4 x steps, B, N_TOT)
    assert np.array_equal(eta[0::4], eta[3::4])
    eta = eta[0::4]
    for lo, hi, amp in ((0, 150, 0.03), (150, 300, 0.08)):
        e = eta[:, lo:hi]
        assert abs(e[0].std() / amp - 1) < 0.06                   # first step: 1350 samples, s.e. 1.9%
        assert abs(e.std() / amp - 1) < 0.02
        rho = np.mean(e[1:] * e[:-1]) / np.mean(e * e)
        assert abs(rho - np.exp(-dt / tau_n)) < 2e-3, rho


def test_white_response_noise_is_added_to_sqrt_y_plus_in_every_downstream_term():
    '''derivatives() with a given eta against the reference equations with sqrt(y+) -> sqrt(y+) + eta
    in the recurrent drive, the pool (squared) and c; NEW equations (c is rectified there).'''
    rng = np.random.default_rng(125)
    theta = rng.uniform(0.001, 0.02, K)
    Nmat = rng.uniform(0.0, 2.0, (N_TOT, N_TOT))
    W_yy = 0.8 * np.eye(N_RF) + 0.1 * rng.standard_normal((N_RF, N_RF))
    p = sv.ModelParams()
    for opt in (sv.NEW, replace(sv.NEW, gating='a', gain_placement='principal', variance_signal='filtered')):
        m = small_model(opt, theta=theta, N_matrix=Nmat, W_yy=W_yy, noise=sv.NoiseOptions('white_response', 0.05))
        state = random_state(rng, 2, m)
        z = rng.standard_normal((2, N_TOT))
        eta = 0.05 * rng.standard_normal((2, N_TOT))
        d = m.derivatives(state, z, None, True, {'eta': eta})
        for b in range(2):
            ref = reference_rhs(W_SMALL, Nmat, W_yy, p, opt, theta, N_RF, N_SETS, split_state(m, state[b]),
                                z[b], eta=eta[b])
            got = split_state(m, d[b])
            for name in ('dy', 'du', 'da', 'dg', 'dv'):
                assert np.abs(got[name[1:]] - ref[name]).max() < 1e-12, (name, opt)
            dmu_c = np.stack([W_SMALL.T @ got['mu'][0], W_SMALL.T @ got['mu'][1]])
            assert np.abs(dmu_c - ref['dmu_c']).max() < 1e-12


# ============================================================================== 8. Poisson noise
@pytest.mark.parametrize('ref_level,bin_ms', [(1.0, 1.0), (2.0, 0.5)])
def test_poisson_input_estimate_is_unbiased_with_the_documented_variance(ref_level, bin_ms):
    '''z_hat = ref n / (R bin), n ~ Poisson(R bin z / ref): E[z_hat] = z and
    var(z_hat) = z ref / (R bin), R in spikes/s and bin in SECONDS (bin_ms x 1e-3).
    Observed through the linear model (normalization off, zero gains): y relaxes to z_hat within
    each bin, so at bin ends y is AR(1) with rho = exp(-bin/tau_y):
        E[y] = z,   var(y) = var(z_hat) (1 - rho) / (1 + rho).
    R = 200 spikes/s: lambda = R bin z / ref is 0.025-0.2 counts per bin.
    Tolerances: mean within 5 s.e. (s.e. from the analytic variance, with the AR(1) inflation
    (1 + rho)/(1 - rho)); variance within 12%: for small lambda the relative s.e. of a Poisson
    variance estimate is ~1/sqrt(lambda n), n = 400 x 190 samples, lambda >= 0.025: <= 2.3%
    (5 s.e. = 11.5%). A wrong time unit would be off by a factor 1e3.'''
    R = 200.0
    p = sv.ModelParams()
    dt = 0.1
    bin_steps = int(round(bin_ms / dt))
    m = small_model(sv.EquationOptions(normalization=False), dt=dt,
                    noise=sv.NoiseOptions('poisson_input', R, ref=ref_level, bin_ms=bin_ms))
    z = np.linspace(0.25, 1.0, N_TOT)
    B, n_bins = 400, 200
    res = m.simulate(np.broadcast_to(z, (B, n_bins, N_TOT)), bin_steps, adapt=False, rng=np.random.default_rng(131))
    y = res['y_end'][:, 10:]
    n = y.shape[0] * y.shape[1]
    rho = np.exp(-bin_ms / p.tau_y)
    var_z = z * ref_level / (R * bin_ms * 1e-3)
    var_y = var_z * (1 - rho) / (1 + rho)
    se_mean = np.sqrt(var_y * (1 + rho) / (1 - rho) / n)
    zscore = (y.mean(axis=(0, 1)) - z) / se_mean
    assert np.abs(zscore).max() < 5, zscore
    ratio = y.var(axis=(0, 1)) / var_y
    assert np.abs(ratio - 1).max() < 0.12, ratio
    print(f'\n[poisson input] ref {ref_level}, bin {bin_ms} ms: mean bias z-scores in [{zscore.min():.2f}, '
          f'{zscore.max():.2f}], max rel. bias {np.abs(y.mean(axis=(0, 1)) / z - 1).max():.1e}; '
          f'var(y)/predicted in [{ratio.min():.3f}, {ratio.max():.3f}]')


def test_poisson_input_infinite_rate_is_noise_free():
    theta, z, _, params = _fast_scenario(seed=132, n_pres=6)
    p = sv.ModelParams(**params)
    for opt in (sv.CODEBASE, sv.NEW):
        ref = small_model(opt, theta=theta, params=p).simulate(z, 80)
        res = small_model(opt, theta=theta, params=p, noise=sv.NoiseOptions('poisson_input', np.inf)).simulate(z, 80)
        assert np.array_equal(res['y_end'], ref['y_end']) and np.array_equal(res['g_end'], ref['g_end'])


def test_FINDING_poisson_input_is_noise_free_until_the_next_bin_after_a_stimulus_change():
    '''FINDING. "z replaced, in every bin, by ref n / (R bin)". z is reset to the noise-free drive
    at the start of every presentation and only resampled when t % bin_steps == 0, so when the
    presentation length is not a multiple of the bin, the steps between a presentation onset and
    the next bin boundary use the noise-free z (simulation_variants.py lines 339, 344-346).
    Here: bin 1 ms (10 steps), presentations of 25 steps; with R = 50 spikes/s the estimate is a
    multiple of ref / (R bin) = 20, so a noisy z can never equal the drive 0.3.'''
    m = small_model(sv.NEW, noise=sv.NoiseOptions('poisson_input', 50.0, bin_ms=1.0))
    calls = spy_on(m)
    z = np.full((1, 4, N_TOT), 0.3)
    m.simulate(z, 25, adapt=False, rng=np.random.default_rng(133))
    clean_steps = [t for t in range(100) if np.array_equal(calls[4 * t]['z'], z[:, 0])]
    print(f'\n[poisson input bins] steps integrated with the noise-free drive: {clean_steps}')
    assert clean_steps == [], f'{len(clean_steps)} of 100 steps used the noise-free input'


def test_poisson_response_rate_estimate_is_unbiased_with_the_documented_variance():
    '''y+_hat = ref n / (R bin), n ~ Poisson(R bin y+ / ref), y+ = max(y, 0)^2 read at the start of
    the bin and held for the bin. With normalization off and zero gains y sits at z regardless of
    the noise, so y+ is known. E[y+_hat] = y+ (5 s.e.), var = y+ ref / (R bin[s]) (12%, as for the
    Poisson input), y+_hat = 0 where y < 0, and the estimate is constant within a bin.
    Also reports (does not assert) the bias of sqrt(y+_hat), the quantity that the recurrent
    drive and c actually receive.'''
    R, bin_ms, dt = 200.0, 1.0, 0.1
    m = small_model(sv.EquationOptions(normalization=False), dt=dt, noise=sv.NoiseOptions('poisson_response', R, bin_ms=bin_ms))
    z = np.linspace(-0.2, 1.0, N_TOT)
    B = 400
    st = m.zero_state(B)
    st[:, m.sl['y']] = z
    calls = spy_on(m)
    m.simulate(np.broadcast_to(z, (B, 1, N_TOT)), 1500, state0=st, adapt=False, rng=np.random.default_rng(134))
    rate = np.stack([c['extra']['rate_hat'] for c in calls[0::4]])       # (steps, B, N_TOT)
    ys = np.stack([c['state'][:, m.sl['y']] for c in calls[0::4]])
    assert np.abs(ys - z).max() < 1e-12
    for k in range(1, 10):
        assert np.array_equal(rate[k::10], rate[0::10]), 'the estimate changed within a bin'
    r = rate[0::10]                                                      # (150 bins, B, N_TOT)
    n = r.shape[0] * r.shape[1]
    yplus = np.maximum(z, 0) ** 2
    pos = yplus > 0
    assert np.all(r[..., ~pos] == 0)
    var = yplus * 1.0 / (R * bin_ms * 1e-3)
    zscore = (r.mean(axis=(0, 1))[pos] - yplus[pos]) / np.sqrt(var[pos] / n)
    assert np.abs(zscore).max() < 5, zscore
    big = yplus * R * bin_ms * 1e-3 >= 0.025                              # enough counts for the variance check
    ratio = r.var(axis=(0, 1))[big] / var[big]
    assert np.abs(ratio - 1).max() < 0.12, ratio
    sqrt_ratio = np.sqrt(r).mean(axis=(0, 1))[pos] / np.sqrt(yplus[pos])
    print(f'\n[poisson response] R = {R:g}/s, bin {bin_ms} ms: mean(y+_hat)/y+ in '
          f'[{(r.mean(axis=(0, 1))[pos] / yplus[pos]).min():.3f}, {(r.mean(axis=(0, 1))[pos] / yplus[pos]).max():.3f}], '
          f'z-scores within {np.abs(zscore).max():.2f}; var ratio in [{ratio.min():.3f}, {ratio.max():.3f}]; '
          f'mean(sqrt(y+_hat))/sqrt(y+) = {np.round(sqrt_ratio, 3)} for sqrt(y+) = {np.round(np.sqrt(yplus[pos]), 2)}')
