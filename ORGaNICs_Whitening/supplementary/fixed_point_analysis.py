'''
fixed_point_analysis.py

Supplementary Note 3: is it valid to read probe responses from the analytical fixed point
(Eq. S17, response_tools.get_response_offline) instead of simulating them? Figs. 1, 2 and 4 of the
main text use the analytical fixed point; Fig. 3 reads responses from the simulation.

Protocol (power-law setup, analysis/power_law.py: von Mises ensemble, small stimuli covering the
cRF only, contrast 1, 30 ms per stimulus, mid-contrast target, no surround delay):
  1. Adapt V1Dynamics_Surround from a zero state for ADAPT_MS ms.
  2. Keep g and mu from the end of adaptation; reset y, u, a, v to 0.
  3. Present one small probe (cRF only) for PROBE_MS ms with the full dynamics running
     (g and mu keep evolving), and compare the cRF firing rates y+ = floor(y)^2 with the
     analytical fixed point evaluated with the instantaneous gains g(t).

Run from ORGaNICs_Whitening/:
    python supplementary/fixed_point_analysis.py           # simulate, save .npz, plot
    python supplementary/fixed_point_analysis.py --plot    # replot from the saved .npz
Writes supplementary/results/fixed_point_analysis.npz (+ _params.json) and
supplementary/figures/fig_fixed_point_analysis.{pdf,png}.
'''

import argparse
import datetime
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, 'analysis'))
sys.path.insert(0, HERE)

import plotstyle as ps                                          # noqa: E402  (Agg backend)
import matplotlib.pyplot as plt                                 # noqa: E402

from simulation import Frame, V1Dynamics_Surround               # noqa: E402
from stimuli import StimulusGenerator                           # noqa: E402
from tunings import V1Tunings                                   # noqa: E402
from response_tools import (offline_gain_operator, to_steps, frozen_derivatives,  # noqa: E402
                            get_response_traced)

# ---- Configuration (as in analysis/power_law.py) ----
SEED = 0
DT = 0.1                      # ms
N_RF, N_SETS = 13, 6
FRAME_PATH = os.path.join(REPO_ROOT, 'data/frames/N13_mercedes_K182_Frame.csv')
TARGET_COV_PATH = os.path.join(REPO_ROOT, 'data/target_covs/uniform_target_covariance_mid_c.csv')
ADAPT_LOC = 'adapt CRF only'  # small stimuli
ADAPT_CONTRAST = 1.0
STIMULUS_MS = 30.0            # each ensemble stimulus is held for 30 ms
ENV_CENTER_DEG, ENV_KAPPA = 90.0, 3.0   # power_law.py environment B
ADAPT_MS = 2000.0             # adaptation before the probe
PROBE_MS = 100.0              # probe window
PROBE_THETA_DEG = 45.0        # probe orientation
PROBE_CONTRAST = 0.8          # as power_law.PROBE_CONTRAST
N_GAINS_SHOWN = 8             # gains plotted: the largest cRF gains at the start of the probe

RESULTS = os.path.join(HERE, 'results', 'fixed_point_analysis.npz')
FIG_BASE = os.path.join(HERE, 'figures', 'fig_fixed_point_analysis')


def build():
    tunings = V1Tunings(N=N_RF)
    frame = Frame(csv_path=FRAME_PATH)
    dyn = V1Dynamics_Surround(tunings, frame, dt=DT, N_RF=N_RF, N_SETS=N_SETS,
                              target_covariance_path=TARGET_COV_PATH, gains_nonneg=True)
    dyn.calibrate_theta_t(C_zz_uniform=dyn.uniform_target_covariance, uniform_target=True)
    assert dyn.n_lag == 0, 'this analysis uses no surround delay'
    return dyn


def analytic_rates(dyn, z, g_cRF, g_surround):
    '''Eq. S17 with the given gains: membrane potentials y* and z' (both N_TOT).'''
    M = offline_gain_operator(dyn, g_cRF, g_surround)
    z_prime = np.linalg.solve(np.eye(z.size) + M, z)
    y_star = z_prime / np.sqrt(dyn.sigma ** 2 + dyn.N_matrix @ z_prime ** 2)
    return y_star, z_prime


def fast_spectrum(dyn, z, g_cRF, g_surround, mu_cRF, mu_surround, settle_ms=1000.0):
    '''Eigenvalues (1/ms) of the Jacobian of the fast subsystem (y, u, a, v; gains frozen) at the
    fixed point reached by settling the frozen-gain dynamics for settle_ms. Central differences.'''
    y, u, a, v_c, v_s = get_response_traced(dyn, z, g_cRF, g_surround, mu_cRF, mu_surround,
                                            n_steps=to_steps(settle_ms, DT, 'settle_ms'))
    x0 = np.concatenate([y[:, -1], u[:, -1], a[:, -1], v_c[:, -1], v_s[:, -1]])
    f = lambda x: frozen_derivatives(x, z, dyn, g_cRF, g_surround)
    residual = np.abs(f(x0)).max()
    J = np.zeros((x0.size, x0.size))
    for j in range(x0.size):
        h = 1e-7 * max(1.0, abs(x0[j]))
        e = np.zeros_like(x0)
        e[j] = h
        J[:, j] = (f(x0 + e) - f(x0 - e)) / (2 * h)
    return np.linalg.eigvals(J), residual


def simulate():
    np.random.seed(SEED)
    dyn = build()
    adapt_steps = to_steps(ADAPT_MS, DT, 'ADAPT_MS')
    probe_steps = to_steps(PROBE_MS, DT, 'PROBE_MS')
    stim_steps = to_steps(STIMULUS_MS, DT, 'STIMULUS_MS')
    # The generator rounds down to whole stimuli, so generate one extra stimulus and trim
    stim = StimulusGenerator(N_RF=N_RF, N_SETS=N_SETS, num_angles=N_RF,
                             stream_length=adapt_steps + stim_steps, contrast=ADAPT_CONTRAST)

    # 1. Ensemble adaptation from a zero state
    stream = stim.generate_surround_ensembles(ADAPT_LOC, von_mises=True,
                                              von_mises_center=ENV_CENTER_DEG,
                                              von_mises_kappa=ENV_KAPPA,
                                              duration=stim_steps)[:, :adapt_steps]
    assert stream.shape[1] == adapt_steps
    dyn.run_simulation(stream)
    adapted = dyn.unpack_state(dyn.last_state)

    # 2. Keep g and mu, reset y, u, a, v
    state0 = np.zeros(dyn.state_size)
    for name, off in [('g_cRF', dyn._off_g_cRF), ('g_surround', dyn._off_g_surround),
                      ('mu_cRF', dyn._off_mu_cRF), ('mu_surround', dyn._off_mu_surround)]:
        state0[off:off + dyn.frame.K] = adapted[name]

    # 3. Probe with the full dynamics
    z = stim.oriented_drive(np.radians(PROBE_THETA_DEG), PROBE_CONTRAST, ADAPT_LOC)
    (y_hist, _, _, _, _, g_cRF_hist, g_surround_hist, *_) = dyn.run_simulation(
        np.repeat(z[:, None], probe_steps, axis=1), initial_state=state0)
    assert np.all(g_cRF_hist >= 0) and np.all(g_surround_hist >= 0)

    # Control: the same probe with every gain at zero (single-layer ORGaNICs; tau_g -> inf keeps g = 0)
    tau_g = dyn.tau_g
    dyn.tau_g = np.inf
    try:
        y_hist_g0 = dyn.run_simulation(np.repeat(z[:, None], probe_steps, axis=1))[0]
    finally:
        dyn.tau_g = tau_g

    # Linear stability of the fast subsystem at the fixed point (gains frozen at probe onset)
    eig, residual = fast_spectrum(dyn, z, adapted['g_cRF'], adapted['g_surround'],
                                  adapted['mu_cRF'], adapted['mu_surround'])

    # Analytical fixed point at every step, with the instantaneous gains
    y_star = np.zeros_like(y_hist)
    zp_min = np.inf
    for t in range(probe_steps):
        ys, zp = analytic_rates(dyn, z, g_cRF_hist[:, t], g_surround_hist[:, t])
        y_star[:, t] = ys
        zp_min = min(zp_min, zp.min())
    # ... and with the gains frozen at the start of the probe (what the main-text figures use)
    y_star0, zp0 = analytic_rates(dyn, z, adapted['g_cRF'], adapted['g_surround'])

    rate = y_hist ** 2                       # y_hist is already rectified: firing rate y+ = floor(y)^2
    rate_star = np.maximum(y_star, 0) ** 2
    crf = slice(0, N_RF)
    mse = ((rate[crf] - rate_star[crf]) ** 2).mean(axis=0)
    mse0 = ((rate[crf] - (np.maximum(y_star0, 0) ** 2)[crf, None]) ** 2).mean(axis=0)

    t_ms = (np.arange(probe_steps) + 1) * DT
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    np.savez(RESULTS, t_ms=t_ms, rate_cRF=rate[crf], rate_star_cRF=rate_star[crf],
             rate_star0_cRF=(np.maximum(y_star0, 0) ** 2)[crf], g_cRF=g_cRF_hist,
             g_cRF_start=adapted['g_cRF'], mse=mse, mse_frozen=mse0, z_prime_min=zp_min,
             z_prime0=zp0, theta_t=dyn.theta_t, rate_cRF_g0=(y_hist_g0 ** 2)[crf],
             eig=eig, fp_residual=residual)
    params = dict(seed=SEED, dt_ms=DT, N_RF=N_RF, N_SETS=N_SETS, frame=os.path.basename(FRAME_PATH),
                  target_cov=os.path.basename(TARGET_COV_PATH), adapt_loc=ADAPT_LOC,
                  adapt_contrast=ADAPT_CONTRAST, stimulus_ms=STIMULUS_MS,
                  env_center_deg=ENV_CENTER_DEG, env_kappa=ENV_KAPPA, adapt_ms=ADAPT_MS,
                  probe_ms=PROBE_MS, probe_theta_deg=PROBE_THETA_DEG, probe_contrast=PROBE_CONTRAST,
                  surround_delay_ms=dyn.surround_delay,
                  tau_ms=dict(y=dyn.tau_y, a=dyn.tau_a, u=dyn.tau_u, v=dyn.tau_v, g=dyn.tau_g,
                              mu=dyn.tau_mu),
                  sigma=dyn.sigma, beta=dyn.beta,
                  git_commit=subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT,
                                            capture_output=True, text=True).stdout.strip(),
                  date=datetime.datetime.now().isoformat(timespec='seconds'))
    with open(RESULTS.replace('.npz', '_params.json'), 'w') as f:
        json.dump(params, f, indent=2)
    print('saved', RESULTS)


def report(d):
    t, mse = d['t_ms'], d['mse']
    peak = d['rate_star_cRF'][:, -1].max()
    print(f"min z' over the probe: {float(d['z_prime_min']):.4g}")
    print(f"peak fixed-point rate (cRF): {peak:.4g}")
    for tq in (10, 25, 50, 75, 100):
        i = np.searchsorted(t, tq - 1e-9)
        print(f"  t = {t[i]:5.1f} ms: MSE = {mse[i]:.3e}  RMSE / peak rate = {np.sqrt(mse[i]) / peak:.3e}"
              f"  (frozen-gain fixed point: {d['mse_frozen'][i]:.3e})")
    eig = d['eig']
    slow = eig[np.argmax(eig.real)]
    osc = eig[np.abs(eig.imag) > 1e-9]
    dom = osc[np.argmax(osc.real)] if osc.size else np.nan
    print(f"fast subsystem: {eig.size} eigenvalues, max Re = {eig.real.max():.4g} /ms "
          f"(decay time {-1 / eig.real.max():.3g} ms), residual |f(x*)| = {float(d['fp_residual']):.2e}")
    print(f"  slowest mode {slow:.4g} /ms; least-damped oscillatory mode {dom:.4g} /ms -> "
          f"{abs(dom.imag) / (2 * np.pi) * 1e3:.3g} Hz, decay time {-1 / dom.real:.3g} ms")
    g0r = d['rate_cRF_g0']
    print(f"g = 0 control: peak transient {g0r.max():.3g}, rate at 100 ms max {g0r[:, -1].max():.3g} "
          f"(overshoot x{g0r.max() / g0r[:, -1].max():.2g}); adapted: peak {d['rate_cRF'].max():.3g}, "
          f"x{d['rate_cRF'].max() / d['rate_cRF'][:, -1].max():.2g}")
    g0, g = d['g_cRF_start'], d['g_cRF']
    print(f"cRF gains: {np.sum(g0 > 0)} of {g0.size} positive at probe onset; "
          f"max |g(100 ms) - g(0)| = {np.abs(g[:, -1] - g0).max():.3e} (max g = {g0.max():.3e})")


def plot():
    d = np.load(RESULTS)
    report(d)
    ps.apply()
    t = d['t_ms']
    peak = d['rate_star_cRF'][:, -1].max()   # peak fixed-point rate across the cRF neurons

    fig, axes = plt.subplots(1, 3, figsize=(ps.FULL_WIDTH, 55 * ps.MM),
                             gridspec_kw=dict(wspace=0.5))
    ax_g, ax_y, ax_e = axes
    fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.22)

    # Gains: the largest cRF gains at the start of the probe, shades of green
    idx = np.argsort(d['g_cRF_start'])[::-1][:N_GAINS_SHOWN]
    greens = plt.get_cmap('Greens')(np.linspace(0.45, 0.95, N_GAINS_SHOWN))
    for c, k in zip(greens, idx):
        ax_g.plot(t, d['g_cRF'][k], color=c, lw=1.0)
    ax_g.set_ylabel('Gain subset')

    # cRF firing rates, shades of red
    reds = plt.get_cmap('Reds')(np.linspace(0.35, 0.95, N_RF))
    for i in range(N_RF):
        ax_y.plot(t, d['rate_cRF'][i], color=reds[i], lw=1.0)
    ax_y.set_ylabel(r'Firing rate, $y^{+}$')

    # RMSE between simulated and analytical rates, relative to the peak fixed-point rate
    ax_e.semilogy(t, np.sqrt(d['mse']) / peak, color='black', lw=1.25)
    ax_e.set_ylabel('RMSE / peak rate')

    for ax in axes:
        ax.set_xlim(0, t[-1])
        ax.set_xticks([0, 50, 100])
        ax.set_xlabel('Time (ms)')

    ps.label_panels(fig, axes)
    os.makedirs(os.path.dirname(FIG_BASE), exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{FIG_BASE}.{ext}', dpi=ps.DPI)
    print('saved', FIG_BASE + '.{pdf,png}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--plot', action='store_true', help='replot from the saved results only')
    if not ap.parse_args().plot:
        simulate()
    plot()
