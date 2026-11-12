'''
figS4_power_law.py  ->  figures/FigS4_power_law.png, data/FigS4_power_law.npz

Does normalization break the power law between response ratio and stimulus-probability ratio?

Protocol (analysis/Dario_plots.py, Fig. 4 of the main text), with the new equations (Eqs. 1-6):
three von Mises orientation environments A (peak 30 deg, kappa 6), B (90 deg, kappa 3),
C (150 deg, kappa 1.5); small adaptors (cRF only), contrast 0.8, 30 ms per presentation,
140 presentations (4.2 s), no surround delay; target = 0.15 x uniform_target_covariance_mid_c.
Gains are frozen at the end of adaptation and steady-state responses to large test gratings at
the 13 preferred orientations are computed (analytic fixed point; checked against the full
dynamics, see `readout_check`). Response magnitude r = l2 norm of the cRF firing rates
(max(y,0)^2). Power law fitted without intercept, pooled over the pairs A/B, A/C, B/C.

Conditions: normalization on (Eqs. 1-6) and off (Eqs. 9-12: beta = 1, no u, a, W_yy, ungated
feedback, same theta_t). 10 trials: seeds 300-309 (A), 400-409 (B), 500-509 (C).
Part 2: test contrast varied from 1% to 100% (adapting contrast fixed at 0.8).

Run:  python figS4_power_law.py [--recompute]     (about 3 minutes on 4 cores)
'''

import sys
from dataclasses import replace
from multiprocessing import Pool

import numpy as np
from scipy.special import i0

import common as C
import plotstyle as ps
import simulation_variants as sv

NAME = 'FigS4_power_law'
ENVS = {'A': (30.0, 6.0), 'B': (90.0, 3.0), 'C': (150.0, 1.5)}
PAIRS = [('A', 'B'), ('A', 'C'), ('B', 'C')]
N_TRIALS = 10
N_PRES, PRES_MS = 140, 30.0
TEST_CONTRASTS = np.logspace(-2, 0, 13)
RANGE = 1.0
K = 182
MID_TARGET = C.TARGET_SCALE * np.loadtxt(C.TARGET_COV_PATH.replace('low_c', 'mid_c'), delimiter=',')
PROBE_ANGLES = np.linspace(0, np.pi, C.N_RF, endpoint=False)


def density(theta, center_deg, kappa):
    '''Orientation density of the folded von Mises environments (Dario_plots.py).'''
    return np.cosh(kappa * np.cos(theta - np.deg2rad(center_deg))) / (np.pi * i0(kappa))


def model(norm):
    theta = sv.theta_t_from_target(C.load_frame(), MID_TARGET)
    return C.make_model(replace(sv.NEW, normalization=norm), theta_t=theta)


def adapt(norm):
    z = np.stack([C.ensemble_presentations(N_PRES, False, 300 + 100 * e + i, C.SMALL, von_mises=True,
                                           von_mises_center=ENVS[env][0], von_mises_kappa=ENVS[env][1])[0]
                  for e, env in enumerate(ENVS) for i in range(N_TRIALS)])
    res = C.adapt(model(norm), z, pres_ms=PRES_MS, delay=0.0, record_g_every_pres=10)
    return norm, res['g_end'].astype(np.float32)          # (3 N_TRIALS, 14, 2K), environment-major


def magnitudes(norm, g, contrast, method='analytic'):
    '''Response magnitude to each probe orientation for every adapted state: (n_states, 13).'''
    m = model(norm)
    z = C.oriented(PROBE_ANGLES, contrast, C.LARGE)
    out, zmin = [], np.inf
    for gi in g:
        if method == 'analytic':
            fp = m.analytic_fixed_point(z, gi[:K], gi[K:])
            y, zmin = fp['y'], min(zmin, fp['z_prime'][:, :C.N_RF].min())
        else:
            y = m.settle(z, gi[:K], gi[K:], settle_ms=600.0)['y_end'][:, 0]
        out.append(np.linalg.norm(np.maximum(y[:, :C.N_RF], 0) ** 2, axis=1))
    return np.array(out), zmin


def fit_power_law(x, y):
    beta = np.sum(x * y) / np.sum(x ** 2)
    return beta, 1 - np.sum((y - beta * x) ** 2) / np.sum((y - y.mean()) ** 2)


def log_ratios(mag):
    '''mag: (3, 13) magnitudes for environments A, B, C -> pooled (log10 p ratio, log10 r ratio).'''
    prob = {e: density(PROBE_ANGLES, *ENVS[e]) for e in ENVS}
    r = dict(zip(ENVS, mag))
    x = np.concatenate([np.log10(prob[a] / prob[b]) for a, b in PAIRS])
    y = np.concatenate([np.log10(r[a] / r[b]) for a, b in PAIRS])
    return x, y


def fits(mag_trials):
    '''mag_trials: (3 N_TRIALS, 13) -> per trial (beta_full, R2_full, beta_in, R2_in).'''
    out = []
    for t in range(N_TRIALS):
        x, y = log_ratios(mag_trials[t::N_TRIALS][:3] if False else mag_trials.reshape(3, N_TRIALS, -1)[:, t])
        keep = np.abs(x) <= RANGE
        out.append([*fit_power_law(x, y), *fit_power_law(x[keep], y[keep])])
    return np.array(out)


def simulate():
    with Pool(2) as pool:
        gains = dict(pool.map(adapt, [True, False]))
    out = dict(test_contrasts=TEST_CONTRASTS, n_trials=N_TRIALS)
    for norm in (True, False):
        tag = 'on' if norm else 'off'
        g = gains[norm].astype(float)
        out[f'g_{tag}'] = gains[norm]
        mags, zmins = zip(*[magnitudes(norm, g[:, -1], c) for c in TEST_CONTRASTS])
        out[f'mag_{tag}'] = np.array(mags)                     # (n_contrasts, 3 N_TRIALS, 13)
        out[f'zprime_min_{tag}'] = np.array(zmins)
    # readout check: analytic fixed point against the full dynamics, first trial of each environment
    i80 = int(np.argmin(np.abs(TEST_CONTRASTS - 0.8)))
    for norm in (True, False):
        tag = 'on' if norm else 'off'
        g = gains[norm].astype(float)[::N_TRIALS, -1]
        dyn, _ = magnitudes(norm, g, TEST_CONTRASTS[i80], method='dynamics')
        out[f'readout_check_{tag}'] = np.abs(dyn / out[f'mag_{tag}'][i80][::N_TRIALS] - 1).max()
    return out


def analyse(d):
    a = {}
    i80 = int(np.argmin(np.abs(d['test_contrasts'] - 0.8)))
    for tag in ('on', 'off'):
        a[f'fits_{tag}'] = np.array([fits(m) for m in d[f'mag_{tag}']])    # (n_contrasts, N_TRIALS, 4)
        a[f'fits80_{tag}'] = a[f'fits_{tag}'][i80]
        a[f'xy_{tag}'] = log_ratios(d[f'mag_{tag}'][i80].reshape(3, N_TRIALS, -1)[:, 0])
        g = d[f'g_{tag}'].astype(float)[:, :, :K]
        a[f'gmean_{tag}'] = g.mean(axis=(0, 2))
        # fits at every saved checkpoint of the adaptation run (test contrast 0.8)
        gfull = d[f'g_{tag}'].astype(float)
        a[f'fits_t_{tag}'] = np.array([fits(magnitudes(tag == 'on', gfull[:, k], 0.8)[0])
                                       for k in range(gfull.shape[1])])     # (n_checkpoints, N_TRIALS, 4)
        a[f'active_{tag}'] = (g[:, -1] > 1e-3).sum(axis=1)
    a['i80'] = i80
    return a


def plot(d, a):
    import matplotlib.pyplot as plt
    ps.apply()
    col = {'on': ps.COLORS['norm_on'], 'off': ps.COLORS['norm_off']}
    fig, ((ax_r, ax_b), (ax_c, ax_t)) = plt.subplots(2, 2, figsize=(ps.FULL_WIDTH, 125 * ps.MM))

    def violins(ax, column, ylabel):
        for i, tag in enumerate(('on', 'off')):
            vals = a[f'fits80_{tag}'][:, column]
            parts = ax.violinplot(vals, positions=[i], widths=0.7, showextrema=False)
            for body in parts['bodies']:
                body.set_facecolor(col[tag]); body.set_edgecolor(col[tag]); body.set_alpha(0.35); body.set_linewidth(1.0)
            rng = np.random.default_rng(i)
            ax.scatter(i + rng.uniform(-0.09, 0.09, len(vals)), vals, s=9, color=col[tag], zorder=3, linewidths=0)
            ax.plot([i - 0.2, i + 0.2], [np.median(vals)] * 2, color='k', lw=1.5, zorder=4)
        ax.set_xticks([0, 1], labels=['on', 'off'])
        ax.set_xlim(-0.6, 1.6)
        ax.set_xlabel('Normalization')
        ax.set_ylabel(ylabel)

    violins(ax_r, 1, r'Power-law fit, $R^2$ (full range)')
    ax_r.set_ylim(0, 1.02)
    ax_r.set_yticks([0, 0.5, 1])
    ax_b.axhspan(-0.4, -0.15, color=ps.COLORS['reference'], alpha=0.22, lw=0, zorder=0)
    violins(ax_b, 0, r'Exponent $\beta$ (full range)')
    ax_b.text(0.5, -0.275, 'experiment', fontsize=8, color='#4D4D4D', ha='center', va='center')

    c = d['test_contrasts'] * 100
    for column, ls, label in ((1, '-', 'full range'), (3, (0, (4, 2)), r'$|\log_{10}(p_X/p_Y)| \leq 1$')):
        m_ = a['fits_on'][:, :, column].mean(axis=1)
        s_ = a['fits_on'][:, :, column].std(axis=1, ddof=1) / np.sqrt(N_TRIALS)       # s.e.m. over trials
        ax_c.fill_between(c, m_ - s_, m_ + s_, color=col['on'], alpha=0.2, lw=0)
        ax_c.plot(c, m_, color=col['on'], lw=ps.LW_MAIN, ls=ls, label=label)
    ax_c.plot(c, a['fits_off'][:, :, 1].mean(axis=1), color=col['off'], lw=ps.LW_MAIN, label='normalization off,\nfull range')
    ps.ref_line(ax_c, 80.0, axis='v')
    ax_c.set_xscale('log')
    ax_c.set_xlim(1, 100)
    ax_c.set_xticks([1, 10, 100], labels=['1', '10', '100'])
    ax_c.set_ylim(-0.45, 1.02)
    ax_c.set_yticks([0, 0.5, 1])
    ps.ref_line(ax_c, 0.0)
    ax_c.set_xlabel('Test contrast (%)')
    ax_c.set_ylabel(r'Power-law fit, $R^2$')
    ax_c.legend(loc='center left', fontsize=8, handlelength=2.2, bbox_to_anchor=(0.0, 0.46), labelspacing=0.2)
    # d: fit quality versus adaptation time
    t = (np.arange(a['fits_t_on'].shape[0]) + 1) * 10 * PRES_MS / 1000.0
    for tag, label in (('on', 'normalization on'), ('off', 'normalization off')):
        m_, s_ = a[f'fits_t_{tag}'][:, :, 1].mean(axis=1), a[f'fits_t_{tag}'][:, :, 1].std(axis=1, ddof=1)
        ax_t.fill_between(t, m_ - s_, m_ + s_, color=col[tag], alpha=0.2, lw=0)
        ax_t.plot(t, m_, color=col[tag], lw=ps.LW_MAIN, label=label)
    ax_t.set_xlim(0, 4.3)
    ax_t.set_xticks([0, 1, 2, 3, 4])
    ax_t.set_ylim(0, 1.02)
    ax_t.set_yticks([0, 0.5, 1])
    ax_t.set_xlabel('Adaptation time (s)')
    ax_t.set_ylabel(r'Power-law fit, $R^2$ (full range)')
    ax_t.legend(loc='lower left', fontsize=8)
    fig.subplots_adjust(left=0.09, right=0.985, top=0.95, bottom=0.095, wspace=0.34, hspace=0.42)
    ps.label_panels(fig, [ax_r, ax_b, ax_c, ax_t])
    return ps.save(fig, NAME)


def report(d, a):
    for tag in ('on', 'off'):
        f = a[f'fits80_{tag}']
        gm = a[f'gmean_{tag}']
        print(f'  normalization {tag}: full range beta = {f[:, 0].mean():.3f} +/- {f[:, 0].std(ddof=1):.3f}, '
              f'R2 = {f[:, 1].mean():.3f} +/- {f[:, 1].std(ddof=1):.3f} (min {f[:, 1].min():.3f}, max {f[:, 1].max():.3f}); '
              f'|x| <= 1: beta = {f[:, 2].mean():.3f} +/- {f[:, 2].std(ddof=1):.3f}, R2 = {f[:, 3].mean():.3f} +/- {f[:, 3].std(ddof=1):.3f}')
        print(f'     mean cRF gain at 0.3, 2.1, 4.2 s: {gm[0]:.4f}, {gm[6]:.4f}, {gm[-1]:.4f}; growth over the last 0.9 s '
              f'{100 * (gm[-1] - gm[-4]) / max(gm[-1], 1e-12) / 0.9:.1f} %/s; active gains (> 1e-3) per run: '
              f'{a[f"active_{tag}"].min()}-{a[f"active_{tag}"].max()} of {K}; min z\' over probes {d[f"zprime_min_{tag}"].min():+.1e}; '
              f'analytic vs dynamics readout {float(d[f"readout_check_{tag}"]):.1e}')
        x, _ = a[f'xy_{tag}']
    print(f'  range of log10(pX/pY): {x.min():.2f} to {x.max():.2f}; points with |x| <= 1: {(np.abs(x) <= 1).sum()} of {x.size}')
    c = d['test_contrasts']
    for column, label in ((1, 'full'), (3, '|x|<=1')):
        r = a['fits_on'][:, :, column].mean(axis=1)
        print(f'  normalization on, R2 ({label}) vs test contrast: ' + ', '.join(f'{100 * ci:.0f}%: {ri:.2f}' for ci, ri in zip(c[::2], r[::2])))
    b = a['fits_on'][:, :, 0].mean(axis=1)
    print('  normalization on, beta (full) vs test contrast: ' + ', '.join(f'{100 * ci:.0f}%: {bi:.2f}' for ci, bi in zip(c[::2], b[::2])))
    r = a['fits_off'][:, :, 1].mean(axis=1)
    print(f'  normalization off, R2 (full) vs test contrast: min {r.min():.3f}, max {r.max():.3f}')
    t = (np.arange(a['fits_t_on'].shape[0]) + 1) * 10 * PRES_MS / 1000.0
    for tag in ('on', 'off'):
        f = a[f'fits_t_{tag}'].mean(axis=1)
        print(f'  normalization {tag}, vs adaptation time (s): ' + '; '.join(
            f'{ti:.1f}: beta {fi[0]:.3f}, R2 {fi[1]:.2f}, R2(|x|<=1) {fi[3]:.2f}' for ti, fi in zip(t[[0, 2, 4, 6, 9, 13]], f[[0, 2, 4, 6, 9, 13]])))


if __name__ == '__main__':
    d = None if '--recompute' in sys.argv else C.load_data(NAME)
    if d is None:
        C.save_data(NAME, **simulate())
        d = C.load_data(NAME)
    a = analyse(d)
    report(d, a)
    print('saved', plot(d, a))
