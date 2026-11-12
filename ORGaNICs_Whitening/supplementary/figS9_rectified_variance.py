'''
figS9_rectified_variance.py  ->  figures/FigS9_rectified_variance.png, data/FigS9_rectified_variance.npz

Does a rectified variance signal give a suppression profile closer to experiment?
(Supplementary Note 4.)

Standard gain equation:   tau_g dg/dt = (c - mu)^2 - theta_t                 (Eq. 8 of the main text)
Rectified gain equation:  tau_g dg/dt = max(c - mu, 0)^2 - theta_t / 2
(theta_t / 2 because the rectified square of a zero-mean Gaussian is half its variance.)

Both are run with the new equations, 125 presentations x 32 ms = 4.0 s, contrast 0.8, surround
delay 40 ms, 8 repeats per ensemble (seeds 100-107 uniform, 200-207 biased), gains from zero.
  Small adaptors: gains frozen, steady-state responses to large test gratings at 26 orientations
                  (analytic fixed point for all-ones weights without noise; full dynamics otherwise).
  Large adaptors: in-stream readout (last step of every presentation, second half of the run).
Configurations: baseline (all-ones weights, no noise, surround width 0.75) and, if different,
the configuration closest to the target in Fig. S8 (read from data/FigS8_large_stimuli.npz).

Run:  python figS9_rectified_variance.py [--recompute]     (about 5 minutes on 4 cores)
'''

import sys
from dataclasses import replace
from multiprocessing import Pool

import numpy as np

import common as C
import plotstyle as ps
import simulation_variants as sv
import figS8_large_stimuli as S8

NAME = 'FigS9_rectified_variance'
N_SEEDS = S8.N_SEEDS
K = 182
ANGLES = np.linspace(0, np.pi, 2 * C.N_RF, endpoint=False)


def build_model(weights, noise, rectified, best, noise_rows=None):
    N = np.ones((C.N_TOT, C.N_TOT)) if weights == 'ones' else C.normalization_weights(best['sigma_s'], best['sigma_th'])
    opts = replace(sv.NEW, rectified_variance=rectified)
    theta = sv.theta_t_from_target(C.load_frame(), C.target_covariance()) * (0.5 if rectified else 1.0)
    amp = best['amplitude'] if noise_rows is None else np.full(noise_rows, best['amplitude'])
    nz = sv.NoiseOptions(best['kind'], amp, ref=best['ref']) if noise else sv.NoiseOptions()
    return C.make_model(opts, N_matrix=N, noise=nz, theta_t=theta)


def small_run(args):
    ci, weights, noise, rectified, best = args
    m = build_model(weights, noise, rectified, best)
    z, _ = C.uniform_and_biased(C.ADAPT_PRES, N_SEEDS)
    g = C.adapt(m, z, rng=np.random.default_rng(6))['g_end'][:, -1]
    probes = C.oriented(ANGLES, C.PROBE_CONTRAST, C.LARGE)
    zmin = np.nan
    if noise:
        rows_g = np.repeat(g, len(ANGLES), axis=0)
        mr = build_model(weights, noise, rectified, best, noise_rows=len(rows_g))
        rates = C.time_averaged_rates(mr, np.tile(probes, (len(g), 1)), rows_g, rng=np.random.default_rng(7))
        tc = rates[:, :C.N_RF].reshape(len(g), len(ANGLES), C.N_RF).transpose(0, 2, 1)
    elif weights != 'ones':
        rows_g = np.repeat(g, len(ANGLES), axis=0)
        y = m.settle(np.tile(probes, (len(g), 1)), rows_g[:, :K], rows_g[:, K:], settle_ms=600.0)['y_end'][:, 0]
        tc = (np.maximum(y[:, :C.N_RF], 0) ** 2).reshape(len(g), len(ANGLES), C.N_RF).transpose(0, 2, 1)
    else:
        fps = [m.analytic_fixed_point(probes, gi[:K], gi[K:]) for gi in g]
        tc = np.stack([(np.maximum(fp['y'][:, :C.N_RF], 0) ** 2).T for fp in fps])
        zmin = min(fp['z_prime'][:, :C.N_RF].min() for fp in fps)
    return ('small', ci, rectified), tc.reshape(2, N_SEEDS, C.N_RF, len(ANGLES)).astype(np.float32), \
        g[:, :K].mean(axis=1).reshape(2, N_SEEDS), zmin


def large_run(args):
    ci, weights, noise, rectified, best, width_index = args
    _, tc, g = S8.run_group((weights, noise, best, rectified))
    return ('large', ci, rectified), tc[width_index], g[width_index], np.nan


def simulate():
    d8 = C.load_data(S8.NAME)
    a8 = S8.analyse(d8)
    best = {k[5:]: (str(d8[k]) if k == 'best_kind' else float(d8[k])) for k in d8.files if k.startswith('best_')}
    b = a8['best_row']
    configs = [('ones', False, 1)]
    if b is not a8['baseline']:
        configs.append((b['weights'], b['noise'], S8.SURROUND_WIDTHS.index(b['w_surround'])))
    # The surround's tuning width only enters through large adaptors: for small adaptors a
    # configuration with the baseline's weights and noise is the baseline, and is not rerun.
    small_cfg = [ci for ci, c in enumerate(configs) if ci == 0 or c[:2] != configs[0][:2]]
    jobs_small = [(ci, configs[ci][0], configs[ci][1], r, best) for ci in small_cfg for r in (False, True)]
    jobs_large = [(ci, w, n, r, best, wi) for ci, (w, n, wi) in enumerate(configs) for r in (False, True)]
    with Pool(4) as pool:
        res = pool.map(small_run, jobs_small) + pool.map(large_run, jobs_large)
    out = dict(angles=ANGLES, n_seeds=N_SEEDS, config_weights=np.array([c[0] for c in configs]),
               config_noise=np.array([c[1] for c in configs]),
               config_width=np.array([S8.SURROUND_WIDTHS[c[2]] for c in configs]),
               small_same_as_baseline=np.array([ci not in small_cfg for ci in range(len(configs))]),
               **{f'best_{k}': v for k, v in best.items()})
    for (size, ci, r), tc, g, zmin in res:
        out[f'tc_{size}_{ci}_{int(r)}'], out[f'gmean_{size}_{ci}_{int(r)}'], out[f'zmin_{size}_{ci}_{int(r)}'] = tc, g, zmin
    for ci in range(len(configs)):
        if ci not in small_cfg:
            for r in (0, 1):
                for key in ('tc', 'gmean', 'zmin'):
                    out[f'{key}_small_{ci}_{r}'] = out[f'{key}_small_0_{r}']
    return out


def analyse(d):
    a = dict(rel=C.rel_to_adaptor(C.pref_deg()))
    a['order'] = np.argsort(a['rel'])
    n_cfg = len(d['config_weights'])
    for size in ('small', 'large'):
        for ci in range(n_cfg):
            for r in (0, 1):
                tc = d[f'tc_{size}_{ci}_{r}'].astype(float)
                p, s = S8.profile_from(tc)
                uni_peak = np.nanmean(tc[0], axis=0).max(axis=1, keepdims=True)
                a[f'{size}_{ci}_{r}'] = dict(profile=p, sem=s, distance=C.profile_distance(p), features=C.profile_features(p),
                                             curves=np.nanmean(tc[1], axis=0) / uni_peak, gmean=d[f'gmean_{size}_{ci}_{r}'].mean(axis=1))
    a['n_cfg'] = n_cfg
    return a


def plot(d, a):
    import matplotlib.pyplot as plt
    ps.apply()
    rel, order = a['rel'], a['order']
    col = {0: ps.COLORS['new'], 1: 'k'}
    lab = {0: 'standard', 1: 'rectified'}
    fig, axes = plt.subplots(2, 2, figsize=(ps.FULL_WIDTH, 120 * ps.MM), gridspec_kw=dict(width_ratios=[1.45, 1]))
    stim_small = np.degrees(d['angles'])
    stim_large = C.pref_deg()
    for row, (size, stim) in enumerate((('small', stim_small), ('large', stim_large))):
        ax_c, ax_p = axes[row]
        x = C.rel_to_adaptor(stim)
        xo = np.argsort(x)
        for r in (0, 1):
            res = a[f'{size}_0_{r}']
            for j, i in enumerate(order[::2]):
                ax_c.plot(x[xo], res['curves'][i][xo], color=col[r], lw=3.0 if r == 0 else 1.2, alpha=0.5 if r == 0 else 1.0,
                          label=lab[r] if j == 0 else None)
            m_, s_ = res['profile'][order], res['sem'][order]
            ax_p.fill_between(rel[order], m_ - s_, m_ + s_, color=col[r], alpha=0.2, lw=0)
            ax_p.plot(rel[order], m_, color=col[r], lw=ps.LW_MAIN, marker='o', ms=3.5, label=lab[r])
            if a['n_cfg'] > 1 and not (size == 'small' and bool(d['small_same_as_baseline'][1])):
                ax_p.plot(rel[order], a[f'{size}_1_{r}']['profile'][order], color=col[r], lw=1.25, ls=(0, (4, 2)))
        ax_p.plot(rel[order], C.target_profile(rel[order]), color=ps.COLORS['reference'], lw=1.5, ls=(0, (1, 1.5)), label='target')
        for ax in (ax_c, ax_p):
            ps.ref_line(ax, 1.0)
            ax.set_xlim(-90, 90)
        ax_c.set_xticks([-90, -45, 0, 45, 90])
        ax_p.set_xticks([-90, 0, 90])
        ax_c.set_ylim(0, 1.6)
        ax_c.set_yticks([0, 0.5, 1.0, 1.5])
        ax_c.set_ylabel(f'Normalized response\n({size} adaptors)')
        ax_p.set_ylim(0.5, 1.5)
        ax_p.set_yticks([0.6, 0.8, 1.0, 1.2, 1.4])
        ax_p.set_ylabel('Peak response,\nbiased / uniform')
    axes[0, 0].legend(loc='upper left', ncol=2, columnspacing=1.5, bbox_to_anchor=(0.0, 1.04))
    axes[0, 1].legend(loc='upper center', fontsize=8, ncol=3, columnspacing=0.8, handlelength=1.4, bbox_to_anchor=(0.5, 1.05))
    axes[1, 0].set_xlabel('Stimulus orientation relative to adaptor (deg)')
    axes[1, 1].set_xlabel('Preferred orientation\nrelative to adaptor (deg)')
    fig.subplots_adjust(left=0.105, right=0.985, top=0.955, bottom=0.12, wspace=0.3, hspace=0.2)
    ps.label_panels(fig, axes.ravel())
    return ps.save(fig, NAME)


def report(d, a):
    rel, order = a['rel'], a['order']
    orth = np.abs(rel) >= 69
    for ci in range(a['n_cfg']):
        name = 'baseline' if ci == 0 else (f'closest to target in Fig. S8 ({str(d["config_weights"][ci])} weights, noise {bool(d["config_noise"][ci])}, '
                                           f'surround width {float(d["config_width"][ci]):g})')
        print(f'  configuration: {name}')
        for size in ('small', 'large'):
            for r in (0, 1):
                res = a[f'{size}_{ci}_{r}']
                far, at, depth = res['features']
                print(f'    {size} adaptors, {"rectified" if r else "standard "}: distance {res["distance"]:.3f} | ratio far {far:.3f}, at adaptor {at:.3f}, '
                      f'orthogonal (|rel| >= 69 deg) {res["profile"][orth].mean():.3f} | mean gain uniform/biased {res["gmean"][0]:.3f}/{res["gmean"][1]:.3f} | '
                      f'min z\' {float(d[f"zmin_{size}_{ci}_{r}"]):+.1e}')
                print('       profile', np.round(res['profile'][order], 3), 's.e.m. max', res['sem'].max().round(3))
            diff = a[f'{size}_{ci}_1']['profile'] - a[f'{size}_{ci}_0']['profile']
            print(f'    {size}: rectified - standard: at adaptor {diff[C.ADAPTOR_IDX]:+.3f}, orthogonal {diff[orth].mean():+.3f}, '
                  f'flanks (28-55 deg) {diff[(np.abs(rel) > 20) & (np.abs(rel) < 60)].mean():+.3f}')


if __name__ == '__main__':
    d = None if '--recompute' in sys.argv else C.load_data(NAME)
    if d is None:
        C.save_data(NAME, **simulate())
        d = C.load_data(NAME)
    a = analyse(d)
    report(d, a)
    print('saved', plot(d, a))
