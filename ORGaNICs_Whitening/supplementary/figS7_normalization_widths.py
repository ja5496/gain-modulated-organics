'''
figS7_normalization_widths.py  ->  figures/FigS7_normalization_widths.png, data/FigS7_normalization_widths.npz

How do the spatial and orientation widths of the normalization pool shape the suppression
profile? (Supplementary Note 3, part 2.)

Normalization weights: common.normalization_weights (Gaussian in RF distance and in preferred
orientation, rescaled to the all-ones pooled drive for an adapting grating).
Grid: sigma_s = 0.5, 1, 2, 5 cRF diameters and inf; sigma_theta = 15, 30, 60 deg and inf.
(inf, inf) is the all-ones pool.

Protocol (small adaptors, frozen-gain readout), no noise: new equations; uniform and biased
ensembles of small gratings (contrast 0.8, 32 ms, 125 presentations = 4.0 s, surround delay
40 ms), 4 repeats each (seeds 100-103, 200-203), gains from zero. Gains are frozen and the FULL
dynamics are integrated for 600 ms for large test gratings (contrast 0.8) at 26 orientations.
Profile = peak after biased adaptation / peak after uniform adaptation. The same settled states
are compared with the all-ones analytic fixed point (Eqs. 10-11 of the main text).

Run:  python figS7_normalization_widths.py [--recompute]     (about 8 minutes on 4 cores)
'''

import sys
from multiprocessing import Pool

import numpy as np

import common as C
import plotstyle as ps
import simulation_variants as sv

NAME = 'FigS7_normalization_widths'
SIGMA_S = [0.5, 1.0, 2.0, 5.0, np.inf]
SIGMA_TH = [15.0, 30.0, 60.0, np.inf]
N_SEEDS = 4
K = 182
ANGLES = np.linspace(0, np.pi, 2 * C.N_RF, endpoint=False)
CONTRASTS = np.logspace(-2, 0, 13)


def run_config(args):
    sigma_s, sigma_th = args
    N = C.normalization_weights(sigma_s, sigma_th)
    m = C.make_model(sv.NEW, N_matrix=N)
    z, _ = C.uniform_and_biased(C.ADAPT_PRES, N_SEEDS)
    g = C.adapt(m, z)['g_end'][:, -1]                                        # (2 N_SEEDS, 2K)
    probes = C.oriented(ANGLES, C.PROBE_CONTRAST, C.LARGE)
    rows_g = np.repeat(g, len(ANGLES), axis=0)
    y = m.settle(np.tile(probes, (len(g), 1)), rows_g[:, :K], rows_g[:, K:], settle_ms=600.0)['y_end'][:, 0]
    y = y.reshape(len(g), len(ANGLES), -1)
    tc = (np.maximum(y[:, :, :C.N_RF], 0) ** 2).transpose(0, 2, 1)             # (states, N_RF, angles)
    ref = C.make_model(sv.NEW)                                                # all-ones analytic fixed point
    y_an = np.stack([ref.analytic_fixed_point(probes, gi[:K], gi[K:])['y'] for gi in g])
    dev = np.abs(y[:, :, :C.N_RF] - y_an[:, :, :C.N_RF]).max() / np.abs(y_an[:, :, :C.N_RF]).max()
    # unadapted contrast-response function at the preferred orientation, relative to all-ones
    ones = np.ones_like(N)
    pref6 = np.radians(C.pref_deg()[C.ADAPTOR_IDX])
    crf = []
    for loc in (C.SMALL, C.LARGE):
        zc = np.concatenate([C.oriented(pref6, c, loc) for c in CONTRASTS])
        crf.append(C.unadapted_response(zc, N)[:, C.ADAPTOR_IDX] / C.unadapted_response(zc, ones)[:, C.ADAPTOR_IDX])
    return dict(tc=tc.astype(np.float32), g=g[:, :K].mean(axis=1), dev=dev, crf=np.array(crf),
                u_max=float(np.sqrt(np.clip(N @ (np.maximum(y, 0) ** 2).reshape(-1, y.shape[2]).T, 0, None).max())))


def simulate():
    grid = [(s, t) for s in SIGMA_S for t in SIGMA_TH]
    with Pool(4) as pool:
        res = pool.map(run_config, grid)
    shape = (len(SIGMA_S), len(SIGMA_TH))
    return dict(sigma_s=np.array(SIGMA_S), sigma_th=np.array(SIGMA_TH), n_seeds=N_SEEDS, angles=ANGLES, contrasts=CONTRASTS,
                tc=np.array([r['tc'] for r in res]).reshape(*shape, 2 * N_SEEDS, C.N_RF, len(ANGLES)),
                gmean=np.array([r['g'] for r in res]).reshape(*shape, 2 * N_SEEDS),
                deviation=np.array([r['dev'] for r in res]).reshape(shape),
                crf_ratio=np.array([r['crf'] for r in res]).reshape(*shape, 2, len(CONTRASTS)))


def analyse(d):
    n = int(d['n_seeds'])
    peak = d['tc'].astype(float).max(axis=-1)                                 # (5, 4, 2n, N_RF)
    ratio = peak[:, :, n:] / peak[:, :, :n].mean(axis=2, keepdims=True)
    a = dict(profile=ratio.mean(axis=2), sem=ratio.std(axis=2, ddof=1) / np.sqrt(n), rel=C.rel_to_adaptor(C.pref_deg()))
    a['order'] = np.argsort(a['rel'])
    a['distance'] = np.array([[C.profile_distance(p) for p in row] for row in a['profile']])
    a['features'] = np.array([[C.profile_features(p) for p in row] for row in a['profile']])
    finite = a['distance'].copy()
    finite[-1, -1] = np.inf                                                   # exclude the all-ones pool itself
    a['best'] = np.unravel_index(np.argmin(finite), finite.shape)
    return a


def width_label(v, unit=''):
    return r'$\infty$' if np.isinf(v) else f'{v:g}{unit}'


def plot(d, a):
    import matplotlib.pyplot as plt
    ps.apply()
    rel, order = a['rel'], a['order']
    fig = plt.figure(figsize=(ps.FULL_WIDTH, 74 * ps.MM))
    ax_h = fig.add_axes([0.1, 0.2, 0.3, 0.68])
    ax_p = fig.add_axes([0.63, 0.2, 0.35, 0.68])
    dist = a['distance']
    im = ax_h.imshow(dist.T, origin='lower', cmap=ps.SEQUENTIAL + '_r', aspect='auto',
                     vmin=np.floor(dist.min() * 100) / 100, vmax=np.ceil(dist.max() * 100) / 100)
    for i in range(dist.shape[0]):
        for j in range(dist.shape[1]):
            ax_h.text(i, j, f'{dist[i, j]:.3f}', ha='center', va='center', fontsize=8,
                      color='w' if dist[i, j] > (im.norm.vmin + im.norm.vmax) / 2 else 'k')
    bi, bj = a['best']
    ax_h.add_patch(plt.Rectangle((bi - 0.5, bj - 0.5), 1, 1, fill=False, ec=ps.OKABE_ITO['vermillion'], lw=2.0, clip_on=False))
    ax_h.set_xticks(range(len(d['sigma_s'])), labels=[width_label(v) for v in d['sigma_s']])
    ax_h.set_yticks(range(len(d['sigma_th'])), labels=[width_label(v) for v in d['sigma_th']])
    ax_h.set_xlabel(r'Spatial width $\sigma_s$ (cRF diameters)')
    ax_h.set_ylabel(r'Orientation width $\sigma_\theta$ (deg)')
    for sp in ax_h.spines.values():
        sp.set_visible(True)
    ax_h.tick_params(length=0)
    cax = fig.add_axes([0.415, 0.2, 0.012, 0.68])
    cb = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=8, length=2)
    cb.set_label('Distance to target profile', fontsize=9, labelpad=3)

    ps.ref_line(ax_p, 1.0)
    ax_p.plot(rel[order], C.target_profile(rel[order]), color=ps.COLORS['reference'], lw=1.5, ls=(0, (1, 1.5)), label='target')
    for (i, j), color, label in (((-1, -1), 'k', 'all-ones pool'),
                                 ((bi, bj), ps.OKABE_ITO['vermillion'],
                                  rf'best: $\sigma_s$ = {width_label(d["sigma_s"][bi])}, $\sigma_\theta$ = {width_label(d["sigma_th"][bj], "°")}')):
        m_, s_ = a['profile'][i, j][order], a['sem'][i, j][order]
        ax_p.fill_between(rel[order], m_ - s_, m_ + s_, color=color, alpha=0.2, lw=0)
        ax_p.plot(rel[order], m_, color=color, lw=ps.LW_MAIN, marker='o', ms=3.5, label=label)
    ax_p.set_xlim(-90, 90)
    ax_p.set_xticks([-90, -45, 0, 45, 90])
    ax_p.set_ylim(0.5, 1.5)
    ax_p.set_yticks([0.6, 0.8, 1.0, 1.2, 1.4])
    ax_p.set_xlabel('Preferred orientation relative to adaptor (deg)', x=0.45)
    ax_p.set_ylabel('Peak response, biased / uniform')
    ax_p.legend(loc='upper center', fontsize=8, ncol=1, bbox_to_anchor=(0.5, 1.04), labelspacing=0.2)
    ps.label_panels(fig, [ax_h, ax_p])
    return ps.save(fig, NAME)


def report(d, a):
    print(f'  target profile: far {C.TARGET_FAR}, at adaptor {C.TARGET_FAR - C.TARGET_DEPTH}, s.d. {C.TARGET_WIDTH} deg; distance = r.m.s. difference')
    print('  sigma_s, sigma_theta: distance | ratio far, at adaptor, depth | mean cRF gain uniform/biased | '
          'max deviation of settled y from Eqs. 10-11 (rel.) | unadapted CRF vs all-ones: small min-max, large min-max')
    n = int(d['n_seeds'])
    for i, s in enumerate(d['sigma_s']):
        for j, t in enumerate(d['sigma_th']):
            far, at, depth = a['features'][i, j]
            c = d['crf_ratio'][i, j]
            mark = '  <- best' if (i, j) == tuple(a['best']) else ''
            print(f'  {s:4g}, {t:4g}: {a["distance"][i, j]:.3f} | {far:.3f}, {at:.3f}, {depth:+.3f} | '
                  f'{d["gmean"][i, j, :n].mean():.3f}/{d["gmean"][i, j, n:].mean():.3f} | {d["deviation"][i, j]:.1e} | '
                  f'{c[0].min():.2f}-{c[0].max():.2f}, {c[1].min():.2f}-{c[1].max():.2f}{mark}')


if __name__ == '__main__':
    d = None if '--recompute' in sys.argv else C.load_data(NAME)
    if d is None:
        C.save_data(NAME, **simulate())
        d = C.load_data(NAME)
    a = analyse(d)
    report(d, a)
    print('saved', plot(d, a))
