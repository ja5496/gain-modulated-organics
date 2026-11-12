'''
figS8_large_stimuli.py  ->  figures/FigS8_large_stimuli.png, data/FigS8_large_stimuli.npz

Large adaptors: can noise, normalization weights and the orientation tuning of the surround
together account for the reported suppression profile? (Supplementary Note 3, part 3.)
A sensitivity analysis over 12 configurations, not a fit.

Protocol (large adaptors, in-stream readout): new equations; uniform and biased ensembles of
large gratings (cRF + surround, contrast 0.8, 32 ms, 125 presentations = 4.0 s, surround delay
40 ms > presentation), 8 repeats each (seeds 100-107, 200-207), gains from zero. Responses are
read at the last time step of every presentation in the second half of the run (2-4 s) and
averaged per orientation and over repeats. Profile = each neuron's peak after biased adaptation
divided by its peak after uniform adaptation.

Configurations = {all-ones weights, best weights of Fig. S7} x {no noise, best noise of Fig. S6}
x {surround input tuning width 0.4, 0.75 (default), 1.5 rad}. The best weights and noise are read
from data/FigS7_normalization_widths.npz and data/FigS6_noise.npz (smallest distance to target
among the non-trivial settings). Noise generator seed 5.

Run:  python figS8_large_stimuli.py [--recompute]     (about 4 minutes on 4 cores)
'''

import sys
from multiprocessing import Pool

import numpy as np

import common as C
import plotstyle as ps
import simulation_variants as sv
import figS6_noise as S6
import figS7_normalization_widths as S7

NAME = 'FigS8_large_stimuli'
N_SEEDS = 8
K = 182
SURROUND_WIDTHS = [0.4, 0.75, 1.5]


def best_settings():
    d6, d7 = C.load_data(S6.NAME), C.load_data(S7.NAME)
    a6, a7 = S6.analyse(d6), S7.analyse(d7)
    kind, level = min(((k, lv) for k in d6['kinds'] for lv in range(1, 6)), key=lambda kl: a6[f'dist_{kl[0]}'][kl[1]])
    amp_all, ref = S6.amplitudes(str(kind))
    i, j = a7['best']
    return dict(kind=str(kind), level=int(level), amplitude=float(amp_all[level]), ref=float(ref),
                sigma_s=float(d7['sigma_s'][i]), sigma_th=float(d7['sigma_th'][j]))


def large_presentations(biased, seed, w_surround):
    z, centres = C.ensemble_presentations(C.ADAPT_PRES, biased, seed, C.LARGE)
    if w_surround != C.TUNING_WIDTH:
        gen = C.stimulus_generator()
        delta = (gen.theta_RF[None, :] - centres[:, None] + np.pi / 2) % np.pi - np.pi / 2
        profile = C.ENSEMBLE_CONTRAST * np.exp(-delta ** 2 / (2 * w_surround ** 2)) / gen.max_drive
        z[:, C.N_RF:] = np.tile(profile, (1, C.N_SETS - 1))
    return z, centres


def configurations(best):
    return [dict(weights=w, noise=n, w_surround=ws) for w in ('ones', 'best') for n in (False, True) for ws in SURROUND_WIDTHS]


def run_group(args):
    '''One process: the three surround widths for one (weights, noise) pair.'''
    weights, noise, best, variance_rectified = args
    N = np.ones((C.N_TOT, C.N_TOT)) if weights == 'ones' else C.normalization_weights(best['sigma_s'], best['sigma_th'])
    zs, cs = [], []
    for ws in SURROUND_WIDTHS:
        for biased, s0 in ((False, 100), (True, 200)):
            for i in range(N_SEEDS):
                z, c = large_presentations(biased, s0 + i, ws)
                zs.append(z)
                cs.append(c)
    opts = sv.NEW if not variance_rectified else sv.EquationOptions(**{**sv.NEW.__dict__, 'rectified_variance': True})
    theta = sv.theta_t_from_target(C.load_frame(), C.target_covariance()) * (0.5 if variance_rectified else 1.0)
    nz = sv.NoiseOptions(best['kind'], best['amplitude'], ref=best['ref']) if noise else sv.NoiseOptions()
    m = C.make_model(opts, N_matrix=N, noise=nz, theta_t=theta)
    res = C.adapt(m, np.stack(zs), rng=np.random.default_rng(5))
    tc = np.stack([C.instream_tuning(y, c)[:C.N_RF] for y, c in zip(res['y_end'], cs)])      # (rows, N_RF, 13)
    shape = (len(SURROUND_WIDTHS), 2, N_SEEDS)
    return (weights, noise), tc.reshape(*shape, C.N_RF, C.N_RF).astype(np.float32), res['g_end'][:, -1, :K].mean(axis=1).reshape(shape)


def simulate(variance_rectified=False):
    best = best_settings()
    groups = [(w, n, best, variance_rectified) for w in ('ones', 'best') for n in (False, True)]
    with Pool(4) as pool:
        res = dict((k, (tc, g)) for k, tc, g in pool.map(run_group, groups))
    out = dict(surround_widths=np.array(SURROUND_WIDTHS), n_seeds=N_SEEDS, **{f'best_{k}': v for k, v in best.items()})
    for (w, n), (tc, g) in res.items():
        out[f'tc_{w}_{int(n)}'], out[f'gmean_{w}_{int(n)}'] = tc, g
    return out


def profile_from(tc):
    '''tc (2, N_SEEDS, N_RF, 13): repeat-averaged in-stream tuning curves -> profile, and its
    s.e.m. from a jackknife over repeats.'''
    def prof(sel):
        uni, bia = np.nanmean(tc[0][sel], axis=0), np.nanmean(tc[1][sel], axis=0)
        return bia.max(axis=1) / uni.max(axis=1)
    n = tc.shape[1]
    full = prof(np.arange(n))
    jack = np.array([prof(np.delete(np.arange(n), i)) for i in range(n)])
    return full, np.sqrt((n - 1) / n * np.sum((jack - jack.mean(axis=0)) ** 2, axis=0))


def analyse(d):
    a = dict(rel=C.rel_to_adaptor(C.pref_deg()), rows=[])
    a['order'] = np.argsort(a['rel'])
    for w in ('ones', 'best'):
        for n in (0, 1):
            for k, ws in enumerate(d['surround_widths']):
                p, s = profile_from(d[f'tc_{w}_{n}'][k].astype(float))
                a['rows'].append(dict(weights=w, noise=bool(n), w_surround=float(ws), profile=p, sem=s,
                                      distance=C.profile_distance(p), features=C.profile_features(p),
                                      gmean=d[f'gmean_{w}_{n}'][k].mean(axis=1)))
    a['baseline'] = next(r for r in a['rows'] if r['weights'] == 'ones' and not r['noise'] and r['w_surround'] == C.TUNING_WIDTH)
    a['best_row'] = min(a['rows'], key=lambda r: r['distance'])
    return a


def row_label(r):
    return f"{'all-ones' if r['weights'] == 'ones' else 'tuned'} pool, {'noise' if r['noise'] else 'no noise'}, $w_s$ = {r['w_surround']:g}"


def plot(d, a):
    import matplotlib.pyplot as plt
    ps.apply()
    rel, order = a['rel'], a['order']
    fig = plt.figure(figsize=(ps.FULL_WIDTH, 92 * ps.MM))
    ax_p = fig.add_axes([0.085, 0.155, 0.4, 0.76])
    ax_d = fig.add_axes([0.79, 0.155, 0.195, 0.76])
    ps.ref_line(ax_p, 1.0)
    for r in a['rows']:
        ax_p.plot(rel[order], r['profile'][order], color=ps.COLORS['reference'], lw=ps.LW_THIN, alpha=0.6, zorder=1)
    ax_p.plot(rel[order], C.target_profile(rel[order]), color=ps.COLORS['reference'], lw=1.5, ls=(0, (1, 1.5)), label='target', zorder=2)
    for r, color, label in ((a['baseline'], 'k', 'baseline'), (a['best_row'], ps.OKABE_ITO['vermillion'], 'closest to target')):
        m_, s_ = r['profile'][order], r['sem'][order]
        ax_p.fill_between(rel[order], m_ - s_, m_ + s_, color=color, alpha=0.2, lw=0, zorder=3)
        ax_p.plot(rel[order], m_, color=color, lw=ps.LW_MAIN, marker='o', ms=3.5, label=label, zorder=4)
    ax_p.set_xlim(-90, 90)
    ax_p.set_xticks([-90, -45, 0, 45, 90])
    ax_p.set_ylim(0.5, 1.6)
    ax_p.set_yticks([0.6, 0.8, 1.0, 1.2, 1.4])
    ax_p.set_xlabel('Preferred orientation relative to adaptor (deg)')
    ax_p.set_ylabel('Peak response, biased / uniform')
    ax_p.legend(loc='upper center', fontsize=8, ncol=3, columnspacing=1.0, handlelength=1.5, bbox_to_anchor=(0.5, 1.04))

    rows = sorted(a['rows'], key=lambda r: -r['distance'])
    for i, r in enumerate(rows):
        color = 'k' if r is a['baseline'] else ps.OKABE_ITO['vermillion'] if r is a['best_row'] else ps.COLORS['reference']
        ax_d.plot([0, r['distance']], [i, i], color=color, lw=1.0)
        ax_d.plot(r['distance'], i, 'o', color=color, ms=4.5)
    ax_d.set_yticks(range(len(rows)), labels=[row_label(r) for r in rows], fontsize=8)
    ax_d.set_xlim(0, None)
    ax_d.set_ylim(-0.6, len(rows) - 0.4)
    ax_d.set_xlabel('Distance to target')
    ax_d.tick_params(axis='y', length=0)
    ps.label_panels(fig, [ax_p, ax_d])
    return ps.save(fig, NAME)


def report(d, a):
    print(f'  best noise from Fig. S6: {str(d["best_kind"])}, level {int(d["best_level"])} (amplitude {float(d["best_amplitude"]):.4g}); '
          f'best weights from Fig. S7: sigma_s = {float(d["best_sigma_s"]):g}, sigma_theta = {float(d["best_sigma_th"]):g} deg')
    print(f'  {len(a["rows"])} configurations. weights, noise, surround width: distance | ratio far, at adaptor, depth | mean gain uniform/biased')
    for r in sorted(a['rows'], key=lambda r: r['distance']):
        far, at, depth = r['features']
        tag = ' (baseline)' if r is a['baseline'] else ''
        print(f'  {r["weights"]:5s} {str(r["noise"]):5s} {r["w_surround"]:4g}: {r["distance"]:.3f} | {far:.3f}, {at:.3f}, {depth:+.3f} | '
              f'{r["gmean"][0]:.3f}/{r["gmean"][1]:.3f}{tag}')
    b = a['baseline']
    print('  baseline profile:', np.round(b['profile'][a['order']], 3), ' s.e.m.', np.round(b['sem'][a['order']], 3))
    print('  closest profile :', np.round(a['best_row']['profile'][a['order']], 3))


if __name__ == '__main__':
    d = None if '--recompute' in sys.argv else C.load_data(NAME)
    if d is None:
        C.save_data(NAME, **simulate())
        d = C.load_data(NAME)
    a = analyse(d)
    report(d, a)
    print('saved', plot(d, a))
