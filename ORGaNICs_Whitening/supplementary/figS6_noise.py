'''
figS6_noise.py  ->  figures/FigS6_noise.png, data/FigS6_noise.npz

Does noise produce untuned suppression? (Supplementary Note 3, part 1.)

Protocol (small adaptors, frozen-gain readout): new equations; uniform and biased ensembles of
small gratings (cRF only, contrast 0.8, 32 ms, 125 presentations = 4.0 s, surround delay 40 ms),
4 repeats per ensemble and noise level (seeds 100-103 uniform, 200-203 biased), gains start at
zero in both. Noise is injected into the dynamics during adaptation AND during the readout.
Readout: gains frozen, large test gratings (contrast 0.8) at 26 orientations, full dynamics with
the same noise, firing rates averaged over 150 ms after 150 ms of settling and over repeats.
Suppression profile = peak of each neuron's tuning curve after biased adaptation divided by its
peak (row 1) after uniform adaptation or (row 2) in the non-adapted circuit (zero gains) at the
same noise level. theta_t is the same in every run.

Noise (simulation_variants.NoiseOptions; tau_n = 1 ms, bin = 1 ms):
  white, input     s.d. (after 1 ms low-pass) = f x peak input drive of a small grating,
                   f = 0, 0.05, 0.1, 0.2, 0.4, 0.8; Euler-Maruyama, fresh every step
  white, response  s.d. = f x peak unadapted response sqrt(y+), f = 0, 0.03, 0.06, 0.12, 0.18, 0.24
  Poisson, input   rate scale R = inf, 3e4, 1e4, 3e3, 1e3, 3e2 spikes/s at the peak input drive
  Poisson, response R = inf, 1e5, 3e4, 1e4, 3e3, 1e3 spikes/s at the peak unadapted firing rate
Time-step check: both white-noise columns are repeated at dt = 0.05 ms for the fourth level.
Random generator seeds: 1 (adaptation), 2 (readout), 3 and 4 (time-step check).

Run:  python figS6_noise.py [--recompute]     (about 12 minutes on 4 cores)
'''

import sys
from multiprocessing import Pool

import numpy as np

import common as C
import plotstyle as ps
import simulation_variants as sv

NAME = 'FigS6_noise'
N_SEEDS = 4
K = 182
KINDS = ['white_input', 'white_response', 'poisson_input', 'poisson_response']
ANGLES = np.linspace(0, np.pi, 2 * C.N_RF, endpoint=False)
PREF = np.linspace(0, np.pi, C.N_RF, endpoint=False)
Z_PEAK = float(C.oriented(PREF[6], C.ENSEMBLE_CONTRAST, C.SMALL)[0, 6])
R_PEAK = float(np.sqrt(C.tuning_curves(C.make_model(sv.NEW), np.zeros(K), np.zeros(K), PREF[6:7], location=C.SMALL)[6, 0]))
FRACTIONS = {'white_input': [0, 0.05, 0.1, 0.2, 0.4, 0.8], 'white_response': [0, 0.03, 0.06, 0.12, 0.18, 0.24]}
RATES = {'poisson_input': [np.inf, 3e4, 1e4, 3e3, 1e3, 3e2], 'poisson_response': [np.inf, 1e5, 3e4, 1e4, 3e3, 1e3]}
DT_CHECK_LEVEL = 3


def amplitudes(kind):
    if kind == 'white_input':
        return np.array(FRACTIONS[kind]) * Z_PEAK, 1.0
    if kind == 'white_response':
        return np.array(FRACTIONS[kind]) * R_PEAK, 1.0
    return np.array(RATES[kind]), (Z_PEAK if kind == 'poisson_input' else R_PEAK ** 2)


def run_kind(args):
    '''Adaptation and readout for one noise kind. Returns tuning curves
    (n_levels, 2 N_SEEDS + 1, N_RF, n_angles): uniform repeats, biased repeats, unadapted.'''
    kind, dt, levels, seed = args
    amp_all, ref = amplitudes(kind)
    amp = amp_all[levels]
    nl = len(amp)
    z, _ = C.uniform_and_biased(C.ADAPT_PRES, N_SEEDS)                      # (2 N_SEEDS, n_pres, N_TOT)
    m = C.make_model(sv.NEW, dt=dt, noise=sv.NoiseOptions(kind, np.repeat(amp, 2 * N_SEEDS), ref=ref))
    res = C.adapt(m, np.tile(z, (nl, 1, 1)), rng=np.random.default_rng(seed))
    g = res['g_end'][:, -1].reshape(nl, 2 * N_SEEDS, 2 * K)
    g = np.concatenate([g, np.zeros((nl, 1, 2 * K))], axis=1)               # + the unadapted circuit
    n_state, n_ang = g.shape[1], len(ANGLES)
    probes = C.oriented(ANGLES, C.PROBE_CONTRAST, C.LARGE)
    rows_g = np.repeat(g.reshape(nl * n_state, 2 * K), n_ang, axis=0)
    rows_z = np.tile(probes, (nl * n_state, 1))
    rows_amp = np.repeat(amp, n_state * n_ang)
    mr = C.make_model(sv.NEW, dt=dt, noise=sv.NoiseOptions(kind, rows_amp, ref=ref))
    rates = C.time_averaged_rates(mr, rows_z, rows_g, rng=np.random.default_rng(seed + 1))
    tc = rates[:, :C.N_RF].reshape(nl, n_state, n_ang, C.N_RF).transpose(0, 1, 3, 2)
    return kind, dt, tc.astype(np.float32), g[:, :-1, :K].mean(axis=2).astype(np.float32)


def simulate():
    jobs = [(k, C.DT, list(range(6)), 1) for k in KINDS]
    jobs += [(k, 0.05, [DT_CHECK_LEVEL], 3) for k in ('white_input', 'white_response')]
    with Pool(4) as pool:
        results = pool.map(run_kind, jobs)
    out = dict(kinds=np.array(KINDS), angles=ANGLES, z_peak=Z_PEAK, r_peak=R_PEAK, n_seeds=N_SEEDS, dt_check_level=DT_CHECK_LEVEL)
    for kind, dt, tc, gmean in results:
        tag = kind if dt == C.DT else f'{kind}_dt05'
        out[f'tc_{tag}'], out[f'gmean_{tag}'] = tc, gmean
    for k in FRACTIONS:
        out[f'levels_{k}'] = np.array(FRACTIONS[k])
    for k in RATES:
        out[f'levels_{k}'] = np.array(RATES[k])
    return out


def profiles(tc, n):
    '''tc (n_levels, 2n + 1, N_RF, n_angles) -> profiles relative to uniform-adapted and to
    unadapted, each (n_levels, N_RF), plus the s.e.m. over biased repeats of the first.'''
    peak = tc.astype(float).max(axis=3)                    # per state: peak over orientations
    uni, bia, una = peak[:, :n].mean(axis=1), peak[:, n:2 * n], peak[:, -1]
    rel_uni = bia / uni[:, None]
    return rel_uni.mean(axis=1), bia.mean(axis=1) / una, rel_uni.std(axis=1, ddof=1) / np.sqrt(n), uni / una


def analyse(d):
    n = int(d['n_seeds'])
    a = dict(rel=C.rel_to_adaptor(C.pref_deg()))
    a['order'] = np.argsort(a['rel'])
    for kind in d['kinds']:
        pu, pn, sem, un = profiles(d[f'tc_{kind}'], n)
        a[f'uni_{kind}'], a[f'una_{kind}'], a[f'sem_{kind}'], a[f'uniform_vs_unadapted_{kind}'] = pu, pn, sem, un
        a[f'dist_{kind}'] = np.array([C.profile_distance(p) for p in pu])
        a[f'feat_{kind}'] = np.array([C.profile_features(p) for p in pu])
        una_tc = d[f'tc_{kind}'].astype(float)[:, -1]                       # unadapted tuning curves per level
        a[f'modulation_{kind}'] = ((una_tc.max(axis=2) - una_tc.min(axis=2)) / una_tc.max(axis=2)).mean(axis=1)
        if f'tc_{kind}_dt05' in d:
            p05, n05, s05, _ = profiles(d[f'tc_{kind}_dt05'], n)
            lv = int(d['dt_check_level'])
            a[f'dtcheck_{kind}'] = np.array([np.abs(p05[0] - pu[lv]).max(), np.sqrt(s05[0] ** 2 + sem[lv] ** 2).max(),
                                             np.abs(n05[0] - pn[lv]).max()])
    return a


TITLES = {'white_input': 'White noise,\ninput', 'white_response': 'White noise,\nresponse',
          'poisson_input': 'Poisson noise,\ninput', 'poisson_response': 'Poisson noise,\nresponse'}


def plot(d, a):
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors as mcolors
    ps.apply()
    rel, order = a['rel'], a['order']
    cmap = plt.get_cmap(ps.SEQUENTIAL)
    level_colors = ['k'] + [cmap(v) for v in np.linspace(0.12, 0.92, 5)]
    fig, axes = plt.subplots(2, 4, figsize=(ps.FULL_WIDTH, 105 * ps.MM), sharex=True, sharey='row')
    for j, kind in enumerate(d['kinds']):
        for row, key in enumerate(('uni', 'una')):
            ax = axes[row, j]
            ps.ref_line(ax, 1.0)
            for lv in range(5, -1, -1):
                ax.plot(rel[order], a[f'{key}_{kind}'][lv][order], color=level_colors[lv],
                        lw=ps.LW_MAIN if lv else 2.25, zorder=10 if lv == 0 else 5 - lv)
            if row == 0:
                ax.plot(rel[order], C.target_profile(rel[order]), color=ps.COLORS['reference'], lw=1.25, ls=(0, (1, 1.5)), zorder=1)
                ax.set_title(TITLES[kind], fontsize=10, pad=6)
            ax.set_xlim(-90, 90)
            ax.set_xticks([-60, 0, 60])
    axes[0, 0].set_ylabel('Peak response,\nbiased / uniform')
    axes[1, 0].set_ylabel('Peak response,\nbiased / unadapted')
    axes[0, 0].set_ylim(0.5, 1.45)
    axes[0, 0].set_yticks([0.6, 0.8, 1.0, 1.2, 1.4])
    axes[1, 0].set_ylim(0, 1.05)
    axes[1, 0].set_yticks([0, 0.5, 1.0])
    axes[0, 0].text(-84, 0.535, 'target', fontsize=8, color='#4D4D4D', va='bottom')
    fig.supxlabel('Preferred orientation relative to adaptor (deg)', fontsize=10, y=0.012, x=0.5)
    fig.subplots_adjust(left=0.105, right=0.905, top=0.875, bottom=0.135, wspace=0.14, hspace=0.16)
    cax = fig.add_axes([0.932, 0.27, 0.012, 0.4])
    cb = fig.colorbar(cm.ScalarMappable(norm=mcolors.BoundaryNorm(np.arange(-0.5, 6), 6),
                                        cmap=mcolors.ListedColormap(level_colors)), cax=cax, ticks=[0, 5])
    cb.ax.set_yticklabels(['none', 'max'], fontsize=9)
    cb.ax.tick_params(length=0)
    cb.ax.set_title('Noise', fontsize=9, pad=5)
    ps.label_panels(fig, [axes[0, 0], axes[1, 0]])
    return ps.save(fig, NAME)


def report(d, a):
    print(f'  peak input drive of a small grating {float(d["z_peak"]):.4f}; peak unadapted response sqrt(y+) {float(d["r_peak"]):.4f}; '
          f'target profile: far {C.TARGET_FAR}, at adaptor {C.TARGET_FAR - C.TARGET_DEPTH}, s.d. {C.TARGET_WIDTH} deg')
    for kind in d['kinds']:
        print(f'  {kind}: levels {list(d[f"levels_{kind}"])}')
        print('     level: mean gain uniform/biased | uniform-adapted/unadapted peak | ratio far, at adaptor (vs uniform) | '
              'biased/unadapted far, at adaptor | distance to target | tuning modulation of unadapted curves')
        g = d[f'gmean_{kind}'].astype(float)
        n = int(d['n_seeds'])
        for lv in range(6):
            far, at, _ = a[f'feat_{kind}'][lv]
            far2, at2, _ = C.profile_features(a[f'una_{kind}'][lv])
            print(f'     {lv}: {g[lv, :n].mean():.3f}/{g[lv, n:].mean():.3f} | {a[f"uniform_vs_unadapted_{kind}"][lv].mean():.3f} | '
                  f'{far:.3f}, {at:.3f} | {far2:.3f}, {at2:.3f} | {a[f"dist_{kind}"][lv]:.3f} | {a[f"modulation_{kind}"][lv]:.2f}')
        if f'dtcheck_{kind}' in a:
            c = a[f'dtcheck_{kind}']
            lv = int(d['dt_check_level'])
            print(f'     time-step check (level {lv}, dt 0.05 vs 0.1 ms, independent noise draws): max profile difference {c[0]:.3f} '
                  f'(largest combined s.e.m. {c[1]:.3f}); vs unadapted {c[2]:.3f}; mean gain {d[f"gmean_{kind}_dt05"][0].mean():.4f} '
                  f'vs {d[f"gmean_{kind}"][lv].mean():.4f}')


if __name__ == '__main__':
    d = None if '--recompute' in sys.argv else C.load_data(NAME)
    if d is None:
        C.save_data(NAME, **simulate())
        d = C.load_data(NAME)
    a = analyse(d)
    report(d, a)
    print('saved', plot(d, a))
