'''
figS3_tuning_old_vs_new.py  ->  figures/FigS3_tuning_old_vs_new.png, data/FigS3_tuning_old_vs_new.npz

Effect of the equation changes on a headline result: tuning curves after adaptation to the
biased ensemble (Fig. 3c of the main text), with the corrected protocol.

Protocol (small adaptors, frozen-gain readout): the circuit adapts, from zero gains, to the
uniform ensemble and to the biased ensemble (gains active in both), with small stimuli
(cRF only, contrast 0.8, 32 ms per presentation, surround delay 40 ms). 8 repeats (random
sequences) per ensemble: seeds 100-107 (uniform) and 200-207 (biased). Gains (cRF and surround, as reached) are then frozen and
steady-state responses to large test gratings (cRF + surround, contrast 0.8) are measured at
every orientation. Tuning curves are averaged over repeats and divided by each neuron's peak
after uniform adaptation. Main readout after 260 presentations (8.32 s, the duration used in
analysis/adapt_ensembles.py is 8.4 s); the run continues to 1250 presentations (40 s).

Equation variants (simulation_variants.EquationOptions):
    old           the codebase (simulation_whiten.V1Dynamics_Surround)
    new           Eqs. 1-6 of the instructions
    gating        codebase + feedback gated by u+                      (listed change 1)
    placement     codebase + gain inside the v equation                (listed change 2)
    variance      codebase + gains driven by (c - mu)^2 instead of the tau_v-filtered (v - mu)^2
    rectify_c     codebase + c = W^T max(y, 0) instead of W^T y
    new_matched   new, with theta_t and tau_g both divided by kappa, where
                  kappa = 1 - (tau_v/T)(1 - exp(-T/tau_v)) = 0.385 is the fraction of the variance
                  of c that survives the tau_v filter for presentations of T = 32 ms. The codebase
                  gain equation, (kappa Var(c) - theta_t)/tau_g, then equals the new one.

Run:  python figS3_tuning_old_vs_new.py [--recompute]     (about 15 minutes on 4 cores)
'''

import sys
from multiprocessing import Pool

import numpy as np

import common as C
import plotstyle as ps
import simulation_variants as sv

NAME = 'FigS3_tuning_old_vs_new'
N_SEEDS = 8
N_PRES = 1250
REC_EVERY = 10
READOUT_PRES = 260
CHECKPOINTS = np.array([10, 20, 30, 40, 50, 60, 80, 100, 130, 160, 200, 260, 320, 390, 460, 520, 600,
                        680, 780, 880, 980, 1040, 1120, 1200, 1250])
VARIANT_NAMES = ['old', 'new', 'gating', 'placement', 'variance', 'rectify_c', 'new_matched']
TAU_V, TAU_G = sv.ModelParams().tau_v, sv.ModelParams().tau_g
KAPPA = 1 - (TAU_V / C.PRES_MS) * (1 - np.exp(-C.PRES_MS / TAU_V))
REL_ANGLES = np.arange(-90.0, 90.0 + 1e-9, 2.0)        # test orientations relative to the adaptor (deg)
K = 182


def model_for(name):
    if name == 'new_matched':
        theta = sv.theta_t_from_target(C.load_frame(), C.target_covariance()) / KAPPA
        return C.make_model(sv.NEW, theta_t=theta, params=sv.ModelParams(tau_g=TAU_G / KAPPA))
    return C.make_model(C.VARIANTS[name])


def run_variant(name):
    z = np.stack([C.ensemble_presentations(N_PRES, False, 100 + i, C.SMALL)[0] for i in range(N_SEEDS)]
                 + [C.ensemble_presentations(N_PRES, True, 200 + i, C.SMALL)[0] for i in range(N_SEEDS)])
    res = C.adapt(model_for(name), z, record_g_every_pres=REC_EVERY)
    # Both gain sets are kept: the surround only sees the blank screen, but its responses still
    # vary through the shared normalization pool, so its gains do not stay at zero.
    return name, res['g_end'][:, CHECKPOINTS // REC_EVERY - 1].astype(np.float32)


def readout_dynamics(args):
    '''Peak-normalizable tuning curves from the full dynamics, for one variant at READOUT_PRES.'''
    name, g = args                                       # g: (2 N_SEEDS, 2K) = [cRF, surround]
    m = model_for(name)
    angles = np.radians(C.pref_deg()[C.ADAPTOR_IDX] + REL_ANGLES)
    return name, np.stack([C.tuning_curves(m, gi[:K], gi[K:], angles, method='dynamics') for gi in g])


def simulate():
    with Pool(4) as pool:
        gains = dict(pool.imap_unordered(run_variant, VARIANT_NAMES))
        i_read = int(np.flatnonzero(CHECKPOINTS == READOUT_PRES)[0])
        curves = dict(pool.imap_unordered(readout_dynamics,
                                          [(n, gains[n][:, i_read].astype(float)) for n in VARIANT_NAMES]))
    out = dict(variants=np.array(VARIANT_NAMES), checkpoints=CHECKPOINTS, rel_angles=REL_ANGLES,
               pres_ms=C.PRES_MS, kappa=KAPPA, readout_pres=READOUT_PRES, n_seeds=N_SEEDS)
    for n in VARIANT_NAMES:
        out[f'g_{n}'] = gains[n]                         # (2 N_SEEDS, n_checkpoints, 2K); uniform seeds first
        out[f'tc_{n}'] = curves[n].astype(np.float32)    # (2 N_SEEDS, N_RF, n_angles), full dynamics
    return out


# ---------------------------------------------------------------------------------- analysis
def analytic_peaks(d, name):
    '''Peak response of each cRF neuron (analytic fixed point) for every seed and checkpoint:
    (2 N_SEEDS, n_checkpoints, N_RF).'''
    m = C.make_model(sv.CODEBASE)
    angles = np.radians(C.pref_deg()[C.ADAPTOR_IDX] + d['rel_angles'])
    g = d[f'g_{name}'].astype(float)
    return np.array([[C.tuning_curves(m, gk[:K], gk[K:], angles).max(axis=1) for gk in gi] for gi in g])


def profile(peaks_uniform, peaks_biased):
    '''Suppression profile per repeat: biased-adapted peak / mean uniform-adapted peak.
    Inputs (N_SEEDS, ..., N_RF). Returns (mean, s.e.m.) over repeats.'''
    ratio = peaks_biased / peaks_uniform.mean(axis=0)
    return ratio.mean(axis=0), ratio.std(axis=0, ddof=1) / np.sqrt(ratio.shape[0])


def analyse(d):
    n = int(d['n_seeds'])
    a = dict(rel_pref=C.rel_to_adaptor(C.pref_deg()), t=d['checkpoints'] * float(d['pres_ms']) / 1000.0)
    order = np.argsort(a['rel_pref'])
    a['order'] = order
    m0 = C.make_model(sv.CODEBASE)
    angles = np.radians(C.pref_deg()[C.ADAPTOR_IDX] + d['rel_angles'])
    a['unadapted_peak'] = C.tuning_curves(m0, np.zeros(K), np.zeros(K), angles).max(axis=1)
    for name in d['variants']:
        tc = d[f'tc_{name}'].astype(float)
        peak_u = tc[:n].max(axis=2)
        a[f'curves_{name}'] = tc[n:].mean(axis=0) / peak_u.mean(axis=0)[:, None]      # (N_RF, n_angles)
        a[f'prof_{name}'], a[f'sem_{name}'] = profile(peak_u, tc[n:].max(axis=2))
        pk = analytic_peaks(d, name)
        a[f'prof_t_{name}'], a[f'sem_t_{name}'] = profile(pk[:n], pk[n:])               # (n_ck, N_RF)
        a[f'uni_vs_unadapted_{name}'] = (pk[:n].mean(axis=0) / a['unadapted_peak']).mean(axis=1)
        g = d[f'g_{name}'].astype(float)[:, :, :K]                                      # cRF gains
        a[f'gmean_{name}'] = np.stack([g[:n].mean(axis=(0, 2)), g[n:].mean(axis=(0, 2))])
        a[f'gsur_{name}'] = d[f'g_{name}'].astype(float)[:, :, K:].mean(axis=(0, 2))
        i_read = int(np.flatnonzero(d['checkpoints'] == d['readout_pres'])[0])
        a[f'analytic_vs_dynamics_{name}'] = np.abs(a[f'prof_t_{name}'][i_read] - a[f'prof_{name}']).max()
    return a


def plot(d, a):
    import matplotlib.pyplot as plt
    ps.apply()
    col_old, col_new = ps.COLORS['old'], ps.COLORS['new']
    x = d['rel_angles']
    rel, order, t = a['rel_pref'], a['order'], a['t']
    t_read = float(d['readout_pres']) * float(d['pres_ms']) / 1000.0

    fig = plt.figure(figsize=(ps.FULL_WIDTH, 130 * ps.MM))
    top, h1, bot, h2 = 0.615, 0.335, 0.125, 0.295
    ax_tc = fig.add_axes([0.085, top, 0.525, h1])
    ax_pr = fig.add_axes([0.765, top, 0.22, h1])
    ax_ch = fig.add_axes([0.115, bot, 0.205, h2])
    ax_g = fig.add_axes([0.435, bot, 0.215, h2])
    ax_t = fig.add_axes([0.77, bot, 0.215, h2])

    # a: tuning curves, old and new overlaid (every second neuron, for legibility)
    shown = order[::2]
    for j, i in enumerate(shown):
        ax_tc.plot(x, a['curves_old'][i], color=col_old, lw=3.2, alpha=0.5, solid_capstyle='butt',
                   label='old equations (codebase)' if j == 0 else None)
    for j, i in enumerate(shown):
        ax_tc.plot(x, a['curves_new'][i], color=col_new, lw=1.3, label='new equations' if j == 0 else None)
    ps.ref_line(ax_tc, 1.0)
    ax_tc.set_xlim(-90, 90)
    ax_tc.set_xticks([-90, -45, 0, 45, 90])
    ax_tc.set_ylim(0, 1.45)
    ax_tc.set_yticks([0, 0.5, 1.0])
    ax_tc.set_xlabel('Stimulus orientation relative to adaptor (deg)')
    ax_tc.set_ylabel('Normalized response')
    ax_tc.legend(loc='upper left', ncol=2, columnspacing=1.5, bbox_to_anchor=(0.0, 1.03))

    # b: suppression profile
    def prof_line(ax, name, color, **kw):
        m_, s_ = a[f'prof_{name}'][order], a[f'sem_{name}'][order]
        ax.fill_between(rel[order], m_ - s_, m_ + s_, color=color, alpha=0.2, lw=0)
        return ax.plot(rel[order], m_, color=color, **kw)[0]

    ps.ref_line(ax_pr, 1.0)
    prof_line(ax_pr, 'old', col_old, lw=ps.LW_MAIN, marker='o', ms=3.5, label='old')
    prof_line(ax_pr, 'new', col_new, lw=ps.LW_MAIN, marker='o', ms=3.5, label='new')
    ax_pr.set_xlim(-90, 90)
    ax_pr.set_xticks([-90, 0, 90])
    ax_pr.set_ylim(0.85, 1.27)
    ax_pr.set_yticks([0.9, 1.0, 1.1, 1.2])
    ax_pr.set_xlabel('Preferred orientation\nrelative to adaptor (deg)')
    ax_pr.set_ylabel('Peak response,\nbiased / uniform')
    ax_pr.legend(loc='upper center', ncol=2, columnspacing=1.2, bbox_to_anchor=(0.5, 1.05))

    # c: each change on its own, as the change of the profile relative to the old equations
    ps.ref_line(ax_ch, 0.0)
    single = [('variance', r'variance from $c$', OK['black'], '-', ps.LW_MAIN),
              ('gating', r'gate $u^+$', OK['green'], '-', 3.0),
              ('placement', r'gain in $v$ eq.', OK['purple'], (0, (3, 2)), ps.LW_MAIN),
              ('rectify_c', r'rectified $c$', OK['orange'], '-', ps.LW_MAIN)]
    ax_ch.plot(rel[order], (a['prof_new'] - a['prof_old'])[order], color=col_new, lw=4.5, alpha=0.45,
               label='all (new)')
    for name, label, color, ls, lw in single:
        ax_ch.plot(rel[order], (a[f'prof_{name}'] - a['prof_old'])[order], color=color, lw=lw, ls=ls, label=label)
    ax_ch.set_xlim(-90, 90)
    ax_ch.set_xticks([-90, 0, 90])
    ax_ch.set_ylim(-0.115, 0.135)
    ax_ch.set_yticks([-0.1, 0, 0.1], labels=['−0.1', '0', '0.1'])
    ax_ch.set_xlabel('Preferred orientation\nrelative to adaptor (deg)')
    ax_ch.set_ylabel('Change in ratio\n(variant − old)')
    handles, labels_ = ax_ch.get_legend_handles_labels()
    ax_ch.legend(handles[2:], labels_[2:], loc='lower center', fontsize=8, handlelength=1.5,
                 bbox_to_anchor=(0.5, -0.03), labelspacing=0.2)
    ax_ch.text(0, 0.128, 'all changes (new)', fontsize=8, color=col_new, ha='center', va='top')
    ax_ch.text(0, 0.103, r'variance from $c$ only', fontsize=8, color=OK['black'], ha='center', va='top')

    # d: gains versus adaptation time
    ax_g.plot(t, a['gmean_old'][1], color=col_old, lw=ps.LW_MAIN, label='old')
    ax_g.plot(t, a['gmean_new'][1], color=col_new, lw=ps.LW_MAIN, label='new')
    if 'gmean_new_matched' in a:
        ax_g.plot(t, a['gmean_new_matched'][1], color=col_new, lw=ps.LW_MAIN, ls=(0, (1, 1.6)),
                  label='new, matched')
    ps.ref_line(ax_g, t_read, axis='v')
    ax_g.set_xlim(0, 40)
    ax_g.set_xticks([0, 10, 20, 30, 40])
    ax_g.set_ylim(0, 0.4)
    ax_g.set_yticks([0, 0.2, 0.4])
    ax_g.set_xlabel('Adaptation time (s)')
    ax_g.set_ylabel('Mean gain')
    ax_g.legend(loc='lower right', fontsize=8, bbox_to_anchor=(1.02, -0.02), handlelength=1.8)

    # e: suppression at the adaptor versus adaptation time
    i_ad = C.ADAPTOR_IDX
    ps.ref_line(ax_t, 1.0)
    for name, color, ls in (('old', col_old, '-'), ('new', col_new, '-'), ('new_matched', col_new, (0, (1, 1.6)))):
        if f'prof_t_{name}' not in a:
            continue
        m_, s_ = a[f'prof_t_{name}'][:, i_ad], a[f'sem_t_{name}'][:, i_ad]
        if ls == '-':
            ax_t.fill_between(t, m_ - s_, m_ + s_, color=color, alpha=0.2, lw=0)
        ax_t.plot(t, m_, color=color, lw=ps.LW_MAIN, ls=ls)
    ps.ref_line(ax_t, t_read, axis='v')
    ax_t.set_xlim(0, 40)
    ax_t.set_xticks([0, 10, 20, 30, 40])
    ax_t.set_ylim(0.8, 1.1)
    ax_t.set_yticks([0.8, 0.9, 1.0, 1.1])
    ax_t.set_xlabel('Adaptation time (s)')
    ax_t.set_ylabel('Ratio at the adaptor')
    ax_t.text(t_read + 1.2, 0.812, 'readout in a–c', fontsize=8, color='#4D4D4D', va='bottom')

    ps.label_panels(fig, [ax_tc, ax_pr, ax_ch, ax_g, ax_t])
    return ps.save(fig, NAME)


OK = ps.OKABE_ITO


def report(d, a):
    rel, order = a['rel_pref'], a['order']
    i_read = int(np.flatnonzero(d['checkpoints'] == d['readout_pres'])[0])
    print(f'  kappa = {float(d["kappa"]):.3f}; readout after {int(d["readout_pres"])} presentations '
          f'({a["t"][i_read]:.2f} s); preferred orientation relative to adaptor: {rel[order].round(0)}')
    for name in d['variants']:
        p, s = a[f'prof_{name}'][order], a[f'sem_{name}'][order]
        diff = np.abs(a[f'prof_{name}'] - a['prof_old']).max()
        print(f'  {name:13s} profile {np.round(p, 3)}')
        print(f'  {"":13s} s.e.m.  {np.round(s, 3)} | max |diff from old| {diff:.4f} | mean gain uniform/biased '
              f'{a[f"gmean_{name}"][0, i_read]:.3f}/{a[f"gmean_{name}"][1, i_read]:.3f} | uniform-adapted / unadapted '
              f'peak {a[f"uni_vs_unadapted_{name}"][i_read]:.3f} | analytic vs dynamics readout {a[f"analytic_vs_dynamics_{name}"]:.1e}')
    for name in ('old', 'new'):
        g = a[f'gmean_{name}'][1]
        t = a['t']
        print(f'  {name}: mean gain (biased) at t = {t[[11, 15, 18, -1]].round(1)} s: {g[[11, 15, 18, -1]].round(3)}; '
              f'growth over the last 8 s: {100 * (g[-1] - g[-5]) / g[-1] / (t[-1] - t[-5]):.2f} %/s; '
              f'adaptor ratio at 8.3 s and 40 s: {a[f"prof_t_{name}"][i_read, C.ADAPTOR_IDX]:.3f}, '
              f'{a[f"prof_t_{name}"][-1, C.ADAPTOR_IDX]:.3f}')


if __name__ == '__main__':
    d = None if '--recompute' in sys.argv else C.load_data(NAME)
    if d is None:
        C.save_data(NAME, **simulate())
        d = C.load_data(NAME)
    a = analyse(d)
    report(d, a)
    print('saved', plot(d, a))
