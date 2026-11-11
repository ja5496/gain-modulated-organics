'''
figS2_settling.py  ->  figures/FigS2_settling.png, data/FigS2_settling.npz

Settling time after a stimulus switch, for the four combinations of the two listed changes
(feedback gate: a+/(1+a+) vs u+; gain: in the principal equation vs in the v equation).

Protocol: gains are frozen at the values reached after adapting the codebase equations to the
biased ensemble (small stimuli, 260 presentations x 32 ms = 8.32 s, seed 200). The circuit starts
at its steady state for a small grating at the adaptor orientation (cRF neuron 6) and, at t = 0,
the grating switches to the preferred orientation of cRF neuron 10 (55 deg away), as happens
between presentations of the ensemble. The mean squared deviation of the 13 cRF principal
neurons from the new fixed point is followed for 100 ms.

Parameters: codebase time constants, dt = 0.05 ms (RK4), surround delay 40 ms, contrast 0.8.

Run:  python figS2_settling.py [--recompute]
'''

import sys
from dataclasses import replace

import numpy as np

import common as C
import plotstyle as ps
import simulation_variants as sv

NAME = 'FigS2_settling'
DT = 0.05
T_MAX = 100.0
N_ADAPT_PRES = 260
SEED = 200
FROM_NEURON, TO_NEURON = 6, 10

COMBOS = {            # label: (options, colour, line style, line width)
    'old gate, old gain placement (codebase)': sv.CODEBASE,
    'new gate, old gain placement':            replace(sv.CODEBASE, gating='u'),
    'old gate, new gain placement':            replace(sv.CODEBASE, gain_placement='interneuron'),
    'new gate, new gain placement':            replace(sv.CODEBASE, gating='u', gain_placement='interneuron'),
}


def simulate():
    pref = np.linspace(0, np.pi, C.N_RF, endpoint=False)
    z_adapt, _ = C.ensemble_presentations(N_ADAPT_PRES, True, SEED, C.SMALL)
    g = C.adapt(C.make_model(sv.CODEBASE), z_adapt)['g_end'][0, -1]
    z0 = C.oriented(pref[FROM_NEURON], C.PROBE_CONTRAST, C.SMALL)
    z1 = C.oriented(pref[TO_NEURON], C.PROBE_CONTRAST, C.SMALL)
    out = dict(g=g, labels=np.array(list(COMBOS)), dt=DT, pres_ms=C.PRES_MS)
    n = C.N_RF
    for i, (label, opt) in enumerate(COMBOS.items()):
        m = C.make_model(opt, dt=DT)
        K = m.K
        start = m.settle(z0, g[:K], g[K:], settle_ms=1000.0)['state']
        target = m.settle(z1, g[:K], g[K:], settle_ms=1000.0, state0=start)['state']
        q0 = m.q_of(start)                      # the surround's delayed signal before the switch
        m.set_surround_delay(C.SURROUND_DELAY)
        res = m.simulate(z1[:, None, :], int(round(T_MAX / DT)), state0=start, adapt=False,
                         q_baseline=q0, record_traces=True)
        y = res['traces']['y'][0, :, :n]
        out[f'y_{i}'] = np.concatenate([start[:, m.sl['y']][:, :n], y])        # includes t = 0
        out[f'y_target_{i}'] = target[0, m.sl['y']][:n]
        fp = m.analytic_fixed_point(z1, g[:K], g[K:])
        out[f'y_analytic_{i}'] = fp['y'][0, :n]
        out[f'zprime_min_{i}'] = fp['z_prime'].min()
    out['t'] = np.arange(out['y_0'].shape[0]) * DT
    return out


def plot(d):
    import matplotlib.pyplot as plt
    ps.apply()
    t = d['t']
    labels = list(d['labels'])
    styles = [dict(color=ps.COLORS['old'], lw=5.0, alpha=0.35, solid_capstyle='butt'),
              dict(color=ps.COLORS['new'], lw=2.75, alpha=0.6),
              dict(color=ps.COLORS['old'], lw=ps.LW_MAIN, ls=(0, (4, 2))),
              dict(color=ps.COLORS['new'], lw=ps.LW_MAIN, ls=(0, (1, 1.6)))]
    short = ['old gate, old gain placement (codebase)', 'new gate, old gain placement',
             'old gate, new gain placement', 'new gate, new gain placement']
    pres = float(d['pres_ms'])

    fig, (ax, ax_d) = plt.subplots(1, 2, figsize=(ps.FULL_WIDTH, 82 * ps.MM))
    mse = [np.mean((d[f'y_{i}'] - d[f'y_target_{i}']) ** 2, axis=1) for i in range(4)]
    for i in range(4):
        ax.plot(t, mse[i], label=short[i], **styles[i])
    ax.set_yscale('log')
    ax.set_xlim(0, T_MAX)
    ax.set_ylim(1e-9, 3e-2)
    ax.set_yticks([1e-8, 1e-6, 1e-4, 1e-2])
    ax.set_xlabel('Time after stimulus switch (ms)')
    ax.set_ylabel('Squared deviation from\nnew fixed point (mean over neurons)')
    ps.ref_line(ax, pres, axis='v')
    ax.text(pres + 2, 1.4e-2, 'next stimulus\n(32 ms)', fontsize=8, color='#4D4D4D', va='top')
    fig.legend(*ax.get_legend_handles_labels(), loc='upper center', ncol=2, fontsize=9, handlelength=2.6,
               columnspacing=2.5, bbox_to_anchor=(0.5, 1.0))

    # Deviation of each variant's trajectory from the codebase trajectory
    floor = 1e-17
    for i in (1, 2, 3):
        dev = np.abs(d[f'y_{i}'] - d['y_0']).max(axis=1)
        ax_d.plot(t[1:], np.maximum(dev[1:], floor), **{**styles[i], 'alpha': 1.0})
    ax_d.set_yscale('log')
    ax_d.set_xlim(0, T_MAX)
    ax_d.set_ylim(3e-18, 1)
    ax_d.set_yticks([1e-16, 1e-12, 1e-8, 1e-4, 1])
    ax_d.set_xlabel('Time after stimulus switch (ms)')
    ax_d.set_ylabel('Largest difference from codebase\nresponse (units of $y$)')
    ps.ref_line(ax_d, pres, axis='v')
    peak = np.abs(d['y_0']).max()
    ax_d.axhline(peak, color=ps.COLORS['principal'], lw=1.0, ls=(0, (1, 2)))
    ax_d.text(98, peak * 0.35, 'largest response', fontsize=8, color='#4D4D4D', ha='right', va='top')
    ax_d.text(98, 3e-5, 'new gate', fontsize=8, color=ps.COLORS['new'], ha='right', va='bottom')
    ax_d.text(98, 4e-16, 'new gain placement only\n(round-off)', fontsize=8, color=ps.COLORS['old'], ha='right', va='bottom')

    fig.subplots_adjust(left=0.115, right=0.985, top=0.8, bottom=0.145, wspace=0.42)
    ps.label_panels(fig, [ax, ax_d])
    return ps.save(fig, NAME)


def report(d):
    t = d['t']
    i32 = int(round(32.0 / d['dt']))
    print(f'  frozen gains: mean {d["g"][:182].mean():.3f}, max {d["g"][:182].max():.3f}')
    for i, label in enumerate(d['labels']):
        mse = np.mean((d[f'y_{i}'] - d[f'y_target_{i}']) ** 2, axis=1)
        rel = np.abs(d[f'y_target_{i}'] - d[f'y_analytic_{i}']).max() / np.abs(d[f'y_analytic_{i}']).max()
        dev = np.abs(d[f'y_{i}'] - d['y_0']).max()
        # time to reduce the squared deviation 100-fold
        below = np.flatnonzero(mse <= 1e-2 * mse[0])
        t100 = t[below[0]] if len(below) else np.inf
        print(f'  {label:40s}: MSE(0) {mse[0]:.2e}, MSE(32 ms)/MSE(0) {mse[i32] / mse[0]:.3f}, '
              f'100-fold reduction at {t100:.1f} ms, max |y - y_codebase| {dev:.1e}, '
              f'settled vs analytic fixed point {rel:.1e} (min z\' {d[f"zprime_min_{i}"]:+.1e})')
    mse = np.mean((d['y_0'] - d['y_target_0']) ** 2, axis=1)
    sl = slice(int(50 / d['dt']), int(90 / d['dt']))
    rate = -np.polyfit(t[sl], np.log(mse[sl]), 1)[0]
    print(f'  late decay of the squared deviation: time constant {2 / rate:.1f} ms for the deviation itself')


if __name__ == '__main__':
    d = None if '--recompute' in sys.argv else C.load_data(NAME)
    if d is None:
        C.save_data(NAME, **simulate())
        d = C.load_data(NAME)
    report(d)
    print('saved', plot(d))
