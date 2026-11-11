'''
figS1_timescales.py  ->  figures/FigS1_timescales.png, data/FigS1_timescales.npz

Time-scale diagnostic. The circuit sits at its steady state under the blank screen; at t = 0 a
small grating (cRF only, contrast 0.8, at the preferred orientation of cRF neuron 6) switches on.
Gains are zero (no adaptation), so the old and new equations are identical here. The responses
of the 13 cRF principal neurons y and of their normalization interneurons u and a are followed
for 50 ms, with the two gating signals u+ (new) and a+/(1+a+) (old).

Parameters: codebase time constants (tau_y 0.4, tau_a 0.2, tau_u 30 ms), dt = 0.01 ms (RK4),
surround delay 40 ms, all-ones normalization weights. No random numbers are used.

Run:  python figS1_timescales.py [--recompute]
'''

import sys

import numpy as np

import common as C
import plotstyle as ps
import simulation_variants as sv

NAME = 'FigS1_timescales'
DT = 0.01
T_MAX = 50.0
STIM_NEURON = 6


def simulate():
    m = C.make_model(sv.CODEBASE, dt=DT)
    theta = np.linspace(0, np.pi, C.N_RF, endpoint=False)[STIM_NEURON]
    z = C.oriented(theta, C.PROBE_CONTRAST, C.SMALL)
    start = m.settle(C.blank(), 0.0, 0.0, settle_ms=C.BASELINE_SETTLE)['state']
    q0 = m.q_of(start)
    m.set_surround_delay(C.SURROUND_DELAY)
    res = m.simulate(z[:, None, :], int(round(T_MAX / DT)), state0=start, adapt=False,
                     q_baseline=q0, record_traces=True)
    tr = res['traces']
    n = C.N_RF
    fp = m.analytic_fixed_point(z, np.zeros(m.K), np.zeros(m.K))
    return dict(t=(np.arange(tr['y'].shape[1]) + 1) * DT,
                y=tr['y'][0, :, :n], u=tr['u'][0, :, :n], a=tr['a'][0, :, :n],
                y_start=start[0, m.sl['y']][:n], u_start=start[0, m.sl['u']][:n], a_start=start[0, m.sl['a']][:n],
                y_ss=fp['y'][0, :n], u_ss=fp['u'][0, :n], a_ss=fp['a'][0, :n],
                tau=np.array([m.p.tau_y, m.p.tau_a, m.p.tau_u]), dt=DT, pres_ms=C.PRES_MS)


def settle_time(t, x, x_ss, frac=0.05):
    '''Last time at which |x - x_ss| exceeds frac of the total excursion |x(0) - x_ss|... of the
    mean trace; np.inf if not settled by the end.'''
    dev = np.abs(x - x_ss)
    span = np.abs(x).max() if np.isclose(dev[0], 0) else max(dev.max(), 1e-30)
    above = np.flatnonzero(dev > frac * span)
    if len(above) == 0:
        return t[0]
    return np.inf if above[-1] == len(t) - 1 else t[above[-1] + 1]


def plot(d):
    import matplotlib.pyplot as plt
    ps.apply()
    t = d['t']
    t_min = 0.1
    pres = float(d['pres_ms'])
    fig, axes = plt.subplots(2, 2, figsize=(ps.FULL_WIDTH, 112 * ps.MM), sharex=True)
    (ax_y, ax_u), (ax_a, ax_g) = axes

    def population(ax, x, x_ss, color, ylabel):
        ax.axhline(x_ss.mean(), color=color, lw=1.0, ls=(0, (1, 2)), zorder=1)
        ax.plot(t, x, color=color, lw=ps.LW_THIN, alpha=ps.ALPHA_THIN)
        ax.plot(t, x.mean(axis=1), color=color, lw=2.25)
        ax.set_ylabel(ylabel)

    population(ax_y, d['y'], d['y_ss'], ps.COLORS['principal'], r'Principal neurons, $y$')
    population(ax_u, d['u'], d['u_ss'], ps.COLORS['new'], r'Normalization neurons, $u$')
    population(ax_a, d['a'], d['a_ss'], ps.COLORS['old'], r'Normalization neurons, $a$')

    u_gate = np.sqrt(np.maximum(d['u'], 0)).mean(axis=1)
    a_plus = np.maximum(d['a'], 0)
    a_gate = (a_plus / (1 + a_plus)).mean(axis=1)
    ax_g.plot(t, a_gate, color=ps.COLORS['old'], lw=4.0, alpha=0.5, label=r'old:  $a^+/(1+a^+)$')
    ax_g.plot(t, u_gate, color=ps.COLORS['new'], lw=ps.LW_MAIN, ls=(0, (3, 2)), label=r'new:  $u^+$')
    ax_g.set_ylabel('Feedback gate')
    ax_g.legend(loc='upper left', bbox_to_anchor=(0.0, 1.03))

    # Inset: the difference between the two gates, on its own scale
    ins = ax_g.inset_axes([0.12, 0.34, 0.43, 0.28])
    ins.plot(t, (u_gate - a_gate) * 1e3, color=ps.COLORS['principal'], lw=1.25)
    ins.set_xscale('log')
    ins.set_xlim(t_min, T_MAX)
    ins.set_xticks([0.1, 1, 10, 50], labels=['0.1', '1', '10', '50'])
    ins.set_ylim(0, 0.65)
    ins.set_yticks([0, 0.5])
    ins.tick_params(labelsize=8, length=2.5, pad=2)
    ins.set_title(r'new $-$ old  ($\times 10^{-3}$)', fontsize=8, pad=3)
    ins.minorticks_off()

    for ax in axes.ravel():
        ax.set_xscale('log')
        ax.set_xlim(t_min, T_MAX)
        ax.set_xticks([0.1, 1, 10, 50], labels=['0.1', '1', '10', '50'])
        ps.ref_line(ax, pres, axis='v')
    for ax in (ax_a, ax_g):
        ax.set_xlabel('Time after stimulus onset (ms)')
    ax_y.set_ylim(0, 0.5)
    ax_y.set_yticks([0, 0.2, 0.4])
    ax_u.set_ylim(0.0125, 0.047)
    ax_u.set_yticks([0.02, 0.03, 0.04])
    ax_a.set_ylim(0.14, 0.28)
    ax_a.set_yticks([0.15, 0.20, 0.25])
    ax_g.set_ylim(0.13, 0.215)
    ax_g.set_yticks([0.14, 0.16, 0.18, 0.20])

    # annotations
    note = dict(fontsize=8, color='#4D4D4D')
    ax_u.text(pres / 1.12, 0.0133, 'one presentation\n(32 ms)', ha='right', va='bottom', **note)
    ax_y.text(0.03, 0.96, '13 cRF neurons (thin)\nand their mean (thick)', transform=ax_y.transAxes, va='top', **note)
    ax_y.text(0.115, d['y_ss'].mean() - 0.01, 'steady state\n(mean)', va='top', **note)
    ax_u.text(0.115, d['u_ss'].mean() - 0.0008, 'steady state', va='top', **note)
    ax_u.text(0.03, 0.62, 'identical in all\n13 neurons', transform=ax_u.transAxes, va='top', **note)
    ax_a.text(0.115, d['a_ss'].mean() - 0.0035, 'steady state', va='top', **note)
    ax_a.text(0.03, 0.62, 'identical in all\n13 neurons', transform=ax_a.transAxes, va='top', **note)

    fig.subplots_adjust(left=0.085, right=0.985, top=0.955, bottom=0.105, wspace=0.3, hspace=0.2)
    ps.label_panels(fig, [ax_y, ax_u, ax_a, ax_g])
    return ps.save(fig, NAME)


def report(d):
    t = d['t']
    u_gate = np.sqrt(np.maximum(d['u'], 0)).mean(axis=1)
    a_plus = np.maximum(d['a'], 0)
    a_gate = (a_plus / (1 + a_plus)).mean(axis=1)
    i32 = int(round(32.0 / d['dt'])) - 1
    ymean, yss = d['y'].mean(axis=1), d['y_ss'].mean()
    print(f'  y: peak of the mean {ymean.max():.3f} at t = {t[ymean.argmax()]:.1f} ms; steady state {yss:.3f}; '
          f'at 32 ms {ymean[i32]:.3f} ({100 * (ymean[i32] / yss - 1):+.0f}% vs steady state)')
    half = t[np.flatnonzero(ymean >= 0.5 * (ymean[0] + ymean.max()))[0]]
    print(f'  y: half-rise time {half:.2f} ms')
    frac_u = (d['u'][i32, 0] - d['u_start'][0]) / (d['u_ss'][0] - d['u_start'][0])
    print(f'  u: {100 * frac_u:.0f}% of its excursion completed at 32 ms; max u+ = {u_gate.max():.3f}')
    print(f'  gates: max |u+ - a+/(1+a+)| = {np.abs(u_gate - a_gate).max():.1e} '
          f'({100 * np.abs(u_gate - a_gate).max() / u_gate.max():.2f}% of the gate) at t = {t[np.abs(u_gate - a_gate).argmax()]:.1f} ms')
    print(f'  spread of u across cRF neurons: {np.ptp(d["u"], axis=1).max():.1e}')


if __name__ == '__main__':
    d = None if '--recompute' in sys.argv else C.load_data(NAME)
    if d is None:
        C.save_data(NAME, **simulate())
        d = C.load_data(NAME)
    report(d)
    print('saved', plot(d))
