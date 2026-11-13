'''
fig_norm_timecourse.py - Fig. 1 of the normalization-settling diagnostic.

Population-mean principal-neuron (y) and normalization-interneuron (u, a) responses to a
full-field grating switched on at t = 0, for V1Dynamics_Surround's time constants ("original")
and for the tuned (tau_u, tau_a) chosen by analysis/normalization_settling.py, against the
analytic steady state. Dots mark each population's 1 % settling time.

Input:  results/normalization_settling/normalization_settling.npz
Output: results/normalization_settling/figures/fig_norm_timecourse.{pdf,png}
Run from ORGaNICs_Whitening/:  python figures/fig_norm_timecourse.py
'''

import os
import numpy as np
import matplotlib.pyplot as plt
from _style import REPO_ROOT, DOUBLE_COL, MM, COLORS, panel_letters, save

RESULTS = os.path.join(REPO_ROOT, 'results', 'normalization_settling')
T_MAX = 200.0       # ms shown
DEADLINE = 30.0     # ms


def draw(d):
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, 52 * MM), constrained_layout=True)
    pops = [('y', r'Mean principal response $\langle y \rangle$', d['y_ss'].mean()),
            ('u', r'Mean interneuron $\langle u \rangle$', float(d['u_ss'])),
            ('a', r'Mean interneuron $\langle a \rangle$', float(d['a_ss']))]
    for ax, (pop, ylabel, x_ss) in zip(axes, pops):
        ax.axvspan(0, DEADLINE, color=COLORS['window'], lw=0, zorder=0)
        ax.axhline(x_ss, color=COLORS['reference'], lw=0.75, ls=(0, (3, 2)), zorder=1,
                   label='Analytic steady state')
        for cond, prefix in [('original', 'orig'), ('tuned', 'tuned')]:
            t = d[f'{prefix}_t']
            x = d[f'{prefix}_{pop}_mean']
            keep = t <= T_MAX
            tau_u, tau_a = float(d[f'{prefix}_tau_u']), float(d[f'{prefix}_tau_a'])
            label = (f"{cond.capitalize()} ($\\tau_u$ = {tau_u:.3g} ms, $\\tau_a$ = {tau_a:.3g} ms)")
            ax.plot(t[keep], x[keep], color=COLORS[cond], lw=1.0, zorder=3, label=label)
            T = float(d[f'{prefix}_T_{pop}'])
            if np.isfinite(T) and T <= T_MAX:
                k = np.argmin(np.abs(t - T))
                ax.plot(T, x[k], 'o', ms=3.2, color=COLORS[cond], mec='white', mew=0.6, zorder=4)
        ax.set_xlim(-5, T_MAX)
        ax.set_xlabel('Time from stimulus onset (ms)')
        ax.set_ylabel(ylabel + ' (a.u.)')
        ax.set_ylim(bottom=0)
        ax.text(DEADLINE + 2, 0.98, f'{DEADLINE:.0f} ms', transform=ax.get_xaxis_transform(),
                ha='left', va='top', fontsize=5.5, color='#555555')
    axes[0].legend(loc='upper right', bbox_to_anchor=(1.0, 0.9))
    panel_letters(axes)
    return fig


if __name__ == '__main__':
    d = np.load(os.path.join(RESULTS, 'normalization_settling.npz'))
    for p in save(draw(d), os.path.join(RESULTS, 'figures'), 'fig_norm_timecourse'):
        print('wrote', p)
