'''
fig_norm_discrepancy.py - Fig. 2 of the normalization-settling diagnostic.

Discrepancy between the principal neurons' firing rates and the analytic normalized state,
    e(t) = || floor(y(t))^2 - y*^2 ||_2 / || y*^2 ||_2,   y*^2 = z^2 / (sigma^2 + N z^2),
after a full-field grating is switched on at t = 0, for the original and tuned time constants.
Dots mark the time after which e(t) stays below 1 %.

Input:  results/normalization_settling/normalization_settling.npz
Output: results/normalization_settling/figures/fig_norm_discrepancy.{pdf,png}
Run from ORGaNICs_Whitening/:  python figures/fig_norm_discrepancy.py
'''

import os
import numpy as np
import matplotlib.pyplot as plt
from _style import REPO_ROOT, SINGLE_COL, MM, COLORS, save

RESULTS = os.path.join(REPO_ROOT, 'results', 'normalization_settling')
T_MAX = 200.0
DEADLINE = 30.0
TOL = 0.01


def draw(d):
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 62 * MM), constrained_layout=True)
    ax.axvspan(0, DEADLINE, color=COLORS['window'], lw=0, zorder=0)
    ax.axhline(TOL, color=COLORS['reference'], lw=0.75, ls=(0, (3, 2)), zorder=1)
    ax.text(T_MAX, TOL * 1.25, '1 %', ha='right', va='bottom', fontsize=6, color='#555555')
    for cond, prefix in [('original', 'orig'), ('tuned', 'tuned')]:
        t, e = d[f'{prefix}_t'], d[f'{prefix}_rate_err']
        keep = (t > 0) & (t <= T_MAX)
        tau_u, tau_a = float(d[f'{prefix}_tau_u']), float(d[f'{prefix}_tau_a'])
        ax.plot(t[keep], e[keep], color=COLORS[cond], lw=1.0, zorder=3,
                label=f"{cond.capitalize()} ($\\tau_u$ = {tau_u:.3g} ms, $\\tau_a$ = {tau_a:.3g} ms)")
        T = float(d[f'{prefix}_T_rate'])
        if np.isfinite(T) and T <= T_MAX:
            ax.plot(T, TOL, 'o', ms=3.2, color=COLORS[cond], mec='white', mew=0.6, zorder=4)
    ax.set_yscale('log')
    ax.set_xlim(0, T_MAX)
    ax.set_ylim(1e-4, 30)
    ax.set_xlabel('Time from stimulus onset (ms)')
    ax.set_ylabel('Relative error of firing rates\nfrom normalized state')
    ax.text(DEADLINE + 2, 0.98, f'{DEADLINE:.0f} ms', transform=ax.get_xaxis_transform(),
            ha='left', va='top', fontsize=5.5, color='#555555')
    ax.legend(loc='upper right')
    return fig


if __name__ == '__main__':
    d = np.load(os.path.join(RESULTS, 'normalization_settling.npz'))
    for p in save(draw(d), os.path.join(RESULTS, 'figures'), 'fig_norm_discrepancy'):
        print('wrote', p)
