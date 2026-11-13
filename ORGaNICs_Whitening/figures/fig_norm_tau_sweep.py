'''
fig_norm_tau_sweep.py - Fig. 3 of the normalization-settling diagnostic.

Settling time of the slowest population, max(T_y, T_u, T_a), over the (tau_a, tau_u) grid
(tau_y fixed). The black outline encloses grid points where every neuron settles within 30 ms
and the fixed point is linearly stable; hatched cells have an unstable fixed point. Markers:
original (V1Dynamics_Surround) and tuned time constants.

Input:  results/normalization_settling/normalization_settling.npz
Output: results/normalization_settling/figures/fig_norm_tau_sweep.{pdf,png}
Run from ORGaNICs_Whitening/:  python figures/fig_norm_tau_sweep.py
'''

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, ListedColormap
from _style import REPO_ROOT, SINGLE_COL, MM, COLORS, save

RESULTS = os.path.join(REPO_ROOT, 'results', 'normalization_settling')
VMAX = 60.0     # ms: the sweep window; longer settling times are shown at the top of the scale


def log_edges(grid):
    '''Cell edges for a log-spaced grid: geometric midpoints, extended by half a step at the ends.'''
    lg = np.log(grid)
    mid = (lg[1:] + lg[:-1]) / 2
    return np.exp(np.concatenate([[lg[0] - (mid[0] - lg[0])], mid, [lg[-1] + (lg[-1] - mid[-1])]]))


def outline(ax, mask, xe, ye, **kw):
    '''Draws the boundary of the True cells of mask (rows: y, cols: x).'''
    ny, nx = mask.shape
    m = np.pad(mask, 1)
    for i in range(ny):
        for j in range(nx):
            if not mask[i, j]:
                continue
            if not m[i + 1, j]:     # left neighbour
                ax.plot([xe[j], xe[j]], [ye[i], ye[i + 1]], **kw)
            if not m[i + 1, j + 2]: # right
                ax.plot([xe[j + 1], xe[j + 1]], [ye[i], ye[i + 1]], **kw)
            if not m[i, j + 1]:     # below
                ax.plot([xe[j], xe[j + 1]], [ye[i], ye[i]], **kw)
            if not m[i + 2, j + 1]: # above
                ax.plot([xe[j], xe[j + 1]], [ye[i + 1], ye[i + 1]], **kw)


def draw(d):
    tu, ta = d['tau_u_grid'], d['tau_a_grid']
    T_max = d['T_sweep'].max(axis=2)
    unstable = d['max_re_eig'] >= 0
    shown = np.where(np.isfinite(T_max), T_max, VMAX)
    shown = np.ma.masked_where(unstable, shown)
    xe, ye = log_edges(ta), log_edges(tu)

    fig, ax = plt.subplots(figsize=(SINGLE_COL, 72 * MM), constrained_layout=True)
    # Single-hue grey ramp, truncated so the slowest cells stay mid-dark and the black
    # feasibility outline remains visible against them.
    cmap = ListedColormap(plt.get_cmap('Greys')(np.linspace(0.03, 0.72, 256)))
    cmap.set_over(cmap(1.0))
    mesh = ax.pcolormesh(xe, ye, shown, cmap=cmap, norm=Normalize(0, VMAX), shading='flat',
                         edgecolors='white', linewidth=0.3)
    # Unstable fixed points: hatched, no fill.
    for i, j in zip(*np.nonzero(unstable)):
        ax.fill_between([xe[j], xe[j + 1]], ye[i], ye[i + 1], facecolor='white',
                        edgecolor='#555555', hatch='xxxx', lw=0)
    outline(ax, d['feasible'], xe, ye, color='black', lw=1.1, solid_capstyle='projecting', zorder=4)

    i, j = d['tuned_idx']
    ax.plot(float(d['orig_tau_a']), float(d['orig_tau_u']), 'o', ms=5, color=COLORS['original'],
            mec='white', mew=0.8, zorder=5, label='Original')
    ax.plot(ta[j], tu[i], 'D', ms=4.5, color=COLORS['tuned'], mec='white', mew=0.8, zorder=5,
            label='Tuned')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(xe[0], xe[-1])
    ax.set_ylim(ye[0], ye[-1])
    ax.set_xlabel(r'$\tau_a$ (ms)')
    ax.set_ylabel(r'$\tau_u$ (ms)')
    ax.set_xticks([0.05, 0.1, 0.2, 0.5, 1, 2])
    ax.set_xticklabels(['0.05', '0.1', '0.2', '0.5', '1', '2'])
    ax.set_yticks([0.5, 1, 2, 5, 10, 20, 50])
    ax.set_yticklabels(['0.5', '1', '2', '5', '10', '20', '50'])
    ax.minorticks_off()

    cb = fig.colorbar(mesh, ax=ax, extend='max', fraction=0.06, pad=0.02, aspect=25)
    cb.set_label('Settling time, slowest population (ms)')
    cb.ax.axhline(30, color='black', lw=1.1)
    cb.outline.set_linewidth(0.5)
    cb.ax.tick_params(width=0.5, length=2)

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    handles = ax.get_legend_handles_labels()[0] + [
        Line2D([], [], color='black', lw=1.1, label=r'$\leq$ 30 ms, stable'),
        Patch(facecolor='white', edgecolor='#555555', hatch='xxxx', lw=0, label='Unstable')]
    ax.legend(handles=handles, loc='lower left', bbox_to_anchor=(0.0, 1.01), ncol=2,
              fontsize=6, borderaxespad=0.0, columnspacing=1.2)
    return fig


if __name__ == '__main__':
    d = np.load(os.path.join(RESULTS, 'normalization_settling.npz'))
    for p in save(draw(d), os.path.join(RESULTS, 'figures'), 'fig_norm_tau_sweep'):
        print('wrote', p)
