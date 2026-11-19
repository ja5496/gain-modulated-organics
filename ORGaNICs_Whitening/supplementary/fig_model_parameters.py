'''
fig_model_parameters.py

Supplementary Note 2 figure: the model parameters shared by every simulation in the main text.

Every matrix is read from the model classes themselves, built exactly as the analysis scripts
build them (analysis/adapt_single_stimulus.py, Kohn_distinct_plots.py, adapt_ensembles.py,
power_law.py: N_RF = 13, N_SETS = 6, frame N13_mercedes_K182, tuning width 0.75 rad), so the
figure cannot drift from the code.

    a  network layout: RF sets, neuron and interneuron counts
    b  feedforward drive z_i(theta) of the 13 neurons of one RF (contrast 1)
    c  target covariance of one RF (the mid-contrast target; the low-contrast one is a scaled copy)
    d  recurrent weights W_yy (78 x 78)
    e  normalization weights N (78 x 78)
    f  full frame W_yv (78 x 1092): block diagonal, so gain feedback is local to each RF set
    g  frame W_yv of one RF set (13 x 182)

Run from ORGaNICs_Whitening/:  python supplementary/fig_model_parameters.py
Writes supplementary/figures/fig_model_parameters.{pdf,png}.
'''

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, HERE)

import plotstyle as ps                                   # noqa: E402  (sets the Agg backend)
import matplotlib.pyplot as plt                          # noqa: E402
from matplotlib.gridspec import GridSpec                 # noqa: E402
from matplotlib.patches import Circle   # noqa: E402
from scipy.linalg import block_diag                      # noqa: E402

from simulation import Frame, V1Dynamics_Surround        # noqa: E402
from stimuli import StimulusGenerator                    # noqa: E402
from tunings import V1Tunings                            # noqa: E402

# ---- Shared configuration (as in the analysis scripts) ----
N_RF, N_SETS = 13, 6
TUNING_WIDTH = 0.75
FRAME_PATH = os.path.join(REPO_ROOT, 'data/frames/N13_mercedes_K182_Frame.csv')
TARGET_COV_MID = os.path.join(REPO_ROOT, 'data/target_covs/uniform_target_covariance_mid_c.csv')
TARGET_COV_LOW = os.path.join(REPO_ROOT, 'data/target_covs/uniform_target_covariance_low_c.csv')
SET_LABELS = ['C', 'S1', 'S2', 'S3', 'S4', 'S5']
DIVERGING = 'RdBu_r'


def build_model():
    tunings = V1Tunings(N=N_RF)
    frame = Frame(csv_path=FRAME_PATH)
    dyn = V1Dynamics_Surround(tunings, frame, dt=0.1, N_RF=N_RF, N_SETS=N_SETS,
                              target_covariance_path=TARGET_COV_MID, gains_nonneg=True)
    stim = StimulusGenerator(N_RF=N_RF, N_SETS=N_SETS, num_angles=N_RF, tuning_width=TUNING_WIDTH)
    return tunings, frame, dyn, stim


def summary(frame, dyn, stim):
    '''Numbers quoted in the Supplementary Note text.'''
    K = frame.K
    W = frame.W
    eig = np.linalg.eigvalsh(W @ W.T)
    G = np.abs(W.T @ W)
    np.fill_diagonal(G, 0)
    C_mid = dyn.uniform_target_covariance
    C_low = np.loadtxt(TARGET_COV_LOW, delimiter=',')
    ev = np.linalg.eigvalsh(C_mid)[::-1]
    info = dict(
        N_principal=N_RF * N_SETS, N_u=N_RF * N_SETS, N_a=N_RF * N_SETS, K_per_RF=K,
        N_v=K * N_SETS, K_min=N_RF * (N_RF + 1) // 2,
        frame_col_norm_range=(np.linalg.norm(W, axis=0).min(), np.linalg.norm(W, axis=0).max()),
        frame_eig_WWt=(eig.min(), eig.max()), frame_max_coherence=G.max(),
        W_yy_is_identity=np.allclose(dyn.W_yy, np.eye(N_RF * N_SETS)),
        N_is_all_ones=np.allclose(dyn.N_matrix, 1.0),
        target_var_mid=np.diag(C_mid).mean(), target_var_low=np.diag(C_low).mean(),
        low_over_mid=np.trace(C_low) / np.trace(C_mid),
        target_top_eigs_mid=ev[:3], target_frac_var_top2=ev[:2].sum() / ev.sum(),
        max_drive=stim.max_drive, baseline_drive=0.15 / stim.max_drive,
        peak_drive=1.0 / stim.max_drive,
    )
    return info


def block_lines(ax, n_blocks, size, color='0.55', lw=0.4):
    for k in range(1, n_blocks):
        ax.axhline(k * size - 0.5, color=color, lw=lw)
        ax.axvline(k * size - 0.5, color=color, lw=lw)


def set_ticks(ax):
    centers = np.arange(N_SETS) * N_RF + (N_RF - 1) / 2
    ax.set_xticks(centers, SET_LABELS)
    ax.set_yticks(centers, SET_LABELS)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.6)


def colorbar(fig, im, ax, ticks=None, label=None):
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, ticks=ticks)
    cb.outline.set_linewidth(0.5)
    cb.ax.tick_params(labelsize=8, width=0.6, length=2)
    if label:
        cb.set_label(label, fontsize=8)
    return cb


def matrix_panel(fig, ax, M, vmax, ticks, label=None):
    im = ax.imshow(M, cmap=DIVERGING, vmin=-vmax, vmax=vmax, interpolation='nearest')
    block_lines(ax, N_SETS, N_RF)
    set_ticks(ax)
    ax.set_xlabel('From RF set')
    ax.set_ylabel('To RF set')
    colorbar(fig, im, ax, ticks=ticks, label=label)


def draw_layout(ax, ax_txt, info):
    '''RF sets: cRF (C) in the middle, five surround sets around it. Red web: normalization pools
    over every neuron. Blue rings: adaptation is local to each RF set. Counts in ax_txt.'''
    ax.set_xlim(-1.42, 1.42)
    ax.set_ylim(-1.42, 1.42)
    ax.set_aspect('equal')
    ax.axis('off')
    r_ring, r_c, r_l = 0.95, 0.25, 0.1
    centers = [(0.0, 0.0)] + [(r_ring * np.cos(np.pi / 2 + 2 * np.pi * k / 5),
                               r_ring * np.sin(np.pi / 2 + 2 * np.pi * k / 5)) for k in range(5)]
    norm_c, adapt_c = ps.OKABE_ITO['vermillion'], ps.OKABE_ITO['blue']
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            ax.plot(*zip(centers[i], centers[j]), color=norm_c, lw=0.6, zorder=1)
    for (x, y), lab in zip(centers, SET_LABELS):
        ang = np.arctan2(y, x) if lab != 'C' else -np.pi / 2 + np.pi / 5
        ax.add_patch(Circle((x + (r_c + r_l) * np.cos(ang), y + (r_c + r_l) * np.sin(ang)), r_l,
                            facecolor='none', edgecolor=adapt_c, lw=1.0, zorder=2))
        ax.add_patch(Circle((x, y), r_c, facecolor='white', edgecolor='black', lw=1.0, zorder=3))
        ax.text(x, y, lab, ha='center', va='center', fontsize=7, zorder=4)

    ax_txt.axis('off')
    lines = [
        ('RF sets', f'{N_SETS} (C + 5 surround)'),
        ('Principal neurons $y$', f'{N_SETS} $\\times$ {N_RF} = {info["N_principal"]}'),
        ('Normalization $u$, $a$', f'{info["N_u"]} each'),
        ('Gain-modulated $v$', f'{N_SETS} $\\times$ {info["K_per_RF"]} = {info["N_v"]:,}'),
    ]
    for k, (name, val) in enumerate(lines):
        ax_txt.text(0.0, 1.0 - 0.215 * k, name, ha='left', va='top', fontsize=8,
                    transform=ax_txt.transAxes)
        ax_txt.text(0.0, 1.0 - 0.215 * k - 0.095, val, ha='left', va='top', fontsize=8,
                    color='0.35', transform=ax_txt.transAxes)
    ax_txt.plot([0.0, 0.12], [0.1, 0.1], color=norm_c, lw=1.0, transform=ax_txt.transAxes,
                clip_on=False)
    ax_txt.text(0.17, 0.1, 'normalization', va='center', fontsize=8, transform=ax_txt.transAxes)
    ax_txt.plot([0.06], [0.0], 'o', ms=5, mfc='none', mec=adapt_c, mew=1.0,
                transform=ax_txt.transAxes, clip_on=False)
    ax_txt.text(0.17, 0.0, 'adaptation', va='center', fontsize=8, transform=ax_txt.transAxes)


def main():
    ps.apply()
    tunings, frame, dyn, stim = build_model()
    info = summary(frame, dyn, stim)
    for k, v in info.items():
        print(f'{k:>24s}: {v}')

    W = frame.W
    W_full = block_diag(*[W] * N_SETS)               # (78, 1092)

    fig = plt.figure(figsize=(ps.FULL_WIDTH, 168 * ps.MM))
    outer = GridSpec(3, 1, figure=fig, height_ratios=[1.1, 1.0, 0.6],
                     left=0.08, right=0.94, bottom=0.07, top=0.965, hspace=0.6)
    top_outer = outer[0].subgridspec(1, 2, width_ratios=[1.65, 2.35], wspace=0.25)
    top_a = top_outer[0].subgridspec(1, 2, width_ratios=[1.1, 0.9], wspace=0.05)
    top_bc = top_outer[1].subgridspec(1, 2, width_ratios=[1, 1.15], wspace=0.7)
    mid = outer[1].subgridspec(1, 3, width_ratios=[1, 1, 1.35], wspace=0.6)

    # a: layout and counts
    ax_a = fig.add_subplot(top_a[0])
    draw_layout(ax_a, fig.add_subplot(top_a[1]), info)

    # b: feedforward drive of one RF's 13 neurons
    ax_b = fig.add_subplot(top_bc[0])
    theta = np.linspace(0, np.pi, 721)
    Z = np.stack([stim.oriented_drive(t, 1.0, 'adapt CRF only')[:N_RF] for t in theta], axis=1)
    cols = plt.get_cmap(ps.SEQUENTIAL)(np.linspace(0, 0.9, N_RF))
    for i in range(N_RF):
        ax_b.plot(np.degrees(theta), Z[i], color=cols[i], lw=0.9)
    ps.ref_line(ax_b, info['baseline_drive'], axis='h', color='black', lw=0.9)
    ax_b.set_xlim(0, 180)
    ax_b.set_xticks([0, 90, 180])
    ax_b.set_ylim(0, None)
    ax_b.set_xlabel('Stimulus orient. (deg)')
    ax_b.set_ylabel('Input drive, $z_i$')

    # c: target covariance (one RF)
    ax_c = fig.add_subplot(top_bc[1])
    C = dyn.uniform_target_covariance
    vmax = np.abs(C).max()
    step = 180 / N_RF
    im = ax_c.imshow(C, cmap=DIVERGING, vmin=-vmax, vmax=vmax, interpolation='nearest',
                     extent=[-step / 2, 180 - step / 2, 180 - step / 2, -step / 2])
    ax_c.set_xticks([0, 90, 180 - step])
    ax_c.set_xticklabels(['0', '90', f'{180 - step:.0f}'])
    ax_c.set_yticks([0, 90, 180 - step])
    ax_c.set_yticklabels(['0', '90', f'{180 - step:.0f}'])
    ax_c.set_xlabel('Pref. orient. (deg)')
    ax_c.set_ylabel('Pref. orient. (deg)')
    for s in ax_c.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.6)
    cb = colorbar(fig, im, ax_c, ticks=[-0.004, 0, 0.004])
    cb.ax.set_yticklabels(['$-4$', '0', '4'])
    cb.set_label(r'$\times 10^{-3}$', fontsize=8, labelpad=1)

    # d, e: 78 x 78 matrices
    ax_d = fig.add_subplot(mid[0])
    matrix_panel(fig, ax_d, dyn.W_yy, 1.0, [-1, 0, 1])
    ax_e = fig.add_subplot(mid[1])
    matrix_panel(fig, ax_e, dyn.N_matrix, 1.0, [-1, 0, 1])

    # f: full block-diagonal frame W_yv (78 x 1092)
    ax_f = fig.add_subplot(mid[2])
    im = ax_f.imshow(W_full, cmap=DIVERGING, vmin=-1, vmax=1, aspect='auto',
                     interpolation='nearest')
    for k in range(1, N_SETS):
        ax_f.axhline(k * N_RF - 0.5, color='0.55', lw=0.4)
        ax_f.axvline(k * frame.K - 0.5, color='0.55', lw=0.4)
    ax_f.set_yticks(np.arange(N_SETS) * N_RF + (N_RF - 1) / 2, SET_LABELS)
    ax_f.set_xticks(np.arange(N_SETS) * frame.K + (frame.K - 1) / 2, SET_LABELS)
    ax_f.tick_params(length=0)
    for sp in ax_f.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.6)
    ax_f.set_xlabel('RF set of $v$')
    ax_f.set_ylabel('RF set of $y$')
    colorbar(fig, im, ax_f, ticks=[-1, 0, 1])

    # g: one RF's frame W_yv (13 x 182)
    ax_g = fig.add_subplot(outer[2])
    im = ax_g.imshow(W, cmap=DIVERGING, vmin=-1, vmax=1, aspect='auto', interpolation='nearest',
                     extent=[0.5, frame.K + 0.5, 180 - step / 2, -step / 2])
    ax_g.set_yticks([0, 90, 180 - step])
    ax_g.set_yticklabels(['0', '90', f'{180 - step:.0f}'])
    ax_g.set_xticks([1, 50, 100, 150, frame.K])
    ax_g.set_xlabel('Gain-modulated interneuron $v_k$ (one RF set)')
    ax_g.set_ylabel('Pref. orient. (deg)')
    for s in ax_g.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.6)
    cb = fig.colorbar(im, ax=ax_g, fraction=0.02, pad=0.012, ticks=[-1, 0, 1])
    cb.outline.set_linewidth(0.5)
    cb.ax.tick_params(labelsize=8, width=0.6, length=2)

    ps.label_panels(fig, [ax_a, ax_b, ax_c, ax_d, ax_e, ax_f, ax_g])

    out_dir = os.path.join(HERE, 'figures')
    os.makedirs(out_dir, exist_ok=True)
    for ext in ('pdf', 'png'):
        path = os.path.join(out_dir, f'fig_model_parameters.{ext}')
        fig.savefig(path, dpi=ps.DPI)
        print('saved', path)
    plt.close(fig)


if __name__ == '__main__':
    main()
