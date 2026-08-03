"""
test_frame.py

Tests whether unequal v² values across frame vectors are a geometric property
of the frame itself (W), independent of network dynamics.

Procedure:
1. Build a small frame (N=11, K=66) via frame_whiten.py
2. Generate 10 raised-cosine input vectors centered at different neuron indices
3. Compute v = W.T @ y for each input; record v² element-wise
4. Print ranked table and plot average v² (± std) vs abs(sum(w)) (Figure 1)
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from frame_whiten import Frame

np.random.seed(20)

# ── Parameters ────────────────────────────────────────────────────────────────
N            = 11
TUNING_WIDTH = 0.5
N_INPUTS     = 10      # raised-cosine inputs centered at indices 0..N_INPUTS-1

# ── Helpers ───────────────────────────────────────────────────────────────────
def build_inputs(N, n_inputs, tuning_width=0.5):
    """Return list of n_inputs raised-cosine vectors (each shape (N,))."""
    theta = np.linspace(0, np.pi, N, endpoint=False)
    inputs = []
    for c in range(n_inputs):
        y = np.exp(tuning_width * np.cos(2 * (theta - theta[c])))
        y /= y.max()
        y -= np.mean(y)
        inputs.append(y)
    return inputs

def compute_vsq_stats(W, inputs):
    """Return avg_vsq (K,) and std_vsq (K,) over the input stream."""
    vsq_all = np.stack([(W.T @ y) ** 2 for y in inputs], axis=0)  # (n_inputs, K)
    return vsq_all.mean(axis=0), vsq_all.std(axis=0)

# ── Build mercedes frame and compute stats ────────────────────────────────────
frame = Frame(dim=N)
W = frame.W          # (N, K)
K = frame.K

inputs = build_inputs(N, N_INPUTS)
avg_vsq, std_vsq = compute_vsq_stats(W, inputs)
sum_w = np.abs(W.sum(axis=0))   # (K,)
sort_idx = np.argsort(avg_vsq)[::-1]

# ── Build Gaussian frame and compute stats ────────────────────────────────────
frame_g = Frame(dim=N, frame_type='gaussian')
W_g = frame_g.W
avg_vsq_g, std_vsq_g = compute_vsq_stats(W_g, inputs)
sort_idx_g = np.argsort(avg_vsq_g)[::-1]

# ── Terminal output ────────────────────────────────────────────────────────────
vec_str_width = N * 8
header = f"{'Rank':<6}{'Avg v²':<14}{'Std v²':<14}  Frame vector"
print("\n" + header)
print("-" * (len(header) + vec_str_width))
for rank, idx in enumerate(sort_idx):
    vec_str = "  ".join(f"{x:+.4f}" for x in W[:, idx])
    print(f"{rank:<6}{avg_vsq[idx]:<14.5f}{std_vsq[idx]:<14.5f}  [{vec_str}]")

# ── Figure 1 ──────────────────────────────────────────────────────────────────
fig1, (ax_bar, ax_bar_g, ax_top, ax_bot) = plt.subplots(1, 4, figsize=(24, 5))

# Left: bar chart sorted descending
ax_bar.bar(np.arange(K), avg_vsq[sort_idx], color='steelblue', edgecolor='none')
ax_bar.axhline(avg_vsq.mean(), color='red', linestyle='--', linewidth=1.5,
               label=f'mean = {avg_vsq.mean():.4f}')
ax_bar.set_xlabel("Frame vector rank (sorted by avg v²)", fontsize=12)
ax_bar.set_ylabel("Average v²", fontsize=12)
ax_bar.set_title("Avg v² per frame vector\n(sorted descending)", fontsize=13, fontweight='bold')
ax_bar.legend()

# Second: Gaussian frame histogram sorted descending
ax_bar_g.bar(np.arange(K), avg_vsq_g[sort_idx_g], color='mediumpurple', edgecolor='none')
ax_bar_g.axhline(avg_vsq_g.mean(), color='red', linestyle='--', linewidth=1.5,
                 label=f'mean = {avg_vsq_g.mean():.4f}')
ax_bar_g.set_xlabel("Frame vector rank (sorted by avg v²)", fontsize=12)
ax_bar_g.set_ylabel("Average v²", fontsize=12)
ax_bar_g.set_title("Gaussian frame: avg v²\n(sorted descending)", fontsize=13, fontweight='bold')
ax_bar_g.legend()

# Third: top-3 frame vectors by avg v² rank
neuron_idx = np.arange(N)
blue_shades = ['#08306b', '#2171b5', '#6baed6']
red_shades  = ['#cb181d', '#fb6a4a', '#fcae91']

for rank, color in zip(range(3), blue_shades):
    idx = sort_idx[rank]
    ax_top.plot(neuron_idx, W[:, idx], color=color, linewidth=2, marker='o', markersize=5,
                label=f'rank {rank} (avg v²={avg_vsq[idx]:.3f})')
ax_top.axhline(0, color='grey', linewidth=0.8, linestyle=':')
ax_top.set_xlabel("Neuron index", fontsize=12)
ax_top.set_ylabel("Component value", fontsize=12)
ax_top.set_title("Top 3 frame vectors", fontsize=13, fontweight='bold')
ax_top.set_xticks(neuron_idx)
ax_top.legend(fontsize=8)

# Right: bottom-3 frame vectors by avg v² rank
for rank, color in zip(range(K-3, K), red_shades):
    idx = sort_idx[rank]
    ax_bot.plot(neuron_idx, W[:, idx], color=color, linewidth=2, marker='o', markersize=5,
                label=f'rank {rank} (avg v²={avg_vsq[idx]:.3f})')
ax_bot.axhline(0, color='grey', linewidth=0.8, linestyle=':')
ax_bot.set_xlabel("Neuron index", fontsize=12)
ax_bot.set_ylabel("Component value", fontsize=12)
ax_bot.set_title("Bottom 3 frame vectors", fontsize=13, fontweight='bold')
ax_bot.set_xticks(neuron_idx)
ax_bot.legend(fontsize=8)

fig1.suptitle(f"Frame geometry analysis  (N={N}, K={K}, {N_INPUTS} inputs)",
              fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
