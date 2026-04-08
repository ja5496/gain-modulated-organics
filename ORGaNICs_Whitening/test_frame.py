"""
test_frame.py

Tests whether unequal v² values across frame vectors are a geometric property
of the frame itself (W), independent of network dynamics.

Procedure:
1. Build a small frame (N=11, K=66) via frame_whiten.py
2. Generate 10 raised-cosine input vectors centered at different neuron indices
3. Compute v = W.T @ y for each input; record v² element-wise
4. Plot average v² (± std) vs abs(sum(w)) per frame vector (Figure 1)
5. Replicate the scatter with quadratic fits across N=5,10,20,30,40,50 (Figure 2)
"""

import numpy as np
import matplotlib.pyplot as plt
from frame_whiten import Frame

np.random.seed(20)

# ── Parameters ────────────────────────────────────────────────────────────────
N            = 50
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
        inputs.append(y)
    return inputs

def compute_vsq_stats(W, inputs):
    """Return avg_vsq (K,) and std_vsq (K,) over the input stream."""
    vsq_all = np.stack([(W.T @ y) ** 2 for y in inputs], axis=0)  # (n_inputs, K)
    return vsq_all.mean(axis=0), vsq_all.std(axis=0)

# ── Figure 1: N=11 frame with error bars ─────────────────────────────────────
frame = Frame(dim=N)
W = frame.W          # (N, K)
K = frame.K

inputs = build_inputs(N, N_INPUTS)
avg_vsq, std_vsq = compute_vsq_stats(W, inputs)
sum_w = np.abs(W.sum(axis=0))   # (K,)
sort_idx = np.argsort(avg_vsq)[::-1]

fig1, (ax_bar, ax_scatter, ax_ratio) = plt.subplots(1, 3, figsize=(18, 5))

# Left: bar chart sorted descending
ax_bar.bar(np.arange(K), avg_vsq[sort_idx], color='steelblue', edgecolor='none')
ax_bar.axhline(avg_vsq.mean(), color='red', linestyle='--', linewidth=1.5,
               label=f'mean = {avg_vsq.mean():.4f}')
ax_bar.set_xlabel("Frame vector rank (sorted by avg v²)", fontsize=12)
ax_bar.set_ylabel("Average v²", fontsize=12)
ax_bar.set_title("Avg v² per frame vector\n(sorted descending)", fontsize=13, fontweight='bold')
ax_bar.legend()

# Middle: scatter avg v² ± std vs abs(sum(w)) with quadratic fit
ax_scatter.errorbar(sum_w, avg_vsq, yerr=std_vsq,
                    fmt='o', color='steelblue', alpha=0.7,
                    ecolor='steelblue', elinewidth=1, capsize=3,
                    markersize=5, markeredgewidth=0)
quad_coeffs = np.polyfit(sum_w, avg_vsq, 2)
x_fit1 = np.linspace(sum_w.min(), sum_w.max(), 300)
ax_scatter.plot(x_fit1, np.polyval(quad_coeffs, x_fit1), color='red', linewidth=1.5,
                label=f'quadratic: {quad_coeffs[0]:.3f}x² + {quad_coeffs[1]:.3f}x + {quad_coeffs[2]:.3f}')
ax_scatter.legend(fontsize=9)
ax_scatter.set_xlabel("|Sum of frame vector components|", fontsize=12)
ax_scatter.set_ylabel("Average v²", fontsize=12)
ax_scatter.set_title("Avg v² (± std) vs |Sum(w)|", fontsize=13, fontweight='bold')

# Right: normalized ratio plot
ratio = avg_vsq / (0.4 * (sum_w**2) + 0.05)
ax_ratio.scatter(sum_w, ratio, color='steelblue', alpha=0.7, edgecolors='none', s=40)
ax_ratio.axhline(1, color='black', linestyle=':', linewidth=1.5)
ax_ratio.set_xlabel("|Sum(w)|", fontsize=12)
ax_ratio.set_ylabel("avg v² / (0.4 · (Sum(w)² + 0.05))", fontsize=12)
ax_ratio.set_title("Normalized ratio vs |Sum(w)|", fontsize=13, fontweight='bold')

fig1.suptitle(f"Frame geometry analysis  (N={N}, K={K}, {N_INPUTS} inputs)",
              fontsize=14, fontweight='bold')
plt.tight_layout()

# ── Figure 2: N = 5, 10, 20, 30, 40, 50 comparison ──────────────────────────
fig2, axes2 = plt.subplots(2, 3, figsize=(16, 10))

for ax, n in zip(axes2.flat, [5, 10, 20, 30, 40, 50]):
    f = Frame(dim=n)
    w = f.W
    k = f.K
    n_inp = min(n, 10)
    inp = build_inputs(n, n_inp)
    avg, _ = compute_vsq_stats(w, inp)
    sw = np.abs(w.sum(axis=0))

    ax.scatter(sw, avg, color='steelblue', alpha=0.7, edgecolors='none', s=40)

    coeffs = np.polyfit(sw, avg, 2)
    x_fit = np.linspace(sw.min(), sw.max(), 300)
    ax.plot(x_fit, np.polyval(coeffs, x_fit), color='red', linewidth=1.5,
            label=f'{coeffs[0]:.3f}x² + {coeffs[1]:.3f}x + {coeffs[2]:.3f}')
    ax.legend(fontsize=9)

    ax.set_xlabel("|Sum(w)|", fontsize=12)
    ax.set_ylabel("Average v²", fontsize=12)
    ax.set_title(f"N={n}, K={k}", fontsize=13, fontweight='bold')

fig2.suptitle("Avg v² vs |Sum(w)| across frame sizes",
              fontsize=14, fontweight='bold')
plt.tight_layout()

plt.show()
