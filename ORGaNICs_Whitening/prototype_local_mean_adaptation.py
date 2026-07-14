import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sys, os

sys.path.insert(0, "/Users/jakeabraham/Mind Lab/gain-modulated-organics/ORGaNICs_Whitening")
from tunings_whiten import V1Tunings
from stimuli_whiten import StimulusGenerator

N = 169
tunings = V1Tunings(N=N)
theta = tunings.theta
out_dir = "/private/tmp/claude-501/-Users-jakeabraham-Mind-Lab-gain-modulated-organics-ORGaNICs-Whitening/bcb94f6a-c6be-41b5-9487-b09030c903b3/scratchpad"

stim_gen = StimulusGenerator(N=N, num_angles=N, stream_length=N, contrast=0.05)
adaptor_idx = N // 2
adaptor_rad = stim_gen.theta_inputs[adaptor_idx]
n_non_adaptor = N - 1
n_adaptor_reps = n_non_adaptor // 2
non_adaptor_thetas = np.concatenate([stim_gen.theta_inputs[:adaptor_idx], stim_gen.theta_inputs[adaptor_idx + 1:]])
centers_bias = np.concatenate([non_adaptor_thetas, np.full(n_adaptor_reps, adaptor_rad)])
np.random.seed(0)
np.random.shuffle(centers_bias)
delta = stim_gen.theta_inputs[:, None] - centers_bias[None, :]
delta = (delta + np.pi / 2) % np.pi - np.pi / 2
seq_bias = np.exp(-delta**2 / (2 * stim_gen.tuning_width**2))
seq_bias = stim_gen.contrast * 15 * seq_bias / np.max(seq_bias)
stimuli_bias = seq_bias.T

seq_uni, _ = stim_gen.generate_input_ensembles(biased=False, return_angles=True, duration=1)
stimuli_uni = seq_uni.T

def local_pool_basis(M):
    centers = np.arange(M) * np.pi / M
    spacing = np.pi / M
    half_width = spacing
    B = np.zeros((N, M))
    for k, c in enumerate(centers):
        d = theta - c
        d = (d + np.pi / 2) % np.pi - np.pi / 2
        w = np.zeros(N)
        mask = np.abs(d) < half_width
        w[mask] = np.cos((np.pi / 2) * d[mask] / half_width)
        B[:, k] = w
    return B, centers

M = 24
B, centers = local_pool_basis(M)

# --- mean-driven gain rule (no matrix fitting at all) ---
mean_drive_uni = (stimuli_uni @ B).mean(axis=0)
mean_drive_bias = (stimuli_bias @ B).mean(axis=0)

sigma_g = np.median(mean_drive_uni)   # baseline = typical drive under "no adaptation"
G_max, n = 5.0, 2.0
def gain_rule(m):
    return G_max * m**n / (sigma_g**n + m**n)

g_uni = gain_rule(mean_drive_uni)

# --- corrected "sustained single-adaptor" condition ---
# The mixture-ensemble comparison above is the wrong paradigm: it makes the
# adaptor compete with 168 other orientations for a fixed trial budget, so
# every OTHER orientation's mean drive drops just because it got fewer trials
# -- an artifact of shared budget, not biology. Real sustained adaptation shows
# ONE stimulus repeatedly; every pool that stimulus doesn't drive should see
# NO change at all, not a competing-ensemble-induced decrease.
z_adaptor = np.exp(-(((theta - adaptor_rad + np.pi/2) % np.pi - np.pi/2)**2) / (2*stim_gen.tuning_width**2))
z_adaptor = stim_gen.contrast * 15 * z_adaptor / np.max(z_adaptor)
adaptation_drive = B.T @ z_adaptor   # each pool's direct drive from the SUSTAINED adaptor alone

Boost_max, x0 = 3.0, np.median(adaptation_drive)
adaptation_boost = Boost_max * adaptation_drive**n / (x0**n + adaptation_drive**n)
g_bias = g_uni + adaptation_boost   # baseline (unchanged where adaptor doesn't reach) + pure nonneg boost

M_uni = np.eye(N) + (B * g_uni) @ B.T
M_bias = np.eye(N) + (B * g_bias) @ B.T

print("=== Gain profile ===")
print("uniform gains: min/max/std:", g_uni.min(), g_uni.max(), g_uni.std())
print("biased  gains: min/max/std:", g_bias.min(), g_bias.max(), g_bias.std())
print("biased gain at adaptor pool (idx 12):", g_bias[12], " at far pool (idx 0):", g_bias[0])

# --- tightness (unchanged, geometry didn't change) ---
sq_sum = np.sum(B**2, axis=1)
print("\ntightness sum_k w_k^2: min/max/std:", sq_sum.min(), sq_sum.max(), sq_sum.std())

# --- locality test: narrow off-adaptor probe (biased circuit) ---
def gaussian_probe(center, width, meansub=True):
    d = theta - center
    d = (d + np.pi / 2) % np.pi - np.pi / 2
    p = np.exp(-d**2 / (2 * width**2))
    return p - p.mean() if meansub else p

test_probe = gaussian_probe(adaptor_rad + 0.6, 0.1, meansub=False)
feedback_bias = (B * g_bias) @ B.T @ test_probe
print("\n=== Locality check (mean-driven gains) ===")
print("feedback far from probe (idx 0-10):", np.round(feedback_bias[0:10], 4))
print("feedback near probe (idx 108-118):", np.round(feedback_bias[108:118], 4))

# --- Gain-vs-location plot ---
fig1, ax1 = plt.subplots(figsize=(7, 4))
ax1.plot(np.degrees(centers), g_uni, 'o-', label='uniform (control)')
ax1.plot(np.degrees(centers), g_bias, 'o-', label='biased (adapted)')
ax1.axvline(np.degrees(adaptor_rad), color='r', linestyle='--', label='adaptor location')
ax1.set_xlabel('Pool center (deg)'); ax1.set_ylabel('Gain (mean-drive rule)')
ax1.legend()
plt.tight_layout()
fig1.savefig(os.path.join(out_dir, "meandriven_gains.png"), dpi=110)

# --- Suppression tuning curve: response at test orientation's own neuron, ---
# --- biased (adapted) vs uniform (control), as function of offset from adaptor ---
offsets_deg = np.linspace(-90, 90, 37)
resp_uni, resp_bias = [], []
for off_deg in offsets_deg:
    center = adaptor_rad + np.radians(off_deg)
    # Plain nonnegative bump (an actual stimulus, not mean-subtracted -- that
    # convention was only needed for the earlier exact-whitening comparisons).
    probe = gaussian_probe(center, stim_gen.tuning_width, meansub=False)
    y_uni = np.linalg.solve(M_uni, probe)
    y_bias = np.linalg.solve(M_bias, probe)
    # Matched-filter readout (project response back onto the probe's own shape)
    # instead of a single neuron's value -- robust to the pool-boundary ripple
    # in a 24-pool piecewise-local basis, analogous to reading out a small
    # population rather than one noiseless unit.
    resp_uni.append((probe @ y_uni) / (probe @ probe))
    resp_bias.append((probe @ y_bias) / (probe @ probe))

resp_uni = np.array(resp_uni)
resp_bias = np.array(resp_bias)
suppression_pct = 100 * (1 - resp_bias / resp_uni)

fig2, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4))
axA.plot(offsets_deg, resp_uni, label='uniform (control)')
axA.plot(offsets_deg, resp_bias, label='biased (adapted)')
axA.set_xlabel('Test orientation - adaptor (deg)')
axA.set_ylabel('Matched-filter response amplitude')
axA.legend(); axA.set_title('Raw response')

axB.plot(offsets_deg, suppression_pct, color='k')
axB.axhline(0, color='gray', linewidth=0.8)
axB.set_xlabel('Test orientation - adaptor (deg)')
axB.set_ylabel('% suppression relative to control')
axB.set_title('Adaptation tuning curve')
plt.tight_layout()
fig2.savefig(os.path.join(out_dir, "suppression_tuning_curve.png"), dpi=110)

print("\nSaved: meandriven_gains.png, suppression_tuning_curve.png")
print("suppression % at offset 0 (at adaptor):", suppression_pct[np.argmin(np.abs(offsets_deg))])
print("suppression % at offset ~90 (orthogonal):", suppression_pct[np.argmin(np.abs(offsets_deg-90))])
