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

# --- SHORT-timescale signal: mean drive (== Poisson-noise variance), dominated
# by the sustained adaptor's direct, deterministic drive -- peaks AT the adaptor.
lambda_baseline = (stimuli_uni @ B).mean(axis=0)
z_adaptor = np.exp(-(((theta - adaptor_rad + np.pi/2) % np.pi - np.pi/2)**2) / (2*stim_gen.tuning_width**2))
z_adaptor = stim_gen.contrast * 15 * z_adaptor / np.max(z_adaptor)
adaptation_drive = B.T @ z_adaptor

def shrink_gain(excess, Boost_max=3.0, n=2.0):
    excess = np.maximum(excess, 0.0)
    x0 = np.median(excess[excess > 1e-9]) if np.any(excess > 1e-9) else 1.0
    return Boost_max * excess**n / (x0**n + excess**n)

g_uni = 2.5 * np.ones(M)
g_short = g_uni + shrink_gain(adaptation_drive / lambda_baseline)   # relative Poisson-variance excess

# --- LONG-timescale signal: cross-stimulus ensemble variance (the "true",
# noise-averaged-out statistic) -- this is what dipped at the adaptor and rose at
# the flanks in the earlier local-pool-vs-covariance-target analysis. Compute it
# directly and plainly here: per-pool variance of pooled drive across the actual
# stimulus ensembles, biased vs uniform.
var_uni = (stimuli_uni @ B).var(axis=0)
var_bias = (stimuli_bias @ B).var(axis=0)
g_long = g_uni + shrink_gain(var_bias / var_uni - 1.0)

print("=== Gain profiles at adaptor (idx 12) vs far (idx 0) ===")
print(f"short-timescale (mean-driven):     adaptor={g_short[12]:.3f}  far={g_short[0]:.3f}")
print(f"long-timescale (variance-driven):  adaptor={g_long[12]:.3f}  far={g_long[0]:.3f}")

M_uni = np.eye(N) + (B * g_uni) @ B.T
M_short = np.eye(N) + (B * g_short) @ B.T
M_long = np.eye(N) + (B * g_long) @ B.T

def gaussian_probe(center, width=None):
    width = width or stim_gen.tuning_width
    d = theta - center
    d = (d + np.pi/2) % np.pi - np.pi/2
    return np.exp(-d**2 / (2*width**2))

def decode_circular_mean(y):
    # doubled-angle population vector (period pi domain) -> (decoded angle, resultant length)
    z = np.sum(np.maximum(y, 0) * np.exp(2j*theta))
    denom = np.sum(np.maximum(y, 0))
    angle = 0.5 * np.angle(z)
    length = np.abs(z) / denom if denom > 0 else 0.0
    return angle, length

def wrap_pi(x):
    return (x + np.pi/2) % np.pi - np.pi/2

# --- Shift (attraction/repulsion) as a function of test offset from adaptor ---
offsets_deg = np.linspace(-80, 80, 33)
shift_short, shift_long = [], []
width_short, width_long = [], []
for off_deg in offsets_deg:
    center = adaptor_rad + np.radians(off_deg)
    probe = gaussian_probe(center)
    y_uni = np.linalg.solve(M_uni, probe)
    y_short = np.linalg.solve(M_short, probe)
    y_long = np.linalg.solve(M_long, probe)

    ang_uni, len_uni = decode_circular_mean(y_uni)
    ang_short, len_short = decode_circular_mean(y_short)
    ang_long, len_long = decode_circular_mean(y_long)

    shift_short.append(np.degrees(wrap_pi(ang_short - ang_uni)))
    shift_long.append(np.degrees(wrap_pi(ang_long - ang_uni)))
    width_short.append(len_short / len_uni)   # >1 == sharper/narrower than control
    width_long.append(len_long / len_uni)

shift_short, shift_long = np.array(shift_short), np.array(shift_long)
width_short, width_long = np.array(width_short), np.array(width_long)

print("\n=== Shift sign check (positive test offset -> positive shift = repulsion, negative = attraction) ===")
for lbl, arr in (('short', shift_short), ('long', shift_long)):
    near_plus30 = arr[np.argmin(np.abs(offsets_deg - 30))]
    print(f"{lbl}-timescale: shift at +30 deg test offset = {near_plus30:+.2f} deg")

print("\n=== Sharpness (resultant length ratio) at the adaptor's own location ===")
idx0 = np.argmin(np.abs(offsets_deg))
print(f"short-timescale sharpness ratio at adaptor: {width_short[idx0]:.4f}")
print(f"long-timescale  sharpness ratio at adaptor: {width_long[idx0]:.4f}")

fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 4.5))
axA.plot(offsets_deg, shift_short, label='short-timescale (mean-driven)')
axA.plot(offsets_deg, shift_long, label='long-timescale (variance-driven)')
axA.axhline(0, color='gray', linewidth=0.8)
axA.plot(offsets_deg, offsets_deg*0, 'k:', alpha=0.3)
axA.set_xlabel('Test orientation - adaptor (deg)')
axA.set_ylabel('Decoded shift (deg): + = repulsion, - = attraction')
axA.set_title('Tuning-curve shift')
axA.legend()

axB.plot(offsets_deg, width_short, label='short-timescale')
axB.plot(offsets_deg, width_long, label='long-timescale')
axB.axhline(1, color='gray', linewidth=0.8)
axB.set_xlabel('Test orientation - adaptor (deg)')
axB.set_ylabel('Sharpness ratio (adapted / control)')
axB.set_title('Tuning-curve sharpening (>1 = narrower)')
axB.legend()
plt.tight_layout()
fig.savefig(os.path.join(out_dir, "timescale_consistency.png"), dpi=110)
print("\nSaved timescale_consistency.png")
