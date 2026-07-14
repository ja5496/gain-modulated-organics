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

def make_bump(center_idx):
    center = stim_gen.theta_inputs[center_idx]
    d = stim_gen.theta_inputs - center
    d = (d + np.pi/2) % np.pi - np.pi/2
    z = np.exp(-d**2 / (2*stim_gen.tuning_width**2))
    return stim_gen.contrast * 15 * z / np.max(z)

def simulate_bursty(p_adapt, T, alphas, L_burst_mean, seed):
    """Burst-structured stream: the 'current' stimulus persists for a
    geometrically-distributed burst length (mean L_burst_mean) before a NEW
    stimulus is drawn (adaptor w.p. p_adapt, else uniform). Within a burst the
    stimulus is fixed; only per-trial Poisson-like neural noise varies. This is
    what creates a genuine short-window (stuck-in-one-burst) vs long-window
    (spans many bursts) distinction -- i.i.d. per-trial resampling doesn't have
    this property at all."""
    rng = np.random.default_rng(seed)
    means = {a: np.zeros(M) for a in alphas}
    vars_ = {a: np.ones(M) for a in alphas}

    def draw_stimulus():
        idx = adaptor_idx if rng.random() < p_adapt else rng.integers(0, N)
        return B.T @ make_bump(idx)

    current_drive = draw_stimulus()
    burst_left = rng.geometric(1.0 / L_burst_mean)
    for t in range(T):
        if burst_left <= 0:
            current_drive = draw_stimulus()
            burst_left = rng.geometric(1.0 / L_burst_mean)
        burst_left -= 1
        noisy = current_drive + rng.normal(0.0, np.sqrt(np.maximum(current_drive, 1e-6)))
        for a in alphas:
            means[a] += a * (noisy - means[a])
            vars_[a] += a * ((noisy - means[a])**2 - vars_[a])
    return means, vars_

alphas = [0.2, 0.0005]   # fast: window~5 (< burst length); slow: window~2000 (>> burst length)
L_burst_mean = 25         # average fixation/burst length in trials
T = 400000
N_SEEDS = 4

print("Running bursty biased + control streams (this takes a bit)...")
means_bias_list, vars_bias_list, means_uni_list, vars_uni_list = [], [], [], []
for seed in range(N_SEEDS):
    mb, vb = simulate_bursty(p_adapt=1/3, T=T, alphas=alphas, L_burst_mean=L_burst_mean, seed=seed)
    mu, vu = simulate_bursty(p_adapt=0.0, T=T, alphas=alphas, L_burst_mean=L_burst_mean, seed=seed + 1000)
    means_bias_list.append(mb); vars_bias_list.append(vb)
    means_uni_list.append(mu); vars_uni_list.append(vu)
    print(f"  seed {seed} done")

means_bias = {a: np.mean([m[a] for m in means_bias_list], axis=0) for a in alphas}
vars_bias = {a: np.mean([v[a] for v in vars_bias_list], axis=0) for a in alphas}
means_uni = {a: np.mean([m[a] for m in means_uni_list], axis=0) for a in alphas}
vars_uni = {a: np.mean([v[a] for v in vars_uni_list], axis=0) for a in alphas}

# Fano-factor-style excess variance at the SLOW timescale: subtract off the
# Poisson-noise-explained contribution (~mean) to isolate genuine
# across-stimulus variance, rather than using the raw (noise-dominated) total.
slow_a = alphas[1]
excess_bias = np.maximum(vars_bias[slow_a] - means_bias[slow_a], 0.0)
excess_uni = np.maximum(vars_uni[slow_a] - means_uni[slow_a], 0.0)

adaptor_pool_idxs = [11, 12, 13]
far_pool_idxs = [23, 0, 1]

def shrink_gain(excess, Boost_max=3.0, n=2.0):
    excess = np.maximum(excess, 0.0)
    x0 = np.median(excess[excess > 1e-9]) if np.any(excess > 1e-9) else 1.0
    return Boost_max * excess**n / (x0**n + excess**n)

print(f"\n=== Gain profile, adaptor-region vs far-region (burst mean={L_burst_mean}) ===")
gain_profiles = {}
for a in alphas:
    rel_var = vars_bias[a] / np.maximum(vars_uni[a], 1e-9)
    g = 2.5 + shrink_gain(rel_var - 1.0)
    gain_profiles[a] = g
    g_adaptor = g[adaptor_pool_idxs].mean()
    g_far = g[far_pool_idxs].mean()
    window = int(round(1/a))
    print(f"alpha={a:8.4f} (window~{window:5d}), RAW variance: gain@adaptor={g_adaptor:.3f}  gain@far={g_far:.3f}  contrast={g_adaptor-g_far:+.3f}")

# NEW: slow-timescale gain from Fano-factor-style EXCESS variance instead of
# raw variance -- explicitly subtracts the Poisson-noise floor first.
rel_excess = excess_bias / np.maximum(excess_uni, 1e-9)
g_excess = 2.5 + shrink_gain(rel_excess - 1.0)
gain_profiles['excess'] = g_excess
g_adaptor_ex = g_excess[adaptor_pool_idxs].mean()
g_far_ex = g_excess[far_pool_idxs].mean()
print(f"slow, EXCESS variance (Var-Mean):        gain@adaptor={g_adaptor_ex:.3f}  gain@far={g_far_ex:.3f}  contrast={g_adaptor_ex-g_far_ex:+.3f}")

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(np.degrees(centers), gain_profiles[alphas[0]], 'o-', label=f'fast, raw variance (window~{int(1/alphas[0])})')
ax.plot(np.degrees(centers), gain_profiles[alphas[1]], 'o-', label=f'slow, raw variance (window~{int(1/alphas[1])})')
ax.plot(np.degrees(centers), gain_profiles['excess'], 'o-', label='slow, EXCESS variance (Var-Mean)')
ax.axvline(np.degrees(adaptor_rad), color='r', linestyle='--', alpha=0.5, label='adaptor location')
ax.set_xlabel('Pool center (deg)'); ax.set_ylabel('Gain')
ax.set_title(f'Bursty stream (mean burst={L_burst_mean} trials): raw vs excess variance')
ax.legend(fontsize=8)
plt.tight_layout()
fig.savefig(os.path.join(out_dir, "bursty_timescale.png"), dpi=110)
print("\nSaved bursty_timescale.png")
