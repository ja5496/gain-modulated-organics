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

# --- Fixed developmental baseline: Poisson RATE (== mean == variance) per pool
# under generic/uniform viewing ---
lambda_baseline = (stimuli_uni @ B).mean(axis=0)

# --- Sustained single-adaptor drive (deterministic, so its OWN across-trial
# variance is zero -- the whole point is that trial-to-trial variability has to
# come from neural response noise, not stimulus variability, once the stimulus
# is literally constant) ---
z_adaptor = np.exp(-(((theta - adaptor_rad + np.pi/2) % np.pi - np.pi/2)**2) / (2*stim_gen.tuning_width**2))
z_adaptor = stim_gen.contrast * 15 * z_adaptor / np.max(z_adaptor)
adaptation_drive = B.T @ z_adaptor
lambda_current = lambda_baseline + adaptation_drive   # ongoing baseline + sustained adaptor drive

def build_gains(var_baseline, var_current, Boost_max=3.0, n=2.0):
    rel_var = var_current / var_baseline           # == 1 where adaptor doesn't reach; >1 where it does
    excess = np.maximum(rel_var - 1.0, 0.0)         # shrink-only: never negative
    x0 = np.median(excess[excess > 1e-9]) if np.any(excess > 1e-9) else 1.0
    boost = Boost_max * excess**n / (x0**n + excess**n)
    g_baseline_rule = 2.5 * np.ones(M)              # flat baseline gain (matches earlier prototype's ~2.5)
    return g_baseline_rule, g_baseline_rule + boost, rel_var

# Poisson noise: Var == mean (unit Fano factor)
g_uni_poisson, g_bias_poisson, relvar_poisson = build_gains(lambda_baseline, lambda_current)

# Constant-CV / superlinear noise: Var ~ mean^2 (steeper contrast-like scaling)
g_uni_cv, g_bias_cv, relvar_cv = build_gains(lambda_baseline**2, lambda_current**2)

print("=== Relative variance per pool (Poisson: Var=mean) ===")
print("at adaptor pool (idx 12):", relvar_poisson[12], " at far pool (idx 0):", relvar_poisson[0])
print("=== Relative variance per pool (constant-CV: Var=mean^2) ===")
print("at adaptor pool (idx 12):", relvar_cv[12], " at far pool (idx 0):", relvar_cv[0])

def gaussian_probe(center, width, meansub=False):
    d = theta - center
    d = (d + np.pi / 2) % np.pi - np.pi / 2
    p = np.exp(-d**2 / (2 * width**2))
    return p - p.mean() if meansub else p

def suppression_curve(g_uni, g_bias):
    M_uni = np.eye(N) + (B * g_uni) @ B.T
    M_bias = np.eye(N) + (B * g_bias) @ B.T
    offsets_deg = np.linspace(-90, 90, 37)
    resp_uni, resp_bias = [], []
    for off_deg in offsets_deg:
        center = adaptor_rad + np.radians(off_deg)
        probe = gaussian_probe(center, stim_gen.tuning_width)
        y_uni = np.linalg.solve(M_uni, probe)
        y_bias = np.linalg.solve(M_bias, probe)
        resp_uni.append((probe @ y_uni) / (probe @ probe))
        resp_bias.append((probe @ y_bias) / (probe @ probe))
    resp_uni, resp_bias = np.array(resp_uni), np.array(resp_bias)
    return offsets_deg, 100 * (1 - resp_bias / resp_uni)

offsets_deg, supp_poisson = suppression_curve(g_uni_poisson, g_bias_poisson)
_, supp_cv = suppression_curve(g_uni_cv, g_bias_cv)

print("\n=== Suppression curve (Poisson variance) ===")
print("at adaptor:", supp_poisson[np.argmin(np.abs(offsets_deg))], " at 90 deg:", supp_poisson[np.argmin(np.abs(offsets_deg-90))])
print("=== Suppression curve (constant-CV variance) ===")
print("at adaptor:", supp_cv[np.argmin(np.abs(offsets_deg))], " at 90 deg:", supp_cv[np.argmin(np.abs(offsets_deg-90))])

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(offsets_deg, supp_poisson, label='Poisson noise (Var = mean)')
ax.plot(offsets_deg, supp_cv, label='constant-CV noise (Var = mean^2)')
ax.axhline(0, color='gray', linewidth=0.8)
ax.set_xlabel('Test orientation - adaptor (deg)')
ax.set_ylabel('% suppression relative to control')
ax.set_title('Adaptation tuning curve: variance-based (two noise models)')
ax.legend()
plt.tight_layout()
fig.savefig(os.path.join(out_dir, "relative_variance_suppression.png"), dpi=110)
print("\nSaved relative_variance_suppression.png")
