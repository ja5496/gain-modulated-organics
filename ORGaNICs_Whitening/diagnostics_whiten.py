"""
diagnostics_whiten.py
---------------------
Three standalone diagnostics for exploring model behavior.

1. Scale sweep      – how probe tuning curves change with the `scale` parameter
                      (raw, unnormalized — shows gain and shape together)
2. a = 0 probe      – run adaptive sims on uniform/biased; probe with a forced to 0
                      so only gain adaptation is active; show 2 normalized plots
3. Contrast response function – peak response vs. contrast (sigmoidal, σ = half-max contrast)

Run from the ORGaNICs_Whitening/ directory:
    python diagnostics_whiten.py
"""

import gc
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
import time

from tunings_whiten import V1Tunings
from stimuli_whiten import StimulusGenerator
from simulation_whiten import V1Dynamics, Frame


# ── Shared parameters ──────────────────────────────────────────────────────────
N           = 169
STREAM_LEN  = 10920   # num_inputs=546, one_third_split=182≈1 full cycle; num_angles=169 divides evenly across N_BINS=13
PROBE_RES   = 90
PROBE_STEPS = 100
N_BINS      = 13
Z_SPONT     = 0.1
AVG_WINDOW  = 2000

np.random.seed(20)


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _make_canonical_frame():
    """Load N169 frame and canonicalize so W @ W.T == (K/N)*I."""
    frame = Frame(csv_path="Frames/N169_Frame.csv")
    S = frame.W @ frame.W.T
    eigvals, eigvecs = np.linalg.eigh(S)
    S_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    N_neu, K_neu = frame.W.shape
    frame.W = np.sqrt(K_neu / N_neu) * (S_inv_sqrt @ frame.W)
    return frame


def gaussian_rectify(y, threshold=0.5, sigma=0.2, r_max=1.0):
    return 0.5 * (1 + erf((y - threshold) / (sigma * np.sqrt(2)))) * r_max


def run_probe(frame, tunings, fixed_gains, probe_angles,
              frozen_u=None, frozen_a=None, z_spont=Z_SPONT,
              scale=1.0, force_a_zero=False):
    """
    Probe tuning curves with optional overrides:
      scale        – Michelson contrast c ∈ [0, 1]. This is converted into an LGN 
                     drive using a Naka-Rushton saturation function.
      force_a_zero – sets a=0 throughout (disables normalization)
    """
    N_neu = frame.dim
    n_probes = len(probe_angles)
    tuning_curves = np.zeros((N_neu, n_probes))
    W_yy = tunings.W_yy

    dt = 0.05; tau_y = 1.0; tau_u = 2.0; tau_a = 5.0
    beta = 1.0; sigma_c = 0.05; tuning_width = 0.5

    # --- Naka-Rushton LGN Input Mapping ---
    R_max_lgn = 2.5
    c_50_lgn = 0.2
    n_exp = 2.0
    # Convert linear contrast (scale) into a saturated biological drive
    contrast_drive = R_max_lgn * (scale**n_exp) / (scale**n_exp + c_50_lgn**n_exp)

    for i, angle in enumerate(probe_angles):
        y = z_spont * np.ones(N_neu)
        u = np.copy(frozen_u) if frozen_u is not None else np.zeros(N_neu)
        a = np.zeros(N_neu) if force_a_zero else (
            np.copy(frozen_a) if frozen_a is not None else np.zeros(N_neu))

        z_t = np.exp(tuning_width * np.cos(2 * (tunings.theta - angle)))
        # Apply the saturated contrast drive here instead of raw scale
        z_t = (z_t / np.max(z_t)) * contrast_drive  

        for _ in range(PROBE_STEPS):
            u_plus      = gaussian_rectify(u)
            y_plus      = gaussian_rectify(y)
            a_plus      = np.zeros(N_neu) if force_a_zero else gaussian_rectify(a)
            sqrt_y_plus = np.sqrt(y_plus)

            v_t           = frame.W.T @ y
            gain_feedback = (frame.W @ (fixed_gains * v_t)) if fixed_gains is not None else 0.0
            recurrent_drive = (1.0 / (1.0 + a_plus)) * (W_yy @ sqrt_y_plus)
            input_drive     = (beta * z_t) / 2 + z_spont
            pool_term       = tunings.N_matrix @ (y_plus * (u_plus ** 2))

            dy = (-y + input_drive + recurrent_drive - gain_feedback) / tau_y
            du = (-u + (sigma_c**2) + pool_term) / tau_u
            da = np.zeros_like(a) if force_a_zero else (-a + u_plus + a * u_plus) / tau_a

            y += dt * dy
            u += dt * du
            a += dt * da

        tuning_curves[:, i] = gaussian_rectify(y)

    return tuning_curves


def minmax_norm(binned, ref_binned=None):
    """Per-bin min-max normalization.
    ref_binned: reference condition (e.g. uniform). If None, each bin is its own reference.
    """
    ref = binned if ref_binned is None else ref_binned
    bin_max = np.max(ref, axis=1, keepdims=True)
    bin_min = np.min(ref, axis=1, keepdims=True)
    return (binned - bin_min) / (bin_max - bin_min + 1e-9)


def get_binned_curves(tuning_curves, neuron_preferences, probe_angles, n_bins=N_BINS):
    discrete_step = np.pi / len(neuron_preferences)
    bin_edges     = np.linspace(0, np.pi, n_bins + 1) - (discrete_step / 2)
    binned        = np.zeros((n_bins, len(probe_angles)))
    idx           = np.clip(np.digitize(neuron_preferences, bin_edges) - 1, 0, n_bins - 1)
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            binned[b] = np.mean(tuning_curves[mask], axis=0)
    return binned


def get_binned_activity(rates, neuron_bin_idx, window=AVG_WINDOW, n_bins=N_BINS):
    """Average steady-state firing per orientation bin."""
    duration = 20; keep = 5
    steady   = rates[:, -window:]
    n_t      = steady.shape[1]
    time_mask = (np.arange(n_t) % duration) >= (duration - keep)
    means    = np.mean(steady[:, time_mask], axis=1)
    binned   = np.zeros(n_bins)
    for b in range(n_bins):
        mask = neuron_bin_idx == b
        if mask.any():
            binned[b] = np.mean(means[mask])
    return binned


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── Shared initialisation ─────────────────────────────────────────────────
    print("Initializing shared resources...")
    tunings   = V1Tunings(N=N)
    frame     = _make_canonical_frame()
    stim_gen  = StimulusGenerator(N=N, num_angles=N, stream_length=STREAM_LEN)

    adaptor_idx = N // 2
    adaptor_rad = stim_gen.theta_inputs[adaptor_idx]
    adaptor_deg = adaptor_rad * 180 / np.pi

    # Bin geometry shared across all diagnostics
    discrete_step_rad = np.pi / N
    bin_edges_rad     = np.linspace(0, np.pi, N_BINS + 1) - (discrete_step_rad / 2)
    bin_centers       = (bin_edges_rad[:-1] + bin_edges_rad[1:]) / 2
    bin_centers_deg   = bin_centers * 180 / np.pi
    neuron_bin_idx    = np.clip(np.digitize(tunings.theta, bin_edges_rad) - 1, 0, N_BINS - 1)

    seq_uni  = stim_gen.generate_input_ensembles(biased=False)
    seq_bias = stim_gen.generate_input_ensembles(biased=True)

    probe_angles     = np.linspace(0, np.pi, PROBE_RES)
    probe_angles_deg = probe_angles * 180 / np.pi

    blue_colors = plt.cm.Blues(np.linspace(0.4, 1.0, N_BINS))

    # x-axis relative to adaptor (for probe curves)
    x_axis      = (probe_angles_deg - adaptor_deg + 90) % 180 - 90
    sort_idx    = np.argsort(x_axis)
    x_sorted    = x_axis[sort_idx]

    # x-axis relative to adaptor (for binned-activity curves)
    x_peak      = (bin_centers_deg - adaptor_deg + 90) % 180 - 90
    sort_idx_2  = np.argsort(x_peak)
    x_peak_s    = x_peak[sort_idx_2]


    # ══════════════════════════════════════════════════════════════════════════
    # DIAGNOSTIC 1: Scale sweep
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("DIAGNOSTIC 1: Probe tuning curves vs. input contrast")
    print("=" * 60)

    scales       = [0.1, 0.25, 0.5, 0.75, 1.0]
    scale_colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(scales)))
    mid_bin      = N_BINS // 2

    scale_binned = {}
    for scale in scales:
        tc = run_probe(frame, tunings, fixed_gains=None, probe_angles=probe_angles,
                       scale=scale)
        scale_binned[scale] = get_binned_curves(tc, tunings.theta, probe_angles)

    fig1, ax1 = plt.subplots(1, 1, figsize=(7, 5))
    for si, scale in enumerate(scales):
        ax1.plot(x_sorted, scale_binned[scale][mid_bin][sort_idx],
                 color=scale_colors[si], linewidth=2, label=f'contrast = {scale}')
    ax1.set_title(f"Tuning Curve vs Contrast",
                  fontweight='bold')
    ax1.set_xlabel("Orientation Relative to Stimulus (°)")
    ax1.set_ylabel("Response (raw)")
    ax1.set_xlim(-90, 90)
    ax1.legend(fontsize='small')
    ax1.grid(False)
    fig1.suptitle("Diagnostic 1: Effect of Input Scale on Tuning Curves",
                  fontweight='bold', fontsize=13)
    plt.tight_layout()
    plt.show()


    # ══════════════════════════════════════════════════════════════════════════
    # DIAGNOSTIC 2: Gain adaptation only (a = 0 during probing)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("DIAGNOSTIC 2: Probe with a = 0 (gain adaptation only)")
    print("=" * 60)

    print("Running adaptive simulation (uniform)...")
    engine_uni2 = V1Dynamics(tunings, frame, adaptive=True)
    rates_uni2, gains_uni2, u_uni2, a_uni2 = engine_uni2.run_simulation(seq_uni)
    final_gains_uni2 = gains_uni2[:, -1]
    final_u_uni2     = u_uni2[:, -1]
    del rates_uni2, gains_uni2, u_uni2, a_uni2; gc.collect()

    print("Running adaptive simulation (biased)...")
    engine_bias2 = V1Dynamics(tunings, frame, adaptive=True)
    rates_bias2, gains_bias2, u_bias2, a_bias2 = engine_bias2.run_simulation(seq_bias)
    final_gains_bias2 = gains_bias2[:, -1]
    final_u_bias2     = u_bias2[:, -1]
    del rates_bias2, gains_bias2, u_bias2, a_bias2; gc.collect()

    print("Probing both conditions with a = 0...")
    tc_uni_a0  = run_probe(frame, tunings, final_gains_uni2, probe_angles,
                           frozen_u=final_u_uni2, force_a_zero=True)
    tc_bias_a0 = run_probe(frame, tunings, final_gains_bias2, probe_angles,
                           frozen_u=final_u_bias2, force_a_zero=True)

    b_uni_a0  = get_binned_curves(tc_uni_a0,  tunings.theta, probe_angles)
    b_bias_a0 = get_binned_curves(tc_bias_a0, tunings.theta, probe_angles)

    n_uni_a0  = minmax_norm(b_uni_a0,  ref_binned=b_uni_a0)
    n_bias_a0 = minmax_norm(b_bias_a0, ref_binned=b_uni_a0)

    fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5), sharey=True,
                               gridspec_kw={'wspace': 0.1})
    for ax, title, data in zip(
        axes2,
        ["Uniform Ensemble (a = 0)", "Biased Ensemble (a = 0)"],
        [n_uni_a0, n_bias_a0],
    ):
        for i in range(N_BINS):
            ax.plot(x_sorted, data[i][sort_idx], color=blue_colors[i], linewidth=1.5)
        ax.set_title(title, fontweight='bold')
        ax.set_xlim(-90, 90)
        ax.set_xlabel("Orientation Relative to Adaptor (°)")
        ax.grid(False)
    axes2[0].set_ylabel("Normalized Response")
    fig2.suptitle("Diagnostic 2: Gain Adaptation Only (a = 0 during probe)",
                  fontweight='bold', fontsize=13)
    plt.tight_layout()
    plt.show()

