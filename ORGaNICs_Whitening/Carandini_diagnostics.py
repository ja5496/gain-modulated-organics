"""
Carandini_diagnostics.py

Diagnostic plots for the Carandini/Benucci adaptation experiment.

Diagnostic 1 — Adaptation Convergence:
    Shows how the adaptive + biased tuning curves evolve as the adaptation
    stream length increases (100, 1000, 3000, 6000 steps).

Diagnostic 2 — Sigma Sweep:
    Shows how the adaptive + biased tuning curves change when varying the
    inhibitory spread (sigma_inh) of the recurrent weight matrix W_yy,
    while keeping excitatory spread (sigma_exc) fixed at 0.2.

Diagnostic 3 — Tuning Width Sweep:
    Replicates the first plot in Carandini_plots.py (adaptive + biased tuning
    curves) for raised-cosine stimulus tuning widths of 1 through 6, arranged
    in a 2 × 3 grid.

Diagnostic 4 — White Noise Probe:
    Probes the adaptive model with broadband white noise added at every step.
"""

import numpy as np
import matplotlib.pyplot as plt
import gc

from tunings_whiten import V1Tunings
from stimuli_whiten import StimulusGenerator
from simulation_whiten import Frame, V1Dynamics
from Carandini_plots import run_probe, get_binned_curves, gaussian_rectify

# ---- Shared Parameters ----
N = 169
N_BINS = 13
PROBE_RES = 36

np.random.seed(999)

# Load frame once (expensive, shared across diagnostics)
frame = Frame(csv_path="Frames/N169_Frame.csv")

# Probe setup
probe_angles = np.linspace(0, np.pi, PROBE_RES)
probe_angles_deg = probe_angles * 180 / np.pi

# Adaptor orientation
stim_gen_ref = StimulusGenerator(N=N, K=N, stream_length=1)
adaptor_idx = N // 2
adaptor_rad = stim_gen_ref.theta_inputs[adaptor_idx]
adaptor_deg = adaptor_rad * 180 / np.pi
x_axis = probe_angles_deg - adaptor_deg

# Colors
blue_colors = plt.cm.Blues(np.linspace(0.4, 1.0, N_BINS))


def plot_tuning_panel(ax, binned_norm, title=None):
    """Draw one 'bottom-right style' subplot: 13 binned tuning curves."""
    for i in range(N_BINS):
        ax.plot(x_axis, binned_norm[i], color=blue_colors[i], linewidth=1.5)
    ax.axvline(0, color='red', linestyle='--', alpha=0.5)
    ax.set_xlim(-90, 90)
    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title, fontweight='bold')


def normalize_with_reference(binned_bias, ref_max, ref_min):
    """Normalize biased curves using the uniform reference scale."""
    return (binned_bias - ref_min) / (ref_max - ref_min + 1e-9)


def run_probe_noisy(frame, tunings, fixed_gains, probe_angles,
                    noise_level=0.2, probe_steps=100):
    """Like run_probe but adds independent Gaussian white noise to the input
    at every integration step, simulating broadband orientation noise."""
    N_loc = frame.dim
    n_probes = len(probe_angles)
    tuning_curves = np.zeros((N_loc, n_probes))

    W_yy = tunings.W_yy
    dt = 0.05
    tau_y, tau_u, tau_a = 1.0, 2.0, 5.0
    beta = 1.0
    sigma_const = 0.05

    y = np.zeros(N_loc)
    u = np.zeros(N_loc)
    a = np.zeros(N_loc)

    for i, angle in enumerate(probe_angles):
        # Clean signal component (same as run_probe)
        diff = np.abs(tunings.theta - angle)
        diff = np.minimum(diff, 2 * np.pi - diff)
        z_signal = np.exp(-(diff ** 2) / (2 * (np.pi / 8) ** 2))

        for _ in range(probe_steps):
            # Fresh white noise at every step, rectified to keep inputs >= 0
            noise = np.random.normal(0, noise_level, N_loc)
            z_t = np.maximum(z_signal + noise, 0)

            u_plus = gaussian_rectify(u)
            y_plus = gaussian_rectify(y)
            a_plus = gaussian_rectify(a)
            sqrt_y_plus = np.sqrt(y_plus)

            v_t = frame.W.T @ y
            if fixed_gains is not None:
                gain_feedback = frame.W @ (fixed_gains * v_t)
            else:
                gain_feedback = 0.0

            recurrent_drive = (1.0 / (1.0 + a_plus)) * (W_yy @ sqrt_y_plus)
            input_drive = (beta * z_t) / 2

            pool_term = tunings.N_matrix @ (y_plus * (u_plus ** 2))

            dy = (-y + input_drive + recurrent_drive - gain_feedback) / tau_y
            du = (-u + (sigma_const ** 2) + pool_term) / tau_u
            da = (-a + u_plus + a * u_plus) / tau_a

            y += dt * dy
            u += dt * du
            a += dt * da

        tuning_curves[:, i] = gaussian_rectify(y)

    return tuning_curves


# =============================================================================
# DIAGNOSTIC 1: Adaptation Convergence
# =============================================================================

def diagnostic_convergence():
    """
    Shows how the adaptive + biased tuning curves look after
    100, 1000, 3000, and 6000 adaptation steps.
    """
    print("\n" + "=" * 60)
    print("  DIAGNOSTIC 1: Adaptation Convergence")
    print("=" * 60)

    checkpoints = [ 2000, 6000, 8000, 10000]
    max_steps = max(checkpoints)

    tunings = V1Tunings(N=N)
    stim_gen = StimulusGenerator(N=N, K=N, stream_length=max_steps)

    # Generate ensembles (same seed → reproducible)
    seq_bias = stim_gen.generate_input_ensembles(biased=True)
    seq_uni = stim_gen.generate_input_ensembles(biased=False)

    # --- Uniform reference (fully adapted) ---
    print("\nAdapting to Uniform Ensemble (reference)...")
    engine_uni = V1Dynamics(tunings, frame, adaptive=True)
    _, gains_hist_uni = engine_uni.run_simulation(seq_uni)
    final_gains_uni = gains_hist_uni[:, -1].copy()
    del gains_hist_uni, engine_uni
    gc.collect()

    print("Probing Uniform reference...")
    tc_uni_raw = run_probe(frame, tunings, final_gains_uni, probe_angles)
    binned_uni = get_binned_curves(tc_uni_raw, tunings.theta, probe_angles, N_BINS)
    ref_max = np.max(binned_uni)
    ref_min = np.min(binned_uni)

    # --- Biased adaptation (full run, keep gains_hist for checkpoints) ---
    print("\nAdapting to Biased Ensemble...")
    engine_bias = V1Dynamics(tunings, frame, adaptive=True)
    _, gains_hist_bias = engine_bias.run_simulation(seq_bias)
    del engine_bias
    gc.collect()

    # --- Probe at each checkpoint ---
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes_flat = axes.flatten()

    for idx, step in enumerate(checkpoints):
        print(f"Probing at step {step}...")
        frozen_gains = gains_hist_bias[:, step - 1].copy()
        tc_raw = run_probe(frame, tunings, frozen_gains, probe_angles)
        binned = get_binned_curves(tc_raw, tunings.theta, probe_angles, N_BINS)
        binned_norm = normalize_with_reference(binned, ref_max, ref_min)

        ax = axes_flat[idx]
        plot_tuning_panel(ax, binned_norm, title=f"Step {step}")
        if idx >= 2:
            ax.set_xlabel("Orientation Relative to Adaptor (\u00b0)")
        if idx % 2 == 0:
            ax.set_ylabel("Normalized Response")

    del gains_hist_bias
    gc.collect()

    fig.suptitle("Diagnostic 1: Adaptation Convergence (Biased Ensemble)",
                 fontweight='bold', fontsize=13)
    plt.tight_layout()
    plt.show()


# =============================================================================
# DIAGNOSTIC 2: Sigma Sweep
# =============================================================================

def diagnostic_sigma_sweep():
    """
    Shows the adaptive + biased tuning curves for different inhibitory
    spread widths (sigma_inh), with sigma_exc fixed at 0.2.
    """
    print("\n" + "=" * 60)
    print("  DIAGNOSTIC 2: Sigma Sweep (sigma_exc=0.2)")
    print("=" * 60)

    STREAM_LENGTH = 8000
    sigma_exc = 0.15
    sigma_inh_values = [0.4, 0.5, 0.6, 0.7]

    stim_gen = StimulusGenerator(N=N, K=N, stream_length=STREAM_LENGTH)
    seq_bias = stim_gen.generate_input_ensembles(biased=True)
    seq_uni = stim_gen.generate_input_ensembles(biased=False)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes_flat = axes.flatten()

    for idx, sigma_inh in enumerate(sigma_inh_values):
        label = f"\u03c3_exc={sigma_exc}, \u03c3_inh={sigma_inh}"
        print(f"\n--- {label} ---")

        tunings = V1Tunings(N=N, sigma_exc=sigma_exc, sigma_inh=sigma_inh)

        # Adapt to uniform → normalization reference
        print("  Adapting to Uniform...")
        engine_uni = V1Dynamics(tunings, frame, adaptive=True)
        _, gains_hist_uni = engine_uni.run_simulation(seq_uni)
        final_gains_uni = gains_hist_uni[:, -1].copy()
        del gains_hist_uni, engine_uni
        gc.collect()

        print("  Probing Uniform...")
        tc_uni_raw = run_probe(frame, tunings, final_gains_uni, probe_angles)
        binned_uni = get_binned_curves(tc_uni_raw, tunings.theta, probe_angles, N_BINS)
        ref_max = np.max(binned_uni)
        ref_min = np.min(binned_uni)

        # Adapt to biased → probe → normalize
        print("  Adapting to Biased...")
        engine_bias = V1Dynamics(tunings, frame, adaptive=True)
        _, gains_hist_bias = engine_bias.run_simulation(seq_bias)
        final_gains_bias = gains_hist_bias[:, -1].copy()
        del gains_hist_bias, engine_bias
        gc.collect()

        print("  Probing Biased...")
        tc_bias_raw = run_probe(frame, tunings, final_gains_bias, probe_angles)
        binned_bias = get_binned_curves(tc_bias_raw, tunings.theta, probe_angles, N_BINS)
        binned_norm = normalize_with_reference(binned_bias, ref_max, ref_min)

        ax = axes_flat[idx]
        plot_tuning_panel(ax, binned_norm, title=label)
        if idx >= 2:
            ax.set_xlabel("Orientation Relative to Adaptor (\u00b0)")
        if idx % 2 == 0:
            ax.set_ylabel("Normalized Response")

    fig.suptitle("Diagnostic 2: Sigma Sweep (Adaptive + Biased)",
                 fontweight='bold', fontsize=13)
    plt.tight_layout()
    plt.show()


# =============================================================================
# DIAGNOSTIC 3: Tuning Width Sweep
# =============================================================================

def diagnostic_tuning_width_sweep():
    """
    Replicates the first plot in Carandini_plots.py (adaptive + biased tuning
    curves) for raised-cosine tuning widths of 1 through 6.

    Layout: 2 × 3 grid, one panel per tuning_width value.
    Each panel shows the 13 binned tuning curves after adaptation to the biased
    ensemble, normalized to the uniform-adapted reference — identical to the
    bottom-right panel of the Carandini_plots Figure 1.
    """
    print("\n" + "=" * 60)
    print("  DIAGNOSTIC 3: Tuning Width Sweep (w = 1 – 6)")
    print("=" * 60)

    STREAM_LENGTH = 8000
    tuning_widths = [1, 1.5, 2, 2.5]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes_flat = axes.flatten()

    for idx, tw in enumerate(tuning_widths):
        label = f"Tuning Width = {tw}"
        print(f"\n--- {label} ---")

        tunings = V1Tunings(N=N)
        stim_gen = StimulusGenerator(N=N, K=N, stream_length=STREAM_LENGTH,
                                     tuning_width=tw)

        seq_uni = stim_gen.generate_input_ensembles(biased=False)
        seq_bias = stim_gen.generate_input_ensembles(biased=True)

        # Uniform adaptation → normalization reference
        print("  Adapting to Uniform...")
        engine_uni = V1Dynamics(tunings, frame, adaptive=True)
        _, gains_hist_uni = engine_uni.run_simulation(seq_uni)
        final_gains_uni = gains_hist_uni[:, -1].copy()
        del gains_hist_uni, engine_uni
        gc.collect()

        print("  Probing Uniform...")
        tc_uni_raw = run_probe(frame, tunings, final_gains_uni, probe_angles)
        binned_uni = get_binned_curves(tc_uni_raw, tunings.theta,
                                       probe_angles, N_BINS)
        ref_max = np.max(binned_uni)
        ref_min = np.min(binned_uni)

        # Biased adaptation → probe → normalize
        print("  Adapting to Biased...")
        engine_bias = V1Dynamics(tunings, frame, adaptive=True)
        _, gains_hist_bias = engine_bias.run_simulation(seq_bias)
        final_gains_bias = gains_hist_bias[:, -1].copy()
        del gains_hist_bias, engine_bias
        gc.collect()

        print("  Probing Biased...")
        tc_bias_raw = run_probe(frame, tunings, final_gains_bias, probe_angles)
        binned_bias = get_binned_curves(tc_bias_raw, tunings.theta,
                                        probe_angles, N_BINS)
        binned_norm = normalize_with_reference(binned_bias, ref_max, ref_min)

        ax = axes_flat[idx]
        plot_tuning_panel(ax, binned_norm, title=label)
        if idx >= 2:
            ax.set_xlabel("Orientation Relative to Adaptor (\u00b0)")
        if idx % 2 == 0:
            ax.set_ylabel("Normalized Response")

    fig.suptitle("Diagnostic 3: Tuning Width Sweep (Adaptive + Biased, w = 1–6)",
                 fontweight='bold', fontsize=13)
    plt.tight_layout()
    plt.show()


# =============================================================================
# DIAGNOSTIC 4: White Noise Probe
# =============================================================================

def diagnostic_noise_probe(noise_level=0.4):
    """
    Probes the adaptive model with white noise added at all orientations.
    Shows tuning curves for both the uniform- and biased-adapted states
    side by side (1x2), so the effect of adaptation under noisy conditions
    is visible.
    """
    print("\n" + "=" * 60)
    print(f"  DIAGNOSTIC 4: Noisy Probe (noise_level={noise_level})")
    print("=" * 60)

    STREAM_LENGTH = 6000
    tunings = V1Tunings(N=N)
    stim_gen = StimulusGenerator(N=N, K=N, stream_length=STREAM_LENGTH)

    seq_uni = stim_gen.generate_input_ensembles(biased=False)
    seq_bias = stim_gen.generate_input_ensembles(biased=True)

    # --- Adapt to Uniform ---
    print("\nAdapting to Uniform Ensemble...")
    engine_uni = V1Dynamics(tunings, frame, adaptive=True)
    _, gains_hist_uni = engine_uni.run_simulation(seq_uni)
    final_gains_uni = gains_hist_uni[:, -1].copy()
    del gains_hist_uni, engine_uni
    gc.collect()

    # --- Adapt to Biased ---
    print("Adapting to Biased Ensemble...")
    engine_bias = V1Dynamics(tunings, frame, adaptive=True)
    _, gains_hist_bias = engine_bias.run_simulation(seq_bias)
    final_gains_bias = gains_hist_bias[:, -1].copy()
    del gains_hist_bias, engine_bias
    gc.collect()

    # --- Noisy probes ---
    print("Probing Uniform state (with noise)...")
    tc_uni_raw = run_probe_noisy(frame, tunings, final_gains_uni,
                                 probe_angles, noise_level=noise_level)
    print("Probing Biased state (with noise)...")
    tc_bias_raw = run_probe_noisy(frame, tunings, final_gains_bias,
                                  probe_angles, noise_level=noise_level)

    # --- Bin & Normalize (uniform reference) ---
    binned_uni = get_binned_curves(tc_uni_raw, tunings.theta,
                                   probe_angles, N_BINS)
    binned_bias = get_binned_curves(tc_bias_raw, tunings.theta,
                                    probe_angles, N_BINS)
    ref_max = np.max(binned_uni)
    ref_min = np.min(binned_uni)

    norm_uni = normalize_with_reference(binned_uni, ref_max, ref_min)
    norm_bias = normalize_with_reference(binned_bias, ref_max, ref_min)

    # --- Plot 1x2 ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    plot_tuning_panel(axes[0], norm_uni, title="Uniform Ensemble")
    axes[0].set_ylabel("Normalized Response")
    axes[0].set_xlabel("Orientation Relative to Adaptor (\u00b0)")

    plot_tuning_panel(axes[1], norm_bias, title="Biased Ensemble")
    axes[1].set_xlabel("Orientation Relative to Adaptor (\u00b0)")

    fig.suptitle(f"Diagnostic 4: Adaptive Tuning with White Noise "
                 f"(\u03c3_noise = {noise_level})",
                 fontweight='bold', fontsize=13)
    plt.tight_layout()
    plt.show()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    #diagnostic_convergence()
    diagnostic_sigma_sweep()
    diagnostic_tuning_width_sweep()
    #diagnostic_noise_probe()
