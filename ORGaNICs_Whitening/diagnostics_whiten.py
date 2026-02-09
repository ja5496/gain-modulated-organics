"""
diagnostics_whiten.py
---------------------
Diagnostic suite for investigating unexpected firing at non-input-driven
orientations in the V1 whitening simulation.

Run from the ORGaNICs_Whitening/ directory:
    python diagnostics_whiten.py

Each diagnostic is a standalone function that prints numerical summaries
and produces matplotlib figures.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from tqdm import tqdm
import time
import sys
import os

from tunings_whiten import V1Tunings
from stimuli_whiten import StimulusGenerator
from simulation_whiten import V1Dynamics, Frame


# ────────────────────────────────────────────────────────────────────
# Shared setup
# ────────────────────────────────────────────────────────────────────

N_NEURONS = 60
CSV_PATH = "N60_Frame.csv"

# Default simulation regimes (same as simulation_whiten.py)
DEFAULT_REGIMES = [
    {'n_steps': 5000, 'contrast': 0.75, 'orientation': np.pi / 2, 'label': 'Bright 90°'},
    {'n_steps': 5000, 'contrast': 0.2,  'orientation': np.pi / 2, 'label': 'Dim 90°'},
    {'n_steps': 5000, 'contrast': 0.5,  'orientation': 0.0,       'label': 'Medium 0°'},
]


def _gaussian_rectify(y, threshold=0.0, sigma=0.1, r_max=1.0):
    """Standalone copy of V1Dynamics.gaussian_rectify for probes."""
    return 0.5 * (1 + erf((y - threshold) / (sigma * np.sqrt(2)))) * r_max


def _steady_state_profile(rates, tunings, regime_idx, regimes, window=200):
    """Average firing over the last `window` steps of the given regime."""
    t_start = sum(r['n_steps'] for r in regimes[:regime_idx])
    t_end = t_start + regimes[regime_idx]['n_steps']
    return np.mean(rates[:, max(0, t_end - window):t_end], axis=1)


# ────────────────────────────────────────────────────────────────────
# DIAGNOSTIC 1 — Gaussian Rectification Baseline Leak
# ────────────────────────────────────────────────────────────────────

def diagnostic_1a(tunings):
    """Numerical probe: quantify the gaussian_rectify baseline and its
    downstream effect on recurrent drive."""
    print("\n" + "=" * 60)
    print("DIAGNOSTIC 1A: Gaussian Rectification Baseline Probe")
    print("=" * 60)

    test_values = np.array([-1.0, -0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5, 1.0])
    print(f"\n{'y':>6s}  {'g_rect(y)':>10s}  {'sqrt(g_rect)':>12s}")
    print("-" * 32)
    for yv in test_values:
        gr = _gaussian_rectify(yv)
        print(f"{yv:>+6.1f}  {gr:>10.6f}  {np.sqrt(gr):>12.6f}")

    # Baseline recurrent drive when all neurons are at y = 0
    baseline_y = np.zeros(N_NEURONS)
    baseline_y_plus = _gaussian_rectify(baseline_y)
    baseline_sqrt = np.sqrt(baseline_y_plus)
    baseline_rec = tunings.W_yy @ baseline_sqrt

    print(f"\nAll-neurons-at-zero recurrent drive (W_yy @ sqrt(y_plus)):")
    print(f"  y_plus baseline value:  {baseline_y_plus[0]:.4f}")
    print(f"  sqrt(y_plus) baseline:  {baseline_sqrt[0]:.4f}")
    print(f"  recurrent drive range:  [{baseline_rec.min():.4f}, {baseline_rec.max():.4f}]")
    print(f"  recurrent drive mean:   {baseline_rec.mean():.4f}")


def diagnostic_1b(tunings, frame, stim_gen):
    """Ablation: compare original gaussian_rectify vs hard ReLU."""
    print("\n" + "=" * 60)
    print("DIAGNOSTIC 1B: Gaussian Rectify vs Hard ReLU Ablation")
    print("=" * 60)

    class V1DynamicsHardReLU(V1Dynamics):
        def gaussian_rectify(self, y, threshold=0.0, sigma=0.1, r_max=1.0):
            return np.maximum(y, 0.0)

    inputs = stim_gen.generate_sequence(DEFAULT_REGIMES)

    engine_orig = V1Dynamics(tunings, frame, dt=0.05)
    rates_orig, _ = engine_orig.run_simulation(inputs)

    engine_relu = V1DynamicsHardReLU(tunings, frame, dt=0.05)
    rates_relu, _ = engine_relu.run_simulation(inputs)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    profile_orig = _steady_state_profile(rates_orig, tunings, 0, DEFAULT_REGIMES)
    profile_relu = _steady_state_profile(rates_relu, tunings, 0, DEFAULT_REGIMES)

    theta_deg = tunings.theta * 180 / np.pi

    axes[0].plot(theta_deg, profile_orig, 'r-', lw=2, label='Gaussian Rectify')
    axes[0].plot(theta_deg, profile_relu, 'b--', lw=2, label='Hard ReLU')
    axes[0].set_title("Steady-State Tuning (Bright 90° regime)")
    axes[0].set_xlabel("Preferred Orientation (deg)")
    axes[0].set_ylabel("Activity")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].semilogy(theta_deg, profile_orig + 1e-10, 'r-', lw=2, label='Gaussian Rectify')
    axes[1].semilogy(theta_deg, profile_relu + 1e-10, 'b--', lw=2, label='Hard ReLU')
    axes[1].set_title("Log Scale (reveals baseline floor)")
    axes[1].set_xlabel("Preferred Orientation (deg)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("DIAGNOSTIC 1B: Gaussian Rectify vs Hard ReLU")
    plt.tight_layout()
    plt.show()


def diagnostic_1c(tunings, frame, stim_gen):
    """Threshold sweep for the gaussian rectification."""
    print("\n" + "=" * 60)
    print("DIAGNOSTIC 1C: Gaussian Rectify Threshold Sweep")
    print("=" * 60)

    thresholds = [0.0, 0.1, 0.2, 0.3, 0.5]
    short_regime = [{'n_steps': 5000, 'contrast': 0.75, 'orientation': np.pi / 2, 'label': '90°'}]
    inputs = stim_gen.generate_sequence(short_regime)

    fig, axes = plt.subplots(1, len(thresholds), figsize=(4 * len(thresholds), 4), sharey=True)
    theta_deg = tunings.theta * 180 / np.pi

    for idx, thr in enumerate(thresholds):
        class V1DynamicsThresh(V1Dynamics):
            _thr = thr
            def gaussian_rectify(self, y, threshold=None, sigma=0.1, r_max=1.0):
                return 0.5 * (1 + erf((y - self._thr) / (sigma * np.sqrt(2)))) * r_max

        eng = V1DynamicsThresh(tunings, frame, dt=0.05)
        r, _ = eng.run_simulation(inputs)

        profile = np.mean(r[:, -200:], axis=1)
        gr_at_zero = 0.5 * (1 + erf(-thr / (0.1 * np.sqrt(2))))

        axes[idx].plot(theta_deg, profile, 'k-', lw=2)
        axes[idx].set_title(f"thr={thr:.1f}\ng_rect(0)={gr_at_zero:.4f}")
        axes[idx].set_xlabel("Pref (deg)")
        axes[idx].grid(True, alpha=0.3)

    axes[0].set_ylabel("Activity")
    plt.suptitle("DIAGNOSTIC 1C: Gaussian Rectify Threshold Sweep")
    plt.tight_layout()
    plt.show()


# ────────────────────────────────────────────────────────────────────
# DIAGNOSTIC 2 — Stimulus Bandwidth
# ────────────────────────────────────────────────────────────────────

def diagnostic_2a():
    """Stimulus profile analysis: compare bandwidths at different kappa."""
    print("\n" + "=" * 60)
    print("DIAGNOSTIC 2A: Stimulus Bandwidth Analysis")
    print("=" * 60)

    theta = np.linspace(0, np.pi, N_NEURONS, endpoint=False)
    stim_orientation = np.pi / 2
    contrast = 0.75
    kappas = [6.0, 8.0, 12.0, 20.0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for kappa in kappas:
        profile = np.exp(kappa * np.cos(2 * (theta - stim_orientation)))
        profile_scaled = 2 * profile / np.max(profile) * contrast
        axes[0].plot(theta * 180 / np.pi, profile_scaled, lw=2, label=f'kappa={kappa}')
        axes[1].semilogy(theta * 180 / np.pi, profile_scaled + 1e-15, lw=2, label=f'kappa={kappa}')

    axes[0].set_title("Stimulus Profile (Linear)")
    axes[1].set_title("Stimulus Profile (Log Scale)")
    for ax in axes:
        ax.set_xlabel("Preferred Orientation (deg)")
        ax.set_ylabel("Drive magnitude")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle("DIAGNOSTIC 2A: Stimulus Bandwidth")
    plt.tight_layout()
    plt.show()

    # Numerical table
    offsets_deg = [0, 10, 20, 30, 45, 60, 90]
    print(f"\n{'Offset':>8s}", end="")
    for k in kappas:
        print(f"  {'k=' + str(k):>12s}", end="")
    print()
    print("-" * (8 + 14 * len(kappas)))

    for off in offsets_deg:
        off_rad = off * np.pi / 180
        print(f"{off:>6d}°", end="")
        for k in kappas:
            val = 2 * np.exp(k * np.cos(2 * off_rad)) / np.exp(k) * contrast
            print(f"  {val:>12.6f}", end="")
        print()


def diagnostic_2b(tunings, frame, stim_gen):
    """Kappa sweep: run simulations at different stimulus sharpness."""
    print("\n" + "=" * 60)
    print("DIAGNOSTIC 2B: Stimulus Kappa Sweep")
    print("=" * 60)

    kappas = [6.0, 8.0, 12.0, 20.0]
    theta = np.linspace(0, np.pi, N_NEURONS, endpoint=False)

    fig, axes = plt.subplots(1, len(kappas), figsize=(4 * len(kappas), 4), sharey=True)

    for idx, kappa in enumerate(kappas):
        profile = np.exp(kappa * np.cos(2 * (theta - np.pi / 2)))
        profile = 2 * profile / np.max(profile) * 0.75
        inputs = np.tile(profile, (3000, 1)).T

        eng = V1Dynamics(tunings, frame, dt=0.05)
        r, _ = eng.run_simulation(inputs)

        avg = np.mean(r[:, -200:], axis=1)
        axes[idx].plot(tunings.theta * 180 / np.pi, avg, 'k-', lw=2)
        axes[idx].set_title(f"kappa={kappa}")
        axes[idx].set_xlabel("Pref (deg)")
        axes[idx].grid(True, alpha=0.3)

    axes[0].set_ylabel("Activity")
    plt.suptitle("DIAGNOSTIC 2B: Stimulus Kappa Sweep")
    plt.tight_layout()
    plt.show()


# ────────────────────────────────────────────────────────────────────
# DIAGNOSTIC 3 — Recurrent Weight Matrix Structure
# ────────────────────────────────────────────────────────────────────

def diagnostic_3a(tunings):
    """Structural analysis of W_yy: profile, eigenspectrum, E/I budget."""
    print("\n" + "=" * 60)
    print("DIAGNOSTIC 3A: Recurrent Weight Matrix (W_yy) Structure")
    print("=" * 60)

    W_yy = tunings.W_yy
    theta_deg = tunings.theta * 180 / np.pi
    eigvals = np.sort(np.real(np.linalg.eigvals(W_yy)))[::-1]

    center = N_NEURONS // 2  # neuron preferring ~90 deg
    profile = W_yy[center, :]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (a) Connection profile for center neuron
    axes[0, 0].plot(theta_deg, profile, 'k-', lw=2)
    axes[0, 0].axhline(0, color='red', ls='--', alpha=0.5)
    sign_changes = np.where(np.diff(np.sign(profile)))[0]
    for sc in sign_changes:
        axes[0, 0].axvline(theta_deg[sc], color='blue', ls=':', alpha=0.7)
    axes[0, 0].set_title(f"W_yy[{center},:] (neuron at {theta_deg[center]:.0f}°)")
    axes[0, 0].set_xlabel("Target Orientation (deg)")
    axes[0, 0].set_ylabel("Weight")
    axes[0, 0].grid(True, alpha=0.3)

    print(f"\nCenter neuron: {center} (pref = {theta_deg[center]:.1f}°)")
    for sc in sign_changes:
        print(f"  Zero crossing at ~{theta_deg[sc]:.1f}°")

    # (b) Eigenvalue spectrum
    axes[0, 1].plot(eigvals, 'ko-', ms=3)
    axes[0, 1].axhline(1.0, color='red', ls='--', label='max_eig = 1')
    axes[0, 1].axhline(0, color='gray', ls='-', alpha=0.3)
    axes[0, 1].set_title("Eigenvalue Spectrum")
    axes[0, 1].set_xlabel("Index")
    axes[0, 1].set_ylabel("Eigenvalue")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    print(f"  Eigenvalue range: [{eigvals[-1]:.4f}, {eigvals[0]:.4f}]")
    print(f"  Positive eigenvalues: {np.sum(eigvals > 0)}")
    print(f"  Negative eigenvalues: {np.sum(eigvals < 0)}")

    # (c) W_yy heatmap
    vlim = np.max(np.abs(W_yy))
    im = axes[0, 2].imshow(W_yy, aspect='auto', cmap='RdBu_r', origin='lower',
                            vmin=-vlim, vmax=vlim)
    axes[0, 2].set_title("W_yy Heatmap")
    axes[0, 2].set_xlabel("Neuron j")
    axes[0, 2].set_ylabel("Neuron i")
    plt.colorbar(im, ax=axes[0, 2])

    # (d) Recurrent drive comparison: gaussian_rectify vs hard ReLU baseline
    y_single = np.zeros(N_NEURONS)
    y_single[center] = 1.0

    y_plus_gr = _gaussian_rectify(y_single)
    y_plus_relu = np.maximum(y_single, 0.0)

    rec_gr = W_yy @ np.sqrt(y_plus_gr)
    rec_relu = W_yy @ np.sqrt(y_plus_relu)

    axes[1, 0].plot(theta_deg, rec_gr, 'r-', lw=2, label='gaussian_rectify')
    axes[1, 0].plot(theta_deg, rec_relu, 'b--', lw=2, label='hard ReLU')
    axes[1, 0].set_title(f"Recurrent drive: 1 neuron at {theta_deg[center]:.0f}°")
    axes[1, 0].set_xlabel("Orientation (deg)")
    axes[1, 0].set_ylabel("W_yy @ sqrt(y_plus)")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # (e) E/I budget
    total_exc = np.sum(profile[profile > 0])
    total_inh = np.sum(profile[profile < 0])
    axes[1, 1].bar(['Excitation', 'Inhibition', 'Net'],
                    [total_exc, total_inh, total_exc + total_inh],
                    color=['green', 'red', 'gray'])
    axes[1, 1].set_title(f"E/I Budget (neuron {center})")
    axes[1, 1].set_ylabel("Total weight")
    axes[1, 1].grid(True, alpha=0.3)

    print(f"  Total excitation: {total_exc:.4f}")
    print(f"  Total inhibition: {total_inh:.4f}")
    print(f"  E/I ratio: {abs(total_exc / total_inh):.4f}")

    # (f) Spatial extent
    exc_extent = np.sum(profile > 0) * (180.0 / N_NEURONS)
    inh_extent = np.sum(profile < 0) * (180.0 / N_NEURONS)
    axes[1, 2].bar(['Excitatory\nextent', 'Inhibitory\nextent'],
                    [exc_extent, inh_extent], color=['green', 'red'])
    axes[1, 2].set_ylabel("Degrees")
    axes[1, 2].set_title("Spatial Extent")
    axes[1, 2].grid(True, alpha=0.3)

    print(f"  Excitatory extent: {exc_extent:.0f}°")
    print(f"  Inhibitory extent: {inh_extent:.0f}°")

    plt.suptitle("DIAGNOSTIC 3A: W_yy Structure Analysis")
    plt.tight_layout()
    plt.show()


def diagnostic_3b(frame, stim_gen):
    """Sigma sweep for W_yy: grid of (sigma_exc) x (sigma_inh)."""
    print("\n" + "=" * 60)
    print("DIAGNOSTIC 3B: W_yy Sigma Sweep")
    print("=" * 60)

    sigma_exc_vals = [0.10, 0.15, 0.20]
    sigma_inh_vals = [0.3, 0.4, 0.6]

    short_regime = [{'n_steps': 3000, 'contrast': 0.75, 'orientation': np.pi / 2, 'label': ''}]
    inputs = stim_gen.generate_sequence(short_regime)

    fig, axes = plt.subplots(len(sigma_exc_vals), len(sigma_inh_vals),
                              figsize=(5 * len(sigma_inh_vals), 4 * len(sigma_exc_vals)),
                              sharey=True, sharex=True)

    for i, se in enumerate(sigma_exc_vals):
        for j, si in enumerate(sigma_inh_vals):
            t_test = V1Tunings(N=N_NEURONS, sigma_exc=se, sigma_inh=si)
            eng = V1Dynamics(t_test, frame, dt=0.05)
            r, _ = eng.run_simulation(inputs)

            avg = np.mean(r[:, -200:], axis=1)
            axes[i, j].plot(t_test.theta * 180 / np.pi, avg, 'k-', lw=2)
            axes[i, j].set_title(f"σ_exc={se}, σ_inh={si}")
            axes[i, j].grid(True, alpha=0.3)

            if j == 0:
                axes[i, j].set_ylabel(f"Activity")
            if i == len(sigma_exc_vals) - 1:
                axes[i, j].set_xlabel("Pref (deg)")

    plt.suptitle("DIAGNOSTIC 3B: W_yy Sigma Sweep (Steady State at 90°)")
    plt.tight_layout()
    plt.show()


# ────────────────────────────────────────────────────────────────────
# DIAGNOSTIC 6 — Full Instrumented Simulation
# ────────────────────────────────────────────────────────────────────

def diagnostic_6a(tunings, frame, stim_gen):
    """Instrumented simulation recording every internal variable for
    three representative neurons: peak, flank, orthogonal."""
    print("\n" + "=" * 60)
    print("DIAGNOSTIC 6A: Full Instrumented Simulation")
    print("=" * 60)

    inputs = stim_gen.generate_sequence(DEFAULT_REGIMES)
    N, n_steps = inputs.shape
    K = frame.K

    # Identify representative neurons
    peak_idx = np.argmin(np.abs(tunings.theta - np.pi / 2))
    flank_idx = np.argmin(np.abs(tunings.theta - np.pi / 4))
    orth_idx = 0
    rep_idxs = [peak_idx, flank_idx, orth_idx]
    rep_labels = {peak_idx: f'Peak ({tunings.theta[peak_idx]*180/np.pi:.0f}°)',
                  flank_idx: f'Flank ({tunings.theta[flank_idx]*180/np.pi:.0f}°)',
                  orth_idx: f'Orth ({tunings.theta[orth_idx]*180/np.pi:.0f}°)'}
    rep_colors = {peak_idx: 'red', flank_idx: 'orange', orth_idx: 'blue'}

    # Allocate trace arrays (only for representative neurons to save memory)
    traces = {name: np.zeros((3, n_steps)) for name in
              ['y_raw', 'y_plus', 'input_drive', 'recurrent_drive',
               'gain_feedback', 'norm_factor']}

    engine = V1Dynamics(tunings, frame, dt=0.05)

    state = np.zeros(3 * N + K)
    state[3 * N:3 * N + K] = 0.2

    print("Running instrumented simulation...")
    for t in tqdm(range(n_steps)):
        z_t = inputs[:, t]

        y = state[0:N]
        u = state[N:2 * N]
        a = state[2 * N:3 * N]
        g = state[3 * N:3 * N + K]
        g_c = np.maximum(g, 0.0)

        y_plus = _gaussian_rectify(y)
        u_plus = _gaussian_rectify(u)
        a_plus = _gaussian_rectify(a)
        sqrt_y_plus = np.sqrt(y_plus)

        v_t = frame.W.T @ y
        gain_feedback = frame.W @ (g_c * v_t)
        recurrent_drive = (1.0 / (1.0 + a_plus)) * (tunings.W_yy @ sqrt_y_plus)
        input_drive = (engine.beta * z_t) / 2
        norm_factor = 1.0 / (1.0 + a_plus)

        for ri, nidx in enumerate(rep_idxs):
            traces['y_raw'][ri, t] = y[nidx]
            traces['y_plus'][ri, t] = y_plus[nidx]
            traces['input_drive'][ri, t] = input_drive[nidx]
            traces['recurrent_drive'][ri, t] = recurrent_drive[nidx]
            traces['gain_feedback'][ri, t] = gain_feedback[nidx]
            traces['norm_factor'][ri, t] = norm_factor[nidx]

        # RK4 step (reuse engine's _derivatives)
        k1 = engine._derivatives(state, z_t)
        k2 = engine._derivatives(state + 0.5 * engine.dt * k1, z_t)
        k3 = engine._derivatives(state + 0.5 * engine.dt * k2, z_t)
        k4 = engine._derivatives(state + engine.dt * k3, z_t)
        state += (engine.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        state[3 * N:3 * N + K] = np.maximum(state[3 * N:3 * N + K], 0.0)

    # Plot
    trace_names = ['y_raw', 'y_plus', 'input_drive', 'recurrent_drive',
                   'gain_feedback', 'norm_factor']
    trace_titles = ['Raw y (membrane)', 'y_plus (gaussian_rectify)',
                    'Input drive', 'Recurrent drive',
                    'Gain feedback', 'Norm factor 1/(1+a+)']

    fig, axes = plt.subplots(len(trace_names), 1, figsize=(12, 3 * len(trace_names)),
                              sharex=True)

    for ax, name, title in zip(axes, trace_names, trace_titles):
        for ri, nidx in enumerate(rep_idxs):
            ax.plot(traces[name][ri, :], color=rep_colors[nidx], lw=1.2,
                    label=rep_labels[nidx])
        ax.set_title(title)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

        # Regime dividers
        t_cursor = 0
        for r in DEFAULT_REGIMES:
            t_cursor += r['n_steps']
            ax.axvline(t_cursor, color='gray', ls='--', alpha=0.5)

    axes[-1].set_xlabel("Time step")
    plt.suptitle("DIAGNOSTIC 6A: Internal State Traces (3 Representative Neurons)")
    plt.tight_layout()
    plt.show()


def diagnostic_6b(tunings, frame, stim_gen):
    """Compare displayed rate max(y,0) vs actual y_plus = gaussian_rectify(y)."""
    print("\n" + "=" * 60)
    print("DIAGNOSTIC 6B: Displayed Rate vs Actual y_plus")
    print("=" * 60)

    inputs = stim_gen.generate_sequence(DEFAULT_REGIMES)
    engine = V1Dynamics(tunings, frame, dt=0.05)
    rates, _ = engine.run_simulation(inputs)

    # Re-run just to get final state
    N = N_NEURONS
    K = frame.K
    state = np.zeros(3 * N + K)
    state[3 * N:3 * N + K] = 0.2

    for t in range(inputs.shape[1]):
        z_t = inputs[:, t]
        k1 = engine._derivatives(state, z_t)
        k2 = engine._derivatives(state + 0.5 * engine.dt * k1, z_t)
        k3 = engine._derivatives(state + 0.5 * engine.dt * k2, z_t)
        k4 = engine._derivatives(state + engine.dt * k3, z_t)
        state += (engine.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        state[3 * N:3 * N + K] = np.maximum(state[3 * N:3 * N + K], 0.0)

    final_y = state[0:N]
    displayed = np.maximum(final_y, 0.0)
    actual_y_plus = _gaussian_rectify(final_y)

    theta_deg = tunings.theta * 180 / np.pi

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(theta_deg, displayed, 'b-', lw=2, label='Displayed: max(y, 0)')
    axes[0].plot(theta_deg, actual_y_plus, 'r-', lw=2, label='Actual: gaussian_rectify(y)')
    axes[0].set_title("Final Timestep: Displayed Rate vs Actual y_plus")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylabel("Value")

    axes[1].plot(theta_deg, actual_y_plus - displayed, 'k-', lw=2)
    axes[1].axhline(0.5, color='red', ls='--', label='Expected baseline gap (0.5)')
    axes[1].set_title("Difference: y_plus - displayed rate")
    axes[1].set_xlabel("Preferred Orientation (deg)")
    axes[1].set_ylabel("Delta")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("DIAGNOSTIC 6B: Hidden Activity from Gaussian Rectification")
    plt.tight_layout()
    plt.show()


# ────────────────────────────────────────────────────────────────────
# Main entry point
# ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading shared resources...")
    tunings = V1Tunings(N=N_NEURONS)
    frame = Frame(csv_path=CSV_PATH)
    stim_gen = StimulusGenerator(N=N_NEURONS)

    # ── Diagnostic 1: Gaussian Rectification Baseline ──
    diagnostic_1a(tunings)
    diagnostic_1b(tunings, frame, stim_gen)
    diagnostic_1c(tunings, frame, stim_gen)

    # ── Diagnostic 2: Stimulus Bandwidth ──
    diagnostic_2a()
    diagnostic_2b(tunings, frame, stim_gen)

    # ── Diagnostic 3: Recurrent Weight Matrix ──
    diagnostic_3a(tunings)
    diagnostic_3b(frame, stim_gen)

    # ── Diagnostic 4: Full Instrumented Simulation ──
    diagnostic_6a(tunings, frame, stim_gen)
    diagnostic_6b(tunings, frame, stim_gen)

    print("\n" + "=" * 60)
    print("All diagnostics complete.")
    print("=" * 60)
