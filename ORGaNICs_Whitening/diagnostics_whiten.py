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


# ================================================================
#                  GAIN ADAPTATION DIAGNOSTICS
# ================================================================
#
# Diagnostics 7–11 investigate why the 1830 gain variables collapse
# to zero instead of adapting to whiten neural output.  The core
# issue is that the gain update rule  dg/dt = (v_t^2 - 1) / tau_g
# requires v_t^2 ≈ 1, but the overcomplete frame spreads energy so
# that mean(v_t^2) ≈ ||y||^2 / N, which is << 1 for sparse activity.
# ================================================================


def _run_no_gain_steady_state(tunings, frame, stim_gen, regime_idx=0):
    """Run a simulation with gains forced to zero and return the
    steady-state y vector for the given regime."""

    class V1DynamicsNoGain(V1Dynamics):
        def _derivatives(self, state, z_t):
            N, K = self.v1.N, self.frame.K
            y  = state[0:N]
            u  = state[N:2*N]
            a  = state[2*N:3*N]

            u_plus = self.gaussian_rectify(u)
            y_plus = self.gaussian_rectify(y)
            a_plus = self.gaussian_rectify(a)
            sqrt_y_plus = np.sqrt(y_plus)

            recurrent_drive = (1.0 / (1.0 + a_plus)) * (self.v1.W_yy @ sqrt_y_plus)
            input_drive = (self.beta * z_t) / 2

            sigma_term = self.sigma ** 2
            pool_term  = self.v1.N_matrix @ (y_plus * (u_plus ** 2))

            dy_dt = (-y + input_drive + recurrent_drive) / self.tau_y
            du_dt = (-u + sigma_term + pool_term) / self.tau_u
            da_dt = (-a + u_plus + a * u_plus + self.alpha * du_dt) / self.tau_a
            dg_dt = np.zeros(K)

            return np.concatenate([dy_dt, du_dt, da_dt, dg_dt])

    regime = DEFAULT_REGIMES[regime_idx]
    inputs = stim_gen.generate_sequence([regime])

    eng = V1DynamicsNoGain(tunings, frame, dt=0.05)
    rates, _ = eng.run_simulation(inputs)

    y_ss = rates[:, -1]  # last timestep (already rectified by max(y,0))
    return y_ss


# ────────────────────────────────────────────────────────────────────
# DIAGNOSTIC 7 — v_t^2 Magnitude and Energy Budget
# ────────────────────────────────────────────────────────────────────

def diagnostic_7(tunings, frame, stim_gen):
    """Quantify why dg/dt < 0: the frame energy budget makes v_t^2 = 1
    unreachable given the actual ||y||^2."""
    print("\n" + "=" * 60)
    print("DIAGNOSTIC 7: v_t^2 Energy Budget")
    print("=" * 60)

    y_ss = _run_no_gain_steady_state(tunings, frame, stim_gen, regime_idx=0)

    N = tunings.N
    K = frame.K
    W = frame.W

    v_t = W.T @ y_ss
    v_t_sq = v_t ** 2

    y_norm_sq = np.sum(y_ss ** 2)
    sum_vt_sq = np.sum(v_t_sq)
    mean_vt_sq = np.mean(v_t_sq)
    frame_ratio = K / N
    needed_y_norm_sq = N  # for mean(v_t^2) = 1

    print(f"\n  Steady-state (no gains, Bright 90° regime):")
    print(f"    ||y_ss||^2                     = {y_norm_sq:.4f}")
    print(f"    ||y_ss||^2 / N                 = {y_norm_sq / N:.4f}")
    print(f"    mean(v_t^2)                    = {mean_vt_sq:.6f}")
    print(f"    sum(v_t^2)                     = {sum_vt_sq:.4f}")
    print(f"    (K/N) * ||y||^2                = {frame_ratio * y_norm_sq:.4f}")
    print(f"    Tight frame check (ratio):       {sum_vt_sq / (frame_ratio * y_norm_sq):.4f}  (should be ~1)")
    print(f"    Target: mean(v_t^2) = 1")
    print(f"    Required ||y||^2 for target:    {needed_y_norm_sq:.1f}")
    print(f"    Shortfall factor:                {needed_y_norm_sq / y_norm_sq:.2f}x")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(v_t_sq, bins=80, color='steelblue', alpha=0.7, edgecolor='white')
    axes[0].axvline(1.0, color='red', ls='--', lw=2, label='Target v_t^2 = 1')
    axes[0].axvline(mean_vt_sq, color='orange', ls='-', lw=2, label=f'mean = {mean_vt_sq:.4f}')
    axes[0].set_title("Distribution of v_t^2 (no-gain steady state)")
    axes[0].set_xlabel("v_t^2")
    axes[0].set_ylabel("Count")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    theta_deg = tunings.theta * 180 / np.pi
    axes[1].bar(theta_deg, y_ss, width=180 / N * 0.8, color='steelblue', alpha=0.7)
    axes[1].axhline(0, color='gray', ls='-', alpha=0.3)
    axes[1].set_title(f"y_ss profile  (||y||^2 = {y_norm_sq:.2f}, need {needed_y_norm_sq})")
    axes[1].set_xlabel("Preferred Orientation (deg)")
    axes[1].set_ylabel("y")
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("DIAGNOSTIC 7: v_t^2 Energy Budget")
    plt.tight_layout()
    plt.show()

    return y_ss  # reuse in diagnostic_8


# ────────────────────────────────────────────────────────────────────
# DIAGNOSTIC 8 — Centering Analysis
# ────────────────────────────────────────────────────────────────────

def diagnostic_8(tunings, frame, y_ss):
    """Test whether centering y before projection moves v_t^2 closer to 1."""
    print("\n" + "=" * 60)
    print("DIAGNOSTIC 8: Centering Analysis")
    print("=" * 60)

    N = tunings.N
    W = frame.W

    y_mean = np.mean(y_ss)
    y_centered = y_ss - y_mean

    v_t_raw = W.T @ y_ss
    v_t_cen = W.T @ y_centered

    raw_sq = v_t_raw ** 2
    cen_sq = v_t_cen ** 2

    y_norm_sq = np.sum(y_ss ** 2)
    y_cen_norm_sq = np.sum(y_centered ** 2)
    dc_energy = N * y_mean ** 2

    print(f"\n  mean(y_ss)                  = {y_mean:.4f}")
    print(f"  ||y_ss||^2                  = {y_norm_sq:.4f}")
    print(f"  ||y_centered||^2            = {y_cen_norm_sq:.4f}")
    print(f"  DC energy (N * mean^2)      = {dc_energy:.4f}  ({100 * dc_energy / y_norm_sq:.1f}% of total)")
    print(f"  mean(v_t_raw^2)             = {np.mean(raw_sq):.6f}")
    print(f"  mean(v_t_centered^2)        = {np.mean(cen_sq):.6f}")
    print(f"  ||y_centered||^2 / N        = {y_cen_norm_sq / N:.6f}")
    print(f"  Target                      = 1.0")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    bins = np.linspace(0, max(np.max(raw_sq), np.max(cen_sq), 1.2), 80)
    axes[0].hist(raw_sq, bins=bins, alpha=0.5, color='red', label=f'Raw  mean={np.mean(raw_sq):.4f}')
    axes[0].hist(cen_sq, bins=bins, alpha=0.5, color='blue', label=f'Centered  mean={np.mean(cen_sq):.4f}')
    axes[0].axvline(1.0, color='black', ls='--', lw=2, label='Target = 1')
    axes[0].set_title("v_t^2 distributions: raw vs centered")
    axes[0].set_xlabel("v_t^2")
    axes[0].set_ylabel("Count")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    theta_deg = tunings.theta * 180 / np.pi
    axes[1].bar(theta_deg - 0.8, y_ss, width=1.4, color='red', alpha=0.5, label='Raw y_ss')
    axes[1].bar(theta_deg + 0.8, y_centered, width=1.4, color='blue', alpha=0.5, label='Centered y_ss')
    axes[1].axhline(0, color='gray', ls='-', alpha=0.5)
    axes[1].axhline(y_mean, color='red', ls=':', alpha=0.7, label=f'mean = {y_mean:.3f}')
    axes[1].set_title("y_ss: raw vs centered")
    axes[1].set_xlabel("Preferred Orientation (deg)")
    axes[1].set_ylabel("y")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("DIAGNOSTIC 8: Centering Analysis")
    plt.tight_layout()
    plt.show()


# ────────────────────────────────────────────────────────────────────
# DIAGNOSTIC 9 — Gain Dynamics Trajectory
# ────────────────────────────────────────────────────────────────────

def diagnostic_9(tunings, frame, stim_gen):
    """Visualize the gain collapse: track mean(g), mean(v_t^2), mean(dg/dt),
    and ||y||^2 at every timestep."""
    print("\n" + "=" * 60)
    print("DIAGNOSTIC 9: Gain Dynamics Trajectory")
    print("=" * 60)

    inputs = stim_gen.generate_sequence(DEFAULT_REGIMES)
    N, n_steps = inputs.shape
    K = frame.K

    engine = V1Dynamics(tunings, frame, dt=0.05)

    state = np.zeros(3 * N + K)
    state[3 * N:3 * N + K] = 0.0  # gains start at 0 (matches current simulation_whiten.py)

    mean_g    = np.zeros(n_steps)
    mean_vtsq = np.zeros(n_steps)
    mean_dgdt = np.zeros(n_steps)
    y_norm_sq = np.zeros(n_steps)

    print("Running gain trajectory simulation...")
    for t in tqdm(range(n_steps)):
        z_t = inputs[:, t]

        y = state[0:N]
        g = state[3 * N:3 * N + K]

        v_t = frame.W.T @ y
        dg = (v_t * v_t - 1) / engine.tau_g

        mean_g[t]    = np.mean(g)
        mean_vtsq[t] = np.mean(v_t ** 2)
        mean_dgdt[t] = np.mean(dg)
        y_norm_sq[t] = np.sum(y ** 2)

        # RK4 step
        k1 = engine._derivatives(state, z_t)
        k2 = engine._derivatives(state + 0.5 * engine.dt * k1, z_t)
        k3 = engine._derivatives(state + 0.5 * engine.dt * k2, z_t)
        k4 = engine._derivatives(state + engine.dt * k3, z_t)
        state += (engine.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        state[3 * N:3 * N + K] = np.maximum(state[3 * N:3 * N + K], 0.0)

    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

    traces_data   = [mean_g,    mean_vtsq,    mean_dgdt,    y_norm_sq]
    traces_titles = ['mean(g)',  'mean(v_t^2)', 'mean(dg/dt)', '||y||^2']
    ref_lines     = [None,       1.0,           0.0,           N]
    ref_labels    = [None,       'target = 1',  'dg/dt = 0',   f'||y||^2 = N = {N}']
    colors        = ['purple',   'teal',        'crimson',     'steelblue']

    for ax, data, title, ref, ref_lbl, clr in zip(
            axes, traces_data, traces_titles, ref_lines, ref_labels, colors):
        ax.plot(data, color=clr, lw=1.5)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        if ref is not None:
            ax.axhline(ref, color='gray', ls='--', lw=1.5, label=ref_lbl)
            ax.legend(loc='upper right', fontsize=8)

        t_cursor = 0
        for r in DEFAULT_REGIMES:
            t_cursor += r['n_steps']
            ax.axvline(t_cursor, color='gray', ls='--', alpha=0.4)

    axes[-1].set_xlabel("Time step")
    plt.suptitle("DIAGNOSTIC 9: Gain Dynamics Trajectory")
    plt.tight_layout()
    plt.show()

    print(f"\n  Final mean(g):      {mean_g[-1]:.6f}")
    print(f"  Final mean(v_t^2):  {mean_vtsq[-1]:.6f}")
    print(f"  Final ||y||^2:      {y_norm_sq[-1]:.4f}")


# ────────────────────────────────────────────────────────────────────
# DIAGNOSTIC 10 — Mean-Subtracted Projection Ablation
# ────────────────────────────────────────────────────────────────────

def diagnostic_10(tunings, frame, stim_gen):
    """Compare original v_t = W.T @ y  vs  centered v_t = W.T @ (y - mean(y)).
    Does centering fix gain adaptation?"""
    print("\n" + "=" * 60)
    print("DIAGNOSTIC 10: Mean-Subtracted Projection Ablation")
    print("=" * 60)

    class V1DynamicsCentered(V1Dynamics):
        def _derivatives(self, state, z_t):
            N, K = self.v1.N, self.frame.K
            y = state[0:N]
            u = state[N:2 * N]
            a = state[2 * N:3 * N]
            g = state[3 * N:3 * N + K]

            u_plus = self.gaussian_rectify(u)
            y_plus = self.gaussian_rectify(y)
            a_plus = self.gaussian_rectify(a)
            sqrt_y_plus = np.sqrt(y_plus)

            # KEY CHANGE: center y before frame projection
            v_t = self.frame.W.T @ (y - np.mean(y))

            gain_feedback = self.frame.W @ (g * v_t)
            recurrent_drive = (1.0 / (1.0 + a_plus)) * (self.v1.W_yy @ sqrt_y_plus)
            input_drive = (self.beta * z_t) / 2

            sigma_term = self.sigma ** 2
            pool_term = self.v1.N_matrix @ (y_plus * (u_plus ** 2))

            dy_dt = (-y + input_drive + recurrent_drive - gain_feedback) / self.tau_y
            du_dt = (-u + sigma_term + pool_term) / self.tau_u
            da_dt = (-a + u_plus + a * u_plus + self.alpha * du_dt) / self.tau_a
            dg_dt = (v_t * v_t - 1) / self.tau_g

            return np.concatenate([dy_dt, du_dt, da_dt, dg_dt])

    inputs = stim_gen.generate_sequence(DEFAULT_REGIMES)
    N, n_steps = inputs.shape
    K = frame.K

    # --- Run original ---
    eng_orig = V1Dynamics(tunings, frame, dt=0.05)
    _, gains_orig = eng_orig.run_simulation(inputs)

    # Collect mean(v_t^2) for original
    state_orig = np.zeros(3 * N + K)
    vtsq_orig = np.zeros(n_steps)
    for t in range(n_steps):
        z_t = inputs[:, t]
        k1 = eng_orig._derivatives(state_orig, z_t)
        k2 = eng_orig._derivatives(state_orig + 0.5 * eng_orig.dt * k1, z_t)
        k3 = eng_orig._derivatives(state_orig + 0.5 * eng_orig.dt * k2, z_t)
        k4 = eng_orig._derivatives(state_orig + eng_orig.dt * k3, z_t)
        state_orig += (eng_orig.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        state_orig[3 * N:3 * N + K] = np.maximum(state_orig[3 * N:3 * N + K], 0.0)
        v = frame.W.T @ state_orig[0:N]
        vtsq_orig[t] = np.mean(v ** 2)

    # --- Run centered ---
    eng_cen = V1DynamicsCentered(tunings, frame, dt=0.05)
    _, gains_cen = eng_cen.run_simulation(inputs)

    state_cen = np.zeros(3 * N + K)
    vtsq_cen = np.zeros(n_steps)
    for t in range(n_steps):
        z_t = inputs[:, t]
        k1 = eng_cen._derivatives(state_cen, z_t)
        k2 = eng_cen._derivatives(state_cen + 0.5 * eng_cen.dt * k1, z_t)
        k3 = eng_cen._derivatives(state_cen + 0.5 * eng_cen.dt * k2, z_t)
        k4 = eng_cen._derivatives(state_cen + eng_cen.dt * k3, z_t)
        state_cen += (eng_cen.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        state_cen[3 * N:3 * N + K] = np.maximum(state_cen[3 * N:3 * N + K], 0.0)
        y_c = state_cen[0:N]
        v = frame.W.T @ (y_c - np.mean(y_c))
        vtsq_cen[t] = np.mean(v ** 2)

    # --- Plot ---
    subset_k = np.linspace(0, K - 1, 10, dtype=int)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)

    axes[0, 0].plot(gains_orig[subset_k, :].T, alpha=0.6)
    axes[0, 0].set_title("Gain evolution — Original")
    axes[0, 0].set_ylabel("g (subset)")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(gains_cen[subset_k, :].T, alpha=0.6)
    axes[0, 1].set_title("Gain evolution — Centered v_t")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(vtsq_orig, color='red', lw=1.5)
    axes[1, 0].axhline(1.0, color='gray', ls='--', label='target = 1')
    axes[1, 0].set_title("mean(v_t^2) — Original")
    axes[1, 0].set_ylabel("mean(v_t^2)")
    axes[1, 0].set_xlabel("Time step")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(vtsq_cen, color='blue', lw=1.5)
    axes[1, 1].axhline(1.0, color='gray', ls='--', label='target = 1')
    axes[1, 1].set_title("mean(v_t^2) — Centered")
    axes[1, 1].set_xlabel("Time step")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    for ax_row in axes:
        for ax in ax_row:
            t_cursor = 0
            for r in DEFAULT_REGIMES:
                t_cursor += r['n_steps']
                ax.axvline(t_cursor, color='gray', ls='--', alpha=0.4)

    plt.suptitle("DIAGNOSTIC 10: Original vs Centered Projection")
    plt.tight_layout()
    plt.show()

    print(f"\n  Original — final gain range:  [{np.min(gains_orig[:, -1]):.6f}, {np.max(gains_orig[:, -1]):.6f}]")
    print(f"  Centered — final gain range:  [{np.min(gains_cen[:, -1]):.6f}, {np.max(gains_cen[:, -1]):.6f}]")
    print(f"  Original — final mean(v_t^2): {vtsq_orig[-1]:.6f}")
    print(f"  Centered — final mean(v_t^2): {vtsq_cen[-1]:.6f}")


# ────────────────────────────────────────────────────────────────────
# DIAGNOSTIC 11 — Gain Rule Variants
# ────────────────────────────────────────────────────────────────────

def diagnostic_11(tunings, frame, stim_gen):
    """Compare three gain rule variants to find one that produces
    stable, nonzero gains:
      (a) Centered projection, target = 1
      (b) Raw projection, adaptive target = ||y||^2 / N
      (c) Centered projection, adaptive target = ||y_cen||^2 / N
    """
    print("\n" + "=" * 60)
    print("DIAGNOSTIC 11: Gain Rule Variants")
    print("=" * 60)

    N = tunings.N
    K = frame.K
    inputs = stim_gen.generate_sequence(DEFAULT_REGIMES)
    n_steps = inputs.shape[1]

    # --- Variant (a): Centered, target = 1 ---
    class VariantA(V1Dynamics):
        def _derivatives(self, state, z_t):
            NN, KK = self.v1.N, self.frame.K
            y = state[0:NN]; u = state[NN:2*NN]; a = state[2*NN:3*NN]; g = state[3*NN:3*NN+KK]
            u_plus = self.gaussian_rectify(u)
            y_plus = self.gaussian_rectify(y)
            a_plus = self.gaussian_rectify(a)
            sqrt_y_plus = np.sqrt(y_plus)
            v_t = self.frame.W.T @ (y - np.mean(y))  # centered
            gain_feedback = self.frame.W @ (g * v_t)
            recurrent_drive = (1.0 / (1.0 + a_plus)) * (self.v1.W_yy @ sqrt_y_plus)
            input_drive = (self.beta * z_t) / 2
            sigma_term = self.sigma ** 2
            pool_term = self.v1.N_matrix @ (y_plus * (u_plus ** 2))
            dy_dt = (-y + input_drive + recurrent_drive - gain_feedback) / self.tau_y
            du_dt = (-u + sigma_term + pool_term) / self.tau_u
            da_dt = (-a + u_plus + a * u_plus + self.alpha * du_dt) / self.tau_a
            dg_dt = (v_t * v_t - 1) / self.tau_g  # target = 1
            return np.concatenate([dy_dt, du_dt, da_dt, dg_dt])

    # --- Variant (b): Raw, adaptive target ---
    class VariantB(V1Dynamics):
        def _derivatives(self, state, z_t):
            NN, KK = self.v1.N, self.frame.K
            y = state[0:NN]; u = state[NN:2*NN]; a = state[2*NN:3*NN]; g = state[3*NN:3*NN+KK]
            u_plus = self.gaussian_rectify(u)
            y_plus = self.gaussian_rectify(y)
            a_plus = self.gaussian_rectify(a)
            sqrt_y_plus = np.sqrt(y_plus)
            v_t = self.frame.W.T @ y  # raw
            gain_feedback = self.frame.W @ (g * v_t)
            recurrent_drive = (1.0 / (1.0 + a_plus)) * (self.v1.W_yy @ sqrt_y_plus)
            input_drive = (self.beta * z_t) / 2
            sigma_term = self.sigma ** 2
            pool_term = self.v1.N_matrix @ (y_plus * (u_plus ** 2))
            dy_dt = (-y + input_drive + recurrent_drive - gain_feedback) / self.tau_y
            du_dt = (-u + sigma_term + pool_term) / self.tau_u
            da_dt = (-a + u_plus + a * u_plus + self.alpha * du_dt) / self.tau_a
            target = np.sum(y ** 2) / NN  # adaptive target
            dg_dt = (v_t * v_t - target) / self.tau_g
            return np.concatenate([dy_dt, du_dt, da_dt, dg_dt])

    # --- Variant (c): Centered + adaptive target ---
    class VariantC(V1Dynamics):
        def _derivatives(self, state, z_t):
            NN, KK = self.v1.N, self.frame.K
            y = state[0:NN]; u = state[NN:2*NN]; a = state[2*NN:3*NN]; g = state[3*NN:3*NN+KK]
            u_plus = self.gaussian_rectify(u)
            y_plus = self.gaussian_rectify(y)
            a_plus = self.gaussian_rectify(a)
            sqrt_y_plus = np.sqrt(y_plus)
            y_cen = y - np.mean(y)
            v_t = self.frame.W.T @ y_cen  # centered
            gain_feedback = self.frame.W @ (g * v_t)
            recurrent_drive = (1.0 / (1.0 + a_plus)) * (self.v1.W_yy @ sqrt_y_plus)
            input_drive = (self.beta * z_t) / 2
            sigma_term = self.sigma ** 2
            pool_term = self.v1.N_matrix @ (y_plus * (u_plus ** 2))
            dy_dt = (-y + input_drive + recurrent_drive - gain_feedback) / self.tau_y
            du_dt = (-u + sigma_term + pool_term) / self.tau_u
            da_dt = (-a + u_plus + a * u_plus + self.alpha * du_dt) / self.tau_a
            target = np.sum(y_cen ** 2) / NN  # adaptive target on centered signal
            dg_dt = (v_t * v_t - target) / self.tau_g
            return np.concatenate([dy_dt, du_dt, da_dt, dg_dt])

    variants = [
        ('(a) Centered, target=1', VariantA),
        ('(b) Raw, adaptive target', VariantB),
        ('(c) Centered + adaptive', VariantC),
    ]

    results = {}
    for label, cls in variants:
        print(f"\n  Running variant: {label}")
        eng = cls(tunings, frame, dt=0.05)
        _, gains = eng.run_simulation(inputs)

        # Collect mean(v_t^2) trace via a fresh run
        st = np.zeros(3 * N + K)
        vtsq_trace = np.zeros(n_steps)
        target_trace = np.zeros(n_steps)

        for t in range(n_steps):
            z_t = inputs[:, t]
            y = st[0:N]
            if 'Centered' in label:
                y_proj = y - np.mean(y)
            else:
                y_proj = y
            v = frame.W.T @ y_proj
            vtsq_trace[t] = np.mean(v ** 2)
            if 'adaptive' in label:
                target_trace[t] = np.sum(y_proj ** 2) / N
            else:
                target_trace[t] = 1.0

            k1 = eng._derivatives(st, z_t)
            k2 = eng._derivatives(st + 0.5 * eng.dt * k1, z_t)
            k3 = eng._derivatives(st + 0.5 * eng.dt * k2, z_t)
            k4 = eng._derivatives(st + eng.dt * k3, z_t)
            st += (eng.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            st[3 * N:3 * N + K] = np.maximum(st[3 * N:3 * N + K], 0.0)

        results[label] = {
            'gains': gains,
            'vtsq': vtsq_trace,
            'target': target_trace,
        }

    # --- Plot gains ---
    subset_k = np.linspace(0, K - 1, 10, dtype=int)
    fig, axes = plt.subplots(2, 3, figsize=(18, 8), sharex=True)

    for col, (label, _) in enumerate(variants):
        res = results[label]
        axes[0, col].plot(res['gains'][subset_k, :].T, alpha=0.6)
        axes[0, col].set_title(f"Gains — {label}")
        axes[0, col].grid(True, alpha=0.3)
        if col == 0:
            axes[0, col].set_ylabel("g (10 subset)")

        axes[1, col].plot(res['vtsq'], color='teal', lw=1.5, label='mean(v_t^2)')
        axes[1, col].plot(res['target'], color='red', ls='--', lw=1.5, label='target')
        axes[1, col].set_title(f"v_t^2 vs target — {label}")
        axes[1, col].set_xlabel("Time step")
        axes[1, col].legend(fontsize=8)
        axes[1, col].grid(True, alpha=0.3)
        if col == 0:
            axes[1, col].set_ylabel("value")

        for ax_row in axes:
            t_cursor = 0
            for r in DEFAULT_REGIMES:
                t_cursor += r['n_steps']
                ax_row[col].axvline(t_cursor, color='gray', ls='--', alpha=0.4)

    plt.suptitle("DIAGNOSTIC 11: Gain Rule Variants")
    plt.tight_layout()
    plt.show()

    for label, _ in variants:
        res = results[label]
        g_final = res['gains'][:, -1]
        stabilized = np.mean(g_final) > 0.001
        print(f"\n  {label}:")
        print(f"    Final gain range: [{np.min(g_final):.6f}, {np.max(g_final):.6f}]")
        print(f"    Final mean(v_t^2): {res['vtsq'][-1]:.6f}")
        print(f"    Gains stabilized: {'YES' if stabilized else 'NO'}")


# ────────────────────────────────────────────────────────────────────
# Main entry point
# ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading shared resources...")
    tunings = V1Tunings(N=N_NEURONS)
    frame = Frame(csv_path=CSV_PATH)
    stim_gen = StimulusGenerator(N=N_NEURONS)

    # ── Diagnostic 1: Gaussian Rectification Baseline ──
    '''diagnostic_1a(tunings)
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
'''
    # ── Diagnostic 5: Gain Adaptation ──
    y_ss = diagnostic_7(tunings, frame, stim_gen)
    diagnostic_8(tunings, frame, y_ss)
    diagnostic_9(tunings, frame, stim_gen)
    diagnostic_10(tunings, frame, stim_gen)
    diagnostic_11(tunings, frame, stim_gen)

    print("\n" + "=" * 60)
    print("All diagnostics complete.")
    print("=" * 60)
