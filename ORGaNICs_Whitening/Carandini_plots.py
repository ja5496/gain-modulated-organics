"""
Carandini_plots.py

Replicates cat V1 adaptation experiments (e.g., Benucci et al.) using ORGaNICs.
Compares steady-state tuning curves after adaptation to Uniform vs. Biased ensembles.

Methodology:
1. "Adaptation Phase": Run the model on a long stream of stimuli to evolve gains.
2. "Probe Phase": Freeze gains and measure responses to a clean sweep of test orientations.
"""

import numpy as np
import matplotlib.pyplot as plt
import gc
from tqdm import tqdm
from scipy.special import erf

# Import from existing codebase
from tunings_whiten import V1Tunings
from stimuli_whiten import StimulusGenerator
from simulation_whiten import Frame, V1Dynamics

# ---- Parameters ----
N = 169                  # Number of primary neurons
N_BINS = 13              # Aggregation bins for visualization
STREAM_LENGTH = 8000     # Length of adaptation stream (steps)
PROBE_STEPS = 100        # Steps to settle for each probe stimulus
PROBE_RES = 36           # Resolution of tuning curve probe (number of angles)

np.random.seed(42)

def gaussian_rectify(y, threshold=0.5, sigma=0.25, r_max=1.0):
    return 0.5 * (1 + erf((y - threshold) / (sigma * np.sqrt(2)))) * r_max

def run_probe(frame, tunings, fixed_gains, probe_angles):
    """
    Measures tuning curves by simulating the network response to specific 
    probe orientations while holding gains CONSTANT.
    """
    N, K = frame.dim, frame.K
    n_probes = len(probe_angles)
    tuning_curves = np.zeros((N, n_probes))
    
    # Pre-compute recurrent weights to save time
    W_yy = tunings.W_yy
    
    # We implement a lightweight integration loop here to ensure 
    # gains remain absolutely frozen during probing.
    dt = 0.05
    tau_y = 1.0
    tau_u = 2.0
    tau_a = 5.0
    beta = 1.0
    sigma_const = 0.05
    
    # Reset state for the probe phase
    y = np.zeros(N)
    u = np.zeros(N)
    a = np.zeros(N)
    
    # print(f"  Probing {n_probes} orientations...", end="", flush=True)
    
    for i, angle in enumerate(probe_angles):
        # 1. Construct Input for this probe angle
        # We assume a standard contrast for the probe (e.g., 1.0)
        # Using the same tuning logic as StimulusGenerator
        diff = np.abs(tunings.theta - angle)
        diff = np.minimum(diff,  np.pi - diff)
        # Gaussian input profile
        z_t = np.exp(- (diff ** 2) / (2 * (np.pi/8) ** 2)) 
        
        # 2. Settle to steady state
        for _ in range(PROBE_STEPS):
            # Rectifications
            u_plus = gaussian_rectify(u)
            y_plus = gaussian_rectify(y)
            a_plus = gaussian_rectify(a)
            sqrt_y_plus = np.sqrt(y_plus)
            
            # Circuit Inputs
            # GAIN FEEDBACK (Frozen)
            v_t = frame.W.T @ y
            if fixed_gains is not None:
                gain_feedback = frame.W @ (fixed_gains * v_t)
            else:
                gain_feedback = 0.0
                
            recurrent_drive = (1.0 / (1.0 + a_plus)) * (W_yy @ sqrt_y_plus)
            input_drive = (beta * z_t) / 2
            
            # Derivatives
            pool_term = tunings.N_matrix @ (y_plus * (u_plus ** 2))
            
            dy = (-y + input_drive + recurrent_drive - gain_feedback) / tau_y
            du = (-u + (sigma_const**2) + pool_term) / tau_u
            da = (-a + u_plus + a*u_plus) / tau_a # alpha assumed 0 for probe
            
            y += dt * dy
            u += dt * du
            a += dt * da
        
        # Record steady state firing rate
        tuning_curves[:, i] = gaussian_rectify(y)
        
    # print(" Done.")
    return tuning_curves

def get_binned_curves(tuning_curves, neuron_preferences, probe_angles, n_bins=13):
    """
    Aggregates individual neuron tuning curves into N_BINS groups based on 
    their preferred orientation.
    """
    # Define bins
    bin_edges = np.linspace(0, np.pi, n_bins + 1)
    
    # Result container: (n_bins, n_probe_angles)
    binned_response = np.zeros((n_bins, len(probe_angles)))
    
    # Assign neurons to bins
    neuron_bin_indices = np.digitize(neuron_preferences, bin_edges) - 1
    neuron_bin_indices = np.clip(neuron_bin_indices, 0, n_bins - 1)
    
    for b in range(n_bins):
        mask = neuron_bin_indices == b
        if np.any(mask):
            # Average the curves of all neurons in this bin
            binned_response[b, :] = np.mean(tuning_curves[mask, :], axis=0)
            
    return binned_response

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    
    # 1. Initialize
    print("Initializing...")
    tunings = V1Tunings(N=N)
    frame = Frame(csv_path="Frames/N169_Frame.csv")
    
    # Initialize Generator with the desired stream length
    stim_gen = StimulusGenerator(N=N, K=N, stream_length=STREAM_LENGTH)
    
    # Define the "Adaptor" location (matches logic in stimuli_whiten.py: K // 2 + 1)
    adaptor_idx = N // 2
    adaptor_rad = stim_gen.theta_inputs[adaptor_idx]
    adaptor_deg = adaptor_rad * 180 / np.pi
    
    # 2. Generate Adaptation Streams
    print("Generating adaptation streams...")
    # Uniform
    seq_uni = stim_gen.generate_input_ensembles(biased=False)
    # Biased
    seq_bias = stim_gen.generate_input_ensembles(biased=True)
    
    # Recover orientations for the histogram (argmax of input drive)
    # Since generate_input_ensembles doesn't return the angles, we infer them.
    uni_indices = np.argmax(seq_uni, axis=0)
    bias_indices = np.argmax(seq_bias, axis=0)
    
    hist_uni = stim_gen.theta_inputs[uni_indices] * 180/np.pi
    hist_bias = stim_gen.theta_inputs[bias_indices] * 180/np.pi

    # 3. Run Simulations & Probes
    
    # Define Probe Angles (Clean sweep 0 to 180)
    probe_angles = np.linspace(0, np.pi, PROBE_RES)
    probe_angles_deg = probe_angles * 180 / np.pi
    
    results = {}
    
    # --- SCENARIO A: ORGaNICs (Non-Adaptive) ---
    print("\n--- Running Non-Adaptive Models ---")
    # For non-adaptive, gains are effectively 0. 
    # The "adaptation" history doesn't change the weights, so we just probe directly with gains=0.
    tc_raw_control = run_probe(frame, tunings, fixed_gains=None, probe_angles=probe_angles)
    
    # For the sake of the plot structure, we assign this same curve to both conditions
    results['org_uni'] = tc_raw_control
    results['org_bias'] = tc_raw_control
    
    # --- SCENARIO B: Adaptive ORGaNICs ---
    print("\n--- Running Adaptive Models ---")
    
    # 1. Adapt to Uniform
    print("Adapting to Uniform Ensemble...")
    engine_uni = V1Dynamics(tunings, frame, adaptive=True)
    _, gains_hist_uni = engine_uni.run_simulation(seq_uni)
    final_gains_uni = gains_hist_uni[:, -1] # Extract final state
    
    # 2. Probe Uniform State
    print("Probing Uniform State...")
    results['adp_uni'] = run_probe(frame, tunings, final_gains_uni, probe_angles)
    
    # 3. Adapt to Biased
    print("Adapting to Biased Ensemble...")
    engine_bias = V1Dynamics(tunings, frame, adaptive=True)
    _, gains_hist_bias = engine_bias.run_simulation(seq_bias)
    final_gains_bias = gains_hist_bias[:, -1] # Extract final state
    
    # 4. Probe Biased State
    print("Probing Biased State...")
    results['adp_bias'] = run_probe(frame, tunings, final_gains_bias, probe_angles)

    # 4. Processing & Normalization
    print("\nProcessing data for plotting...")
    
    # Helper to bin and normalize
    def process_pair(tc_uni_raw, tc_bias_raw):
        # 1. Bin data
        binned_uni = get_binned_curves(tc_uni_raw, tunings.theta, probe_angles, N_BINS)
        binned_bias = get_binned_curves(tc_bias_raw, tunings.theta, probe_angles, N_BINS)
        
        # 2. Normalize based on UNIFORM response
        # We want Uniform Peak = 1, Uniform Min = 0
        glob_max = np.max(binned_uni)
        glob_min = np.min(binned_uni)
        
        norm_uni = (binned_uni - glob_min) / (glob_max - glob_min + 1e-9)
        norm_bias = (binned_bias - glob_min) / (glob_max - glob_min + 1e-9)
        
        return norm_uni, norm_bias

    # Process both rows
    row2_uni, row2_bias = process_pair(results['org_uni'], results['org_bias'])
    row3_uni, row3_bias = process_pair(results['adp_uni'], results['adp_bias'])

    # 5. Plotting
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), gridspec_kw={'height_ratios': [0.8, 1.5, 1.5]})
    
    # Setup x-axis relative to adaptor and wrap to [-90, 90)
    x_axis = (probe_angles_deg - adaptor_deg + 90) % 180 - 90

    # Sort the axis so matplotlib doesn't draw lines across the chart
    sort_idx = np.argsort(x_axis)
    x_axis_sorted = x_axis[sort_idx]
    
    # Colors
    blue_colors = plt.cm.Blues(np.linspace(0.4, 1.0, N_BINS))
    
    # --- ROW 1: Histograms ---
    bins_hist = np.linspace(0, 180, N_BINS + 1)
    
    # Uniform Hist
    axes[0, 0].hist(hist_uni, bins=bins_hist, color='black', rwidth=0.9)
    axes[0, 0].set_title("Uniform Ensemble", fontweight='bold')
    axes[0, 0].set_ylabel("Count")
    
    # Biased Hist
    axes[0, 1].hist(hist_bias, bins=bins_hist, color='black', rwidth=0.9)
    axes[0, 1].set_title("Biased Ensemble", fontweight='bold')
    
    # Clean up Row 1
    for ax in axes[0]:
        ax.axvline(adaptor_deg, color='red', linestyle='--', alpha=0.5)
        ax.set_xlim(0, 180)
        ax.tick_params(labelbottom=False)

    # --- ROW 2: Non-Adaptive ---
    for i in range(N_BINS):
        # Apply sort_idx to the y-data as well
        axes[1, 0].plot(x_axis_sorted, row2_uni[i][sort_idx], color=blue_colors[i], linewidth=1.5)
        axes[1, 1].plot(x_axis_sorted, row2_bias[i][sort_idx], color=blue_colors[i], linewidth=1.5)
        
    axes[1, 0].set_ylabel("Non-Adaptive\nNormalized Response", fontweight='bold')
    axes[1, 0].set_title("Response (Control)")
    
    # --- ROW 3: Adaptive ---
    for i in range(N_BINS):
        axes[2, 0].plot(x_axis_sorted, row3_uni[i][sort_idx], color=blue_colors[i], linewidth=1.5)
        axes[2, 1].plot(x_axis_sorted, row3_bias[i][sort_idx], color=blue_colors[i], linewidth=1.5)
        
    axes[2, 0].set_ylabel("Adaptive\nNormalized Response", fontweight='bold')
    axes[2, 0].set_title("Response (Adapted)")

    # --- Global Formatting ---
    for r in [1, 2]:
        for c in [0, 1]:
            ax = axes[r, c]
            ax.axvline(0, color='red', linestyle='--', alpha=0.5, label='Adaptor')
            ax.set_xlim(-90, 90) # Centered view
            ax.grid(True, alpha=0.3)
            
            if r == 2:
                ax.set_xlabel("Orientation Relative to Adaptor (°)")
    
    plt.tight_layout()
    plt.show()

    # =================================================================
    # FIGURE 2: Average Peak Response per Orientation Bin
    # =================================================================
    # Uses longer 10,000-step simulations and measures peak firing rates
    # directly from the running simulation (last 2,000 steps).
    print("\n" + "=" * 50)
    print("  FIGURE 2: Average Peak Response per Orientation Bin")
    print("=" * 50)

    STREAM_LENGTH_2 = 14000
    AVG_WINDOW = 2000

    stim_gen_2 = StimulusGenerator(N=N, K=N, stream_length=STREAM_LENGTH_2)
    seq_uni_2 = stim_gen_2.generate_input_ensembles(biased=False)
    seq_bias_2 = stim_gen_2.generate_input_ensembles(biased=True)

    # Bin setup
    bin_edges = np.linspace(0, np.pi, N_BINS + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_centers_deg = bin_centers * 180 / np.pi
    neuron_bin_idx = np.digitize(tunings.theta, bin_edges) - 1
    neuron_bin_idx = np.clip(neuron_bin_idx, 0, N_BINS - 1)

    def get_binned_peaks(rates, window):
        """Peak response per neuron over last `window` steps, averaged per bin."""
        peaks = np.max(rates[:, -window:], axis=1)
        binned = np.zeros(N_BINS)
        for b in range(N_BINS):
            mask = neuron_bin_idx == b
            if mask.any():
                binned[b] = np.mean(peaks[mask])
        return binned

    # 1. Adaptive + Uniform
    print("\nAdaptive + Uniform (10k steps)...")
    engine = V1Dynamics(tunings, frame, adaptive=True)
    rates, _ = engine.run_simulation(seq_uni_2)
    peaks_adp_uni = get_binned_peaks(rates, AVG_WINDOW)
    del rates, engine
    gc.collect()

    # 2. Adaptive + Biased
    print("Adaptive + Biased (10k steps)...")
    engine = V1Dynamics(tunings, frame, adaptive=True)
    rates, _ = engine.run_simulation(seq_bias_2)
    peaks_adp_bias = get_binned_peaks(rates, AVG_WINDOW)
    del rates, engine
    gc.collect()

    # 3. ORGaNICs + Biased
    print("ORGaNICs + Biased (10k steps)...")
    engine = V1Dynamics(tunings, frame, adaptive=False)
    rates, _ = engine.run_simulation(seq_bias_2)
    peaks_org_bias = get_binned_peaks(rates, AVG_WINDOW)
    del rates, engine
    gc.collect()

    # Plot
    # Wrap and sort for Figure 2
    x_peak = (bin_centers_deg - adaptor_deg + 90) % 180 - 90
    sort_idx_2 = np.argsort(x_peak)
    x_peak_sorted = x_peak[sort_idx_2]
    
    fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    # Left: Adaptive + Uniform
    axes2[0].plot(x_peak_sorted, peaks_adp_uni[sort_idx_2], 'o-', color='steelblue',
                  linewidth=2, markersize=5, label='Adaptive')
    axes2[0].set_title("Adaptive: Uniform Ensemble", fontweight='bold')
    axes2[0].set_ylabel("Average Peak Response")
    axes2[0].set_xlabel("Orientation Relative to Adaptor (°)")
    axes2[0].axvline(0, color='red', linestyle='--', alpha=0.5)
    axes2[0].set_xlim(-90, 90)
    axes2[0].grid(True, alpha=0.3)

    # Right: Biased — Adaptive vs ORGaNICs
    axes2[1].plot(x_peak_sorted, peaks_adp_bias[sort_idx_2], 'o-', color='steelblue',
                  linewidth=2, markersize=5, label='Adaptive')
    axes2[1].plot(x_peak_sorted, peaks_org_bias[sort_idx_2], 's--', color='coral',
                  linewidth=2, markersize=5, label='ORGaNICs')
    axes2[1].set_title("Biased Ensemble", fontweight='bold')
    axes2[1].set_xlabel("Orientation Relative to Adaptor (°)")
    axes2[1].axvline(0, color='red', linestyle='--', alpha=0.5)
    axes2[1].set_xlim(-90, 90)
    axes2[1].legend()
    axes2[1].grid(True, alpha=0.3)

    fig2.suptitle("Average Peak Response (last 2000 of 14000 steps)",
                  fontweight='bold', fontsize=13)
    plt.tight_layout()
    plt.show()