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
from tunings_whiten import V1Tunings
from stimuli_whiten import StimulusGenerator
from simulation_whiten import Frame, V1Dynamics
import matplotlib.pyplot as plt

# ---- Parameters ----
N = 169                  # Number of primary neurons
N_BINS = 13              # Aggregation bins for visualization
STREAM_LENGTH = 10140     # Length of adaptation stream (steps)
PROBE_STEPS = 100        # Steps to settle for each probe stimulus
PROBE_RES = 180          # Resolution of tuning curve probe (number of angles)

np.random.seed(20)

def gaussian_rectify(y, threshold=0.5, sigma=0.25, r_max=1.0):
    return 0.5 * (1 + erf((y - threshold) / (sigma * np.sqrt(2)))) * r_max

def run_probe(frame, tunings, fixed_gains, probe_angles, frozen_u=None, frozen_a=None):
    """
    Measures tuning curves by simulating the network response to specific 
    probe orientations while holding gains constant. u, and a are taken from 
    their last values and then adapted.
    """
    N, K = frame.dim, frame.K
    n_probes = len(probe_angles)
    tuning_curves = np.zeros((N, n_probes))
    
    W_yy = tunings.W_yy
    
    dt = 0.05
    tau_y = 1.0
    tau_u = 2.0
    tau_a = 5.0
    beta = 1.0
    sigma_const = 0.05
    
    for i, angle in enumerate(probe_angles):
        
        # y must start at a baseline to settle to the new probe stimulus
        scale = 0.1
        y = scale*np.ones(N) # 
        
        # Let u and a freely adapt from their most recent state
        u = np.copy(frozen_u) if frozen_u is not None else np.zeros(N)
        a = np.copy(frozen_a) if frozen_a is not None else np.zeros(N)
        
        # 1. Construct Input for this probe angle
        diff = np.abs(tunings.theta - angle)
        diff = np.minimum(diff,  np.pi - diff)
        z_t = np.exp(- (diff ** 2) / (2 * (np.pi/8) ** 2)) 
        
        # 2. Settle to steady state
        for _ in range(PROBE_STEPS):
            # Rectifications
            u_plus = gaussian_rectify(u)
            y_plus = gaussian_rectify(y)
            a_plus = gaussian_rectify(a)
            sqrt_y_plus = np.sqrt(y_plus)
            
            # Circuit Inputs
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
            da = (-a + u_plus + a*u_plus) / tau_a 
            
            y += dt * dy
            u += dt * du
            a += dt * da
        
        # Record steady state firing rate
        tuning_curves[:, i] = gaussian_rectify(y)
        
    return tuning_curves

def get_binned_curves(tuning_curves, neuron_preferences, probe_angles, n_bins=13):
    """
    Aggregates individual neuron tuning curves into N_BINS groups based on 
    their preferred orientation.
    """
    # Calculate the exact mathematical space between your 169 neurons
    N_neurons = len(neuron_preferences)
    discrete_step = np.pi / N_neurons
    
    # Shift the bin edges left by half a step to avoid boundary collisions
    bin_edges = np.linspace(0, np.pi, n_bins + 1) - (discrete_step / 2)
    
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
    # 1. Calculate the norm of each row, keeping the 2D shape for broadcasting
    row_norms = np.linalg.norm(frame.W, axis=1, keepdims=True)
    
    # 2. Find the average scale of the original matrix
    mean_norm = np.mean(row_norms)
    
    # 3. Equalize the rows, but multiply by the mean_norm to keep the original scale
    frame.W = (frame.W / row_norms) * mean_norm

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
    
    # 1. Non-Adaptive Uniform
    engine_org_uni = V1Dynamics(tunings, frame, adaptive=False)
    org_uniform_rates, _, u_hist_org_uni, a_hist_org_uni = engine_org_uni.run_simulation(seq_uni)
    
    results['org_uni'] = run_probe(frame, tunings, fixed_gains=None, probe_angles=probe_angles,
                                   frozen_u=u_hist_org_uni[:, -1], frozen_a=a_hist_org_uni[:, -1])
                                   
    # 2. Non-Adaptive Biased
    engine_org_bias = V1Dynamics(tunings, frame, adaptive=False)
    org_bias_rates, _, u_hist_org_bias, a_hist_org_bias = engine_org_bias.run_simulation(seq_bias)
    
    results['org_bias'] = run_probe(frame, tunings, fixed_gains=None, probe_angles=probe_angles,
                                    frozen_u=u_hist_org_bias[:, -1], frozen_a=a_hist_org_bias[:, -1])
    
    # --- SCENARIO B: Adaptive ORGaNICs ---
    print("\n--- Running Adaptive Models ---")
    
    # 1. Adapt to Uniform
    print("Adapting to Uniform Ensemble...")
    engine_uni = V1Dynamics(tunings, frame, adaptive=True)
    
    # Grab the histories
    adapt_uniform_rates, gains_hist_uni, u_hist_uni, a_hist_uni = engine_uni.run_simulation(seq_uni)
    
    # Grab the final state for all three variables
    final_gains_uni = gains_hist_uni[:, -1] 
    final_u_uni = u_hist_uni[:, -1]
    final_a_uni = a_hist_uni[:, -1]
    
    # 2. Probe Uniform State 
    print("Probing Uniform State...")
    results['adp_uni'] = run_probe(frame, tunings, final_gains_uni, probe_angles, 
                                   frozen_u=final_u_uni, frozen_a=final_a_uni)
    
    # 3. Adapt to Biased
    print("Adapting to Biased Ensemble...")
    engine_bias = V1Dynamics(tunings, frame, adaptive=True)
    adapt_biased_rates, gains_hist_bias, u_hist_bias, a_hist_bias = engine_bias.run_simulation(seq_bias)
    
    # Grab the final column for all three variables again
    final_gains_bias = gains_hist_bias[:, -1] 
    final_u_bias = u_hist_bias[:, -1]
    final_a_bias = a_hist_bias[:, -1]
    
    # 4. Probe Biased State
    print("Probing Biased State...")
    results['adp_bias'] = run_probe(frame, tunings, final_gains_bias, probe_angles,
                                    frozen_u=final_u_bias, frozen_a=final_a_bias)

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
    discrete_step = 180 / N 
    
    # Shift the bin edges left by half a step to avoid boundary collisions
    bins_hist = np.linspace(0, 180, N_BINS + 1) - (discrete_step / 2)
    
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
    # FIGURE 2: Average Steady-State Response per Orientation Bin
    # =================================================================
    # Uses 10140-step simulations and measure avg firing rates
    # directly from the running simulation (last 4,000 steps).

    print("\n" + "=" * 50)
    print("  FIGURE 2: Average Response per Orientation Bin")
    print("=" * 50)

    AVG_WINDOW = 4000

    discrete_step_rad = np.pi / N
    bin_edges = np.linspace(0, np.pi, N_BINS + 1) - (discrete_step_rad / 2)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_centers_deg = bin_centers * 180 / np.pi
    neuron_bin_idx = np.digitize(tunings.theta, bin_edges) - 1
    neuron_bin_idx = np.clip(neuron_bin_idx, 0, N_BINS - 1)

    def get_binned_activity(rates, window):
        """Average response per neuron over last `window` steps (only steady_state), averaged per bin."""
        
        duration = 20
        keep = 5 # keeps the last 5 responses to each stimuli to only count the steady state responses.
        
        steady_rates = rates[:, -window:]
        n_time_steps = steady_rates.shape[1]
        
        # 1. Temporal Mask: Cycle of length `duration`, keep the last `keep` steps
        time_mask = (np.arange(n_time_steps) % duration) >= (duration - keep)
        
        # 2. Get steady state rates and average across time to get mean per neuron
        means = np.mean(steady_rates[:, time_mask], axis=1) 
        
        binned = np.zeros(N_BINS)
        for b in range(N_BINS):
            # 3. Spatial Mask: applied to the neurons
            bin_mask = neuron_bin_idx == b
            if bin_mask.any():
                binned[b] = np.mean(means[bin_mask])
                
        return binned

    # 1. Adaptive + Uniform
    print("\nAdaptive + Uniform (10k steps)...")
    peaks_adp_uni = get_binned_activity(adapt_uniform_rates, AVG_WINDOW)
    del adapt_uniform_rates
    gc.collect()

    # 2. Adaptive + Biased
    print("Adaptive + Biased (10k steps)...")
    peaks_adp_bias = get_binned_activity(adapt_biased_rates, AVG_WINDOW)
    del adapt_biased_rates
    gc.collect()

    # 3. ORGaNICs + Biased
    print("ORGaNICs + Biased (10k steps)...")
    peaks_org_bias = get_binned_activity(org_bias_rates, AVG_WINDOW)
    del org_bias_rates
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
    axes2[0].set_ylabel("Average Response")
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

    fig2.suptitle("Average Steady State Response",
                  fontweight='bold', fontsize=13)
    plt.tight_layout()
    plt.show()