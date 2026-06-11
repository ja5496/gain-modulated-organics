"""
Carandini_plots.py

Replicates cat V1 adaptation experiments (e.g., Benucci et al.) using ORGaNICs + a 
whitening objective. Compares steady-state tuning curves after adaptation to Uniform 
vs. Biased ensembles.

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

# ---- Parameters ----
N = 169                  # Number of primary neurons
N_BINS = 13              # Aggregation bins for visualization
STREAM_LENGTH = 10140    # Length of adaptation stream (steps)
PROBE_STEPS = 100        # Steps to settle for each probe stimulus
PROBE_RES = 360          # Resolution of tuning curve probe (number of angles)
Z_SPONT = 0.1            # Tonic LGN background drive (tune to control spontaneous rate;
                         # ~0.16 of max firing at Z_SPONT=0.3 with threshold=0.5, sigma=0.2)

np.random.seed(20)

def gaussian_rectify(y, threshold=0.6, sigma=0.35, r_max=1.0):
    return 0.5 * (1 + erf((y - threshold) / (sigma * np.sqrt(2)))) * r_max

def run_probe(frame, tunings, stim_gen, fixed_gains, probe_angles, frozen_u=None, frozen_a=None,
              frozen_avg_z=None, frozen_v=None, z_spont=0.1):
    """
    Measures tuning curves by simulating the network response to specific
    probe orientations while holding gains constant. u, and a are taken from
    their last values and then adapted.
    """
    N, K = frame.dim, frame.K
    n_probes = len(probe_angles)
    tuning_curves = np.zeros((N, n_probes))

    W_yy = tunings.W_yy

    dt = 0.1
    tau_y = 0.4
    tau_u = 0.8
    tau_a = 2.0
    tau_v = 50.0
    # Freeze beta at the end-of-adaptation state; fall back to 1.0 if no avg_z was tracked
    beta = 1 - 0.2 * frozen_avg_z if frozen_avg_z is not None else 1.0
    sigma = 0.1

    for i, angle in enumerate(probe_angles):

        # Start y near the spontaneous resting state driven by z_spont
        y = np.zeros(N)

        # Let u and a freely adapt from their most recent state
        u = np.copy(frozen_u)
        a = np.copy(frozen_a)
        v = frozen_v.copy() if frozen_v is not None else np.zeros(K)

        # Construct probe stimulus identically to generate_input_ensembles 
        delta = stim_gen.theta_inputs - angle
        delta = (delta + np.pi/2) % np.pi - np.pi/2  # same wrapping as StimulusGenerator
        z_t = np.exp(-delta**2 / (2 * stim_gen.tuning_width**2)) #+ 0.3
        #z_t = np.exp(stim_gen.tuning_width * np.cos(2 * delta)) # RAISED COSINE
        contrast = 0.1
        scale = 15 # COEFFICIENT OF ~15 ACHIEVES CORRECT SATURATION FOR CONTRAST OF 1
        z_t = contrast * scale * z_t / np.max(z_t)

        # 2. Settle to steady state
        for step in range(PROBE_STEPS):
            # Rectifications
            u_plus = gaussian_rectify(u)
            y_plus = gaussian_rectify(y)
            a_plus = gaussian_rectify(a)
            sqrt_y_plus = np.sqrt(y_plus)

            # Circuit Inputs
            if fixed_gains is not None:
                gain_feedback = frame.W @ (fixed_gains * v)
            else:
                gain_feedback = 0.0

            recurrent_drive = (1.0 / (1.0 + a_plus)) * (W_yy @ sqrt_y_plus)
            input_drive = (beta * z_t) / 2

            # Derivatives
            pool_term = tunings.N_matrix @ (y_plus * (u_plus ** 2))

            dy = (-y + input_drive + recurrent_drive - gain_feedback) / tau_y
            du = (-u + (sigma / 2)**2 + pool_term) / tau_u
            da = (-a + u_plus + a*u_plus) / tau_a
            dv = (-v + frame.W.T @ y) / tau_v

            y += dt * dy
            u += dt * du
            a += dt * da
            v += dt * dv

            # --- DIAGNOSTIC: print drive magnitudes on first angle, last step ---
            if i == 0 and step == PROBE_STEPS - 1 and fixed_gains is not None:
                print(f"  [DIAG probe i=0 final step]")
                print(f"    mean |input_drive|    = {np.mean(np.abs(input_drive)):.4f}")
                print(f"    mean |recurrent_drive|= {np.mean(np.abs(recurrent_drive)):.4f}")
                print(f"    mean |gain_feedback|  = {np.mean(np.abs(gain_feedback)):.4f}")
                print(f"    max  |gain_feedback|  = {np.max(np.abs(gain_feedback)):.4f}")
                print(f"    mean y = {np.mean(y):.4f},  max y = {np.max(y):.4f},  min y = {np.min(y):.4f}")

        # Record steady state firing rate
        tuning_curves[:, i] = gaussian_rectify(y)

        # --- DIAGNOSTIC: print raw tuning curve spread after first few angles ---
        if i < 3 and fixed_gains is not None:
            print(f"  [DIAG probe i={i}] mean firing rate = {np.mean(tuning_curves[:, i]):.4f},  "
                  f"max = {np.max(tuning_curves[:, i]):.4f}")

    return tuning_curves

def get_binned_curves(tuning_curves, neuron_preferences, probe_angles, n_bins=13):
    """
    Aggregates individual neuron tuning curves into N_BINS groups based on 
    their preferred orientation.
    """
    N_neurons = len(neuron_preferences)
    discrete_step = np.pi / N_neurons
    
    bin_edges = np.linspace(0, np.pi, n_bins + 1) - (discrete_step / 2)
    binned_response = np.zeros((n_bins, len(probe_angles)))
    
    neuron_bin_indices = np.digitize(neuron_preferences, bin_edges) - 1
    neuron_bin_indices = np.clip(neuron_bin_indices, 0, n_bins - 1)
    
    for b in range(n_bins):
        mask = neuron_bin_indices == b
        if np.any(mask):
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
    stim_gen = StimulusGenerator(N=N, num_angles=N, stream_length=STREAM_LENGTH, contrast=0.1)
    
    adaptor_idx = N // 2
    adaptor_rad = stim_gen.theta_inputs[adaptor_idx]
    adaptor_deg = adaptor_rad * 180 / np.pi
    
    # 2. Generate Adaptation Streams
    print("Generating adaptation streams...")
    seq_uni = stim_gen.generate_input_ensembles(biased=False)
    seq_bias = stim_gen.generate_input_ensembles(biased=True)
    
    uni_indices = np.argmax(seq_uni, axis=0)
    bias_indices = np.argmax(seq_bias, axis=0)
    
    hist_uni = stim_gen.theta_inputs[uni_indices] * 180/np.pi
    hist_bias = stim_gen.theta_inputs[bias_indices] * 180/np.pi

    # 3. Run Simulations & Probes
    probe_angles = np.linspace(0, np.pi, PROBE_RES)
    probe_angles_deg = probe_angles * 180 / np.pi
    
    results = {}
    
    # --- SCENARIO A: Non-Adaptive ORGaNICs ---
    print("\n--- Running Non-Adaptive Models ---")
    
    engine_non_adapt = V1Dynamics(tunings, frame, adaptive=False, input_adaptive=False)

    org_uniform_rates, _, u_hist_org_uni, a_hist_org_uni, _, _, _ = engine_non_adapt.run_simulation(seq_uni)
    results['org_uni'] = run_probe(frame, tunings, stim_gen, fixed_gains=None, probe_angles=probe_angles,
                                   frozen_u=u_hist_org_uni[:, -1], frozen_a=a_hist_org_uni[:, -1])

    org_bias_rates, _, u_hist_org_bias, a_hist_org_bias, _, _, _ = engine_non_adapt.run_simulation(seq_bias)
    results['org_bias'] = run_probe(frame, tunings, stim_gen, fixed_gains=None, probe_angles=probe_angles,
                                    frozen_u=u_hist_org_bias[:, -1], frozen_a=a_hist_org_bias[:, -1])

    # --- SCENARIO B: Adaptive ORGaNICs ---
    print("\n--- Running Adaptive Models ---")

    print("Adapting to Uniform Ensemble...")
    engine_adapt = V1Dynamics(tunings, frame, adaptive=True, input_adaptive=False)
    adapt_uniform_rates, gains_hist_uni, u_hist_uni, a_hist_uni, v_hist_uni, avg_z_hist_uni, avg_vsq_hist_uni = engine_adapt.run_simulation(seq_uni)

    final_gains_uni = gains_hist_uni[:, -1]
    final_u_uni = u_hist_uni[:, -1]
    final_a_uni = a_hist_uni[:, -1]

    print("Probing Uniform State...")
    print(f"  [DIAG] final_gains_uni: mean={np.mean(final_gains_uni):.4f}, "
          f"max={np.max(final_gains_uni):.4f}, std={np.std(final_gains_uni):.4f}")
    results['adp_uni'] = run_probe(frame, tunings, stim_gen, final_gains_uni, probe_angles,
                                   frozen_u=final_u_uni, frozen_a=final_a_uni,
                                   frozen_avg_z=avg_z_hist_uni[:, -1],
                                   frozen_v=v_hist_uni[:, -1])

    print("Adapting to Biased Ensemble...")
    adapt_biased_rates, gains_hist_bias, u_hist_bias, a_hist_bias, v_hist_bias, avg_z_hist_bias, avg_vsq_hist_bias = engine_adapt.run_simulation(seq_bias)

    final_gains_bias = gains_hist_bias[:, -1]
    final_u_bias = u_hist_bias[:, -1]
    final_a_bias = a_hist_bias[:, -1]

    print("Probing Biased State...")
    print(f"  [DIAG] final_gains_bias: mean={np.mean(final_gains_bias):.4f}, "
          f"max={np.max(final_gains_bias):.4f}, std={np.std(final_gains_bias):.4f}")
    results['adp_bias'] = run_probe(frame, tunings, stim_gen, final_gains_bias, probe_angles,
                                    frozen_u=final_u_bias, frozen_a=final_a_bias,
                                    frozen_avg_z=avg_z_hist_bias[:, -1],
                                    frozen_v=v_hist_bias[:, -1])

    # 4. Processing & Normalization
    print("\nProcessing data for plotting...")
    
    def process_pair(tc_uni_raw, tc_bias_raw):
        binned_uni = get_binned_curves(tc_uni_raw, tunings.theta, probe_angles, N_BINS)
        binned_bias = get_binned_curves(tc_bias_raw, tunings.theta, probe_angles, N_BINS)
        
        bin_max = np.max(binned_uni, axis=1, keepdims=True)
        bin_min = np.min(binned_uni, axis=1, keepdims=True)

        norm_uni = (binned_uni - bin_min) / (bin_max - bin_min + 1e-9)
        norm_bias = (binned_bias - bin_min) / (bin_max - bin_min + 1e-9)
        
        return norm_uni, norm_bias

    row2_uni, row2_bias = process_pair(results['org_uni'], results['org_bias'])
    row3_uni, row3_bias = process_pair(results['adp_uni'], results['adp_bias'])

    # 5. Plotting (FIGURE 1)
    fig, axes = plt.subplots(3, 2, figsize=(10, 9), sharey='row', gridspec_kw={'height_ratios': [0.8, 1.0, 1.2]})
    
    x_axis = (probe_angles_deg - adaptor_deg + 90) % 180 - 90
    sort_idx = np.argsort(x_axis)
    x_axis_sorted = x_axis[sort_idx]
    
    blue_colors = plt.cm.Blues(np.linspace(0.2, 1.0, N_BINS))
    
    discrete_step = 180 / N 
    bins_hist = np.linspace(0, 180, N_BINS + 1) - (discrete_step / 2)
    weights_uni = np.ones_like(hist_uni) / len(hist_uni)
    weights_bias = np.ones_like(hist_bias) / len(hist_bias)

    axes[0, 0].hist(hist_uni, bins=bins_hist, weights=weights_uni, color='black', rwidth=0.9)
    axes[0, 0].set_title("Uniform Ensemble", fontweight='bold', fontsize=18)
    axes[0, 0].set_ylabel("Probability", fontsize=18)
    
    axes[0, 1].hist(hist_bias, bins=bins_hist, weights=weights_bias, color='black', rwidth=0.9)
    axes[0, 1].set_title("Biased Ensemble", fontweight='bold', fontsize=18)
    
    for ax in axes[0]:
        ax.set_xlim(0, 180)
        ax.tick_params(labelbottom=False)

    for i in range(N_BINS):
        axes[1, 0].plot(x_axis_sorted, row2_uni[i][sort_idx], color=blue_colors[i], linewidth=2.0)
        axes[1, 1].plot(x_axis_sorted, row2_bias[i][sort_idx], color=blue_colors[i], linewidth=2.0)

    axes[1, 0].set_ylabel("Non-Adaptive\nORGaNICs Response", fontsize=18)

    for i in range(N_BINS):
        axes[2, 0].plot(x_axis_sorted, row3_uni[i][sort_idx], color=blue_colors[i], linewidth=2.0)
        axes[2, 1].plot(x_axis_sorted, row3_bias[i][sort_idx], color=blue_colors[i], linewidth=2.0)
        
    axes[2, 0].set_ylabel("Adaptive ORGaNICs\n Response", fontsize=18)

    for r in [1, 2]:
        for c in [0, 1]:
            ax = axes[r, c]
            ax.set_xlim(-90, 90) 
            ax.grid(False, alpha=0.3)
            
            if r == 2:
                ax.set_xlabel("Stimulus Orientation(°)", fontsize = 18)
    
    plt.tight_layout()
    plt.show()

    # =================================================================
    # FIGURE 2: Biased-Ensemble Average Response (from Tuning Curves)
    # =================================================================

    # Build biased ensemble stream:
    # - one of each non-adaptor orientation (N-1 = 168)
    # - adaptor repeated (N-1)//2 = 84 times so it is exactly 1/3 of the stream
    n_non_adaptor = N - 1
    n_adaptor_reps = n_non_adaptor // 2
    non_adaptor_angles = np.concatenate([
        stim_gen.theta_inputs[:adaptor_idx],
        stim_gen.theta_inputs[adaptor_idx + 1:]
    ])
    biased_stream = np.concatenate([
        non_adaptor_angles,
        np.full(n_adaptor_reps, adaptor_rad)
    ])

    # Map each stream orientation to the nearest probe-angle index
    probe_idx_for_stream = np.argmin(
        np.abs(biased_stream[:, None] - probe_angles[None, :]), axis=1
    )

    # Average tuning-curve response across the stream for each neuron
    avg_org2 = np.mean(results['org_bias'][:, probe_idx_for_stream], axis=1)
    avg_adp2 = np.mean(results['adp_bias'][:, probe_idx_for_stream], axis=1)

    # Normalize so the population mean sits at y = 1
    norm_avg_org2 = avg_org2 / np.mean(avg_org2)
    norm_avg_adp2 = avg_adp2 / np.mean(avg_adp2)

    # Sort neurons by preferred orientation relative to the adaptor
    neuron_prefs_deg = tunings.theta * 180 / np.pi
    x_neuron = (neuron_prefs_deg - adaptor_deg + 90) % 180 - 90
    sort_neuron_idx = np.argsort(x_neuron)

    FIG2_DARK_ORANGE = '#800020'
    FIG2_DARK_GREY   = '#333333'

    fig2, ax2 = plt.subplots(1, 1, figsize=(6, 4))

    ax2.axhline(1, color='grey', linestyle='--', linewidth=1.2, zorder=1)
    ax2.plot(x_neuron[sort_neuron_idx], norm_avg_org2[sort_neuron_idx],
             color=FIG2_DARK_GREY, linewidth=2.5, label='Non-Adaptive ORGaNICs')
    ax2.plot(x_neuron[sort_neuron_idx], norm_avg_adp2[sort_neuron_idx],
             color=FIG2_DARK_ORANGE, linewidth=2.5, label='Adaptive ORGaNICs')

    ax2.set_title("Average Activity (Biased Ensemble)", fontweight='bold', fontsize=18)
    ax2.set_ylabel("Normalized Response", fontweight='bold', fontsize=14)
    ax2.set_xlabel("Neuron Preference (deg)", fontweight='bold', fontsize=14)
    ax2.set_xlim(-90, 90)
    ax2.legend(loc='upper right')
    ax2.grid(False)

    for spine in ax2.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(2.5)

    plt.tight_layout()
    plt.show()

    # Free simulation rate arrays before Figure 3
    del org_uniform_rates, org_bias_rates, adapt_uniform_rates, adapt_biased_rates
    gc.collect()

    # =================================================================
    # FIGURE 3: Subset of Gain Dynamics (Last 1000 Steps)
    # =================================================================

    LAST_STEPS = 10000
    N_GAIN_SUBSET = 50
    DARK_ORANGE = '#CC5500'
    DARK_GREY = '#333333'
    DARK_BLUE = '#00008B'

    gain_subset_idx = np.random.choice(N, N_GAIN_SUBSET, replace=False)

    gains_uni_sub  = gains_hist_uni[gain_subset_idx, :LAST_STEPS]
    gains_bias_sub = gains_hist_bias[gain_subset_idx, :LAST_STEPS]

    v_sq_uni  = v_hist_uni[gain_subset_idx, :LAST_STEPS] ** 2
    v_sq_bias = v_hist_bias[gain_subset_idx, :LAST_STEPS] ** 2

    time_steps = np.arange(LAST_STEPS)

    fig4, axes4 = plt.subplots(3, 1, figsize=(7, 9), sharex=True)

    vsq_colors = plt.cm.tab20(np.linspace(0, 1, N_GAIN_SUBSET))
    for i in range(N_GAIN_SUBSET):
        axes4[0].plot(time_steps, gains_uni_sub[i],  color=vsq_colors[i], alpha=0.5, linewidth=2)
        axes4[1].plot(time_steps, gains_bias_sub[i], color=vsq_colors[i], alpha=0.5, linewidth=2)

    avg_vsq_uni     = avg_vsq_hist_uni[-LAST_STEPS:]

    axes4[2].plot(time_steps, avg_vsq_uni,     color='green',     linestyle='--', linewidth=3)

    # Dummy handle so v² appears once in the legend
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color='grey',       alpha=0.6,      linewidth=2, label='f(v²) subset'),
        Line2D([0], [0], color='lightblue', linestyle='--', linewidth=2, label='mean f(v²)'),
        Line2D([0], [0], color='green',     linestyle='--', linewidth=3, label='avg_vsq (dynamics)'),
    ]
    axes4[2].legend(handles=legend_handles, loc='upper right',
                    fontsize=13, prop={'weight': 'bold', 'size': 13})

    titles   = ["Subset of Gain Dynamics — Uniform Ensemble",
                 "Subset of Gain Dynamics — Biased Ensemble",
                 ""]
    ylabels  = ["Gain Value", "Gain Value", "Value"]

    for ax, title, ylabel in zip(axes4, titles, ylabels):
        if title:
            ax.set_title(title, fontsize=15, fontweight='bold', color='black', pad=10)
        ax.set_ylabel(ylabel, fontsize=13, fontweight='bold', color='black', labelpad=8)
        ax.grid(False)

        for spine in ax.spines.values():
            spine.set_edgecolor(DARK_GREY)
            spine.set_linewidth(2.5)

        ax.tick_params(axis='both', colors=DARK_GREY, width=2.5, length=6, labelsize=11)
        ax.yaxis.label.set_color(DARK_GREY)

    axes4[2].set_xlabel("Time-step", fontsize=13, fontweight='bold', color='black', labelpad=8)
    axes4[2].xaxis.label.set_color(DARK_GREY)

    plt.tight_layout()
    plt.show()

    # =================================================================
    # FIGURE 4: Tuning Curve Peak Shifts (Adaptive ORGaNICs)
    # =================================================================

    binned_adp_uni  = get_binned_curves(results['adp_uni'],  tunings.theta, probe_angles, N_BINS)
    binned_adp_bias = get_binned_curves(results['adp_bias'], tunings.theta, probe_angles, N_BINS)

    peak_deg_uni  = probe_angles_deg[np.argmax(binned_adp_uni,  axis=1)]
    peak_deg_bias = probe_angles_deg[np.argmax(binned_adp_bias, axis=1)]
    peak_shifts   = peak_deg_bias - peak_deg_uni

    bin_numbers = np.arange(1, N_BINS + 1)

    fig_shifts, ax_shifts = plt.subplots(1, 1, figsize=(6, 4))

    ax_shifts.axhline(0, color='grey', linestyle='-', linewidth=1.5, zorder=1)
    ax_shifts.plot(bin_numbers, peak_shifts, 'o-', color='#00008B', linewidth=2.5, markersize=8, zorder=2)

    ax_shifts.set_title("Tuning Shifts", fontweight='bold', fontsize=15)
    ax_shifts.set_ylabel("Degrees", fontweight='bold', fontsize=13)
    ax_shifts.set_xlabel("Bin", fontweight='bold', fontsize=13)
    ax_shifts.set_xticks(bin_numbers)
    ax_shifts.grid(False)

    for spine in ax_shifts.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(2.5)

    plt.tight_layout()
    plt.show()

    # =================================================================
    # FIGURE 5: Calculated Average Stimuli (avg_z)
    # =================================================================

    neuron_angles_deg = tunings.theta * 180 / np.pi

    fig_avgz, ax_avgz = plt.subplots(1, 1, figsize=(7, 5))

    ax_avgz.plot(neuron_angles_deg, avg_z_hist_uni[:, -1],
                 color='#6BAED6', linewidth=3.5, label='Uniform Ensemble')
    ax_avgz.plot(neuron_angles_deg, avg_z_hist_bias[:, -1],
                 color='#08306B', linewidth=3.5, label='Biased Ensemble')

    ax_avgz.set_title("Calculated Average Stimuli", fontweight='bold', fontsize=18)
    ax_avgz.set_xlabel("Stimulus Angle (°)", fontweight='bold', fontsize=15)
    ax_avgz.set_ylabel("avg_z", fontweight='bold', fontsize=15)
    ax_avgz.set_xlim(0, 180)
    ax_avgz.grid(False)
    ax_avgz.legend(fontsize=13)
    ax_avgz.tick_params(axis='both', width=2.5, length=6, labelsize=13)

    for spine in ax_avgz.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(2.5)

    plt.tight_layout()
    plt.show()
