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

# ---- Parameters ----
N = 169                  # Number of primary neurons
N_BINS = 13              # Aggregation bins for visualization
STREAM_LENGTH = 10140    # Length of adaptation stream (steps)
PROBE_STEPS = 100        # Steps to settle for each probe stimulus
PROBE_RES = 180          # Resolution of tuning curve probe (number of angles)
Z_SPONT = 0.1            # Tonic LGN background drive (tune to control spontaneous rate;
                         # ~0.16 of max firing at Z_SPONT=0.3 with threshold=0.5, sigma=0.2)

np.random.seed(20)

def gaussian_rectify(y, threshold=0.5, sigma=0.2, r_max=1.0): 
    return 0.5 * (1 + erf((y - threshold) / (sigma * np.sqrt(2)))) * r_max

def run_probe(frame, tunings, fixed_gains, probe_angles, frozen_u=None, frozen_a=None,
              z_spont=0.3, scale=0.5):
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
    sigma_const = 0.10

    # --- Naka-Rushton LGN Input Mapping ---
    R_max_lgn = 1.5
    c_50_lgn = sigma_const  # Using sigma_const as the semi-saturation parameter
    n_exp = 1.0
    
    # Convert linear contrast (scale) into a saturated biological drive
    contrast_drive = R_max_lgn * (scale**n_exp) / (scale**n_exp + c_50_lgn**n_exp)

    for i, angle in enumerate(probe_angles):

        # Start y near the spontaneous resting state driven by z_spont
        y = z_spont * np.ones(N)

        # Let u and a freely adapt from their most recent state
        u = np.copy(frozen_u) if frozen_u is not None else np.zeros(N)
        a = np.copy(frozen_a) if frozen_a is not None else np.zeros(N)

        # 1. Construct Input for this probe angle
        tuning_width = 0.5 
        
        # Von Mises / Raised Cosine
        z_t = np.exp(tuning_width * np.cos(2 * (tunings.theta - angle)))
        
        # Apply the saturated contrast drive here instead of raw scale
        z_t = (z_t / np.max(z_t)) * contrast_drive

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
            input_drive = (beta * z_t) / 2 + z_spont

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
    
    S = frame.W @ frame.W.T                          
    eigvals, eigvecs = np.linalg.eigh(S)
    S_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    N_neu, K_neu = frame.W.shape
    frame.W = np.sqrt(K_neu / N_neu) * (S_inv_sqrt @ frame.W)

    WWT = frame.W @ frame.W.T
    eigvals_check = np.linalg.eigvalsh(WWT)
    print(f"Eigenvalue ratio: {eigvals_check.max()/eigvals_check.min():.6f}")

    stim_gen = StimulusGenerator(N=N, K=N, stream_length=STREAM_LENGTH)
    
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
    
    # --- SCENARIO A: ORGaNICs (Non-Adaptive) ---
    print("\n--- Running Non-Adaptive Models ---")
    
    engine_org_uni = V1Dynamics(tunings, frame, adaptive=False)
    org_uniform_rates, _, u_hist_org_uni, a_hist_org_uni, _, _ = engine_org_uni.run_simulation(seq_uni)
    
    results['org_uni'] = run_probe(frame, tunings, fixed_gains=None, probe_angles=probe_angles,
                                   frozen_u=u_hist_org_uni[:, -1], frozen_a=a_hist_org_uni[:, -1],
                                   z_spont=Z_SPONT)
                                   
    engine_org_bias = V1Dynamics(tunings, frame, adaptive=False)
    org_bias_rates, _, u_hist_org_bias, a_hist_org_bias, _, _ = engine_org_bias.run_simulation(seq_bias)
    
    results['org_bias'] = run_probe(frame, tunings, fixed_gains=None, probe_angles=probe_angles,
                                    frozen_u=u_hist_org_bias[:, -1], frozen_a=a_hist_org_bias[:, -1],
                                    z_spont=Z_SPONT)
    
    # --- SCENARIO B: Adaptive ORGaNICs ---
    print("\n--- Running Adaptive Models ---")
    
    print("Adapting to Uniform Ensemble...")
    engine_uni = V1Dynamics(tunings, frame, adaptive=True)
    adapt_uniform_rates, gains_hist_uni, u_hist_uni, a_hist_uni, v_hist_uni, avg_vsq_hist_uni = engine_uni.run_simulation(seq_uni)
    
    final_gains_uni = gains_hist_uni[:, -1] 
    final_u_uni = u_hist_uni[:, -1]
    final_a_uni = a_hist_uni[:, -1]
    
    print("Probing Uniform State...")
    results['adp_uni'] = run_probe(frame, tunings, final_gains_uni, probe_angles,
                                   frozen_u=final_u_uni, frozen_a=final_a_uni,
                                   z_spont=Z_SPONT)
    
    print("Adapting to Biased Ensemble...")
    engine_bias = V1Dynamics(tunings, frame, adaptive=True)
    adapt_biased_rates, gains_hist_bias, u_hist_bias, a_hist_bias, v_hist_bias, avg_vsq_hist_bias = engine_bias.run_simulation(seq_bias)
    
    final_gains_bias = gains_hist_bias[:, -1] 
    final_u_bias = u_hist_bias[:, -1]
    final_a_bias = a_hist_bias[:, -1]
    
    print("Probing Biased State...")
    results['adp_bias'] = run_probe(frame, tunings, final_gains_bias, probe_angles,
                                    frozen_u=final_u_bias, frozen_a=final_a_bias,
                                    z_spont=Z_SPONT)

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
    
    blue_colors = plt.cm.RdPu(np.linspace(0.2, 1.0, N_BINS))
    
    discrete_step = 180 / N 
    bins_hist = np.linspace(0, 180, N_BINS + 1) - (discrete_step / 2)
    weights_uni = np.ones_like(hist_uni) / len(hist_uni)
    weights_bias = np.ones_like(hist_bias) / len(hist_bias)

    axes[0, 0].hist(hist_uni, bins=bins_hist, weights=weights_uni, color='black', rwidth=0.9)
    axes[0, 0].set_title("Uniform Ensemble", fontweight='bold')
    axes[0, 0].set_ylabel("Probability")
    
    axes[0, 1].hist(hist_bias, bins=bins_hist, weights=weights_bias, color='black', rwidth=0.9)
    axes[0, 1].set_title("Biased Ensemble", fontweight='bold')
    
    for ax in axes[0]:
        ax.set_xlim(0, 180)
        ax.tick_params(labelbottom=False)

    for i in range(N_BINS):
        axes[1, 0].plot(x_axis_sorted, row2_uni[i][sort_idx], color=blue_colors[i], linewidth=1.5)
        axes[1, 1].plot(x_axis_sorted, row2_bias[i][sort_idx], color=blue_colors[i], linewidth=1.5)
        
    axes[1, 0].set_ylabel("Non-Adaptive\nNormalized Response", fontweight='bold')
    
    for i in range(N_BINS):
        axes[2, 0].plot(x_axis_sorted, row3_uni[i][sort_idx], color=blue_colors[i], linewidth=1.5)
        axes[2, 1].plot(x_axis_sorted, row3_bias[i][sort_idx], color=blue_colors[i], linewidth=1.5)
        
    axes[2, 0].set_ylabel("Adaptive\nNormalized Response", fontweight='bold')

    for r in [1, 2]:
        for c in [0, 1]:
            ax = axes[r, c]
            ax.set_xlim(-90, 90) 
            ax.grid(False, alpha=0.3)
            
            if r == 2:
                ax.set_xlabel("Orientation Relative to Adaptor (°)")
    
    plt.tight_layout()
    plt.show()

    # =================================================================
    # FIGURE 2: Average Steady-State Response per Orientation Bin
    # =================================================================
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
        duration = 20
        keep = 5 
        steady_rates = rates[:, -window:]
        n_time_steps = steady_rates.shape[1]
        
        time_mask = (np.arange(n_time_steps) % duration) >= (duration - keep)
        means = np.mean(steady_rates[:, time_mask], axis=1) 
        
        binned = np.zeros(N_BINS)
        for b in range(N_BINS):
            bin_mask = neuron_bin_idx == b
            if bin_mask.any():
                binned[b] = np.mean(means[bin_mask])
                
        return binned

    print("\nAdaptive + Uniform (10k steps)...")
    peaks_adp_uni = get_binned_activity(adapt_uniform_rates, AVG_WINDOW)
    del adapt_uniform_rates; gc.collect()

    print("Adaptive + Biased (10k steps)...")
    peaks_adp_bias = get_binned_activity(adapt_biased_rates, AVG_WINDOW)
    del adapt_biased_rates; gc.collect()

    print("ORGaNICs + Biased (10k steps)...")
    peaks_org_bias = get_binned_activity(org_bias_rates, AVG_WINDOW)
    del org_bias_rates; gc.collect()

    x_peak = (bin_centers_deg - adaptor_deg + 90) % 180 - 90
    sort_idx_2 = np.argsort(x_peak)
    x_peak_sorted = x_peak[sort_idx_2]

    norm_adp_bias = peaks_adp_bias / np.mean(peaks_adp_bias)
    norm_org_bias = peaks_org_bias / np.mean(peaks_org_bias)

    fig2, ax2 = plt.subplots(1, 1, figsize=(6, 4))

    ax2.axhline(1, color='grey', linestyle='--', linewidth=1.2, zorder=1)
    ax2.plot(x_peak_sorted, norm_adp_bias[sort_idx_2], 'o-', color='steelblue',
             linewidth=2, markersize=5, label='Adaptive')
    ax2.plot(x_peak_sorted, norm_org_bias[sort_idx_2], 's-', color='coral',
             linewidth=2, markersize=5, label='Non-Adaptive')
    ax2.set_title("Biased Ensemble: Normalized Response", fontweight='bold')
    ax2.set_ylabel("Response / Mean Response")
    ax2.set_xlabel("Orientation (°)")
    ax2.set_xlim(-90, 90)
    ax2.legend()
    ax2.grid(False)

    fig2.suptitle("Average Steady State Response", fontweight='bold', fontsize=13)
    plt.tight_layout()
    plt.show()

    # =================================================================
    # FIGURE 3: Contrast Response Function
    # =================================================================
    '''print("\n" + "=" * 50)
    print("  FIGURE 3: Contrast Response Function")
    print("=" * 50)

    contrasts = np.geomspace(0.01, 1.0, 40)
    crf_peak = np.zeros((N, len(contrasts))) # Stores peak response for each neuron at each contrast (one stimulus orientation)

    # Collect the min/max response for each neuron given a contrast, then bin the activity.
    for ci, contrast in enumerate(contrasts):
        print(f"  Probing contrast = {contrast:.3f}...")
        # Probing without gain adaptation to get the baseline CRF
        crf_tuning_curves = run_probe(frame, tunings, fixed_gains=None, probe_angles=probe_angles,
                       scale=contrast, z_spont=Z_SPONT)
        crf_peak[:, ci] = np.max(crf_tuning_curves, axis=1)  

    crf_binned = np.zeros((N_BINS, len(contrasts))) # Stores binned peak responses at each contrast
    for b in range(N_BINS):
        mask = neuron_bin_idx == b # Note this is NOT stimulus masking, but instead a technique to single out bins
        if mask.any():
            crf_binned[b] = np.mean(crf_peak[mask], axis=0) 

    # Compute sigmas dynamically based on baseline firing
    sigmas = np.zeros(N_BINS) # Calculated semi-saturation constant for each bin (should roughly be the same)
    for b in range(N_BINS):
        baseline = crf_binned[b, 0] # Min firing rate for each bin
        peak = np.max(crf_binned[b]) # Max firing rate for each bin
        half_max = baseline + (peak - baseline) / 2.0 # Extract the half max firing rate
        
        above = np.where(crf_binned[b] >= half_max)[0] # For that bin, extract the first response above the half-max firing rate
        if len(above) > 0 and above[0] > 0: # Confirming that it exists and is non-negative
            # Record index, responses, contrasts right before and right after reaching half-max threshold
            i0, i1 = above[0] - 1, above[0] 
            r0, r1 = crf_binned[b, i0], crf_binned[b, i1] 
            c0, c1 = contrasts[i0], contrasts[i1] 
            sigmas[b] = c0 + (half_max - r0) / (r1 - r0) * (c1 - c0) # Estimate of sigmas using linear interpolation
        elif len(above) > 0:
            sigmas[b] = contrasts[above[0]]

    fig3, ax3 = plt.subplots(1, 1, figsize=(7, 5))
    
    # We reuse the RdPu colormap from Figure 1
    for b in range(N_BINS):
        ax3.plot(contrasts, crf_binned[b], color=blue_colors[b], linewidth=1.5)

    mid = N_BINS // 2 + 1
    baseline_mid = crf_binned[mid, 0]
    peak_mid = np.max(crf_binned[mid])
    half_max_mid = baseline_mid + (peak_mid - baseline_mid) / 2.0
    
    ax3.axhline(half_max_mid, color='grey', linestyle=':', linewidth=1.0)
    ax3.axvline(sigmas[mid], color='grey', linestyle=':', linewidth=1.0,
                label=f'σ = {sigmas[mid]:.2f}')

    ax3.set_xscale('log')
    ax3.set_title("Contrast Response Function", fontweight='bold')
    ax3.set_xlabel("Contrast (log scale)")
    ax3.set_ylabel("Peak Response")
    ax3.legend(fontsize='small')
    ax3.grid(False)
    fig3.suptitle("Contrast Response Function",
                  fontweight='bold', fontsize=13)
    plt.tight_layout()
    plt.show()'''

    # =================================================================
    # FIGURE 4: Subset of Gain Dynamics (Last 1000 Steps)
    # =================================================================
    print("\n" + "=" * 50)
    print("  FIGURE 4: Subset of Gain Dynamics")
    print("=" * 50)

    LAST_STEPS = 1000
    N_GAIN_SUBSET = 50
    DARK_ORANGE = '#CC5500'
    DARK_GREY = '#333333'
    DARK_BLUE = '#00008B'

    gain_subset_idx = np.random.choice(N, N_GAIN_SUBSET, replace=False)

    gains_uni_sub  = gains_hist_uni[gain_subset_idx, -LAST_STEPS:]
    gains_bias_sub = gains_hist_bias[gain_subset_idx, -LAST_STEPS:]

    v_sq_uni  = v_hist_uni[gain_subset_idx, -LAST_STEPS:] ** 2
    v_sq_bias = v_hist_bias[gain_subset_idx, LAST_STEPS:] ** 2

    time_steps = np.arange(LAST_STEPS)

    fig4, axes4 = plt.subplots(3, 1, figsize=(7, 9), sharex=True)

    for i in range(N_GAIN_SUBSET):
        axes4[0].plot(time_steps, gains_uni_sub[i],  color=DARK_ORANGE, alpha=0.25, linewidth=2)
        axes4[1].plot(time_steps, gains_bias_sub[i], color=DARK_ORANGE, alpha=0.25, linewidth=2)

    c_vsq = engine_uni.c_vsq
    v_sq_c_uni = v_sq_uni / (v_sq_uni + c_vsq)

    for i in range(N_GAIN_SUBSET):
        axes4[2].plot(time_steps, v_sq_c_uni[i], color=DARK_BLUE, alpha=0.25, linewidth=2)

    mean_v_sq_c_uni = np.mean(v_sq_c_uni, axis=0)
    avg_vsq_uni     = avg_vsq_hist_uni[-LAST_STEPS:]

    axes4[2].plot(time_steps, mean_v_sq_c_uni, color='lightblue', linestyle='--', linewidth=2)
    axes4[2].plot(time_steps, avg_vsq_uni,     color='green',     linestyle='--', linewidth=3)

    # Dummy handle so v² appears once in the legend
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color=DARK_BLUE,   alpha=0.6,      linewidth=2, label='f(v²) subset'),
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