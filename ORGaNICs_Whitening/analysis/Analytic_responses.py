"""
Analytic_responses.py

Calculates neural firing rates for a known input distribution using analytical expressions 
and a self-consistency loop.

Methodology:
1. Optimal gains are calculated assuming a context of normalized responses to a given input distribution.
2. Expected response of primary neurons over the context, mu = <y>, is calculated via a self-consistency loop
3. A closed-form expression for firing rates as a function of input, optimal gains, and mu is used to calculate 
    steady state responses.
"""

import os
import sys
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import numpy as np
import matplotlib.pyplot as plt
from simulation_whiten import V1Dynamics, Frame
from tunings_whiten import V1Tunings
from stimuli_whiten import StimulusGenerator
from tqdm import tqdm
import scipy
from scipy.optimize import minimize, Bounds, nnls

sigma = 0.1       # normalization constant (matches V1Dynamics default)
N_matrix = None   # set in __main__ after V1Tunings is instantiated

def get_optimal_gains(stimuli, frame, label='', no_norm=False, poisson_variance=False):
    N, K = frame.shape  # N = 13, K = 91

    # Vectorize covariance generation (Removes the slow Python loop)
    stimuli = np.asarray(stimuli)
    Beta = 0.5
    raw_input_drive = stimuli * Beta
    Z_sq = (stimuli) ** 2
    denom = np.sqrt(sigma**2 + (N_matrix @ Z_sq.T).T)
    covariance_array = stimuli / denom

    if no_norm == True:
        Covariance = np.cov(raw_input_drive, rowvar=False)
    else:
        Covariance = np.cov(covariance_array, rowvar=False)

    if poisson_variance:
        # Deterministically inject Poisson-like variance (Var = Mean) onto the
        # covariance diagonal, distinct from the random poisson_noise sampling
        mean_drive = np.mean(raw_input_drive if no_norm else covariance_array, axis=0)
        Covariance = Covariance + 0.1*np.diag(np.maximum(mean_drive, 0.0))

    # Fast symmetric matrix square root (Safer and faster than general scipy.linalg.sqrtm)
    eigvals, eigvecs = np.linalg.eigh(Covariance)
    sqrt_Cov = eigvecs @ np.diag(np.sqrt(np.maximum(eigvals, 0))) @ eigvecs.T
    A = sqrt_Cov - np.eye(N)

    # Compute the exact right-hand side vector (K,) without full matrix multiplications
    diag_WTAW = np.diag(frame.T @ A @ frame)

    # Compute the exact left-hand side matrix (K, K)
    WTW = frame.T @ frame                              
    WTW_sq = WTW ** 2                                  # Element-wise square
    inv_WTW_sq = np.linalg.pinv(WTW_sq)
    g_opt = inv_WTW_sq @ diag_WTAW
    
    # Enforce non-negativity
    #g_opt = np.maximum(g_opt, 0.0)

    # DIAGNOSTIC: sqrt(Covariance) vs its I + W@diag(g_opt)@W.T factorization
    fig_diag, ax_diag = plt.subplots(1, 2, figsize=(8, 4))
    vmin, vmax = sqrt_Cov.min(), sqrt_Cov.max()
    ax_diag[0].imshow(sqrt_Cov, vmin=vmin, vmax=vmax); ax_diag[0].set_title("sqrt(Cov)")
    ax_diag[1].imshow(np.eye(N) + frame @ np.diag(g_opt) @ frame.T, vmin=vmin, vmax=vmax); ax_diag[1].set_title("I + W g W.T")
    plt.tight_layout(); plt.show()

    return g_opt

def get_optimal_gains_target(stimuli, frame, label='', no_norm=False, uniform_stimuli=None,
                                poisson_variance=False):
    N, K = frame.shape  # N = 13, K = 91

    # Covariance generation
    stimuli = np.asarray(stimuli)
    Beta = 0.5
    raw_input_drive = stimuli * Beta
    Z_sq = (raw_input_drive) ** 2
    denom = np.sqrt(sigma**2 + (N_matrix @ Z_sq.T).T)
    covariance_array = raw_input_drive / denom

    if no_norm == True:
        Covariance = np.cov(raw_input_drive, rowvar=False)
    else:
        Covariance = np.cov(covariance_array, rowvar=False)

    if poisson_variance:
        # Deterministically inject Poisson-like variance (Var = Mean) onto the
        # covariance diagonal, distinct from the random poisson_noise sampling
        # above -- no RNG, no perturbation of off-diagonal (correlation) terms.
        mean_drive = np.mean(raw_input_drive if no_norm else covariance_array, axis=0)
        Covariance = Covariance + 0.125 * np.diag(np.maximum(mean_drive, 0.0))

    # GET MODIFED WHITENING MATRIX THAT SCALES ONLY LARGE VARIANCES
    eigenvalues, eigenvectors = np.linalg.eigh(Covariance)
    safe_lambdas = np.maximum(eigenvalues, 1e-9)

    if uniform_stimuli is not None:
        # Target variance from the uniform ensemble
        uniform_stimuli = np.asarray(uniform_stimuli)
        uniform_raw_drive = uniform_stimuli * Beta
        uniform_Z_sq = uniform_raw_drive ** 2
        uniform_denom = np.sqrt(sigma**2 + (N_matrix @ uniform_Z_sq.T).T)
        uniform_covariance_array = uniform_raw_drive / uniform_denom
        uniform_Covariance = (np.cov(uniform_raw_drive, rowvar=False) if no_norm
                              else np.cov(uniform_covariance_array, rowvar=False))
        target = np.mean(np.diag(uniform_Covariance))
    else:
        target = np.mean(eigenvalues) # Set the mean variance as the upper bound ("target")

    d = np.minimum(1.0, np.sqrt(target / safe_lambdas))
    T = eigenvectors @ np.diag(d) @ eigenvectors.T

    # NOW COMPUTE OPTIMAL GAINS WITH LYNDON'S EQUATION A.5 
    T_inv = np.linalg.inv(T)
    A = T_inv - np.eye(N) # Modified transformation for the optimal gains
    diag_WTAW = np.diag(frame.T @ A @ frame)
    WTW = frame.T @ frame                              
    WTW_sq = WTW ** 2                                  # Element-wise square
    inv_WTW_sq = np.linalg.pinv(WTW_sq)
    g_opt = inv_WTW_sq @ diag_WTAW


    # DIAGNOSTIC: sqrt(Covariance) vs its I + W@diag(g_opt)@W.T factorization
    fig_diag, ax_diag = plt.subplots(1, 2, figsize=(8, 4))
    vmin, vmax = T_inv.min(), T_inv.max()
    ax_diag[0].imshow(T_inv, vmin=vmin, vmax=vmax); ax_diag[0].set_title("T^-1")
    ax_diag[1].imshow(np.eye(N) + frame @ np.diag(g_opt) @ frame.T, vmin=vmin, vmax=vmax); ax_diag[1].set_title("I + W g W.T")
    plt.tight_layout(); plt.show()

    return g_opt

def get_mu(stimuli, frame, optimal_gains, alpha=0.1, Beta=0.5):

    # Self-consistency loop to calculate mu given the input dataset and optimal gains

    N, K = frame.shape
    M = (frame * optimal_gains) @ frame.T  # avoids building a K×K diagonal matrix
    mu = np.zeros(N)
    diff = 1

    pbar = tqdm(desc="  mu convergence", unit="iter")
    while diff > 1e-6:
        y_total = 0

        for z in tqdm(stimuli, desc="    stimuli", leave=False):
            z_prime = 2*(Beta * z - M @ mu)
            y_total += z_prime / np.sqrt(sigma**2 + N_matrix @ (z_prime * z_prime))

        mu_new = y_total / len(stimuli)
        mu_old = mu.copy()
        diff = np.linalg.norm(mu_new - mu_old)
        mu += alpha * (mu_new - mu_old)
        print(np.mean(mu))
        pbar.set_postfix(diff=f"{diff:.2e}")
        pbar.update(1)
    pbar.close()

    return mu
        
def get_response_perceptual(stimulus, mu, M, Beta=0.5):
    gain_feedback = M @ mu
    z_prime = 2*(Beta * stimulus - gain_feedback)
    y = z_prime / np.sqrt(sigma**2 + N_matrix @ (z_prime**2))

    rectified_y = y 
    return rectified_y

def get_response_simple(stimulus, mu, M, Beta=0.5):
    gain_feedback = M @ mu
    z_normalized = stimulus / np.sqrt(sigma**2 + N_matrix @ (stimulus**2))
    y = z_normalized - gain_feedback

    rectified_y = y 
    return rectified_y

if __name__ == "__main__":

    N = 13
    N_BINS = 13

    # Initialize model components
    print("Initializing...")
    tunings   = V1Tunings(N=N)
    frame_obj = Frame(csv_path=os.path.join(REPO_ROOT, "data/frames/N13_mercedes_Frame.csv"))
    W = frame_obj.W  # raw (N, K) numpy array used by the analytic functions

    # Set globals required by the analytic functions above
    N_matrix = tunings.N_matrix

    # (a) Stimulus streams.
    duration   = 1
    num_angles = N
    stim_gen = StimulusGenerator(N=N, num_angles=num_angles,
                                 stream_length=num_angles, contrast=0.05)

    print("Generating stimulus streams...")
    seq_uni, centers_uni = stim_gen.generate_input_ensembles(
        biased=False, return_angles=True, duration=duration)
    stimuli_uni = list(seq_uni.T)

    # Build biased stream manually for equal non-adaptor representation
    adaptor_idx = num_angles // 2
    adaptor_rad = stim_gen.theta_inputs[adaptor_idx]
    n_non_adaptor  = num_angles - 1           # 168
    n_adaptor_reps = n_non_adaptor // 2       # 84  →  adaptor ≈ 1/3 of total

    non_adaptor_thetas = np.concatenate([
        stim_gen.theta_inputs[:adaptor_idx],
        stim_gen.theta_inputs[adaptor_idx + 1:]
    ])
    centers_bias = np.concatenate([
        non_adaptor_thetas,
        np.full(n_adaptor_reps, adaptor_rad)
    ])
    np.random.shuffle(centers_bias)

    delta = stim_gen.theta_inputs[:, None] - centers_bias[None, :]
    delta = (delta + np.pi / 2) % np.pi - np.pi / 2
    seq_bias = np.exp(-delta**2 / (2 * stim_gen.tuning_width**2))
    seq_bias = stim_gen.contrast * 15 * seq_bias / np.max(seq_bias)
    stimuli_bias = list(seq_bias.T)

    # (b) Optimal gains for each context
    print("Computing optimal gains (uniform)...")
    g_opt_uni  = get_optimal_gains_target(stimuli_uni,  W, label='uniform', poisson_variance=True)
    print("Computing optimal gains (biased)...")
    g_opt_bias = get_optimal_gains_target(stimuli_bias, W, label='biased', poisson_variance=True, uniform_stimuli=stimuli_uni)

    # Gain histogram — small plot, dark green bins, bold axes, no gridlines
    DARK_GREEN = '#006400'

    fig3, ax3 = plt.subplots(figsize=(4, 3))
    ax3.hist(g_opt_uni,  bins=20, color=DARK_GREEN, rwidth=0.9,
             label='Uniform', alpha=0.85)
    ax3.hist(g_opt_bias, bins=20, color='#228B22',  rwidth=0.9,
             label='Biased',  alpha=0.70)
    ax3.set_xlabel("Gain Value", fontsize=14, fontweight='bold')
    ax3.set_ylabel("Count",      fontsize=14, fontweight='bold')
    ax3.set_title("Gain Distribution", fontsize=13, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(False)
    for spine in ax3.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(2.5)
    ax3.tick_params(axis='both', width=2.5, length=6, labelsize=11)
    plt.tight_layout()
    plt.show()

    
    # (c) Self-consistent mu for each context
    print("Computing mu (uniform)...")
    mu_uni  = get_mu(stimuli_uni,  W, g_opt_uni)
    print("Computing mu (biased)...")
    mu_bias = get_mu(stimuli_bias, W, g_opt_bias)

    fig_mu, ax_mu = plt.subplots(figsize=(6, 3))
    ax_mu.plot(mu_uni,  label='uniform')
    ax_mu.plot(mu_bias, label='biased')
    ax_mu.set_xlabel("Neuron"); ax_mu.set_ylabel("mu"); ax_mu.legend(); plt.tight_layout(); plt.show()

    
    # (d) Distinct stimulus vectors — probed on a fine angle grid (independent of
    #     the N=13 input channels) so the resulting tuning curves come out smooth.
    N_PROBES = 180  # 1-degree resolution over the 180-degree orientation range
    probe_angles = np.linspace(0, np.pi, N_PROBES, endpoint=False)

    distinct_stimuli = []
    for angle in probe_angles:
        delta = stim_gen.theta_inputs - angle
        delta = (delta + np.pi / 2) % np.pi - np.pi / 2
        z = np.exp(-delta**2 / (2 * stim_gen.tuning_width**2))
        z = stim_gen.contrast * 15 * z / np.max(z)
        distinct_stimuli.append(z)

    n_distinct     = len(distinct_stimuli)
    responses_uni  = np.zeros((N, n_distinct))
    responses_bias = np.zeros((N, n_distinct))

    # Precompute the N×N feedback matrices once (avoids recomputing per stimulus)
    M_uni  = (W @ np.diag(g_opt_uni))  @ W.T
    M_bias = (W @ np.diag(g_opt_bias)) @ W.T

    
    # DIAGNOSTIC: gain feedback (M @ mu, per get_response) for each ensemble
    gain_feedback_uni  = - M_uni  @ mu_uni
    gain_feedback_bias = - M_bias @ mu_bias

    fig_gf, ax_gf = plt.subplots(figsize=(6, 3))
    ax_gf.plot(gain_feedback_uni,  label='uniform',
               color='#0b3d91', linewidth=3.0)
    ax_gf.plot(gain_feedback_bias, label='biased',
               color='#b35900', linewidth=3.0)
    ax_gf.set_xlabel("Neuron Index", fontsize=16, fontweight='bold')
    ax_gf.set_ylabel("Gain Feedback", fontsize=16, fontweight='bold')
    ax_gf.tick_params(axis='both', width=2.5, length=6, labelsize=12)
    for spine in ax_gf.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(2.5)
    ax_gf.legend()
    plt.tight_layout(); plt.show()


    print("Computing steady-state responses...")
    for j, z in enumerate(tqdm(distinct_stimuli)):
        #responses_uni[:, j]  = get_response_perceptual(z, mu_uni,  M_uni)
        #responses_bias[:, j] = get_response_perceptual(z, mu_bias, M_bias)
        responses_uni[:, j]  = get_response_simple(z, mu_uni, M_uni)
        responses_bias[:, j] = get_response_simple(z, mu_bias, M_bias)

    # (e) Tuning curves: each neuron's response across the 180° input range.
    probe_angles_deg = probe_angles * 180 / np.pi

    tuning_curves_uni  = responses_uni   # shape (N, n_distinct)
    tuning_curves_bias = responses_bias  

    def half_wave_rectify(response):
        return (np.maximum(response, 0))**2

    # (f) Tuning curves — binned (same logic/dimensions as Carandini_plots.py) or
    #     unbinned (raw single-neuron curves, no averaging across preference bins).
    def get_tuning_curves(tuning_curves, neuron_preferences, probe_angs, n_bins=13, binned=False):
        rectified = half_wave_rectify(tuning_curves)

        if not binned:
            return rectified

        N_neurons     = len(neuron_preferences)
        discrete_step = np.pi / N_neurons
        bin_edges     = np.linspace(0, np.pi, n_bins + 1) - (discrete_step / 2)
        binned_response    = np.zeros((n_bins, len(probe_angs)))
        neuron_bin_indices = np.digitize(neuron_preferences, bin_edges) - 1
        neuron_bin_indices = np.clip(neuron_bin_indices, 0, n_bins - 1)
        for b in range(n_bins):
            mask = neuron_bin_indices == b
            if np.any(mask):
                binned_response[b, :] = np.mean(rectified[mask, :], axis=0)
        return binned_response


    def process_pair(tc_uni_raw, tc_bias_raw, binned=False):
        if binned==True:
            binned_uni  = get_tuning_curves(tc_uni_raw,  tunings.theta, probe_angles, N_BINS, binned=True)
            binned_bias = get_tuning_curves(tc_bias_raw, tunings.theta, probe_angles, N_BINS, binned=True)
            bin_max = np.max(binned_uni, axis=1, keepdims=True)
            bin_min = np.min(binned_uni, axis=1, keepdims=True)
            norm_uni  = (binned_uni  - bin_min) / (bin_max - bin_min + 1e-9)
            norm_bias = (binned_bias - bin_min) / (bin_max - bin_min + 1e-9)
        if binned==False:
            unbinned_uni  = get_tuning_curves(tc_uni_raw,  tunings.theta, probe_angles, N_BINS, binned=False)
            unbinned_bias = get_tuning_curves(tc_bias_raw, tunings.theta, probe_angles, N_BINS, binned=False)
            curve_max = np.max(unbinned_uni, axis=1, keepdims=True)
            curve_min = np.min(unbinned_uni, axis=1, keepdims=True)
            norm_uni  = (unbinned_uni  - curve_min) / (curve_max - curve_min + 1e-9)
            norm_bias = (unbinned_bias - curve_min) / (curve_max - curve_min + 1e-9)

        return norm_uni, norm_bias


    binned_uni, binned_bias = process_pair(tuning_curves_uni, tuning_curves_bias, binned=True)
    tc_uni, tc_bias = process_pair(tuning_curves_uni, tuning_curves_bias, binned=False)
    

    # (g) Figure 1 — top row (input histograms) and bottom row (analytic tuning curves)
    adaptor_idx = num_angles // 2
    adaptor_rad = stim_gen.theta_inputs[adaptor_idx]
    adaptor_deg = adaptor_rad * 180 / np.pi

    uni_angles_deg  = centers_uni  * 180 / np.pi
    bias_angles_deg = centers_bias * 180 / np.pi

    discrete_step_hist = 180 / N
    bins_hist    = np.linspace(0, 180, N_BINS + 1) - (discrete_step_hist / 2)
    weights_uni  = np.ones_like(uni_angles_deg)  / len(uni_angles_deg)
    weights_bias = np.ones_like(bias_angles_deg) / len(bias_angles_deg)

    x_axis        = (probe_angles_deg - adaptor_deg + 90) % 180 - 90
    sort_idx      = np.argsort(x_axis)
    x_axis_sorted = x_axis[sort_idx]

    blue_colors = plt.cm.Blues(np.linspace(0.2, 1.0, N_BINS))

    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharey='row',
                             gridspec_kw={'height_ratios': [0.8, 1.0]})

    axes[0, 0].hist(uni_angles_deg,  bins=bins_hist, weights=weights_uni,
                    color='black', rwidth=0.9)
    axes[0, 0].set_title("Uniform Ensemble",  fontweight='bold', fontsize=18)
    axes[0, 0].set_ylabel("Probability", fontsize=18)

    axes[0, 1].hist(bias_angles_deg, bins=bins_hist, weights=weights_bias,
                    color='black', rwidth=0.9)
    axes[0, 1].set_title("Biased Ensemble", fontweight='bold', fontsize=18)

    for ax in axes[0]:
        ax.set_xlim(bins_hist[0], bins_hist[-1])
        ax.tick_params(labelbottom=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    for i in range(N_BINS):
        axes[1, 0].plot(x_axis_sorted, binned_uni[i][sort_idx],
                        color=blue_colors[i], linewidth=2.0)
        axes[1, 1].plot(x_axis_sorted, binned_bias[i][sort_idx],
                        color=blue_colors[i], linewidth=2.0)

    axes[1, 0].set_ylabel("Response", fontsize=18)

    for c in [0, 1]:
        ax = axes[1, c]
        ax.set_xlim(-90, 90)
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin - 0.05 * (ymax - ymin), ymax)
        ax.grid(False)
        ax.set_xlabel("Stimulus Orientation (°)", fontsize=18)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()

    # Figure 2 — average activity per neuron over the biased ensemble
    n_non_adaptor  = N - 1
    n_adaptor_reps = n_non_adaptor // 2
    non_adaptor_angles = np.concatenate([
        stim_gen.theta_inputs[:adaptor_idx],
        stim_gen.theta_inputs[adaptor_idx + 1:]
    ])
    biased_stream = np.concatenate([
        non_adaptor_angles,
        np.full(n_adaptor_reps, adaptor_rad)
    ])

    probe_idx_for_stream = np.argmin(
        np.abs(biased_stream[:, None] - probe_angles[None, :]), axis=1
    )

    avg_uni2  = np.mean(tuning_curves_uni[:,  probe_idx_for_stream], axis=1)
    avg_bias2 = np.mean(tuning_curves_bias[:, probe_idx_for_stream], axis=1)

    # Estimate avg firing rate from avg membrane potential
    rec_avg_uni2 = half_wave_rectify(avg_uni2)
    rec_avg_bias2 = half_wave_rectify(avg_bias2)

    norm_avg_uni2  = rec_avg_uni2  / (np.mean(rec_avg_uni2)  + 1e-9)
    norm_avg_bias2 = rec_avg_bias2 / (np.mean(rec_avg_bias2) + 1e-9)

    neuron_prefs_deg = tunings.theta * 180 / np.pi
    x_neuron         = (neuron_prefs_deg - adaptor_deg + 90) % 180 - 90
    sort_neuron_idx  = np.argsort(x_neuron)

    # Peak-shift panel — same logic as Figure 4 in Carandini_plots.py: locate
    # each bin's tuning-curve peak (argmax is invariant to the per-bin min/max
    # normalization already applied to binned_uni/binned_bias) and compare
    # uniform vs biased peak orientation.
    peak_deg_uni  = probe_angles_deg[np.argmax(binned_uni,  axis=1)]
    peak_deg_bias = probe_angles_deg[np.argmax(binned_bias, axis=1)]
    peak_shifts   = peak_deg_bias - peak_deg_uni
    bin_numbers   = np.arange(1, N_BINS + 1)

    fig2, (ax2, ax_shift) = plt.subplots(1, 2, figsize=(11, 4))
    ax2.axhline(1, color='grey', linestyle='--', linewidth=1.2, zorder=1)
    ax2.plot(x_neuron[sort_neuron_idx], norm_avg_uni2[sort_neuron_idx],
             color='#333333', linewidth=2.5, label='Uniform')
    ax2.plot(x_neuron[sort_neuron_idx], norm_avg_bias2[sort_neuron_idx],
             color='#800020', linewidth=2.5, label='Biased')
    ax2.set_title("Average Activity (Biased Ensemble)", fontweight='bold', fontsize=18)
    ax2.set_ylabel("Normalized Response", fontweight='bold', fontsize=14)
    ax2.set_xlabel("Neuron Preference (deg)",  fontweight='bold', fontsize=14)
    ax2.set_xlim(-90, 90)
    ax2.legend(loc='upper right')
    ax2.grid(False)
    for spine in ax2.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(2.5)

    ax_shift.axhline(0, color='grey', linestyle='-', linewidth=1.5, zorder=1)
    ax_shift.plot(bin_numbers, peak_shifts, 'o-', color='#00008B',
                  linewidth=2.5, markersize=8, zorder=2)
    ax_shift.set_title("Tuning Shifts", fontweight='bold', fontsize=18)
    ax_shift.set_ylabel("Degrees", fontweight='bold', fontsize=14)
    ax_shift.set_xlabel("Bin", fontweight='bold', fontsize=14)
    ax_shift.set_xticks(bin_numbers)
    ax_shift.grid(False)
    for spine in ax_shift.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(2.5)

    plt.tight_layout()
    plt.show()