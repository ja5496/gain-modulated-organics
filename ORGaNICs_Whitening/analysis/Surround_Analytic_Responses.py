'''
Surround_Analytic_Responses.py

Calculates neural firing rates for a known input distribution using analytical expressions.
Interactions between classical receptive field (cRF) neurons and surround (non-cRF) neurons are modeled.
The normalization pool includes all neurons, while adaptation (gain feedback) is local to each RF.

Population structure: 7 sets of 13 primary neurons each (91 total). One set (CRF_IDX) is the
classical RF; the other 6 are surround sets. All 7 sets share the same tuning-curve basis and the
same per-RF frame, so the 6 surround sets are treated as fully interchangeable (no distance-dependent
weighting) -- only cRF vs. non-cRF membership matters.

Methodology:
1. Optimal gains for a single 13-neuron RF pool are computed once per ensemble type (baseline
   uniform, or adaptor-biased) and shared across every set that sees that ensemble.
2. For each of the 4 adaptation conditions (none / cRF-only / non-cRF-only / both), a block-diagonal
   feedback matrix M (91x91) is assembled from those per-RF gains, and mu = <y> is found via a joint
   self-consistency loop over the full 91-neuron population (normalization pools across all 91;
   gain feedback stays within each 13-neuron block).
3. Steady-state responses to probe stimuli (shown to the full population)
   are computed in closed form from the probe, optimal gains, and mu.
'''

import os
import sys
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from simulation_whiten import Frame
from tunings_whiten import V1Tunings
from stimuli_whiten import StimulusGenerator
from analysis import Analytic_responses as ar
from typing import Literal
from scipy.linalg import block_diag

N_RF       = 13                    # primary neurons per receptive field
N_SETS     = 7                     # 1 classical RF (cRF) + 6 surround sets
N_TOTAL    = N_RF * N_SETS         # full primary-neuron population
CRF_IDX    = 0                     # which of the 7 sets is the cRF (arbitrary; sets are symmetric)
FRAME_PATH = os.path.join(REPO_ROOT, "data/frames/N13_mercedes_Frame.csv")
N_matrix = np.ones((N_TOTAL, N_TOTAL))
sigma = 14
Beta  = 0.5


ENSEMBLE_CONTRAST = 0.8      # contrast of the adaptation ensembles (baseline & adaptor)
TUNING_WIDTH      = 0.75
N_CONTRASTS   = 20
CRF_CONTRASTS = np.logspace(-2, 0, N_CONTRASTS)
PROBE_CONTRAST = 0.05   
N_PROBES       = 720

# Setting colors for plot lines (designated by what section of the visual field is adapted)
COLOR_NONE   = 'black'
COLOR_CRF    = '#FDE68A'     # pastel yellow
COLOR_NONCRF = 'red'
COLOR_BOTH   = 'darkorange'


def half_wave_rectify(y):
    return np.maximum(y, 0.0) ** 2

def probe_input_drive(input_theta, contrast, tuning_width=TUNING_WIDTH):
    '''Always probing with a stimulus that cover both CRF and surround, no matter the adaptation state'''
    theta_grid = np.linspace(0, np.pi, N_RF, endpoint=False) # Evenly spaced orientation preferences for neurons
    delta = theta_grid - input_theta # Distance between neuron preference from stimulus orientation
    delta = (delta + np.pi / 2) % np.pi - np.pi / 2
    profile = np.exp(-delta**2 / (2 * tuning_width**2))  
    RF_gaussian_drive = contrast * 15 * profile #/ np.max(profile) # Gaussian input response profile scaled by contrast

    full_drive =  np.concatenate([RF_gaussian_drive]  * (N_SETS))
    return full_drive

def block_diag_M(frame, g_opt, adapt_location: Literal['adapt CRF only', 'adapt surround only', 'adapt CRF and surround', 'no adaptation']):
    match adapt_location:
        case 'adapt CRF only':
            M_adapt_local = frame @ np.diag(g_opt) @ frame.T
            M_zeros_local = np.zeros((N_RF, N_RF))
            repeats = N_SETS - 1 # number of times the zero feedback must be copied to cover the surround
            M = block_diag(M_adapt_local, *[M_zeros_local] * repeats)
            return M
        case 'adapt surround only':
            M_adapt_local = frame @ np.diag(g_opt) @ frame.T
            M_zeros_local = np.zeros((N_RF, N_RF))
            repeats = N_SETS - 1 # number of times the adapt feedback must be copied to cover the surround
            M = block_diag(M_zeros_local, *[M_adapt_local] * repeats)
            return M
        case 'adapt CRF and surround':
            M_adapt_local = frame @ np.diag(g_opt) @ frame.T
            repeats = N_SETS # number of times the adaptation feedback must be copied to cover CRF and surround
            M = block_diag(*[M_adapt_local] * repeats)
            return M
        case 'no adaptation':
            M_zeros_local = np.zeros((N_RF, N_RF))
            repeats = N_SETS  # number of times the zero feedback must be copied to cover CRF and surround
            M = block_diag(*[M_zeros_local] * repeats)
            return M
        
def full_spatial_stimuli(stimuli, adapt_location: Literal['adapt CRF only', 'adapt surround only', 'adapt CRF and surround', 'no adaptation']):
    match adapt_location:
        case 'adapt CRF only':
            baseline = np.full(N_RF, 1.0)            
            repeats = N_SETS - 1
            full_stimuli = np.concatenate([stimuli] + [baseline] * repeats)
            return full_stimuli
        case 'adapt surround only':
            baseline = np.full(N_RF, 1.0)
            repeats = N_SETS - 1
            full_stimuli = np.concatenate([baseline] + [stimuli] * repeats)
            return full_stimuli
        case 'adapt CRF and surround':
            full_stimuli = np.concatenate([stimuli] * N_SETS)
            return full_stimuli
        case 'no adaptation':
            baseline = np.full(N_RF, 1.0)            
            full_stimuli = np.concatenate([baseline] * N_SETS)
            return full_stimuli

def get_mu(stimuli, M, alpha=0.1, Beta=0.5):
    # Self-consistency loop to calculate mu given the input dataset and optimal gains
    mu = np.zeros(N_TOTAL)
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
    
def get_response(stimulus, mu, M, Beta=0.5):
    gain_feedback = M @ mu
    z_prime = 2 * (Beta * stimulus - gain_feedback)
    y = z_prime / np.sqrt(sigma**2 + N_matrix @ (z_prime**2))

    rectified_y = half_wave_rectify(y)
    return rectified_y


CONDITIONS = ['no adaptation', 'adapt CRF only', 'adapt surround only', 'adapt CRF and surround']
CONDITION_LABEL = {
    'no adaptation':          'No adaptation',
    'adapt CRF only':         'Classical RF adapted',
    'adapt surround only':    'Surround adapted',
    'adapt CRF and surround': 'RF + surround adapted',
}
CONDITION_COLOR = {
    'no adaptation':          COLOR_NONE,
    'adapt CRF only':         COLOR_CRF,
    'adapt surround only':    COLOR_NONCRF,
    'adapt CRF and surround': COLOR_BOTH,
}

if __name__ == "__main__":

    print("Initializing tunings and frame...")
    tunings = V1Tunings(N=N_RF)
    frame   = Frame(csv_path=FRAME_PATH)

    # Optimal gains are derived from a single 13-neuron RF pool, so the covariance/
    # normalization pool used to derive them is local (13x13), not the full 91x91 pool.
    ar.sigma    = sigma
    ar.N_matrix = np.ones((N_RF, N_RF))

    # ---- Local (single-RF) biased ensemble: shared context for every adaptation case ----
    stim_gen = StimulusGenerator(N=N_RF, num_angles=N_RF, stream_length=N_RF,
                                 tuning_width=TUNING_WIDTH, contrast=ENSEMBLE_CONTRAST)

    print("Generating local uniform and biased ensembles...")
    seq_uni, centers_uni = stim_gen.generate_input_ensembles(
        biased=False, return_angles=True, duration=1)
    stimuli_uni = list(seq_uni.T)

    adaptor_idx = N_RF // 2
    adaptor_rad = stim_gen.theta_inputs[adaptor_idx]

    # Build the biased stream manually for equal non-adaptor representation --
    # generate_input_ensembles(biased=True) overwrites the first third of its
    # (pre-shuffle) index array with the adaptor index, so when stream_length
    # == num_angles (one un-shuffled cycle) that wipes out the first floor(N/3)
    # orientations entirely instead of just up-weighting the adaptor.
    n_non_adaptor  = N_RF - 1
    n_adaptor_reps = n_non_adaptor // 2

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

    print("Computing optimal gains (biased)...")
    g_opt = ar.get_optimal_gains_target(stimuli_bias, frame.W, label='biased',
                                        poisson_variance=True, uniform_stimuli=stimuli_uni)

    print("Building M for each condition...")
    M_by_condition = {cond: block_diag_M(frame.W, g_opt, cond) for cond in CONDITIONS}

    # ==========================================================================
    # CHECK 2 -- Plot probe_input_drive, centered at the biased ensemble's adaptor
    # ==========================================================================
    print("Plotting probe_input_drive...")
    probe_drive = probe_input_drive(adaptor_rad, PROBE_CONTRAST)

    fig_probe, ax_probe = plt.subplots(figsize=(9, 3))
    ax_probe.plot(probe_drive, color='#333333', linewidth=2.0)
    for s in range(1, N_SETS):
        ax_probe.axvline(s * N_RF - 0.5, color='grey', linestyle='--', linewidth=1.0)
    ax_probe.set_xlabel("Neuron index (7 sets x 13 neurons; set 0 = cRF)")
    ax_probe.set_ylabel("Drive")
    ax_probe.set_title("Probe centered at adaptor orientation "
                        f"({adaptor_rad * 180 / np.pi:.1f} deg)", fontweight='bold')
    plt.tight_layout(); plt.show()

    mu_by_condition = {}
    for cond in CONDITIONS:
        print(f"Computing mu ({CONDITION_LABEL[cond]})...")
        full_stimuli = [full_spatial_stimuli(z, cond) for z in stimuli_bias]
        mu_by_condition[cond] = get_mu(full_stimuli, M_by_condition[cond])

    # ==========================================================================
    # CHECK 5 -- Contrast response functions of the cRF neuron that prefers the adaptor
    # ==========================================================================
    print("Computing contrast response functions...")
    crf_target_idx = CRF_IDX * N_RF + adaptor_idx

    def crf_curve(cond):
        resp = np.zeros(N_CONTRASTS)
        for i, c in enumerate(CRF_CONTRASTS):
            probe = probe_input_drive(adaptor_rad, c)
            y = get_response(probe, mu_by_condition[cond], M_by_condition[cond])
            resp[i] = y[crf_target_idx]
        return resp

    curves_by_condition = {cond: crf_curve(cond) for cond in CONDITIONS}

    fig_crf, ax_crf = plt.subplots(figsize=(7, 5.5))
    for cond in CONDITIONS:
        ax_crf.plot(CRF_CONTRASTS, curves_by_condition[cond], color=CONDITION_COLOR[cond],
                    linewidth=3.5, label=CONDITION_LABEL[cond])

    # Half-saturation contrast of the unadapted (control) cRF curve -- interpolated
    # in log-contrast space between the two samples straddling half of its max.
    control_curve = curves_by_condition['no adaptation']
    half_max = control_curve.max() / 2.0
    idx = np.argmax(control_curve >= half_max)
    if idx == 0:
        c50 = CRF_CONTRASTS[0]
    else:
        log_c_lo, log_c_hi = np.log10(CRF_CONTRASTS[idx - 1]), np.log10(CRF_CONTRASTS[idx])
        r_lo, r_hi = control_curve[idx - 1], control_curve[idx]
        log_c50 = log_c_lo + (half_max - r_lo) * (log_c_hi - log_c_lo) / (r_hi - r_lo)
        c50 = 10 ** log_c50

    ax_crf.axvline(c50, color=COLOR_NONE, linestyle='--', linewidth=1.5, alpha=0.8)
    ax_crf.plot(c50, half_max, 'o', color=COLOR_NONE, markersize=8, zorder=5)
    ax_crf.text(c50, 0.95, f"c50 = {c50:.3g}", transform=ax_crf.get_xaxis_transform(),
                fontsize=11, fontweight='bold', color=COLOR_NONE, ha='center', va='top')

    ax_crf.set_xscale('log')
    ax_crf.set_title("Contrast Response Functions", fontsize=16, fontweight='bold')
    ax_crf.set_xlabel("Contrast", fontsize=14, fontweight='bold')
    ax_crf.set_yticks([])
    ax_crf.grid(False)
    ax_crf.spines['top'].set_visible(False)
    ax_crf.spines['right'].set_visible(False)
    ax_crf.spines['left'].set_visible(False)
    ax_crf.spines['bottom'].set_linewidth(2.5)
    ax_crf.tick_params(axis='x', width=2.5, length=6, labelsize=12)
    ax_crf.legend(fontsize=10, frameon=False)
    plt.tight_layout(); plt.show()

    # ==========================================================================
    # CHECK 6 -- Gain feedback (M @ mu) for each adaptation case
    # ==========================================================================
    print("Computing gain feedback...")
    gain_feedback_by_condition = {
        cond: M_by_condition[cond] @ mu_by_condition[cond] for cond in CONDITIONS
    }

    fig_gf, ax_gf = plt.subplots(figsize=(9, 4))
    for cond in CONDITIONS:
        ax_gf.plot(gain_feedback_by_condition[cond], color=CONDITION_COLOR[cond],
                   linewidth=2.5, label=CONDITION_LABEL[cond])
    for s in range(1, N_SETS):
        ax_gf.axvline(s * N_RF - 0.5, color='grey', linestyle='--', linewidth=1.0)
    ax_gf.set_xlabel("Neuron index (7 sets x 13 neurons; set 0 = cRF)")
    ax_gf.set_ylabel("Gain feedback")
    ax_gf.set_title("Gain feedback (M @ mu) per adaptation case", fontweight='bold')
    ax_gf.legend()
    plt.tight_layout(); plt.show()

    # ==========================================================================
    # CHECK 7 -- Tuning curve of a neuron adjacent to the adaptor-preferring neuron
    # ==========================================================================
    print("Computing flank-neuron tuning curves...")
    flank_idx = (adaptor_idx - 1) % N_RF
    flank_target_idx = CRF_IDX * N_RF + flank_idx

    TUNING_CONDITIONS = ['no adaptation', 'adapt surround only', 'adapt CRF only']
    probe_angles = np.linspace(0, np.pi, N_PROBES, endpoint=False)
    probe_angles_deg = probe_angles * 180 / np.pi

    def flank_tuning_curve(cond):
        resp = np.zeros(N_PROBES)
        for i, ang in enumerate(probe_angles):
            probe = probe_input_drive(ang, PROBE_CONTRAST)
            y = get_response(probe, mu_by_condition[cond], M_by_condition[cond])
            resp[i] = y[flank_target_idx]
        return resp

    flank_curves = {cond: flank_tuning_curve(cond) for cond in TUNING_CONDITIONS}

    # Test for a shift in tuning preference: locate each curve's peak (parabolic
    # interpolation around the argmax sample for sub-resolution precision) and
    # compare the two adapted conditions against the unadapted control.
    def curve_peak_deg(curve):
        i = int(np.argmax(curve))
        if 0 < i < len(curve) - 1:
            y0, y1, y2 = curve[i - 1], curve[i], curve[i + 1]
            denom = (y0 - 2 * y1 + y2)
            frac = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
        else:
            frac = 0.0
        step = probe_angles_deg[1] - probe_angles_deg[0]
        return probe_angles_deg[i] + frac * step

    peak_deg = {cond: curve_peak_deg(flank_curves[cond]) for cond in TUNING_CONDITIONS}
    control_peak = peak_deg['no adaptation']

    print("Flank neuron tuning-preference shift (vs. control):")
    for cond in TUNING_CONDITIONS:
        shift = ((peak_deg[cond] - control_peak + 90) % 180) - 90
        print(f"  {CONDITION_LABEL[cond]:25s} peak={peak_deg[cond]:7.2f} deg   shift={shift:+6.2f} deg")

    fig_flank, ax_flank = plt.subplots(figsize=(7, 5.5))
    for cond in TUNING_CONDITIONS:
        ax_flank.plot(probe_angles_deg, flank_curves[cond], color=CONDITION_COLOR[cond],
                      linewidth=3.5, label=CONDITION_LABEL[cond])

    # Vertical arrow marking the adaptor orientation
    adaptor_deg = adaptor_rad * 180 / np.pi
    ax_flank.annotate('', xy=(adaptor_deg, 0.80), xytext=(adaptor_deg, 0.94),
                       xycoords=('data', 'axes fraction'), textcoords=('data', 'axes fraction'),
                       arrowprops=dict(arrowstyle='-|>', color='black', linewidth=2.5, mutation_scale=18))
    ax_flank.text(adaptor_deg, 0.96, "adaptor", transform=ax_flank.get_xaxis_transform(),
                  fontsize=10, fontweight='bold', color='black', ha='center', va='bottom')

    ax_flank.set_title("Tuning Curve (Flank Neuron)", fontsize=16, fontweight='bold', pad=16)
    ax_flank.set_xlabel("stimulus orientation (deg)", fontsize=14, fontweight='bold')
    ax_flank.set_yticks([])
    ax_flank.grid(False)
    ax_flank.spines['top'].set_visible(False)
    ax_flank.spines['right'].set_visible(False)
    ax_flank.spines['left'].set_visible(False)
    ax_flank.spines['bottom'].set_linewidth(2.5)
    ax_flank.tick_params(axis='x', width=2.5, length=6, labelsize=12)
    ax_flank.legend(fontsize=10, frameon=False)
    plt.tight_layout(); plt.show()

    # ==========================================================================
    # FIGURE 1 (surround) -- recreates Figure 1 from Analytic_responses.py with
    # the exact same layout/style and the exact same input-ensemble histograms
    # (top row: Uniform Ensemble / Biased Ensemble). The only thing that changes
    # is the adaptation state driving the bottom-row tuning curves: instead of
    # separate uniform/biased gain contexts (that script had no surround), the
    # left column uses the 'no adaptation' state and the right column uses the
    # 'adapt CRF only' state from this script's block-diagonal model. The probe
    # still sweeps the full 0-180 deg range and always drives the whole cRF +
    # surround population (probe_input_drive), matching the original's coverage.
    # ==========================================================================
    print("Recreating Figure 1 (cRF-only-adapted state)...")
    N_BINS = N_RF

    uni_angles_deg  = centers_uni  * 180 / np.pi
    bias_angles_deg = centers_bias * 180 / np.pi

    discrete_step_hist = 180 / N_RF
    bins_hist    = np.linspace(0, 180, N_BINS + 1) - (discrete_step_hist / 2)
    weights_uni  = np.ones_like(uni_angles_deg)  / len(uni_angles_deg)
    weights_bias = np.ones_like(bias_angles_deg) / len(bias_angles_deg)

    # Tuning curves for the cRF neurons only, under each adaptation state,
    # probed across the full orientation range (probe_angles/N_PROBES already
    # cover the whole 0-180 deg range and the whole cRF+surround population).
    crf_slice = slice(CRF_IDX * N_RF, (CRF_IDX + 1) * N_RF)

    def crf_tuning_curves(cond):
        resp = np.zeros((N_RF, N_PROBES))
        for i, ang in enumerate(probe_angles):
            probe = probe_input_drive(ang, PROBE_CONTRAST)
            y = get_response(probe, mu_by_condition[cond], M_by_condition[cond])
            resp[:, i] = y[crf_slice]
        return resp

    tc_none = crf_tuning_curves('no adaptation')
    tc_crf  = crf_tuning_curves('adapt CRF and surround') #adapt CRF only

    # Bin by neuron preference (same logic/dimensions as get_tuning_curves in
    # Analytic_responses.py); get_response already half-wave rectifies, so no
    # extra rectification step is needed here.
    def bin_by_preference(response, neuron_preferences, n_bins=N_BINS):
        discrete_step = np.pi / len(neuron_preferences)
        bin_edges = np.linspace(0, np.pi, n_bins + 1) - (discrete_step / 2)
        binned = np.zeros((n_bins, response.shape[1]))
        bin_idx = np.digitize(neuron_preferences, bin_edges) - 1
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)
        for b in range(n_bins):
            mask = bin_idx == b
            if np.any(mask):
                binned[b, :] = np.mean(response[mask, :], axis=0)
        return binned

    binned_none = bin_by_preference(tc_none, tunings.theta)
    binned_crf  = bin_by_preference(tc_crf,  tunings.theta)

    # Normalize both panels against the 'no adaptation' (reference/control)
    # panel's per-bin min/max, exactly as process_pair normalizes bias against
    # the uniform reference in Analytic_responses.py.
    bin_max = np.max(binned_none, axis=1, keepdims=True)
    bin_min = np.min(binned_none, axis=1, keepdims=True)
    norm_none = (binned_none - bin_min) / (bin_max - bin_min + 1e-9)
    norm_crf  = (binned_crf  - bin_min) / (bin_max - bin_min + 1e-9)

    x_axis        = (probe_angles_deg - adaptor_deg + 90) % 180 - 90
    sort_idx      = np.argsort(x_axis)
    x_axis_sorted = x_axis[sort_idx]

    blue_colors = plt.cm.Blues(np.linspace(0.2, 1.0, N_BINS))

    fig1s, axes1s = plt.subplots(2, 2, figsize=(10, 6), sharey='row',
                                 gridspec_kw={'height_ratios': [0.8, 1.0]})

    axes1s[0, 0].hist(uni_angles_deg,  bins=bins_hist, weights=weights_uni,
                      color='black', rwidth=0.9)
    axes1s[0, 0].set_title("Uniform Ensemble",  fontweight='bold', fontsize=18)
    axes1s[0, 0].set_ylabel("Probability", fontsize=18)

    axes1s[0, 1].hist(bias_angles_deg, bins=bins_hist, weights=weights_bias,
                      color='black', rwidth=0.9)
    axes1s[0, 1].set_title("Biased Ensemble", fontweight='bold', fontsize=18)

    for ax in axes1s[0]:
        ax.set_xlim(bins_hist[0], bins_hist[-1])
        ax.tick_params(labelbottom=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    for i in range(N_BINS):
        axes1s[1, 0].plot(x_axis_sorted, norm_none[i][sort_idx],
                          color=blue_colors[i], linewidth=2.0)
        axes1s[1, 1].plot(x_axis_sorted, norm_crf[i][sort_idx],
                          color=blue_colors[i], linewidth=2.0)

    axes1s[1, 0].set_ylabel("Response", fontsize=18)

    for c in [0, 1]:
        ax = axes1s[1, c]
        ax.set_xlim(-90, 90)
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin - 0.05 * (ymax - ymin), ymax)
        ax.grid(False)
        ax.set_xlabel("Stimulus Orientation (°)", fontsize=18)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()
