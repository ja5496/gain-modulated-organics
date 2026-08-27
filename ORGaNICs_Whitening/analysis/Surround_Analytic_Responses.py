'''
Surround_Analytic_Responses.py

Calculates neural firing rates for a known input distribution using analytical expressions.
Interactions between classical receptive field (cRF) neurons and surround (non-cRF) neurons are modeled.
The normalization pool includes all neurons, while adaptation is local to each RF.

Population structure: 7 sets of 13 primary neurons each (91 total). One set (CRF_IDX) is the
classical RF; the other 6 are surround sets. All 7 sets share the same tuning-curve basis and the
same per-RF frame.

Methodology:
1. Optimal gains for a single 13-neuron RF pool are computed once per adaptation condition
   (cRF-only / surround-only / cRF+surround).
2. For each of the 4 adaptation conditions (none / cRF-only / non-cRF-only / both), a block-diagonal
   feedback matrix M (91x91) is assembled from that condition's own per-RF gains, and mu = <y> is
   found via a joint self-consistency loop over the full 91-neuron population (normalization pools
   across all 91; gain feedback stays within each 13-neuron block).
3. Steady-state responses to probe stimuli (shown to the full population)
   are computed in closed form from the probe, optimal gains, and mu.
'''

import os
import sys
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from tqdm import tqdm
from simulation_whiten import Frame
from tunings_whiten import V1Tunings
from stimuli_whiten import StimulusGenerator
from analysis import Analytic_responses as ar
from typing import Literal
from scipy.linalg import block_diag

N_RF       = 13                    # Primary neurons per receptive field
N_SETS     = 7                     # 1 classical RF (cRF) + 6 surround sets
N_TOTAL    = N_RF * N_SETS         # Full primary-neuron population
CRF_IDX    = 0                     # Which of the 7 sets is the cRF (arbitrary; sets are symmetric)
FRAME_PATH = os.path.join(REPO_ROOT, "data/frames/N13_mercedes_Frame.csv")
# Same target-covariance file V1Dynamics_Surround loads for theta_t (simulation_whiten.py) -
# used below so the PCA whitening diagnostic's "T" matches the live model's target exactly.
TARGET_COV_PATH = os.path.join(REPO_ROOT, "data/target_covs/uniform_target_covariance.csv")
N_matrix = np.ones((N_TOTAL, N_TOTAL))
sigma = 0.35
Beta  = 0.5

# Semi-saturation constant the LIVE network actually normalizes with (V1Dynamics_Surround.sigma,
# simulation_whiten.py) -- distinct from this file's own `sigma` above (used only in get_response /
# get_response_new / get_mu's *response* denominator). ar.get_optimal_gains_target's pooled
# denominator (Analytic_responses._pooled_denom) reads the ar module's own `sigma` global, which
# defaults to 0.1 (Analytic_responses.py's own stale default) unless overridden -- must be set to
# THIS value so get_optim's covariance/target is computed against the same normalized stimulus
# environment the online gain dynamics see. Mirrors Surround_simulated_responses.py's own
# "AR.sigma = dyn.sigma" line.
SIM_SIGMA = 0.15


ENSEMBLE_CONTRAST = 1.0      # Contrast of the adaptation ensembles (baseline & adaptor)
TUNING_WIDTH      = 0.75     # Width of the gaussian stimulus profiles
N_CONTRASTS   = 20           # Number of contrasts to probe with
CRF_CONTRASTS = np.logspace(-2, 0, N_CONTRASTS)
PROBE_CONTRAST = 1.0         
N_PROBES       = 720

# Setting colors for plot lines (designated by what section of the visual field is adapted)
COLOR_NONE   = 'black'
COLOR_CRF    = '#FDE68A'     # pastel yellow
COLOR_NONCRF = 'red'
COLOR_BOTH   = 'darkorange'


def half_wave_rectify(y):      # Estimates primary neuron firing rate from membrane potential
    return np.maximum(y, 0.0) ** 2

def probe_local_profile(input_theta, contrast, tuning_width=TUNING_WIDTH):
    '''Local (single-RF, N_RF-long) Gaussian probe profile, before spatial embedding.'''
    theta_grid = np.linspace(0, np.pi, N_RF, endpoint=False)    # Evenly spaced orientation preferences for neurons
    delta = theta_grid - input_theta                            # Distance between neuron preference from stimulus orientation
    delta = (delta + np.pi / 2) % np.pi - np.pi / 2             # Shift by 90 degrees to match other scripts
    profile = np.exp(-delta**2 / (2 * tuning_width**2))         # Gaussian profile
    profile = contrast * profile / np.linalg.norm(profile)      # Normalize and scale by contrast
    return profile

def probe_input_drive(input_theta, contrast, tuning_width=TUNING_WIDTH):
    '''Always probing with a stimulus that covers both cRF and surround, no matter the adaptation state.'''
    return np.concatenate([probe_local_profile(input_theta, contrast, tuning_width)] * N_SETS)

def block_diag_M(frame, g_opt, adapt_location: Literal['adapt CRF only', 'adapt surround only', 'adapt CRF and surround', 'no adaptation']):
    '''M = W @ diag(g) W.T '''
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

def block_diag_W(frame, g_opt, adapt_location: Literal['adapt CRF only', 'adapt surround only', 'adapt CRF and surround', 'no adaptation']):
    '''
    Block-diagonal embedding of the shared per-RF frame
    '''
    K = frame.shape[1]
    zeros_local = np.zeros(K)
    W_full = block_diag(*[frame] * N_SETS)
    match adapt_location:
        case 'adapt CRF only':
            g_full = np.concatenate([g_opt] + [zeros_local] * (N_SETS - 1))
        case 'adapt surround only':
            g_full = np.concatenate([zeros_local] + [g_opt] * (N_SETS - 1))
        case 'adapt CRF and surround':
            g_full = np.concatenate([g_opt] * N_SETS)
        case 'no adaptation':
            g_full = np.concatenate([zeros_local] * N_SETS)
    return W_full, g_full

def full_spatial_stimuli(stimuli, adapt_location: Literal['adapt CRF only', 'adapt surround only', 'adapt CRF and surround', 'no adaptation']):
    match adapt_location:
        case 'adapt CRF only':
            baseline = np.full(N_RF, 0.2)            
            repeats = N_SETS - 1
            full_stimuli = np.concatenate([stimuli] + [baseline] * repeats)
            return full_stimuli
        case 'adapt surround only':
            baseline = np.full(N_RF, 0.2)
            repeats = N_SETS - 1
            full_stimuli = np.concatenate([baseline] + [stimuli] * repeats)
            return full_stimuli
        case 'adapt CRF and surround':
            full_stimuli = np.concatenate([stimuli] * N_SETS)
            return full_stimuli
        case 'no adaptation':
            baseline = np.full(N_RF, 0.2)            
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
    ''' 
    Assumes the time constant of the variance interneuron is extremely slow so that it settles 
    as an average: v = W.T @ mu. Since this is effectively a constant vector, it cannot factorize 
    into the proper covariance tranformation on the inputs. 
    '''
    gain_feedback = M @ mu                                      # Gain feedback ~ constant vector from slow v, g
    z_prime = 2 * (Beta * stimulus - gain_feedback)             # Modified approx input drive
    y = z_prime / np.sqrt(sigma**2 + N_matrix @ (z_prime**2))   # Normalize the approx input drive

    rectified_y = half_wave_rectify(y)
    return rectified_y

def get_response_new(stimulus, M, Beta=0.5):
    ''' Calculation of fixed point with fast v dynamics. Response is consistent with normalized,
    whitened inputs.'''

    z_prime = np.linalg.inv(np.eye(N_TOTAL) + M) @ stimulus
    y = z_prime / np.sqrt(sigma**2 + N_matrix @ (z_prime**2))
    rectified_y = half_wave_rectify(y)

    return rectified_y

CONDITIONS = ['no adaptation', 'adapt CRF only', 'adapt surround only', 'adapt CRF and surround']
CONDITION_LABEL = {
    'no adaptation':          'no adaptation',
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

    # ---- Local (single-RF) biased ensemble: shared context for every adaptation case ----
    stim_gen = StimulusGenerator(N=N_RF, num_angles=N_RF, stream_length=N_RF,
                                 tuning_width=TUNING_WIDTH, contrast=ENSEMBLE_CONTRAST)

    print("Generating local uniform and biased ensembles...")
    seq_uni, centers_uni = stim_gen.generate_input_ensembles(
        biased=False, return_angles=True, duration=1)
    stimuli_uni = list(seq_uni.T)

    adaptor_idx = N_RF // 2
    adaptor_rad = stim_gen.theta_inputs[adaptor_idx]

    # Build the biased stream manually for equal non-adaptor representation 
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
    seq_bias = stim_gen.contrast * seq_bias / np.linalg.norm(seq_bias)
    stimuli_bias = list(seq_bias.T)

    uniform_target_covariance = np.loadtxt(TARGET_COV_PATH, delimiter=",")
    assert uniform_target_covariance.shape == (N_RF, N_RF), (
        f"uniform_target_covariance at {TARGET_COV_PATH} has shape "
        f"{uniform_target_covariance.shape}, expected ({N_RF}, {N_RF})."
    )

    # ---- Optimal gains, one set per stimulus condition ----
    GAIN_CONDITIONS = ['adapt CRF only', 'adapt surround only', 'adapt CRF and surround']
    # NOTE: pool_uni is unused below -- get_optimal_gains_target is always called with an explicit
    # target_covariance (uniform_target_covariance), so its uniform_stimuli/pool_uniform_stimuli
    # re-derivation path (the only place a uniform-ensemble pool would be needed) never runs. Kept
    # here only in case that path is exercised later.
    pool_uni = np.array([full_spatial_stimuli(z, 'adapt CRF and surround') for z in stimuli_uni])
    print("Computing optimal gains...")
    ar.sigma = SIM_SIGMA   # match the live network's semi-saturation constant (see SIM_SIGMA above)
    g_opt_by_condition = {}
    for cond in GAIN_CONDITIONS:
        # Full 91-neuron population drive under THIS condition -- same numerator (stimuli_bias,
        # this RF's own local biased drive) for every condition, but the pool (denominator) differs
        # per condition since full_spatial_stimuli places the biased ensemble in a different
        # cRF/surround location each time. Passing this as pool_stimuli makes get_optimal_gains_target
        # normalize by the FULL population's pooled energy (matching the live circuit's global
        # normalization pool, V1Dynamics_Surround.N_matrix = ones((N_TOT, N_TOT))) instead of just
        # this RF's own N_RF=13-neuron local energy -- i.e. gains are now computed against the actual
        # normalized stimulus environment each condition produces, not an unsuppressed local proxy.
        pool_bias = np.array([full_spatial_stimuli(z, cond) for z in stimuli_bias])
        g_opt_by_condition[cond] = ar.get_optimal_gains_target(
            stimuli_bias, frame.W, label=f'biased ({cond})',
            target_covariance=uniform_target_covariance,
            pool_stimuli=pool_bias)

    print("Building M for each condition...")
    M_by_condition = {
        cond: block_diag_M(frame.W, g_opt_by_condition[cond], cond) if cond in g_opt_by_condition
              else block_diag_M(frame.W, None, cond)
        for cond in CONDITIONS
    }

    # Block-diagonal (W_full, g_full) per condition 
    Wg_full_by_condition = {
        cond: block_diag_W(frame.W, g_opt_by_condition[cond], cond) if cond in g_opt_by_condition
              else block_diag_W(frame.W, None, cond)
        for cond in CONDITIONS
    }

    # ==========================================================================
    # CHECK 1 -- Plot probe_input_drive, centered at the biased ensemble's adaptor
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
        #mu_by_condition[cond] = get_mu(full_stimuli, M_by_condition[cond])


    # ==========================================================================
    # Figure 1 -- Contrast response functions of the cRF neuron that prefers the adaptor
    # ==========================================================================
    print("Computing contrast response functions...")
    crf_target_idx = CRF_IDX * N_RF + adaptor_idx

    def crf_curve(cond):
        resp = np.zeros(N_CONTRASTS)
        W_full, g_full = Wg_full_by_condition[cond]
        for i, c in enumerate(CRF_CONTRASTS):
            probe = probe_input_drive(adaptor_rad, c)
            #y = get_response(probe, mu_by_condition[cond], M_by_condition[cond])        # Approximates slow v steady state
            y = get_response_new(probe, M_by_condition[cond]) # Fast v steady state
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
    # CHECK 3 -- Tuning curve of a neuron adjacent to the adaptor-preferring neuron
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
            y = get_response_new(probe, M_by_condition[cond])
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
    peak_shift_deg = {cond: ((peak_deg[cond] - control_peak + 90) % 180) - 90 for cond in TUNING_CONDITIONS}

    print("Flank neuron tuning-preference shift (vs. control):")
    for cond in TUNING_CONDITIONS:
        print(f"  {CONDITION_LABEL[cond]:25s} peak={peak_deg[cond]:7.2f} deg   "
              f"shift={peak_shift_deg[cond]:+6.2f} deg")

    # Compact legend labels folding in each condition's peak shift (in degrees) relative to
    # the no-adaptation control, so the shift is readable directly off the plot.
    FLANK_LEGEND_LABEL = {
        'no adaptation':       'No adaptation',
        'adapt surround only': f"Surround: {peak_shift_deg['adapt surround only']:+.2f}°",
        'adapt CRF only':      f"cRF: {peak_shift_deg['adapt CRF only']:+.2f}°",
    }

    fig_flank, ax_flank = plt.subplots(figsize=(7, 5.5))
    for cond in TUNING_CONDITIONS:
        ax_flank.plot(probe_angles_deg, flank_curves[cond], color=CONDITION_COLOR[cond],
                      linewidth=3.5, label=FLANK_LEGEND_LABEL[cond])

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
    ax_flank.legend(fontsize=15, frameon=False)
    plt.tight_layout(); plt.show()

    # ==========================================================================
    # FIGURE 2 Tuning Curves 
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
            #y = get_response(probe, mu_by_condition[cond], M_by_condition[cond])
            y = get_response_new(probe, M_by_condition[cond])
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

    # ==========================================================================
    # PCA whitening diagnostic: the biased single-RF stimulus ensemble (stimuli_bias, the
    # same array get_optimal_gains_target above is fit on) before vs. after three
    # covariance transforms:
    #   (1) full whitening, from the RAW stimuli's own covariance.
    #   (2) "T" - Analytic_responses.get_optimal_gains_target's shrink-to-target transform,
    #       built from the covariance of the NORMALIZED stimuli z/sqrt(sigma^2+||z||^2)
    #       (no Beta: Beta's only role is cancelling the factor of 2 in the y-dynamics fixed
    #       point y* = 2*(Beta*z - fb)/..., it has no place in this covariance/basis calc) -
    #       this is the actual quantity divisive normalization drives responses toward, so T
    #       is derived on the same footing as the target covariance it shrinks toward. Only
    #       eigen-directions whose variance exceeds the model's target get pulled down to
    #       that target; directions already below it are untouched. Uses the SAME target
    #       covariance file V1Dynamics_Surround loads for theta_t (simulation_whiten.py /
    #       TARGET_COV_PATH), so T here matches what the live gain-feedback circuit in
    #       Surround_simulated_responses.py actually approximates.
    #   (3) top-eigenvalue clip, from the RAW stimuli's own covariance (same basis as (1)):
    #       leaves every eigen-direction untouched EXCEPT the single largest eigenvalue,
    #       whose variance is shrunk down to exactly the second-largest eigenvalue - isolates
    #       how much of the stimuli-vs-response structural difference the single most
    #       dominant direction (e.g. the adaptor orientation) accounts for on its own.
    # Same red=stimuli / blue=response color scheme as the PCA scatter in
    # Surround_simulated_responses.py's Figure 7. 3 panels (one per transform), each
    # overlaying its stimuli/response pair on a SHARED joint-PCA frame so their relative
    # scale is directly visible (not each rescaled to fill its own axes).
    # ==========================================================================
    print("Building PCA whitening diagnostic (biased ensemble)...")
    stimuli_bias_matrix = np.array(stimuli_bias)   # (n_bias, N_RF)

    Covariance_bias = np.cov(stimuli_bias_matrix, rowvar=False)
    eigvals_bias, eigvecs_bias = np.linalg.eigh(Covariance_bias)
    safe_lambdas = np.maximum(eigvals_bias, 1e-9)

    # (1) Full whitening: Cov^-1/2, raw-stimuli basis.
    W_whiten_full = eigvecs_bias @ np.diag(np.sqrt(0.02) / np.sqrt(safe_lambdas)) @ eigvecs_bias.T
    stim_whitened_full = (W_whiten_full @ stimuli_bias_matrix.T).T

    # (2) T: shrink-to-target whitening, normalized-stimuli basis. sigma=0.25 matches
    # frame_whiten.compute_uniform_target_covariance / V1Dynamics_Surround's default - the
    # same sigma that produced target_covariance itself.
    SIGMA_NORM = 0.25
    pooled_energy = np.sum(stimuli_bias_matrix ** 2, axis=1, keepdims=True)   # (n_bias, 1)
    stimuli_bias_normalized = stimuli_bias_matrix / np.sqrt(SIGMA_NORM**2 + pooled_energy)

    Covariance_norm = np.cov(stimuli_bias_normalized, rowvar=False)
    eigvals_norm, eigvecs_norm = np.linalg.eigh(Covariance_norm)
    safe_lambdas_norm = np.maximum(eigvals_norm, 1e-9)

    target_covariance = np.loadtxt(TARGET_COV_PATH, delimiter=",")
    target_variance = np.mean(np.diag(target_covariance))
    d_target = np.minimum(1.0, np.sqrt(target_variance / safe_lambdas_norm))
    T = eigvecs_norm @ np.diag(d_target) @ eigvecs_norm.T
    stim_T = (T @ stimuli_bias_normalized.T).T

    # (3) Top-eigenvalue clip: same raw-stimuli eigenbasis as (1), but only the single
    # largest eigenvalue is shrunk - down to exactly the second-largest - and every other
    # eigen-direction's coefficient stays at 1.
    sorted_desc = np.argsort(eigvals_bias)[::-1]
    top_idx, second_idx = sorted_desc[0], sorted_desc[1]
    second_highest_eigval = eigvals_bias[second_idx]
    d_clip_top = np.ones_like(safe_lambdas)
    d_clip_top[top_idx] = np.sqrt(second_highest_eigval / safe_lambdas[top_idx])
    W_clip_top = eigvecs_bias @ np.diag(d_clip_top) @ eigvecs_bias.T
    stim_clip_top = (W_clip_top @ stimuli_bias_matrix.T).T

    def pca_project_pair(raw, transformed):
        '''Joint PCA on [raw; transformed] so both scatters share one 2D coordinate frame -
        preserves their relative scale instead of each getting its own independent axes.'''
        combined = np.concatenate([raw, transformed], axis=0)
        centered = combined - combined.mean(axis=0, keepdims=True)
        cov2 = np.cov(centered, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov2)
        top2 = eigvecs[:, np.argsort(eigvals)[::-1][:2]]
        proj = centered @ top2
        n = raw.shape[0]
        return proj[:n], proj[n:]

    def cov_ellipse(ax, points, color, n_std=1.0):
        '''Draws a 2*n_std-sigma covariance ellipse characterizing a 2D point cloud and
        returns its (eigval_1, eigval_2) variances along the major/minor axes.'''
        center = points.mean(axis=0)
        cov2 = np.cov(points, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov2)
        order = np.argsort(eigvals)[::-1]
        eigvals, eigvecs = eigvals[order], eigvecs[:, order]
        angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
        width, height = 2 * n_std * np.sqrt(np.maximum(eigvals, 0))
        ax.add_patch(Ellipse(center, width, height, angle=angle,
                              facecolor='none', edgecolor=color, linewidth=2.5))
        return eigvals

    raw_proj_full, whitened_proj_full = pca_project_pair(stimuli_bias_matrix, stim_whitened_full)
    norm_proj_T,   T_proj             = pca_project_pair(stimuli_bias_normalized, stim_T)
    raw_proj_clip, clip_proj          = pca_project_pair(stimuli_bias_matrix, stim_clip_top)

    fig_pca, (ax_full, ax_T, ax_clip) = plt.subplots(1, 3, figsize=(19, 6.5))
    panels = [
        (ax_full, raw_proj_full, whitened_proj_full, 'Stimuli',            'Whitened Response',
         r"Full Whitening ($\Sigma^{-1/2}$)"),
        (ax_T,    norm_proj_T,   T_proj,              'Normalized Stimuli', 'T Response',
         r"Target-Covariance Whitening ($T$)"),
        (ax_clip, raw_proj_clip, clip_proj,           'Stimuli',            'Clipped Response',
         "Top-Eigenvalue Clip"),
    ]
    for ax, raw_proj, resp_proj, raw_label, resp_label, title in panels:
        # alpha < 1 so exactly-overlapping points compound into a visibly darker/denser
        # patch instead of one opaque marker silently hiding how many points land there.
        ax.scatter(raw_proj[:, 0],  raw_proj[:, 1],  color='red',  alpha=0.6, s=45, label=raw_label)
        ax.scatter(resp_proj[:, 0], resp_proj[:, 1], color='blue', alpha=0.6, s=45, label=resp_label)
        eig_raw  = cov_ellipse(ax, raw_proj,  'red',  n_std=1.0)
        eig_resp = cov_ellipse(ax, resp_proj, 'blue', n_std=1.0)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(fontsize=12, loc='upper right', frameon=False)
        ax.text(0.02, 0.98, rf"{raw_label} $\lambda$: {eig_raw[0]:.3g}, {eig_raw[1]:.3g}",
                transform=ax.transAxes, color='red', fontsize=10, fontweight='bold', va='top', ha='left')
        ax.text(0.02, 0.91, rf"{resp_label} $\lambda$: {eig_resp[0]:.3g}, {eig_resp[1]:.3g}",
                transform=ax.transAxes, color='blue', fontsize=10, fontweight='bold', va='top', ha='left')

    fig_pca.suptitle("PCA Whitening Diagnostic (Biased Ensemble)", fontsize=18, fontweight='bold')
    plt.tight_layout()

    # ==========================================================================
    # FIGURE 2b -- Same tuning-curve plot as Figure 2, but the "adapted" panel is produced by
    # simply applying the top-eigenvalue-clip transform (W_clip_top, from the "Top-Eigenvalue
    # Clip" ellipse panel above) directly to the probe stimulus, instead of running the actual
    # optimal-gain adaptation model. M/mu are held at the 'no adaptation' (all-zero) values used
    # for tc_none - there is no gain-feedback loop here at all, only the bare linear clip of the
    # single dominant stimulus eigen-direction, before divisive normalization. This isolates how
    # much of Figure 2's tuning-curve effect (surround suppression / sharpening near the adaptor)
    # that one clipped eigenvalue can reproduce on its own, with no adaptive circuitry involved.
    # ==========================================================================
    print("Recreating Figure 2 with the top-eigenvalue clip standing in for the gain model...")

    M_none  = M_by_condition['no adaptation']

    def crf_tuning_curves_clip():
        resp = np.zeros((N_RF, N_PROBES))
        for i, ang in enumerate(probe_angles):
            local_probe   = probe_local_profile(ang, PROBE_CONTRAST)   # (N_RF,) local drive
            local_clipped = W_clip_top @ local_probe                    # top-eigenvalue clip
            probe = np.concatenate([local_clipped] * N_SETS)
            y = get_response_new(probe, M_none)
            resp[:, i] = y[crf_slice]
        return resp

    tc_clip = crf_tuning_curves_clip()
    binned_clip = bin_by_preference(tc_clip, tunings.theta)
    # Normalized against the SAME 'no adaptation' bin min/max as Figure 2, so the two figures'
    # right-hand panels are directly comparable.
    norm_clip = (binned_clip - bin_min) / (bin_max - bin_min + 1e-9)

    fig1s_clip, axes1s_clip = plt.subplots(2, 2, figsize=(10, 6), sharey='row',
                                            gridspec_kw={'height_ratios': [0.8, 1.0]})

    axes1s_clip[0, 0].hist(uni_angles_deg, bins=bins_hist, weights=weights_uni,
                           color='black', rwidth=0.9)
    axes1s_clip[0, 0].set_title("Uniform Ensemble", fontweight='bold', fontsize=18)
    axes1s_clip[0, 0].set_ylabel("Probability", fontsize=18)

    axes1s_clip[0, 1].hist(bias_angles_deg, bins=bins_hist, weights=weights_bias,
                           color='black', rwidth=0.9)
    axes1s_clip[0, 1].set_title("Biased Ensemble", fontweight='bold', fontsize=18)

    for ax in axes1s_clip[0]:
        ax.set_xlim(bins_hist[0], bins_hist[-1])
        ax.tick_params(labelbottom=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    for i in range(N_BINS):
        axes1s_clip[1, 0].plot(x_axis_sorted, norm_none[i][sort_idx],
                               color=blue_colors[i], linewidth=2.0)
        axes1s_clip[1, 1].plot(x_axis_sorted, norm_clip[i][sort_idx],
                               color=blue_colors[i], linewidth=2.0)

    axes1s_clip[1, 0].set_ylabel("Response", fontsize=18)
    axes1s_clip[1, 0].set_title("No Adaptation", fontsize=14, fontweight='bold')
    axes1s_clip[1, 1].set_title("Top-Eigenvalue Clip", fontsize=14, fontweight='bold')

    for c in [0, 1]:
        ax = axes1s_clip[1, c]
        ax.set_xlim(-90, 90)
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin - 0.05 * (ymax - ymin), ymax)
        ax.grid(False)
        ax.set_xlabel("Stimulus Orientation (°)", fontsize=18)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig1s_clip.suptitle("Tuning Curves via Top-Eigenvalue Clip (cf. Figure 2's gain-adapted model)",
                         fontsize=14, fontweight='bold')
    plt.tight_layout()

    # NOTE: the eigenvalue-difference diagnostic (biased vs. uniform covariance spectrum
    # comparison) has moved to figures.py, which now also covers the Poisson-variance and
    # double-peaked ensemble cases. Run `python analysis/figures.py` for that figure.

    plt.show()
