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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # analysis/ -- for `import Surround_simulated_responses`

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from tqdm import tqdm
from simulation_whiten import Frame, V1Dynamics_Surround
from tunings_whiten import V1Tunings
from stimuli_whiten import StimulusGenerator
from analysis import Analytic_responses as ar
import Surround_simulated_responses as SSR   # reuses run_adaptation_phase -- see gain computation below
from typing import Literal
from scipy.linalg import block_diag

N_RF       = 13                    # Primary neurons per receptive field
N_SETS     = 5                    # 1 classical RF (cRF) + 6 surround sets
N_TOTAL    = N_RF * N_SETS         # Full primary-neuron population
CRF_IDX    = 0                     # Which of the 7 sets is the cRF (arbitrary; sets are symmetric)
FRAME_PATH = os.path.join(REPO_ROOT, "data/frames/N13_mercedes_Frame.csv")
# Same target-covariance file V1Dynamics_Surround loads for theta_t (simulation_whiten.py) -
# used below so the PCA whitening diagnostic's "T" matches the live model's target exactly.
TARGET_COV_PATH = os.path.join(REPO_ROOT, "data/target_covs/uniform_target_covariance.csv")
N_matrix = np.ones((N_TOTAL, N_TOTAL))

SIM_SIGMA = 0.15
sigma = SIM_SIGMA
Beta  = 0.5


ENSEMBLE_CONTRAST = 1.0      # Contrast of the adaptation ensembles (baseline & adaptor)
TUNING_WIDTH      = 0.75     # Width of the gaussian stimulus profiles
N_CONTRASTS   = 20           # Number of contrasts to probe with
CRF_CONTRASTS = np.logspace(-2, 0, N_CONTRASTS)
PROBE_CONTRAST = 0.8         
N_PROBES       = 720

# Setting colors for plot lines (designated by what section of the visual field is adapted)
COLOR_NONE   = 'black'
COLOR_CRF    = '#FDE68A'     # pastel yellow
COLOR_NONCRF = 'red'
COLOR_BOTH   = 'darkorange'


def half_wave_rectify(y):      # Estimates primary neuron firing rate from membrane potential
    return np.maximum(y, 0.0) ** 2

def probe_local_profile(input_theta, contrast, tuning_width=TUNING_WIDTH):
    '''Local (single-RF, N_RF-long) Gaussian probe profile, before spatial embedding. Currently
    unused by any figure below (probe_input_drive, which normalizes AFTER replication -- not this
    function's own per-block normalization -- is what every contrast/tuning-curve figure uses);
    kept as a building block for any single-local-block diagnostic added later.'''
    theta_grid = np.linspace(0, np.pi, N_RF, endpoint=False)    # Evenly spaced orientation preferences for neurons
    delta = theta_grid - input_theta                            # Distance between neuron preference from stimulus orientation
    delta = (delta + np.pi / 2) % np.pi - np.pi / 2             # Shift by 90 degrees to match other scripts
    profile = np.exp(-delta**2 / (2 * tuning_width**2))         # Gaussian profile
    profile = contrast * profile / np.linalg.norm(profile)      # Normalize and scale by contrast
    return profile

def probe_input_drive(input_theta, contrast, tuning_width=TUNING_WIDTH):
    '''Always probing with a stimulus that covers both cRF and surround, no matter the adaptation
    state. Normalizes the FULL replicated (N_TOTAL,) vector to unit norm, matching
    Surround_simulated_responses.py's own probe_input_drive exactly -- normalizing each local
    block to unit norm BEFORE replicating (the old behavior here) inflates the total probe energy
    by an extra factor of sqrt(N_SETS), which shifts the effective semi-saturation contrast by
    that same factor (c50 = sigma/sqrt(N_SETS) instead of sigma, for the M=0 baseline). That's
    what was making sigma=0.15 look "wrong" -- the response formula was correct, the probe
    amplitude convention was not. See conversation for the derivation and the sigma=0.35 vs 0.15
    ~sqrt(5)=2.236 numerical match that gave it away.'''
    theta_grid = np.linspace(0, np.pi, N_RF, endpoint=False)
    delta = theta_grid - input_theta
    delta = (delta + np.pi / 2) % np.pi - np.pi / 2
    profile = np.exp(-delta**2 / (2 * tuning_width**2))
    full_profile = np.concatenate([profile] * N_SETS)
    return contrast * full_profile / np.linalg.norm(full_profile)

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


def get_response(stimulus, M, Beta=0.5):
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

    # ---- Gains: loaded from data/optimal_gains/, precomputed by precompute_adapted_gains.py as
    # the MEAN of N=16 independent live V1Dynamics_Surround adaptation runs (RK4,
    # dg/dt=(v-W.T@mu)^2-theta_t, gains_nonneg=True) -- not fit via Analytic_responses.
    # get_optimal_gains_target's closed-form covariance-shrink optimum (diverges from the live
    # circuit: its g>=0 clip is commented out, so it isn't solving the same nonnegativity-
    # constrained optimization -- see conversation). A single live run was also tried directly in
    # this script, but its result varied several percent run-to-run from the unseeded Poisson
    # stream noise; averaging over many seeds offline (once, via precompute_adapted_gains.py)
    # removes that variance while keeping this script itself fast and deterministic. Each cached
    # file's columns are (g_cRF_mean, g_surround_mean, g_cRF_std) -- see
    # data/optimal_gains/meta.json for the seeds/stream-length/duration used to generate them.
    GAINS_DIR = os.path.join(REPO_ROOT, "data", "optimal_gains")
    GAIN_FILE = {
        'adapt CRF only':         'adapt_CRF_only.csv',
        'adapt surround only':    'adapt_surround_only.csv',
        'adapt CRF and surround': 'adapt_CRF_and_surround.csv',
    }
    print("Loading precomputed, seed-averaged adaptation gains...")
    cached_gains = {}
    for cond, fname in GAIN_FILE.items():
        path = os.path.join(GAINS_DIR, fname)
        assert os.path.exists(path), (
            f"Missing precomputed gains at {path} -- run `python precompute_adapted_gains.py` "
            f"from the repo root first."
        )
        cols = np.loadtxt(path, delimiter=",")
        cached_gains[cond] = (cols[:, 0], cols[:, 1])   # (g_cRF_mean, g_surround_mean)
    cached_gains['no adaptation'] = (np.zeros(frame.K), np.zeros(frame.K))

    def M_from_cached(cond):
        '''M = block_diag(W diag(g_cRF) W.T, [W diag(g_surround) W.T] * (N_SETS-1)) -- matches
        V1Dynamics_Surround's own replication assumption (one shared g_surround, tiled across all
        surround RFs; see full_gain_feedback in simulation_whiten.py's _derivatives).'''
        g_cRF, g_surround = cached_gains[cond]
        M_cRF = frame.W @ np.diag(g_cRF) @ frame.W.T
        M_surround = frame.W @ np.diag(g_surround) @ frame.W.T
        return block_diag(M_cRF, *[M_surround] * (N_SETS - 1))

    print("Building M for each condition (from precomputed, seed-averaged gains)...")
    M_by_condition = {cond: M_from_cached(cond) for cond in CONDITIONS}

    stim_gen = StimulusGenerator(N_RF=N_RF, N_SETS=N_SETS, num_angles=N_RF,
                                 stream_length=SSR.ADAPT_STREAM_LENGTH,
                                 tuning_width=TUNING_WIDTH, contrast=ENSEMBLE_CONTRAST)
    adaptor_idx = N_RF // 2
    adaptor_rad = stim_gen.theta_inputs[adaptor_idx]

    uniform_target_covariance = np.loadtxt(TARGET_COV_PATH, delimiter=",")
    assert uniform_target_covariance.shape == (N_RF, N_RF), (
        f"uniform_target_covariance at {TARGET_COV_PATH} has shape "
        f"{uniform_target_covariance.shape}, expected ({N_RF}, {N_RF})."
    )

    # centers_uni/stimuli_uni, centers_bias/stimuli_bias: purely for Figure 2's histogram and the
    # PCA whitening diagnostic below -- a representative ensemble draw, not tied to gain
    # computation (gains come from the cache above), so a single fresh, noisy draw is enough here.
    stream_uni, centers_uni = stim_gen.generate_surround_ensembles(
        'adapt CRF and surround', biased=False, duration=SSR.DURATION,
        add_poisson_noise=True, return_angles=True)
    stimuli_uni = list(stream_uni[:N_RF, :].T)
    stream_bias, centers_bias = stim_gen.generate_surround_ensembles(
        'adapt CRF and surround', biased=True, duration=SSR.DURATION,
        add_poisson_noise=True, return_angles=True)
    stimuli_bias = list(stream_bias[:N_RF, :].T)

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
        for i, c in enumerate(CRF_CONTRASTS):
            probe = probe_input_drive(adaptor_rad, c)
            y = get_response(probe, M_by_condition[cond]) # Fast v steady state
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
            y = get_response(probe, M_by_condition[cond])
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
            y = get_response(probe, M_by_condition[cond])
            resp[:, i] = y[crf_slice]
        return resp

    tc_none = crf_tuning_curves('no adaptation')
    tc_crf  = crf_tuning_curves('adapt CRF only') 

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

   