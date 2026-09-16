import os
import sys
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from simulation_whiten import Frame, V1Dynamics_Surround
from tunings_whiten import V1Tunings
from stimuli_whiten import StimulusGenerator
from typing import Literal
import Analytic_responses as AR
import frame_whiten

N_RF       = 13                    # Number of primary neurons per receptive field
N_SETS     = 1                     # 1 classical RF (cRF) + 6 surround sets
CRF_IDX    = 0                     # Index of cRF (arbitrary; sets are symmetric)
FRAME_PATH = os.path.join(REPO_ROOT, "data/frames/N13_mercedes_Frame.csv")
TARGET_COV_PATH = os.path.join(REPO_ROOT, "data/target_covs/uniform_target_covariance_low_c.csv")

ENSEMBLE_CONTRAST    = 0.6       # contrast of the adaptation ensembles (baseline & adaptor)
THETA_T_CONTRAST     = 0.25      # contrast used ONLY to calibrate theta_t (see run_adaptation_phase)
TUNING_WIDTH         = 0.75
ADAPT_STREAM_LENGTH  = 1500000  # 101920   # timesteps of adaptation stimulus (dt=0.1 -> 1092s =~ 11x tau_g)
DURATION             = 200     # timesteps each individual adaptation stimulus is held for
N_SETTLE_STEPS       = 1500     # timesteps to settle y/u/a to steady state per probe (dt=0.1 -> 30s)

N_CONTRASTS    = 20
CRF_CONTRASTS  = np.logspace(-2, 0, N_CONTRASTS)
PROBE_CONTRAST = 0.6
N_PROBES       = 180

# Setting colors for plot lines (designated by what section of the visual field is adapted)
COLOR_NONE   = 'black'
COLOR_CRF    = '#FDE68A'     # pastel yellow
COLOR_NONCRF = 'red'
COLOR_BOTH   = 'darkorange'

# Full adaptation-phase histories (y_hist, g_cRF_hist, g_surround_hist, stream), keyed by
# condition - populated by run_adaptation_phase below, reused by Figures 2 and 7 without any
# additional simulation runs.
SIM_HISTORY = {}

CONDITIONS = ['no adaptation', 'adapt CRF only']
ACTIVE_CONDITIONS = CONDITIONS
CONDITION_LABEL = {
    'no adaptation':          'No adaptation',
    'adapt CRF only':         'Classical RF adapted',
}

CONDITION_COLOR = {
    'no adaptation':          COLOR_NONE,
    'adapt CRF only':         COLOR_CRF,
}

# "no adaptation" -> unbiased ensemble to both regions: the zero-gain-feedback control condition,
# and also what run_adaptation_phase uses to calibrate theta_t (see below).
ADAPT_LOCATION_FOR_COND = {
    'no adaptation':          'adapt CRF and surround',
    'adapt CRF only':         'adapt CRF only',
}

BIASED_FOR_COND = {
    'no adaptation':          False,
    'adapt CRF only':         True,
}


def run_adaptation(dyn, stim_gen, cond, adapt_location=None, biased=None):
    '''
    Simulates the adaptation state for one condition. Returns (g_cRF, g_surround, v_cRF,
    v_surround, mu_cRF, mu_surround, stream) - stream is cached so later diagnostics can reuse
    this exact run instead of re-simulating.

    adapt_location/biased default to ADAPT_LOCATION_FOR_COND[cond]/BIASED_FOR_COND[cond], but can
    be overridden to run a condition not registered in those dicts (e.g. Figure 6's extra
    "adapt CRF only, uniform ensemble" control - same adapt_location as 'adapt CRF only', biased=False).

    "no adaptation" runs a real, unbiased ensemble to both regions - needed to calibrate
    theta_t (see dyn.calibrate_theta_t) - but still forces zero gain feedback in the returned
    values: it's the pure-normalization control condition, not genuine adaptation. Runs before
    the other 3 conditions (first in CONDITIONS), so they adapt against the calibrated target.
    Correctness of the calibration depends on THIS run's own g_cRF/g_surround having stayed at
    exactly zero throughout - guaranteed by theta_t's sentinel value at V1Dynamics_Surround
    construction (see there), not by anything in this function.

    This reference stream is generated at THETA_T_CONTRAST, not ENSEMBLE_CONTRAST - stim_gen's
    own contrast is temporarily swapped for this one call and restored immediately after, so the
    calibration measures variance against a fixed internal prior rather than against whatever
    contrast the actual experiment happens to use for its adaptation ensembles. (Nothing else
    about the stream changes: same adapt_location/biased/duration/noise as the other conditions.)

    For the other conditions, whichever region does NOT get the varying/adaptor ensemble only
    sees the flat, orientation-less baseline, so its gain feedback is forced to zero too.
    '''
    K, N_RF = dyn.frame.K, dyn.N_RF
    adapt_location = ADAPT_LOCATION_FOR_COND[cond] if adapt_location is None else adapt_location
    biased = BIASED_FOR_COND[cond] if biased is None else biased

    if cond == 'no adaptation':
        true_contrast = stim_gen.contrast
        stim_gen.contrast = THETA_T_CONTRAST
        try:
            stream, centers = stim_gen.generate_surround_ensembles(
                adapt_location, biased=biased, duration=DURATION,
                add_poisson_noise=True, return_angles=True)
        finally:
            stim_gen.contrast = true_contrast
    else:
        stream, centers = stim_gen.generate_surround_ensembles(
            adapt_location, biased=biased, duration=DURATION,
            add_poisson_noise=True, return_angles=True)

    if cond == 'no adaptation':
        (y_hist, u_hist, a_hist, g_cRF_hist, g_surround_hist, v_cRF_hist, v_surround_hist,
         mu_cRF_hist, mu_surround_hist) = dyn.run_simulation(stream)
        SIM_HISTORY[cond] = dict(y_hist=y_hist, g_cRF_hist=g_cRF_hist,
                                  g_surround_hist=g_surround_hist, stream=stream)

        assert np.all(g_cRF_hist == 0) and np.all(g_surround_hist == 0), (
            "'no adaptation' run's own gains moved away from zero - theta_t's sentinel "
            "(see V1Dynamics_Surround.__init__) no longer holds, or calibrate_theta_t was "
            "already called on this dyn instance. The calibration below would be measuring a "
            "partially-adapted reference, not a genuinely unbiased one."
        )
        dyn.calibrate_theta_t(v_cRF_hist, v_surround_hist, mu_cRF_hist, mu_surround_hist,
                              circular_target=True)

        zeros_K = np.zeros(K)
        return (zeros_K, zeros_K, v_cRF_hist[:, -1], v_surround_hist[:, -1],
                mu_cRF_hist[:, -1], mu_surround_hist[:, -1], (stream, centers))

    (y_hist, u_hist, a_hist, g_cRF_hist, g_surround_hist, v_cRF_hist, v_surround_hist,
     mu_cRF_hist, mu_surround_hist) = dyn.run_simulation(stream)
    SIM_HISTORY[cond] = dict(y_hist=y_hist, g_cRF_hist=g_cRF_hist,
                              g_surround_hist=g_surround_hist, stream=stream)

    N_TOT = dyn.N_RF * dyn.N_SETS
    state = dyn.last_state
    g_cRF       = state[3*N_TOT:3*N_TOT+K]
    g_surround  = state[3*N_TOT+K:3*N_TOT+2*K]
    v_cRF       = state[3*N_TOT+2*K:3*N_TOT+3*K]
    v_surround  = state[3*N_TOT+3*K:3*N_TOT+4*K]
    mu_cRF      = state[3*N_TOT+4*K:3*N_TOT+4*K+N_RF]
    mu_surround = state[3*N_TOT+4*K+N_RF:3*N_TOT+4*K+2*N_RF]

    # Confirm the fix actually holds for this condition: theta_t must sit below at least
    # SOME interneurons' achieved variance, or every gain is clipped to zero (see
    # dyn.calibrate_theta_t's docstring). Checked directly here, every run, rather than
    # trusted - a warning below means this condition's stimulus statistics didn't exceed
    # the calibrated target anywhere, not that the calibration itself is broken.
    g_active = g_cRF
    n_active = int(np.sum(g_active > 1e-3))
    print(f"  [{cond}] gains active (>1e-3): {n_active}/{K} interneurons "
          f"(mean={g_active.mean():.4g}, max={g_active.max():.4g})")
    if n_active == 0:
        print(f"  WARNING: [{cond}] every gain collapsed to zero - this condition's stimulus "
              f"never drove any interneuron's variance above the calibrated theta_t.")

    # g_surround/v_surround/mu_surround stay exactly zero at N_SETS=1 (see _derivatives'
    # N_SETS>=2 guard) - returned anyway so this matches the 'no adaptation' branch's 7-tuple
    # shape above (same convention as Surround_simulated_responses.py's run_adaptation_phase).
    return g_cRF, g_surround, v_cRF, v_surround, mu_cRF, mu_surround, (stream, centers)

# ==========================================================================
# Whitening-error diagnostic (see Panel 2 below): rather than the covariance of a running,
# empirically-simulated response (noisy, and lags behind the live gain state), the error at
# each sampled time step is computed ANALYTICALLY from the current gains via
#     Cyy(t) = J(t) @ A(t) @ Cxx_raw @ A(t).T @ J(t).T
# where A(t) = (I + W diag(g_cRF(t)) W^T)^-1 is the linear gain-feedback fixed point
# (get_response_offline's operator; see Surround_simulated_responses.py), and J(t) is the
# Jacobian of the divisive-normalization nonlinearity y = z'/sqrt(sigma^2 + N(z'^2)),
# evaluated at z' = A(t) @ z0 for a single CLEAN (no Poisson noise), fixed-contrast probe z0 -
# clean so the linearization point is deterministic rather than itself noisy, and fixed-contrast
# so (with the all-ones N_matrix at N_SETS=1) the normalization's pooled denominator is the same
# regardless of which orientation z0 happens to be.
# ==========================================================================
ERROR_STRIDE = 1000    # subsample the (K, ADAPT_STREAM_LENGTH) gain history
ERROR_TYPE   = 'operator'    # passed straight to AR.compute_error ('fro' | 'spectral' | 'operator')
GAIN_SUBSET_N = 5       # size of the "small set" of gains averaged for Panel 1

SEED = 0   # fixes the shared adaptation stream / clean probe stream, drawn before any frame
           # generation below so their values don't depend on whether a frame is freshly built
           # or loaded from a cached CSV (see ensure_frame_csv).

FRAME_DIR = os.path.join(REPO_ROOT, "data/frames")
MERCEDES_2X_K = 182   # 2x the baseline mercedes frame's K = N_RF*(N_RF+1)/2 = 91

# Clean-ensemble generation matches frame_whiten.compute_uniform_target_covariance's own
# recipe exactly (same N_RF, contrast, stream_length, no noise) - that function is what
# produced uniform_target_covariance_low_c.csv in the first place.
CLEAN_CONTRAST = 0.1
CLEAN_STREAM_LENGTH = 10920


def ensure_frame_csv(path, build_fn):
    '''Builds and caches a frame (via build_fn() -> W array) to `path` if it doesn't already
    exist, so re-running this script doesn't rebuild a (possibly slow) frame from scratch.'''
    if not os.path.exists(path):
        print(f"  Building and caching frame -> {path}")
        W = build_fn()
        np.savetxt(path, W, delimiter=",")
    else:
        print(f"  Using cached frame -> {path}")
    return path


def analytic_response_operator(dyn, g_cRF, z0):
    '''
    M_total = J @ A: the linearized end-to-end operator from raw stimulus to response at a
    given (fixed) gain vector g_cRF - A is the gain-feedback fixed point, J the Jacobian of
    the divisive normalization at A @ z0 (see the module docstring above). Shared by
    whitening_error_trace (per time step, reduced to a scalar via compute_error) and the
    final-transformed-covariance figure (used directly as a matrix).
    '''
    W = dyn.frame.W
    N = W.shape[0]
    I_N = np.eye(N)
    A = np.linalg.inv(I_N + W @ np.diag(g_cRF) @ W.T)
    z_prime = A @ z0
    d = dyn.sigma**2 + dyn.N_matrix @ (z_prime**2)
    J = np.diag(1.0 / np.sqrt(d)) - np.diag(z_prime / d**1.5) @ dyn.N_matrix @ np.diag(z_prime)
    return J @ A


def whitening_error_trace(dyn, g_cRF_hist, Cxx_raw, z0, stride=ERROR_STRIDE, error_type=ERROR_TYPE):
    '''Analytic whitening-error trace (see module docstring above) at every `stride`-th step.'''
    T = g_cRF_hist.shape[1]
    steps = np.arange(0, T, stride)
    errors = np.zeros(len(steps))
    for i, t in enumerate(tqdm(steps, desc="  whitening error", leave=False)):
        M_total = analytic_response_operator(dyn, g_cRF_hist[:, t], z0)
        errors[i] = AR.compute_error(M_total, Cxx_raw, clamp=True,
                                      target_covariance=dyn.uniform_target_covariance,
                                      error_type=error_type)
    return steps, errors


def gain_subset_average(g_cRF_hist, n_subset=GAIN_SUBSET_N, stride=ERROR_STRIDE):
    '''Mean of a small, evenly-spaced subset of gains at every `stride`-th step.'''
    K, T = g_cRF_hist.shape
    idx = np.linspace(0, K - 1, min(n_subset, K)).astype(int)
    steps = np.arange(0, T, stride)
    return steps, g_cRF_hist[idx][:, steps].mean(axis=0)


if __name__ == "__main__":

    np.random.seed(SEED)

    print("Generating the shared adaptation-phase stimulus stream...")
    tunings = V1Tunings(N=N_RF)
    stim_gen = StimulusGenerator(N_RF=N_RF, N_SETS=N_SETS, num_angles=N_RF,
                                  stream_length=ADAPT_STREAM_LENGTH,
                                  tuning_width=TUNING_WIDTH, contrast=ENSEMBLE_CONTRAST)
    shared_stream, shared_centers = stim_gen.generate_surround_ensembles(
        'adapt CRF only', biased=True, duration=DURATION, add_poisson_noise=True, return_angles=True)

    print("Generating the clean (no-noise, single-contrast) reference ensemble...")
    clean_stim_gen = StimulusGenerator(N_RF=N_RF, N_SETS=1, num_angles=N_RF,
                                        stream_length=CLEAN_STREAM_LENGTH, contrast=CLEAN_CONTRAST)
    clean_stream = clean_stim_gen.generate_surround_ensembles(
        'adapt CRF only', biased=False, add_poisson_noise=False)   # (N_RF, T_clean)
    Cxx_raw = np.cov(clean_stream)
    z0 = clean_stream[:, 0]   # any one clean sample - normalization scales them all alike (see above)

    C_zz_uniform = np.loadtxt(TARGET_COV_PATH, delimiter=",")   # uniform_target_covariance_low_c

    print("Building frame variants...")
    frame_paths = {
        'mercedes_baseline': FRAME_PATH,
        'mercedes_2x': ensure_frame_csv(
            os.path.join(FRAME_DIR, f"N{N_RF}_mercedes_K{MERCEDES_2X_K}_Frame.csv"),
            lambda: frame_whiten.Frame(dim=N_RF, frame_type='mercedes', K=MERCEDES_2X_K).W),
        'spectral': ensure_frame_csv(
            os.path.join(FRAME_DIR, f"N{N_RF}_spectral_lowc_Frame.csv"),
            lambda: frame_whiten.Frame(dim=N_RF, frame_type='spectral', target_covariance=C_zz_uniform).W),
    }
    VARIANT_LABEL = {
        'mercedes_baseline': 'Mercedes (K=91, baseline)',
        'mercedes_2x':       f'Mercedes (K={MERCEDES_2X_K})',
        'spectral':          'Spectral (eigenbasis, K=13)',
    }
    VARIANT_COLOR = {
        'mercedes_baseline': 'black',
        'mercedes_2x':       '#1f77b4',
        'spectral':          '#d62728',
    }

    results = {}
    for name, path in frame_paths.items():
        print(f"=== Frame variant: {name} ===")
        frame = Frame(csv_path=path)
        dyn = V1Dynamics_Surround(tunings, frame, N_RF=N_RF, N_SETS=N_SETS,
                                   target_covariance_path=TARGET_COV_PATH, gains_nonneg=True)
        # theta_t from the SAME uniform_target_covariance_low_c for every variant - a pure
        # function of frame.W and C_zz_uniform, so no preliminary "no adaptation" run is needed.
        dyn.calibrate_theta_t(None, None, None, None, C_zz_uniform=C_zz_uniform,
                              uniform_target=True, circular_target=False)

        (y_hist, u_hist, a_hist, g_cRF_hist, g_surround_hist, v_cRF_hist, v_surround_hist,
         mu_cRF_hist, mu_surround_hist) = dyn.run_simulation(shared_stream)

        gain_steps, gain_avg = gain_subset_average(g_cRF_hist)
        error_steps, errors = whitening_error_trace(dyn, g_cRF_hist, Cxx_raw, z0)

        # Final transformed covariance (Figure 3 below): same J @ A operator as the error
        # trace above, evaluated at the LAST gain state and applied directly as a matrix
        # rather than reduced to a scalar via compute_error.
        M_final = analytic_response_operator(dyn, g_cRF_hist[:, -1], z0)
        Cyy_final = M_final @ Cxx_raw @ M_final.T

        results[name] = dict(gain_steps=gain_steps, gain_avg=gain_avg,
                              error_steps=error_steps, errors=errors, Cyy_final=Cyy_final)

    # ==========================================================================
    # Two-panel convergence comparison: one curve per frame variant in each panel.
    # ==========================================================================
    fig, (ax_gain, ax_err) = plt.subplots(1, 2, figsize=(13, 5))

    for name in frame_paths:
        r = results[name]
        ax_gain.plot(r['gain_steps'], r['gain_avg'], color=VARIANT_COLOR[name],
                     linewidth=2.0, label=VARIANT_LABEL[name])
        ax_err.plot(r['error_steps'], r['errors'], color=VARIANT_COLOR[name],
                    linewidth=2.0, label=VARIANT_LABEL[name])

    ax_gain.set_xlabel("Time step", fontsize=13, fontweight='bold')
    ax_gain.set_ylabel(f"Mean of {GAIN_SUBSET_N} gains", fontsize=13, fontweight='bold')
    ax_gain.set_title("Gain convergence", fontsize=14, fontweight='bold')
    ax_gain.spines['top'].set_visible(False)
    ax_gain.spines['right'].set_visible(False)
    ax_gain.legend(fontsize=10, frameon=False)

    ax_err.set_yscale('log')
    ax_err.set_xlabel("Time step", fontsize=13, fontweight='bold')
    ax_err.set_ylabel(f"Whitening error ({ERROR_TYPE})", fontsize=13, fontweight='bold')
    ax_err.set_title("Convergence error", fontsize=14, fontweight='bold')
    ax_err.spines['top'].set_visible(False)
    ax_err.spines['right'].set_visible(False)
    ax_err.legend(fontsize=10, frameon=False)

    plt.tight_layout()

    # ==========================================================================
    # Figure 3: final transformed (predicted) response covariance Cyy = M_final @ Cxx_raw @
    # M_final.T per frame variant, M_final = J @ A at the LAST gain state (same operator as
    # whitening_error_trace's per-step M_total, here kept as a matrix instead of reduced to a
    # scalar). Shared color scale across all 3 panels for a fair visual comparison - they're
    # all the same (N_RF, N_RF) shape regardless of the frame's K, since J and A both live in
    # primary-neuron space.
    # ==========================================================================
    all_vals = np.concatenate([results[name]['Cyy_final'].ravel() for name in frame_paths])
    vmin, vmax = all_vals.min(), all_vals.max()

    fig3, axes3 = plt.subplots(1, 3, figsize=(15, 5))
    im3 = None
    for ax, name in zip(axes3, frame_paths):
        im3 = ax.imshow(results[name]['Cyy_final'], cmap='viridis', vmin=vmin, vmax=vmax, aspect='auto')
        ax.set_title(VARIANT_LABEL[name], fontsize=13, fontweight='bold')
        ax.set_xlabel("Neuron index", fontsize=11, fontweight='bold')
        ax.set_ylabel("Neuron index", fontsize=11, fontweight='bold')
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1.5)
    fig3.colorbar(im3, ax=axes3.ravel().tolist(), fraction=0.03, pad=0.02)
    fig3.suptitle("Final Transformed Response Covariance ($C_{yy}$)", fontsize=15, fontweight='bold')

    plt.show()

