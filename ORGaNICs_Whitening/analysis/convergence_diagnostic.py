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

N_RF       = 13                    # Number of primary neurons per receptive field
N_SETS     = 1                     # 1 classical RF (cRF) + 6 surround sets
CRF_IDX    = 0                     # Index of cRF (arbitrary; sets are symmetric)
FRAME_PATH = os.path.join(REPO_ROOT, "data/frames/N13_mercedes_Frame.csv")
TARGET_COV_PATH = os.path.join(REPO_ROOT, "data/target_covs/uniform_target_covariance.csv")

ENSEMBLE_CONTRAST    = 1.0       # contrast of the adaptation ensembles (baseline & adaptor)
THETA_T_CONTRAST     = 0.25      # contrast used ONLY to calibrate theta_t (see run_adaptation_phase)
TUNING_WIDTH         = 0.75
ADAPT_STREAM_LENGTH  = 1000000  # 101920   # timesteps of adaptation stimulus (dt=0.1 -> 1092s =~ 11x tau_g)
DURATION             = 200     # timesteps each individual adaptation stimulus is held for
N_SETTLE_STEPS       = 1500     # timesteps to settle y/u/a to steady state per probe (dt=0.1 -> 30s)

N_CONTRASTS    = 20
CRF_CONTRASTS  = np.logspace(-2, 0, N_CONTRASTS)
PROBE_CONTRAST = 0.8
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


def probe_input_drive(input_theta, contrast, tuning_width=TUNING_WIDTH):
    '''
    Probe with a stimulus that covers both cRF and surround.
    '''
    theta_grid = np.linspace(0, np.pi, N_RF, endpoint=False)  # Evenly spaced orientation preferences for neurons
    delta = theta_grid - input_theta                          # Distance between neuron pref from stimulus orientation
    delta = (delta + np.pi / 2) % np.pi - np.pi / 2
    profile = np.exp(-delta**2 / (2 * tuning_width**2))

    full_profile = np.concatenate([profile] * N_SETS)
    full_drive = contrast * full_profile / np.linalg.norm(full_profile)  # Normalize and scale by contrast
    return full_drive


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


def run_adaptation_phase(dyn, stim_gen, cond, adapt_location=None, biased=None):
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

#        assert np.all(g_cRF_hist == 0) and np.all(g_surround_hist == 0), (
#            "'no adaptation' run's own gains moved away from zero - theta_t's sentinel "
#            "(see V1Dynamics_Surround.__init__) no longer holds, or calibrate_theta_t was "
#            "already called on this dyn instance. The calibration below would be measuring a "
#            "partially-adapted reference, not a genuinely unbiased one."
#        )
#        dyn.calibrate_theta_t(v_cRF_hist, v_surround_hist, mu_cRF_hist, mu_surround_hist)

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

def frozen_derivatives(state, z_t, dyn, g_cRF):
    '''
    y/u/a/v_cRF/v_surround dynamics, matching V1Dynamics_Surround._derivatives, but with
    g_cRF/g_surround held fixed.

    Synced (per Asit's equations, pasted 2026-08-26) to match two fixes already applied to
    _derivatives -- this function is a hand-maintained mirror and had drifted out of sync,
    silently running the OLD (incorrect) forms for every probe while the adaptation phase
    used the corrected ones:
      1. recurrent_drive uses ONLY sqrt(y+) (Asit: W_yy @ sqrt(y1+), rectified/one-sided).
         The old sqrt_y_plus - sqrt_y_minus reduces to y itself (max(y,0)-max(-y,0) = y,
         identically for every real y) -- silently cancelling the rectification and
         reintroducing full signed-y linear recurrent coupling. Matches V1Dynamics's
         existing (already-correct) recurrent_drive line for corroboration.
      2. da_dt uses raw `a`, not `a_plus`, in the a*u+ term (Asit: a ⊙ u+, using bold/
         unrectified a). Asit's equation also has an additive alpha*du/dt term; alpha=0 in
         Asit's own convention, so it correctly contributes nothing and needs no term here.
    '''
    N_TOT = dyn.N_RF * dyn.N_SETS
    N_RF = dyn.N_RF
    K = dyn.frame.K

    y = state[0:N_RF]
    u = state[N_RF:2*N_RF]
    a = state[2*N_RF:3*N_RF]
    v_cRF = state[3*N_RF:3*N_RF+K]

    u_plus = dyn.half_wave_rectify(u, 0.5)
    y_plus = dyn.half_wave_rectify(y, 2.0)
    a_plus = dyn.half_wave_rectify(a, 1.0)
    sqrt_y_plus = np.sqrt(y_plus)

    dv_cRF_dt = (-v_cRF + dyn.frame.W.T @ y) / dyn.tau_v

    cRF_gain_feedback = (a_plus / (1 + a_plus)) * dyn.frame.W @ (g_cRF * v_cRF)

    recurrent_drive = (1.0 / (1.0 + a_plus)) * (dyn.W_yy @ sqrt_y_plus)
    input_drive = dyn.beta * z_t

    sigma_term = (dyn.sigma / 2) ** 2
    pool_term = dyn.N_matrix @ (y_plus * (u_plus ** 2))

    dy_dt = (-y + input_drive + recurrent_drive - cRF_gain_feedback) / dyn.tau_y
    du_dt = (-u + sigma_term + pool_term) / dyn.tau_u
    da_dt = (-a + (1 + a) * u_plus) / dyn.tau_a

    return np.concatenate([dy_dt, du_dt, da_dt, dv_cRF_dt])

def get_response(dyn, stimulus, g_cRF, mu_cRF, n_steps=N_SETTLE_STEPS):
    '''
    Settles the system (y, u, a) to steady state given a fixed probe stimulus, with g_cRF/g_surround
    frozen. v is initialized to W.T @ mu_{cRF,surround}.

    Starts y/u/a from a zero initial state every call, so probes are independent of sweep
    order/history. Returns (y_final, v_cRF_final, v_surround_final).
    '''
    N_TOT = dyn.N_RF * dyn.N_SETS
    K = dyn.frame.K
    dt = dyn.dt

    v_cRF_init = dyn.frame.W.T @ mu_cRF

    state = np.zeros(3 * N_TOT + 2 * K)
    state[3*N_TOT:3*N_TOT+K] = v_cRF_init

    for _ in range(n_steps):
        k1 = frozen_derivatives(state, stimulus, dyn, g_cRF)
        k2 = frozen_derivatives(state + 0.5 * dt * k1, stimulus, dyn, g_cRF)
        k3 = frozen_derivatives(state + 0.5 * dt * k2, stimulus, dyn, g_cRF)
        k4 = frozen_derivatives(state + dt * k3, stimulus, dyn, g_cRF)
        state += (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    y_final = state[0:N_TOT]
    v_cRF_final = state[3*N_TOT:3*N_TOT+K]

    y_final_rect = dyn.half_wave_rectify(y_final, 2.0)
    return y_final_rect, v_cRF_final


if __name__ == "__main__":

    print("Initializing tunings, frame, and dynamics...")
    tunings = V1Tunings(N=N_RF)
    frame   = Frame(csv_path=FRAME_PATH)
    dyn     = V1Dynamics_Surround(tunings, frame, N_RF=N_RF, N_SETS=N_SETS,
                                   target_covariance_path=TARGET_COV_PATH, gains_nonneg=True)

    stim_gen = StimulusGenerator(N_RF=N_RF, N_SETS=N_SETS, num_angles=N_RF,
                                  stream_length=ADAPT_STREAM_LENGTH,
                                  tuning_width=TUNING_WIDTH, contrast=ENSEMBLE_CONTRAST)

    adaptor_idx = stim_gen.num_angles // 2          # matches generate_surround_ensembles' own adaptor
    adaptor_rad = stim_gen.theta_inputs[adaptor_idx]
    crf_target_idx = CRF_IDX * N_RF + adaptor_idx   # num_angles == N_RF, so this is an exact match

    print("Running adaptation phase for each condition...")
    frozen_gains = {}
    for cond in ACTIVE_CONDITIONS:
        print(f"  Adapting: {CONDITION_LABEL[cond]}")
        frozen_gains[cond] = run_adaptation_phase(dyn, stim_gen, cond)


    # ==========================================================================
    # theoretical optimal g_cRF (Analytic_responses.get_optimal_gains_target),
    # ==========================================================================
    print("Computing theoretical optimal gains for comparison against the simulated network...")
    AR.N_matrix = tunings.N_matrix   # single-RF (N_RF, N_RF) pooling matrix - matches get_optimal_gains_target's expected shape
    AR.sigma = dyn.sigma             # match the live model's sigma, not Analytic_responses.py's own default

    # Reuses the 'adapt CRF only' run already done above - no re-simulation. NOTE: get_optimal_gains_target
    # still compares against dyn.uniform_target_covariance (the offline, feedforward-formula target), NOT
    # dyn.theta_t (now empirically calibrated online, in interneuron- not neuron-space) - these two
    # "theoretical" and "live" targets are no longer the same object, by construction.
    GAIN_CHECK_COND = 'adapt CRF only'
    frozen_g_cRF, _, _, _, _, _, (gain_check_stream, _) = frozen_gains[GAIN_CHECK_COND]
    K = dyn.frame.K

    stimuli_for_theory = gain_check_stream[:N_RF, :].T   # (T, N_RF) - cRF block only, matches frame.W's shape
    g_optimal_cRF = AR.get_optimal_gains_target(
        stimuli_for_theory, dyn.frame.W, target_covariance=dyn.uniform_target_covariance) # CHANGED FROM dyn.uniform_target_covariance)

    # ==========================================================================
    # Figure 1: subset of g_cRF gains vs. time step, for one adaptive simulation - checks
    # that the interneuron gains actually settle to a steady state during the adaptation
    # phase. Reuses the gain history already captured in SIM_HISTORY by run_adaptation_phase
    # (no new simulation).
    # ==========================================================================
    print("Plotting gain-settling time course...")
    GAIN_TIMECOURSE_COND = 'adapt CRF only'
    g_cRF_hist = SIM_HISTORY[GAIN_TIMECOURSE_COND]['g_cRF_hist']   # (K, n_steps)
    n_steps_gain = g_cRF_hist.shape[1]
    gain_subset_idx = np.linspace(0, K - 1, 5).astype(int)
    time_steps = np.arange(n_steps_gain)
    subset_colors = ['#800020', '#002060', '#228B22', '#B35900', '#4B0082']

    fig_gopt, ax_gopt = plt.subplots(figsize=(9, 5))
    for i, gi in enumerate(gain_subset_idx):
        ax_gopt.plot(time_steps, g_cRF_hist[gi, :], color=subset_colors[i], linewidth=3.0)
    ax_gopt.set_ylabel("Gain Subset", fontsize=18, fontweight='bold')
    ax_gopt.set_xlabel("Time Step", fontsize=18, fontweight='bold')
    ax_gopt.tick_params(axis='both', width=2.5, length=6, labelsize=14)
    ax_gopt.grid(False)
    ax_gopt.spines['top'].set_visible(False)
    ax_gopt.spines['right'].set_visible(False)
    ax_gopt.spines['left'].set_linewidth(2.5)
    ax_gopt.spines['bottom'].set_linewidth(2.5)
    plt.tight_layout()

    # ==========================================================================
    # Diagnostic 2: covariance of the actual input stimuli vs. the factorization each
    # gain vector implies. (I + W @ diag(g) @ W.T) is the matrix that maps steady-state
    # input -> the linearized recurrent-plus-gain-feedback response (y* = M^-1 @ z, so
    # M^-1 is what actually gets applied to z) - if g were truly optimal for this
    # stimulus covariance, (I + W @ diag(g) @ W.T)^-1 should resemble it.
    # ==========================================================================
    print("Plotting stimulus covariance vs. gain-implied factorizations...")
    stimulus_covariance = np.cov(stimuli_for_theory, rowvar=False)   # (N_RF, N_RF)

    I_N = np.eye(N_RF)
    M_opt_inv = np.linalg.inv(I_N + dyn.frame.W @ np.diag(g_optimal_cRF) @ dyn.frame.W.T)
    M_frozen_inv = np.linalg.inv(I_N + dyn.frame.W @ np.diag(frozen_g_cRF) @ dyn.frame.W.T)

    vmin, vmax = stimulus_covariance.min(), stimulus_covariance.max()
    fig_fact, axes_fact = plt.subplots(1, 3, figsize=(15, 5))
    for ax, mat, title in zip(axes_fact,
                               [stimulus_covariance, M_opt_inv, M_frozen_inv],
                               ["Cov(input stimuli)",
                                r"$(I + W\,\mathrm{diag}(g_{opt,target})\,W^T)^{-1}$",
                                r"$(I + W\,\mathrm{diag}(g_{frozen})\,W^T)^{-1}$"]):
        im = ax.imshow(mat, cmap='viridis', vmin=vmin, vmax=vmax, aspect='auto')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlabel("cRF neuron index", fontsize=11, fontweight='bold')
        ax.set_ylabel("cRF neuron index", fontsize=11, fontweight='bold')
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(2.0)
    fig_fact.suptitle(f"Stimulus covariance vs. gain-implied factorizations ({CONDITION_LABEL[GAIN_CHECK_COND]})",
                       fontsize=14, fontweight='bold')
    plt.tight_layout()


    # ==========================================================================
    # Diagnostic 3: covariance matrices of the cRF-only stimulus ensemble vs. the network's own
    # adapted steady-state responses to it, over the SAME (N_RF, N_RF) cRF-only block.
    # Panel 1: Cov(stimulus), panel 2: Cov(response) -- both computed over the LAST HALF of
    # the 'adapt CRF only' adaptation stream (tau_g=2500, ADAPT_STREAM_LENGTH=100000 -> ~40
    # tau_g total, so by the halfway point gains have long since converged; Figure 2, same
    # condition, already plots this settling time course). This is the network's actual
    # online-adapted response (live g_cRF/g_surround ODE state), not a frozen-g probe.
    # Scoped to the cRF's own N_RF=13 neurons -- the surround blocks see only the flat
    # baseline in this condition, so they carry no cRF stimulus structure. Reuses
    # SIM_HISTORY captured during the adaptation phase -- no new simulation.
    # ==========================================================================
    print("Computing stimulus vs. adapted-response covariance matrices...")
    COV_COND = 'adapt CRF only'
    cov_stream = SIM_HISTORY[COV_COND]['stream']   # (N_TOT, n_steps)
    cov_y_hist = SIM_HISTORY[COV_COND]['y_hist']   # (N_TOT, n_steps)
    half_cov = cov_stream.shape[1] // 2

    stim_cov = np.cov(cov_stream[:N_RF, half_cov:])   # (N_RF, N_RF)
    resp_cov = np.cov(cov_y_hist[:N_RF, half_cov:])   # (N_RF, N_RF)

    fig_cov, (ax_stim_cov, ax_resp_cov) = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, mat, title in zip([ax_stim_cov, ax_resp_cov], [stim_cov, resp_cov],
                               ["Stimulus Covariance", "Adapted Response Covariance"]):
        im = ax.imshow(mat, cmap='viridis', aspect='auto')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel("cRF neuron index", fontsize=11, fontweight='bold')
        ax.set_ylabel("cRF neuron index", fontsize=11, fontweight='bold')
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(2.0)
    fig_cov.suptitle(f"cRF Covariance: Stimulus vs. Adapted Response ({CONDITION_LABEL[COV_COND]})",
                      fontsize=15, fontweight='bold')
    plt.tight_layout()

    plt.show()
