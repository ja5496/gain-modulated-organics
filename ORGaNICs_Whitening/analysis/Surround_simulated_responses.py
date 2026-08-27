'''
Surround_simulated_responses.py

Simulates neural firing rates using a model of joint adaptation + normalization.
Interactions between classical receptive field (cRF) neurons and surround (non-cRF) neurons are modeled.
The normalization pool includes all neurons, while adaptation (gain feedback) is local to each RF.

Population structure: 7 sets of 13 primary neurons each (91 total). One set is the
classical RF; the other 6 are surround sets. All 7 sets share the same tuning-curve basis and the
same per-RF frame.

Methodology :
1. Adaptation phase: for each of the 4 conditions, generate_surround_ensembles (stimuli_whiten.py)
   builds a long stimulus stream with the biased/adaptor ensemble routed to whichever region(s)
   the condition adapts. V1Dynamics_Surround.run_simulation integrates the state forward until adaptation
   settles. 
2. To probe, g_cRF, g_surround, mu_cRF, mu_surround are extracted from the final
   adaptation-phase state.
3. get_response settles the fast (y, u, a) dynamics to steady state for a given probe stimulus,
   holding g_cRF/g_surround fixed (no further gain adaptation), but letting v_cRF/v_surround evolve
   dynamically - initialized to W.T @ mu (the frame-projected mean, i.e. "no fluctuation yet") and
   integrated for N_SETTLE_STEPS. So, gain feedback can partially re-equilibrate according to the 
   input drive.
4. The contrast response function is the settled response of the cRF neuron whose tuning
   preference is the adaptor orientation, probed with stimuli centered at that same orientation.
'''

import os
import sys
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from tqdm import tqdm
from simulation_whiten import Frame, V1Dynamics_Surround
from tunings_whiten import V1Tunings
from stimuli_whiten import StimulusGenerator
from typing import Literal
import Analytic_responses as AR

N_RF       = 13                    # Number of primary neurons per receptive field
N_SETS     = 5                     # 1 classical RF (cRF) + 6 surround sets
CRF_IDX    = 0                     # Index of cRF (arbitrary; sets are symmetric)
FRAME_PATH = os.path.join(REPO_ROOT, "data/frames/N13_mercedes_Frame.csv")
TARGET_COV_PATH = os.path.join(REPO_ROOT, "data/target_covs/uniform_target_covariance.csv")

ENSEMBLE_CONTRAST    = 1.0       # contrast of the adaptation ensembles (baseline & adaptor)
TUNING_WIDTH         = 0.75
ADAPT_STREAM_LENGTH  = 100000  # 101920   # timesteps of adaptation stimulus (dt=0.1 -> 1092s =~ 11x tau_g)
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


CONDITIONS = ['no adaptation', 'adapt CRF only', 'adapt surround only', 'adapt CRF and surround']
ACTIVE_CONDITIONS = CONDITIONS
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
# "no adaptation" -> unbiased ensemble to both regions: the zero-gain-feedback control condition,
# and also what run_adaptation_phase uses to calibrate theta_t (see below).
ADAPT_LOCATION_FOR_COND = {
    'no adaptation':          'adapt CRF and surround',
    'adapt CRF only':         'adapt CRF only',
    'adapt surround only':   'adapt surround only',
    'adapt CRF and surround': 'adapt CRF and surround',
}
BIASED_FOR_COND = {
    'no adaptation':          False,
    'adapt CRF only':         True,
    'adapt surround only':    True,
    'adapt CRF and surround': True,
}


def run_adaptation_phase(dyn, stim_gen, cond):
    '''
    Simulates the adaptation state for one condition. Returns (g_cRF, g_surround, v_cRF,
    v_surround, mu_cRF, mu_surround, stream) - stream is cached so later diagnostics can reuse
    this exact run instead of re-simulating.

    "no adaptation" runs a real, unbiased ensemble to both regions - needed to calibrate
    theta_t (see dyn.calibrate_theta_t) - but still forces zero gain feedback in the returned
    values: it's the pure-normalization control condition, not genuine adaptation. Runs before
    the other 3 conditions (first in CONDITIONS), so they adapt against the calibrated target.
    Correctness of the calibration depends on THIS run's own g_cRF/g_surround having stayed at
    exactly zero throughout - guaranteed by theta_t's sentinel value at V1Dynamics_Surround
    construction (see there), not by anything in this function.

    For the other three conditions, whichever region does NOT get the biased/adaptor ensemble only
    sees the flat, orientation-less baseline, so its gain feedback is forced to zero too.
    '''
    K, N_RF = dyn.frame.K, dyn.N_RF

    stream, centers = stim_gen.generate_surround_ensembles(
        ADAPT_LOCATION_FOR_COND[cond], biased=BIASED_FOR_COND[cond], duration=DURATION,
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
        dyn.calibrate_theta_t(v_cRF_hist, v_surround_hist, mu_cRF_hist, mu_surround_hist)

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

    if cond == 'adapt CRF only':
        g_surround = np.zeros(K)
        v_surround = np.zeros(K)
        mu_surround = np.zeros(N_RF)
    elif cond == 'adapt surround only':
        g_cRF = np.zeros(K)
        v_cRF = np.zeros(K)
        mu_cRF = np.zeros(N_RF)

    # Confirm the fix actually holds for this condition: theta_t must sit below at least
    # SOME interneurons' achieved variance, or every gain is clipped to zero (see
    # dyn.calibrate_theta_t's docstring). Checked directly here, every run, rather than
    # trusted - a warning below means this condition's stimulus statistics didn't exceed
    # the calibrated target anywhere, not that the calibration itself is broken.
    g_active = g_cRF if cond != 'adapt surround only' else g_surround
    n_active = int(np.sum(g_active > 1e-3))
    print(f"  [{cond}] gains active (>1e-3): {n_active}/{K} interneurons "
          f"(mean={g_active.mean():.4g}, max={g_active.max():.4g})")
    if n_active == 0:
        print(f"  WARNING: [{cond}] every gain collapsed to zero - this condition's stimulus "
              f"never drove any interneuron's variance above the calibrated theta_t.")

    return g_cRF, g_surround, v_cRF, v_surround, mu_cRF, mu_surround, (stream, centers)

def frozen_derivatives(state, z_t, dyn, g_cRF, g_surround):
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

    y = state[0:N_TOT]
    u = state[N_TOT:2*N_TOT]
    a = state[2*N_TOT:3*N_TOT]
    v_cRF = state[3*N_TOT:3*N_TOT+K]
    v_surround = state[3*N_TOT+K:3*N_TOT+2*K]

    u_plus = dyn.half_wave_rectify(u, 0.5)
    y_plus = dyn.half_wave_rectify(y, 2.0)
    a_plus = dyn.half_wave_rectify(a, 1.0)
    sqrt_y_plus = np.sqrt(y_plus)

    dv_cRF_dt = (-v_cRF + dyn.frame.W.T @ y[:N_RF]) / dyn.tau_v
    dv_surround_dt = (-v_surround + dyn.frame.W.T @ y[N_RF:2*N_RF]) / dyn.tau_v

    cRF_gain_feedback = dyn.frame.W @ (g_cRF * v_cRF)
    surround_gain_feedback = dyn.frame.W @ (g_surround * v_surround)
    full_gain_feedback = (a_plus / (1 + a_plus)) * np.concatenate([cRF_gain_feedback] + [surround_gain_feedback] * (dyn.N_SETS - 1))

    recurrent_drive = (1.0 / (1.0 + a_plus)) * (dyn.W_yy @ sqrt_y_plus)
    input_drive = dyn.beta * z_t

    sigma_term = (dyn.sigma / 2) ** 2
    pool_term = dyn.N_matrix @ (y_plus * (u_plus ** 2))

    dy_dt = (-y + input_drive + recurrent_drive - full_gain_feedback) / dyn.tau_y
    du_dt = (-u + sigma_term + pool_term) / dyn.tau_u
    da_dt = (-a + (1 + a) * u_plus) / dyn.tau_a

    return np.concatenate([dy_dt, du_dt, da_dt, dv_cRF_dt, dv_surround_dt])

def get_response(dyn, stimulus, g_cRF, g_surround, mu_cRF, mu_surround, n_steps=N_SETTLE_STEPS):
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
    v_surround_init = dyn.frame.W.T @ mu_surround

    state = np.zeros(3 * N_TOT + 2 * K)
    state[3*N_TOT:3*N_TOT+K] = v_cRF_init
    state[3*N_TOT+K:3*N_TOT+2*K] = v_surround_init

    for _ in range(n_steps):
        k1 = frozen_derivatives(state, stimulus, dyn, g_cRF, g_surround)
        k2 = frozen_derivatives(state + 0.5 * dt * k1, stimulus, dyn, g_cRF, g_surround)
        k3 = frozen_derivatives(state + 0.5 * dt * k2, stimulus, dyn, g_cRF, g_surround)
        k4 = frozen_derivatives(state + dt * k3, stimulus, dyn, g_cRF, g_surround)
        state += (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    y_final = state[0:N_TOT]
    v_cRF_final = state[3*N_TOT:3*N_TOT+K]
    v_surround_final = state[3*N_TOT+K:3*N_TOT+2*K]

    y_final_rect = dyn.half_wave_rectify(y_final, 2.0)
    return y_final_rect, v_cRF_final, v_surround_final

def get_response_traced(dyn, stimulus, g_cRF, g_surround, mu_cRF, mu_surround, n_steps=N_SETTLE_STEPS):
    '''
    Identical dynamics to get_response (same frozen g_cRF/g_surround, same v initialized to
    W.T @ mu_{cRF,surround}), but returns the FULL time course of y, u, a, v_cRF, v_surround
    over all n_steps instead of only the final settled state.

    Exists to directly check a specific hypothesis (per user request, Problem 2): does
    N_SETTLE_STEPS actually give y/u/a/v time to reach their true fixed point for a NEW
    probe stimulus, or is get_response's reported "steady state" still measuring a
    transient that has not finished relaxing away from v's initial condition? That initial
    condition, W.T @ mu, is exactly the "variance settles over a long time" approximation
    Jake's notes (Sec. 1) identify as inconsistent with Lyndon's framework -- it reflects
    the LONG-RUN adaptation-phase average, not necessarily anything close to what this one
    new probe would settle to.

    Returns (y_hist, u_hist, a_hist, v_cRF_hist, v_surround_hist), each (dim, n_steps).
    '''
    N_TOT = dyn.N_RF * dyn.N_SETS
    K = dyn.frame.K
    dt = dyn.dt

    v_cRF_init = dyn.frame.W.T @ mu_cRF
    v_surround_init = dyn.frame.W.T @ mu_surround

    state = np.zeros(3 * N_TOT + 2 * K)
    state[3*N_TOT:3*N_TOT+K] = v_cRF_init
    state[3*N_TOT+K:3*N_TOT+2*K] = v_surround_init

    y_hist = np.zeros((N_TOT, n_steps))
    u_hist = np.zeros((N_TOT, n_steps))
    a_hist = np.zeros((N_TOT, n_steps))
    v_cRF_hist = np.zeros((K, n_steps))
    v_surround_hist = np.zeros((K, n_steps))

    for t in range(n_steps):
        k1 = frozen_derivatives(state, stimulus, dyn, g_cRF, g_surround)
        k2 = frozen_derivatives(state + 0.5 * dt * k1, stimulus, dyn, g_cRF, g_surround)
        k3 = frozen_derivatives(state + 0.5 * dt * k2, stimulus, dyn, g_cRF, g_surround)
        k4 = frozen_derivatives(state + dt * k3, stimulus, dyn, g_cRF, g_surround)
        state += (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        y_hist[:, t] = state[0:N_TOT]
        u_hist[:, t] = state[N_TOT:2*N_TOT]
        a_hist[:, t] = state[2*N_TOT:3*N_TOT]
        v_cRF_hist[:, t] = state[3*N_TOT:3*N_TOT+K]
        v_surround_hist[:, t] = state[3*N_TOT+K:3*N_TOT+2*K]

    return y_hist, u_hist, a_hist, v_cRF_hist, v_surround_hist

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
    # Figure 1: Gain Feedback matrices (feedback depends on stimuli orientation + neuron
    # preference). One panel per biased condition. In 'adapt CRF only'/'adapt surround only'
    # the non-adapted region has g=0 (its matrix is exactly zero), so the adapted region's
    # matrix is shown. In 'adapt CRF and surround' both regions see the identical biased
    # ensemble through the identical joint normalization pool, so cRF_matrix and
    # surround_matrix come out numerically equal - either one alone is the right thing to
    # plot (summing them would double-count and mismatch the color scale against the other
    # two panels).
    # ==========================================================================
    print("Computing gain-feedback matrices (probe orientation x neuron index)...")
    theta_RF_deg = np.degrees(stim_gen.theta_RF)

    N_GAIN_PROBES = N_RF   # 13 evenly spaced probes
    gain_probe_thetas = np.linspace(0, np.pi, N_GAIN_PROBES, endpoint=False)
    gain_probe_deg = np.degrees(gain_probe_thetas)

    def gain_feedback_matrix(cond):
        '''Row i = settled gain feedback on every neuron (columns) when probed with a stimulus
        centered at gain_probe_thetas[i]. Sign convention matches the old plot: negated, so
        a positive entry means suppressive.'''
        g_cRF, g_surround, _, _, mu_cRF, mu_surround, _ = frozen_gains[cond]
        cRF_matrix = np.zeros((N_GAIN_PROBES, N_RF))
        surround_matrix = np.zeros((N_GAIN_PROBES, N_RF))
        for i, theta in enumerate(tqdm(gain_probe_thetas, desc=f"    {cond}", leave=False)):
            probe = probe_input_drive(theta, PROBE_CONTRAST)
            _, v_cRF_settled, v_surround_settled = get_response(dyn, probe, g_cRF, g_surround, mu_cRF, mu_surround)
            cRF_matrix[i, :] = - dyn.frame.W @ (g_cRF * v_cRF_settled)
            surround_matrix[i, :] = - dyn.frame.W @ (g_surround * v_surround_settled)
        return cRF_matrix, surround_matrix

    FIG1_CONDITIONS = ['adapt CRF only', 'adapt surround only', 'adapt CRF and surround']
    FIG1_TITLE = {
        'adapt CRF only':         'Classical RF Adapted',
        'adapt surround only':    'Surround Adapted',
        'adapt CRF and surround': 'cRF and Surround Adapted',
    }
    gain_matrices = {}
    for cond in FIG1_CONDITIONS:
        cRF_matrix, surround_matrix = gain_feedback_matrix(cond)
        # Pick the region that's actually adapted; for 'adapt CRF and surround' both matrices
        # are equal by symmetry, so cRF_matrix is as good a choice as surround_matrix.
        gain_matrices[cond] = surround_matrix if cond == 'adapt surround only' else cRF_matrix

    # Shared, zero-centered color scale across all 3 panels so magnitudes/signs are comparable.
    all_gain_vals = np.concatenate([m.ravel() for m in gain_matrices.values()])
    gain_vmax = np.max(np.abs(all_gain_vals)) if all_gain_vals.size else 1.0
    gain_vmin = -gain_vmax

    fig1_tick_pos = np.linspace(0, 180, 4)

    fig_gain, axes_gain = plt.subplots(1, len(FIG1_CONDITIONS),
                                        figsize=(5.5 * len(FIG1_CONDITIONS), 6),
                                        sharex=True, sharey=True)
    im = None
    for col, cond in enumerate(FIG1_CONDITIONS):
        ax = axes_gain[col]
        im = ax.imshow(gain_matrices[cond], cmap='RdBu_r', vmin=gain_vmin, vmax=gain_vmax,
                        aspect='auto', origin='lower', extent=[0, 180, 0, 180])
        ax.axhline(np.degrees(adaptor_rad), color='gray', linestyle=':', linewidth=1.2)
        ax.axvline(np.degrees(adaptor_rad), color='gray', linestyle=':', linewidth=1.2)
        ax.set_title(FIG1_TITLE[cond], fontsize=18, fontweight='bold')
        ax.set_xticks(fig1_tick_pos)
        ax.set_yticks(fig1_tick_pos)
        ax.tick_params(labelsize=14, width=2.0, length=6)
        ax.set_xlabel("Neuron Preference", fontsize=16, fontweight='bold')
        if col == 0:
            ax.set_ylabel("Stimulus Orientation", fontsize=16, fontweight='bold')
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1.5)

    cbar = fig_gain.colorbar(im, ax=axes_gain.ravel().tolist(), fraction=0.03, pad=0.02)
    cbar.set_label("Gain Feedback", fontsize=16, fontweight='bold')
    cbar.ax.tick_params(labelsize=13)
    fig_gain.suptitle("Gain Feedback Matrices", fontsize=22, fontweight='bold')


    # ==========================================================================
    # Diagnostic: theoretical optimal g_cRF (Analytic_responses.get_optimal_gains_target),
    # computed from the EXACT SAME stimulus stream the network is about to see and the SAME
    # target covariance the simulation uses for theta_t, vs. the network's actual frozen
    # g_cRF from running that identical stream through the real RK4 dynamics - checking
    # whether the simulated gain feedback matches what theory says it should be.
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
    # Figure 2: subset of g_cRF gains vs. time step, for one adaptive simulation - checks
    # that the interneuron gains actually settle to a steady state during the adaptation
    # phase. Reuses the gain history already captured in SIM_HISTORY by run_adaptation_phase
    # (no new simulation).
    # ==========================================================================
    print("Plotting gain-settling time course...")
    GAIN_TIMECOURSE_COND = 'adapt CRF and surround'
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
    # Diagnostic: covariance of the actual input stimuli vs. the factorization each
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
    # Contrast response functions of the cRF neuron that prefers the adaptor
    # ==========================================================================
    print("Computing contrast response functions...")

    def crf_curve(cond):
        g_cRF, g_surround, _, _, mu_cRF, mu_surround, _ = frozen_gains[cond]
        resp = np.zeros(N_CONTRASTS)
        for i, c in enumerate(tqdm(CRF_CONTRASTS, desc=f"    {cond}", leave=False)):
            probe = probe_input_drive(adaptor_rad, c)
            y, _, _ = get_response(dyn, probe, g_cRF, g_surround, mu_cRF, mu_surround)
            resp[i] = y[crf_target_idx]
        return resp

    curves_by_condition = {cond: crf_curve(cond) for cond in ACTIVE_CONDITIONS}

    fig_crf, ax_crf = plt.subplots(figsize=(7, 5.5))
    for cond in ACTIVE_CONDITIONS:
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
    plt.tight_layout()

    # ==========================================================================
    # Figure 2 (recreated from Surround_Analytic_Responses.py, online adapted state):
    # cRF tuning curves, no-adaptation vs. cRF-ONLY-adapted (surround left at baseline, so
    # any tuning-curve change is due entirely to the cRF's own local gain feedback, not
    # surround-driven suppression). Unlike that script's get_response_moments (assumes v
    # instantly factorizes the covariance transform) or get_response (assumes v is frozen at
    # W.T@mu), this uses the SAME get_response as every other figure above: g frozen, v
    # dynamically settling from W.T@mu over N_SETTLE_STEPS - the actual online model, not
    # either closed-form extreme.
    # ==========================================================================
    print("Recreating Figure 2 (cRF tuning curves, online adapted state)...")
    N_BINS = N_RF
    crf_slice = slice(CRF_IDX * N_RF, (CRF_IDX + 1) * N_RF)
    probe_angles = np.linspace(0, np.pi, N_PROBES, endpoint=False)
    probe_angles_deg = np.degrees(probe_angles)
    adaptor_deg = np.degrees(adaptor_rad)

    _, _, _, _, _, _, (_, centers_none) = frozen_gains['no adaptation']
    _, _, _, _, _, _, (_, centers_crf)  = frozen_gains['adapt CRF only']
    uni_angles_deg  = np.degrees(centers_none)   # stimulus centers actually shown during that run
    bias_angles_deg = np.degrees(centers_crf)

    def crf_tuning_curves(cond):
        g_cRF, g_surround, _, _, mu_cRF, mu_surround, _ = frozen_gains[cond]
        resp = np.zeros((N_RF, N_PROBES))
        for i, ang in enumerate(tqdm(probe_angles, desc=f"    {cond}", leave=False)):
            probe = probe_input_drive(ang, PROBE_CONTRAST)
            y, _, _ = get_response(dyn, probe, g_cRF, g_surround, mu_cRF, mu_surround)
            # get_response only half-wave-rectifies (max(y,0)); square to get the firing-rate
            # estimate, matching half_wave_rectify(y, alpha=2.0) used everywhere else (y is
            # already >=0 here, so squaring is equivalent and needs no re-clipping).
            resp[:, i] = y[crf_slice]
        return resp

    tc_none = crf_tuning_curves('no adaptation')
    tc_crf  = crf_tuning_curves('adapt CRF only')

    def bin_by_preference(response, neuron_preferences, n_bins=N_BINS):
        '''Matches Surround_Analytic_Responses.py's bin_by_preference.'''
        discrete_step = np.pi / len(neuron_preferences)
        bin_edges = np.linspace(0, np.pi, n_bins + 1) - (discrete_step / 2)
        binned = np.zeros((n_bins, response.shape[1]))
        bin_idx = np.clip(np.digitize(neuron_preferences, bin_edges) - 1, 0, n_bins - 1)
        for b in range(n_bins):
            mask = bin_idx == b
            if np.any(mask):
                binned[b, :] = np.mean(response[mask, :], axis=0)
        return binned

    binned_none = bin_by_preference(tc_none, tunings.theta)
    binned_crf  = bin_by_preference(tc_crf,  tunings.theta)

    # Normalize each neuron's curve (both panels) to ITS OWN non-adapted peak response, not a
    # min/max rescale - preserves the true (non-forced-to-0) floor and makes both panels directly
    # comparable as "fraction of that neuron's unadapted peak firing rate."
    peak_none = np.max(binned_none, axis=1, keepdims=True)
    norm_none = binned_none / (peak_none + 1e-12)
    norm_crf  = binned_crf  / (peak_none + 1e-12)

    discrete_step_hist = 180 / N_RF
    bins_hist = np.linspace(0, 180, N_BINS + 1) - (discrete_step_hist / 2)
    weights_uni  = np.ones_like(uni_angles_deg)  / len(uni_angles_deg)
    weights_bias = np.ones_like(bias_angles_deg) / len(bias_angles_deg)

    x_axis = (probe_angles_deg - adaptor_deg + 90) % 180 - 90
    sort_idx = np.argsort(x_axis)
    x_axis_sorted = x_axis[sort_idx]

    blue_colors = plt.cm.Blues(np.linspace(0.2, 1.0, N_BINS))

    fig_tc, axes_tc = plt.subplots(2, 2, figsize=(10, 6), sharey='row',
                                    gridspec_kw={'height_ratios': [0.8, 1.0]})

    axes_tc[0, 0].hist(uni_angles_deg, bins=bins_hist, weights=weights_uni, color='black', rwidth=0.9)
    axes_tc[0, 0].set_title("Uniform Ensemble", fontweight='bold', fontsize=18)
    axes_tc[0, 0].set_ylabel("Probability", fontsize=18)

    axes_tc[0, 1].hist(bias_angles_deg, bins=bins_hist, weights=weights_bias, color='black', rwidth=0.9)
    axes_tc[0, 1].set_title("Biased Ensemble", fontweight='bold', fontsize=18)

    for ax in axes_tc[0]:
        ax.set_xlim(bins_hist[0], bins_hist[-1])
        ax.tick_params(labelbottom=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    for i in range(N_BINS):
        axes_tc[1, 0].plot(x_axis_sorted, norm_none[i][sort_idx], color=blue_colors[i], linewidth=2.0)
        axes_tc[1, 1].plot(x_axis_sorted, norm_crf[i][sort_idx],  color=blue_colors[i], linewidth=2.0)

    axes_tc[1, 0].set_ylabel("Response", fontsize=18)
    for c in [0, 1]:
        ax = axes_tc[1, c]
        ax.set_xlim(-90, 90)
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin - 0.05 * (ymax - ymin), ymax)
        ax.grid(False)
        ax.set_xlabel("Stimulus Orientation (°)", fontsize=18)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()

    # ==========================================================================
    # Figure 6: Tuning curve of the flank neuron (adjacent to the adaptor-preferring
    # neuron), matching the style of Surround_Analytic_Responses.py's "Tuning Curve (Flank
    # Neuron)" plot exactly - only the response computation differs (here: settled RK4
    # dynamics via get_response/crf_tuning_curves; there: closed-form get_response). Reuses
    # tc_none and tc_crf ('adapt CRF only', now identical to Figure 5's own condition) from
    # Figure 5; only the surround-only curve is newly computed here, from an adaptation state
    # already produced in the adaptation-phase loop above (no new simulation).
    # ==========================================================================
    print("Computing flank-neuron tuning curves (simulated)...")
    flank_idx = (adaptor_idx - 1) % N_RF
    FLANK_CONDITIONS = ['no adaptation', 'adapt surround only', 'adapt CRF only']

    tc_surround_only = crf_tuning_curves('adapt surround only')
    tc_by_flank_cond = {
        'no adaptation':       tc_none,
        'adapt surround only': tc_surround_only,
        'adapt CRF only':      tc_crf,
    }
    flank_curves = {cond: tc_by_flank_cond[cond][flank_idx, :] for cond in FLANK_CONDITIONS}

    # Peak location per curve (parabolic interpolation around the argmax sample for
    # sub-resolution precision - matches Surround_Analytic_Responses.py's curve_peak_deg),
    # then the shift of each adapted condition's peak relative to the no-adaptation control,
    # wrapped to the nearest equivalent orientation (+-90 deg).
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

    peak_deg = {cond: curve_peak_deg(flank_curves[cond]) for cond in FLANK_CONDITIONS}
    control_peak = peak_deg['no adaptation']
    peak_shift_deg = {cond: ((peak_deg[cond] - control_peak + 90) % 180) - 90 for cond in FLANK_CONDITIONS}

    FLANK_LEGEND_LABEL = {
        'no adaptation':       'No adaptation',
        'adapt surround only': f"Surround: {peak_shift_deg['adapt surround only']:+.2f}°",
        'adapt CRF only':      f"cRF: {peak_shift_deg['adapt CRF only']:+.2f}°",
    }

    fig_flank, ax_flank = plt.subplots(figsize=(7, 5.5))
    for cond in FLANK_CONDITIONS:
        ax_flank.plot(probe_angles_deg, flank_curves[cond], color=CONDITION_COLOR[cond],
                      linewidth=3.5, label=FLANK_LEGEND_LABEL[cond])

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
    plt.tight_layout()

    # ==========================================================================
    # Figure 7: PCA scatter of stimuli vs. steady-state responses (biased ensemble = 'adapt
    # CRF and surround'), over the last quarter of the adaptation stream. Scoped to the cRF's
    # OWN N_RF neurons only, not the full N_RF*N_SETS population (per user request) - the 6
    # replica surround blocks see a heavily-correlated copy of the same stimulus, so a joint
    # PCA over all of them was dominated by that between-block redundancy rather than
    # within-RF whitening quality. One point per distinct stimulus presentation, not per
    # timestep: both the stimulus and the response are sampled at the LAST timestep of each
    # DURATION-length hold, so the response is the settled, steady-state reaction to that
    # exposure. PCA is fit jointly on stimuli + responses so both point clouds share one 2D
    # coordinate frame, making their covariance ellipses directly comparable. Reuses the
    # stream/y_hist already captured in SIM_HISTORY during the adaptation phase - no new
    # simulation.
    # ==========================================================================
    print("Building PCA scatter of stimuli vs. steady-state responses...")
    PCA_COND = 'adapt CRF and surround'
    pca_stream = SIM_HISTORY[PCA_COND]['stream']   # (N_TOT, n_steps)
    pca_y_hist = SIM_HISTORY[PCA_COND]['y_hist']    # (N_TOT, n_steps)
    n_steps_pca = pca_stream.shape[1]

    quarter_start = n_steps_pca - n_steps_pca // 4
    quarter_start = int(np.ceil(quarter_start / DURATION)) * DURATION  # align to a block boundary
    last_exposure_idx = np.arange(quarter_start + DURATION - 1, n_steps_pca, DURATION)

    # ---- Step 1 (per user request): confirm these responses are gain-ADAPTED, not
    # frozen/early-transient. pca_y_hist/pca_stream come straight from run_simulation
    # (called inside run_adaptation_phase), which integrates g_cRF/g_surround as live ODE
    # state (tau_g=500) jointly with y/u/a/v -- this is NOT the frozen-g get_response path
    # used for every probe/tuning-curve figure above. ADAPT_STREAM_LENGTH=100000 steps at
    # dt=0.1 -> 10000 time units ~ 20*tau_g, so gains should be fully converged well
    # before the last-quarter window sampled below (Figure 2, same condition, already
    # plots this settling time course). Confirmed numerically here from the actual g_cRF
    # history over exactly the sampled window, rather than just the theoretical estimate.
    g_cRF_hist_pca = SIM_HISTORY[PCA_COND]['g_cRF_hist']
    g_window = g_cRF_hist_pca[:, quarter_start:]
    g_drift = np.linalg.norm(g_window[:, -1] - g_window[:, 0]) / (np.linalg.norm(g_window[:, 0]) + 1e-12)
    print(f"  g_cRF relative drift over the sampled (last-quarter) window: {g_drift:.2%} "
          f"({'converged -- gains are adapted' if g_drift < 0.01 else 'STILL DRIFTING -- window may be too early'})")

    # cRF block ONLY (first N_RF rows), not the full N_TOT-dim population: the 6 surround
    # blocks see a heavily-correlated, near-redundant copy of the same stimulus (confirmed
    # separately: cross-block response correlation ~0.55 even with independent per-neuron
    # noise), so a joint PCA over all 91 dims was picking up that between-block redundancy
    # structure as much as any within-RF whitening quality - not the intended test.
    stim_points = pca_stream[:N_RF, last_exposure_idx].T   # (n_blocks, N_RF) - one row per presentation, cRF only
    resp_points = pca_y_hist[:N_RF, last_exposure_idx].T   # (n_blocks, N_RF) - steady-state response, cRF only
    n_blocks = stim_points.shape[0]
    print(f"  {n_blocks} distinct stimulus presentations in the last quarter of the stream.")

    combined = np.concatenate([stim_points, resp_points], axis=0)
    combined_centered = combined - combined.mean(axis=0, keepdims=True)
    joint_cov = np.cov(combined_centered, rowvar=False)
    eigvals_j, eigvecs_j = np.linalg.eigh(joint_cov)
    top2 = eigvecs_j[:, np.argsort(eigvals_j)[::-1][:2]]

    proj = combined_centered @ top2
    stim_proj = proj[:n_blocks]
    resp_proj = proj[n_blocks:]

    # ---- Step 2 (per user request): does the pure LINEAR-ALGEBRA gain-feedback
    # factorization (I + W diag(g_opt) W^T)^-1, applied directly to the SAME (now cRF-only)
    # stim_points used above, circularize their covariance the way the full dynamical
    # model's responses (panel 1) apparently do NOT? g_opt is the theoretically optimal
    # gain vector (Analytic_responses.get_optimal_gains_target), computed fresh from THIS
    # condition's own cRF-block stream (the network's full adaptation history, not just
    # the last-quarter window) -- NOT reused from the earlier 'adapt CRF only' diagnostic
    # above, which is a different run with different stream statistics. Scoped to the
    # cRF's own N_RF dims to match stim_points above -- no N_SETS block-replication needed
    # now. This isolates the gain-feedback linear algebra from every other piece of the
    # full model (rectification, the u/a divisive-normalization pool, v's dynamic
    # settling, recurrent W_yy drive) -- if THIS circularizes but panel 1 doesn't, the
    # discrepancy lives in one of those other mechanisms, not in the gains being wrong.
    print("Computing optimal-gain linear factorization for the same stimulus points...")
    stimuli_pca_cRF_block = pca_stream[:N_RF, :].T   # (n_steps_pca, N_RF) - full stream, matches the earlier g_optimal_cRF diagnostic's convention
    g_optimal_pca = AR.get_optimal_gains_target(
        stimuli_pca_cRF_block, dyn.frame.W, target_covariance=dyn.uniform_target_covariance)

    M_opt_inv = np.linalg.inv(np.eye(N_RF) + dyn.frame.W @ np.diag(g_optimal_pca) @ dyn.frame.W.T)

    linfact_points = (M_opt_inv @ stim_points.T).T            # (n_blocks, N_RF)
    linfact_proj = (linfact_points - combined.mean(axis=0, keepdims=True)) @ top2

    def cov_ellipse(ax, points, color, n_std=2.0):
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

    fig_pca, (ax_pca, ax_linfact) = plt.subplots(1, 2, figsize=(14, 7))
    # alpha < 1 so exactly-overlapping points compound into a visibly darker/denser patch
    # instead of one opaque marker silently hiding how many points actually land there.
    ax_pca.scatter(stim_proj[:, 0], stim_proj[:, 1], color='red', alpha=0.1, s=45, label='Stimuli')
    ax_pca.scatter(resp_proj[:, 0], resp_proj[:, 1], color='blue', alpha=0.1, s=45, label='Responses')
    eig_stim = cov_ellipse(ax_pca, stim_proj, 'red')
    eig_resp = cov_ellipse(ax_pca, resp_proj, 'blue')

    ax_pca.set_title("Adaptation PCA", fontsize=22, fontweight='bold')
    ax_pca.set_xticks([])
    ax_pca.set_yticks([])
    ax_pca.legend(fontsize=16, loc='upper right', frameon=False)
    ax_pca.text(0.02, 0.98, rf"Stimuli $\lambda$: {eig_stim[0]:.3g}, {eig_stim[1]:.3g}",
                transform=ax_pca.transAxes, color='red', fontsize=12, fontweight='bold', va='top', ha='left')
    ax_pca.text(0.02, 0.92, rf"Responses $\lambda$: {eig_resp[0]:.3g}, {eig_resp[1]:.3g}",
                transform=ax_pca.transAxes, color='blue', fontsize=12, fontweight='bold', va='top', ha='left')

    # Same stim_proj (same points, same top2 projection) as panel 1, so panel 1's red
    # ellipse and this panel's red ellipse are identical by construction -- only the
    # green (linear-factorization) cloud is new.
    ax_linfact.scatter(stim_proj[:, 0], stim_proj[:, 1], color='red', alpha=0.1, s=45, label='Stimuli')
    ax_linfact.scatter(linfact_proj[:, 0], linfact_proj[:, 1], color='green', alpha=0.1, s=45,
                        label=r'$(I+W\,\mathrm{diag}(g_{opt})\,W^T)^{-1}$ Stimuli')
    eig_stim2 = cov_ellipse(ax_linfact, stim_proj, 'red')
    eig_linfact = cov_ellipse(ax_linfact, linfact_proj, 'green')

    ax_linfact.set_title("Optimal-Gain Linear Factorization", fontsize=20, fontweight='bold')
    ax_linfact.set_xticks([])
    ax_linfact.set_yticks([])
    ax_linfact.legend(fontsize=13, loc='upper right', frameon=False)
    ax_linfact.text(0.02, 0.98, rf"Stimuli $\lambda$: {eig_stim2[0]:.3g}, {eig_stim2[1]:.3g}",
                     transform=ax_linfact.transAxes, color='red', fontsize=12, fontweight='bold', va='top', ha='left')
    ax_linfact.text(0.02, 0.92, rf"Lin. fact. $\lambda$: {eig_linfact[0]:.3g}, {eig_linfact[1]:.3g}",
                     transform=ax_linfact.transAxes, color='green', fontsize=12, fontweight='bold', va='top', ha='left')
    plt.tight_layout()

    # ==========================================================================
    # Figure 8 (per user request, Problem 2): single-probe dynamics diagnostic. Traces
    # y (cRF + surround adaptor-preferring neurons), u, a+, and ||v|| over the FULL
    # N_SETTLE_STEPS settling window, for 'adapt CRF only' (Problem 2's own condition) at
    # contrasts 0.6 and 1.0 -- directly matching "recurrent drive should change a lot from
    # contrast 0.6 to 1.0" and testing whether get_response's reported "steady state" has
    # actually converged within N_SETTLE_STEPS, or is still a transient relaxing away from
    # v's initial condition (W.T@mu, the long-run adaptation-phase average -- see
    # get_response_traced's docstring and Jake's notes Sec. 1 on why that initial condition
    # is NOT the same thing as Lyndon's fast-v factorization).
    # ==========================================================================
    print("Tracing single-probe dynamics (y, u, a, v) for Problem 2 diagnostic...")
    DIAG_COND = 'adapt CRF only'
    g_cRF_diag, g_surround_diag, _, _, mu_cRF_diag, mu_surround_diag, _ = frozen_gains[DIAG_COND]
    surround_target_idx = N_RF * 1 + adaptor_idx   # adaptor-preferring neuron, first surround block

    DIAG_CONTRASTS = [0.6, 1.0]
    DIAG_COLORS = {0.6: '#4C72B0', 1.0: '#C44E52'}
    traces = {}
    for c in DIAG_CONTRASTS:
        probe = probe_input_drive(adaptor_rad, c)
        traces[c] = get_response_traced(dyn, probe, g_cRF_diag, g_surround_diag, mu_cRF_diag, mu_surround_diag)

    time_axis = np.arange(N_SETTLE_STEPS) * dyn.dt

    def rel_change_last_10pct(trace_1d):
        '''Relative change over the last 10% of the settle window -- near 0 means
        converged; still-substantial means get_response's "final state" was a transient.'''
        i90 = int(0.9 * len(trace_1d))
        return abs(trace_1d[-1] - trace_1d[i90]) / (abs(trace_1d[-1]) + 1e-9)

    fig_diag, axes_diag = plt.subplots(2, 2, figsize=(12, 8))
    ax_y, ax_u, ax_a, ax_v = axes_diag[0, 0], axes_diag[0, 1], axes_diag[1, 0], axes_diag[1, 1]

    for c in DIAG_CONTRASTS:
        y_hist, u_hist, a_hist, v_cRF_hist, v_surround_hist = traces[c]
        color = DIAG_COLORS[c]
        u_mean = u_hist.mean(axis=0)
        a_plus_mean = dyn.half_wave_rectify(a_hist, 1.0).mean(axis=0)
        v_cRF_norm = np.linalg.norm(v_cRF_hist, axis=0)
        v_surround_norm = np.linalg.norm(v_surround_hist, axis=0)

        ax_y.plot(time_axis, y_hist[crf_target_idx], color=color, linewidth=2.5, label=f"cRF, c={c}")
        ax_y.plot(time_axis, y_hist[surround_target_idx], color=color, linewidth=2.0, linestyle='--', label=f"surround, c={c}")
        ax_u.plot(time_axis, u_mean, color=color, linewidth=2.5, label=f"c={c}")
        ax_a.plot(time_axis, a_plus_mean, color=color, linewidth=2.5, label=f"c={c}")
        ax_v.plot(time_axis, v_cRF_norm, color=color, linewidth=2.5, label=f"||v_cRF||, c={c}")
        ax_v.plot(time_axis, v_surround_norm, color=color, linewidth=2.0, linestyle='--', label=f"||v_surround||, c={c}")

        print(f"  c={c}: relative change over the LAST 10% of the settle window -- "
              f"y[cRF]={rel_change_last_10pct(y_hist[crf_target_idx]):.2%}, "
              f"y[surround]={rel_change_last_10pct(y_hist[surround_target_idx]):.2%}, "
              f"mean(u)={rel_change_last_10pct(u_mean):.2%}, "
              f"mean(a+)={rel_change_last_10pct(a_plus_mean):.2%}, "
              f"||v_cRF||={rel_change_last_10pct(v_cRF_norm):.2%}, "
              f"||v_surround||={rel_change_last_10pct(v_surround_norm):.2%}")

    ax_y.set_title("y (membrane potential)", fontweight='bold')
    ax_u.set_title("u (mean over population)", fontweight='bold')
    ax_a.set_title(r"$a_+$ (mean over population)", fontweight='bold')
    ax_v.set_title("||v|| (cRF vs. surround)", fontweight='bold')
    for ax in (ax_y, ax_u, ax_a, ax_v):
        ax.set_xlabel("Time within probe (settle window)", fontsize=11)
        ax.legend(fontsize=9, frameon=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    fig_diag.suptitle(f"Single-Probe Dynamics ({CONDITION_LABEL[DIAG_COND]}, adaptor orientation)",
                       fontsize=15, fontweight='bold')
    plt.tight_layout()

    plt.show()
