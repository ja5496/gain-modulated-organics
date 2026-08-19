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
from tqdm import tqdm
from simulation_whiten import Frame, V1Dynamics_Surround
from tunings_whiten import V1Tunings
from stimuli_whiten import StimulusGenerator
from typing import Literal
import Analytic_responses as AR

N_RF       = 13                    # Number of primary neurons per receptive field
N_SETS     = 7                     # 1 classical RF (cRF) + 6 surround sets
N_TOTAL    = N_RF * N_SETS         # Full primary-neuron population
CRF_IDX    = 0                     # Index of cRF (arbitrary; sets are symmetric)
FRAME_PATH = os.path.join(REPO_ROOT, "data/frames/N13_mercedes_Frame.csv")
TARGET_COV_PATH = os.path.join(REPO_ROOT, "data/target_covs/uniform_target_covariance.csv")

ENSEMBLE_CONTRAST = 0.6       # contrast of the adaptation ensembles (baseline & adaptor)
TUNING_WIDTH      = 0.75
ADAPT_STREAM_LENGTH = 100000  # 101920   # timesteps of adaptation stimulus (dt=0.1 -> 1092s =~ 11x tau_g)
DURATION      = 200           # timesteps each individual adaptation stimulus is held for
N_SETTLE_STEPS      = 300     # timesteps to settle y/u/a to steady state per probe (dt=0.1 -> 30s)

N_CONTRASTS   = 20
CRF_CONTRASTS = np.logspace(-2, 0, N_CONTRASTS)
PROBE_CONTRAST = 0.6
N_PROBES       = 720

# Setting colors for plot lines (designated by what section of the visual field is adapted)
COLOR_NONE   = 'black'
COLOR_CRF    = '#FDE68A'     # pastel yellow
COLOR_NONCRF = 'red'
COLOR_BOTH   = 'darkorange'


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

    "no adaptation" runs a real, unbiased ensemble to both regions (needed to calibrate theta_t,
    below) but still forces zero gain feedback - it's the pure-normalization control condition,
    not genuine adaptation.

    theta_t calibration (cond == 'no adaptation' only): dyn.theta_t (loaded from
    uniform_target_covariance.csv, an idealized *feedforward* approximation - see
    frame_whiten.compute_uniform_target_covariance) is overwritten with the EMPIRICAL variance of
    (v - W.T@mu) from this live recurrent run, pooling cRF+surround for more samples. This makes
    theta_t consistent with what this model can actually achieve, rather than a static formula that
    ignores recurrent excitation, the u/a pool, and gain feedback. Runs before the other 3
    conditions (first in CONDITIONS), so they adapt against the calibrated target.

    For the other three conditions, whichever region does NOT get the biased/adaptor ensemble only
    sees the flat, orientation-less baseline, so its gain feedback is forced to zero too.
    '''
    K, N_RF = dyn.frame.K, dyn.N_RF

    stream = stim_gen.generate_surround_ensembles(
        ADAPT_LOCATION_FOR_COND[cond], biased=BIASED_FOR_COND[cond], duration=DURATION, add_poisson_noise=False)

    if cond == 'no adaptation':
        (_, _, _, _, _, v_cRF_hist, v_surround_hist,
         mu_cRF_hist, mu_surround_hist) = dyn.run_simulation(stream)

        half = stream.shape[1] // 2   # skip mu's own warm-up transient
        resid_cRF = v_cRF_hist[:, half:] - dyn.frame.W.T @ mu_cRF_hist[:, half:]
        resid_surround = v_surround_hist[:, half:] - dyn.frame.W.T @ mu_surround_hist[:, half:]
        dyn.theta_t = np.var(np.concatenate([resid_cRF, resid_surround], axis=1), axis=1)

        zeros_K = np.zeros(K)
        return (zeros_K, zeros_K, v_cRF_hist[:, -1], v_surround_hist[:, -1],
                mu_cRF_hist[:, -1], mu_surround_hist[:, -1], stream)

    dyn.run_simulation(stream)

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

    return g_cRF, g_surround, v_cRF, v_surround, mu_cRF, mu_surround, stream

def frozen_derivatives(state, z_t, dyn, g_cRF, g_surround):
    '''
    y/u/a/v_cRF/v_surround dynamics, matching V1Dynamics_Surround._derivatives, but with
    g_cRF/g_surround held fixed.
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
    y_minus = dyn.half_wave_rectify(-y, 2.0)
    a_plus = dyn.half_wave_rectify(a, 1.0)
    sqrt_y_plus = np.sqrt(y_plus)
    sqrt_y_minus = np.sqrt(y_minus)

    dv_cRF_dt = (-v_cRF + dyn.frame.W.T @ y[:N_RF]) / dyn.tau_v
    dv_surround_dt = (-v_surround + dyn.frame.W.T @ y[N_RF:2*N_RF]) / dyn.tau_v

    cRF_gain_feedback = dyn.frame.W @ (g_cRF * v_cRF)
    surround_gain_feedback = dyn.frame.W @ (g_surround * v_surround)
    full_gain_feedback = np.concatenate([cRF_gain_feedback] + [surround_gain_feedback] * (dyn.N_SETS - 1))

    recurrent_drive = (1.0 / (1.0 + a_plus)) * (dyn.W_yy @ (sqrt_y_plus - sqrt_y_minus))  
    input_drive = dyn.beta * z_t

    sigma_term = (dyn.sigma / 2) ** 2
    pool_term = dyn.N_matrix @ (y_plus * (u_plus ** 2))

    dy_dt = (-y + input_drive + recurrent_drive - full_gain_feedback) / dyn.tau_y
    du_dt = (-u + sigma_term + pool_term) / dyn.tau_u
    da_dt = (-a + (1 + a_plus) * u_plus) / dyn.tau_a

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

    y_final = np.maximum(state[0:N_TOT], 0)
    v_cRF_final = state[3*N_TOT:3*N_TOT+K]
    v_surround_final = state[3*N_TOT+K:3*N_TOT+2*K]
    return y_final, v_cRF_final, v_surround_final

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
    # Diagnostic: Gain Feedback matrix (feedback depends on stimuli orientation + neuron preference)
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

    gain_matrices = {cond: gain_feedback_matrix(cond) for cond in ACTIVE_CONDITIONS}

    # Shared, zero-centered color scale across every panel so magnitudes/signs are comparable
    # condition-to-condition and region-to-region.
    all_gain_vals = np.concatenate([m.ravel() for pair in gain_matrices.values() for m in pair])
    gain_vmax = np.max(np.abs(all_gain_vals)) if all_gain_vals.size else 1.0
    gain_vmin = -gain_vmax

    fig_gain, axes_gain = plt.subplots(2, len(ACTIVE_CONDITIONS),
                                        figsize=(3.6 * len(ACTIVE_CONDITIONS), 7.5),
                                        sharex=True, sharey=True)
    im = None
    for col, cond in enumerate(ACTIVE_CONDITIONS):
        cRF_matrix, surround_matrix = gain_matrices[cond]
        for row, (matrix, region_label) in enumerate(zip([cRF_matrix, surround_matrix], ["cRF", "Surround"])):
            ax = axes_gain[row, col]
            im = ax.imshow(matrix, cmap='RdBu_r', vmin=gain_vmin, vmax=gain_vmax,
                            aspect='auto', origin='lower',
                            extent=[0, 180, 0, 180])
            ax.axhline(np.degrees(adaptor_rad), color='gray', linestyle=':', linewidth=1.2)
            ax.axvline(np.degrees(adaptor_rad), color='gray', linestyle=':', linewidth=1.2)
            if row == 0:
                ax.set_title(CONDITION_LABEL[cond], fontsize=11, fontweight='bold')
            if col == 0:
                ax.set_ylabel(f"{region_label}\nProbe orientation (deg)", fontsize=10, fontweight='bold')
            if row == 1:
                ax.set_xlabel("Neuron preference (deg)", fontsize=10, fontweight='bold')
            for spine in ax.spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(1.5)

    fig_gain.colorbar(im, ax=axes_gain.ravel().tolist(), fraction=0.02, pad=0.02,
                       label="Gain feedback (+ = suppressive)")
    fig_gain.suptitle("Gain-feedback matrix: probe orientation x neuron preference", fontsize=15, fontweight='bold')

    '''    # ==========================================================================
    # Diagnostic: average cRF response vector alongside the average cRF stimulus
    # <z_t>, both averaged over the ENTIRE adaptation stream (not just the tail) -
    # checking whether y_avg's jaggedness over a short window is just an artifact
    # of too few stimulus presentations to average over, or something structural.
    # ==========================================================================
    print("Running traced adaptation phases (all conditions) to compare average cRF responses...")
    TRACE_COND = 'adapt CRF only'   # which condition's stimulus stream the right panel shows

    y_avg_by_cond = {}
    trace_cond_stream = None
    for cond in ACTIVE_CONDITIONS:
        trace_stream = stim_gen.generate_surround_ensembles(
            ADAPT_LOCATION_FOR_COND[cond], biased=BIASED_FOR_COND[cond],
            duration=DURATION, add_poisson_noise=True)
        (y_hist, u_hist, a_hist, g_cRF_hist, g_surround_hist, v_cRF_hist, v_surround_hist,
         mu_cRF_hist, mu_surround_hist) = dyn.run_simulation(trace_stream)
        y_avg_by_cond[cond] = y_hist[:N_RF, :].mean(axis=1)   # average cRF response vector, entire simulation
        if cond == TRACE_COND:
            trace_cond_stream = trace_stream

    N_AVG_WINDOW = trace_cond_stream.shape[1]   # entire simulation, not just the last few thousand steps
    z_avg = trace_cond_stream[:N_RF, :].mean(axis=1)   # average cRF stimulus <z_t>, same window, TRACE_COND only

    fig_trace, (ax_y, ax_z) = plt.subplots(1, 2, figsize=(13, 5))

    for cond in ACTIVE_CONDITIONS:
        ax_y.plot(theta_RF_deg, y_avg_by_cond[cond], color=CONDITION_COLOR[cond],
                  linewidth=3, marker='o', markersize=5, label=CONDITION_LABEL[cond])
    ax_y.set_title(f"Average cRF response by condition\n(entire simulation, {N_AVG_WINDOW} steps)",
                    fontsize=13, fontweight='bold')
    ax_y.set_xlabel("Preferred orientation (deg)", fontsize=12, fontweight='bold')
    ax_y.set_ylabel(r"$\bar{y}_{cRF}$", fontsize=13, fontweight='bold')
    ax_y.grid(False)
    ax_y.spines['top'].set_visible(False)
    ax_y.spines['right'].set_visible(False)
    ax_y.legend(fontsize=9, frameon=False)

    ax_z.plot(theta_RF_deg, z_avg, color='black', linewidth=3, marker='o', markersize=5)
    ax_z.set_title(f"Average cRF stimulus\n(entire simulation, {N_AVG_WINDOW} steps, {CONDITION_LABEL[TRACE_COND]})",
                    fontsize=13, fontweight='bold')
    ax_z.set_xlabel("Preferred orientation (deg)", fontsize=12, fontweight='bold')
    ax_z.set_ylabel(r"$\langle z_t \rangle$", fontsize=13, fontweight='bold')
    ax_z.grid(False)
    ax_z.spines['top'].set_visible(False)
    ax_z.spines['right'].set_visible(False)

    plt.tight_layout()'''

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
    frozen_g_cRF, _, _, _, _, _, gain_check_stream = frozen_gains[GAIN_CHECK_COND]
    K = dyn.frame.K

    stimuli_for_theory = gain_check_stream[:N_RF, :].T   # (T, N_RF) - cRF block only, matches frame.W's shape
    g_optimal_cRF = AR.get_optimal_gains_target(
        stimuli_for_theory, dyn.frame.W, target_covariance=dyn.uniform_target_covariance)

    fig_gopt, ax_gopt = plt.subplots(figsize=(9, 5))
    gain_idx = np.arange(K)
    ax_gopt.plot(gain_idx, g_optimal_cRF, color='black', linewidth=2.5, label='theoretical optimal gains')
    ax_gopt.plot(gain_idx, frozen_g_cRF, color=COLOR_CRF, linewidth=2.5, linestyle='--', label='simulated frozen g_cRF')
    corr = np.corrcoef(g_optimal_cRF, frozen_g_cRF)[0, 1]
    rel_rms_err = np.linalg.norm(g_optimal_cRF - frozen_g_cRF) / np.linalg.norm(g_optimal_cRF)
    ax_gopt.set_title(f"Simulated vs. theoretical optimal g_cRF ({CONDITION_LABEL[GAIN_CHECK_COND]})\n"
                       f"corr = {corr:.3f}, rel. RMS err = {rel_rms_err:.3f}",
                       fontsize=13, fontweight='bold')
    ax_gopt.set_xlabel("Gain index", fontsize=12, fontweight='bold')
    ax_gopt.set_ylabel("Gain value", fontsize=12, fontweight='bold')
    ax_gopt.grid(False)
    ax_gopt.spines['top'].set_visible(False)
    ax_gopt.spines['right'].set_visible(False)
    ax_gopt.legend(fontsize=10, frameon=False)
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
    ax_crf.set_title("Contrast Response Functions (Simulated Adaptation)", fontsize=16, fontweight='bold')
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
