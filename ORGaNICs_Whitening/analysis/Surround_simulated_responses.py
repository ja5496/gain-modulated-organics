'''
Surround_simulated_responses.py

Simulates neural firing rates in response to different input distributions after adaptation.
Interactions between classical receptive field (cRF) neurons and surround (non-cRF) neurons are modeled.
The normalization pool includes all neurons, while adaptation (gain feedback) is local to each RF.

Population structure: 7 sets of 13 primary neurons each (91 total). One set is the
classical RF; the other 6 are surround sets. All 7 sets share the same tuning-curve basis and the
same per-RF frame, so the 6 surround sets are treated as fully interchangeable (no distance-dependent
weighting) -- only cRF vs. non-cRF membership matters.

Methodology (mirrors Surround_analytic_responses.py's structure, but the adaptation state is
simulated via the full ORGaNICs RK4 dynamics in V1Dynamics_Surround rather than computed via a
self-consistency loop):
1. Adaptation phase: for each of the 4 conditions, generate_surround_ensembles (stimuli_whiten.py)
   builds a long stimulus stream with the biased/adaptor ensemble routed to whichever region(s)
   the condition adapts, and V1Dynamics_Surround.run_simulation integrates the full state
   (y, u, a, g_cRF, g_surround, v_cRF, v_surround) forward until adaptation settles. "no adaptation"
   is just adaptation to the uniform (unbiased) ensemble shown to both cRF and surround.
2. The adapted g_cRF, g_surround, v_cRF, v_surround are frozen (read off the final state) - these
   fully determine a fixed gain-feedback vector, matching the role of M @ mu in the analytic script.
3. get_response settles the fast (y, u, a) dynamics to steady state for a given probe stimulus,
   holding that gain feedback fixed (i.e. no further adaptation during the probe).
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

N_RF       = 13                    # primary neurons per receptive field
N_SETS     = 7                     # 1 classical RF (cRF) + 6 surround sets
N_TOTAL    = N_RF * N_SETS         # full primary-neuron population
CRF_IDX    = 0                     # which of the 7 sets is the cRF (arbitrary; sets are symmetric)
FRAME_PATH = os.path.join(REPO_ROOT, "data/frames/N13_mercedes_Frame.csv")
TARGET_COV_PATH = os.path.join(REPO_ROOT, "data/target_covs/uniform_target_covariance.csv")

ENSEMBLE_CONTRAST = 0.8      # contrast of the adaptation ensembles (baseline & adaptor)
TUNING_WIDTH      = 0.75
ADAPT_STREAM_LENGTH = 10920   # timesteps of adaptation stimulus (dt=0.1 -> 1092s =~ 11x tau_g)
ADAPT_DURATION      = 20      # timesteps each individual adaptation stimulus is held for
N_SETTLE_STEPS      = 300     # timesteps to settle y/u/a to steady state per probe (dt=0.1 -> 30s)

N_CONTRASTS   = 20
CRF_CONTRASTS = np.logspace(-2, 0, N_CONTRASTS)
PROBE_CONTRAST = 0.15
N_PROBES       = 720

# Setting colors for plot lines (designated by what section of the visual field is adapted)
COLOR_NONE   = 'black'
COLOR_CRF    = '#FDE68A'     # pastel yellow
COLOR_NONCRF = 'red'
COLOR_BOTH   = 'darkorange'


def probe_input_drive(input_theta, contrast, tuning_width=TUNING_WIDTH):
    '''Always probing with a stimulus that covers both cRF and surround, no matter the adaptation state.'''
    theta_grid = np.linspace(0, np.pi, N_RF, endpoint=False)  # Evenly spaced orientation preferences for neurons
    delta = theta_grid - input_theta  # Distance between neuron preference from stimulus orientation
    delta = (delta + np.pi / 2) % np.pi - np.pi / 2
    profile = np.exp(-delta**2 / (2 * tuning_width**2))
    RF_gaussian_drive = contrast * 1 * profile / np.linalg.norm(profile)  # Gaussian input response profile scaled by contrast

    full_drive = np.concatenate([RF_gaussian_drive] * N_SETS)
    return full_drive


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
# "no adaptation" is adaptation to the uniform ensemble, shown to both cRF and surround (routed via
# 'adapt CRF and surround' with biased=False) - every other condition shows the biased/adaptor
# ensemble only to the region(s) it names, with the rest of the population held at baseline
# (see generate_surround_ensembles in stimuli_whiten.py).
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
    Simulates the adaptation state for one condition: builds the appropriate stimulus stream
    via generate_surround_ensembles and integrates V1Dynamics_Surround's full dynamics forward
    until the gains settle. Returns the frozen (g_cRF, g_surround, v_cRF, v_surround).
    '''
    stream = stim_gen.generate_surround_ensembles(
        ADAPT_LOCATION_FOR_COND[cond], biased=BIASED_FOR_COND[cond], duration=ADAPT_DURATION, add_poisson_noise=False)
    dyn.run_simulation(stream)

    K = dyn.frame.K
    N_TOT = dyn.N_RF * dyn.N_SETS
    state = dyn.last_state
    g_cRF = state[3*N_TOT:3*N_TOT+K]
    g_surround = state[3*N_TOT+K:3*N_TOT+2*K]
    v_cRF = state[3*N_TOT+2*K:3*N_TOT+3*K]
    v_surround = state[3*N_TOT+3*K:3*N_TOT+4*K]
    return g_cRF, g_surround, v_cRF, v_surround


def frozen_derivatives(state, z_t, dyn, full_gain_feedback):
    '''y/u/a dynamics only, matching V1Dynamics_Surround._derivatives, but with the gain-feedback
    term fixed rather than derived from evolving g/v state - used to settle the fast dynamics to
    steady state under frozen (post-adaptation) gains.'''
    N_TOT = dyn.N_RF * dyn.N_SETS
    y = state[0:N_TOT]
    u = state[N_TOT:2*N_TOT]
    a = state[2*N_TOT:3*N_TOT]

    u_plus = dyn.half_wave_rectify(u, 0.5)
    y_plus = dyn.half_wave_rectify(y, 2.0)
    y_minus = dyn.half_wave_rectify(-y, 2.0)
    a_plus = dyn.half_wave_rectify(a, 1.0)
    sqrt_y_plus = np.sqrt(y_plus)
    sqrt_y_minus = np.sqrt(y_minus)

    recurrent_drive = (1.0 / (1.0 + a_plus)) * (dyn.W_yy @ (sqrt_y_plus - sqrt_y_minus))
    input_drive = dyn.beta * z_t

    sigma_term = (dyn.sigma / 2) ** 2
    pool_term = dyn.N_matrix @ (y_plus * (u_plus ** 2))

    dy_dt = (-y + input_drive + recurrent_drive - full_gain_feedback) / dyn.tau_y
    du_dt = (-u + sigma_term + pool_term) / dyn.tau_u
    da_dt = (-a + (1+ a_plus) * u_plus) / dyn.tau_a

    return np.concatenate([dy_dt, du_dt, da_dt])


def get_response(dyn, stimulus, g_cRF, g_surround, v_cRF, v_surround, n_steps=N_SETTLE_STEPS):
    '''
    Settles the system (y, u, a) to steady state given a fixed probe stimulus and frozen adapted
    gains (g_cRF, g_surround, v_cRF, v_surround) - no further gain adaptation happens here. Starts
    from a zero initial state every call, so probes are independent of sweep order/history.
    '''
    N_TOT = dyn.N_RF * dyn.N_SETS
    dt = dyn.dt

    cRF_gain_feedback = dyn.frame.W @ (g_cRF * v_cRF)
    surround_gain_feedback = dyn.frame.W @ (g_surround * v_surround)
    full_gain_feedback = np.concatenate([cRF_gain_feedback] + [surround_gain_feedback] * (dyn.N_SETS - 1))

    state = np.zeros(3 * N_TOT)
    for _ in range(n_steps):
        k1 = frozen_derivatives(state, stimulus, dyn, full_gain_feedback)
        k2 = frozen_derivatives(state + 0.5 * dt * k1, stimulus, dyn, full_gain_feedback)
        k3 = frozen_derivatives(state + 0.5 * dt * k2, stimulus, dyn, full_gain_feedback)
        k4 = frozen_derivatives(state + dt * k3, stimulus, dyn, full_gain_feedback)
        state += (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    return np.maximum(state[0:N_TOT], 0)


if __name__ == "__main__":

    print("Initializing tunings, frame, and dynamics...")
    tunings = V1Tunings(N=N_RF)
    frame   = Frame(csv_path=FRAME_PATH)
    dyn     = V1Dynamics_Surround(tunings, frame, N_RF=N_RF, N_SETS=N_SETS,
                                   target_covariance_path=TARGET_COV_PATH)

    stim_gen = StimulusGenerator(N_RF=N_RF, N_SETS=N_SETS, num_angles=N_RF,
                                  stream_length=ADAPT_STREAM_LENGTH,
                                  tuning_width=TUNING_WIDTH, contrast=ENSEMBLE_CONTRAST)

    adaptor_idx = stim_gen.num_angles // 2          # matches generate_surround_ensembles' own adaptor
    adaptor_rad = stim_gen.theta_inputs[adaptor_idx]
    crf_target_idx = CRF_IDX * N_RF + adaptor_idx   # num_angles == N_RF, so this is an exact match

    print("Running adaptation phase for each condition...")
    frozen_gains = {}
    for cond in CONDITIONS:
        print(f"  Adapting: {CONDITION_LABEL[cond]}")
        frozen_gains[cond] = run_adaptation_phase(dyn, stim_gen, cond)

    # ==========================================================================
    # Contrast response functions of the cRF neuron that prefers the adaptor
    # ==========================================================================
    print("Computing contrast response functions...")

    def crf_curve(cond):
        g_cRF, g_surround, v_cRF, v_surround = frozen_gains[cond]
        resp = np.zeros(N_CONTRASTS)
        for i, c in enumerate(tqdm(CRF_CONTRASTS, desc=f"    {cond}", leave=False)):
            probe = probe_input_drive(adaptor_rad, c)
            y = get_response(dyn, probe, g_cRF, g_surround, v_cRF, v_surround)
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
