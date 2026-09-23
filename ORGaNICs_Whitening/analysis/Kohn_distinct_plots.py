"""
Kohn_distinct_plots.py

Tests whether adaptation DURATION and SPATIAL EXTENT produce distinct flank-neuron tuning-curve
effects (Kohn 2015-style dissociation), using V1Dynamics_Surround's single continuous gain
trajectory (one tau_g) checkpointed at three points, rather than two separate mechanisms.
Unlike Dario_plots.py/Surround_simulated_responses.py, the adaptor here is a single SUSTAINED
(unchanging) oriented stimulus, matching classic single-adaptor physiology paradigms.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from tunings_whiten import V1Tunings
from stimuli_whiten import StimulusGenerator
from scipy.linalg import block_diag
from simulation_whiten import Frame, V1Dynamics_Surround
from Surround_simulated_responses import get_response_offline

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_RF, N_SETS = 13, 6
FRAME_PATH = os.path.join(REPO_ROOT, "data/frames/N13_mercedes_K182_Frame.csv")
TARGET_COV_PATH = os.path.join(REPO_ROOT, "data/target_covs/uniform_target_covariance_mid_c.csv")
TUNING_WIDTH = 0.75
THETA_RF = np.linspace(0, np.pi, N_RF, endpoint=False)

ADAPTOR_THETA = np.pi / 2       # radians (centered in the [0, pi) probe sweep -> mid-plot on x-axis)
ADAPTOR_CONTRAST = 1.0
PROBE_CONTRAST = 0.6
N_PROBES = 90
THETA_T_CONTRAST = 0.6         # fixed low contrast used only to calibrate theta_t

# Conduction delay (seconds) on the surround's contribution to the cRF's normalization pool -
# see V1Dynamics_Surround.set_surround_delay. 0.0 reproduces the pre-delay model exactly.
SURROUND_DELAY = 4000.0

# Step indices along the ONE adaptation trajectory to checkpoint, labeled per spec - dt=0.1
# makes these labels nominal (short/medium/long), not dimensionally exact.
CHECKPOINTS = {'Short': 4000, 'Medium': 40000, 'Long': 100000}
N_STEPS = max(CHECKPOINTS.values()) + 1

CONDITIONS = ['adapt CRF only', 'adapt CRF and surround']
COND_LABEL = {'adapt CRF only': 'cRF only', 'adapt CRF and surround': 'cRF + Surround'}


def oriented_drive(theta, contrast, adapt_location, baseline=0.20):
    '''Gaussian tuning-curve profile at theta, routed to adapt_location's region(s) (flat
    baseline elsewhere), tiled across N_SETS, normalized and contrast-scaled. Used for both
    the sustained adaptor and the (spatially-matched) probe stimuli.'''
    delta = (THETA_RF - theta + np.pi / 2) % np.pi - np.pi / 2
    profile = np.exp(-delta**2 / (2 * TUNING_WIDTH**2))
    base = np.full(N_RF, baseline)
    full = (np.concatenate([profile] + [base] * (N_SETS - 1)) if adapt_location == 'adapt CRF only'
            else np.concatenate([profile] * N_SETS))
    return contrast * full / np.linalg.norm(full)


def run_adaptation_phase(dyn, stim_gen, cond, surround_msg_baseline=0.0):
    '''Adapts for N_STEPS and returns {label: (g_cRF, g_surround, mu_cRF, mu_surround)} at each
    CHECKPOINT. 'no adaptation' instead calibrates theta_t from a genuinely unbiased stream
    (stim_gen is only ever used for this) and returns the (zero-gain, baseline-mu) reference
    state for the non-adapted control curve: (g_cRF, g_surround, mu_cRF, mu_surround). The
    'no adaptation' pass always runs with whatever delay dyn currently has (the caller is
    expected to run it before calling dyn.set_surround_delay, so it establishes a clean,
    delay-free baseline - see __main__). surround_msg_baseline is only consulted by
    run_simulation while surround_delay>0 (see V1Dynamics_Surround.run_simulation).'''
    if cond == 'no adaptation':
        stream = stim_gen.generate_surround_ensembles('no adaptation', add_poisson_noise=False)
        (_, _, _, _, _, g_cRF_hist, g_surround_hist, v_cRF_hist, v_surround_hist,
         mu_cRF_hist, mu_surround_hist, _) = dyn.run_simulation(stream)
        assert np.all(g_cRF_hist == 0) and np.all(g_surround_hist == 0), (
            "calibration run's gains moved away from zero - theta_t sentinel no longer holds."
        )
        dyn.calibrate_theta_t(C_zz_uniform=dyn.uniform_target_covariance, uniform_target=True)
        zeros_K = np.zeros(dyn.frame.K)
        return zeros_K, zeros_K, mu_cRF_hist[:, -1], mu_surround_hist[:, -1]

    stimulus = oriented_drive(ADAPTOR_THETA, ADAPTOR_CONTRAST, cond)
    stream = np.tile(stimulus[:, None], N_STEPS)   # sustained: same stimulus every timestep
    (_, _, _, _, _, g_cRF_hist, g_surround_hist, _, _,
     mu_cRF_hist, mu_surround_hist, _) = dyn.run_simulation(stream, surround_msg_baseline=surround_msg_baseline)

    return {label: (g_cRF_hist[:, t], g_surround_hist[:, t], mu_cRF_hist[:, t], mu_surround_hist[:, t])
            for label, t in CHECKPOINTS.items()}


if __name__ == "__main__":
    print("Initializing tunings, frame, stimulus generator, and dynamics...")
    tunings = V1Tunings(N=N_RF)
    frame = Frame(csv_path=FRAME_PATH)
    stim_gen = StimulusGenerator(N_RF=N_RF, N_SETS=N_SETS, num_angles=N_RF, stream_length=N_STEPS,
                                  tuning_width=TUNING_WIDTH, contrast=THETA_T_CONTRAST)
    dyn = V1Dynamics_Surround(tunings, frame, N_RF=N_RF, N_SETS=N_SETS,
                               target_covariance_path=TARGET_COV_PATH, gains_nonneg=True)

    print("Calibrating theta_t...")
    baseline_state = run_adaptation_phase(dyn, stim_gen, 'no adaptation')  # runs at surround_delay=0

    # Surround's settled non-adapting contribution, used to hold the delay buffer during the
    # first surround_delay seconds of the adaptation runs below (see run_simulation's
    # surround_msg_baseline docstring) - not zero, since the surround is never actually silent.
    surround_msg_baseline = dyn._surround_msg(dyn.last_state)
    dyn.set_surround_delay(SURROUND_DELAY)

    print(f"Running adaptation phase (sustained adaptor, cRF-only and cRF+surround, "
          f"surround_delay={SURROUND_DELAY}s)...")
    frozen_states = {cond: run_adaptation_phase(dyn, stim_gen, cond, surround_msg_baseline=surround_msg_baseline)
                      for cond in CONDITIONS}

    # Flank neuron: positive-angle neighbor of the adaptor (right of it on a tuning plot).
    adaptor_idx = int(np.argmin(np.abs(THETA_RF - ADAPTOR_THETA)))
    flank_idx = (adaptor_idx + 1) % N_RF
    probe_angles = np.linspace(0, np.pi, N_PROBES, endpoint=False)

    def offline_gain_operator(g_cRF, g_surround):
        '''(N_TOTAL, N_TOTAL) block-diagonal M = W diag(g) W.T feeding get_response_offline's
        (I+M)^-1 fixed point - same cRF/surround block layout as frozen_derivatives'
        full_gain_feedback: the cRF block uses g_cRF, every surround block reuses g_surround.'''
        W = dyn.frame.W
        M_cRF = W @ np.diag(g_cRF) @ W.T
        M_surround = W @ np.diag(g_surround) @ W.T
        return block_diag(M_cRF, *([M_surround] * (N_SETS - 1)))

    def flank_tuning_curve(cond, g_cRF, g_surround, mu_cRF, mu_surround, desc):
        '''Flank-neuron response swept over probe_angles, probe shape matching cond.'''
        curve = np.zeros(N_PROBES)
        for i, theta in enumerate(tqdm(probe_angles, desc=desc, leave=False)):
            probe = oriented_drive(theta, PROBE_CONTRAST, cond)
            M = offline_gain_operator(g_cRF, g_surround)
            y = get_response_offline(dyn, probe, M)
            curve[i] = y[flank_idx]
        return curve

    print("Computing flank-neuron tuning curves...")
    tuning_curves = {(cond, label): flank_tuning_curve(cond, *frozen_states[cond][label], f"{cond} @ {label}")
                      for cond in CONDITIONS for label in CHECKPOINTS}

    print("Computing non-adapted reference curves...")
    baseline_curves = {cond: flank_tuning_curve(cond, *baseline_state, f"{cond} @ baseline")
                        for cond in CONDITIONS}

    probe_deg = np.degrees(probe_angles)
    fig, axes = plt.subplots(2, 3, figsize=(10, 6), sharex=True, sharey='row')
    for row, cond in enumerate(CONDITIONS):
        for col, label in enumerate(CHECKPOINTS):
            ax = axes[row, col]
            ax.plot(probe_deg, baseline_curves[cond], color='gray', ls='--', lw=3.0, label='No adaptation')
            ax.plot(probe_deg, tuning_curves[(cond, label)], color="#89021D", lw=3.5, label='Adapted')
            ax.axvline(np.degrees(ADAPTOR_THETA), color='gray', ls=':', lw=1.2)
            if row == 0:
                ax.set_title(label, fontsize=20, fontweight='bold')
            if col == 0:
                ax.set_ylabel(COND_LABEL[cond], fontsize=20, fontweight='bold')
            for spine in ax.spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(2.0)
            ax.spines[['top', 'right']].set_visible(False)
            ax.set_xticks([0, 90, 180])
            ax.tick_params(axis='x', labelsize=16)
            ax.tick_params(axis='y', left=False, labelleft=False)
    #axes[0, 0].legend(fontsize=10, frameon=False)

    fig.supxlabel('Stimulus Orientation (deg)', fontsize=20, fontweight='bold')
    plt.tight_layout()
    plt.show()
