"""
Kohn_distinct_plots.py

Tests whether adaptation DURATION and SPATIAL EXTENT produce distinct flank-neuron tuning-curve
effects (Kohn 2015-style dissociation), using V1Dynamics_Surround's single continuous gain
trajectory (one tau_g) checkpointed at three points, rather than two separate mechanisms.
Unlike Dario_plots.py/adapt_ensembles.py, the adaptor here is a single SUSTAINED
(unchanging) oriented stimulus, matching classic single-adaptor physiology paradigms (as in
adapt_single_stimulus.py, which instead probes only the final gains of one run).
"""

import os
import sys
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import numpy as np
import matplotlib.pyplot as plt
from tunings_whiten import V1Tunings
from stimuli_whiten import StimulusGenerator
from simulation_whiten import Frame, V1Dynamics_Surround
from response_tools import get_response_offline, offline_gain_operator, settled_surround_msg, to_steps

# ---- Durations (ms, the units of the model's time constants) ----
DT = 0.1                        # simulation time step
# Conduction delay on the surround's contribution to the cRF's normalization pool.
# V1Dynamics_Surround.set_surround_delay takes it in ms and converts it to timesteps itself.
# 0.0 reproduces the pre-delay model exactly.
SURROUND_DELAY = 40.0
# Times along the ONE adaptation trajectory to checkpoint
CHECKPOINTS = {'Short': 400.0, 'Medium': 4000.0, 'Long': 10000.0}
BASELINE_SETTLE_TIME = 500.0    # time the network settles under the flat baseline before adapting

# ---- The same durations in timesteps ----
CHECKPOINT_STEPS = {label: to_steps(t, DT, f"CHECKPOINTS['{label}']") for label, t in CHECKPOINTS.items()}
N_STEPS = max(CHECKPOINT_STEPS.values())   # adaptation run length: up to the last checkpoint
BASELINE_SETTLE_STEPS = to_steps(BASELINE_SETTLE_TIME, DT, 'BASELINE_SETTLE_TIME')

N_RF, N_SETS = 13, 6
FRAME_PATH = os.path.join(REPO_ROOT, "data/frames/N13_mercedes_K182_Frame.csv")
TARGET_COV_PATH = os.path.join(REPO_ROOT, "data/target_covs/uniform_target_covariance_high_c.csv")
TUNING_WIDTH = 0.75

ADAPTOR_THETA = np.pi / 2       # radians (centered in the [0, pi) probe sweep -> mid-plot on x-axis)
ADAPTOR_CONTRAST = 0.8
PROBE_CONTRAST = 1.0
N_PROBES = 90
BASELINE_CONTRAST = 0.3         # contrast of the flat, orientation-less baseline seen before adapting

CONDITIONS = ['adapt CRF only', 'adapt CRF and surround']
COND_LABEL = {'adapt CRF only': 'cRF only', 'adapt CRF and surround': 'cRF + Surround'}


def adapted_gains(dyn, stim_gen, cond, surround_msg_baseline):
    '''Holds the adaptor (routed per cond) for N_STEPS from a zero initial state and returns
    {checkpoint label: (g_cRF, g_surround)}.'''
    adaptor = stim_gen.oriented_drive(ADAPTOR_THETA, ADAPTOR_CONTRAST, cond)
    stream = np.broadcast_to(adaptor[:, None], (adaptor.size, N_STEPS))   # same stimulus every timestep
    _, _, _, _, _, g_cRF_hist, g_surround_hist, *_ = dyn.run_simulation(
        stream, surround_msg_baseline=surround_msg_baseline)
    # History column t is the state after t + 1 steps, so checkpoint n steps is column n - 1.
    # Copied so the full histories can be freed.
    return {label: (g_cRF_hist[:, n - 1].copy(), g_surround_hist[:, n - 1].copy())
            for label, n in CHECKPOINT_STEPS.items()}


if __name__ == "__main__":
    print("Initializing tunings, frame, stimulus generator, and dynamics...")
    tunings = V1Tunings(N=N_RF)
    frame = Frame(csv_path=FRAME_PATH)
    stim_gen = StimulusGenerator(N_RF=N_RF, N_SETS=N_SETS, num_angles=N_RF, tuning_width=TUNING_WIDTH)
    dyn = V1Dynamics_Surround(tunings, frame, dt=DT, N_RF=N_RF, N_SETS=N_SETS,
                               target_covariance_path=TARGET_COV_PATH, gains_nonneg=True)

    print("Calibrating theta_t...")
    dyn.calibrate_theta_t(C_zz_uniform=dyn.uniform_target_covariance, uniform_target=True)

    print("Settling the surround's signal under the flat baseline (no delay)...")
    baseline = stim_gen.oriented_drive(0.0, BASELINE_CONTRAST, 'no adaptation')   # theta unused
    surround_msg_baseline = settled_surround_msg(dyn, baseline, BASELINE_SETTLE_STEPS)
    dyn.set_surround_delay(SURROUND_DELAY)

    print(f"Running adaptation phase (sustained adaptor, cRF-only and cRF+surround, "
          f"surround_delay={SURROUND_DELAY:g}ms)...")
    gains = {cond: adapted_gains(dyn, stim_gen, cond, surround_msg_baseline) for cond in CONDITIONS}

    # Flank neuron: positive-angle neighbor of the adaptor (right of it on a tuning plot).
    adaptor_idx = int(np.argmin(np.abs(tunings.theta - ADAPTOR_THETA)))
    flank_idx = (adaptor_idx + 1) % N_RF
    probe_angles = np.linspace(0, np.pi, N_PROBES, endpoint=False)
    # One probe per column, shaped like the condition's adaptor
    probes = {cond: np.stack([stim_gen.oriented_drive(theta, PROBE_CONTRAST, cond) for theta in probe_angles],
                             axis=1)
              for cond in CONDITIONS}

    def flank_tuning_curve(cond, g_cRF, g_surround):
        '''Flank-neuron response to cond's probes, with the gains frozen at (g_cRF, g_surround).'''
        M = offline_gain_operator(dyn, g_cRF, g_surround)
        return get_response_offline(dyn, probes[cond], M)[flank_idx]

    print("Computing flank-neuron tuning curves...")
    zeros_K = np.zeros(dyn.frame.K)
    baseline_curves = {cond: flank_tuning_curve(cond, zeros_K, zeros_K) for cond in CONDITIONS}   # non-adapted
    tuning_curves = {(cond, label): flank_tuning_curve(cond, *gains[cond][label])
                     for cond in CONDITIONS for label in CHECKPOINTS}

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
