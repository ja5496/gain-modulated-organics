'''
adapt_single_stimulus.py

Adaptation to a single, prolonged oriented stimulus in the joint normalization + adaptation model
(V1Dynamics_Surround). The normalization pool includes all neurons, while adaptation (gain
feedback) is local to each RF.

Population: N_SETS sets of N_RF primary neurons (6 x 13 = 78): one classical RF (cRF) and 5
surround sets, all sharing the same tuning-curve basis and per-RF frame.

Methodology:
1. For each condition, one grating at ADAPTOR_THETA (StimulusGenerator.oriented_drive, routed to
   the adapted region(s), flat baseline elsewhere) is held for ADAPTATION_DURATION ms. The whole
   network's final state is kept, including the region that only saw the flat baseline (which
   adapts too).
2. The final gains are frozen and the network is probed with full-field gratings (response_tools:
   closed-form fixed point by default, settled RK4 dynamics if online=True).
3. Figures: a subset of g_cRF vs time; contrast-response functions of the cRF subpopulation
   (l2 norm of its response) at the adaptor orientation; tuning curves of a flank neuron 20.8 deg
   from the adaptor.
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
from response_tools import (offline_gain_operator, get_response_offline, get_response_online,
                            curve_peak_deg, to_steps)

# ---- Durations (ms, the units of the model's time constants) ----
DT                   = 0.1         # simulation time step (in ms)
ADAPTATION_DURATION  = 2100.0      # time (in ms) the adaptor is held for 
PROBE_SETTLE_TIME    = 150.0       # time (in ms) each online probe (online=True) is settled for

# ---- The same durations in timesteps ----
ADAPTATION_STEPS     = to_steps(ADAPTATION_DURATION, DT, 'ADAPTATION_DURATION')
PROBE_SETTLE_STEPS   = to_steps(PROBE_SETTLE_TIME, DT, 'PROBE_SETTLE_TIME')

N_RF       = 13                    # Number of primary neurons per receptive field
N_SETS     = 6                     # 1 classical RF (cRF) + 5 surround sets
CRF_IDX    = 0                     # Index of cRF (arbitrary; sets are symmetric)
FRAME_PATH = os.path.join(REPO_ROOT, "data/frames/N13_mercedes_K182_Frame.csv")
TARGET_COV_PATH = os.path.join(REPO_ROOT, "data/target_covs/uniform_target_covariance_high_c.csv")

ADAPTOR_CONTRAST     = 0.8         # contrast of the sustained adaptor (same as adapt_ensembles.py's ensembles)
TUNING_WIDTH         = 0.75
ADAPTOR_THETA        = np.pi / 2   # radians: midway between neurons 6 (83.08 deg) and 7 (96.92 deg)

# If True, probes settle the full RK4 dynamics with gains frozen (get_response_online - slow,
# exact). If False, they use the closed-form fixed point (get_response_offline - fast).
online = False

N_CONTRASTS    = 20
CRF_CONTRASTS  = np.logspace(-2, 0, N_CONTRASTS)
PROBE_CONTRAST = 1.0
N_PROBES       = 180

# Setting colors for plot lines (designated by what section of the visual field is adapted)
COLOR_NONE   = 'black'
COLOR_CRF    = '#FDE68A'     # pastel yellow
COLOR_NONCRF = 'red'
COLOR_BOTH   = 'darkorange'

CONDITIONS = ['no adaptation', 'adapt CRF only', 'adapt surround only', 'adapt CRF and surround']
CONDITION_LABEL = {
    'no adaptation':          'No adaptation',
    'adapt CRF only':         'cRF adapted',
    'adapt surround only':    'Surround adapted',
    'adapt CRF and surround': 'cRF + surround adapted',
}
CONDITION_COLOR = {
    'no adaptation':          COLOR_NONE,
    'adapt CRF only':         COLOR_CRF,
    'adapt surround only':    COLOR_NONCRF,
    'adapt CRF and surround': COLOR_BOTH,
}
GAIN_TIMECOURSE_COND = 'adapt CRF and surround'   # condition whose gains are plotted vs time

SIM_HISTORY = {}   # g_cRF history (K, n_steps) of each adaptation run, keyed by condition


def run_adaptation_phase(dyn, stim_gen, cond):
    '''
    Holds the adaptor (ADAPTOR_THETA, routed per cond) for ADAPTATION_DURATION ms from a zero
    initial state and returns the final (g_cRF, g_surround, v_cRF, v_surround, mu_cRF, mu_surround)
    of the whole network - including, for single-region conditions, the region that only saw the
    flat baseline. 'no adaptation' returns all zeros without simulating (zero gain feedback is its
    known steady state). theta_t must already be calibrated.
    '''
    K, N_RF = dyn.frame.K, dyn.N_RF

    if cond == 'no adaptation':
        zeros_K = np.zeros(K)
        zeros_N = np.zeros(N_RF)
        return zeros_K, zeros_K, zeros_K, zeros_K, zeros_N, zeros_N

    adaptor = stim_gen.oriented_drive(ADAPTOR_THETA, ADAPTOR_CONTRAST, cond)
    stream = np.tile(adaptor[:, None], ADAPTATION_STEPS)   # same stimulus at every timestep

    (_, _, _, _, _, g_cRF_hist, _, _, _, _, _, _) = dyn.run_simulation(stream)
    SIM_HISTORY[cond] = g_cRF_hist

    unpacked = dyn.unpack_state(dyn.last_state)
    g_cRF       = unpacked['g_cRF']
    g_surround  = unpacked['g_surround']
    v_cRF       = unpacked['v_cRF']
    v_surround  = unpacked['v_surround']
    mu_cRF      = unpacked['mu_cRF']
    mu_surround = unpacked['mu_surround']

    # If the adaptor never drives an interneuron's variance above theta_t, every gain clips to 0.
    g_active = g_cRF if cond != 'adapt surround only' else g_surround
    n_active = int(np.sum(g_active > 1e-3))
    print(f"  [{cond}] gains active (>1e-3): {n_active}/{K} interneurons "
          f"(mean={g_active.mean():.4g}, max={g_active.max():.4g})")
    if n_active == 0:
        print(f"  WARNING: [{cond}] every gain collapsed to zero - the adaptor never drove any "
              f"interneuron's variance above the calibrated theta_t.")

    return g_cRF, g_surround, v_cRF, v_surround, mu_cRF, mu_surround


if __name__ == "__main__":

    print("Initializing tunings, frame, and dynamics...")
    tunings  = V1Tunings(N=N_RF)
    frame    = Frame(csv_path=FRAME_PATH)
    dyn      = V1Dynamics_Surround(tunings, frame, dt=DT, N_RF=N_RF, N_SETS=N_SETS,
                                    target_covariance_path=TARGET_COV_PATH, gains_nonneg=True)
    stim_gen = StimulusGenerator(N_RF=N_RF, N_SETS=N_SETS, num_angles=N_RF, tuning_width=TUNING_WIDTH)

    crf_slice = slice(CRF_IDX * N_RF, (CRF_IDX + 1) * N_RF)   # cRF subpopulation

    print("Calibrating theta_t from the uniform target covariance...")
    dyn.calibrate_theta_t(C_zz_uniform=dyn.uniform_target_covariance, uniform_target=True)

    print("Running adaptation phase for each condition...")
    frozen_gains = {}
    for cond in CONDITIONS:
        print(f"  Adapting: {CONDITION_LABEL[cond]}")
        frozen_gains[cond] = run_adaptation_phase(dyn, stim_gen, cond)

    # ==========================================================================
    # Subset of g_cRF vs. time step - checks that the gains settle during adaptation.
    # ==========================================================================
    print("Plotting gain-settling time course...")
    K = dyn.frame.K
    g_cRF_hist = SIM_HISTORY[GAIN_TIMECOURSE_COND]   # (K, n_steps)
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
    fig_gopt.tight_layout()

    # ==========================================================================
    # Frozen-gain probes. Probes always cover cRF and surround, whatever region was adapted.
    # ==========================================================================
    def full_field_probe(theta, contrast):
        return stim_gen.oriented_drive(theta, contrast, 'adapt CRF and surround')

    def probe_response(cond, probe, M):
        '''Response of every neuron to one probe with cond's gains frozen (M: offline operator).'''
        if online:
            g_cRF, g_surround, _, _, mu_cRF, mu_surround = frozen_gains[cond]
            y, _, _ = get_response_online(dyn, probe, g_cRF, g_surround, mu_cRF, mu_surround,
                                          n_steps=PROBE_SETTLE_STEPS)
            return y
        return get_response_offline(dyn, probe, M)

    def gain_operator(cond):
        g_cRF, g_surround, *_ = frozen_gains[cond]
        return offline_gain_operator(dyn, g_cRF, g_surround)

    # ==========================================================================
    # Contrast response functions of the cRF subpopulation to the adaptor-orientation probe
    # ==========================================================================
    print("Computing contrast response functions...")

    def crf_curve(cond):
        '''cRF population response magnitude (l2 norm over the N_RF cRF neurons) vs contrast.'''
        M = gain_operator(cond)
        resp = np.zeros(N_CONTRASTS)
        for i, c in enumerate(tqdm(CRF_CONTRASTS, desc=f"    {cond}", leave=False)):
            resp[i] = np.linalg.norm(probe_response(cond, full_field_probe(ADAPTOR_THETA, c), M)[crf_slice])
        return resp

    curves_by_condition = {cond: crf_curve(cond) for cond in CONDITIONS}

    fig_combined, (ax_crf, ax_flank) = plt.subplots(1, 2, figsize=(8, 3.5))

    # Normalize by the no-adaptation peak so the unadapted response is the reference (= 1)
    crf_norm = curves_by_condition['no adaptation'].max()
    for cond in CONDITIONS:
        ax_crf.plot(CRF_CONTRASTS, curves_by_condition[cond] / crf_norm, color=CONDITION_COLOR[cond],
                    linewidth=3.5, label=CONDITION_LABEL[cond])

    ax_crf.set_xscale('log')
    ax_crf.set_xlabel("Contrast", fontsize=20)
    ax_crf.set_ylabel("Norm. Response", fontsize=20)
    ax_crf.set_ylim(bottom=0)
    ax_crf.set_yticks([0, 1])
    ax_crf.grid(False)
    ax_crf.spines['top'].set_visible(False)
    ax_crf.spines['right'].set_visible(False)
    ax_crf.spines['left'].set_linewidth(2.5)
    ax_crf.spines['bottom'].set_linewidth(2.5)
    ax_crf.tick_params(axis='both', width=2.5, length=6, labelsize=18)
    ax_crf.legend(fontsize=10, frameon=False)

    # ==========================================================================
    # Tuning curve of the flank neuron
    # ==========================================================================
    print("Computing flank-neuron tuning curves...")
    probe_angles = np.linspace(0, np.pi, N_PROBES, endpoint=False)
    probe_angles_deg = np.degrees(probe_angles)
    adaptor_deg = np.degrees(ADAPTOR_THETA)
    # Flank neuron: negative-angle neighbor of neuron 6 (83.08 deg) -> neuron 5 (69.23 deg), 20.8 deg
    # below the 90 deg adaptor (the mirror image of Kohn_distinct_plots.py's flank, 20.8 deg above).
    adaptor_idx = stim_gen.num_angles // 2
    flank_idx = (adaptor_idx - 1) % N_RF
    FLANK_CONDITIONS = ['no adaptation', 'adapt surround only', 'adapt CRF only']

    def crf_tuning_curves(cond):
        '''(N_RF, N_PROBES) cRF firing rates to full-field probes over probe_angles.'''
        M = gain_operator(cond)
        resp = np.zeros((N_RF, N_PROBES))
        for i, ang in enumerate(tqdm(probe_angles, desc=f"    {cond}", leave=False)):
            resp[:, i] = probe_response(cond, full_field_probe(ang, PROBE_CONTRAST), M)[crf_slice]
        return resp

    flank_curves = {cond: crf_tuning_curves(cond)[flank_idx, :] for cond in FLANK_CONDITIONS}

    # Shift of each adapted condition's peak relative to the no-adaptation control, wrapped to +-90 deg.
    peak_deg = {cond: curve_peak_deg(flank_curves[cond], probe_angles_deg) for cond in FLANK_CONDITIONS}
    control_peak = peak_deg['no adaptation']
    peak_shift_deg = {cond: ((peak_deg[cond] - control_peak + 90) % 180) - 90 for cond in FLANK_CONDITIONS}

    FLANK_LEGEND_LABEL = {
        'no adaptation':       None,   # redundant: this is the zero-shift reference curve
        'adapt surround only': fr"$\Delta\theta$ = {peak_shift_deg['adapt surround only']:+.2f}°",
        'adapt CRF only':      fr"$\Delta\theta$ = {peak_shift_deg['adapt CRF only']:+.2f}°",
    }

    flank_norm = flank_curves['no adaptation'].max()
    for cond in FLANK_CONDITIONS:
        plot_kwargs = dict(color=CONDITION_COLOR[cond], linewidth=3.5)
        if FLANK_LEGEND_LABEL[cond] is not None:
            plot_kwargs['label'] = FLANK_LEGEND_LABEL[cond]
        ax_flank.plot(probe_angles_deg, flank_curves[cond] / flank_norm, **plot_kwargs)

    ax_flank.axvline(control_peak, color='grey', linestyle='--', linewidth=1.0, zorder=0)

    ax_flank.annotate('', xy=(adaptor_deg, 0.80), xytext=(adaptor_deg, 0.94),
                       xycoords=('data', 'axes fraction'), textcoords=('data', 'axes fraction'),
                       arrowprops=dict(arrowstyle='-|>', color='black', linewidth=2.5, mutation_scale=18))

    ax_flank.set_xlabel("Stimulus Orientation", fontsize=20)
    ax_flank.set_xlim(0, 180)
    ax_flank.set_xticks([0, 90, 180])
    ax_flank.set_xticklabels(['0', '90', '180'], fontsize=18)
    ax_flank.set_ylim(bottom=0)
    ax_flank.set_yticks([0, 1])
    ax_flank.grid(False)
    ax_flank.spines['top'].set_visible(False)
    ax_flank.spines['right'].set_visible(False)
    ax_flank.spines['left'].set_linewidth(2.5)
    ax_flank.spines['bottom'].set_linewidth(2.5)
    ax_flank.tick_params(axis='x', width=2.5, length=8)
    ax_flank.tick_params(axis='y', width=2.5, length=6, labelsize=18)
    ax_flank.legend(fontsize=12, frameon=False, handlelength=1.0)
    fig_combined.tight_layout()

    plt.show()
