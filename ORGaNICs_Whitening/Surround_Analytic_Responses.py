'''
Surround_Analytic_Responses.py

Calculates neural firing rates for a known input distribution using analytical expressions.
Interactions between classical receptive field (cRF) neurons and surround (non-cRF) neurons are modeled.
The normalization pool includes all neurons, while adaptation (gain feedback) is local to each RF.

Population structure: 7 sets of 13 primary neurons each (91 total). One set (CRF_IDX) is the
classical RF; the other 6 are surround sets. All 7 sets share the same tuning-curve basis and the
same per-RF frame, so the 6 surround sets are treated as fully interchangeable (no distance-dependent
weighting) -- only cRF vs. non-cRF membership matters.

Methodology:
1. Optimal gains for a single 13-neuron RF pool are computed once per ensemble type (baseline
   uniform, or adaptor-biased) and shared across every set that sees that ensemble.
2. For each of the 4 adaptation conditions (none / cRF-only / non-cRF-only / both), a block-diagonal
   feedback matrix M (91x91) is assembled from those per-RF gains, and mu = <y> is found via a joint
   self-consistency loop over the full 91-neuron population (normalization pools across all 91;
   gain feedback stays within each 13-neuron block).
3. Steady-state responses to probe stimuli (shown to a single target set only, baseline elsewhere)
   are computed in closed form from the probe, optimal gains, and mu.
'''

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from simulation_whiten import Frame
from tunings_whiten import V1Tunings
from stimuli_whiten import StimulusGenerator
import Analytic_responses as ar

# ----------------------------------------------------------------------------
# Population structure
# ----------------------------------------------------------------------------
N_RF       = 13                    # primary neurons per receptive field
N_SETS     = 7                     # 1 classical RF (cRF) + 6 surround sets
N_TOTAL    = N_RF * N_SETS         # full primary-neuron population
CRF_IDX    = 0                     # which of the 7 sets is the cRF (arbitrary; sets are symmetric)
FRAME_PATH = "Frames/N13_mercedes_Frame.csv"

sigma = 0.15
Beta  = 0.5

# ----------------------------------------------------------------------------
# Tunable knobs
# ----------------------------------------------------------------------------
ENSEMBLE_CONTRAST = 0.6      # contrast of the adaptation ensembles (baseline & adaptor)
TUNING_WIDTH      = 0.75

N_CONTRASTS   = 20
CRF_CONTRASTS = np.logspace(-2, 0, N_CONTRASTS)

PROBE_CONTRAST = 0.3       # fixed test contrast for the full orientation tuning-curve sweep --
                              # chosen in the range where Figure 1 shows adaptation curves are
                              # still separated (they reconverge above ~0.1 contrast)
N_PROBES       = 180

OFF_ADAPTOR_DEG = 20.0        # "flank" neuron offset from the adaptor orientation

COLOR_NONE   = 'black'
COLOR_CRF    = '#FDE68A'     # pastel yellow
COLOR_NONCRF = 'darkorange'
COLOR_BOTH   = 'darkred'


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def half_wave_rectify(y):
    return np.maximum(y, 0.0) ** 2

def make_grating(theta_pref, contrast, theta_grid, tuning_width=TUNING_WIDTH):
    delta = theta_grid - theta_pref
    delta = (delta + np.pi / 2) % np.pi - np.pi / 2
    profile = np.exp(-delta**2 / (2 * tuning_width**2))
    return contrast * 15 * profile / np.max(profile)

def block_diag_M(block_types, W_local, g_opt):
    '''block_types: length-N_SETS list of 'baseline' or 'adaptor', one per set.'''
    blocks = {kind: (W_local @ np.diag(g_opt[kind])) @ W_local.T for kind in ('baseline', 'adaptor')}
    M = np.zeros((N_TOTAL, N_TOTAL))
    for s, kind in enumerate(block_types):
        sl = slice(s * N_RF, (s + 1) * N_RF)
        M[sl, sl] = blocks[kind]
    return M

def joint_stream(block_types, ensembles):
    '''ensembles: {'baseline': (13,T) array, 'adaptor': (13,T) array}. Every surround set shares
    the single physical stream for its condition (a uniform annulus), rather than independent draws.'''
    T = ensembles['baseline'].shape[1]
    joint = np.zeros((N_TOTAL, T))
    for s, kind in enumerate(block_types):
        joint[s * N_RF:(s + 1) * N_RF, :] = ensembles[kind]
    return list(joint.T)

def get_mu_spatial(stimuli, M, N_matrix_full, alpha=0.1, Beta=Beta):
    '''Self-consistency loop for mu over the full population. M is block-diagonal, so gain
    feedback from one set's adaptation cannot leak into another set's z_prime.'''
    N = M.shape[0]
    mu = np.zeros(N)
    diff = 1.0
    pbar = tqdm(desc="  mu convergence", unit="iter")
    while diff > 1e-6:
        y_total = np.zeros(N)
        for z in stimuli:
            z_prime = 2 * (Beta * z - M @ mu)
            y_total += z_prime / np.sqrt(sigma**2 + N_matrix_full @ (z_prime * z_prime))
        mu_new = y_total / len(stimuli)
        diff = np.linalg.norm(mu_new - mu)
        mu += alpha * (mu_new - mu)
        pbar.set_postfix(diff=f"{diff:.2e}")
        pbar.update(1)
    pbar.close()
    return mu

def get_response(local_stimulus, mu, M, N_matrix_full, set_idx, Beta=Beta):
    '''Steady-state response of set_idx's own 13 neurons to local_stimulus, given the
    population's adaptation state (mu, M). Sets other than set_idx are NOT re-probed with a
    fictitious zero stimulus -- that would inject their (unrectified) gain-feedback residual
    into the shared pool as if it were real activity, which grows *with* adaptation instead of
    shrinking. Instead they contribute their own already-known steady-state activity, mu, so a
    more strongly adapted (lower-mu) set genuinely relieves the shared pool.'''
    sl = slice(set_idx * N_RF, (set_idx + 1) * N_RF)
    gain_feedback_target = (M @ mu)[sl]
    z_prime_target = 2 * (Beta * local_stimulus - gain_feedback_target)

    pool_vec = mu ** 2
    pool_vec[sl] = z_prime_target ** 2
    pool = N_matrix_full @ pool_vec

    return z_prime_target / np.sqrt(sigma**2 + pool[sl])


if __name__ == "__main__":

    print("Initializing tunings and frame...")
    tunings = V1Tunings(N=N_RF)
    frame   = Frame(csv_path=FRAME_PATH)
    W_local = frame.W                            # (13, 91) frame shared by every RF

    N_matrix_local = np.ones((N_RF, N_RF))        # local pool -- optimal-gains calc only
    N_matrix_full  = np.ones((N_TOTAL, N_TOTAL))  # global pool -- mu / response dynamics

    ar.sigma    = sigma
    ar.N_matrix = N_matrix_local

    # ---- Local (single-RF) ensembles used to derive shareable optimal gains ----
    stim_gen = StimulusGenerator(N=N_RF, num_angles=N_RF, stream_length=N_RF,
                                 tuning_width=TUNING_WIDTH, contrast=ENSEMBLE_CONTRAST)

    print("Generating local baseline / adaptor ensembles...")
    seq_baseline = stim_gen.generate_input_ensembles(biased=False, duration=1)
    seq_adaptor  = stim_gen.generate_input_ensembles(biased=True,  duration=1)
    stimuli_baseline = list(seq_baseline.T)
    stimuli_adaptor  = list(seq_adaptor.T)

    print("Computing optimal gains (baseline)...")
    g_opt_baseline = ar.get_optimal_gains_target(stimuli_baseline, W_local,
                                                 label='baseline', poisson_variance=True)
    print("Computing optimal gains (adaptor)...")
    g_opt_adaptor = ar.get_optimal_gains_target(stimuli_adaptor, W_local,
                                                label='adaptor', poisson_variance=True,
                                                uniform_stimuli=stimuli_baseline)
    g_opt = {'baseline': g_opt_baseline, 'adaptor': g_opt_adaptor}

    # ---- Adaptation conditions: which sets see the adaptor ensemble vs. the baseline one ----
    # 'none' is NOT a hard-zero (mu=0, M=0) system -- that would strip out the cRF's own
    # intrinsic baseline gain feedback, which the other 3 conditions all retain, confounding
    # any comparison against them. Every condition here gets a real self-consistency solve;
    # 'none' differs from the others only in that no set sees the adaptor ensemble.
    all_baseline = ['baseline'] * N_SETS
    crf_only     = ['adaptor' if s == CRF_IDX else 'baseline' for s in range(N_SETS)]
    noncrf_only  = ['baseline' if s == CRF_IDX else 'adaptor' for s in range(N_SETS)]
    both         = ['adaptor'] * N_SETS

    ensembles = {'baseline': seq_baseline, 'adaptor': seq_adaptor}

    conditions = {}
    for key, block_types, color, label in [
        ('none',   all_baseline, COLOR_NONE,   'No adaptation'),
        ('crf',    crf_only,     COLOR_CRF,    'cRF adapted'),
        ('noncrf', noncrf_only,  COLOR_NONCRF, 'Non-cRF adapted'),
        ('both',   both,         COLOR_BOTH,   'Both adapted'),
    ]:
        print(f"Computing mu ({label})...")
        M = block_diag_M(block_types, W_local, g_opt)
        stimuli = joint_stream(block_types, ensembles)
        mu = get_mu_spatial(stimuli, M, N_matrix_full)
        conditions[key] = dict(mu=mu, M=M, color=color, label=label)

    # ---- Target neurons: prefers the adaptor orientation, and the ~20-degree-off flank ----
    adaptor_idx = N_RF // 2
    adaptor_ang = tunings.theta[adaptor_idx]

    off_target  = adaptor_ang + np.deg2rad(OFF_ADAPTOR_DEG)
    off_wrapped = (tunings.theta - off_target + np.pi / 2) % np.pi - np.pi / 2
    flank_idx   = np.argmin(np.abs(off_wrapped))
    flank_ang   = tunings.theta[flank_idx]

    condition_order = ['none', 'crf', 'noncrf', 'both']

    # ==========================================================================
    # FIGURE 1 -- Contrast Response Functions (2 panels, 4 curves each)
    # ==========================================================================
    def crf_curve(neuron_idx, theta_pref, cond):
        resp = np.zeros(N_CONTRASTS)
        for i, c in enumerate(CRF_CONTRASTS):
            local_stim = make_grating(theta_pref, c, tunings.theta)
            y = get_response(local_stim, cond['mu'], cond['M'], N_matrix_full, CRF_IDX)
            resp[i] = y[neuron_idx]
        return half_wave_rectify(resp)

    print("Computing CRF curves...")
    fig1, axes1 = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)

    panel_targets = [
        (adaptor_idx, adaptor_ang, "Prefers Adaptor Orientation"),
        (flank_idx,   flank_ang,   f"Prefers ~{OFF_ADAPTOR_DEG:.0f}° Off-Adaptor"),
    ]

    for ax, (n_idx, theta_pref, title) in zip(axes1, panel_targets):
        for key in condition_order:
            cond = conditions[key]
            curve = crf_curve(n_idx, theta_pref, cond)
            ax.plot(CRF_CONTRASTS, curve, color=cond['color'], linewidth=3.5, label=cond['label'])

        ax.set_xscale('log')
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel("Contrast", fontsize=16, fontweight='bold')
        ax.tick_params(axis='both', width=2.5, length=6, labelsize=12)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(2.5)

    axes1[0].set_ylabel("Response", fontsize=16, fontweight='bold')
    axes1[0].legend(fontsize=10, loc='upper left', frameon=False)
    plt.tight_layout()
    plt.show()

    # ==========================================================================
    # FIGURE 2 -- Individual tuning curve for the flank neuron (~20 deg off-adaptor)
    # ==========================================================================
    print("Computing tuning curves...")
    probe_angles = np.linspace(0, np.pi, N_PROBES, endpoint=False)

    def tuning_curve(neuron_idx, cond):
        resp = np.zeros(N_PROBES)
        for i, ang in enumerate(probe_angles):
            local_stim = make_grating(ang, PROBE_CONTRAST, tunings.theta)
            y = get_response(local_stim, cond['mu'], cond['M'], N_matrix_full, CRF_IDX)
            resp[i] = y[neuron_idx]
        return half_wave_rectify(resp)

    fig2, ax2 = plt.subplots(figsize=(7, 5.5))
    probe_deg = probe_angles * 180 / np.pi

    for key in condition_order:
        cond = conditions[key]
        curve = tuning_curve(flank_idx, cond)
        ax2.plot(probe_deg, curve, color=cond['color'], linewidth=3.5, label=cond['label'])

    ax2.set_title(f"Flank Neuron Tuning (~{OFF_ADAPTOR_DEG:.0f}° Off-Adaptor)",
                 fontsize=16, fontweight='bold')
    ax2.set_xlabel("Stimulus Orientation (°)", fontsize=16, fontweight='bold')
    ax2.set_ylabel("Response", fontsize=16, fontweight='bold')
    ax2.tick_params(axis='both', width=2.5, length=6, labelsize=12)
    ax2.grid(False)
    ax2.legend(fontsize=10, loc='upper right', frameon=False)
    for spine in ax2.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(2.5)

    plt.tight_layout()
    plt.show()
