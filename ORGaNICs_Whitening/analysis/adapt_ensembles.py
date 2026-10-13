'''
adapt_ensembles.py

Adaptation to stimulus ensembles in the joint normalization + adaptation model
(V1Dynamics_Surround). The normalization pool includes all neurons, while adaptation (gain
feedback) is local to each RF.

Population: N_SETS sets of N_RF primary neurons (6 x 13 = 78): one classical RF (cRF) and 5
surround sets, all sharing the same tuning-curve basis and per-RF frame.

Methodology:
1. The cRF adapts to a uniform and to a biased (adaptor-heavy) ensemble of gratings, with a flat
   baseline in the surround (generate_surround_ensembles). The biased stimulus sequence is also
   re-run with gains frozen at zero: the non-adapted network.
2. Tuning curves are recorded during each run: over its 2nd half, the response to every
   presentation is read at the presentation's last timestep and averaged per orientation.
3. Ensemble statistics (mean responses, covariances) use the smallest stimulus set with exactly
   the ensemble's distribution (minimal_ensemble), mapped through those tuning curves.
4. Figures: a subset of g_cRF vs time; stimulus covariance vs gain-implied factorizations;
   ensemble histograms + biased-adapted tuning curves; input covariance, tuning repulsion and
   1st/2nd-order equalization vs time.
'''

import os
import sys
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import numpy as np
import matplotlib.pyplot as plt
from simulation_whiten import Frame, V1Dynamics_Surround
from tunings_whiten import V1Tunings
from stimuli_whiten import StimulusGenerator
from response_tools import curve_peak_deg
import Analytic_responses as AR

N_RF       = 13                    # Number of primary neurons per receptive field
N_SETS     = 6                     # 1 classical RF (cRF) + 5 surround sets
CRF_IDX    = 0                     # Index of cRF (arbitrary; sets are symmetric)
CRF_SLICE  = slice(CRF_IDX * N_RF, (CRF_IDX + 1) * N_RF)
FRAME_PATH = os.path.join(REPO_ROOT, "data/frames/N13_mercedes_K182_Frame.csv")
TARGET_COV_PATH = os.path.join(REPO_ROOT, "data/target_covs/uniform_target_covariance_high_c.csv")

ENSEMBLE_CONTRAST    = 0.8       # contrast of the adaptation ensembles (baseline & adaptor)
TUNING_WIDTH         = 0.75
ADAPT_STREAM_LENGTH  = 10000     # timesteps of each adaptation run (dt=0.1 -> 1000s)
DURATION             = 10        # timesteps each individual adaptation stimulus is held for

CONDITION_LABEL = {
    'no adaptation':          'No adaptation',
    'adapt CRF only':         'cRF adapted',
    'adapt surround only':    'Surround adapted',
    'adapt CRF and surround': 'cRF + surround adapted',
    'uniform adaptation':     'Uniform adaptation',
}
# Ensemble routing per condition. 'no adaptation' (unbiased, both regions) has zero gains by
# construction and is never simulated; 'uniform adaptation' is the unbiased ensemble in the cRF,
# genuinely adapted.
ADAPT_LOCATION_FOR_COND = {
    'no adaptation':          'adapt CRF and surround',
    'adapt CRF only':         'adapt CRF only',
    'adapt surround only':    'adapt surround only',
    'adapt CRF and surround': 'adapt CRF and surround',
    'uniform adaptation':     'adapt CRF only',
}
BIASED_FOR_COND = {
    'no adaptation':          False,
    'adapt CRF only':         True,
    'adapt surround only':    True,
    'adapt CRF and surround': True,
    'uniform adaptation':     False,
}

UNIFORM_COND = 'uniform adaptation'   # reference for tuning-curve normalization and repulsion
BIASED_COND  = 'adapt CRF only'       # biased-ensemble adaptation analyzed in every figure
GAIN_TIMECOURSE_COND = BIASED_COND    # condition whose gains are plotted vs time

SIM_HISTORY = {}   # full history of each adaptation run (y, gains, stream, centers), keyed by condition


def run_adaptation_phase(dyn, stim_gen, cond):
    '''
    Adapts to cond's ensemble from a zero initial state and returns the final (g_cRF, g_surround,
    v_cRF, v_surround, mu_cRF, mu_surround, (stream, centers)); the full run is kept in
    SIM_HISTORY[cond].

    'no adaptation' returns zeros without simulating (zero gain feedback is its known steady
    state). For single-region conditions, the region that doesn't get the ensemble only sees the
    flat baseline, so its gains, v and mu are zeroed. theta_t must already be calibrated.
    '''
    K, N_RF = dyn.frame.K, dyn.N_RF

    if cond == 'no adaptation':
        zeros_K = np.zeros(K)
        zeros_N = np.zeros(N_RF)
        return zeros_K, zeros_K, zeros_K, zeros_K, zeros_N, zeros_N, (None, None)

    stream, centers = stim_gen.generate_surround_ensembles(
        ADAPT_LOCATION_FOR_COND[cond], biased=BIASED_FOR_COND[cond], duration=DURATION,
        add_poisson_noise=False, return_angles=True)

    (y_hist, _, _, _, _, g_cRF_hist, g_surround_hist, _, _, _, _, _) = dyn.run_simulation(stream)
    SIM_HISTORY[cond] = dict(y_hist=y_hist, g_cRF_hist=g_cRF_hist,
                              g_surround_hist=g_surround_hist, stream=stream, centers=centers)

    unpacked = dyn.unpack_state(dyn.last_state)
    g_cRF       = unpacked['g_cRF']
    g_surround  = unpacked['g_surround']
    v_cRF       = unpacked['v_cRF']
    v_surround  = unpacked['v_surround']
    mu_cRF      = unpacked['mu_cRF']
    mu_surround = unpacked['mu_surround']

    if cond == 'adapt CRF only':
        g_surround = np.zeros(K)
        v_surround = np.zeros(K)
        mu_surround = np.zeros(N_RF)
    elif cond == 'adapt surround only':
        g_cRF = np.zeros(K)
        v_cRF = np.zeros(K)
        mu_cRF = np.zeros(N_RF)

    # If the ensemble never drives an interneuron's variance above theta_t, every gain clips to 0.
    g_active = g_cRF if cond != 'adapt surround only' else g_surround
    n_active = int(np.sum(g_active > 1e-3))
    print(f"  [{cond}] gains active (>1e-3): {n_active}/{K} interneurons "
          f"(mean={g_active.mean():.4g}, max={g_active.max():.4g})")
    if n_active == 0:
        print(f"  WARNING: [{cond}] every gain collapsed to zero - this condition's stimulus "
              f"never drove any interneuron's variance above the calibrated theta_t.")

    return g_cRF, g_surround, v_cRF, v_surround, mu_cRF, mu_surround, (stream, centers)


def run_gains_frozen(dyn, stream):
    '''
    Runs stream with dg/dt = 0 (tau_g -> inf), so the gains stay at their zero initial value: the
    non-adapted network seeing exactly the same stimuli. Returns y_hist (rectified y).
    '''
    tau_g = dyn.tau_g
    dyn.tau_g = np.inf
    try:
        (y_hist, _, _, _, _, g_cRF_hist, g_surround_hist, *_) = dyn.run_simulation(stream)
    finally:
        dyn.tau_g = tau_g
    assert not g_cRF_hist.any() and not g_surround_hist.any(), "gains moved during the gains-frozen run"
    return y_hist


def orientation_index(angles, num_angles):
    '''Index k of each angle on stimuli_whiten's orientation grid, theta_inputs[k] = k * pi / num_angles.'''
    return np.round(np.asarray(angles) * num_angles / np.pi).astype(int) % num_angles


def tuning_curves(y_hist, centers, num_angles, window=None):
    '''
    In-stream tuning curves: for each presentation (DURATION-step block) whose last timestep lies
    in window = [t_start, t_stop) (default: the run's 2nd half), the response at that last
    timestep, averaged over all presentations of the same orientation.

    Returns (rate, y, counts): mean firing rate (rectified y squared) and mean rectified y, each
    (N_TOT, num_angles), and the number of presentations averaged per orientation.
    '''
    n_steps = y_hist.shape[1]
    t_start, t_stop = window if window is not None else (n_steps // 2, n_steps)
    last = np.arange(DURATION - 1, n_steps, DURATION)
    last = last[(last >= t_start) & (last < t_stop)]
    stim_idx = orientation_index(centers[last], num_angles)
    counts = np.bincount(stim_idx, minlength=num_angles)
    assert np.all(counts > 0), (
        f"orientations {np.flatnonzero(counts == 0)} never presented in window [{t_start}, {t_stop})")

    y = y_hist[:, last]   # run_simulation stores rectified y, max(y, 0)
    y_mean = np.stack([y[:, stim_idx == s].mean(axis=1) for s in range(num_angles)], axis=1)
    rate_mean = np.stack([(y[:, stim_idx == s] ** 2).mean(axis=1) for s in range(num_angles)], axis=1)
    return rate_mean, y_mean, counts


def minimal_ensemble(cond, num_angles):
    '''
    Smallest stimulus set whose distribution equals cond's adaptation ensemble exactly, built by
    generate_surround_ensembles itself (same routing, baseline and contrast) with duration=1.
    Uniform: one of each of the n = num_angles orientations. Biased: the generator overwrites the
    first third of its presentations with the adaptor, so P(adaptor) = (n+2)/(3n) and
    P(other) = 2/(3n): n+2 adaptors + 2 of each other orientation (39 stimuli for n = 13).

    Returns (stimuli (N_TOT, n_stimuli), orientation index of each stimulus).
    '''
    n_stimuli = 3 * num_angles if BIASED_FOR_COND[cond] else num_angles
    ensemble_gen = StimulusGenerator(N_RF=N_RF, N_SETS=N_SETS, num_angles=num_angles,
                                     stream_length=n_stimuli, tuning_width=TUNING_WIDTH,
                                     contrast=ENSEMBLE_CONTRAST)
    stimuli, centers = ensemble_gen.generate_surround_ensembles(
        ADAPT_LOCATION_FOR_COND[cond], biased=BIASED_FOR_COND[cond], duration=1, return_angles=True)
    return stimuli, orientation_index(centers, num_angles)


def bin_by_preference(response, neuron_preferences, n_bins=N_RF):
    '''Averages response rows into n_bins equal bins of neuron preference (one neuron per bin
    when n_bins == N_RF). Matches Surround_Analytic_Responses.py's bin_by_preference.'''
    discrete_step = np.pi / len(neuron_preferences)
    bin_edges = np.linspace(0, np.pi, n_bins + 1) - (discrete_step / 2)
    binned = np.zeros((n_bins, response.shape[1]))
    bin_idx = np.clip(np.digitize(neuron_preferences, bin_edges) - 1, 0, n_bins - 1)
    for b in range(n_bins):
        mask = bin_idx == b
        if np.any(mask):
            binned[b, :] = np.mean(response[mask, :], axis=0)
    return binned


def first_order_equalization(rate, ens_idx):
    '''||r_hat - 1||_2 with r_hat = r / mean(r), where r is each cRF neuron's mean firing rate over
    the ensemble stimuli ens_idx (rate: tuning curves, (N_TOT, num_angles)). 0 = fully equalized.'''
    r = rate[CRF_SLICE][:, ens_idx].mean(axis=1)
    return np.linalg.norm(r / r.mean() - 1)


def response_covariance(y, ens_idx):
    '''cRF covariance of rectified y over the ensemble stimuli ens_idx (y: tuning curves,
    (N_TOT, num_angles)). ddof=0: the exact covariance of the ensemble distribution.'''
    return np.cov(y[CRF_SLICE][:, ens_idx], bias=True)


def second_order_equalization(y, ens_idx, target_cov):
    '''Largest eigenvalue of Cov_responses - Cov_target.'''
    return np.linalg.eigvalsh(response_covariance(y, ens_idx) - target_cov)[-1]


if __name__ == "__main__":

    print("Initializing tunings, frame, and dynamics...")
    tunings = V1Tunings(N=N_RF)
    frame   = Frame(csv_path=FRAME_PATH)
    dyn     = V1Dynamics_Surround(tunings, frame, N_RF=N_RF, N_SETS=N_SETS,
                                   target_covariance_path=TARGET_COV_PATH, gains_nonneg=True)

    stim_gen = StimulusGenerator(N_RF=N_RF, N_SETS=N_SETS, num_angles=N_RF,
                                  stream_length=ADAPT_STREAM_LENGTH,
                                  tuning_width=TUNING_WIDTH, contrast=ENSEMBLE_CONTRAST)
    n_angles = stim_gen.num_angles
    adaptor_idx = n_angles // 2                      # matches generate_surround_ensembles' own adaptor
    adaptor_deg = np.degrees(stim_gen.theta_inputs[adaptor_idx])
    stim_deg = np.degrees(stim_gen.theta_inputs)     # orientations the tuning curves are sampled at
    K = dyn.frame.K

    print("Calibrating theta_t from the uniform target covariance...")
    dyn.calibrate_theta_t(C_zz_uniform=dyn.uniform_target_covariance, uniform_target=True)

    print("Running adaptation phase...")
    final_gains = {}
    for cond in (UNIFORM_COND, BIASED_COND):
        print(f"  Adapting: {CONDITION_LABEL[cond]}")
        final_gains[cond] = run_adaptation_phase(dyn, stim_gen, cond)

    print("  Non-adapted: re-running the biased stimulus sequence with gains frozen at zero...")
    y_hist_nonadapted = run_gains_frozen(dyn, SIM_HISTORY[BIASED_COND]['stream'])

    # ==========================================================================
    # Diagnostic: theoretical optimal g_cRF (Analytic_responses.get_optimal_gains_target) for the
    # same stimulus stream and target covariance, vs. the network's final g_cRF.
    # ==========================================================================
    print("Computing theoretical optimal gains for comparison against the simulated network...")
    AR.N_matrix = tunings.N_matrix   # single-RF (N_RF, N_RF) pooling matrix, as get_optimal_gains_target expects
    AR.sigma = dyn.sigma             # match the live model's sigma, not Analytic_responses.py's default

    frozen_g_cRF, _, _, _, _, _, (gain_check_stream, _) = final_gains[BIASED_COND]
    stimuli_for_theory = gain_check_stream[:N_RF, :].T   # (T, N_RF) - cRF block only, matches frame.W's shape
    g_optimal_cRF = AR.get_optimal_gains_target(
        stimuli_for_theory, dyn.frame.W, target_covariance=dyn.uniform_target_covariance)

    # ==========================================================================
    # Subset of g_cRF vs. time step - checks that the gains settle during adaptation.
    # ==========================================================================
    print("Plotting gain-settling time course...")
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
    fig_gopt.tight_layout()

    # ==========================================================================
    # Diagnostic: covariance of the input stimuli vs. the factorization each gain vector implies.
    # (I + W diag(g) W.T)^-1 maps input to the whitened response; for gains optimal for this
    # stimulus covariance it should resemble it.
    # ==========================================================================
    print("Plotting stimulus covariance vs. gain-implied factorizations...")
    stimulus_covariance = np.cov(stimuli_for_theory, rowvar=False)   # (N_RF, N_RF)

    I_N = np.eye(N_RF)
    M_opt_inv = np.linalg.inv(I_N + dyn.frame.W @ np.diag(g_optimal_cRF) @ dyn.frame.W.T)
    M_frozen_inv = np.linalg.inv(I_N + dyn.frame.W @ np.diag(frozen_g_cRF) @ dyn.frame.W.T)

    fig_fact, axes_fact = plt.subplots(1, 3, figsize=(15, 5))
    for ax, mat, title in zip(axes_fact,
                               [stimulus_covariance, M_opt_inv, M_frozen_inv],
                               ["Cov(input stimuli)",
                                r"$(I + W\,\mathrm{diag}(g_{opt,target})\,W^T)^{-1}$",
                                r"$(I + W\,\mathrm{diag}(g_{frozen})\,W^T)^{-1}$"]):
        # Each panel auto-scaled to its own min/max - the matrices can differ by orders of magnitude.
        im = ax.imshow(mat, cmap='viridis', aspect='auto')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlabel("cRF neuron index", fontsize=11, fontweight='bold')
        ax.set_ylabel("cRF neuron index", fontsize=11, fontweight='bold')
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(2.0)
    fig_fact.suptitle(f"Stimulus covariance vs. gain-implied factorizations ({CONDITION_LABEL[BIASED_COND]})",
                       fontsize=14, fontweight='bold')
    fig_fact.tight_layout()

    # ==========================================================================
    # In-stream tuning curves. Each snapshot is the biased-adaptation state at one time step, and
    # every panel of the time-course figure (except the input covariance) loops over them. To add
    # time points, append snapshots of other windows of the biased run, e.g.
    # snapshot(t, biased['y_hist'], biased['centers'], window=(t - 2000, t), label=...).
    # ==========================================================================
    print("Recording in-stream tuning curves...")

    def snapshot(t, y_hist, centers, window=None, label=''):
        rate, y, counts = tuning_curves(y_hist, centers, n_angles, window)
        print(f"  {label} (t={t}): {counts.sum()} presentations, {counts.min()}-{counts.max()} per orientation")
        return dict(t=t, rate=rate, y=y)

    uniform = SIM_HISTORY[UNIFORM_COND]
    biased  = SIM_HISTORY[BIASED_COND]
    n_steps = biased['y_hist'].shape[1]
    uniform_ref = snapshot(n_steps, uniform['y_hist'], uniform['centers'], label='uniform-adapted')
    snapshots = [
        snapshot(0, y_hist_nonadapted, biased['centers'], label='non-adapted'),   # gains frozen at 0
        snapshot(n_steps, biased['y_hist'], biased['centers'], label='adapted'),  # 2nd-half average
    ]
    adapted_snapshots = [s for s in snapshots if s['t'] > 0]

    # ==========================================================================
    # Ensemble histograms and biased-adapted cRF tuning curves, normalized to the uniform-adapted
    # peaks. All orientation axes are relative to the adaptor.
    # ==========================================================================
    print("Plotting ensemble histograms and tuning curves...")
    N_BINS = N_RF
    uni_angles_deg  = np.degrees(uniform['centers'])   # stimulus centers actually shown during each run
    bias_angles_deg = np.degrees(biased['centers'])

    binned_uniform = bin_by_preference(uniform_ref['rate'][CRF_SLICE], tunings.theta)
    binned_crf     = bin_by_preference(adapted_snapshots[-1]['rate'][CRF_SLICE], tunings.theta)

    # Each neuron's biased-adapted curve as a fraction of ITS OWN uniform-adapted peak.
    peak_uniform = np.max(binned_uniform, axis=1, keepdims=True)
    norm_crf     = binned_crf / (peak_uniform + 1e-12)

    # num_angles == N_RF is odd and the adaptor is one of the presented angles, so the relative
    # angles land exactly on the centers of N_BINS equal bins spanning [-90, 90].
    uni_angles_rel  = (uni_angles_deg  - adaptor_deg + 90) % 180 - 90
    bias_angles_rel = (bias_angles_deg - adaptor_deg + 90) % 180 - 90
    bins_hist = np.linspace(-90, 90, N_BINS + 1)
    weights_uni  = np.ones_like(uni_angles_rel)  / len(uni_angles_rel)
    weights_bias = np.ones_like(bias_angles_rel) / len(bias_angles_rel)

    x_axis = (stim_deg - adaptor_deg + 90) % 180 - 90
    sort_idx = np.argsort(x_axis)
    x_axis_sorted = x_axis[sort_idx]

    blue_colors = plt.cm.Blues(np.linspace(0.2, 1.0, N_BINS))

    fig_tc, axes_tc = plt.subplots(1, 3, figsize=(13, 3.75))
    ax_uni_hist, ax_bias_hist, ax_tc_bias = axes_tc

    ax_uni_hist.hist(uni_angles_rel, bins=bins_hist, weights=weights_uni, color='black', rwidth=0.9)
    ax_bias_hist.hist(bias_angles_rel, bins=bins_hist, weights=weights_bias, color='black', rwidth=0.9)
    # Shared y-scale so the two ensembles' probabilities are directly comparable
    hist_ymax = max(ax_uni_hist.get_ylim()[1], ax_bias_hist.get_ylim()[1])
    for ax in (ax_uni_hist, ax_bias_hist):
        ax.set_ylim(0, hist_ymax)
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)

    for i in range(N_BINS):
        ax_tc_bias.plot(x_axis_sorted, norm_crf[i][sort_idx], color=blue_colors[i], linewidth=2.0)
    ymin, ymax = ax_tc_bias.get_ylim()
    ax_tc_bias.set_ylim(ymin - 0.05 * (ymax - ymin), ymax)
    ax_tc_bias.set_yticks([0, 1])

    for ax in axes_tc:
        ax.set_xlim(-90, 90)
        ax.set_xticks([-90, 0, 90])
        ax.set_xlabel("Stimulus (°)", fontsize=18)
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', labelsize=16)

    ax_tc_bias.spines['left'].set_linewidth(1.8)
    ax_tc_bias.spines['bottom'].set_linewidth(1.8)
    ax_tc_bias.tick_params(axis='both', width=1.8, length=5)
    fig_tc.tight_layout()

    # ==========================================================================
    # Biased adaptation over time: input covariance | tuning repulsion | 1st- and 2nd-order
    # equalization. Ensemble statistics use the biased ensemble's minimal stimulus set.
    # ==========================================================================
    print("Plotting biased-adaptation time course...")
    ens_stimuli, ens_idx = minimal_ensemble(BIASED_COND, n_angles)
    input_cov = np.cov(ens_stimuli[CRF_SLICE], bias=True)   # exact ensemble covariance (ddof=0)

    # Tuning repulsion: shift of each neuron's peak from its uniform-adapted peak, wrapped to +-90 deg.
    pref_uniform_deg = np.array([curve_peak_deg(tc, stim_deg) for tc in uniform_ref['rate'][CRF_SLICE]])
    pref_uniform_rel = (pref_uniform_deg - adaptor_deg + 90) % 180 - 90
    sort_shift = np.argsort(pref_uniform_rel)   # connect neighbours in preference order

    def tuning_shift_deg(snap):
        pref = np.array([curve_peak_deg(tc, stim_deg) for tc in snap['rate'][CRF_SLICE]])
        return (pref - pref_uniform_deg + 90) % 180 - 90

    times = [s['t'] for s in snapshots]
    eq1 = [first_order_equalization(s['rate'], ens_idx) for s in snapshots]
    eq2 = [second_order_equalization(s['y'], ens_idx, dyn.uniform_target_covariance) for s in snapshots]
    for t, e1, e2 in zip(times, eq1, eq2):
        print(f"  t={t}: 1st-order equalization = {e1:.4f}, 2nd-order (lambda_max) = {e2:.4g}")

    fig_eq, (ax_cov_in, ax_shift, ax_eq1, ax_eq2) = plt.subplots(1, 4, figsize=(17.5, 3.75))

    ax_cov_in.imshow(input_cov, cmap='viridis')
    ax_cov_in.set_xticks([])
    ax_cov_in.set_yticks([])
    ax_cov_in.set_xlabel("Neurons", fontsize=18)
    ax_cov_in.set_ylabel("Neurons", fontsize=18)
    ax_cov_in.set_title("Input covariance", fontsize=16)

    # Latest snapshot in navy; earlier ones (when added) in lighter blues.
    shift_colors = list(plt.cm.Blues(np.linspace(0.35, 0.75, len(adapted_snapshots) - 1))) + ['navy']
    for snap, color in zip(adapted_snapshots, shift_colors):
        shift = tuning_shift_deg(snap)
        ax_shift.plot(pref_uniform_rel[sort_shift], shift[sort_shift],
                      color='lightblue', linestyle=':', linewidth=2.0, zorder=2)
        ax_shift.scatter(pref_uniform_rel, shift, color=color, s=50, zorder=3, label=f"t = {snap['t']}")
    if len(adapted_snapshots) > 1:
        ax_shift.legend(fontsize=10, frameon=False)
    ax_shift.set_xlim(-90, 90)
    ax_shift.set_xticks([-90, 0, 90])
    ax_shift.locator_params(axis='y', nbins=3)
    ax_shift.set_xlabel("Neuron Preference (°)", fontsize=18)
    ax_shift.set_ylabel(r"$\Delta\theta$", fontsize=18)
    ax_shift.set_title("Tuning repulsion", fontsize=16)

    for ax, values, ylabel, title in (
            (ax_eq1, eq1, r"$\|\hat{r} - \mathbf{1}\|_2$", "1st-order equalization"),
            (ax_eq2, eq2, r"$\lambda_{max}(C_{resp} - C_{target})$", "2nd-order equalization")):
        ax.axhline(0, color='gray', linestyle='--', linewidth=1.0, zorder=1)
        ax.plot(times, values, color='lightgray', linestyle=':', linewidth=2.0, zorder=2)
        ax.scatter(times, values, color='black', s=60, zorder=3)
        ax.set_xticks(times)
        ax.margins(x=0.15)
        ax.locator_params(axis='y', nbins=4)
        ax.set_xlabel("Time Step", fontsize=18)
        ax.set_ylabel(ylabel, fontsize=16)
        ax.set_title(title, fontsize=16)

    for ax in (ax_shift, ax_eq1, ax_eq2):
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.8)
        ax.spines['bottom'].set_linewidth(1.8)
        ax.tick_params(axis='both', labelsize=16, width=1.8, length=5)
    fig_eq.tight_layout()

    plt.show()
