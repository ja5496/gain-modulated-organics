'''
adapt_ensembles.py

Adaptation to stimulus ensembles in the joint normalization + adaptation model
(V1Dynamics_Surround). The normalization pool includes all neurons, while adaptation (gain
feedback) is local to each RF.

Population: N_SETS sets of N_RF primary neurons (6 x 13 = 78): one classical RF (cRF) and 5
surround sets, all sharing the same tuning-curve basis and per-RF frame.

Methodology:
1. The cRF and surround adapt to a uniform and to a biased (adaptor-heavy) ensemble of full-field
   gratings (generate_surround_ensembles' 'adapt CRF and surround' routing). theta_t comes from
   the single-RF target covariance rescaled by crf_response_scale, since the shared normalization
   pool suppresses the cRF's responses to full-field gratings. In every run, the surround's
   contribution to the cRF's normalization pool arrives SURROUND_DELAY ms late (as in
   Kohn_distinct_plots.py); until it does, the cRF sees the surround's settled contribution under
   a flat, non-adapting stimulus.
2. Tuning curves are recorded during each run: over its 2nd half, the response to every
   presentation is read at the presentation's last timestep and averaged per orientation. Each
   cRF tuning curve is fit with a Gaussian on the orientation circle; tuning shifts and the
   tuning-curve figure use the fits.
3. 1st- and 2nd-order equalization use the cRF responses actually produced during biased
   adaptation (mean firing rates and covariance), over consecutive STATS_WINDOW-ms windows of
   the run. The input covariance uses the smallest stimulus set with exactly the ensemble's
   distribution (minimal_ensemble).
4. Figures: a subset of g_cRF vs time; stimulus covariance vs gain-implied factorizations;
   ensemble histograms + biased-adapted tuning curves (in-stream fits, and offline fixed-point
   responses to a fine orientation grid with each run's final gains frozen); input covariance,
   tuning repulsion and 1st/2nd-order equalization vs time.
'''

import os
import sys
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from simulation_whiten import Frame, V1Dynamics_Surround
from tunings_whiten import V1Tunings
from stimuli_whiten import StimulusGenerator
from response_tools import get_response_offline, offline_gain_operator, settled_surround_msg, to_steps
import Analytic_responses as AR

# ---- Durations (ms, the units of the model's time constants) ----
DT                   = 0.1       # simulation time step
ADAPTATION_DURATION  = 4200.0    # length of each adaptation run
STIMULUS_DURATION    = 30.0      # time each individual adaptation stimulus is held for
STATS_WINDOW         = 4200.0    # length of each window of the adaptation run's 1st/2nd-order statistics
SURROUND_DELAY       = 40.0
BASELINE_SETTLE_TIME = 500.0     # time the network settles under the flat baseline before adapting

# ---- The same durations in timesteps ----
ADAPTATION_STEPS     = to_steps(ADAPTATION_DURATION, DT, 'ADAPTATION_DURATION')
STIMULUS_STEPS       = to_steps(STIMULUS_DURATION, DT, 'STIMULUS_DURATION')
STATS_WINDOW_STEPS   = to_steps(STATS_WINDOW, DT, 'STATS_WINDOW')
BASELINE_SETTLE_STEPS = to_steps(BASELINE_SETTLE_TIME, DT, 'BASELINE_SETTLE_TIME')

N_RF       = 13                    # Number of primary neurons per receptive field
N_SETS     = 6                     # 1 classical RF (cRF) + 5 surround sets
CRF_IDX    = 0                     # Index of cRF (arbitrary; sets are symmetric)
CRF_SLICE  = slice(CRF_IDX * N_RF, (CRF_IDX + 1) * N_RF)
FRAME_PATH = os.path.join(REPO_ROOT, "data/frames/N13_mercedes_K182_Frame.csv")
TARGET_COV_PATH = os.path.join(REPO_ROOT, "data/target_covs/uniform_target_covariance_mid_c.csv")

ENSEMBLE_CONTRAST    = 0.8       # contrast of the adaptation ensembles (baseline & adaptor)
TUNING_WIDTH         = 0.75

CONDITION_LABEL = {
    'no adaptation':          'No adaptation',
    'adapt CRF only':         'cRF adapted',
    'adapt surround only':    'Surround adapted',
    'adapt CRF and surround': 'cRF + surround adapted',
    'uniform adaptation':     'Uniform adaptation',
}
# Ensemble routing per condition. 'no adaptation' (unbiased, both regions) has zero gains by
# construction and is never simulated; 'uniform adaptation' is the same unbiased full-field
# ensemble, genuinely adapted.
ADAPT_LOCATION_FOR_COND = {
    'no adaptation':          'adapt CRF and surround', # CHANGED FROM CRF AND SURROUND
    'adapt CRF only':         'adapt CRF only',
    'adapt surround only':    'adapt surround only',
    'adapt CRF and surround': 'adapt CRF and surround',
    'uniform adaptation':     'adapt CRF and surround', # CHANGED FROM CRF AND SURROUND
}
BIASED_FOR_COND = {
    'no adaptation':          False,
    'adapt CRF only':         True,
    'adapt surround only':    True,
    'adapt CRF and surround': True,
    'uniform adaptation':     False,
}

UNIFORM_COND = 'uniform adaptation'   # reference for tuning-curve normalization and repulsion
BIASED_COND  = 'adapt CRF and surround'       # biased-ensemble adaptation analyzed in every figure # CHANGED FROM CRF AND SURROUND
GAIN_TIMECOURSE_COND = BIASED_COND    # condition whose gains are plotted vs time

SIM_HISTORY = {}   # full history of each adaptation run (y, gains, stream, centers), keyed by condition


def run_adaptation_phase(dyn, stim_gen, cond, surround_msg_baseline=0.0):
    '''
    Adapts to cond's ensemble from a zero initial state and returns the final (g_cRF, g_surround,
    v_cRF, v_surround, mu_cRF, mu_surround, (stream, centers)); the full run is kept in
    SIM_HISTORY[cond].

    'no adaptation' returns zeros without simulating (zero gain feedback is its known steady
    state). For single-region conditions, the region that doesn't get the ensemble only sees the
    flat baseline, so its gains, v and mu are zeroed. theta_t must already be calibrated.
    surround_msg_baseline holds the delayed surround->cRF pooling signal until a real delayed
    reading exists (only read while dyn's surround_delay > 0; see response_tools.settled_surround_msg).
    '''
    K, N_RF = dyn.frame.K, dyn.N_RF

    if cond == 'no adaptation':
        zeros_K = np.zeros(K)
        zeros_N = np.zeros(N_RF)
        return zeros_K, zeros_K, zeros_K, zeros_K, zeros_N, zeros_N, (None, None)

    stream, centers = stim_gen.generate_surround_ensembles(
        ADAPT_LOCATION_FOR_COND[cond], biased=BIASED_FOR_COND[cond], duration=STIMULUS_STEPS,
        add_poisson_noise=False, return_angles=True)

    (y_hist, _, _, _, _, g_cRF_hist, g_surround_hist, _, _, _, _, _) = dyn.run_simulation(
        stream, surround_msg_baseline=surround_msg_baseline)
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


def orientation_index(angles, num_angles):
    '''Index k of each angle on stimuli_whiten's orientation grid, theta_inputs[k] = k * pi / num_angles.'''
    return np.round(np.asarray(angles) * num_angles / np.pi).astype(int) % num_angles


def tuning_curves(y_hist, centers, num_angles, window=None):
    '''
    In-stream tuning curves: for each presentation (STIMULUS_STEPS block) whose last timestep lies
    in window = [t_start, t_stop) (default: the run's 2nd half), the response at that last
    timestep, averaged over all presentations of the same orientation.

    Returns (rate, y, counts): mean firing rate (rectified y squared) and mean rectified y, each
    (N_TOT, num_angles), and the number of presentations averaged per orientation.
    '''
    n_steps = y_hist.shape[1]
    t_start, t_stop = window if window is not None else (n_steps // 2, n_steps)
    last = np.arange(STIMULUS_STEPS - 1, n_steps, STIMULUS_STEPS)
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


def crf_response_scale(dyn, layout, reference='adapt CRF only'):
    '''
    kappa = s^2: the cRF's mean squared response to a uniform ensemble routed to layout, relative
    to the same ensemble routed to reference. Uses the closed-form fixed point
    y = z / sqrt(sigma^2 + N z^2) at g = 0, over one presentation of each orientation.

    The all-ones normalization pool divides every neuron by the whole population's drive, so a
    full-field grating suppresses the cRF by s ~ 0.56 relative to a cRF-only grating
    (kappa ~ 0.31 at contrast 0.8).
    The target covariance at TARGET_COV_PATH is single-RF (no surround suppression), so theta_t is
    rescaled by kappa to sit at the cRF's actual response scale. tau_g is not rescaled: dg/dt
    scales with the squared response, so under full-field stimulation the gains approach this set
    point about 1/kappa times more slowly than cRF-only gains.
    '''
    def mean_power(loc):
        gen = StimulusGenerator(N_RF=N_RF, N_SETS=N_SETS, num_angles=N_RF, stream_length=N_RF,
                                tuning_width=TUNING_WIDTH, contrast=ENSEMBLE_CONTRAST)
        z = gen.generate_surround_ensembles(loc, duration=1)   # each orientation once
        y = z / np.sqrt(dyn.sigma ** 2 + dyn.N_matrix @ z ** 2)
        return np.mean(np.sum(y[CRF_SLICE] ** 2, axis=0))
    return mean_power(layout) / mean_power(reference)


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


def first_order_equalization(y_hist, window):
    '''||r_hat - 1||_2 with r_hat = r / mean(r), where r is each cRF neuron's mean firing rate
    (rectified y squared) over every timestep of y_hist in window = [t_start, t_stop).
    0 = fully equalized.'''
    t_start, t_stop = window
    r = (y_hist[CRF_SLICE, t_start:t_stop] ** 2).mean(axis=1)
    return np.linalg.norm(r / r.mean() - 1)


def response_covariance(y_hist, window):
    '''cRF covariance of the rectified y actually produced during a run (y_hist: (N_TOT, n_steps)),
    over every timestep in window = [t_start, t_stop). ddof=0.'''
    t_start, t_stop = window
    return np.cov(y_hist[CRF_SLICE, t_start:t_stop], bias=True)


def second_order_equalization(y_hist, window, target_cov):
    '''Operator distance: largest eigenvalue of Cov_responses - Cov_target (as in
    Analytic_responses.compute_error's 'operator'), Cov_responses over window of y_hist.'''
    return np.linalg.eigvalsh(response_covariance(y_hist, window) - target_cov)[-1]


def gaussian_tuning(theta_deg, amp, mu, width, base):
    '''Gaussian on the orientation circle: base + amp * exp(-d^2 / (2 width^2)), with d the
    distance from theta to the preferred orientation mu wrapped to +-90 deg.'''
    d = (theta_deg - mu + 90) % 180 - 90
    return base + amp * np.exp(-0.5 * (d / width) ** 2)


def fit_gaussian_tuning(curve, angles_deg):
    '''Least-squares fit of gaussian_tuning to a tuning curve sampled at angles_deg over [0, 180).
    Returns (amp, mu, width, base), mu wrapped to [0, 180).'''
    i = int(np.argmax(curve))
    p0 = [curve[i] - curve.min(), angles_deg[i], 30.0, curve.min()]
    bounds = ([0.0, -np.inf, 1.0, -np.inf], [np.inf, np.inf, 180.0, np.inf])
    params, _ = curve_fit(gaussian_tuning, angles_deg, curve, p0=p0, bounds=bounds, maxfev=10000)
    params[1] %= 180
    return params


if __name__ == "__main__":

    print("Initializing tunings, frame, and dynamics...")
    tunings = V1Tunings(N=N_RF)
    frame   = Frame(csv_path=FRAME_PATH)
    dyn     = V1Dynamics_Surround(tunings, frame, dt=DT, N_RF=N_RF, N_SETS=N_SETS,
                                   target_covariance_path=TARGET_COV_PATH, gains_nonneg=True)

    stim_gen = StimulusGenerator(N_RF=N_RF, N_SETS=N_SETS, num_angles=N_RF,
                                  stream_length=ADAPTATION_STEPS,
                                  tuning_width=TUNING_WIDTH, contrast=ENSEMBLE_CONTRAST)
    n_angles = stim_gen.num_angles
    adaptor_idx = n_angles // 2                      # matches generate_surround_ensembles' own adaptor
    adaptor_deg = np.degrees(stim_gen.theta_inputs[adaptor_idx])
    stim_deg = np.degrees(stim_gen.theta_inputs)     # orientations the tuning curves are sampled at
    K = dyn.frame.K

    print("Calibrating theta_t from the uniform target covariance...")
    # Target rescaled to the cRF's response scale under the uniform ensemble's routing (see
    # crf_response_scale); also the target of the 2nd-order equalization below.
    kappa = crf_response_scale(dyn, ADAPT_LOCATION_FOR_COND[UNIFORM_COND])
    target_cov = dyn.uniform_target_covariance #* kappa
    print(f"  kappa = {kappa:.4f} (cRF response power, "
          f"'{ADAPT_LOCATION_FOR_COND[UNIFORM_COND]}' vs 'adapt CRF only')")
    dyn.calibrate_theta_t(C_zz_uniform=target_cov, uniform_target=True)

    print("Settling the surround's non-adapting normalization signal (no delay)...")
    flat = stim_gen.oriented_drive(0.0, 0.0, 'no adaptation')   # blank screen: the fixed baseline everywhere
    surround_msg_baseline = settled_surround_msg(dyn, flat, BASELINE_SETTLE_STEPS)
    dyn.set_surround_delay(SURROUND_DELAY)
    if dyn.n_lag >= ADAPTATION_STEPS:
        print(f"  NOTE: surround_delay ({SURROUND_DELAY:g}ms) >= run length ({ADAPTATION_DURATION:g}ms): "
              f"the cRF only ever sees this baseline surround signal.")

    print(f"Running adaptation phase (surround_delay={SURROUND_DELAY:g}ms)...")
    final_gains = {}
    for cond in (UNIFORM_COND, BIASED_COND):
        print(f"  Adapting: {CONDITION_LABEL[cond]}")
        final_gains[cond] = run_adaptation_phase(dyn, stim_gen, cond, surround_msg_baseline=surround_msg_baseline)

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
    # the tuning-repulsion panel of the time-course figure loops over them. To add
    # time points, append snapshots of other windows of the biased run, e.g.
    # snapshot(t, biased['y_hist'], biased['centers'], window=(t - 2000, t), label=...).
    # Every cRF tuning curve (firing rate) is fit with gaussian_tuning; the tuning shifts and the
    # tuning-curve figure use the fits.
    # ==========================================================================
    print("Recording in-stream tuning curves...")

    def snapshot(t, y_hist, centers, window=None, label=''):
        rate, y, counts = tuning_curves(y_hist, centers, n_angles, window)
        print(f"  {label} (t={t}): {counts.sum()} presentations, {counts.min()}-{counts.max()} per orientation")
        fits = np.array([fit_gaussian_tuning(tc, stim_deg) for tc in rate[CRF_SLICE]])   # (N_RF, 4)
        return dict(t=t, rate=rate, y=y, fits=fits)

    uniform = SIM_HISTORY[UNIFORM_COND]
    biased  = SIM_HISTORY[BIASED_COND]
    n_steps = biased['y_hist'].shape[1]
    uniform_ref = snapshot(n_steps, uniform['y_hist'], uniform['centers'], label='uniform-adapted')
    snapshots = [
        snapshot(n_steps, biased['y_hist'], biased['centers'], label='adapted'),  # 2nd-half average
    ]

    # ==========================================================================
    # Ensemble histograms and biased-adapted cRF tuning curves (Gaussian fits), normalized to the
    # uniform-adapted fits' peaks. All orientation axes are relative to the adaptor.
    # ==========================================================================
    print("Plotting ensemble histograms and tuning curves...")
    N_BINS = N_RF
    uni_angles_deg  = np.degrees(uniform['centers'])   # stimulus centers actually shown during each run
    bias_angles_deg = np.degrees(biased['centers'])

    # Fits evaluated on a fine grid of orientations relative to the adaptor
    x_axis = np.linspace(-90, 90, 361)
    def fitted_curves(fits):
        return np.array([gaussian_tuning(x_axis + adaptor_deg, *p) for p in fits])   # (N_RF, len(x_axis))

    binned_uniform = bin_by_preference(fitted_curves(uniform_ref['fits']), tunings.theta)
    binned_crf     = bin_by_preference(fitted_curves(snapshots[-1]['fits']), tunings.theta)

    # Each neuron's biased-adapted curve as a fraction of ITS OWN uniform-adapted peak.
    peak_uniform = np.max(binned_uniform, axis=1, keepdims=True)
    norm_crf     = binned_crf / (peak_uniform + 1e-12)

    # Offline tuning curves: closed-form fixed point (get_response_offline) with each run's final
    # gains frozen, probed at every orientation of x_axis (0.5 deg spacing). Probes are routed and
    # scaled like that condition's ensemble, as the in-stream presentations were.
    def offline_tuning(cond):
        g_cRF, g_surround = final_gains[cond][:2]
        probes = np.stack([stim_gen.oriented_drive(theta, ENSEMBLE_CONTRAST, ADAPT_LOCATION_FOR_COND[cond])
                           for theta in np.radians(x_axis + adaptor_deg)], axis=1)   # (N_TOT, len(x_axis))
        M = offline_gain_operator(dyn, g_cRF, g_surround)
        return get_response_offline(dyn, probes, M)[CRF_SLICE]   # firing rates, (N_RF, len(x_axis))

    binned_offline_uniform = bin_by_preference(offline_tuning(UNIFORM_COND), tunings.theta)
    binned_offline_crf     = bin_by_preference(offline_tuning(BIASED_COND), tunings.theta)
    norm_offline_crf = binned_offline_crf / (np.max(binned_offline_uniform, axis=1, keepdims=True) + 1e-12)

    # num_angles == N_RF is odd and the adaptor is one of the presented angles, so the relative
    # angles land exactly on the centers of N_BINS equal bins spanning [-90, 90].
    uni_angles_rel  = (uni_angles_deg  - adaptor_deg + 90) % 180 - 90
    bias_angles_rel = (bias_angles_deg - adaptor_deg + 90) % 180 - 90
    bins_hist = np.linspace(-90, 90, N_BINS + 1)
    weights_uni  = np.ones_like(uni_angles_rel)  / len(uni_angles_rel)
    weights_bias = np.ones_like(bias_angles_rel) / len(bias_angles_rel)

    blue_colors = plt.cm.Blues(np.linspace(0.2, 1.0, N_BINS))

    fig_tc, axes_tc = plt.subplots(1, 4, figsize=(17.3, 3.75))
    ax_uni_hist, ax_bias_hist, ax_tc_bias, ax_tc_offline = axes_tc

    ax_uni_hist.hist(uni_angles_rel, bins=bins_hist, weights=weights_uni, color='black', rwidth=0.9)
    ax_bias_hist.hist(bias_angles_rel, bins=bins_hist, weights=weights_bias, color='black', rwidth=0.9)
    # Shared y-scale so the two ensembles' probabilities are directly comparable
    hist_ymax = max(ax_uni_hist.get_ylim()[1], ax_bias_hist.get_ylim()[1])
    for ax in (ax_uni_hist, ax_bias_hist):
        ax.set_ylim(0, hist_ymax)
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)

    for i in range(N_BINS):
        ax_tc_bias.plot(x_axis, norm_crf[i], color=blue_colors[i], linewidth=2.0)
        ax_tc_offline.plot(x_axis, norm_offline_crf[i], color=blue_colors[i], linewidth=2.0)
    ax_tc_bias.set_title("In-stream (fit)", fontsize=16)
    ax_tc_offline.set_title("Offline (frozen gains)", fontsize=16)
    # Shared y-scale so the in-stream and offline curves are directly comparable
    ymin = min(ax.get_ylim()[0] for ax in (ax_tc_bias, ax_tc_offline))
    ymax = max(ax.get_ylim()[1] for ax in (ax_tc_bias, ax_tc_offline))
    for ax in (ax_tc_bias, ax_tc_offline):
        ax.set_ylim(ymin - 0.05 * (ymax - ymin), ymax)
        ax.set_yticks([0, 1])

    for ax in axes_tc:
        ax.set_xlim(-90, 90)
        ax.set_xticks([-90, 0, 90])
        ax.set_xlabel("Stimulus (°)", fontsize=18)
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', labelsize=16)

    for ax in (ax_tc_bias, ax_tc_offline):
        ax.spines['left'].set_linewidth(1.8)
        ax.spines['bottom'].set_linewidth(1.8)
        ax.tick_params(axis='both', width=1.8, length=5)
    fig_tc.tight_layout()

    # ==========================================================================
    # Biased adaptation over time: input covariance | tuning repulsion | 1st- and 2nd-order
    # equalization. The input covariance uses the biased ensemble's minimal stimulus set; the
    # 1st- and 2nd-order equalization use the responses actually produced in consecutive
    # STATS_WINDOW-ms windows of the biased run.
    # ==========================================================================
    print("Plotting biased-adaptation time course...")
    ens_stimuli, _ = minimal_ensemble(BIASED_COND, n_angles)
    input_cov = np.cov(ens_stimuli[CRF_SLICE], bias=True)   # exact ensemble covariance (ddof=0)

    # Tuning repulsion: shift of each neuron's fitted preference from its uniform-adapted fitted
    # preference, wrapped to +-90 deg.
    pref_uniform_deg = uniform_ref['fits'][:, 1]
    pref_uniform_rel = (pref_uniform_deg - adaptor_deg + 90) % 180 - 90
    sort_shift = np.argsort(pref_uniform_rel)   # connect neighbours in preference order

    def tuning_shift_deg(snap):
        return (snap['fits'][:, 1] - pref_uniform_deg + 90) % 180 - 90

    # One point per window [t - STATS_WINDOW_STEPS, t) of the biased run, plotted at t (timesteps).
    windows = [(t0, t0 + STATS_WINDOW_STEPS) for t0 in range(0, n_steps - STATS_WINDOW_STEPS + 1, STATS_WINDOW_STEPS)]
    times = [t_stop for _, t_stop in windows]
    eq1 = [first_order_equalization(biased['y_hist'], w) for w in windows]
    eq2 = [second_order_equalization(biased['y_hist'], w, target_cov) for w in windows]
    for t, e1, e2 in zip(times, eq1, eq2):
        print(f"  t={t}: 1st-order equalization = {e1:.4f}, 2nd-order (operator distance) = {e2:.4g}")

    fig_eq, (ax_cov_in, ax_shift, ax_eq1, ax_eq2) = plt.subplots(1, 4, figsize=(17.5, 3.75))

    ax_cov_in.imshow(input_cov, cmap='viridis')
    ax_cov_in.set_xticks([])
    ax_cov_in.set_yticks([])
    ax_cov_in.set_xlabel("Neurons", fontsize=18)
    ax_cov_in.set_ylabel("Neurons", fontsize=18)
    ax_cov_in.set_title("Input covariance", fontsize=16)

    # Latest snapshot in navy; earlier ones (when added) in lighter blues.
    shift_colors = list(plt.cm.Blues(np.linspace(0.35, 0.75, len(snapshots) - 1))) + ['navy']
    for snap, color in zip(snapshots, shift_colors):
        shift = tuning_shift_deg(snap)
        ax_shift.plot(pref_uniform_rel[sort_shift], shift[sort_shift],
                      color='lightblue', linestyle=':', linewidth=2.0, zorder=2)
        ax_shift.scatter(pref_uniform_rel, shift, color=color, s=50, zorder=3, label=f"t = {snap['t']}")
    if len(snapshots) > 1:
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
        ax.set_xticks(times, labels=[f"{t // 1000}k" for t in times])
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
