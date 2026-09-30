"""
Dario_plots.py

Replicates mouse V1 adaptation experiments using adaptive ORGaNICs.

Figure 1: Adaptation to three orientation environments A, B, C - von Mises distributions with
peaks evenly spaced in orientation and different widths (see ENVIRONMENTS). Panels: the stimulus
orientation distributions and the power-law test of Tring, Dipoppa &
Ringach (2023, Nat. Commun., "A power law describes the magnitude of adaptation in neural
populations of primary visual cortex") - log ratio of population response magnitudes between
two environments vs. log ratio of the stimulus probabilities.

Figure 3: Contrast response functions after adapting to low/medium/high contrast ensembles.

Both figures run on V1Dynamics_Surround (the unified normalization+whitening+surround engine):
adaptation and probing use the full population (cRF + surround, 'adapt CRF and surround'), and
analysis is restricted to the cRF block since all other blocks are statistically redundant
copies under that adapt_location.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea, HPacker, TextArea, VPacker
from matplotlib.ticker import MultipleLocator
from tqdm import tqdm
from scipy.linalg import block_diag
from scipy.special import i0
from tunings_whiten import V1Tunings
from stimuli_whiten import StimulusGenerator
from simulation_whiten import Frame, V1Dynamics_Surround
from Surround_simulated_responses import get_response_offline, probe_input_drive

# ---- Parameters ----
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_RF = 13                      # Number of primary neurons per receptive field
N_SETS = 6                     # 1 classical RF (cRF) + 5 surround sets; must match Surround_simulated_responses.N_SETS,
                               # which sizes both get_response_offline's (I+M) and probe_input_drive
FRAME_PATH = os.path.join(REPO_ROOT, "data/frames/N13_mercedes_K182_Frame.csv")
TARGET_COV_PATH = os.path.join(REPO_ROOT, "data/target_covs/uniform_target_covariance_mid_c.csv")
ADAPT_LOC = 'adapt CRF only'   # no spatial cRF/surround distinction in these experiments
ADAPT_CONTRAST = 0.8
PROBE_CONTRAST = 0.6
# Figure 1 orientation environments, all von Mises with peaks evenly spaced 60° apart:
# name -> (peak orientation in degrees, concentration kappa)
ENVIRONMENTS = {'A': (30.0, 6.0),    # tall and skinny
                'B': (90.0, 3.0),    # in between
                'C': (150.0, 1.5)}   # short and wide

# tau_g=750 (V1Dynamics_Surround) needs several tau_g of adaptation time to converge; at dt=0.1
# this gives ~4 tau_g (~98% settled) for every adaptation/probe stream below.
STREAM_LENGTH = 100000
PROBE_RES = 13                 # probe orientations; keep a multiple of N_RF so probes land on neuron preferences
RATE_FLOOR = 1e-3              # responses below this (log10 = -3) are treated as zero (excluded from the log moments)
RESTRICTED_FIT_RANGE = 1.0     # Figure 1's dotted power-law fit uses only points with |log10 P ratio| <= this


def offline_gain_operator(dyn, state):
    '''(N_TOTAL, N_TOTAL) block-diagonal M = W diag(g) W.T feeding get_response_offline's
    (I+M)^-1 fixed point, built from the frozen g_cRF/g_surround of a full
    V1Dynamics_Surround state - same cRF/surround block layout as frozen_derivatives'
    full_gain_feedback: the cRF block uses g_cRF, every surround block reuses g_surround.'''
    unpacked = dyn.unpack_state(state)
    W = dyn.frame.W
    M_cRF = W @ np.diag(unpacked['g_cRF']) @ W.T
    M_surround = W @ np.diag(unpacked['g_surround']) @ W.T
    return block_diag(M_cRF, *([M_surround] * (N_SETS - 1)))


def probe_responses(dyn, M, probe_angles, contrast):
    """
    cRF population response vectors (N_RF, n_probes) to Gaussian probes (probe_input_drive) at
    each of probe_angles, with gains frozen in M (from offline_gain_operator), settled via
    get_response_offline.
    """
    resp = np.zeros((N_RF, len(probe_angles)))
    for i, theta in enumerate(probe_angles):
        resp[:, i] = get_response_offline(dyn, probe_input_drive(theta, contrast), M)[:N_RF]
    return resp


def von_mises_orientation_density(theta, center, kappa):
    '''Probability density over orientation [0, pi) of stimuli_whiten's von Mises environments:
    np.random.vonmises(center, kappa) is drawn on the full circle and then folded mod pi, so
    p(theta) = f(theta) + f(theta + pi) = cosh(kappa cos(theta - center)) / (pi I0(kappa)).'''
    return np.cosh(kappa * np.cos(theta - center)) / (np.pi * i0(kappa))


def fit_power_law(log_p_ratio, log_r_ratio):
    '''Least-squares line through the origin in log-log coordinates, i.e. the power law
    r_X/r_Y = (p_X/p_Y)^beta of Tring et al. (fit without intercept). Returns (beta, R^2).'''
    beta = np.sum(log_p_ratio * log_r_ratio) / np.sum(log_p_ratio ** 2)
    residual = log_r_ratio - beta * log_p_ratio
    r_squared = 1 - np.sum(residual ** 2) / np.sum((log_r_ratio - log_r_ratio.mean()) ** 2)
    return beta, r_squared


def add_pair_legend(ax, pairs, pair_styles, env_colors, alpha=1.0, markersize=8, loc='lower left', fontsize=12):
    '''Legend for the X / Y environment pairs (pair_styles: (X, Y) -> (marker color, marker shape))
    with each environment letter in its own color. A matplotlib legend label is a single-colored
    Text, so each row is assembled from offsetbox pieces instead: marker, X, " / ", Y.'''
    def letter(env):
        return TextArea(env, textprops=dict(color=env_colors[env], fontsize=fontsize, fontweight='bold'))

    rows = []
    for X, Y in pairs:
        color, shape = pair_styles[(X, Y)]
        width, height = markersize + 10, markersize + 4
        marker = DrawingArea(width, height)
        marker.add_artist(Line2D([width / 2], [height / 2], marker=shape, color=color, alpha=alpha,
                                 markersize=markersize, linestyle='none'))
        label = HPacker(children=[letter(X), TextArea(' / ', textprops=dict(fontsize=fontsize)), letter(Y)],
                        align='baseline', pad=0, sep=0)
        rows.append(HPacker(children=[marker, label], align='center', pad=0, sep=6))
    title = TextArea('X / Y', textprops=dict(fontsize=fontsize))
    box = VPacker(children=[title, VPacker(children=rows, align='left', pad=0, sep=4)],
                  align='center', pad=0, sep=6)
    legend = AnchoredOffsetbox(loc=loc, child=box, pad=0.5, borderpad=0.6, frameon=True)
    legend.patch.set_boxstyle('round,pad=0,rounding_size=0.2')
    legend.patch.set_edgecolor('0.8')
    ax.add_artist(legend)


def calc_moments(responses):
    '''Calculates base-10 log mean and log variance of the data for comparison with Dario's results'''
    N = responses.shape[0]

    # Create a copy as floats to insert NaNs where responses are below RATE_FLOOR (incl. exact
    # zeros), so near-zero rates don't dominate the log moments
    r_masked = np.array(responses, dtype=float)
    r_masked[r_masked < RATE_FLOOR] = np.nan

    # Calculate base-10 log responses for entries above RATE_FLOOR
    log_r = np.log10(r_masked)

    mu = np.nanmean(log_r, axis=0)
    variance = np.nanvar(log_r, axis=0)

    return mu, variance


def Dario_fig1(dyn, stim_gen):
    """Figure 1: adaptation to three von Mises orientation environments A, B, C (ENVIRONMENTS:
    same shape family, different peak orientations and widths). Left: stimulus orientation
    distributions. Right: power-law test of Tring, Dipoppa & Ringach (2023) - log10 ratio of
    population response magnitudes (l2 norm of the response vector) between two environments X, Y
    vs. log10 ratio of the stimulus probabilities, fit by a line without intercept
    (r_X/r_Y = (p_X/p_Y)^beta, pooled over pairs) - over all points (solid) and over only points
    with |log10 P ratio| <= RESTRICTED_FIT_RANGE (dotted). Only the cRF block is analyzed."""
    print(' ----------- FIGURE 1 -----------')
    # Probe exactly at the neuron preferences (np.linspace(0, pi, N_RF, endpoint=False), as in
    # probe_input_drive/stimuli_whiten); 180° is excluded since it's the same orientation as 0°
    probe_angles = np.linspace(0, np.pi, PROBE_RES, endpoint=False)

    print("\n--- Running Adaptation Stage ---")
    final_states = {}
    for env, (center_deg, kappa) in ENVIRONMENTS.items():
        print(f"Adapting to environment {env} (von Mises, peak {center_deg:g}°, kappa {kappa:g})...")
        dyn.run_simulation(stim_gen.generate_surround_ensembles(ADAPT_LOC, von_mises=True, von_mises_center=center_deg,
                                                                von_mises_kappa=kappa))
        final_states[env] = dyn.last_state

    # --- Probe Stage: gains frozen at each adapted state, probed with the offline fixed point
    # (cRF block only) ---
    print("\n--- Running Probe Stage ---")
    resp = {env: probe_responses(dyn, offline_gain_operator(dyn, state), probe_angles, PROBE_CONTRAST)
            for env, state in final_states.items()}

    # Response magnitude = l2 norm of each probe's population response vector (as in Tring et al.)
    magnitude = {env: np.linalg.norm(r, axis=0) for env, r in resp.items()}

    # Probabilities are the densities the adaptation streams were actually drawn from
    prob = {env: von_mises_orientation_density(probe_angles, np.deg2rad(center_deg), kappa)
            for env, (center_deg, kappa) in ENVIRONMENTS.items()}
    # (X, Y); swapping X and Y mirrors a pair through the origin
    pairs = [('A', 'B'), ('A', 'C'), ('B', 'C')]
    log_p_ratio = {(X, Y): np.log10(prob[X] / prob[Y]) for X, Y in pairs}
    log_r_ratio = {(X, Y): np.log10(magnitude[X] / magnitude[Y]) for X, Y in pairs}

    # Power law r_X/r_Y = (p_X/p_Y)^beta, pooled over pairs: once over all points, once over only
    # the points within RESTRICTED_FIT_RANGE on the probability-ratio axis
    x_all = np.concatenate([log_p_ratio[pair] for pair in pairs])
    y_all = np.concatenate([log_r_ratio[pair] for pair in pairs])
    beta, r_squared = fit_power_law(x_all, y_all)
    in_range = np.abs(x_all) <= RESTRICTED_FIT_RANGE
    beta_in, r_squared_in = fit_power_law(x_all[in_range], y_all[in_range])
    print(f"Power law (no intercept, pooled over {len(pairs)} pairs): beta = {beta:.3f}, R^2 = {r_squared:.3f}")
    print(f"  |log10 P ratio| <= {RESTRICTED_FIT_RANGE:g} only ({in_range.sum()}/{in_range.size} points): "
          f"beta = {beta_in:.3f}, R^2 = {r_squared_in:.3f}")

    # --- Figure ---
    fig, (ax_dist, ax_pow) = plt.subplots(1, 2, figsize=(13, 5.5))
    colors = {'A': '#4169E1', 'B': '#D2042D', 'C': '#B8860B'}   # royal blue, cherry red, dark yellow
    pair_styles = {('A', 'B'): ('#6A3D9A', 'o'), ('A', 'C'): ('#0F6B2F', 's'),   # purple circles, green squares,
                   ('B', 'C'): ('#E07000', '^')}                                  # orange triangles
    point_alpha = 1.0
    point_size = 110   # scatter marker area (pt^2)
    lw = 3
    fs_label = 24
    fs_ticks = 16
    axis_lw = 1.5
    for ax in (ax_dist, ax_pow):
        for spine in ax.spines.values():
            spine.set_linewidth(axis_lw)

    # Left: stimulus orientation distributions, each labeled with its letter right above its peak
    ax = ax_dist
    theta_fine = np.linspace(0, np.pi, 721)
    per_degree = np.pi / 180
    for env, (center_deg, kappa) in ENVIRONMENTS.items():
        center = np.deg2rad(center_deg)
        ax.plot(np.degrees(theta_fine), von_mises_orientation_density(theta_fine, center, kappa) * per_degree,
                color=colors[env], lw=lw)
        peak = von_mises_orientation_density(center, center, kappa) * per_degree
        ax.annotate(env, (center_deg, peak), xytext=(0, 6), textcoords='offset points', ha='center', va='bottom',
                    color=colors[env], fontsize=20, fontweight='bold')
    ax.set_xlabel('Stimulus orientation (°)', fontsize=fs_label)
    ax.set_ylabel('Probability', fontsize=fs_label)
    ax.set_xlim(0, 180)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)   # headroom for the peak letters
    ax.set_xticks([0, 90, 180])
    ax.set_yticks([])
    ax.tick_params(axis='x', length=0, labelsize=fs_ticks)

    # Right: log response-magnitude ratio vs log probability ratio, one marker per probe orientation
    ax = ax_pow
    ax.axhline(0, color='lightgray', lw=1, zorder=0)
    ax.axvline(0, color='lightgray', lw=1, zorder=0)
    for X, Y in pairs:
        color, marker = pair_styles[(X, Y)]
        ax.scatter(log_p_ratio[(X, Y)], log_r_ratio[(X, Y)], color=color, alpha=point_alpha, marker=marker,
                   s=point_size, edgecolors='white', linewidths=1, zorder=3)
    x_fit = np.array([x_all.min(), x_all.max()])
    fit_all_line, = ax.plot(x_fit, beta * x_fit, color='#333333', lw=2, zorder=2,
                            label=fr'$\beta = {beta:.2f}$, $R^2 = {r_squared:.2f}$')
    # Restricted fit, drawn only over the range it was fit on
    x_fit_in = np.clip(x_fit, -RESTRICTED_FIT_RANGE, RESTRICTED_FIT_RANGE)
    fit_in_line, = ax.plot(x_fit_in, beta_in * x_fit_in, color='#333333', lw=2.5, ls=':', zorder=2,
                           label=fr'$\beta = {beta_in:.2f}$, $R^2 = {r_squared_in:.2f}$')
    ax.set_xlabel(r'$\log_{10}\,[P_X(\theta)\,/\,P_Y(\theta)]$', fontsize=fs_label)
    ax.set_ylabel(r'$\log_{10}\,[r_X(\theta)\,/\,r_Y(\theta)]$', fontsize=fs_label)
    ax.xaxis.set_major_locator(MultipleLocator(1))   # a tick label at every integer
    # Three y ticks symmetric about 0: the largest one-significant-figure value inside both y limits
    y_extent = min(abs(lim) for lim in ax.get_ylim())
    decade = 10 ** np.floor(np.log10(y_extent))
    y_tick = np.floor(y_extent / decade + 1e-9) * decade
    ax.set_yticks([-y_tick, 0, y_tick])
    ax.set_yticklabels([f'−{y_tick:g}', '0', f'{y_tick:g}'])
    ax.tick_params(labelsize=fs_ticks, width=axis_lw)
    add_pair_legend(ax, pairs, pair_styles, colors, alpha=point_alpha, markersize=np.sqrt(point_size),
                    loc='lower left')
    ax.legend(handles=[fit_all_line, fit_in_line], loc='upper right', fontsize=14)

    plt.tight_layout()
    plt.show()


def Dario_fig3(dyn, stim_gen):
    '''Contrast response functions after adapting to low/medium/high contrast ensembles
    (orientation content unbiased throughout - only contrast is ensemble-biased).'''
    print(' ----------- FIGURE 3 -----------')

    print("\n--- Running Adaptation Stage ---")
    print("Adapting to high contrast stream...")
    dyn.run_simulation(stim_gen.generate_contrast_stream(peak_ln_contrast=0, adapt_location=ADAPT_LOC))
    state_hi = dyn.last_state

    print("Adapting to medium contrast stream...")
    dyn.run_simulation(stim_gen.generate_contrast_stream(peak_ln_contrast=-1.5, adapt_location=ADAPT_LOC))
    state_med = dyn.last_state

    print("Adapting to low contrast stream...")
    dyn.run_simulation(stim_gen.generate_contrast_stream(peak_ln_contrast=-3, adapt_location=ADAPT_LOC))
    state_lo = dyn.last_state

    probe_contrasts   = np.logspace(np.log10(0.04), np.log10(1.0), 20)
    probe_angles_fig3 = np.linspace(0, np.pi, PROBE_RES, endpoint=False)   # neuron preferences (see Dario_fig1)

    conditions = [
        ('Low',    'green', state_lo),
        ('Medium', 'red',   state_med),
        ('High',   'black', state_hi),
    ]

    mu_curves  = {}
    var_curves = {}

    for label, color, state in conditions:
        M = offline_gain_operator(dyn, state)
        print(f"  Sweeping contrasts for {label} adapted state...")
        mus, vars_ = [], []
        for c in tqdm(probe_contrasts, desc=f"{label} contrast sweep", leave=True):
            resp = np.zeros((N_RF, PROBE_RES))
            for i, angle in enumerate(probe_angles_fig3):
                y = get_response_offline(dyn, probe_input_drive(angle, c), M)
                resp[:, i] = y[:N_RF]
            _, mu_c, var_c = calc_moments(resp)
            mus.append(np.nanmean(mu_c))
            vars_.append(np.nanmean(var_c))
        mu_curves[label]  = np.array(mus)
        var_curves[label] = np.array(vars_)

    fig3, (ax_mu, ax_var) = plt.subplots(1, 2, figsize=(12, 5))
    log_c = np.log10(probe_contrasts)
    fs3  = 18

    for label, color, *_ in conditions:
        ax_mu.plot(log_c, mu_curves[label],  color=color, lw=2, label=label)
        ax_var.plot(log_c, var_curves[label], color=color, lw=2, label=label)

    delta_mu  = np.nanmean(mu_curves['High'])  - np.nanmean(mu_curves['Low'])
    delta_var = np.nanmean(var_curves['High']) - np.nanmean(var_curves['Low'])

    for ax, ylabel, delta in [(ax_mu, r'$\mu$', delta_mu), (ax_var, r'$\sigma^2$', delta_var)]:
        ax.set_xlabel(r'$\log_{10}(\mathrm{contrast})$', fontsize=fs3, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=fs3 + 6, fontweight='bold')
        ax.legend(fontsize=12, loc='upper left')
        ax.text(0.97, 0.97, fr'$\Delta={delta:+.3f}$', transform=ax.transAxes,
                fontsize=13, ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray'))
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['left', 'bottom']].set_color('gray')
        ax.tick_params(colors='gray')

    plt.suptitle('Contrast Response After Adaptation', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    print("Initializing tunings, frame, stimulus generator, and dynamics...")
    tunings = V1Tunings(N=N_RF)
    frame = Frame(csv_path=FRAME_PATH)
    stim_gen = StimulusGenerator(N_RF=N_RF, N_SETS=N_SETS, num_angles=N_RF, stream_length=STREAM_LENGTH, contrast=ADAPT_CONTRAST)
    dyn = V1Dynamics_Surround(tunings, frame, N_RF=N_RF, N_SETS=N_SETS,
                               target_covariance_path=TARGET_COV_PATH, gains_nonneg=True)

    # Calibrate theta_t once, from a genuinely unbiased stream at a fixed low contrast (kept
    # separate from whatever contrast either figure below actually probes at - see
    # Surround_simulated_responses.run_adaptation_phase's 'no adaptation' condition, same idea).
    # Shared by both figures, per the model's "fixed developmental prior" design
    # (docs/whitening_adaptation_notes.md).
    print("Calibrating theta_t...")
    true_contrast = ADAPT_CONTRAST
    dyn.calibrate_theta_t(C_zz_uniform=dyn.uniform_target_covariance, uniform_target=True)

    Dario_fig1(dyn, stim_gen)
    #Dario_fig3(dyn, stim_gen)
