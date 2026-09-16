"""
Dario_plots.py

Replicates mouse V1 adaptation experiments using adaptive ORGaNICs.

Figure 1: Analysis of post-adaptation log-normal components. Stimuli belong to one of three
distributions: (A) Von Mises Centered at 0 degrees (B) Von Mises Centered at 90 degrees or
(C) Uniform across orientations. Recreates plots from Figure 5 of Dario's "Contrast and
Pattern Adaptation..."

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
from tqdm import tqdm
from tunings_whiten import V1Tunings
from stimuli_whiten import StimulusGenerator
from simulation_whiten import Frame, V1Dynamics_Surround
from Surround_simulated_responses import get_response, probe_input_drive

# ---- Parameters ----
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_RF = 13                      # Number of primary neurons per receptive field
N_SETS = 5                     # 1 classical RF (cRF) + 4 surround sets (V1Dynamics_Surround requires >=2)
FRAME_PATH = os.path.join(REPO_ROOT, "data/frames/N13_mercedes_Frame.csv")
TARGET_COV_PATH = os.path.join(REPO_ROOT, "data/target_covs/uniform_target_covariance.csv")
ADAPT_LOC = 'adapt CRF and surround'   # no spatial cRF/surround distinction in these experiments

# tau_g=750 (V1Dynamics_Surround) needs several tau_g of adaptation time to converge; at dt=0.1
# this gives ~4 tau_g (~98% settled) for every adaptation/probe stream below.
STREAM_LENGTH = 30000
THETA_T_CONTRAST = 0.25        # low, fixed contrast used only to calibrate theta_t (see __main__)
PROBE_RES = 20


def probe_ensemble_moments(dyn, y_hist, stim_angles, probe_angle_bins):
    """
    Compute log-normal moments at each probe orientation by averaging over the
    statistical ensemble. y_hist is (n_neurons, T) membrane potentials from the probe
    simulation; stim_angles is (T,) stimulus angle in radians per step.
    Returns P_0, mu, variance each of shape (n_probes,).
    Bins with no matching time steps return np.nan.
    """

    n_probes = len(probe_angle_bins)
    firing_rates = dyn.half_wave_rectify(y_hist, 2.0)   # (n_neurons, T)
    bin_width = np.pi / n_probes

    mu = np.full(n_probes, np.nan)
    variance = np.full(n_probes, np.nan)
    P_0 = np.full(n_probes, np.nan)

    for i, theta in enumerate(probe_angle_bins):
        d = (stim_angles - theta + np.pi / 2) % np.pi - np.pi / 2
        mask = np.abs(d) < bin_width / 2
        if not mask.any():
            continue
        rates = firing_rates[:, mask].flatten().astype(float)
        P_0[i] = np.mean(rates == 0)
        rates[rates == 0] = np.nan
        log_r = np.log(rates)
        mu[i] = np.nanmean(log_r)
        variance[i] = np.nanvar(log_r)

    return P_0, mu, variance


def calc_moments(responses):
    '''Calculates log mean and log variance of the data for comparison with Dario's results'''
    N = responses.shape[0]


    P_0 = np.sum(responses == 0, axis=0) / N

    # Create a copy as floats to insert NaNs where responses are 0
    r_masked = np.array(responses, dtype=float)
    r_masked[r_masked == 0] = np.nan

    # Calculate log responses for non-zero entries
    log_r = np.log(r_masked)

    mu = np.nanmean(log_r, axis=0)
    variance = np.nanvar(log_r, axis=0)

    return P_0, mu, variance


def Dario_fig1(dyn, stim_gen):
    '''Figure 1: log-normal response moments after adapting to Von Mises (0deg, 90deg) vs.
    uniform orientation ensembles. Only the cRF block is analyzed (other N_SETS blocks are
    redundant copies under ADAPT_LOC).'''
    print(' ----------- FIGURE 1 -----------')
    probe_angles = np.linspace(0, np.pi, PROBE_RES)
    probe_angles_deg = probe_angles * 180 / np.pi

    print("\n--- Running Adaptation Stage ---")
    print("Adapting to Ensemble A (Von Mises at 0 degrees)...")
    dyn.run_simulation(stim_gen.generate_surround_ensembles(ADAPT_LOC, von_mises=True, von_mises_center=0))
    final_state_VM_0 = dyn.last_state

    print("Adapting to Ensemble B (Von Mises at 90 degrees)...")
    dyn.run_simulation(stim_gen.generate_surround_ensembles(ADAPT_LOC, von_mises=True, von_mises_center=90))
    final_state_VM_90 = dyn.last_state

    print("Adapting to Ensemble C (Uniform)...")
    dyn.run_simulation(stim_gen.generate_surround_ensembles(ADAPT_LOC))
    final_state_uni = dyn.last_state

    # --- Probe Stage: full (still-plastic) dynamics from the exact final adaptation state,
    # continuing the SAME ensemble statistics, so gains stay near their adapted fixed point ---
    print("\n--- Running Probe Stage ---")
    print("Probing VM_0 context...")
    VM_0_probe_stream, VM_0_probe_angles = stim_gen.generate_surround_ensembles(
        ADAPT_LOC, von_mises=True, von_mises_center=0, return_angles=True)
    y_hist_probe_VM_0, *_ = dyn.run_simulation(VM_0_probe_stream, initial_state=final_state_VM_0)

    print("Probing VM_90 context...")
    VM_90_probe_stream, VM_90_probe_angles = stim_gen.generate_surround_ensembles(
        ADAPT_LOC, von_mises=True, von_mises_center=90, return_angles=True)
    y_hist_probe_VM_90, *_ = dyn.run_simulation(VM_90_probe_stream, initial_state=final_state_VM_90)

    print("Probing uniform context...")
    uni_probe_stream, uni_probe_angles = stim_gen.generate_surround_ensembles(ADAPT_LOC, return_angles=True)
    y_hist_probe_uni, *_ = dyn.run_simulation(uni_probe_stream, initial_state=final_state_uni)

    # --- Compute Moments over ensemble (cRF block only) ---
    P0_VM_0,  mu_VM_0,  var_VM_0  = probe_ensemble_moments(dyn, y_hist_probe_VM_0[:N_RF],  VM_0_probe_angles,  probe_angles)
    P0_VM_90, mu_VM_90, var_VM_90 = probe_ensemble_moments(dyn, y_hist_probe_VM_90[:N_RF], VM_90_probe_angles, probe_angles)
    P0_uni,   mu_uni,   var_uni   = probe_ensemble_moments(dyn, y_hist_probe_uni[:N_RF],   uni_probe_angles,   probe_angles)

    # Interpolate over angle bins that received no samples (NaN) so lines are continuous
    def fill_nans(arr):
        idx = np.arange(len(arr))
        finite = np.isfinite(arr)
        return np.interp(idx, idx[finite], arr[finite]) if not finite.all() else arr

    mu_VM_0  = fill_nans(mu_VM_0);  var_VM_0  = fill_nans(var_VM_0)
    mu_VM_90 = fill_nans(mu_VM_90); var_VM_90 = fill_nans(var_VM_90)
    mu_uni   = fill_nans(mu_uni);   var_uni   = fill_nans(var_uni)

    # --- Context ensemble densities P(θ) at probe orientations ---
    kappa = 4.0
    p_VM_0  = np.exp(kappa * np.cos(2 * (probe_angles - 0.0)))
    p_VM_0 /= np.trapz(p_VM_0, probe_angles)
    p_VM_90  = np.exp(kappa * np.cos(2 * (probe_angles - np.deg2rad(90))))
    p_VM_90 /= np.trapz(p_VM_90, probe_angles)
    p_uni = np.ones_like(probe_angles) / np.pi
    log_p_VM_0  = np.log(p_VM_0)
    log_p_VM_90 = np.log(p_VM_90)
    log_p_uni   = np.log(p_uni)

    # --- Figure ---
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    colors = {'VM_0': '#36454F', 'VM_90': '#228B22', 'uni': '#CC5500'}
    lw = 3
    labels = {'VM_0': 'Von Mises 0°', 'VM_90': 'Von Mises 90°', 'uni': 'Uniform'}
    fs_label = 14
    fs_ylabel = 26
    # Top-left: μ vs orientation
    ax = axes[0, 0]
    ax.plot(probe_angles_deg, mu_VM_0,  color=colors['VM_0'],  lw=lw, label=labels['VM_0'])
    ax.plot(probe_angles_deg, mu_VM_90, color=colors['VM_90'], lw=lw, label=labels['VM_90'])
    ax.plot(probe_angles_deg, mu_uni,   color=colors['uni'],   lw=lw, label=labels['uni'])
    ax.set_xlabel('Orientation (°)', fontsize=fs_label, fontweight='bold')
    ax.set_ylabel(r'$\mu$', fontsize=fs_ylabel, fontweight='bold')
    ax.set_xlim(0, 180)
    ax.legend()
    # Top-right: σ² vs orientation
    ax = axes[0, 1]
    ax.plot(probe_angles_deg, var_VM_0,  color=colors['VM_0'],  lw=lw)
    ax.plot(probe_angles_deg, var_VM_90, color=colors['VM_90'], lw=lw)
    ax.plot(probe_angles_deg, var_uni,   color=colors['uni'],   lw=lw)
    ax.set_xlabel('Orientation (°)', fontsize=fs_label, fontweight='bold')
    ax.set_ylabel(r'$\sigma^2$', fontsize=fs_ylabel, fontweight='bold')
    ax.set_xlim(0, 180)
    # Bottom-left: μ vs log P(θ)
    ax = axes[1, 0]
    ax.plot(log_p_VM_0,  mu_VM_0,  color=colors['VM_0'],  lw=lw, label=labels['VM_0'])
    ax.plot(log_p_VM_90, mu_VM_90, color=colors['VM_90'], lw=lw, label=labels['VM_90'])
    ax.plot(log_p_uni,   mu_uni,   color=colors['uni'],   lw=lw, label=labels['uni'])
    ax.set_xlabel(r'$\log\, P(\theta)$', fontsize=fs_label, fontweight='bold')
    ax.set_ylabel(r'$\mu$', fontsize=fs_ylabel, fontweight='bold')
    # Bottom-right: σ² vs log P(θ)
    ax = axes[1, 1]
    ax.plot(log_p_VM_0,  var_VM_0,  color=colors['VM_0'],  lw=lw)
    ax.plot(log_p_VM_90, var_VM_90, color=colors['VM_90'], lw=lw)
    ax.plot(log_p_uni,   var_uni,   color=colors['uni'],   lw=lw)
    ax.set_xlabel(r'$\log\, P(\theta)$', fontsize=fs_label, fontweight='bold')
    ax.set_ylabel(r'$\sigma^2$', fontsize=fs_ylabel, fontweight='bold')

    plt.suptitle('Log-Normal Moments After Adaptation', fontsize=16, fontweight='bold')
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

    K = dyn.frame.K
    N_TOT = dyn.N_RF * dyn.N_SETS

    def frozen_gains_mu(state):
        '''g_cRF, g_surround, mu_cRF, mu_surround sliced from a full V1Dynamics_Surround state
        (layout: y,u,a | g_cRF,g_surround,v_cRF,v_surround | mu_cRF,mu_surround).'''
        g_cRF       = state[3*N_TOT:3*N_TOT+K]
        g_surround  = state[3*N_TOT+K:3*N_TOT+2*K]
        mu_cRF      = state[3*N_TOT+4*K:3*N_TOT+4*K+N_RF]
        mu_surround = state[3*N_TOT+4*K+N_RF:3*N_TOT+4*K+2*N_RF]
        return g_cRF, g_surround, mu_cRF, mu_surround

    probe_contrasts   = np.logspace(np.log10(0.04), np.log10(1.0), 20)
    probe_angles_fig3 = np.linspace(0, np.pi, PROBE_RES)

    conditions = [
        ('Low',    'green', state_lo),
        ('Medium', 'red',   state_med),
        ('High',   'black', state_hi),
    ]

    mu_curves  = {}
    var_curves = {}

    for label, color, state in conditions:
        g_cRF, g_surround, mu_cRF, mu_surround = frozen_gains_mu(state)
        print(f"  Sweeping contrasts for {label} adapted state...")
        mus, vars_ = [], []
        for c in tqdm(probe_contrasts, desc=f"{label} contrast sweep", leave=True):
            resp = np.zeros((N_RF, PROBE_RES))
            for i, angle in enumerate(probe_angles_fig3):
                y, _, _ = get_response(dyn, probe_input_drive(angle, c), g_cRF, g_surround, mu_cRF, mu_surround)
                resp[:, i] = y[:N_RF]
            _, mu_c, var_c = calc_moments(resp)
            mus.append(np.nanmean(mu_c))
            vars_.append(np.nanmean(var_c))
        mu_curves[label]  = np.array(mus)
        var_curves[label] = np.array(vars_)

    fig3, (ax_mu, ax_var) = plt.subplots(1, 2, figsize=(12, 5))
    ln_c = np.log(probe_contrasts)
    fs3  = 18

    for label, color, *_ in conditions:
        ax_mu.plot(ln_c, mu_curves[label],  color=color, lw=2, label=label)
        ax_var.plot(ln_c, var_curves[label], color=color, lw=2, label=label)

    delta_mu  = np.nanmean(mu_curves['High'])  - np.nanmean(mu_curves['Low'])
    delta_var = np.nanmean(var_curves['High']) - np.nanmean(var_curves['Low'])

    for ax, ylabel, delta in [(ax_mu, r'$\mu$', delta_mu), (ax_var, r'$\sigma^2$', delta_var)]:
        ax.set_xlabel(r'$\ln(\mathrm{contrast})$', fontsize=fs3, fontweight='bold')
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
    stim_gen = StimulusGenerator(N_RF=N_RF, N_SETS=N_SETS, num_angles=N_RF, stream_length=STREAM_LENGTH)
    dyn = V1Dynamics_Surround(tunings, frame, N_RF=N_RF, N_SETS=N_SETS,
                               target_covariance_path=TARGET_COV_PATH, gains_nonneg=True)

    # Calibrate theta_t once, from a genuinely unbiased stream at a fixed low contrast (kept
    # separate from whatever contrast either figure below actually probes at - see
    # Surround_simulated_responses.run_adaptation_phase's 'no adaptation' condition, same idea).
    # Shared by both figures, per the model's "fixed developmental prior" design
    # (docs/whitening_adaptation_notes.md).
    print("Calibrating theta_t...")
    true_contrast = stim_gen.contrast
    stim_gen.contrast = THETA_T_CONTRAST
    try:
        calib_stream = stim_gen.generate_surround_ensembles(ADAPT_LOC, add_poisson_noise=True)
    finally:
        stim_gen.contrast = true_contrast
    (_, _, _, g_cRF_hist, g_surround_hist, v_cRF_hist, v_surround_hist,
     mu_cRF_hist, mu_surround_hist) = dyn.run_simulation(calib_stream)
    assert np.all(g_cRF_hist == 0) and np.all(g_surround_hist == 0), (
        "calibration run's gains moved away from zero - theta_t sentinel no longer holds "
        "(see V1Dynamics_Surround.__init__)."
    )
    dyn.calibrate_theta_t(v_cRF_hist, v_surround_hist, mu_cRF_hist, mu_surround_hist,
                          circular_target=True)

    Dario_fig1(dyn, stim_gen)
    Dario_fig3(dyn, stim_gen)
