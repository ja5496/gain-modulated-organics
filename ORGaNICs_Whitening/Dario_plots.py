"""
Dario_plots.py

Replicates mouse V1 adaptation experiments using adaptive ORGaNICs.

Figure 1: Analysis of post-adaptation log-normal components. Stimuli belong to one of three
distributions: (A) Von Mises Centered at 0 degrees (B) Von Mises Centered at 90 degrees or 
(C) Uniform across orientations. Recreates plots from Figure 5 of Dario's "Contrast and 
Pattern Adaptation..."




"""

import numpy as np
import matplotlib.pyplot as plt
import gc
from tqdm import tqdm
from scipy.special import erf
from tunings_whiten import V1Tunings
from stimuli_whiten import StimulusGenerator
from simulation_whiten import Frame, V1Dynamics

# ---- Parameters ----
N = 169                  # Number of primary neurons
STREAM_LENGTH = 5460    # Length of adaptation stream (steps)
PROBE_STEPS = 20
PROBE_RES = 20

def gaussian_rectify(y, threshold=0.6, sigma=0.35, r_max=1.0):
    return 0.5 * (1 + erf((y - threshold) / (sigma * np.sqrt(2)))) * r_max

def get_responses(frame, tunings, stim_gen, fixed_gains, frozen_u, frozen_a, probe_angles, contrast=1.0):
    """
    Measures response at each orientation between 0 and 180 while holding gains constant. 
    u, and a are taken from their last values and allowed to adapt.

    """
    N, K = frame.dim, frame.K
    n_probes = len(probe_angles)
    responses = np.zeros((N, n_probes))

    W_yy = tunings.W_yy

    dt = 0.1
    tau_y = 0.4
    tau_u = 0.8
    tau_a = 2.0
    tau_v = 100
    beta = 1.0
    sigma = 0.1
    sigma_term = (sigma / 2) ** 2

    for i, angle in enumerate(probe_angles):

        # Start y at 0
        y = np.zeros(N)

        # Let u and a freely adapt from their most recent state
        u = np.copy(frozen_u)
        a = np.copy(frozen_a)
        v = np.zeros(K)

        # Construct probe stimulus identically to generate_input_ensembles
        delta = stim_gen.theta_inputs - angle
        delta = (delta + np.pi/2) % np.pi - np.pi/2  # same wrapping as StimulusGenerator
        scale = 15
        z_t = np.exp(-delta**2 / (2 * stim_gen.tuning_width**2))
        z_t = contrast * scale * z_t / np.max(z_t)

        def derivs(y_, u_, a_, v_):
            u_plus = gaussian_rectify(u_)
            y_plus = gaussian_rectify(y_)
            a_plus = gaussian_rectify(a_)
            sqrt_y_plus = np.sqrt(y_plus)

            if fixed_gains is not None:
                gain_feedback = frame.W @ (fixed_gains * v)
            else:
                gain_feedback = 0.0

            recurrent_drive = (1.0 / (1.0 + a_plus)) * (W_yy @ sqrt_y_plus)
            input_drive = (beta * z_t) / 2
            pool_term = tunings.N_matrix @ (y_plus * (u_plus ** 2))

            dy = (-y_ + input_drive + recurrent_drive - gain_feedback) / tau_y
            du = (-u_ + sigma_term + pool_term) / tau_u
            da = (-a_ + u_plus + a_ * u_plus) / tau_a
            dv = (-v_ + frame.W.T @ y_) / tau_v
            return dy, du, da, dv

        # 2. Settle to steady state
        for _ in range(PROBE_STEPS):
            dy1, du1, da1, dv1 = derivs(y, u, a, v)
            dy2, du2, da2, dv2 = derivs(y + 0.5*dt*dy1, u + 0.5*dt*du1, a + 0.5*dt*da1, v + 0.5*dt*dv1)
            dy3, du3, da3, dv3 = derivs(y + 0.5*dt*dy2, u + 0.5*dt*du2, a + 0.5*dt*da2, v + 0.5*dt*dv2)
            dy4, du4, da4, dv4 = derivs(y +     dt*dy3, u +     dt*du3, a +     dt*da3, v +     dt*dv3)

            y += (dt / 6.0) * (dy1 + 2*dy2 + 2*dy3 + dy4)
            u += (dt / 6.0) * (du1 + 2*du2 + 2*du3 + du4)
            a += (dt / 6.0) * (da1 + 2*da2 + 2*da3 + da4)
            v += (dt / 6.0) * (dv1 + 2*dv2 + 2*dv3 + dv4)

        # Record steady state response (firing rate)
        responses[:, i] = gaussian_rectify(y) # Gaussian rectify to estimate firing rate from membrane potential
        # Note: the first index of responses gives the neuron and the second gives the angle. 
        # So for one neuron i, r_i(theta) = responses[i, theta]

    return responses

def probe_ensemble_moments(y_hist, stim_angles, probe_angle_bins):
    """
    Compute log-normal moments at each probe orientation by averaging over the
    statistical ensemble. y_hist is (N, T) membrane potentials from the probe
    simulation; stim_angles is (T,) stimulus angle in radians per step.
    Returns P_0, mu, variance each of shape (n_probes,).
    Bins with no matching time steps return np.nan.
    """

    n_probes = len(probe_angle_bins)
    firing_rates = gaussian_rectify(y_hist)   # (N, T)
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


def probe_single_stimulus(dynamics, frame, tunings, stim_gen, fixed_gains, frozen_u, frozen_a,
                          angle, contrast):
    """Steady-state firing-rate vector (N,) for one orientation at a given contrast."""
    N_neurons = frame.dim
    W_yy = tunings.W_yy
    dt, tau_y, tau_u, tau_a, beta, sigma = 0.1, 0.4, 0.8, 2.0, 1.0, 0.1

    delta = stim_gen.theta_inputs - angle
    delta = (delta + np.pi/2) % np.pi - np.pi/2
    profile = np.exp(-delta**2 / (2 * stim_gen.tuning_width**2))
    scale = 15 # COEFFICIENT OF ~15 ACHIEVES CORRECT SATURATION FOR CONTRAST OF 1
    z_t = contrast * scale * profile / np.max(profile)

    y = np.zeros(N_neurons)
    u = np.copy(frozen_u)
    a = np.copy(frozen_a)
    v = np.zeros(dynamics.frame.K)
    tau_v = 100
    sigma_term = (dynamics.sigma / 2) ** 2
    
    def derivs(y_, u_, a_, v_):
        u_plus = dynamics.gaussian_rectify(u_)
        y_plus = dynamics.gaussian_rectify(y_)
        a_plus = dynamics.gaussian_rectify(a_)
        sq_y_plus   = np.sqrt(y_plus)

        gain_fb = dynamics.frame.W @ (fixed_gains * v)
        rec     = (1.0 / (1.0 + a_plus)) * (dynamics.v1.W_yy @ sq_y_plus)
        pool_t  = dynamics.v1.N_matrix @ (y_plus * (u_plus ** 2))

        dy = (-y_ + z_t / 2 + rec - gain_fb) / dynamics.tau_y
        du = (-u_ + sigma_term + pool_t)     / dynamics.tau_u
        da = (-a_ + u_plus + a_ * u_plus)    / dynamics.tau_a
        dv = (-v_ + dynamics.frame.W.T @ y_) / tau_v
        return dy, du, da, dv

    for _ in range(PROBE_STEPS):
        dy1, du1, da1, dv1 = derivs(y, u, a, v)
        dy2, du2, da2, dv2 = derivs(y + 0.5*dt*dy1, u + 0.5*dt*du1, a + 0.5*dt*da1, v + 0.5*dt*dv1)
        dy3, du3, da3, dv3 = derivs(y + 0.5*dt*dy2, u + 0.5*dt*du2, a + 0.5*dt*da2, v + 0.5*dt*dv2)
        dy4, du4, da4, dv4 = derivs(y +     dt*dy3, u +     dt*du3, a +     dt*da3, v +     dt*dv3)

        y += (dt / 6.0) * (dy1 + 2*dy2 + 2*dy3 + dy4)
        u += (dt / 6.0) * (du1 + 2*du2 + 2*du3 + du4)
        a += (dt / 6.0) * (da1 + 2*da2 + 2*da3 + da4)
        v += (dt / 6.0) * (dv1 + 2*dv2 + 2*dv3 + dv4)

    firing_rates = dynamics.gaussian_rectify(y)
    return firing_rates


def pool_adaptation_responses(y_hist, contrasts_per_pres,
                               dynamics, target_contrast=0.15, duration=20):
    """
    Pool end-of-presentation firing rates for all presentations whose contrast
    falls within ~±40% of target_contrast (log-scale tolerance).

    Returns (pooled_responses, n_presentations).
    """
    dist = np.abs(contrasts_per_pres - target_contrast)
    tol = target_contrast * (np.exp(0.5) - 1)   # ~±40% relative window in linear space
    near_mask = dist <= tol
    if near_mask.sum() == 0:                     # fallback: nearest 10 presentations
        tol = np.sort(dist)[min(9, len(dist) - 1)]
        near_mask = dist <= tol

    pres_idx = np.where(near_mask)[0]

    pooled = []
    for i in pres_idx:
        t_end = (i + 1) * duration - 1
        pooled.append(dynamics.gaussian_rectify(y_hist[:, t_end]))

    pooled_responses = np.concatenate(pooled) if pooled else np.array([])
    return pooled_responses, len(pres_idx)


if __name__ == "__main__":
    
    
    
    def Dario_fig1():
        # 1. Initialize
        print(' ----------- FIGURE 1 -----------')
        print("Initializing...")
        tunings = V1Tunings(N=N)
        frame = Frame(csv_path="Frames/N169_Frame.csv")
        stim_gen = StimulusGenerator(N=N, num_angles=N, stream_length=STREAM_LENGTH)
        VM_0_stream = stim_gen.generate_input_ensembles(von_mises=True, von_mises_center=0)
        VM_90_stream = stim_gen.generate_input_ensembles(von_mises=True, von_mises_center=90)
        uniform_stream = stim_gen.generate_input_ensembles()  
        probe_angles = np.linspace(0, np.pi, PROBE_RES)
        probe_angles_deg = probe_angles * 180 / np.pi
        results = {}

        # Begin Adaptation Stage
        print("\n--- Running Adaptation Stage ---")
        engine_adapt = V1Dynamics(tunings, frame, adaptive=True, input_adaptive=False)

        print("Adapting to Ensemble A (Von Mises at 0 degrees)...")
        engine_adapt.run_simulation(VM_0_stream)
        final_state_VM_0 = engine_adapt.last_state

        print("Adapting to Ensemble B (Von Mises at 90 degrees)...")
        engine_adapt.run_simulation(VM_90_stream)
        final_state_VM_90 = engine_adapt.last_state

        print("Adapting to Ensemble C (Uniform)...")
        engine_adapt.run_simulation(uniform_stream)
        final_state_uni = engine_adapt.last_state

        # --- Probe Stage: full dynamics from exact final adaptation state ---
        print("\n--- Running Probe Stage ---")

        print("Probing VM_0 context...")
        VM_0_probe_stream, VM_0_probe_angles = stim_gen.generate_input_ensembles(
            von_mises=True, von_mises_center=0, return_angles=True)
        y_hist_probe_VM_0, *_ = engine_adapt.run_simulation(
            VM_0_probe_stream, initial_state=final_state_VM_0)

        print("Probing VM_90 context...")
        VM_90_probe_stream, VM_90_probe_angles = stim_gen.generate_input_ensembles(
            von_mises=True, von_mises_center=90, return_angles=True)
        y_hist_probe_VM_90, *_ = engine_adapt.run_simulation(
            VM_90_probe_stream, initial_state=final_state_VM_90)

        print("Probing uniform context...")
        uni_probe_stream, uni_probe_angles = stim_gen.generate_input_ensembles(return_angles=True)
        y_hist_probe_uni, *_ = engine_adapt.run_simulation(
            uni_probe_stream, initial_state=final_state_uni)

        # --- Compute Moments over ensemble ---
        P0_VM_0,  mu_VM_0,  var_VM_0  = probe_ensemble_moments(y_hist_probe_VM_0,  VM_0_probe_angles,  probe_angles)
        P0_VM_90, mu_VM_90, var_VM_90 = probe_ensemble_moments(y_hist_probe_VM_90, VM_90_probe_angles, probe_angles)
        P0_uni,   mu_uni,   var_uni   = probe_ensemble_moments(y_hist_probe_uni,   uni_probe_angles,   probe_angles)

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


    def Dario_figs2and3():
        print(' ----------- FIGURE 2 -----------')
        # 1. Initialize
        print("Initializing...")
        tunings = V1Tunings(N=N)
        frame = Frame(csv_path="Frames/N169_Frame.csv")
        stim_gen = StimulusGenerator(N=N, num_angles=N, stream_length=STREAM_LENGTH)
        low_contrast_stream = stim_gen.generate_contrast_stream(peak_ln_contrast=-3)
        medium_contrast_stream, _, contrasts_med = stim_gen.generate_contrast_stream(
            peak_ln_contrast=-1.5, return_metadata=True)
        high_contrast_stream, _, contrasts_hi = stim_gen.generate_contrast_stream(
            peak_ln_contrast=0, return_metadata=True)
        results = {}

        # --- Adaptation Stage ---
        print("\n--- Running Adaptation Stage (Figure 2) ---")
        engine_fig2 = V1Dynamics(tunings, frame, adaptive=True, input_adaptive=False)

        print("Adapting to high contrast stream...")
        y_hist_hi, gains_hist_hi, u_hist_hi, a_hist_hi, *_ = engine_fig2.run_simulation(high_contrast_stream)
        final_gains_hi = gains_hist_hi[:, -1]
        final_u_hi     = u_hist_hi[:, -1]
        final_a_hi     = a_hist_hi[:, -1]

        print("Adapting to medium contrast stream...")
        y_hist_med, gains_hist_med, u_hist_med, a_hist_med, *_ = engine_fig2.run_simulation(medium_contrast_stream)
        final_gains_med = gains_hist_med[:, -1]
        final_u_med     = u_hist_med[:, -1]
        final_a_med     = a_hist_med[:, -1]

        print("Adapting to low contrast stream...")
        _, gains_hist_lo, u_hist_lo, a_hist_lo, *_ = engine_fig2.run_simulation(low_contrast_stream)
        final_gains_lo = gains_hist_lo[:, -1]
        final_u_lo     = u_hist_lo[:, -1]
        final_a_lo     = a_hist_lo[:, -1]

        # --- Pool responses from actual adaptation presentations ---
        print("Pooling high contrast adaptation responses...")
        r_hi, n_hi = pool_adaptation_responses(y_hist_hi, contrasts_hi, engine_fig2)
        print(f"  n presentations: {n_hi}")

        print("Pooling medium contrast adaptation responses...")
        r_med, n_med = pool_adaptation_responses(y_hist_med, contrasts_med, engine_fig2)
        print(f"  n presentations: {n_med}")

        # --- Figure 2 ---
        def plot_response_hist(ax, responses, log_scale):
            r = responses[responses > 0]
            if r.size == 0:
                return
            BIN_COLOR = '#CC7000'
            EDGE_COLOR = '#333333'
            if log_scale:
                r_min, r_max = r.min(), r.max()
                if r_min == r_max:
                    return
                bins = np.logspace(np.log10(r_min), np.log10(r_max), 20)
                counts, edges = np.histogram(r, bins=bins, density=False)
                probs = counts / counts.sum()
                centers = np.sqrt(edges[:-1] * edges[1:])
                log_probs = np.where(counts > 0, np.log10(probs), np.nan)
                valid_lp = log_probs[np.isfinite(log_probs)]
                if valid_lp.size == 0:
                    return
                lp_range = valid_lp.max() - valid_lp.min()
                y_bottom = valid_lp.min() - 0.05 * (lp_range if lp_range > 0 else 1)
                heights = np.where(np.isfinite(log_probs), log_probs - y_bottom, 0)
                ax.bar(np.log10(centers), heights,
                    width=np.diff(np.log10(edges)), color=BIN_COLOR,
                    edgecolor=EDGE_COLOR, linewidth=0.8,
                    bottom=y_bottom, alpha=1.0, align='edge')
                ax.set_ylim(bottom=y_bottom)
            else:
                counts, edges = np.histogram(responses, bins=20, density=False)
                probs = counts / counts.sum()
                ax.bar((edges[:-1] + edges[1:]) / 2, probs, width=np.diff(edges),
                    color=BIN_COLOR, edgecolor=EDGE_COLOR, linewidth=0.8,
                    alpha=1.0, align='edge')

        def style_ax(ax):
            ax.spines[['top', 'right']].set_visible(False)
            for spine in ['left', 'bottom']:
                ax.spines[spine].set_color('black')
                ax.spines[spine].set_linewidth(2.0)
            ax.tick_params(colors='black', width=2.0, labelsize=13)
            ax.locator_params(axis='both', nbins=4)

        fig2, axes2 = plt.subplots(2, 2, figsize=(10, 8))
        fs = 18

        col_info = [
            ('High Contrast',   r_hi,  n_hi),
            ('Medium Contrast', r_med, n_med),
        ]
        for col, (title, r, n_pres) in enumerate(col_info):
            ax = axes2[0, col]
            plot_response_hist(ax, r, log_scale=True)
            ax.set_xlabel(r'$\log\, (R)$',    fontsize=fs, fontweight='bold')
            ax.set_ylabel(r'$\log\, P(R)$', fontsize=fs, fontweight='bold')
            ax.set_title(f'{title}\n(n={n_pres})', fontsize=fs - 2, fontweight='bold')
            style_ax(ax)
            ax.locator_params(axis='both', nbins=6)

            ax = axes2[1, col]
            plot_response_hist(ax, r, log_scale=False)
            ax.set_xlabel(r'$Response \,(R)$',    fontsize=fs, fontweight='bold')
            ax.set_ylabel(r'$P(R)$', fontsize=fs, fontweight='bold')
            style_ax(ax)

        plt.suptitle('Response Distribution After Contrast Adaptation', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

        # ---- FIGURE 3 ----
        print(' ----------- FIGURE 3 -----------')
        probe_contrasts   = np.logspace(np.log10(0.04), np.log10(1.0), 20)
        probe_angles_fig3 = np.linspace(0, np.pi, PROBE_RES)

        conditions = [
            ('Low',    'green', final_gains_lo,  final_u_lo,  final_a_lo),
            ('Medium', 'red',   final_gains_med, final_u_med, final_a_med),
            ('High',   'black', final_gains_hi,  final_u_hi,  final_a_hi),
        ]

        mu_curves  = {}
        var_curves = {}

        for label, color, gains, u_f, a_f in conditions:
            print(f"  Sweeping contrasts for {label} adapted state...")
            mus, vars_ = [], []
            for c in tqdm(probe_contrasts, desc=f"{label} contrast sweep", leave=True):
                resp = get_responses(frame, tunings, stim_gen, gains, u_f, a_f,
                                    probe_angles_fig3, contrast=c)
                if label == 'Medium':
                    ()

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

    Dario_fig1()
    #Dario_figs2and3()