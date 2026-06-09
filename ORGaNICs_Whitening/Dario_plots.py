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
PROBE_STEPS = 25
PROBE_RES = 90

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
    beta = 1.0
    sigma = 0.1

    for i, angle in enumerate(probe_angles):

        # Start y at 0
        y = np.zeros(N)

        # Let u and a freely adapt from their most recent state
        u = np.copy(frozen_u) 
        a = np.copy(frozen_a) 

        # Construct probe stimulus identically to generate_input_ensembles 
        delta = stim_gen.theta_inputs - angle
        delta = (delta + np.pi/2) % np.pi - np.pi/2  # same wrapping as StimulusGenerator
        z_t = np.exp(-delta**2 / (2 * stim_gen.tuning_width**2))
        z_t = contrast * z_t / np.max(z_t)

        # 2. Settle to steady state
        for _ in range(PROBE_STEPS):
            # Rectifications
            u_plus = gaussian_rectify(u)
            y_plus = gaussian_rectify(y)
            a_plus = gaussian_rectify(a)
            sqrt_y_plus = np.sqrt(y_plus)

            # Circuit Inputs
            v_t = frame.W.T @ y  
            if fixed_gains is not None:
                gain_feedback = frame.W @ (fixed_gains * v_t)
            else:
                gain_feedback = 0.0

            recurrent_drive = (1.0 / (1.0 + a_plus)) * (W_yy @ sqrt_y_plus)
            input_drive = (beta * z_t) / 2 

            # Derivatives
            pool_term = tunings.N_matrix @ (y_plus * (u_plus ** 2))

            dy = (-y + input_drive + recurrent_drive - gain_feedback) / tau_y
            du = (-u + (sigma / 2)**2 + pool_term) / tau_u
            da = (-a + u_plus + a*u_plus) / tau_a

            y += dt * dy
            u += dt * du
            a += dt * da

        # Record steady state response (firing rate)
        responses[:, i] = gaussian_rectify(y) # Gaussian rectify to estimate firing rate from membrane potential
        # Note: the first index of responses gives the neuron and the second gives the angle. 
        # So for one neuron i, r_i(theta) = responses[i, theta]

    return responses

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
        print(np.mean(v))

    firing_rates = dynamics.gaussian_rectify(y)
    return firing_rates


if __name__ == "__main__":
    
    print(' ----------- FIGURE 1 -----------')
    
    def Dario_fig1():
        # 1. Initialize
        print("Initializing...")
        tunings = V1Tunings(N=N)
        frame = Frame(csv_path="Frames/N169_Frame.csv")
        stim_gen = StimulusGenerator(N=N, num_angles=N, stream_length=STREAM_LENGTH)
        VM_0_stream = stim_gen.generate_input_ensembles(von_mises=True, von_mises_center=0)
        VM_90_stream = stim_gen.generate_input_ensembles(von_mises=True, von_mises_center=90)
        uniform_stream = stim_gen.generate_input_ensembles() / 15  # match VM stream amplitude (scale≈1)
        probe_angles = np.linspace(0, np.pi, PROBE_RES)
        probe_angles_deg = probe_angles * 180 / np.pi
        results = {}

        # Begin Adaptation Stage
        print("\n--- Running Adaptation Stage ---")
        engine_adapt = V1Dynamics(tunings, frame, adaptive=True, input_adaptive=False)

        print("Adapting to Ensemble A (Von Mises at 0 degrees)...")
        VM_0_rates, gains_hist_VM_0, u_hist_VM_0, a_hist_VM_0, v_hist_VM_0, avg_z_hist_VM_0, avg_vsq_hist_VM_0 = engine_adapt.run_simulation(VM_0_stream)
        final_gains_VM_0 = gains_hist_VM_0[:, -1]
        final_u_VM_0 = u_hist_VM_0[:, -1]
        final_a_VM_0 = a_hist_VM_0[:, -1]
        print("Adapting to Ensemble B (Von Mises at 90 degrees)...")
        VM_90_rates, gains_hist_VM_90, u_hist_VM_90, a_hist_VM_90, v_hist_VM_90, avg_z_hist_VM_90, avg_vsq_hist_VM_90 = engine_adapt.run_simulation(VM_90_stream)
        final_gains_VM_90 = gains_hist_VM_90[:, -1]
        final_u_VM_90 = u_hist_VM_90[:, -1]
        final_a_VM_90 = a_hist_VM_90[:, -1]
        print("Adapting to Ensemble C (Uniform)...")
        uniform_rates, gains_hist_uni, u_hist_uni, a_hist_uni, v_hist_uni, avg_z_hist_uni, avg_vsq_hist_uni = engine_adapt.run_simulation(uniform_stream)
        final_gains_uni = gains_hist_uni[:, -1]
        final_u_uni = u_hist_uni[:, -1]
        final_a_uni = a_hist_uni[:, -1]

        # --- Probe Stage ---
        print("\n--- Running Probe Stage ---")

        print("Probing VM_0 context...")
        responses_VM_0 = get_responses(frame, tunings, stim_gen, final_gains_VM_0, final_u_VM_0, final_a_VM_0, probe_angles)
        print("Probing VM_90 context...")
        responses_VM_90 = get_responses(frame, tunings, stim_gen, final_gains_VM_90, final_u_VM_90, final_a_VM_90, probe_angles)
        print("Probing uniform context...")
        responses_uni = get_responses(frame, tunings, stim_gen, final_gains_uni, final_u_uni, final_a_uni, probe_angles)
        # --- Compute Moments ---
        P0_VM_0,  mu_VM_0,  var_VM_0  = calc_moments(responses_VM_0)
        P0_VM_90, mu_VM_90, var_VM_90 = calc_moments(responses_VM_90)
        P0_uni,   mu_uni,   var_uni   = calc_moments(responses_uni)
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
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        colors = {'VM_0': 'steelblue', 'VM_90': 'tomato', 'uni': 'gray'}
        lw = 2
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
        ax.plot(probe_angles_deg, var_VM_0,  color=colors['VM_0'],  lw=lw, label=labels['VM_0'])
        ax.plot(probe_angles_deg, var_VM_90, color=colors['VM_90'], lw=lw, label=labels['VM_90'])
        ax.plot(probe_angles_deg, var_uni,   color=colors['uni'],   lw=lw, label=labels['uni'])
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
        ax.plot(log_p_VM_0,  var_VM_0,  color=colors['VM_0'],  lw=lw, label=labels['VM_0'])
        ax.plot(log_p_VM_90, var_VM_90, color=colors['VM_90'], lw=lw, label=labels['VM_90'])
        ax.plot(log_p_uni,   var_uni,   color=colors['uni'],   lw=lw, label=labels['uni'])
        ax.set_xlabel(r'$\log\, P(\theta)$', fontsize=fs_label, fontweight='bold')
        ax.set_ylabel(r'$\sigma^2$', fontsize=fs_ylabel, fontweight='bold')

        plt.suptitle('Log-Normal Moments After Adaptation', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()


    def Dario_figs2and3():
        print(' ----------- FIGURE 2 + 3 -----------')
        # 1. Initialize
        print("Initializing...")
        tunings = V1Tunings(N=N)
        frame = Frame(csv_path="Frames/N169_Frame.csv")
        dynamics = V1Dynamics(tunings, frame, adaptive=True, input_adaptive=False)
        stim_gen = StimulusGenerator(N=N, num_angles=N, stream_length=STREAM_LENGTH)
        low_contrast_stream = stim_gen.generate_contrast_stream(peak_ln_contrast=-3)
        medium_contrast_stream = stim_gen.generate_contrast_stream(peak_ln_contrast=-1.5)
        high_contrast_stream = stim_gen.generate_contrast_stream(peak_ln_contrast=0)
        probe_contrast = 0.1357
        probe_angle = np.pi / 2          # 90 degrees
        results = {}

        # --- Adaptation Stage ---
        print("\n--- Running Adaptation Stage (Figure 2) ---")
        engine_fig2 = V1Dynamics(tunings, frame, adaptive=True, input_adaptive=False)

        print("Adapting to high contrast stream...")
        _, gains_hist_hi, u_hist_hi, a_hist_hi, *_ = engine_fig2.run_simulation(high_contrast_stream)
        final_gains_hi = gains_hist_hi[:, -1]
        final_u_hi     = u_hist_hi[:, -1]
        final_a_hi     = a_hist_hi[:, -1]

        print("Adapting to medium contrast stream...")
        _, gains_hist_med, u_hist_med, a_hist_med, *_ = engine_fig2.run_simulation(medium_contrast_stream)
        final_gains_med = gains_hist_med[:, -1]
        final_u_med     = u_hist_med[:, -1]
        final_a_med     = a_hist_med[:, -1]

        print("Adapting to low contrast stream...")
        _, gains_hist_lo, u_hist_lo, a_hist_lo, *_ = engine_fig2.run_simulation(low_contrast_stream)
        final_gains_lo = gains_hist_lo[:, -1]
        final_u_lo     = u_hist_lo[:, -1]
        final_a_lo     = a_hist_lo[:, -1]

        # --- Probe ---
        print("Probing high contrast adapted state...")
        r_hi  = probe_single_stimulus(dynamics, frame, tunings, stim_gen,
                                    final_gains_hi, final_u_hi, final_a_hi,
                                    probe_angle, probe_contrast)

        print("Probing medium contrast adapted state...")
        r_med = probe_single_stimulus(dynamics, frame, tunings, stim_gen,
                                    final_gains_med, final_u_med, final_a_med,
                                    probe_angle, probe_contrast)

        # --- Figure 2 ---
        def plot_response_hist(ax, responses, log_scale):
            r = responses[responses > 0]
            print(r)
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

        for col, (title, r) in enumerate(zip(['High Contrast', 'Medium Contrast'], [r_hi, r_med])):
            ax = axes2[0, col]
            plot_response_hist(ax, r, log_scale=True)
            ax.set_xlabel(r'$\log\, r$',    fontsize=fs, fontweight='bold')
            ax.set_ylabel(r'$\log\, P(r)$', fontsize=fs, fontweight='bold')
            ax.set_title(title, fontsize=fs, fontweight='bold')
            style_ax(ax)

            ax = axes2[1, col]
            plot_response_hist(ax, r, log_scale=False)
            ax.set_xlabel(r'$r$',    fontsize=fs, fontweight='bold')
            ax.set_ylabel(r'$P(r)$', fontsize=fs, fontweight='bold')
            style_ax(ax)

        plt.suptitle('Response Distribution After Contrast Adaptation', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

        # ---- FIGURE 3 ----
        print(' ----------- FIGURE 3 -----------')
        probe_contrasts   = np.logspace(np.log10(0.04), np.log10(1.0), 20)
        probe_angles_fig3 = np.linspace(0, np.pi, PROBE_RES)

        conditions = [
            ('Low',    'steelblue', final_gains_lo,  final_u_lo,  final_a_lo),
            ('Medium', 'seagreen',  final_gains_med, final_u_med, final_a_med),
            ('High',   'tomato',    final_gains_hi,  final_u_hi,  final_a_hi),
        ]

        mu_curves  = {}
        var_curves = {}

        for label, color, gains, u_f, a_f in conditions:
            print(f"  Sweeping contrasts for {label} adapted state...")
            mus, vars_ = [], []
            for c in probe_contrasts:
                resp = get_responses(frame, tunings, stim_gen, gains, u_f, a_f,
                                    probe_angles_fig3, contrast=c)
                if label == 'Medium':
                    print()

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

    #Dario_fig1()
    Dario_figs2and3()