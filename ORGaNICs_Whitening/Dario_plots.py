"""
Carandini_plots.py

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
PROBE_STEPS = 100
PROBE_RES = 90

def gaussian_rectify(y, threshold=0.6, sigma=0.35, r_max=1.0):
    return 0.5 * (1 + erf((y - threshold) / (sigma * np.sqrt(2)))) * r_max

def get_responses(frame, tunings, stim_gen, fixed_gains, frozen_u, frozen_a, probe_angles):
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
    # Freeze beta at the end-of-adaptation state; fall back to 1.0 if no avg_z was tracked
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
        z_t = z_t / np.max(z_t)

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


if __name__ == "__main__":
    
    # 1. Initialize
    print("Initializing...")
    tunings = V1Tunings(N=N)
    frame = Frame(csv_path="Frames/N169_Frame.csv")
    stim_gen = StimulusGenerator(N=N, num_angles=N, stream_length=STREAM_LENGTH)
    
    # Initialize inputs
    VM_0_stream = stim_gen.generate_input_ensembles(von_mises=True, von_mises_center=0)
    VM_90_stream = stim_gen.generate_input_ensembles(von_mises=True, von_mises_center=90)
    uniform_stream = stim_gen.generate_input_ensembles()

    # Set up probe
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
    ax.legend()

    # Bottom-left: μ vs log P(θ)
    ax = axes[1, 0]
    ax.plot(log_p_VM_0,  mu_VM_0,  color=colors['VM_0'],  lw=lw, label=labels['VM_0'])
    ax.plot(log_p_VM_90, mu_VM_90, color=colors['VM_90'], lw=lw, label=labels['VM_90'])
    ax.plot(log_p_uni,   mu_uni,   color=colors['uni'],   lw=lw, label=labels['uni'])
    ax.set_xlabel(r'$\log\, P(\theta)$', fontsize=fs_label, fontweight='bold')
    ax.set_ylabel(r'$\mu$', fontsize=fs_ylabel, fontweight='bold')
    ax.legend()

    # Bottom-right: σ² vs log P(θ)
    ax = axes[1, 1]
    ax.plot(log_p_VM_0,  var_VM_0,  color=colors['VM_0'],  lw=lw, label=labels['VM_0'])
    ax.plot(log_p_VM_90, var_VM_90, color=colors['VM_90'], lw=lw, label=labels['VM_90'])
    ax.plot(log_p_uni,   var_uni,   color=colors['uni'],   lw=lw, label=labels['uni'])
    ax.set_xlabel(r'$\log\, P(\theta)$', fontsize=fs_label, fontweight='bold')
    ax.set_ylabel(r'$\sigma^2$', fontsize=fs_ylabel, fontweight='bold')
    ax.legend()

    plt.suptitle('Log-Normal Moments After Adaptation', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()