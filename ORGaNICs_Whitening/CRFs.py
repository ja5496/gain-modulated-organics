"""
---- Contrast_Response_Funcs.py ----
Isolates and plots the Contrast Response Function (CRF) for a specific population 
of V1 neurons. Uses the core ORGaNICs network dynamics and probes with a strictly 
normalized profile to allow exact mathematical control over the contrast from 0 to 1.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

# Assuming these are available in your local directory
from tunings_whiten import V1Tunings
from simulation_whiten import Frame

# ---- Parameters ----
N = 169                  # Number of primary neurons
PROBE_STEPS = 100        # Steps to settle for each probe stimulus
BASELINE = 0.5           # Added baseline firing rate (and mean intensity)
TUNING_WIDTH = 0.5       # Width of the Von Mises input profile

def gaussian_rectify(y, threshold=0.5, sigma=0.2, r_max=1.0): 
    return 0.5 * (1 + erf((y - threshold) / (sigma * np.sqrt(2)))) * r_max

def run_crf_probe(frame, tunings, target_neuron_idx, stimulus_angle, contrast):
    """
    Runs the network to steady state for a single stimulus and returns 
    the response of the target neuron.
    """
    # 1. Raw Von Mises Profile
    raw_profile = np.exp(TUNING_WIDTH * np.cos(2 * (tunings.theta - stimulus_angle)))
    
    # 2. Normalize strictly to [-1, 1]. 
    # This ensures that when we apply the coefficient, it pivots symmetrically around the mean.
    p_min = np.min(raw_profile)
    p_max = np.max(raw_profile)
    f_theta = ((raw_profile - p_min) / (p_max - p_min)) * 2.0 - 1.0
    
    # 3. Calculate coefficient based on C = (Max - Min) / (2 * Mean)
    # This allows 'contrast' to cleanly map from 0.0 to 1.0
    coeff = contrast * BASELINE
    input_drive = BASELINE + (coeff * f_theta)
    
    # 4. Network Initialization (Start at baseline)
    y = np.ones(N) * BASELINE
    u = np.zeros(N)
    a = np.zeros(N)
    
    # Dynamics Parameters (Matched to non-adaptive state)
    dt = 0.05
    tau_y = 1.0
    tau_u = 2.0
    tau_a = 5.0
    sigma_const = 0.10
    W_yy = tunings.W_yy
    
    # 5. Settle to steady state
    for _ in range(PROBE_STEPS):
        u_plus = gaussian_rectify(u)
        y_plus = gaussian_rectify(y)
        a_plus = gaussian_rectify(a)
        sqrt_y_plus = np.sqrt(y_plus)
        
        recurrent_drive = (1.0 / (1.0 + a_plus)) * (W_yy @ sqrt_y_plus)
        pool_term = tunings.N_matrix @ (y_plus * (u_plus ** 2))
        
        # Purely non-adaptive for the probe
        dy = (-y + input_drive + recurrent_drive) / tau_y
        du = (-u + (sigma_const**2) + pool_term) / tau_u
        da = (-a + u_plus + a*u_plus) / tau_a
        
        y += dt * dy
        u += dt * du
        a += dt * da
        
    # Return the rectified steady-state firing rate of just the target bin
    return gaussian_rectify(y)[target_neuron_idx]

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    
    print("Initializing network for CRF extraction...")
    tunings = V1Tunings(N=N)
    frame = Frame(csv_path="Frames/N169_Frame.csv")
    
    # Frame normalization
    S = frame.W @ frame.W.T                          
    eigvals, eigvecs = np.linalg.eigh(S)
    S_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    N_neu, K_neu = frame.W.shape
    frame.W = np.sqrt(K_neu / N_neu) * (S_inv_sqrt @ frame.W)

    # Define our target: The center neuron
    target_idx = N // 2
    angle_pref = tunings.theta[target_idx]
    
    # Define probing conditions
    angle_peak = angle_pref
    angle_20deg = angle_pref + (20 * np.pi / 180.0)
    
    # Logspace from 10^-3 to 10^0 (which is 1.0) with 200 points for a smooth line
    contrasts = np.logspace(-3, 0, 200)
    
    responses_peak = np.zeros_like(contrasts)
    responses_20deg = np.zeros_like(contrasts)
    
    print(f"Target Neuron Preference: {angle_pref * 180 / np.pi:.1f}°")
    print("Probing Contrasts...")
    
    for i, c in enumerate(contrasts):
        responses_peak[i] = run_crf_probe(frame, tunings, target_idx, angle_peak, c)
        responses_20deg[i] = run_crf_probe(frame, tunings, target_idx, angle_20deg, c)
        
    # --- Plotting ---
    print("Plotting CRF...")
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Smooth lines, no markers
    ax.plot(contrasts, responses_peak, '-', color='steelblue', linewidth=3.0, 
            label='Peak (Preferred)')
    ax.plot(contrasts, responses_20deg, '-', color='coral', linewidth=3.0, 
            label='20° Away')
    
    ax.set_title("Contrast Response Function (CRF)", fontweight='bold', fontsize=14)
    ax.set_xlabel("Contrast (Log Scale)", fontsize=12)
    ax.set_ylabel("Steady State Response (Hz)", fontsize=12)
    
    # Logarithmic x-axis
    ax.set_xscale('log')
    ax.set_xlim(1e-3, 1.0)
    
    ax.grid(True, which="both", linestyle='--', alpha=0.5)
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    plt.show()