import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from tunings_whiten import V1Tunings
from stimuli_whiten import StimulusGenerator
from scipy.special import erf

class Frame:
    '''Lightweight Frame class that loads W from a pre-computed csv file.'''
    def __init__(self, csv_path: str):
        print(f"Loading frame from {csv_path}...")
        self.W = np.loadtxt(csv_path, delimiter=",")
        self.dim = self.W.shape[0]
        self.K = self.W.shape[1]
        print(f"Loaded frame (N={self.dim}, K={self.K})")

class V1Dynamics:
    def __init__(self, v1_model, frame, dt=0.05, adaptive=True):
        self.v1 = v1_model
        self.frame = frame
        self.dt = dt 
        self.adaptive = adaptive  
        
        self.tau_y = 1.0      
        self.tau_a = 5.0      
        self.tau_u = 2.0      
        self.tau_g = 200.0    
        self.tau_v = 0.5      
        self.tau_avg = 20.0
        
        self.beta = 1.0 
        self.sigma = 0.05     
        self.alpha = 0.0

    def gaussian_rectify(self, y, threshold=0.5, sigma=0.25, r_max=1.0):
        return 0.5 * (1 + erf((y - threshold) / (sigma * np.sqrt(2)))) * r_max

    def _derivatives(self, state, z_t):
        N, K = self.v1.N, self.frame.K
        
        y = state[0:N]
        u = state[N:2*N]
        a = state[2*N:3*N]
        g = state[3*N:3*N+K]
        avg = state[3*N+2*K:3*N+2*K+1]
        
        u_plus = self.gaussian_rectify(u)
        y_plus = self.gaussian_rectify(y)
        a_plus = self.gaussian_rectify(a)
        sqrt_y_plus = np.sqrt(y_plus) 
        
        if self.adaptive:
            v_t = self.frame.W.T @ y
            gain_feedback = self.frame.W @ (g * v_t)
            davg_dt = (-avg + np.linalg.norm(y)) / self.tau_avg
            dg_dt = (v_t * v_t - (avg)**2/ N) / self.tau_g
            dv_dt = (-v_t + self.frame.W.T @ y) / self.tau_v
        else:
            gain_feedback = 0.0
            dg_dt = np.zeros(K)
            dv_dt = np.zeros(K)
            davg_dt = np.zeros(1)
            dbeta_dt = np.ones(N)

        recurrent_drive = (1.0 / (1.0 + a_plus)) * (self.v1.W_yy @ sqrt_y_plus)
        input_drive = (self.beta * z_t) / 2 # Renamed from self.beta to beta to allow adaptation
        
        sigma_term = (self.sigma) ** 2
        pool_term = self.v1.N_matrix @ (y_plus * (u_plus ** 2))
        
        dy_dt = (-y + input_drive + recurrent_drive - gain_feedback) / self.tau_y
        du_dt = (-u + sigma_term + pool_term) / self.tau_u
        da_dt = (-a + u_plus + a * u_plus + self.alpha * du_dt) / self.tau_a
        
        return np.concatenate([dy_dt, du_dt, da_dt, dg_dt, dv_dt, davg_dt])
        
    def run_simulation(self, stimulus_stream):
        N, n_steps = stimulus_stream.shape 
        K = self.frame.K
        
        state = np.zeros(3*N + 2*K + 1) 
        
        membrane_hist = np.zeros((N, n_steps))
        gains_hist = np.zeros((K, n_steps))
        
        # ---> ADDED: Tracking for u and a so they can be frozen later
        u_hist = np.zeros((N, n_steps))
        a_hist = np.zeros((N, n_steps))
        
        mode_str = "Adaptive" if self.adaptive else "Non-Adaptive"
        print(f"Running {mode_str} Simulation ({n_steps} steps)...") 
        t0 = time.time()
        
        for t in tqdm(range(n_steps)):
            z_t = stimulus_stream[:, t] 
            
            k1 = self._derivatives(state, z_t)
            k2 = self._derivatives(state + 0.5 * self.dt * k1, z_t)
            k3 = self._derivatives(state + 0.5 * self.dt * k2, z_t)
            k4 = self._derivatives(state + self.dt * k3, z_t)
            
            state += (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            
            state[3*N:3*N+K] = np.maximum(state[3*N:3*N+K], 0)
            
            membrane_hist[:, t] = np.maximum(state[0:N], 0)
            u_hist[:, t] = state[N:2*N]
            a_hist[:, t] = state[2*N:3*N]
            gains_hist[:, t] = state[3*N:3*N+K]
            
        print(f"Simulation complete in {time.time() - t0:.2f}s.")
        return membrane_hist, gains_hist, u_hist, a_hist


if __name__ == "__main__":
    
    N_NEURONS = 60
    tunings = V1Tunings(N=N_NEURONS)
    frame = Frame(csv_path="Frames/N60_Frame.csv")
    stim_gen = StimulusGenerator(N=N_NEURONS, K=N_NEURONS)
    
    adapt_engine = V1Dynamics(tunings, frame, dt=0.05, adaptive=True)
    organics_engine = V1Dynamics(tunings, frame, dt=0.05, adaptive=False)
    
    base_regimes = [
        {'n_steps': 5000, 'contrast': 0.9, 'orientation': np.pi/2, 'label': 'Bright 90°'},
        {'n_steps': 5000, 'contrast': 0.6, 'orientation': np.pi/2, 'label': 'Dim 90°'},
        {'n_steps': 5000, 'contrast': 0.6, 'orientation': 0, 'label': 'Medium 0°'},
    ]

    for r in base_regimes: r['noise_level'] = 0.0
    inputs_clean = stim_gen.generate_sequence(base_regimes)

    print("\n======= Simulation 1: Adaptive ORGaNICs =======")
    # Now unpacking 4 variables instead of 2
    rates_adapt, gains_clean, u_adp, a_adp = adapt_engine.run_simulation(inputs_clean)

    print("\n======= Simulation 2: ORGaNICs (non-adapt) =======")
    rates_organics, gains_empty, u_org, a_org = organics_engine.run_simulation(inputs_clean)

    # --- PLOTTING ---
    fig, axes = plt.subplots(3, 2, figsize=(8, 8), gridspec_kw={'height_ratios': [1, 1.5, 1.5]})
    
    vmax_stim = max(inputs_clean.max(), inputs_clean.max())
    vmax_rate = max(np.percentile(rates_adapt, 99.5), np.percentile(rates_organics, 99.5))
    
    total_steps = inputs_clean.shape[1]
    extent = [0, total_steps, 0, 180]

    # Row 1
    axes[0, 0].imshow(inputs_clean, aspect='auto', cmap='hot', origin='lower', vmax=vmax_stim, extent=extent)
    axes[0, 0].set_title("Adaptive ORGaNICs", fontweight='bold')
    axes[0, 0].set_ylabel("Preference (°)", fontsize=14)
    axes[0, 0].tick_params(labelbottom=False)

    axes[0, 1].imshow(inputs_clean, aspect='auto', cmap='hot', origin='lower', vmax=vmax_stim, extent=extent)
    axes[0, 1].set_title("ORGaNICs", fontweight='bold')
    axes[0, 1].tick_params(labelleft=False, labelbottom=False)

    # Row 2
    axes[1, 0].imshow(rates_adapt, aspect='auto', cmap='inferno', origin='lower', vmax=vmax_rate, extent=extent)
    axes[1, 0].set_title("V1 Activity", fontweight='bold')
    axes[1, 0].set_ylabel("Preference (°)", fontsize=14)
    
    axes[1, 1].imshow(rates_organics, aspect='auto', cmap='inferno', origin='lower', vmax=vmax_rate, extent=extent)
    axes[1, 1].set_title("V1 Activity", fontweight='bold')
    axes[1, 1].tick_params(labelleft=False)

    # Row 3 (Population Responses)
    t_cursor = 0
    regime_colors = ['#d62728', '#ff7f0e', '#2ca02c'] 
    ymax_curve = 0 

    for i, r in enumerate(base_regimes):
        t_end = t_cursor + r['n_steps']
        t_start = t_end - 500 
        
        curve_adapt = np.mean(rates_adapt[:, t_start:t_end], axis=1)
        axes[2, 0].plot(tunings.theta * 180 / np.pi, curve_adapt, color=regime_colors[i], linewidth=2, label=r['label'])
        
        curve_organics = np.mean(rates_organics[:, t_start:t_end], axis=1)
        axes[2, 1].plot(tunings.theta * 180 / np.pi, curve_organics, color=regime_colors[i], linewidth=2, label=r['label'])
        
        current_max = max(curve_adapt.max(), curve_organics.max())
        if current_max > ymax_curve: ymax_curve = current_max
            
        t_cursor += r['n_steps']

    # Changed titles to reflect accurate terminology
    axes[2, 0].set_title("Live Population Response", fontweight='bold')
    axes[2, 0].set_xlabel("Preferred Orientation (°)", fontsize=14)
    axes[2, 0].set_ylabel("Response", fontsize=14)
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].set_ylim(0, ymax_curve * 1.1)
    axes[2, 0].legend(fontsize='small', loc='upper right')

    axes[2, 1].set_title("Live Population Response", fontweight='bold')
    axes[2, 1].set_xlabel("Preferred Orientation (°)", fontsize=14)
    axes[2, 1].grid(True, alpha=0.3)
    axes[2, 1].set_ylim(0, ymax_curve * 1.1)
    axes[2, 1].legend(fontsize='small', loc='upper right')

    t_cursor = 0
    for r in base_regimes:
        t_cursor += r['n_steps']
        for ax in axes.flatten()[:4]: 
            ax.axvline(t_cursor, color='white', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.show()

    # --- FIGURE 2: Aggregate Dynamics & Gain Comparison ---
    fig2, ax2 = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    mean_activity_clean = np.mean(rates_adapt, axis=0)
    mean_activity_noisy = np.mean(rates_organics, axis=0)

    ax2[0].plot(mean_activity_clean, color='#1f77b4', linewidth=2, label='Adaptive')
    ax2[0].plot(mean_activity_noisy, color='#d62728', linewidth=2, linestyle='--', label='Non-Adaptive')
    
    ax2[0].set_ylabel("Mean Activity (Hz)", fontsize=18)
    ax2[0].set_title("Overall Response Magnitude", fontweight='bold', fontsize=20)
    ax2[0].legend(loc='upper right')
    ax2[0].tick_params(axis='both', labelsize=16)
    ax2[0].grid(True, alpha=0.3)

    subset_indices = np.linspace(0, frame.K - 1, 5, dtype=int)
    blue_colors = plt.cm.Blues(np.linspace(0.5, 1.0, len(subset_indices)))

    for i, k_idx in enumerate(subset_indices):
        ax2[1].plot(gains_clean[k_idx, :], color=blue_colors[i], linestyle='-', linewidth=1.5, alpha=0.8)

    ax2[1].set_ylabel("Gain Amplitude", fontsize=18)
    ax2[1].set_xlabel("Time Step", fontsize=18)
    ax2[1].set_title(f"Gain Dynamics (Subset of {len(subset_indices)} neurons)", fontweight='bold', fontsize=20)
    ax2[1].tick_params(axis='both', labelsize=16)
    ax2[1].grid(True, alpha=0.3)

    t_cursor = 0
    for r in base_regimes:
        t_cursor += r['n_steps']
        for ax in ax2:
            ax.axvline(t_cursor, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()
    
    
    # =====================================================================
    # ---> EXAMPLE: HOW TO PROBE TRUE TUNING CURVES WITH FROZEN STATES
    # =====================================================================
    print("\n--- Running True Tuning Curve Probes ---")
    
    # 1. Grab the states at the very end of the simulation
    final_gains = gains_clean[:, -1]
    final_u_adp = u_adp[:, -1]
    final_a_adp = a_adp[:, -1]
    
    # 2. Define angles to test
    probe_angles = np.linspace(0, np.pi, 20)
    
    # 3. Run the probe using the engine!
    tuned_curves = adapt_engine.run_probe(probe_angles, final_gains, final_u_adp, final_a_adp)
    
    # 4. If you wanted to plot the tuning curve of the 90-degree neuron:
    # idx_90 = np.argmin(np.abs(tunings.theta - np.pi/2))
    # plt.plot(probe_angles * 180 / np.pi, tuned_curves[idx_90, :])
    # plt.title("True Tuning Curve of 90° Neuron after Adaptation")
    # plt.show()