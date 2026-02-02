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
    def __init__(self, v1_model, frame, dt=0.05):
        self.v1 = v1_model
        self.frame = frame
        self.dt = dt 
        
        self.tau_y = 1.0      # Time constant of primary neurons
        self.tau_a = 5.0      # Time constant of inhibitory divisive neurons
        self.tau_u = 2.0      # Time constant of normalization pool neurons
        self.tau_g = 2000.0   # Time constant of gain adaptation
        self.tau_ybar = 10.0  # Time constant of mean activity tracker 
        
        self.beta = 1.0 
        self.sigma = 0.05     # Semi-saturation constant
        self.alpha = 0.0

    def gaussian_rectify(self, y, threshold=0.0, sigma=0.1, r_max=1.0):
        return 0.5 * (1 + erf((y - threshold) / (sigma * np.sqrt(2)))) * r_max

    def _derivatives(self, state, z_t):
        N, K = self.v1.N, self.frame.K
        
        y = state[0:N]
        u = state[N:2*N]
        a = state[2*N:3*N]
        g = state[3*N:3*N+K]
        y_bar = state[3*N+K]
        
        # Stability: Gains must be non-negative
        g = np.maximum(g, 0.0) 
        
        # Estimate of firing rates from membrane potential using gaussian rectification
        u_plus = self.gaussian_rectify(u)
        y_plus = self.gaussian_rectify(y)
        a_plus = self.gaussian_rectify(a)
        sqrt_y_plus = np.sqrt(y_plus) 
        
        # Center firing rates using the tracked mean
        y_centered = y_plus - y_bar
        
        # Normalize for gain update rule (so v_t^2 targets 1)
        y_norm = np.linalg.norm(y_centered)
        if y_norm > 1e-6:
            y_normalized = y_centered / y_norm * np.sqrt(N)
        else:
            y_normalized = y_centered
        
        # Two separate projections:
        # 1) Normalized projection for gain update rule
        v_t_norm = self.frame.W.T @ y_normalized
        
        # 2) Original projection for feedback onto neurons  
        v_t = self.frame.W.T @ y_centered
        
        # Neural circuit terms
        gain_feedback = self.frame.W @ (g * v_t)
        recurrent_drive = (1.0 / (1.0 + a_plus)) * (self.v1.W_yy @ sqrt_y_plus)
        input_drive = (self.beta * z_t) / 2
        
        sigma_term = (self.sigma) ** 2
        pool_term = self.v1.N_matrix @ (y_plus * (u_plus ** 2))
        mean_y_plus = np.mean(y_plus)
        
        # Differential equations
        dy_dt = (-y + input_drive + recurrent_drive - gain_feedback) / self.tau_y
        du_dt = (-u + sigma_term + pool_term) / self.tau_u
        da_dt = (-a + u_plus + a * u_plus + self.alpha * du_dt) / self.tau_a
        dg_dt = (v_t_norm * v_t_norm - 1) / self.tau_g  # Uses normalized projection
        dy_bar_dt = (-y_bar + mean_y_plus) / self.tau_ybar
        
        return np.concatenate([dy_dt, du_dt, da_dt, dg_dt, [dy_bar_dt]])
        
    def run_simulation(self, stimulus_stream):
        N, n_steps = stimulus_stream.shape 
        K = self.frame.K
        
        state = np.zeros(3*N + K + 1)
        state[3*N:3*N+K] = 0.0  # gains
        state[3*N+K] = 0.5      # y_bar initialized to reasonable baseline
        
        membrane_hist = np.zeros((N, n_steps))
        gains_hist = np.zeros((K, n_steps))
        ybar_hist = np.zeros(n_steps)
        v_squared_hist = np.zeros((K, n_steps))
        
        print(f"Running Simulation...") 
        t0 = time.time()
        
        for t in tqdm(range(n_steps)):
            z_t = stimulus_stream[:, t] 
            
            # RK4
            k1 = self._derivatives(state, z_t)
            k2 = self._derivatives(state + 0.5 * self.dt * k1, z_t)
            k3 = self._derivatives(state + 0.5 * self.dt * k2, z_t)
            k4 = self._derivatives(state + self.dt * k3, z_t)
            
            state += (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            
            # Clamp gains >= 0
            state[3*N:3*N+K] = np.maximum(state[3*N:3*N+K], 0.0)
            
            y = state[0:N]
            g = state[3*N:3*N+K]
            y_bar = state[3*N+K]
            
            # Diagnostic logging
            y_plus_log = self.gaussian_rectify(y)
            y_centered_log = y_plus_log - y_bar
            y_norm_log = np.linalg.norm(y_centered_log)
            if y_norm_log > 1e-6:
                y_normalized_log = y_centered_log / y_norm_log * np.sqrt(N)
            else:
                y_normalized_log = y_centered_log
            v_t_norm_log = self.frame.W.T @ y_normalized_log
            v_squared_hist[:, t] = v_t_norm_log ** 2
            
            membrane_hist[:, t] = np.maximum(y, 0)
            gains_hist[:, t] = g 
            ybar_hist[t] = y_bar
            
        print(f"Simulation complete in {time.time() - t0:.2f}s.")
        
        print(f"\n--- Diagnostics ---")
        print(f"v_t^2 mean: {np.mean(v_squared_hist):.6f}")
        print(f"v_t^2 range: [{np.min(v_squared_hist):.6f}, {np.max(v_squared_hist):.6f}]")
        print(f"y_bar final: {ybar_hist[-1]:.6f}")
        print(f"gains final range: [{np.min(gains_hist[:,-1]):.6f}, {np.max(gains_hist[:,-1]):.6f}]")
        
        return membrane_hist, gains_hist, ybar_hist


if __name__ == "__main__":
    
    N_NEURONS = 60
    tunings = V1Tunings(N=N_NEURONS)
    frame = Frame(csv_path="N60_Frame.csv")
    stim_gen = StimulusGenerator(N=N_NEURONS)
    
    engine = V1Dynamics(tunings, frame, dt=0.05)
    
    regimes = [
        {'n_steps': 5000, 'contrast': 0.75, 'orientation': np.pi/2, 'label': 'Bright 90°'},
        {'n_steps': 20000, 'contrast': 0.2, 'orientation': np.pi/2, 'label': 'Dim 90°'},
        {'n_steps': 5000, 'contrast': 0.5, 'orientation': 0.0, 'label': 'Medium 0°'},
    ]
    inputs = stim_gen.generate_sequence(regimes)
    
    rates, gains, ybar = engine.run_simulation(inputs)
    
    # --- PLOTTING ---
    fig, axes = plt.subplots(4, 1, figsize=(8, 10), sharex=True, 
                             gridspec_kw={'height_ratios': [1, 2, 1, 2]})
    
    axes[0].imshow(inputs, aspect='auto', cmap='binary', origin='lower')
    axes[0].set_ylabel("Orientation")
    axes[0].set_title("Stimulus")
    axes[0].tick_params(left=False, labelleft=False) 
    
    ymax = np.percentile(rates, 99.5)
    im_y = axes[1].imshow(rates, aspect='auto', cmap='inferno', origin='lower', vmax=ymax)
    axes[1].set_ylabel("Neuron Index")
    axes[1].set_title("Firing Rates (y)")
    plt.colorbar(im_y, ax=axes[1], label="Hz", fraction=0.046, pad=0.04)
    
    axes[2].plot(ybar, color='dodgerblue', linewidth=1.5)
    axes[2].set_ylabel(r"$\bar{y}$")
    axes[2].set_title("Mean Activity Tracker")
    axes[2].grid(True, alpha=0.3)
    
    subset_k = np.linspace(0, frame.K-1, 10, dtype=int)
    axes[3].plot(gains[subset_k, :].T, alpha=0.6)
    axes[3].set_ylabel("Gains (subset)")
    axes[3].set_xlabel("Time (steps)")
    axes[3].set_title("Gain Evolution")
    axes[3].grid(True, alpha=0.3)
    
    t_cursor = 0
    for r in regimes:
        t_cursor += r['n_steps']
        for ax in axes:
            ax.axvline(t_cursor, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

    # --- Tuning Curve Check ---
    plt.figure(figsize=(8, 5))
    t_cursor = 0
    colors = ['#d62728', '#ff7f0e', '#2ca02c']
    for i, r in enumerate(regimes):
        t_end = t_cursor + r['n_steps']
        window_slice = rates[:, max(0, t_end-200) : t_end]
        avg_profile = np.mean(window_slice, axis=1)
        plt.plot(tunings.theta * 180 / np.pi, avg_profile, 
                 color=colors[i], linewidth=2.5, label=f"{r['label']}")
        t_cursor += r['n_steps']
    
    plt.title("Steady State Tuning Curves")
    plt.xlabel("Preference (Deg)")
    plt.ylabel("Activity (y)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()