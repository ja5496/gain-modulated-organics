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
        self.adaptive = adaptive  # Toggle for gain modulation/interneurons
        
        self.tau_y = 1.0      # Time constant of primary neurons
        self.tau_a = 5.0      # Time constant of inhibitory divisive neurons
        self.tau_u = 2.0      # Time constant of normalization pool neurons
        self.tau_g = 200.0    # Time constant of gain adaptation
        self.tau_v = 0.5      # Time constant of interneurons (fast so it can accurately update gains)
        
        self.beta = 1.0 
        self.sigma = 0.05     # Semi-saturation constant
        self.alpha = 0.0

    def gaussian_rectify(self, y, threshold=0.4, sigma=0.1, r_max=1.0):
        return 0.5 * (1 + erf((y - threshold) / (sigma * np.sqrt(2)))) * r_max

    def _derivatives(self, state, z_t):
        N, K = self.v1.N, self.frame.K
        
        y = state[0:N]
        u = state[N:2*N]
        a = state[2*N:3*N]
        g = state[3*N:3*N+K]
        # v = state[3*N+K:3*N+2*K] # (Unused in calculation, just tracked)
        
        # Rectifications
        u_plus = self.gaussian_rectify(u)
        y_plus = self.gaussian_rectify(y)
        a_plus = self.gaussian_rectify(a)
        sqrt_y_plus = np.sqrt(y_plus) 
        
        # --- ADAPTIVE LOGIC ---
        if self.adaptive:
            # 1) Normalized projection for gain update rule
            v_t = self.frame.W.T @ y
            
            # 2) Feedback terms
            gain_feedback = self.frame.W @ (g * v_t)
            
            # 3) Derivatives for adaptation variables
            target = np.sum((y) ** 2) / N  # Adaptive target
            dg_dt = (v_t * v_t - target) / self.tau_g
            dv_dt = (-v_t + self.frame.W.T @ y) / self.tau_v
        else:
            # Non-Adaptive: No feedback, no gain changes, no interneuron activity
            gain_feedback = 0.0
            dg_dt = np.zeros(K)
            dv_dt = np.zeros(K)

        # Neural circuit terms
        recurrent_drive = (1.0 / (1.0 + a_plus)) * (self.v1.W_yy @ sqrt_y_plus)
        input_drive = (self.beta * z_t) / 2
        
        sigma_term = (self.sigma) ** 2
        pool_term = self.v1.N_matrix @ (y_plus * (u_plus ** 2))
        
        # Differential equations
        dy_dt = (-y + input_drive + recurrent_drive - gain_feedback) / self.tau_y
        du_dt = (-u + sigma_term + pool_term) / self.tau_u
        da_dt = (-a + u_plus + a * u_plus + self.alpha * du_dt) / self.tau_a
        
        return np.concatenate([dy_dt, du_dt, da_dt, dg_dt, dv_dt])
        
    def run_simulation(self, stimulus_stream):
        N, n_steps = stimulus_stream.shape 
        K = self.frame.K
        
        state = np.zeros(3*N + 2*K)
        #state[3*N:3*N+K] = 0.0  # gains start at 0
        
        membrane_hist = np.zeros((N, n_steps))
        gains_hist = np.zeros((K, n_steps))
        v_squared_hist = np.zeros((K, n_steps))
        
        mode_str = "Adaptive" if self.adaptive else "Non-Adaptive"
        print(f"Running {mode_str} Simulation ({n_steps} steps)...") 
        t0 = time.time()
        
        for t in tqdm(range(n_steps)):
            z_t = stimulus_stream[:, t] 
            
            # RK4 Integration
            k1 = self._derivatives(state, z_t)
            k2 = self._derivatives(state + 0.5 * self.dt * k1, z_t)
            k3 = self._derivatives(state + 0.5 * self.dt * k2, z_t)
            k4 = self._derivatives(state + self.dt * k3, z_t)
            
            state += (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            
            # Clamp gains >= 0
            state[3*N:3*N+K] = np.maximum(state[3*N:3*N+K], 0.0)
            
            y = state[0:N]
            g = state[3*N:3*N+K]
            
            # Diagnostic logging
            v_t_log = state[3*N+K: 3*N+2*K]
            v_squared_hist[:, t] = v_t_log ** 2
            
            membrane_hist[:, t] = np.maximum(y, 0)
            gains_hist[:, t] = g 
            
        print(f"Simulation complete in {time.time() - t0:.2f}s.")
        return membrane_hist, gains_hist


if __name__ == "__main__":
    
    N_NEURONS = 60
    tunings = V1Tunings(N=N_NEURONS)
    frame = Frame(csv_path="Frames/N60_Frame.csv")
    stim_gen = StimulusGenerator(N=N_NEURONS, K=N_NEURONS)
    
    # Initialize Engine (Default is Adaptive)
    adapt_engine = V1Dynamics(tunings, frame, dt=0.05, adaptive=True)
    organics_engine = V1Dynamics(tunings, frame, dt=0.05, adaptive=False)
    
    # --- Define Regimes ---
    # We create two versions: one clean, one with noise
    base_regimes = [
        {'n_steps': 5000, 'contrast': 0.9, 'orientation': np.pi/2, 'label': 'Bright 90°'},
        {'n_steps': 5000, 'contrast': 0.6, 'orientation': np.pi/2, 'label': 'Dim 90°'},
        {'n_steps': 5000, 'contrast': 0.6, 'orientation': 0, 'label': 'Medium 0°'},
    ]

    # Create Clean Inputs
    for r in base_regimes: r['noise_level'] = 0.0
    inputs_clean = stim_gen.generate_sequence(base_regimes)

    # --- Run Simulations ---
    print("\n======= Simulation 1: Adaptive ORGaNICs =======")
    rates_adapt, gains_clean = adapt_engine.run_simulation(inputs_clean)

    print("\n======= Simulation 2: ORGaNICs (non-adapt) =======")
    rates_organics, gains_empty = organics_engine.run_simulation(inputs_clean)

    # --- PLOTTING ---
    # 3 Rows, 2 Columns
    fig, axes = plt.subplots(3, 2, figsize=(8, 8), gridspec_kw={'height_ratios': [1, 1.5, 1.5]})
    
    # Determine common color scaling
    vmax_stim = max(inputs_clean.max(), inputs_clean.max())
    vmax_rate = max(np.percentile(rates_adapt, 99.5), np.percentile(rates_organics, 99.5))
    
    # Define the physical extent of the axes: [x_min, x_max, y_min, y_max]
    # x: 0 to total time steps
    # y: 0 to 180 degrees
    total_steps = inputs_clean.shape[1]
    extent = [0, total_steps, 0, 180]

    # --- Row 1: Stimuli (Hot Colormap) ---
    axes[0, 0].imshow(inputs_clean, aspect='auto', cmap='hot', origin='lower', 
                      vmax=vmax_stim, extent=extent)
    axes[0, 0].set_title("Adaptive ORGaNICs", fontweight='bold')
    axes[0, 0].set_ylabel("Preference (°)", fontsize=14)
    axes[0, 0].tick_params(labelbottom=False)

    axes[0, 1].imshow(inputs_clean, aspect='auto', cmap='hot', origin='lower', 
                      vmax=vmax_stim, extent=extent)
    axes[0, 1].set_title("ORGaNICs", fontweight='bold')
    axes[0, 1].tick_params(labelleft=False, labelbottom=False)

    # --- Row 2: Dynamics (Firing Rates) ---
    axes[1, 0].imshow(rates_adapt, aspect='auto', cmap='inferno', origin='lower', 
                      vmax=vmax_rate, extent=extent)
    axes[1, 0].set_title("V1 Activity", fontweight='bold')
    axes[1, 0].set_ylabel("Preference (°)", fontsize=14)
    
    im = axes[1, 1].imshow(rates_organics, aspect='auto', cmap='inferno', origin='lower', 
                           vmax=vmax_rate, extent=extent)
    axes[1, 1].set_title("V1 Activity", fontweight='bold')
    axes[1, 1].tick_params(labelleft=False)

    # --- Row 3: Tuning Curves (Steady State of ALL Regimes) ---
    t_cursor = 0
    # Colors for the 3 regimes to distinguish them in the line plot
    regime_colors = ['#d62728', '#ff7f0e', '#2ca02c'] 
    
    ymax_curve = 0 # Track max for consistent scaling

    for i, r in enumerate(base_regimes):
        t_end = t_cursor + r['n_steps']
        t_start = t_end - 500 # Average over last 500 steps of the regime
        
        # Clean Tuning
        curve_adapt = np.mean(rates_adapt[:, t_start:t_end], axis=1)
        axes[2, 0].plot(tunings.theta * 180 / np.pi, curve_adapt, 
                        color=regime_colors[i], linewidth=2, label=r['label'])
        
        # Noisy Tuning
        curve_organics = np.mean(rates_organics[:, t_start:t_end], axis=1)
        axes[2, 1].plot(tunings.theta * 180 / np.pi, curve_organics, 
                        color=regime_colors[i], linewidth=2, label=r['label'])
        
        # Track max y for consistent scaling
        current_max = max(curve_adapt.max(), curve_organics.max())
        if current_max > ymax_curve:
            ymax_curve = current_max
            
        t_cursor += r['n_steps']

    # Formatting Row 3
    axes[2, 0].set_title("Steady State Tuning", fontweight='bold')
    axes[2, 0].set_xlabel("Orientation (°)", fontsize=14)
    axes[2, 0].set_ylabel("Response", fontsize=14)
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].set_ylim(0, ymax_curve * 1.1)
    axes[2, 0].legend(fontsize='small', loc='upper right')

    axes[2, 1].set_title("Steady State Tuning", fontweight='bold')
    axes[2, 1].set_xlabel("Orientation (°)", fontsize=14)
    axes[2, 1].grid(True, alpha=0.3)
    axes[2, 1].set_ylim(0, ymax_curve * 1.1)
    axes[2, 1].legend(fontsize='small', loc='upper right')

    # Add vertical lines for regime changes
    t_cursor = 0
    for r in base_regimes:
        t_cursor += r['n_steps']
        # Loop over flattened axes to apply lines to all image plots (first 4 subplots)
        for ax in axes.flatten()[:4]: 
            ax.axvline(t_cursor, color='white', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.show()

    # --- FIGURE 2: Aggregate Dynamics & Gain Comparison ---
    fig2, ax2 = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # 1. Top Plot: Overall Average Activity (Population Mean)
    # We calculate the mean firing rate across all neurons at each time step
    mean_activity_clean = np.mean(rates_adapt, axis=0)
    mean_activity_noisy = np.mean(rates_organics, axis=0)

    ax2[0].plot(mean_activity_clean, color='#1f77b4', linewidth=2, label='Noiseless (Clean)')
    ax2[0].plot(mean_activity_noisy, color='#d62728', linewidth=2, linestyle='--', label='Noisy')
    
    ax2[0].set_ylabel("Mean Activity (Hz)", fontsize=18)
    ax2[0].set_title("Overall Response Magnitude", fontweight='bold', fontsize=20)
    ax2[0].legend(loc='upper right')
    ax2[0].tick_params(axis='both', labelsize=16)
    ax2[0].grid(True, alpha=0.3)

    # 2. Bottom Plot: Subset of Gains (Blue Solid vs Red Dotted)
    # Select a subset of gains to keep the plot readable (e.g., 5 representative neurons)
    subset_indices = np.linspace(0, frame.K - 1, 5, dtype=int)
    
    # Generate gradients for the lines so individual neurons are distinguishable
    blue_colors = plt.cm.Blues(np.linspace(0.5, 1.0, len(subset_indices)))
    red_colors = plt.cm.Reds(np.linspace(0.5, 1.0, len(subset_indices)))

    for i, k_idx in enumerate(subset_indices):
        # Plot Clean Gains (Solid Blue)
        ax2[1].plot(gains_clean[k_idx, :], color=blue_colors[i], linestyle='-', linewidth=1.5, alpha=0.8)
        
        # Plot Noisy Gains (Dotted Red)
        #ax2[1].plot(gains_noisy[k_idx, :], color=red_colors[i], linestyle=':', linewidth=2.0, alpha=0.9)

    ax2[1].set_ylabel("Gain Amplitude", fontsize=18)
    ax2[1].set_xlabel("Time Step", fontsize=18)
    ax2[1].set_title(f"Gain Dynamics (Subset of {len(subset_indices)} neurons)", fontweight='bold', fontsize=20)
    ax2[1].tick_params(axis='both', labelsize=16)
    ax2[1].grid(True, alpha=0.3)

    # Custom Legend for the Gain Plot (Proxy artists)
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='#1f77b4', lw=2, linestyle='-'),
                    Line2D([0], [0], color='#d62728', lw=2, linestyle=':')]
    #ax2[1].legend(custom_lines, ['Gains', 'Noisy Gains'], loc='upper right')

    # Add vertical lines for regime changes
    t_cursor = 0
    for r in base_regimes:
        t_cursor += r['n_steps']
        for ax in ax2:
            ax.axvline(t_cursor, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()