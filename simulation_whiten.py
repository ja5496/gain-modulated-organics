import numpy as np
import matplotlib.pyplot as plt
import time

# --- 1. Dependencies (Tunings & Stimuli) ---

class V1Tunings:
    def __init__(self, N=180, kappa_exc=15.0, kappa_norm=0.05):
        self.N = N
        self.theta = np.linspace(0, np.pi, N, endpoint=False)
        self.b0 = np.ones(N) * 0.2
        
        # Recurrent Excitation (Scaled)
        self.W_rec = self._make_dist_weights(kappa_exc, normalize_by='max') * 0.15
        
        # Normalization Pool
        self.W_norm = self._make_dist_weights(kappa_norm, normalize_by='sum')

    def _make_dist_weights(self, kappa, normalize_by='max'):
        W = np.zeros((self.N, self.N))
        for i in range(self.N):
            d = np.abs(self.theta - self.theta[i])
            d = np.minimum(d, np.pi - d)
            W[i, :] = np.exp(kappa * np.cos(2*d))
        
        if normalize_by == 'sum':
            return W / np.sum(W, axis=1, keepdims=True)
        else:
            return W / np.max(W)

class StimulusGenerator:
    def __init__(self, N=180):
        self.N = N
        self.theta = np.linspace(0, np.pi, N, endpoint=False)
        
    def generate_sequence(self, regimes):
        seq = []
        for r in regimes:
            profile = np.exp(8.0 * np.cos(2*(self.theta - r['orientation'])))
            profile = profile / np.max(profile) * r['contrast']
            seq.append(np.tile(profile, (r['n_steps'], 1)).T)
        return np.hstack(seq)

# --- 2. Main Dynamics Engine ---

class V1Dynamics:
    def __init__(self, v1_model, dt=1.0, tau_v=1.0, tau_a=2.0, tau_u=1.0, tau_adapt=100.0):
        self.v1 = v1_model
        self.dt = dt
        self.tau_v = tau_v
        self.tau_a = tau_a
        self.tau_u = tau_u
        self.tau_adapt = tau_adapt
        
        # User Parameters (Feedforward Adaptation Config)
        self.alpha = 0.15          # Input Fatigue Strength
        self.beta = 0.005          # Lateral Whitening Strength
        self.target_b0 = 0.2       # Resting gain
        self.sigma_noise = 0.1     # Spontaneous drive
        
    def run_simulation(self, stimulus_stream):
        N, n_steps = stimulus_stream.shape
        
        # State Variables
        v = np.zeros(N)
        a = np.zeros(N)
        u = np.zeros(N)
        b0 = self.v1.b0.copy()
        
        # History
        rates_hist = np.zeros((N, n_steps))
        b0_hist = np.zeros((N, n_steps))
        
        print(f"Running ORGaNICs simulation ({n_steps} steps)...")
        t0 = time.time()
        
        for t in range(n_steps):
            
            # 1. Output Activity (y)
            v_rect = np.maximum(v, 0)
            y = v_rect ** 2.0
            
            # 2. Input Drive (z) with Gain
            z = stimulus_stream[:, t]
            input_drive = (b0 / (1.0 + b0)) * z
            
            # 3. Recurrent Drives
            y_hat = self.v1.W_rec @ v_rect
            pool_drive = self.v1.W_norm @ y 
            
            # 4. Updates (Euler Integration)
            
            # u: Normalization Pool
            sigma_term = (self.sigma_noise * b0 / (1.0 + b0))**2
            du = (-u + pool_drive + sigma_term) / self.tau_u
            u += du * self.dt
            u = np.maximum(u, 0)
            
            # a: Normalization Factor
            da = (-a + np.sqrt(u) + a * np.sqrt(u)) / self.tau_a
            a += da * self.dt
            
            # v: Excitatory Potential
            recurrent_drive = y_hat / (1.0 + a)
            dv = (-v + input_drive + recurrent_drive) / self.tau_v
            v += dv * self.dt
            
            # b0: Adaptation (Input Fatigue Logic)
            
            # want gain to increase when signal is distinct but low and decrease when high. right now gain only decreases
            
            db0 = ((self.target_b0 - b0)            
                   - (self.alpha * z)
                   - (self.theta * y)               
                  ) / self.tau_adapt
            
            b0 += db0 * self.dt
            b0 = np.maximum(b0, 0)
            
            rates_hist[:, t] = y
            b0_hist[:, t] = b0
            
        print(f"Simulation complete in {time.time() - t0:.2f}s.")
        return rates_hist, b0_hist

if __name__ == "__main__":
    
    # 1. Initialize
    tunings = V1Tunings(N=180, kappa_exc=15.0, kappa_norm=0.05)
    stim_gen = StimulusGenerator(N=180)
    engine = V1Dynamics(tunings, dt=1.0, tau_adapt=100.0) 
    
    # 2. Define Experiment
    regimes = [
        {'n_steps': 300, 'contrast': 1.0, 'orientation': np.pi/2, 'label': 'Adapt (90°)'},
        {'n_steps': 300, 'contrast': 0.2, 'orientation': np.pi/2, 'label': 'Test (90°)'},
        {'n_steps': 300, 'contrast': 0.4, 'orientation': 0.0,     'label': 'Control (0°)'},
    ]
    inputs = stim_gen.generate_sequence(regimes)
    
    # 3. Run
    rates, gains = engine.run_simulation(inputs)
    
    # --- PLOT 1: Overview ---
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True, 
                             gridspec_kw={'height_ratios': [1, 2, 2]})
    
    axes[0].imshow(inputs, aspect='auto', cmap='binary', origin='lower')
    axes[0].set_ylabel("Pref")
    axes[0].set_title("(A) Input Drive ($z$)")
    axes[0].tick_params(left=False, labelleft=False) 
    
    im_r = axes[1].imshow(rates, aspect='auto', cmap='inferno', origin='lower', 
                          vmax=np.percentile(rates, 99.5))
    axes[1].set_ylabel("Neuron Pref (Deg)")
    axes[1].set_title("(B) Firing Rates ($y$)")
    plt.colorbar(im_r, ax=axes[1], label="Hz", fraction=0.046, pad=0.04)
    
    im_b = axes[2].imshow(gains, aspect='auto', cmap='coolwarm', origin='lower', 
                          vmin=0.0, vmax=0.4)
    axes[2].set_ylabel("Neuron Pref (Deg)")
    axes[2].set_xlabel("Time (ms)")
    axes[2].set_title(f"(C) Input Gain ($b_0$)")
    plt.colorbar(im_b, ax=axes[2], label="Gain", fraction=0.046, pad=0.04)
    
    t_cursor = 0
    for r in regimes:
        t_cursor += r['n_steps']
        for ax in axes: ax.axvline(t_cursor, color='white', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

    # --- PLOT 2: Transient Response (3 Neurons) ---
    zoom_steps = 100
    idx_stim = 90  # Peak 
    idx_flank = 45 # Flank 
    idx_orth = 0   # Orthogonal 

    fig2, ax2 = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
    ax2[0].plot(range(zoom_steps), rates[idx_stim, :zoom_steps], color='crimson', lw=2)
    ax2[0].set_title(f"Peak Driven Neuron ({idx_stim}°)")
    ax2[0].set_ylabel("Hz")
    ax2[0].grid(True, alpha=0.3)

    ax2[1].plot(range(zoom_steps), rates[idx_flank, :zoom_steps], color='forestgreen', lw=2)
    ax2[1].set_title(f"Flank Neuron ({idx_flank}°)")
    ax2[1].set_ylabel("Hz")
    ax2[1].grid(True, alpha=0.3)

    ax2[2].plot(range(zoom_steps), rates[idx_orth, :zoom_steps], color='navy', lw=2)
    ax2[2].set_title(f"Orthogonal Neuron ({idx_orth}°)")
    ax2[2].set_ylabel("Hz")
    ax2[2].set_xlabel("Time (ms)")
    ax2[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

    # --- PLOT 3: Steady State Tuning Curves ---
    # Calculating average response for the last 100 steps of each regime
    
    plt.figure(figsize=(8, 5))
    
    t_cursor = 0
    colors = ['#d62728', '#ff7f0e', '#2ca02c'] # Red, Orange, Green
    
    for i, r in enumerate(regimes):
        t_start = t_cursor
        t_end = t_start + r['n_steps']
        
        # Extract last 100 steps of this regime
        window_slice = rates[:, t_end-100 : t_end]
        avg_profile = np.mean(window_slice, axis=1)
        
        plt.plot(tunings.theta * 180 / np.pi, avg_profile, 
                 color=colors[i], linewidth=2.5, label=f"Regime {i+1}: {r['label']}")
        
        # Advance cursor
        t_cursor += r['n_steps']

    plt.title("Population Tuning Curves (Steady State)")
    plt.xlabel("Neuron Preference (Degrees)")
    plt.ylabel("Avg Firing Rate (Hz)")
    plt.xlim(0, 180)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()