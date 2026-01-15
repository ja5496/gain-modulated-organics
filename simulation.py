import numpy as np
import matplotlib.pyplot as plt
import time

# --- 1. Dependencies (Tunings & Stimuli) ---

class V1Tunings:
    def __init__(self, N=180, kappa_exc=15.0, kappa_norm=4.0):
        self.N = N
        self.theta = np.linspace(0, np.pi, N, endpoint=False)
        self.b0 = np.ones(N) * 0.2  # Initialize at your target_b0
        
        # Initialize Kernels
        self.W_rec = self._make_dist_weights(kappa_exc, normalize_by='max')
        # Use SUM normalization for W_norm so the decorrelation drive is stable
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
    def __init__(self, v1_model, dt=1.0, tau_v=1.0, tau_a=2.0, tau_u=1.0, tau_adapt=200):
        """
        ORGaNICs Dynamics Engine.
        """
        self.v1 = v1_model
        self.dt = dt
        self.tau_v = tau_v
        self.tau_a = tau_a
        self.tau_u = tau_u
        self.tau_adapt = tau_adapt
        
        # YOUR CUSTOM PARAMETERS
        self.alpha = 0.15          # Local Fatigue
        self.beta = 0.01           # Global Homeostasis (Whitening)
        self.target_b0 = 0.2       # Resting gain (Low baseline)
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
            
            # v: Excitatory Potential (Divisive Norm)
            recurrent_drive = y_hat / (1.0 + a)
            dv = (-v + input_drive + recurrent_drive) / self.tau_v
            v += dv * self.dt
            
            # b0: Adaptation (Whitening Logic)
            decorrelation_drive = self.v1.W_norm @ y
            
            db0 = ((self.target_b0 - b0)            # Restoration
                   - (self.alpha * y)               # Local Fatigue
                   - (self.beta * decorrelation_drive) # Lateral Whitening
                  ) / self.tau_adapt
            
            b0 += db0 * self.dt
            b0 = np.maximum(b0, 0)
            
            rates_hist[:, t] = y
            b0_hist[:, t] = b0
            
        print(f"Simulation complete in {time.time() - t0:.2f}s.")
        return rates_hist, b0_hist

if __name__ == "__main__":
    
    # 1. Initialize
    tunings = V1Tunings(N=180, kappa_exc=15.0, kappa_norm=4.0)
    stim_gen = StimulusGenerator(N=180)
    
    # We pass tau_adapt=1000.0 here to see the slow history effects in the plot
    engine = V1Dynamics(tunings, dt=1.0, tau_adapt=200.0) 
    
    # 2. Define Experiment
    regimes = [
        {'n_steps': 200, 'contrast': 1.0, 'orientation': np.pi/2}, # Adapt 90
        {'n_steps': 100, 'contrast': 0.2, 'orientation': np.pi/2}, # Test 90
        {'n_steps': 200, 'contrast': 0.6, 'orientation': 0.0},     # Control 0
    ]
    inputs = stim_gen.generate_sequence(regimes)
    
    # 3. Run
    rates, gains = engine.run_simulation(inputs)
    
    # 4. Compact Plotting (Main Overview)
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
                          vmin=0.0, vmax=0.6)
    axes[2].set_ylabel("Neuron Pref (Deg)")
    axes[2].set_xlabel("Time (ms)")
    axes[2].set_title("(C) Input Gain ($b_0$) | Rest = 0.2")
    plt.colorbar(im_b, ax=axes[2], label="Gain", fraction=0.046, pad=0.04)
    
    t = 0
    for r in regimes:
        t += r['n_steps']
        for ax in axes: ax.axvline(t, color='white', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

    # --- 5. NEW PLOT: Two Neurons (First 100 Steps) ---
    
    # Settings
    zoom_steps = 100
    idx_stim = 90  # Neuron preferred 90 deg (Driven)
    idx_orth = 0   # Neuron preferred 0 deg (Not Driven)

    # Setup Figure with 2 Subplots
    fig2, ax2 = plt.subplots(2, 1, figsize=(7, 6), sharex=True)

    # Subplot 1: The Driven Neuron (90 deg)
    ax2[0].plot(range(zoom_steps), rates[idx_stim, :zoom_steps], 
                color='crimson', linewidth=2, label=f"Neuron {idx_stim}°")
    ax2[0].set_title(f"Driven Neuron ({idx_stim}°): Onset Transient")
    ax2[0].set_ylabel("Firing Rate ($y$)")
    ax2[0].grid(True, alpha=0.3)
    ax2[0].legend(loc='upper right')

    # Subplot 2: The Orthogonal Neuron (0 deg)
    ax2[1].plot(range(zoom_steps), rates[idx_orth, :zoom_steps], 
                color='navy', linewidth=2, label=f"Neuron {idx_orth}°")
    ax2[1].set_title(f"Orthogonal Neuron ({idx_orth}°): Suppression Check")
    ax2[1].set_ylabel("Firing Rate ($y$)")
    ax2[1].set_xlabel("Time (ms)")
    ax2[1].grid(True, alpha=0.3)
    
    # Force y-axis to be small for the second plot so we can see noise
    # (Or keep auto if activity is high)
    # ax2[1].set_ylim(-0.01, 0.1) 

    ax2[1].legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()