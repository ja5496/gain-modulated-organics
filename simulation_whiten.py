import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from tqdm import tqdm
import time

# --- 1. Frame Construction ---

class Frame:
    """Overcomplete frame for gain-modulated whitening"""
    def __init__(self, dim: int):
        self.dim = int(dim)
        self.K = int(self.dim * (self.dim + 1) // 2)
        self.W = self.mercedes()
        self.g = np.ones(self.K)
    
    def mercedes(self) -> np.ndarray:
        """Build overcomplete set of K unit vectors via greedy selection"""
        N, K = self.dim, self.K
        A = np.random.randn(5 * K, N)
        A /= np.linalg.norm(A, axis=1, keepdims=True)
        
        w_1 = A[0]
        A = np.delete(A, 0, axis=0)
        W = np.stack([w_1], axis=1)
        
        for _ in tqdm(range(K - 1), desc="Building frame", leave=False):
            cos = A @ W
            closest = np.max(np.abs(cos), axis=1)
            idx = np.argmin(closest)
            W = np.column_stack([W, A[idx]])
            A = np.delete(A, idx, axis=0)
        
        W /= np.linalg.norm(W, axis=0, keepdims=True)
        return W

# --- 2. Dependencies (Tunings & Stimuli) ---

class V1Tunings:
    def __init__(self, N=60, kappa_exc=45.0, kappa_norm=0.15):
        self.N = N
        self.theta = np.linspace(0, np.pi, N, endpoint=False)
        self.W_yy = self._make_dist_weights(kappa_exc, normalize_by='max') * 0.15
        self.N_matrix = self._make_dist_weights(kappa_norm, normalize_by='sum')

    def _make_dist_weights(self, kappa, normalize_by='max'):
        W = np.zeros((self.N, self.N))
        for i in range(self.N):
            d = np.abs(self.theta - self.theta[i])
            d = np.minimum(d, np.pi - d)
            W[i, :] = np.exp(kappa * np.cos(2*d))
        if normalize_by == 'sum':
            return W / np.sum(W, axis=1, keepdims=True)
        return W / np.max(W)

class StimulusGenerator:
    def __init__(self, N=60):
        self.N = N
        self.theta = np.linspace(0, np.pi, N, endpoint=False)
        
    def generate_sequence(self, regimes):
        seq = []
        for r in regimes:
            profile = np.exp(24.0 * np.cos(2*(self.theta - r['orientation'])))
            profile = profile / np.max(profile) * r['contrast']
            seq.append(np.tile(profile, (r['n_steps'], 1)).T)
        return np.hstack(seq)

# --- 3. Main Dynamics Engine with scipy.integrate ---

class V1DynamicsScipy:
    def __init__(self, v1_model, frame, dt=1.0, tau_y=1.0, tau_a=2.0, 
                 tau_u=1.0, tau_q=1.0, tau_g=10.0):
        self.v1 = v1_model
        self.frame = frame
        self.dt = dt
        self.tau_y = tau_y
        self.tau_a = tau_a
        self.tau_u = tau_u
        self.tau_q = tau_q
        self.tau_g = tau_g
        self.beta = 0.2
        self.sigma = 0.1
        self.alpha = 0.0
        
        # Precompute dimensions
        self.N = v1_model.N
        self.K = frame.K
        
    def _pack_state(self, y, u, a, q, g):
        """Pack state variables into single vector"""
        return np.concatenate([y, u, a, q, g])
    
    def _unpack_state(self, state):
        """Unpack state vector into components"""
        N, K = self.N, self.K
        y = state[0:N]
        u = state[N:2*N]
        a = state[2*N:3*N]
        q = state[3*N:4*N]
        g = state[4*N:4*N+K]
        return y, u, a, q, g
    
    def _derivatives(self, t, state, z_interp):
        """Compute derivatives for solve_ivp"""
        y, u, a, q, g = self._unpack_state(state)
        
        # Get interpolated input at time t
        z_centered = z_interp(t)
        
        # Rectification
        y_pos = np.maximum(y, 0)
        u_pos = np.maximum(u, 0)
        sqrt_y_pos = np.sqrt(y_pos)
        sqrt_u_pos = np.sqrt(u_pos)
        
        # Projection onto frame
        v_t = self.frame.W.T @ y
        
        # dy/dt
        input_drive = (self.beta / 2.0) * z_centered
        recurrent_drive = (1.0 / (1.0 + np.maximum(a, 0))) * (self.v1.W_yy @ sqrt_y_pos)
        gain_feedback = self.frame.W @ (g * v_t)
        dy_dt = (-y + input_drive + recurrent_drive - gain_feedback) / self.tau_y
        
        # du/dt
        sigma_term = (self.sigma / 2.0) ** 2
        pool_term = self.v1.N_matrix @ (y_pos * (u_pos ** 2))
        du_dt = (-u + sigma_term + pool_term) / self.tau_u
        
        # da/dt
        da_dt = (-a + sqrt_u_pos + a * sqrt_u_pos + self.alpha * du_dt) / self.tau_a
        
        # dq/dt
        dq_dt = (-q + sqrt_y_pos) / self.tau_q
        
        # dg/dt
        dg_dt = (v_t * v_t - 1.0) / self.tau_g
        
        return self._pack_state(dy_dt, du_dt, da_dt, dq_dt, dg_dt)
        
    def run_simulation(self, stimulus_stream):
        N, n_steps = stimulus_stream.shape
        K = self.K
        
        # Center inputs
        z_mean = np.mean(stimulus_stream)
        stimulus_centered = stimulus_stream - z_mean
        
        # Create interpolator for continuous-time input
        t_points = np.arange(n_steps)
        def z_interp(t):
            idx = int(np.clip(t, 0, n_steps - 1))
            return stimulus_centered[:, idx]
        
        # Initial state
        y0 = np.zeros(N)
        u0 = np.zeros(N)
        a0 = np.zeros(N)
        q0 = np.zeros(N)
        g0 = self.frame.g.copy()
        state0 = self._pack_state(y0, u0, a0, q0, g0)
        
        # History storage
        rates_hist = np.zeros((N, n_steps))
        gains_hist = np.zeros((K, n_steps))
        
        print(f"Running ORGaNICs+Whitening simulation ({n_steps} steps, K={K})...")
        t0 = time.time()
        
        # Integrate step-by-step with progress bar
        current_state = state0
        for t in tqdm(range(n_steps), desc="Simulating", unit="step"):
            # Solve for one time step
            sol = solve_ivp(
                fun=lambda t_inner, s: self._derivatives(t_inner, s, z_interp),
                t_span=(t, t + self.dt),
                y0=current_state,
                method='RK45',
                t_eval=[t + self.dt],
                rtol=1e-6,
                atol=1e-8
            )
            
            current_state = sol.y[:, -1]
            y, u, a, q, g = self._unpack_state(current_state)
            
            # Store (use y^2 for output)
            y_rect = np.maximum(y, 0)
            rates_hist[:, t] = y_rect ** 2
            gains_hist[:, t] = g
            
        print(f"Simulation complete in {time.time() - t0:.2f}s.")
        return rates_hist, gains_hist


if __name__ == "__main__":
    
    # 1. Initialize (smaller network for faster testing)
    N_NEURONS = 60  # Reduced from 180 -> K = 1830 instead of 16110
    print("Initializing model components...")
    tunings = V1Tunings(N=N_NEURONS, kappa_exc=45.0, kappa_norm=0.15)
    frame = Frame(dim=N_NEURONS)
    stim_gen = StimulusGenerator(N=N_NEURONS)
    engine = V1DynamicsScipy(tunings, frame, dt=1.0, tau_g=10.0)
    
    # 2. Define Experiment (shorter regimes)
    regimes = [
        {'n_steps': 1000, 'contrast': 1.0, 'orientation': np.pi/2, 'label': 'Adapt (90°)'},
        {'n_steps': 1000, 'contrast': 0.2, 'orientation': np.pi/2, 'label': 'Test (90°)'},
        {'n_steps': 1000, 'contrast': 0.4, 'orientation': 0.0,     'label': 'Control (0°)'},
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
    axes[1].set_title("(B) Firing Rates ($y^2$)")
    plt.colorbar(im_r, ax=axes[1], label="Hz", fraction=0.046, pad=0.04)
    
    mean_gains = np.mean(gains, axis=0)
    axes[2].plot(mean_gains, color='blue')
    axes[2].set_ylabel("Mean Gain")
    axes[2].set_xlabel("Time (ms)")
    axes[2].set_title(f"(C) Mean Adaptive Gains (K={frame.K})")
    axes[2].grid(True, alpha=0.3)
    
    t_cursor = 0
    for r in regimes:
        t_cursor += r['n_steps']
        axes[0].axvline(t_cursor, color='white', linestyle='--', alpha=0.5)
        axes[1].axvline(t_cursor, color='white', linestyle='--', alpha=0.5)
        axes[2].axvline(t_cursor, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

    # --- PLOT 2: Transient Response (3 Neurons) ---
    zoom_steps = min(100, rates.shape[1])
    idx_stim = N_NEURONS // 2      # Peak
    idx_flank = N_NEURONS // 4     # Flank
    idx_orth = 0                   # Orthogonal

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
    plt.figure(figsize=(8, 5))
    
    t_cursor = 0
    colors = ['#d62728', '#ff7f0e', '#2ca02c']
    
    for i, r in enumerate(regimes):
        t_start = t_cursor
        t_end = t_start + r['n_steps']
        window_slice = rates[:, max(0, t_end-50) : t_end]
        avg_profile = np.mean(window_slice, axis=1)
        plt.plot(tunings.theta * 180 / np.pi, avg_profile, 
                 color=colors[i], linewidth=2.5, label=f"Regime {i+1}: {r['label']}")
        t_cursor += r['n_steps']

    plt.title("Population Tuning Curves (Steady State)")
    plt.xlabel("Neuron Preference (Degrees)")
    plt.ylabel("Avg Firing Rate (Hz)")
    plt.xlim(0, 180)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()