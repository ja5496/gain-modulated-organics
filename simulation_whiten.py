import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm
import time

# --- 1. Frame Construction (Smoothed) ---

class Frame:
    def __init__(self, dim: int):
        self.dim = int(dim)
        # N=60 -> K=1830
        self.K = int(self.dim * (self.dim + 1) // 2)
        print(f"Building Smooth Mercedes Frame (N={self.dim}, K={self.K})...")
        self.W = self.mercedes()
        
        # --- REVERTED: Initialize gains at 0.0 ---
        self.g = np.zeros(self.K) 
    
    def mercedes(self) -> np.ndarray:
        N, K = self.dim, self.K
        
        # 1. Generate Random Vectors
        A = np.random.randn(5 * K, N)
        
        # 2. SMOOTH the vectors
        for i in range(A.shape[0]):
            A[i, :] = gaussian_filter1d(A[i, :], sigma=2.0, mode='wrap')
            
        A /= np.linalg.norm(A, axis=1, keepdims=True)
        
        # 3. Greedy Selection
        W_list = [A[0]]
        W_curr = A[0][:, None]
        remaining_A = A[1:]
        
        for _ in tqdm(range(K - 1), desc="Frame Init", leave=False):
            dots = remaining_A @ W_curr
            coherences = np.max(np.abs(dots), axis=1)
            best_idx = np.argmin(coherences)
            
            vec = remaining_A[best_idx]
            W_list.append(vec)
            W_curr = np.column_stack([W_curr, vec])
            remaining_A = np.delete(remaining_A, best_idx, axis=0)
            
        return np.array(W_list).T

# --- 2. Dependencies ---

class V1Tunings:
    def __init__(self, N=60, kappa_exc=8.0, kappa_norm=0.1): 
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
            profile = np.exp(6.0 * np.cos(2*(self.theta - r['orientation'])))
            profile = profile / np.max(profile) * r['contrast']
            seq.append(np.tile(profile, (r['n_steps'], 1)).T)
        return np.hstack(seq)

# --- 3. Dynamics Engine ---

class V1DynamicsCustom:
    def __init__(self, v1_model, frame, dt=0.05, tau_g=200.0):
        self.v1 = v1_model
        self.frame = frame
        self.dt = dt 
        
        self.tau_y = 1.0
        self.tau_a = 5.0
        self.tau_u = 2.0
        self.tau_q = 1.0
        self.tau_g = tau_g
        
        self.beta = 1.0       
        self.sigma = 0.05     
        self.alpha = 0.0
        self.g_decay = 0.02   
        
        self.target_energy = 0.05 
        
    def _derivatives(self, state, z_t):
        N, K = self.v1.N, self.frame.K
        
        y = state[0:N]
        u = state[N:2*N]
        a = state[2*N:3*N]
        q = state[3*N:4*N]
        g = state[4*N:4*N+K]
        
        # Stability: Gains must be non-negative
        g = np.maximum(g, 0.0)
        
        y_pos = np.maximum(y, 0)
        u_pos = np.maximum(u, 0)
        sqrt_y_pos = np.sqrt(y_pos)
        sqrt_u_pos = np.sqrt(u_pos)
        
        v_t = self.frame.W.T @ y
        
        gain_feedback = self.frame.W @ (g * v_t)
        recurrent_drive = (1.0 / (1.0 + np.maximum(a, 0))) * (self.v1.W_yy @ sqrt_y_pos)
        input_drive = self.beta * z_t 
        
        dy_dt = (-y + input_drive + recurrent_drive - gain_feedback) / self.tau_y
        
        sigma_term = (self.sigma) ** 2
        pool_term = self.v1.N_matrix @ (y_pos * (u_pos ** 2))
        du_dt = (-u + sigma_term + pool_term) / self.tau_u
        da_dt = (-a + sqrt_u_pos + a * sqrt_u_pos + self.alpha * du_dt) / self.tau_a
        
        dq_dt = (-q + sqrt_y_pos) / self.tau_q
        dg_dt = (v_t * v_t - self.target_energy - self.g_decay * g) / self.tau_g
        
        return np.concatenate([dy_dt, du_dt, da_dt, dq_dt, dg_dt])
        
    def run_simulation(self, stimulus_stream):
        N, n_steps = stimulus_stream.shape
        K = self.frame.K
        
        state = np.zeros(4*N + K)
        
        # --- REVERTED: Start gains at 0.0 ---
        state[4*N:] = 0.0 
        
        rates_hist = np.zeros((N, n_steps))
        gains_hist = np.zeros((K, n_steps))
        
        print(f"Running Stabilized Simulation (Initial Gains=0.0)...")
        t0 = time.time()
        
        for t in tqdm(range(n_steps)):
            z_t = stimulus_stream[:, t]
            
            # RK4
            k1 = self._derivatives(state, z_t)
            k2 = self._derivatives(state + 0.5 * self.dt * k1, z_t)
            k3 = self._derivatives(state + 0.5 * self.dt * k2, z_t)
            k4 = self._derivatives(state + self.dt * k3, z_t)
            
            state += (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            
            # Clamp gains [0, 25]
            state[4*N:] = np.clip(state[4*N:], 0.0, 25.0)
            
            y = state[0:N]
            g = state[4*N:]
            
            rates_hist[:, t] = np.maximum(y, 0)
            gains_hist[:, t] = g
            
        print(f"Simulation complete in {time.time() - t0:.2f}s.")
        return rates_hist, gains_hist


if __name__ == "__main__":
    
    # 1. Initialize
    N_NEURONS = 60
    tunings = V1Tunings(N=N_NEURONS, kappa_exc=8.0, kappa_norm=0.1)
    frame = Frame(dim=N_NEURONS)
    stim_gen = StimulusGenerator(N=N_NEURONS)
    
    engine = V1DynamicsCustom(tunings, frame, dt=0.05, tau_g=200.0)
    
    # 2. Define Experiment
    regimes = [
        {'n_steps': 5000, 'contrast': 0.75, 'orientation': np.pi/2, 'label': 'Adapt (90°)'},
        {'n_steps': 5000, 'contrast': 0.2, 'orientation': np.pi/2, 'label': 'Test (90°)'},
        {'n_steps': 5000, 'contrast': 0.5, 'orientation': 0.0,     'label': 'Control (0°)'},
    ]
    inputs = stim_gen.generate_sequence(regimes)
    
    # 3. Run
    rates, gains = engine.run_simulation(inputs)
    
    # --- PLOTTING ---
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True, 
                             gridspec_kw={'height_ratios': [1, 2, 2]})
    
    axes[0].imshow(inputs, aspect='auto', cmap='binary', origin='lower')
    axes[0].set_ylabel("Pref")
    axes[0].set_title("(A) Input Drive ($z$)")
    axes[0].tick_params(left=False, labelleft=False) 
    
    vmax = np.percentile(rates, 99.5)
    im_r = axes[1].imshow(rates, aspect='auto', cmap='inferno', origin='lower', 
                          vmax=vmax)
    axes[1].set_ylabel("Neuron Pref (Deg)")
    axes[1].set_title(f"(B) Firing Rates ($y$)")
    plt.colorbar(im_r, ax=axes[1], label="Hz", fraction=0.046, pad=0.04)
    
    subset_k = np.linspace(0, frame.K-1, 10, dtype=int)
    axes[2].plot(gains[subset_k, :].T, alpha=0.6)
    axes[2].set_ylabel("Gains (subset)")
    axes[2].set_xlabel("Time (steps)")
    axes[2].set_title(f"(C) Gain Evolution")
    axes[2].grid(True, alpha=0.3)
    
    t_cursor = 0
    for r in regimes:
        t_cursor += r['n_steps']
        for ax in axes: ax.axvline(t_cursor, color='white', linestyle='--', alpha=0.5)

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