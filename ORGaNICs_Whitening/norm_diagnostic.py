import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from tunings_whiten import V1Tunings
from stimuli_whiten import StimulusGenerator
from scipy.special import erf
from scipy.linalg import block_diag

class Norm_Dynamics_1:
    def __init__(self, v1_model, dt=0.1):
        self.v1 = v1_model
        self.dt = dt 

        self.tau_y = 0.2
        self.tau_a = 0.1  
        self.tau_u = 15.0
        
        self.sigma = 0.25
        self.beta = 0.5

    def half_wave_rectify(self, y, Beta=2.0):
        return (np.maximum(y,0)) ** Beta

    def _derivatives(self, state, z_t):
        N = self.v1.N
        
        y = state[0:N]
        u = state[N:2*N]
        a = state[2*N:3*N]

        y_plus = self.half_wave_rectify(y, 2.0)
        y_minus = self.half_wave_rectify(-y, 2.0)           
        u_plus = self.half_wave_rectify(u, 0.5)
        a_plus = self.half_wave_rectify(a, 1.0)
        sqrt_y_plus = np.sqrt(y_plus) 
        sqrt_y_minus = np.sqrt(y_minus) 
        

        recurrent_drive = (1.0 / (1.0 + a_plus)) * (self.v1.W_yy @ (sqrt_y_plus)) 
        input_drive = self.beta * z_t
        
        sigma_term = (self.sigma / 2) ** 2
        pool_term = self.v1.N_matrix @ ((y_plus) *(u_plus ** 2))
        
        # ORGaNICs equations taken from Asit's Heirarchical Model (with gain feedback)
        dy_dt = (-y + input_drive + recurrent_drive) / self.tau_y
        du_dt = (-u + sigma_term + pool_term) / self.tau_u
        da_dt = (-a + u_plus * (1 + a_plus)) / self.tau_a
        
        return np.concatenate([dy_dt, du_dt, da_dt])
        
    def run_simulation(self, stimulus_stream, initial_state=None):
        N, n_steps = stimulus_stream.shape

        if initial_state is not None:
            state = initial_state.copy()
        else:
            state = np.zeros(3*N)

        # Tracking histories for later analysis + figures
        y_hist = np.zeros((N, n_steps))
        u_hist = np.zeros((N, n_steps))
        a_hist = np.zeros((N, n_steps))
        
        print(f"Running Simulation ({n_steps} steps)...") 
        t0 = time.time()
        
        for t in tqdm(range(n_steps)):
            z_t = stimulus_stream[:, t] 
            # RK4 Simulation
            k1 = self._derivatives(state, z_t)
            k2 = self._derivatives(state + 0.5 * self.dt * k1, z_t)
            k3 = self._derivatives(state + 0.5 * self.dt * k2, z_t)
            k4 = self._derivatives(state + self.dt * k3, z_t)
            
            state += (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            
            
            y_hist[:, t] = np.maximum(state[0:N], 0)
            u_hist[:, t] = state[N:2*N]
            a_hist[:, t] = state[2*N:3*N]


        print(f"Simulation complete in {time.time() - t0:.2f}s.")
        self.last_state = state.copy()
        return y_hist, u_hist, a_hist


# ==========================================================================
# Contrast response function of the isolated normalization model
# ==========================================================================
N_RF         = 91     # primary neurons (single population, no surround)
TUNING_WIDTH = 0.75
PROBE_THETA  = 0.0    # orientation the test stimuli (and probed neuron) are centered on

N_SETTLE_STEPS = 300  # timesteps to settle y/u/a to steady state per probe (dt=0.1 -> 30s)

N_CONTRASTS = 20
CONTRASTS   = np.logspace(-2, 0, N_CONTRASTS)   # log-spaced contrasts in (0, 1] - can't include
                                                 # the zero vector on a log axis


def probe_input_drive(input_theta, contrast, N=N_RF, tuning_width=TUNING_WIDTH):
    '''
    Gaussian tuning-curve profile over N neurons, centered at input_theta, normalized to
    unit length and then scaled by contrast - so the input vector has ||z|| = contrast,
    ranging from the zero vector (contrast=0) to the unit-length profile (contrast=1).
    '''
    theta_grid = np.linspace(0, np.pi, N, endpoint=False)
    delta = theta_grid - input_theta
    delta = (delta + np.pi / 2) % np.pi - np.pi / 2
    profile = np.exp(-delta**2 / (2 * tuning_width**2))
    return contrast * profile / np.linalg.norm(profile)


def get_response(dyn, stimulus, n_steps=N_SETTLE_STEPS):
    '''Settles (y, u, a) to steady state from a zero initial state for a fixed stimulus.'''
    N = dyn.v1.N
    dt = dyn.dt
    state = np.zeros(3 * N)
    for _ in range(n_steps):
        k1 = dyn._derivatives(state, stimulus)
        k2 = dyn._derivatives(state + 0.5 * dt * k1, stimulus)
        k3 = dyn._derivatives(state + 0.5 * dt * k2, stimulus)
        k4 = dyn._derivatives(state + dt * k3, stimulus)
        state += (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    return np.maximum(state[0:N], 0)


N_TRACE_STEPS   = 1000                          # timesteps to visualize the settling trajectory
TRACE_CONTRASTS = [0.10, 0.25, 0.45, 0.75]     # one contrast per panel


def get_response_trace(dyn, stimulus, n_steps=N_TRACE_STEPS):
    '''Same RK4 settle loop as get_response, but returns the full y trajectory (N x n_steps)
    from a zero initial state instead of just the final value - shows *how* (or whether) the
    dynamics actually settle rather than just where they happen to land.'''
    N = dyn.v1.N
    dt = dyn.dt
    state = np.zeros(3 * N)
    y_hist = np.zeros((N, n_steps))
    for t in range(n_steps):
        k1 = dyn._derivatives(state, stimulus)
        k2 = dyn._derivatives(state + 0.5 * dt * k1, stimulus)
        k3 = dyn._derivatives(state + 0.5 * dt * k2, stimulus)
        k4 = dyn._derivatives(state + dt * k3, stimulus)
        state += (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        y_hist[:, t] = np.maximum(state[0:N], 0)

    return y_hist


if __name__ == "__main__":
    import os
    REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
    FRAME_PATH = os.path.join(REPO_ROOT, "data/frames/N13_mercedes_Frame.csv")

    class Frame:
        '''Minimal stand-in - Norm_Dynamics only reads frame.K, unused in its (isolated,
        gain-feedback-free) _derivatives, so nothing beyond dim/K needs to be real here.'''
        def __init__(self, csv_path):
            W = np.loadtxt(csv_path, delimiter=",")
            self.dim, self.K = W.shape

    print("Initializing tunings, frame, and isolated normalization dynamics...")
    tunings = V1Tunings(N=N_RF)
    dyn     = Norm_Dynamics_1(tunings)

    target_idx = N_RF // 2   # index of the neuron whose preference == PROBE_THETA

    print("Computing contrast response function...")
    responses = np.zeros(N_CONTRASTS)
    for i, c in enumerate(tqdm(CONTRASTS)):
        probe = probe_input_drive(PROBE_THETA, c)
        y = get_response(dyn, probe)
        responses[i] = y[target_idx]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(CONTRASTS, responses, color='black', linewidth=3.5)
    ax.plot(CONTRASTS, responses, 'o', color='black', markersize=4)

    # Half-saturation contrast -- interpolated in log-contrast space between the two
    # samples straddling half of the curve's max (mirrors Surround_simulated_responses.py).
    half_max = responses.max() / 2.0
    idx = np.argmax(responses >= half_max)
    if idx == 0:
        c50 = CONTRASTS[0]
    else:
        log_c_lo, log_c_hi = np.log10(CONTRASTS[idx - 1]), np.log10(CONTRASTS[idx])
        r_lo, r_hi = responses[idx - 1], responses[idx]
        log_c50 = log_c_lo + (half_max - r_lo) * (log_c_hi - log_c_lo) / (r_hi - r_lo)
        c50 = 10 ** log_c50

    ax.axvline(c50, color='black', linestyle='--', linewidth=1.5, alpha=0.8)
    ax.plot(c50, half_max, 'o', color='red', markersize=8, zorder=5)
    ax.text(c50, 0.95, f"c50 = {c50:.3g}", transform=ax.get_xaxis_transform(),
            fontsize=11, fontweight='bold', color='black', ha='center', va='top')

    ax.set_xscale('log')
    ax.set_title("Contrast Response Function (Isolated Normalization Model)",
                 fontsize=14, fontweight='bold')
    ax.set_xlabel("Contrast (input vector length)", fontsize=14, fontweight='bold')
    ax.set_ylabel("Steady-state response", fontsize=14, fontweight='bold')
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', width=2.5, length=6, labelsize=12)
    plt.tight_layout()

    # ==========================================================================
    # Four-panel settling traces: activity vs. time step at a few example contrasts
    # ==========================================================================
    print("Computing settling traces at four example contrasts...")
    fig_trace, axes_trace = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    time_steps = np.arange(N_TRACE_STEPS)

    for ax, c in zip(axes_trace.flat, TRACE_CONTRASTS):
        probe = probe_input_drive(PROBE_THETA, c)
        y_hist = get_response_trace(dyn, probe)

        for n in range(N_RF):
            ax.plot(time_steps, y_hist[n], color='lightgray', linewidth=1.0)
        ax.plot(time_steps, y_hist[target_idx], color='black', linewidth=2.5)

        ax.set_title(f"Contrast = {c:.2f}", fontsize=13, fontweight='bold')
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=10)

    for ax in axes_trace[-1]:
        ax.set_xlabel("Time step", fontsize=12, fontweight='bold')
    for ax in axes_trace[:, 0]:
        ax.set_ylabel("Activity (y)", fontsize=12, fontweight='bold')

    fig_trace.suptitle("Settling dynamics of the isolated normalization model\n"
                        "(black = target neuron, gray = other 12 neurons)",
                        fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

