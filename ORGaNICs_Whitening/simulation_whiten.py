import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from tunings_whiten import V1Tunings
from stimuli_whiten import StimulusGenerator
from scipy.special import erf
from scipy.linalg import block_diag

'''
---- simulation_whiten.py ----

Stores RK4 Dynamics and simulation code for computational neural modeling of joint 
adaptation and normalization in V1. Dynamics are designed for orientation adaptation. 

'''

class Frame:

    '''  Loads W (Frame connecting primary neurons to interneurons) from a pre-computed 
    csv file. If an accompanying _centers.csv exists alongside the frame file, loads orientation
    centers (radians) for each frame vector into self.centers; otherwise None. '''

    def __init__(self, csv_path: str):
        import os
        print(f"Loading frame from {csv_path}...")
        self.W = np.loadtxt(csv_path, delimiter=",")
        self.dim = self.W.shape[0]
        self.K = self.W.shape[1]
        print(f"Loaded frame (N={self.dim}, K={self.K})")
        centers_path = csv_path.replace(".csv", "_centers.csv")
        if os.path.exists(centers_path):
            self.centers = np.loadtxt(centers_path, delimiter=",")
            print(f"Loaded orientation centers from {centers_path}")
        else:
            self.centers = None

class V1Dynamics:
    def __init__(self, v1_model, frame, dt=0.1, adaptive=True, input_adaptive=True):
        self.v1 = v1_model
        self.frame = frame
        self.dt = dt 
        self.adaptive = adaptive
        self.input_adaptive = input_adaptive

        self.tau_y = 0.2 
        self.tau_a = 1.0  
        self.tau_u = 0.4 
        self.tau_g = 100.0 
        self.tau_v = 100.0 # from 50
        self.tau_avg = 10 
        self.tau_avg_z = 400
        
        self.sigma = 0.1
        self.alpha = 0.0

        if input_adaptive:
            self.beta = None
        else:
            self.beta = 1.0

    def gaussian_rectify(self, y, threshold=0.6, sigma=0.35, r_max=1.0):
        # Rectification function (crudely) estimates firing rates from membrane potential
        return 0.5 * (1 + erf((y - threshold) / (sigma * np.sqrt(2)))) * r_max

    def _derivatives(self, state, z_t):
        N, K = self.v1.N, self.frame.K
        
        y = state[0:N]
        u = state[N:2*N]
        a = state[2*N:3*N]
        g = np.maximum(state[3*N:3*N+K], 0)
        v_state = state[3*N+K:3*N+2*K]
        avg_z = state[3*N+2*K:4*N+2*K]
        avg_vsq = state[4*N+2*K:4*N+2*K+1]
        
        
        u_plus = self.gaussian_rectify(u)
        y_plus = self.gaussian_rectify(y)
        a_plus = self.gaussian_rectify(a)
        sqrt_y_plus = np.sqrt(y_plus) 
        
        # avg_z tracks normalized input; updated independently of whitening gain adaptation
        z_min, z_max = z_t.min(), z_t.max()
        scaled_z_t = (z_t - z_min) / (z_max - z_min + 1e-8)
        if self.input_adaptive:
            davg_z_dt = (-avg_z + scaled_z_t) / self.tau_avg_z
        else:
            davg_z_dt = np.zeros(N)

        if self.adaptive:
            davg_vsq_dt = (-avg_vsq + np.mean(v_state * v_state)) / self.tau_avg # dynamics to calculate mean(v^2)
            dg_dt = (v_state * v_state - avg_vsq) / self.tau_g # target set to the recent average of v^2 (avg_vsq)
            dv_dt = (-v_state + self.frame.W.T @ y) / self.tau_v # dynamics converge to whitening objective
            gain_feedback = self.frame.W @ (g * v_state)
        else:
            gain_feedback = 0.0
            dg_dt = np.zeros(K)
            dv_dt = np.zeros(K)
            davg_vsq_dt = np.zeros(1)

        recurrent_drive = (1.0 / (1.0 + a_plus)) * (self.v1.W_yy @ sqrt_y_plus)
        if self.input_adaptive:
            beta = 1 - 0.4 * avg_z  # Common stimuli are less effective at driving the cortex
        else:
            beta = self.beta

        input_drive = (beta * z_t) / 2
        
        sigma_term = (self.sigma / 2) ** 2
        pool_term = self.v1.N_matrix @ (y_plus * (u_plus ** 2))
        
        # ORGaNICs equations taken from Asit's Heirarchical Model (with gain feedback)
        dy_dt = (-y + input_drive + recurrent_drive - gain_feedback) / self.tau_y
        du_dt = (-u + sigma_term + pool_term) / self.tau_u
        da_dt = (-a + u_plus + a * u_plus + self.alpha * du_dt) / self.tau_a
        
        return np.concatenate([dy_dt, du_dt, da_dt, dg_dt, dv_dt, davg_z_dt, davg_vsq_dt])
        
    def run_simulation(self, stimulus_stream, initial_state=None):
        N, n_steps = stimulus_stream.shape
        K = self.frame.K

        if initial_state is not None:
            state = initial_state.copy()
        else:
            state = np.zeros(4*N + 2*K + 1)
            state[4*N + 2*K] = 1.0  # initialize avg_vsq to a non-zero baseline

        # Tracking histories for later analysis + figures
        y_hist = np.zeros((N, n_steps))
        gains_hist = np.zeros((K, n_steps))
        u_hist = np.zeros((N, n_steps))
        a_hist = np.zeros((N, n_steps))
        v_hist = np.zeros((K, n_steps))
        avg_z_hist = np.zeros((N, n_steps))
        avg_vsq_hist = np.zeros(n_steps)
        
        mode_str = "Adaptive" if self.adaptive else "Non-Adaptive"
        print(f"Running {mode_str} Simulation ({n_steps} steps)...") 
        t0 = time.time()
        
        for t in tqdm(range(n_steps)):
            z_t = stimulus_stream[:, t] 
            # RK4 Simulation
            k1 = self._derivatives(state, z_t)
            k2 = self._derivatives(state + 0.5 * self.dt * k1, z_t)
            k3 = self._derivatives(state + 0.5 * self.dt * k2, z_t)
            k4 = self._derivatives(state + self.dt * k3, z_t)
            
            state += (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            
            state[3*N:3*N+K] = np.maximum(state[3*N:3*N+K], 0)
            
            y_hist[:, t] = np.maximum(state[0:N], 0)
            u_hist[:, t] = state[N:2*N]
            a_hist[:, t] = state[2*N:3*N]
            gains_hist[:, t] = state[3*N:3*N+K]
            v_hist[:, t] = state[3*N+K:3*N+2*K]
            avg_z_hist[:, t] = state[3*N+2*K:4*N+2*K]
            avg_vsq_hist[t] = state[4*N+2*K]

        print(f"Simulation complete in {time.time() - t0:.2f}s.")
        self.last_state = state.copy()
        return y_hist, gains_hist, u_hist, a_hist, v_hist, avg_z_hist, avg_vsq_hist

class V1Dynamics_Surround:
    def __init__(self, v1_model, frame, dt=0.1, N_RF = 13, N_SETS = 7,
                 target_covariance_path="data/target_covs/uniform_target_covariance.csv"):
        self.v1 = v1_model
        self.frame = frame
        self.dt = dt
        self.N_RF = N_RF
        self.N_SETS = N_SETS
        N_TOT = N_RF * N_SETS

        # v1_model.W_yy is the single-location (N_RF, N_RF) recurrent matrix; tile it into
        # the block-diagonal (N_TOT, N_TOT) matrix needed now that y spans all N_SETS
        # locations (recurrence stays local to each location, unlike the pooling below).
        assert v1_model.W_yy.shape == (N_RF, N_RF), (
            f"v1_model.W_yy must be the single-location ({N_RF}, {N_RF}) matrix - "
            f"construct V1Tunings with N=N_RF (got shape {v1_model.W_yy.shape})."
        )
        self.W_yy = block_diag(*[v1_model.W_yy] * N_SETS)
        # Normalization pool spans cRF and surround: every one of the N_TOT neurons
        # pools together, unlike W_yy which stays local per location.
        self.N_matrix = np.ones((N_TOT, N_TOT))

        # Target covariance of one RF's responses to a uniform ensemble (see
        # frame_whiten.py:compute_uniform_target_covariance), used to derive per-interneuron
        # adaptation targets theta_t in _derivatives.
        self.uniform_target_covariance = np.loadtxt(target_covariance_path, delimiter=",")
        assert self.uniform_target_covariance.shape == (N_RF, N_RF), (
            f"uniform_target_covariance at {target_covariance_path} has shape "
            f"{self.uniform_target_covariance.shape}, expected ({N_RF}, {N_RF})."
        )

        self.tau_y = 0.2
        self.tau_a = 1.0  
        self.tau_u = 0.4 
        self.tau_g = 100.0 
        self.tau_v = 100.0 
        self.tau_avg = 10 
        self.tau_avg_z = 400
        
        self.sigma = 0.1  # matches V1Dynamics's default; sigma=15 made the constant
                          # (sigma/2)**2 floor in du_dt overwhelm u from a zero initial
                          # state, causing runaway positive feedback via pool_term's u_plus**2
        self.alpha = 0.0
        self.beta = 0.5

    def half_wave_rectify(self, y):
        return (np.maximum(y,0))**2

    def _derivatives(self, state, z_t):
        K = self.frame.K
        N_SETS = self.N_SETS
        N_RF = self.N_RF
        N_TOT = N_RF * N_SETS

        # Global variables - all span the full population (N_TOT = N_RF * N_SETS)
        y = state[0:N_TOT] # Primary responses across all RFs
        u = state[N_TOT:2*N_TOT] # Normalization pool spanning cRF and surround
        a = state[2*N_TOT:3*N_TOT] # Normalization pool spanning cRF and surround

        # Local variables confined to each RF - assuming all surround gains are the same by symmetry
        g_cRF = state[3*N_TOT:3*N_TOT+K]
        g_surround = state[3*N_TOT+K:3*N_TOT+2*K]
        v_cRF = state[3*N_TOT+2*K:3*N_TOT+3*K]
        v_surround = state[3*N_TOT+3*K:3*N_TOT+4*K]

        u_plus = self.half_wave_rectify(u)
        y_plus = self.half_wave_rectify(y)
        a_plus = self.half_wave_rectify(a)
        sqrt_y_plus = np.sqrt(y_plus)

        # Derive Targets (theta_t) from target covariance matrix: theta_t[i] = w_i @ uniform_target_covariance @ w_i.T,
        # where w_i is the i-th frame vector (column i of self.frame.W) - one target per interneuron.
        theta_t = np.diag(self.frame.W.T @ self.uniform_target_covariance @ self.frame.W)

        # Classical Receptive Field Adaptation Dynamics
        dg_cRF_dt = (v_cRF * v_cRF - theta_t) / self.tau_g # target set to theta_t (see above)
        dv_cRF_dt = (-v_cRF + self.frame.W.T @ y[:N_RF]) / self.tau_v # Estimation of variance of cRF neurons
        cRF_gain_feedback = self.frame.W @ (g_cRF * v_cRF)

        # Surround Adaptation Dynamics
        dg_surround_dt = (v_surround * v_surround - theta_t) / self.tau_g # target set to theta_t (see above)
        dv_surround_dt = (-v_surround + self.frame.W.T @ y[N_RF:2*N_RF]) / self.tau_v # Estimation of variance of surround neurons, using one surround RF and generalizing
        surround_gain_feedback = self.frame.W @ (g_surround * v_surround)

        recurrent_drive = (1.0 / (1.0 + a_plus)) * (self.W_yy @ sqrt_y_plus)
        input_drive = (self.beta * z_t) / 2

        # Local gain feedback matrix that can be applied to the full y dynamics
        full_gain_feedback = np.concatenate([cRF_gain_feedback]+[surround_gain_feedback]*(N_SETS-1))

        sigma_term = (self.sigma / 2) ** 2
        pool_term = self.N_matrix @ (y_plus * (u_plus ** 2))

        # ORGaNICs equations taken from Asit's Heirarchical Model (with gain feedback)
        dy_dt = (-y + input_drive + recurrent_drive ) / self.tau_y # MISSING - full_gain_feedback
        du_dt = (-u + sigma_term + pool_term) / self.tau_u
        da_dt = (-a + u_plus + a * u_plus + self.alpha * du_dt) / self.tau_a
        
        return np.concatenate([dy_dt, du_dt, da_dt, dg_cRF_dt, dg_surround_dt, dv_cRF_dt, dv_surround_dt])
        
    def run_simulation(self, stimulus_stream, initial_state=None):
        N, n_steps = stimulus_stream.shape
        N_TOT = self.N_RF * self.N_SETS
        K = self.frame.K

        assert N == N_TOT, (
            f"stimulus_stream has {N} rows but N_RF*N_SETS={N_TOT} - "
            f"generate it with matching N_RF/N_SETS."
        )

        if initial_state is not None:
            state = initial_state.copy()
        else:
            state = np.zeros(3*N_TOT + 4*K)

        # Tracking histories for later analysis + figures
        y_hist = np.zeros((N_TOT, n_steps))
        g_cRF_hist = np.zeros((K, n_steps))
        g_surround_hist = np.zeros((K, n_steps))
        u_hist = np.zeros((N_TOT, n_steps))
        a_hist = np.zeros((N_TOT, n_steps))
        v_cRF_hist = np.zeros((K, n_steps))
        v_surround_hist = np.zeros((K, n_steps))

        mode_str = "Adaptive"
        print(f"Running {mode_str} Simulation ({n_steps} steps)...")
        t0 = time.time()

        for t in tqdm(range(n_steps)):
            z_t = stimulus_stream[:, t]
            # RK4 Simulation
            k1 = self._derivatives(state, z_t)
            k2 = self._derivatives(state + 0.5 * self.dt * k1, z_t)
            k3 = self._derivatives(state + 0.5 * self.dt * k2, z_t)
            k4 = self._derivatives(state + self.dt * k3, z_t)

            state += (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

            y_hist[:, t] = np.maximum(state[0:N_TOT], 0)
            u_hist[:, t] = state[N_TOT:2*N_TOT]
            a_hist[:, t] = state[2*N_TOT:3*N_TOT]
            g_cRF_hist[:, t] = state[3*N_TOT:3*N_TOT+K]
            g_surround_hist[:, t] = state[3*N_TOT+K:3*N_TOT+2*K]
            v_cRF_hist[:, t] = state[3*N_TOT+2*K:3*N_TOT+3*K]
            v_surround_hist[:, t] = state[3*N_TOT+3*K:3*N_TOT+4*K]


        print(f"Simulation complete in {time.time() - t0:.2f}s.")
        self.last_state = state.copy()
        return y_hist, u_hist, a_hist, g_cRF_hist, g_surround_hist, v_cRF_hist, v_surround_hist

