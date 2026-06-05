import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from tunings_whiten import V1Tunings
from stimuli_whiten import StimulusGenerator
from scipy.special import erf

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

        self.tau_y = 0.2 # from 1
        self.tau_a = 1.0 # from 5   
        self.tau_u = 0.4 # from 2
        self.tau_g = 100.0 
        self.tau_v = 50.0    
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
        
    def run_simulation(self, stimulus_stream):
        N, n_steps = stimulus_stream.shape 
        K = self.frame.K
        
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
        return y_hist, gains_hist, u_hist, a_hist, v_hist, avg_z_hist, avg_vsq_hist
