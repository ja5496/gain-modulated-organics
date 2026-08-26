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

V1Dynamics: Joint normalization + adaptation dynamics for a single RF. Adaptation and normalization are 
both local only to this RF.

V1Dyamics_Surround: Joint normalization + adaptation dynamics for a cRF and its surround. Surround is 
modeled  by many small RFs that are exposed to the same stimuli. Adaptation is local to each RF, whereas 
normalization is global across the cRF + Surround. 

'''

class Frame:

    '''  Loads W (overcomplete frame) from a pre-computed csv file. '''

    def __init__(self, csv_path: str):
        import os
        print(f"Loading frame from {csv_path}...")
        self.W = np.loadtxt(csv_path, delimiter=",")
        self.dim = self.W.shape[0]
        self.K = self.W.shape[1]
        print(f"Loaded frame (N={self.dim}, K={self.K})")

class V1Dynamics:
    def __init__(self, v1_model, frame, dt=0.1, adaptive=True):
        self.v1 = v1_model
        self.frame = frame
        self.dt = dt 
        self.adaptive = adaptive

        self.tau_y = 0.2    # time constant of primary neurons
        self.tau_a = 0.2    # time constant of inhibitory neurons in normalization pool
        self.tau_u = 12.0   # time constant of excitatory neurons in normalization pool
        self.tau_g = 500.0  # time constant of interneuron gains
        self.tau_v = 5.0    # time constant of variance interneurons
        self.tau_avg = 12.0 
        self.tau_avg_z = 400
        
        self.sigma = 0.1 

        self.beta = 0.5

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

        beta = self.beta

        input_drive = beta * z_t
        
        sigma_term = (self.sigma / 2) ** 2
        pool_term = self.v1.N_matrix @ (y_plus * (u_plus ** 2))
        
        # ORGaNICs equations taken from Asit's Heirarchical Model (with gain feedback)
        dy_dt = (-y + input_drive + recurrent_drive - gain_feedback) / self.tau_y
        du_dt = (-u + sigma_term + pool_term) / self.tau_u
        da_dt = (-a + u_plus + a * u_plus) / self.tau_a
        
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
                 target_covariance_path="data/target_covs/uniform_target_covariance.csv",
                 gains_nonneg=False):
        self.v1 = v1_model     # Refers to tunings_whiten.py
        self.frame = frame     # Overcomplete frame (W)
        self.dt = dt           # Time step of simulation
        self.N_RF = N_RF       # Number of neurons in each small receptive field
        self.N_SETS = N_SETS   # Number of total RFs being modeled. 1 is the cRF and all else make up the surround
        N_TOT = N_RF * N_SETS  # Total number of neurons in cRF and Surround
        # If True, g_cRF/g_surround are clamped to >=0 after every RK4 step in run_simulation, for
        # the entire adaptation phase - dg/dt itself has no such floor (it's a plain difference of
        # squared terms, see _derivatives), so without this g can go negative and gain feedback
        # (W @ (g*v)) can turn facilitatory rather than strictly suppressive.
        self.gains_nonneg = gains_nonneg

        # Make single-RF W_yy block-diagonal (N_TOT, N_TOT):
        assert v1_model.W_yy.shape == (N_RF, N_RF), (
            f"v1_model.W_yy must be the single-location ({N_RF}, {N_RF}) matrix - "
            f"construct V1Tunings with N=N_RF (got shape {v1_model.W_yy.shape})."
        )
        self.W_yy = block_diag(*[v1_model.W_yy] * N_SETS)
        # Normalization pool spans cRF and surround: every one of the N_TOT neurons pools together
        self.N_matrix = np.ones((N_TOT, N_TOT))

        # Target covariance of one RF's responses to a uniform ensemble (see
        # frame_whiten.py:compute_uniform_target_covariance)
        self.uniform_target_covariance = np.loadtxt(target_covariance_path, delimiter=",")
        assert self.uniform_target_covariance.shape == (N_RF, N_RF), (
            f"uniform_target_covariance at {target_covariance_path} has shape "
            f"{self.uniform_target_covariance.shape}, expected ({N_RF}, {N_RF})."
        )
        # theta_t starts UNCALIBRATED - a large sentinel, not a real target. It used to be
        # seeded directly from uniform_target_covariance here, but that seeded value was
        # ALWAYS silently overwritten by calibrate_theta_t() (called from
        # Surround_simulated_responses.py's run_adaptation_phase) before any real adaptation
        # run - so setting theta_t here, or in uniform_target_covariance.csv, had no effect on
        # the live simulation, only a misleading appearance of one.
        #
        # That overwrite is NECESSARY, not optional: uniform_target_covariance.csv's value
        # (diag mean ~0.05) sits ABOVE this model's own achievable (v-mean)^2 ceiling
        # (~0.01-0.03 in practice, confirmed by direct measurement). With theta_t set that
        # high, dg/dt = (v-mean)^2 - theta_t is negative for every one of the K interneurons,
        # for the entire run - gains_nonneg then clips every gain to exactly zero at every
        # RK4 step, permanently, regardless of tau_g or run length. Bypassing calibration
        # (e.g. commenting out the overwrite and relying on a fixed theta_t instead)
        # reproduces this exact collapse - confirmed both by direct measurement and by
        # independently hitting the same failure while testing this fix.
        #
        # The sentinel here guarantees gains stay at EXACTLY zero until calibrate_theta_t()
        # runs - including during the reference run calibrate_theta_t() itself is measured
        # from, so that "unadapted" reference is genuinely uncontaminated by any partial
        # gain feedback. Its magnitude is NOT arbitrary: it must sit safely above the
        # achievable ceiling (~0.01-0.06 in practice) WITHOUT being needlessly huge. A
        # too-large sentinel (tried 1e4 first) makes |dg/dt| = |v^2-theta_t|/tau_g large
        # too - and gains_nonneg only clips g at the END of each full RK4 step, not during
        # its four intermediate sub-evaluations, so a large |dg/dt| still produces a large
        # UNCLAMPED excursion in the intermediate g used mid-step, which the gain-feedback
        # term W@(g*v) then injects into y - large enough, empirically, to push this
        # model's own near-critical u/a normalization dynamics (a_ss = u+/(1-u+), diverging
        # as u+ -> 1) into a genuine NaN blowup partway through a real run. Swept
        # {1e4, 100, 10, 1, 0.5} directly against a fixed adaptation stream: 1e4 reliably
        # produced NaN, all of {100, 10, 1, 0.5} stayed clean with g provably never leaving
        # zero. 1.0 keeps a full order-of-magnitude margin on both sides.
        self.theta_t = np.full(self.frame.K, 1.0)

        self.tau_y = 0.2       # time constant of primary neuron (fast)
        self.tau_a = 0.1       # time constant of inhibitory neurons in normalization pool (fast)
        self.tau_u = 15.0      # time constant of excitatory neurons in normalization pool (fast, slower than y, a)
        self.tau_g = 2500.0   # time constant of excitatory neurons in normalization pool (very slow, full context window needed)
        self.tau_v = 15.0    # time constant of excitatory neurons in normalization pool (medium to fast)
        self.tau_mu = 2500.0  # time constant of mean-response tracker (very slow, full context window needed)

        self.sigma = 0.15      # semi-saturation constant in the equations (adjusted to give simulation sigma ~ 0.15)
        self.beta = 0.5        # Constant input gain, beta = 1/2 for normalization fixed point derivation

    def half_wave_rectify(self, y, alpha=2.0):  # Used to estimate firing rates from membrane potential
        return (np.maximum(y,0)) ** alpha       # Rectify and raise to the power Beta (NOT input gain)

    def calibrate_theta_t(self, v_cRF_hist, v_surround_hist, mu_cRF_hist, mu_surround_hist, verbose=True):
        '''
        Sets self.theta_t (in place) from the EMPIRICAL variance of (v - W.T@mu), pooling
        cRF + surround, over the second half of the given histories (skips the slow
        mean-tracker mu's own warm-up transient). Returns the new theta_t.

        Call this ONCE, on a reference run of the UNBIASED/uniform ensemble. Correctness
        depends on that run having g_cRF/g_surround held at exactly zero for its ENTIRE
        duration - guaranteed by theta_t's sentinel value at __init__ (see there), provided
        this method has not already been called on this instance (a second call calibrates
        against a run that itself had nonzero, partially-adapted gain feedback, corrupting
        the "unadapted" reference this is meant to measure).

        This calibration is NECESSARY, not cosmetic: theta_t must sit BELOW at least some
        interneurons' achievable (v-mean)^2 under a BIASED ensemble, or dg/dt = v^2 - theta_t
        is negative everywhere and gains_nonneg clips every gain to exactly zero, permanently
        (see __init__'s comment for the full argument and confirmed numbers). Calibrating
        theta_t to THIS model's own unbiased-ensemble variance - rather than an offline/
        idealized value like uniform_target_covariance.csv - is what gives a later biased
        run's excess variance somewhere to be measured against, matching this codebase's
        existing "shrink to the ensemble's own mean/reference" convention elsewhere (see
        Analytic_responses.get_optimal_gains_target).

        Prints a summary (theta_t mean, before -> after) unless verbose=False - this
        overwrite is intentionally never a silent side effect.
        '''
        half = v_cRF_hist.shape[1] // 2
        resid_cRF = v_cRF_hist[:, half:] - self.frame.W.T @ mu_cRF_hist[:, half:]
        resid_surround = v_surround_hist[:, half:] - self.frame.W.T @ mu_surround_hist[:, half:]
        theta_t_before = self.theta_t.copy()
        self.theta_t = np.var(np.concatenate([resid_cRF, resid_surround], axis=1), axis=1)
        if verbose:
            print(f"  theta_t calibrated: mean {theta_t_before.mean():.5g} -> {self.theta_t.mean():.5g} "
                  f"(min {self.theta_t.min():.5g}, max {self.theta_t.max():.5g})")
        return self.theta_t

    def _derivatives(self, state, z_t):
        K = self.frame.K
        N_SETS = self.N_SETS
        N_RF = self.N_RF
        N_TOT = N_RF * N_SETS

        # Global variables 
        y = state[0:N_TOT]              # Primary responses across all RFs
        u = state[N_TOT:2*N_TOT]        # Normalization pool spanning cRF and surround
        a = state[2*N_TOT:3*N_TOT]      # Normalization pool spanning cRF and surround

        # Local variables 
        g_cRF = state[3*N_TOT:3*N_TOT+K]
        g_surround = state[3*N_TOT+K:3*N_TOT+2*K]
        v_cRF = state[3*N_TOT+2*K:3*N_TOT+3*K]
        v_surround = state[3*N_TOT+3*K:3*N_TOT+4*K]

        # Slow mean trackers:
        mu_cRF = state[3*N_TOT+4*K:3*N_TOT+4*K+N_RF]
        mu_surround = state[3*N_TOT+4*K+N_RF:3*N_TOT+4*K+2*N_RF]

        # Rectifications consistent with Asit's 'Heirarchical ORGaNICs' paper
        u_plus = self.half_wave_rectify(u, 0.5)
        y_plus = self.half_wave_rectify(y, 2.0)
        a_plus = self.half_wave_rectify(a, 1.0)
        sqrt_y_plus = np.sqrt(y_plus)

        theta_t = self.theta_t

        # Slow mean-tracking dynamics:
        dmu_cRF_dt = (-mu_cRF + y[:N_RF]) / self.tau_mu
        dmu_surround_dt = (-mu_surround + y[N_RF:2*N_RF]) / self.tau_mu

        # cRF Adaptation Dynamics
        dg_cRF_dt = ((v_cRF - self.frame.W.T @ mu_cRF) ** 2 - theta_t) / self.tau_g # mean-corrected target set to theta_t (see above)
        dv_cRF_dt = (-v_cRF + self.frame.W.T @ y[:N_RF]) / self.tau_v # Estimation of variance of cRF neurons
        cRF_gain_feedback = self.frame.W @ (g_cRF * v_cRF) # unchanged: suppression still scales with raw v_cRF, not the mean-corrected version

        # Surround Adaptation Dynamics
        dg_surround_dt = ((v_surround - self.frame.W.T @ mu_surround) ** 2 - theta_t) / self.tau_g # mean-corrected target set to theta_t (see above)
        dv_surround_dt = (-v_surround + self.frame.W.T @ y[N_RF:2*N_RF]) / self.tau_v # Estimation of variance of surround neurons, using one surround RF and generalizing
        surround_gain_feedback = self.frame.W @ (g_surround * v_surround) # unchanged: suppression still scales with raw v_surround

        # W_yy @ sqrt(y1+), rectified/one-sided per Asit's equation (DC_y1_dynamics) -- the
        # old (sqrt_y_plus - sqrt_y_minus) reduced to y itself (max(y,0)-max(-y,0) = y,
        # identically), silently cancelling the rectification. Matches V1Dynamics's own
        # (already-correct) recurrent_drive line above.
        recurrent_drive = (1.0 / (1.0 + a_plus)) * (self.W_yy @ sqrt_y_plus)
        input_drive = self.beta * z_t

        # Local gain feedback matrix that can be applied to the full y dynamics
        full_gain_feedback = (a_plus / (1 + a_plus)) * np.concatenate([cRF_gain_feedback]+[surround_gain_feedback]*(N_SETS-1))

        sigma_term = (self.sigma / 2) ** 2
        pool_term = self.N_matrix @ (y_plus * (u_plus ** 2))

        # ORGaNICs equations taken from Asit's Heirarchical Model (with gain feedback)
        dy_dt = (-y + input_drive + recurrent_drive - full_gain_feedback) / self.tau_y
        du_dt = (-u + sigma_term + pool_term) / self.tau_u
        # -a + u+ + a*u+ per Asit's equation (DC_a_dynamics: -a + u+ + a⊙u+ + alpha*du/dt) --
        # raw a, not a_plus, in the multiplicative term (CHANGED FROM (1+a_plus)*u_plus).
        # Asit's own alpha=0, so the additive alpha*du/dt term is correctly absent here, not
        # a missing term.
        da_dt = (-a + (1 + a) * u_plus) / self.tau_a

        return np.concatenate([dy_dt, du_dt, da_dt, dg_cRF_dt, dg_surround_dt, dv_cRF_dt, dv_surround_dt, dmu_cRF_dt, dmu_surround_dt])

    def run_simulation(self, stimulus_stream, initial_state=None):
        N, n_steps = stimulus_stream.shape
        N_TOT = self.N_RF * self.N_SETS
        K = self.frame.K
        N_RF = self.N_RF

        assert N == N_TOT, (
            f"stimulus_stream has {N} rows but N_RF*N_SETS={N_TOT} - "
            f"generate it with matching N_RF/N_SETS."
        )

        if initial_state is not None:
            state = initial_state.copy()
        else:
            state = np.zeros(3*N_TOT + 4*K + 2*N_RF)

        # Tracking histories for later analysis + figures
        y_hist = np.zeros((N_TOT, n_steps))
        g_cRF_hist = np.zeros((K, n_steps))
        g_surround_hist = np.zeros((K, n_steps))
        u_hist = np.zeros((N_TOT, n_steps))
        a_hist = np.zeros((N_TOT, n_steps))
        v_cRF_hist = np.zeros((K, n_steps))
        v_surround_hist = np.zeros((K, n_steps))
        mu_cRF_hist = np.zeros((N_RF, n_steps))
        mu_surround_hist = np.zeros((N_RF, n_steps))

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

            if self.gains_nonneg:
                # g_cRF and g_surround are contiguous in the state layout - clamp both in one call.
                state[3*N_TOT:3*N_TOT+2*K] = np.maximum(state[3*N_TOT:3*N_TOT+2*K], 0)

            y_hist[:, t] = np.maximum(state[0:N_TOT], 0)
            u_hist[:, t] = state[N_TOT:2*N_TOT]
            a_hist[:, t] = state[2*N_TOT:3*N_TOT]
            g_cRF_hist[:, t] = state[3*N_TOT:3*N_TOT+K]
            g_surround_hist[:, t] = state[3*N_TOT+K:3*N_TOT+2*K]
            v_cRF_hist[:, t] = state[3*N_TOT+2*K:3*N_TOT+3*K]
            v_surround_hist[:, t] = state[3*N_TOT+3*K:3*N_TOT+4*K]
            mu_cRF_hist[:, t] = state[3*N_TOT+4*K:3*N_TOT+4*K+N_RF]
            mu_surround_hist[:, t] = state[3*N_TOT+4*K+N_RF:3*N_TOT+4*K+2*N_RF]


        print(f"Simulation complete in {time.time() - t0:.2f}s.")
        self.last_state = state.copy()
        return (y_hist, u_hist, a_hist, g_cRF_hist, g_surround_hist, v_cRF_hist, v_surround_hist,
                mu_cRF_hist, mu_surround_hist)

