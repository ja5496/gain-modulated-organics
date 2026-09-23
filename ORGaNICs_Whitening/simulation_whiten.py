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
                 target_covariance_path="data/target_covs/uniform_target_covariance_mid_c.csv",
                 gains_nonneg=False, surround_delay=0.0):
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
        # Normalization pool spans cRF and surround: every one of the N_TOT neurons pools together.
        # Kept as a public attribute for callers that read it directly (e.g.
        # Surround_simulated_responses.py's get_response_offline/frozen_derivatives,
        # convergence_tests.py's analytic_response_operator) even though _derivatives itself now
        # builds pool_cRF/pool_surround explicitly instead of matrix-multiplying by this.
        self.N_matrix = np.ones((N_TOT, N_TOT))

        # Target covariance of one RF's responses to a uniform ensemble (see
        # frame_whiten.py:compute_uniform_target_covariance)
        self.uniform_target_covariance = np.loadtxt(target_covariance_path, delimiter=",")
        assert self.uniform_target_covariance.shape == (N_RF, N_RF), (
            f"uniform_target_covariance at {target_covariance_path} has shape "
            f"{self.uniform_target_covariance.shape}, expected ({N_RF}, {N_RF})."
        )

        self.theta_t = np.full(self.frame.K, 1.0) # Initial overestimate of target variances (not the values that will be used)

        self.tau_y = 0.2       # time constant of primary neuron (fast)
        self.tau_a = 0.1       # time constant of inhibitory neurons in normalization pool (fast)
        self.tau_u = 15.0      # time constant of excitatory neurons in normalization pool (fast, slower than y, a)
        self.tau_g = 2000.0   # time constant of excitatory neurons in normalization pool (very slow, full context window needed)
        self.tau_v = 20.0    # time constant of excitatory neurons in normalization pool (medium to fast)
        self.tau_mu = 100000.0  # time constant of mean-response tracker (very slow, full context window needed)

        self.sigma = 0.15      # semi-saturation constant in the equations (adjusted to give simulation sigma ~ 0.15)
        self.beta = 0.5        # Constant input gain, beta = 1/2 for normalization fixed point derivation

        # Surround -> cRF communication delay (opt-in; surround_delay=0.0 reproduces the
        # pre-delay model exactly - see _derivatives' n_lag==0 branch). Sampled once per outer
        # RK4 step in run_simulation, not interpolated across substeps - matches how z_t is
        # already held constant across k1..k4.
        self.set_surround_delay(surround_delay)

        # State-vector layout offsets - single source of truth (also used by unpack_state, so
        # callers never need to hand-derive these themselves). u/a are split into cRF/surround
        # pairs (see _derivatives), each N_RF-length like mu_cRF/mu_surround already are - "one
        # shared representative copy, broadcast/tiled across N_SETS-1 surround blocks" is the
        # same convention g_surround/v_surround/mu_surround already use.
        K = self.frame.K
        self._off_u_cRF = N_TOT
        self._off_u_surround = N_TOT + N_RF
        self._off_a_cRF = N_TOT + 2*N_RF
        self._off_a_surround = N_TOT + 3*N_RF
        self._off_g_cRF = N_TOT + 4*N_RF
        self._off_g_surround = self._off_g_cRF + K
        self._off_v_cRF = self._off_g_surround + K
        self._off_v_surround = self._off_v_cRF + K
        self._off_mu_cRF = self._off_v_surround + K
        self._off_mu_surround = self._off_mu_cRF + N_RF
        self.state_size = self._off_mu_surround + N_RF

    def set_surround_delay(self, surround_delay):
        '''Set (or change) the surround->cRF communication delay, in seconds. 0.0 (default)
        reproduces the model's pre-delay behavior exactly. Callable after construction so a
        caller can compare multiple delays without reconstructing/recalibrating theta_t.'''
        assert surround_delay >= 0, f"surround_delay must be >= 0 (got {surround_delay})"
        self.surround_delay = surround_delay
        self.n_lag = int(round(surround_delay / self.dt))

    def unpack_state(self, state):
        '''Single source of truth for slicing a raw V1Dynamics_Surround state vector by name.
        Prefer this over hand-deriving offsets (e.g. from dyn.last_state) - that pattern breaks
        silently whenever the state layout changes.'''
        N_RF, K = self.N_RF, self.frame.K
        return dict(
            y=state[0:self._off_u_cRF],
            u_cRF=state[self._off_u_cRF:self._off_u_cRF+N_RF],
            u_surround=state[self._off_u_surround:self._off_u_surround+N_RF],
            a_cRF=state[self._off_a_cRF:self._off_a_cRF+N_RF],
            a_surround=state[self._off_a_surround:self._off_a_surround+N_RF],
            g_cRF=state[self._off_g_cRF:self._off_g_cRF+K],
            g_surround=state[self._off_g_surround:self._off_g_surround+K],
            v_cRF=state[self._off_v_cRF:self._off_v_cRF+K],
            v_surround=state[self._off_v_surround:self._off_v_surround+K],
            mu_cRF=state[self._off_mu_cRF:self._off_mu_cRF+N_RF],
            mu_surround=state[self._off_mu_surround:self._off_mu_surround+N_RF],
        )

    def half_wave_rectify(self, y, alpha=2.0):  # Used to estimate firing rates from membrane potential
        return (np.maximum(y,0)) ** alpha       # Rectify and raise to the power Beta (NOT input gain)

    def calibrate_theta_t(self, v_cRF_hist=None, v_surround_hist=None, mu_cRF_hist=None, mu_surround_hist=None, C_zz_uniform=None,  
                          verbose=True, circular_target=False, uniform_target=False):
        '''
        `circular` covariance target:
            Sets self.theta_t as the mean of the empirical variance of (v - W.T@mu) over the second half 
            histories so every interneuron shares one isotropic threshold. Returns the new theta_t. 

        `uniform` covariance target:
            Sets self.theta_t from the covariance of the uniform ensemble with no noise (C_zz_uniform). 
            theta_t = diag(W)

        Prints a summary (theta_t mean, before -> after) unless verbose=False - this
        overwrite is intentionally never a silent side effect.
        '''
        if circular_target:
            half = v_cRF_hist.shape[1] // 2
            resid_cRF = v_cRF_hist[:, half:] - self.frame.W.T @ mu_cRF_hist[:, half:]
            resid_surround = v_surround_hist[:, half:] - self.frame.W.T @ mu_surround_hist[:, half:]
            theta_t_before = self.theta_t.copy()
            self.theta_t = np.var(np.concatenate([resid_cRF, resid_surround], axis=1), axis=1)
            # Flatten to one scalar shared by every interneuron (isotropic target, i.e. the implied
            # target covariance is proportional to identity) instead of the per-interneuron profile.
            self.theta_t = np.full_like(self.theta_t, self.theta_t.mean())
            if verbose:
                print(f"  theta_t calibrated: mean {theta_t_before.mean():.5g} -> {self.theta_t.mean():.5g} "
                    f"(min {self.theta_t.min():.5g}, max {self.theta_t.max():.5g})")

        elif uniform_target:
            self.theta_t = np.diag(self.frame.W.T @ C_zz_uniform @ self.frame.W)
            if verbose:
                print(f"  theta_t calculated: mean {self.theta_t.mean():.5g} "
                    f"(min {self.theta_t.min():.5g}, max {self.theta_t.max():.5g})")

        else:
            raise ValueError("No target covariance was set --> cannot compute marginal variance targets (theta_t). " \
            "Please set either 'cicular_target' or 'uniform_target' to 'True' when calling calibrate_theta_t.")
        return self.theta_t

    def _surround_msg(self, state):
        '''Surround's total instantaneous local pooling contribution, Sum_surround(y_plus *
        u_plus**2), extracted from a raw (finalized) state vector. This is the "message" that
        surround_delay clocks between the surround and the cRF (see _derivatives). Same formula
        as _derivatives' surround_msg_now, kept in sync manually since this runs on a finalized
        post-RK4-step state while that one runs on a live sub-step state. 0.0 at N_SETS==1 (no
        surround blocks exist).'''
        N_RF, N_SETS = self.N_RF, self.N_SETS
        if N_SETS < 2:
            return 0.0
        y_plus_surround = self.half_wave_rectify(state[N_RF:N_RF*N_SETS], 2.0)
        u_plus_surround = self.half_wave_rectify(
            state[self._off_u_surround:self._off_u_surround+N_RF], 0.5)
        return float(np.sum(y_plus_surround * (np.tile(u_plus_surround, N_SETS - 1) ** 2)))

    def _derivatives(self, state, z_t, surround_msg_delayed):
        K = self.frame.K
        N_SETS = self.N_SETS
        N_RF = self.N_RF
        N_TOT = N_RF * N_SETS

        # Primary responses across all RFs
        y = state[0:N_TOT]

        # Normalization pool, split into a cRF-gating copy and a surround-gating copy (see
        # set_surround_delay/_surround_msg): each N_RF-length, "one shared representative copy,
        # broadcast/tiled across N_SETS-1 surround blocks" - the same convention already used
        # for g_surround/v_surround/mu_surround below, not a true scalar (the old single `a`'s
        # all-entries-identical behavior was an emergent consequence of the all-ones N_matrix,
        # not a structural guarantee worth hardcoding).
        u_cRF = state[self._off_u_cRF:self._off_u_cRF+N_RF]
        u_surround = state[self._off_u_surround:self._off_u_surround+N_RF]
        a_cRF = state[self._off_a_cRF:self._off_a_cRF+N_RF]
        a_surround = state[self._off_a_surround:self._off_a_surround+N_RF]

        # Local variables
        g_cRF = state[self._off_g_cRF:self._off_g_cRF+K]
        g_surround = state[self._off_g_surround:self._off_g_surround+K]
        v_cRF = state[self._off_v_cRF:self._off_v_cRF+K]
        v_surround = state[self._off_v_surround:self._off_v_surround+K]

        # Slow mean trackers:
        mu_cRF = state[self._off_mu_cRF:self._off_mu_cRF+N_RF]
        mu_surround = state[self._off_mu_surround:self._off_mu_surround+N_RF]

        # Rectifications consistent with Asit's 'Heirarchical ORGaNICs' paper
        u_plus_cRF = self.half_wave_rectify(u_cRF, 0.5)
        u_plus_surround = self.half_wave_rectify(u_surround, 0.5)
        y_plus = self.half_wave_rectify(y, 2.0)
        a_plus_cRF = self.half_wave_rectify(a_cRF, 1.0)
        a_plus_surround = self.half_wave_rectify(a_surround, 1.0)
        sqrt_y_plus = np.sqrt(y_plus)

        # Full N_TOT-length view needed for the population-wide recurrent/gain-feedback terms
        # below - built from the split blocks exactly like full_gain_feedback already
        # concatenates cRF_gain_feedback/surround_gain_feedback further down.
        a_plus_full = np.concatenate([a_plus_cRF] + [a_plus_surround] * (N_SETS - 1))

        theta_t = self.theta_t

        # At N_SETS=1 there is no surround block at all (y only holds the cRF's own N_RF
        # entries) - y[N_RF:2*N_RF] would otherwise silently come back empty and break every
        # broadcast below against the (always N_RF/K-sized) mu_surround/v_surround/g_surround
        # state. Treat the nonexistent surround as driven by zero input instead: mu_surround
        # and v_surround then just decay toward 0 from their (zero) initial state and stay
        # there, and g_surround's target term becomes the constant -theta_t/tau_g, clamped to
        # 0 by run_simulation's gains_nonneg floor - i.e. surround adaptation is genuinely inert,
        # matching the single-RF (no surround) model this represents.
        y_surround = y[N_RF:2*N_RF] if N_SETS >= 2 else np.zeros(N_RF)

        # Slow mean-tracking dynamics:
        dmu_cRF_dt = (-mu_cRF + y[:N_RF]) / self.tau_mu
        dmu_surround_dt = (-mu_surround + y_surround) / self.tau_mu

        # cRF Adaptation Dynamics
        dg_cRF_dt = ((v_cRF - self.frame.W.T @ mu_cRF) ** 2 - theta_t) / self.tau_g # mean-corrected target set to theta_t (see above)
        dv_cRF_dt = (-v_cRF + self.frame.W.T @ y[:N_RF]) / self.tau_v # Estimation of variance of cRF neurons
        cRF_gain_feedback = self.frame.W @ (g_cRF * v_cRF) # unchanged: suppression still scales with raw v_cRF, not the mean-corrected version

        # Surround Adaptation Dynamics
        dg_surround_dt = ((v_surround - self.frame.W.T @ mu_surround) ** 2 - theta_t) / self.tau_g # mean-corrected target set to theta_t (see above)
        dv_surround_dt = (-v_surround + self.frame.W.T @ y_surround) / self.tau_v # Estimation of variance of surround neurons, using one surround RF and generalizing
        surround_gain_feedback = self.frame.W @ (g_surround * v_surround) # unchanged: suppression still scales with raw v_surround

        # W_yy @ sqrt(y1+), rectified/one-sided per Asit's equation (DC_y1_dynamics) -- the
        # old (sqrt_y_plus - sqrt_y_minus) reduced to y itself (max(y,0)-max(-y,0) = y,
        # identically), silently cancelling the rectification. Matches V1Dynamics's own
        # (already-correct) recurrent_drive line above.
        recurrent_drive = (1.0 / (1.0 + a_plus_full)) * (self.W_yy @ sqrt_y_plus)
        input_drive = self.beta * z_t

        # Local gain feedback matrix that can be applied to the full y dynamics
        full_gain_feedback = (a_plus_full / (1 + a_plus_full)) * np.concatenate([cRF_gain_feedback]+[surround_gain_feedback]*(N_SETS-1))

        sigma_term = (self.sigma / 2) ** 2

        # --- Surround-delay pooling split ---
        # cRF's own instantaneous contribution to the shared pool - both pathways below use this
        # fresh, at whichever state _derivatives is evaluated at (k1..k4 alike).
        own_cRF_term = np.sum(y_plus[:N_RF] * (u_plus_cRF ** 2))
        # Surround's total instantaneous contribution: sum over ALL N_SETS-1 surround blocks'
        # own y_plus*u_plus**2 (shared representative u_plus_surround, tiled). 0 at N_SETS==1.
        surround_msg_now = (np.sum(y_plus[N_RF:] * (np.tile(u_plus_surround, N_SETS - 1) ** 2))
                             if N_SETS >= 2 else 0.0)

        # Surround pathway: computed as normal, no delay - identical in value to the old single
        # global pool_term, regardless of surround_delay/n_lag.
        pool_surround = own_cRF_term + surround_msg_now

        # cRF pathway: the surround's contribution arrives surround_delay seconds late. At
        # n_lag==0 (the default), reuse surround_msg_now directly instead of run_simulation's
        # once-per-step buffer sample - otherwise pool_cRF would only agree with pool_surround at
        # the k1 evaluation point and silently diverge at k2/k3/k4 (the buffer is frozen for the
        # whole outer step, this live value isn't), which would make surround_delay=0.0 NOT be an
        # exact reproduction of the pre-delay model. For n_lag>0 there is no live "delayed" value
        # to sample mid-substep, so the buffer is used as designed.
        surround_term_for_cRF = surround_msg_now if self.n_lag == 0 else surround_msg_delayed
        pool_cRF = own_cRF_term + surround_term_for_cRF

        # ORGaNICs equations taken from Asit's Heirarchical Model (with gain feedback)
        dy_dt = (-y + input_drive + recurrent_drive - full_gain_feedback) / self.tau_y
        du_cRF_dt = (-u_cRF + sigma_term + pool_cRF) / self.tau_u
        du_surround_dt = (-u_surround + sigma_term + pool_surround) / self.tau_u
        # -a + u+ + a*u+ per Asit's equation (DC_a_dynamics: -a + u+ + a⊙u+ + alpha*du/dt) --
        # raw a, not a_plus, in the multiplicative term (CHANGED FROM (1+a_plus)*u_plus).
        # Asit's own alpha=0, so the additive alpha*du/dt term is correctly absent here, not
        # a missing term.
        da_cRF_dt = (-a_cRF + (1 + a_cRF) * u_plus_cRF) / self.tau_a
        da_surround_dt = (-a_surround + (1 + a_surround) * u_plus_surround) / self.tau_a

        return np.concatenate([dy_dt, du_cRF_dt, du_surround_dt, da_cRF_dt, da_surround_dt,
                                dg_cRF_dt, dg_surround_dt, dv_cRF_dt, dv_surround_dt,
                                dmu_cRF_dt, dmu_surround_dt])

    def run_simulation(self, stimulus_stream, initial_state=None, surround_msg_baseline=0.0):
        '''surround_msg_baseline: value to use for the delayed surround->cRF pooling signal
        (see _surround_msg) for as long as fewer than n_lag steps have elapsed - i.e. before
        there has been time for a real delayed reading to exist. Defaults to 0.0, which is only
        exercised when surround_delay>0 (irrelevant otherwise, since n_lag==0 never reads this
        buffer - see _derivatives). 0.0 means "surround contributes nothing yet"; callers that
        care about physical realism should instead pass the surround's baseline (non-adapting)
        steady-state contribution - e.g. dyn._surround_msg(dyn.last_state) after a calibration
        run driven by a flat/non-adapting stimulus - so the pre-arrival window is continuous
        with the actual pre-adaptor operating regime instead of an artificial "surround absent"
        transient.'''
        N, n_steps = stimulus_stream.shape
        N_TOT = self.N_RF * self.N_SETS
        K = self.frame.K
        N_RF = self.N_RF
        n_lag = self.n_lag

        assert N == N_TOT, (
            f"stimulus_stream has {N} rows but N_RF*N_SETS={N_TOT} - "
            f"generate it with matching N_RF/N_SETS."
        )

        if initial_state is not None:
            state = initial_state.copy()
        else:
            state = np.zeros(self.state_size)

        # Tracking histories for later analysis + figures
        y_hist = np.zeros((N_TOT, n_steps))
        u_cRF_hist = np.zeros((N_RF, n_steps))
        u_surround_hist = np.zeros((N_RF, n_steps))
        a_cRF_hist = np.zeros((N_RF, n_steps))
        a_surround_hist = np.zeros((N_RF, n_steps))
        g_cRF_hist = np.zeros((K, n_steps))
        g_surround_hist = np.zeros((K, n_steps))
        v_cRF_hist = np.zeros((K, n_steps))
        v_surround_hist = np.zeros((K, n_steps))
        mu_cRF_hist = np.zeros((N_RF, n_steps))
        mu_surround_hist = np.zeros((N_RF, n_steps))

        # Delayed-surround-message buffer (see _derivatives/_surround_msg). surround_msg_hist[t]
        # holds the surround's total pooling contribution computed from the state AFTER step t's
        # RK4 update - read back n_lag steps later, held at surround_msg_baseline before that.
        surround_msg_hist = np.zeros(n_steps)

        mode_str = "Adaptive"
        print(f"Running {mode_str} Simulation ({n_steps} steps)...")
        t0 = time.time()

        for t in tqdm(range(n_steps)):
            z_t = stimulus_stream[:, t]
            lag_idx = t - n_lag
            surround_msg_delayed = surround_msg_hist[lag_idx] if lag_idx >= 0 else surround_msg_baseline

            # RK4 Simulation
            k1 = self._derivatives(state, z_t, surround_msg_delayed)
            k2 = self._derivatives(state + 0.5 * self.dt * k1, z_t, surround_msg_delayed)
            k3 = self._derivatives(state + 0.5 * self.dt * k2, z_t, surround_msg_delayed)
            k4 = self._derivatives(state + self.dt * k3, z_t, surround_msg_delayed)

            state += (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

            if self.gains_nonneg:
                # g_cRF and g_surround are contiguous in the state layout - clamp both in one call.
                state[self._off_g_cRF:self._off_g_cRF+2*K] = np.maximum(state[self._off_g_cRF:self._off_g_cRF+2*K], 0)

            y_hist[:, t] = np.maximum(state[0:N_TOT], 0)
            u_cRF_hist[:, t] = state[self._off_u_cRF:self._off_u_cRF+N_RF]
            u_surround_hist[:, t] = state[self._off_u_surround:self._off_u_surround+N_RF]
            a_cRF_hist[:, t] = state[self._off_a_cRF:self._off_a_cRF+N_RF]
            a_surround_hist[:, t] = state[self._off_a_surround:self._off_a_surround+N_RF]
            g_cRF_hist[:, t] = state[self._off_g_cRF:self._off_g_cRF+K]
            g_surround_hist[:, t] = state[self._off_g_surround:self._off_g_surround+K]
            v_cRF_hist[:, t] = state[self._off_v_cRF:self._off_v_cRF+K]
            v_surround_hist[:, t] = state[self._off_v_surround:self._off_v_surround+K]
            mu_cRF_hist[:, t] = state[self._off_mu_cRF:self._off_mu_cRF+N_RF]
            mu_surround_hist[:, t] = state[self._off_mu_surround:self._off_mu_surround+N_RF]

            surround_msg_hist[t] = self._surround_msg(state)

        print(f"Simulation complete in {time.time() - t0:.2f}s.")
        self.last_state = state.copy()
        return (y_hist, u_cRF_hist, u_surround_hist, a_cRF_hist, a_surround_hist,
                g_cRF_hist, g_surround_hist, v_cRF_hist, v_surround_hist,
                mu_cRF_hist, mu_surround_hist, surround_msg_hist)

