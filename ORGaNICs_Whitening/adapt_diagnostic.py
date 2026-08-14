import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import os
from tunings_whiten import V1Tunings
from stimuli_whiten import StimulusGenerator
from scipy.special import erf
from scipy.linalg import block_diag

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

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

class Adapt_Dynamics:
    def __init__(self, v1_model, frame, dt=0.1, N_RF=13, target_covariance_path="data/target_covs/uniform_target_covariance.csv"):
        self.v1 = v1_model
        self.frame = frame
        self.dt = dt
        self.N_RF = N_RF

        self.tau_y = 0.2
        self.tau_g = 20.0
        self.tau_v = 1.0
        self.beta = 0.5

        self.uniform_target_covariance = np.loadtxt(target_covariance_path, delimiter=",")
        assert self.uniform_target_covariance.shape == (N_RF, N_RF), (
            f"uniform_target_covariance at {target_covariance_path} has shape "
            f"{self.uniform_target_covariance.shape}, expected ({N_RF}, {N_RF}).")

        # Per-interneuron adaptation targets theta_t[i] = w_i @ uniform_target_covariance @ w_i.T
        # are fixed for the lifetime of this object (frame and target covariance never change),
        # so cache them here instead of rebuilding the (K, K) W.T @ Cov @ W product on every
        # single _derivatives call (4x per RK4 step - adds up fast over 100,000+ step runs).
        self.theta_t = np.diag(self.frame.W.T @ self.uniform_target_covariance @ self.frame.W)

    def half_wave_rectify(self, y, Beta=2.0):
        return (np.maximum(y,0)) ** Beta

    def _derivatives(self, state, z_t):
        N, K = self.v1.N, self.frame.K

        y = state[0:N]
        g = state[N:N+K]
        v = state[N+K:N+2*K]

        dg_dt = (v * v - self.theta_t) / self.tau_g # target set to the recent average of v^2 (avg_vsq)
        dv_dt = (-v + self.frame.W.T @ y) / self.tau_v # dynamics converge to whitening objective

        gain_feedback = self.frame.W @ (g * v)
        input_drive = self.beta * z_t

        dy_dt = (-y + input_drive - gain_feedback) / self.tau_y

        return np.concatenate([dy_dt, dg_dt, dv_dt])

    def run_simulation(self, stimulus_stream, initial_state=None):
        N, n_steps = stimulus_stream.shape
        K = self.frame.K

        if initial_state is not None:
            state = initial_state.copy()
        else:
            state = np.zeros(N + 2*K)

        # Tracking histories for later analysis + figures
        y_hist = np.zeros((N, n_steps))
        gains_hist = np.zeros((K, n_steps))
        v_hist = np.zeros((K, n_steps))

        print(f"Running Adaptive Simulation ({n_steps} steps)...")
        t0 = time.time()

        for t in tqdm(range(n_steps)):
            z_t = stimulus_stream[:, t]
            # RK4 Simulation
            k1 = self._derivatives(state, z_t)
            k2 = self._derivatives(state + 0.5 * self.dt * k1, z_t)
            k3 = self._derivatives(state + 0.5 * self.dt * k2, z_t)
            k4 = self._derivatives(state + self.dt * k3, z_t)

            state += (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

            y_hist[:, t] = state[0:N]
            gains_hist[:, t] = state[N:N+K]
            v_hist[:, t] = state[N+K:N+2*K]

        print(f"Simulation complete in {time.time() - t0:.2f}s.")
        self.last_state = state.copy()
        return y_hist, gains_hist, v_hist



if __name__ == "__main__":
    np.random.seed(0)

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------
    N_RF         = 13      # primary neurons
    DURATION     = 20     # timesteps each stimulus is held for
    TUNING_WIDTH = 0.75
    CONTRAST     = 0.5
    N_STEPS      = 100000

    FRAME_PATH      = os.path.join(REPO_ROOT, "data/frames/N13_mercedes_Frame.csv")
    TARGET_COV_PATH = os.path.join(REPO_ROOT, "data/target_covs/uniform_target_covariance.csv")

    print("Initializing tunings, frame, and adaptive whitening dynamics...")
    tunings = V1Tunings(N=N_RF)
    frame   = Frame(FRAME_PATH)          # K = 91 for the N13 mercedes frame
    K       = frame.K
    dyn     = Adapt_Dynamics(tunings, frame, N_RF=N_RF, target_covariance_path=TARGET_COV_PATH)


    def get_optimal_gains_target(stimuli, frame, label='', uniform_stimuli=None,
                                    target_covariance=None):
        '''
        target_covariance (optional): an (N, N) covariance matrix to use directly as the target,
        instead of estimating one from a fresh uniform_stimuli sample - e.g. the same
        uniform_target_covariance.csv the simulation uses to derive theta_t (see
        simulation_whiten.py's V1Dynamics_Surround), so the analytic gains are computed against
        the exact target the live network is actually being pulled toward. Takes precedence over
        uniform_stimuli if both are given.
        '''
        N, K = frame.shape  # N = 13, K = 91

        # Covariance generation
        stimuli = np.asarray(stimuli)
        Beta = 0.5
        input_drive = stimuli * Beta

        Covariance = np.cov(input_drive, rowvar=False)

        # GET MODIFED WHITENING MATRIX THAT SCALES ONLY LARGE VARIANCES
        eigenvalues, eigenvectors = np.linalg.eigh(Covariance)
        safe_lambdas = np.maximum(eigenvalues, 1e-9)

        if target_covariance is not None:
            # Directly-provided target covariance (e.g. uniform_target_covariance.csv) - use its
            # mean diagonal variance as-is rather than re-estimating a target from a fresh sample.
            target = np.diag(target_covariance)
        elif uniform_stimuli is not None:
            # Target variance from the uniform ensemble
            uniform_stimuli = np.asarray(uniform_stimuli)
            uniform_drive = uniform_stimuli * Beta
            uniform_covariance_matrix = uniform_drive
            uniform_Covariance = np.cov(uniform_drive, rowvar=False)
            target = np.diag(uniform_Covariance)
        else:
            target = np.mean(eigenvalues) # Set the mean variance as the upper bound ("target")

        d = np.sqrt(target / safe_lambdas) #np.minimum(1.0, np.sqrt(target / safe_lambdas))
        T = eigenvectors @ np.diag(d) @ eigenvectors.T

        # NOW COMPUTE OPTIMAL GAINS WITH LYNDON'S EQUATION A.5
        T_inv = np.linalg.inv(T)
        A = T_inv - np.eye(N) # Modified transformation for the optimal gains
        diag_WTAW = np.diag(frame.T @ A @ frame)
        WTW = frame.T @ frame
        WTW_sq = WTW ** 2                                  # Element-wise square
        inv_WTW_sq = np.linalg.pinv(WTW_sq)
        g_opt = inv_WTW_sq @ diag_WTAW


        # DIAGNOSTIC: sqrt(Covariance) vs its I + W@diag(g_opt)@W.T factorization
        '''fig_diag, ax_diag = plt.subplots(1, 2, figsize=(8, 4))
        vmin, vmax = T_inv.min(), T_inv.max()
        ax_diag[0].imshow(T_inv, vmin=vmin, vmax=vmax); ax_diag[0].set_title("T^-1")
        ax_diag[1].imshow(np.eye(N) + frame @ np.diag(g_opt) @ frame.T, vmin=vmin, vmax=vmax); ax_diag[1].set_title("I + W g W.T")
        plt.tight_layout(); plt.show()'''

        return g_opt

    def get_response(stimulus, g_frozen):
        T = np.linalg.inv(np.eye(N_RF) + frame.W @ np.diag(g_frozen) @ frame.W.T)
        y = T @ stimulus
        return y

    # ------------------------------------------------------------------
    # Stimuli: uniform and biased ensembles (generate_input_ensembles), matched
    # in vector length/duration/poisson-noise handling to generate_surround_ensembles.
    # ------------------------------------------------------------------
    stim_gen = StimulusGenerator(N=N_RF, num_angles=N_RF, stream_length=N_STEPS,
                                  tuning_width=TUNING_WIDTH, contrast=CONTRAST)

    print("Generating uniform and biased stimulus streams...")
    uniform_stream = stim_gen.generate_input_ensembles(biased=False, duration=DURATION, add_poisson_noise=True, mean_center=True)
    biased_stream  = stim_gen.generate_input_ensembles(biased=True,  duration=DURATION, add_poisson_noise=True, mean_center=True)

    assert uniform_stream.shape == (N_RF, N_STEPS), uniform_stream.shape
    assert biased_stream.shape  == (N_RF, N_STEPS), biased_stream.shape

    # ------------------------------------------------------------------
    # Analytic optimal gains for this environment. V1Dynamics here has no
    # divisive-normalization stage (input_drive is just beta*z_t), and the local
    # get_optimal_gains_target above never applies normalization either - it always
    # uses the raw covariance of the beta-scaled stimuli (Covariance = cov(Beta * stimuli)).
    #
    # target_covariance is explicitly set to dyn.uniform_target_covariance - the exact same
    # array Adapt_Dynamics uses to build theta_t (self.theta_t = diag(W.T @
    # uniform_target_covariance @ W)) - for BOTH the uniform and biased calls, rather than
    # letting each derive its own target from a fresh sample (uniform_stimuli=... / the
    # eigenvalue-mean fallback). Those were two different targets: theta_t (what the online
    # gains are actually pulled toward) vs. whatever get_optimal_gains_target separately
    # estimated from a live stimulus sample - so even starting the online run exactly at
    # g_opt/v_opt, the two would immediately diverge because they were chasing different
    # targets. Now both the analytic gains and the online adaptation target the identical
    # covariance matrix.
    # ------------------------------------------------------------------

    print("Computing analytic optimal gains from the uniform ensemble...")
    g_opt = get_optimal_gains_target(uniform_stream.T, frame.W, label='uniform',
                                      target_covariance=dyn.uniform_target_covariance)

    print("Computing analytic optimal gains from the biased ensemble...")
    g_opt_bias = get_optimal_gains_target(biased_stream.T, frame.W, label='biased',
                                           target_covariance=dyn.uniform_target_covariance)

    # ------------------------------------------------------------------
    # Run both simulations. y starts at 0, but g and v are seeded at their
    # optimal/fixed-point values instead of the default all-zero start: g at
    # that ensemble's analytic optimum (g_opt / g_opt_bias), and v at the
    # steady-state value consistent with g already being optimal - dg_dt=0
    # requires v^2 = theta_t, so v0 = sqrt(theta_t). theta_t is shared across
    # ensembles (derived once from the uniform target covariance), so the same
    # v0 seeds both runs.
    # ------------------------------------------------------------------
    v0 = np.sqrt(dyn.theta_t)

    init_state_uni = np.zeros(N_RF + 2 * K)
    init_state_uni[N_RF:N_RF + K]         = g_opt
    init_state_uni[N_RF + K:N_RF + 2 * K] = v0

    init_state_bias = np.zeros(N_RF + 2 * K)
    init_state_bias[N_RF:N_RF + K]         = g_opt_bias
    init_state_bias[N_RF + K:N_RF + 2 * K] = v0

    print("\n--- Uniform ensemble ---")
    y_uni, g_uni, v_uni = dyn.run_simulation(uniform_stream, initial_state=init_state_uni)

    print("\n--- Biased ensemble ---")
    y_bias, g_bias, v_bias = dyn.run_simulation(biased_stream, initial_state=init_state_bias)

    # ==================================================================
    # Plot 1: subset of optimal gains (dotted) vs. online gains (solid)
    # ==================================================================
    N_SUBSET   = 6
    subset_idx = np.linspace(0, K - 1, N_SUBSET).astype(int)
    colors     = plt.cm.viridis(np.linspace(0.1, 0.9, N_SUBSET))
    time_axis  = np.arange(N_STEPS) * dyn.dt

    fig1, axes1 = plt.subplots(1, 2, figsize=(14, 5))
    for ax, g_hist, g_opt_arr, title in zip(
            axes1, [g_uni, g_bias], [g_opt, g_opt_bias], ['Uniform Ensemble', 'Biased Ensemble']):
        for c, idx in zip(colors, subset_idx):
            ax.plot(time_axis, g_hist[idx], color=c, linewidth=2.0, label=f"g[{idx}]")
            ax.axhline(g_opt_arr[idx], color=c, linestyle=':', linewidth=2.0)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel("Time", fontsize=14, fontweight='bold')
        ax.set_ylabel("Gain", fontsize=14, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(width=2.0, length=6, labelsize=11)
        ax.legend(fontsize=9, ncol=2)
    fig1.suptitle("Online gains (solid) vs. analytic optimal gains (dotted)",
                  fontsize=15, fontweight='bold')
    plt.tight_layout()

    # ==================================================================
    # Plot 2: input stimuli covariance vs. response covariance (2nd half of run)
    # ==================================================================
    half = N_STEPS // 2
    cov_input_uni  = np.cov(uniform_stream, rowvar=True)
    cov_input_bias = np.cov(biased_stream,  rowvar=True)
    cov_resp_uni   = np.cov(y_uni[:, half:],  rowvar=True)
    cov_resp_bias  = np.cov(y_bias[:, half:], rowvar=True)

    fig2, axes2 = plt.subplots(2, 2, figsize=(11, 10))
    mats       = [[cov_input_uni, cov_resp_uni], [cov_input_bias, cov_resp_bias]]
    row_titles = ['Uniform Ensemble', 'Biased Ensemble']
    col_titles = ['Input Stimuli Covariance', 'Response Covariance (2nd half)']
    for i in range(2):
        for j in range(2):
            ax = axes2[i, j]
            im = ax.imshow(mats[i][j], cmap='viridis', aspect='auto')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(f"{row_titles[i]}\n{col_titles[j]}", fontsize=13, fontweight='bold')
            for spine in ax.spines.values():
                spine.set_linewidth(2.0)
            ax.tick_params(width=2.0, length=5, labelsize=10)
    plt.tight_layout()

    # ==================================================================
    # Plot 3: (I + W diag(g) W.T)^-1 for optimal gains vs. online gains (final,
    # most-adapted timestep), one row per ensemble - g_opt/g_uni for Uniform,
    # g_opt_bias/g_bias for Biased.
    # ==================================================================
    g_online_final_uni  = g_uni[:, -1]
    g_online_final_bias = g_bias[:, -1]

    M_opt_uni     = np.eye(N_RF) + frame.W @ np.diag(g_opt)              @ frame.W.T
    M_online_uni  = np.eye(N_RF) + frame.W @ np.diag(g_online_final_uni) @ frame.W.T
    M_opt_bias    = np.eye(N_RF) + frame.W @ np.diag(g_opt_bias)          @ frame.W.T
    M_online_bias = np.eye(N_RF) + frame.W @ np.diag(g_online_final_bias) @ frame.W.T

    inv_opt_uni     = np.linalg.inv(M_opt_uni)
    inv_online_uni  = np.linalg.inv(M_online_uni)
    inv_opt_bias    = np.linalg.inv(M_opt_bias)
    inv_online_bias = np.linalg.inv(M_online_bias)

    fig3, axes3 = plt.subplots(2, 2, figsize=(11, 10))
    mats3      = [[inv_opt_uni, inv_online_uni], [inv_opt_bias, inv_online_bias]]
    row_titles3 = ['Uniform Ensemble', 'Biased Ensemble']
    col_titles3 = ['Optimal Gains', 'Online Gains (final)']
    for i in range(2):
        for j in range(2):
            ax = axes3[i, j]
            # Each panel autoscales to its own data (no shared vmin/vmax) so a panel's
            # structure is still visible even when its magnitude is wildly different from
            # the others - e.g. online gains stuck near 0 vs. optimal gains at O(1).
            im = ax.imshow(mats3[i][j], cmap='viridis', aspect='auto')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(f"{row_titles3[i]}\n{col_titles3[j]}", fontsize=13, fontweight='bold')
            for spine in ax.spines.values():
                spine.set_linewidth(2.0)
            ax.tick_params(width=2.0, length=5, labelsize=10)
    fig3.suptitle(r"$(I + W\,\mathrm{diag}(g)\,W^\top)^{-1}$", fontsize=15, fontweight='bold')
    plt.tight_layout()

    # ==================================================================
    # Plot 4: gain feedback (W @ (g*v)) per primary-neuron index, averaged over
    # the last LAST_N_AVG timesteps of each run - optimal gains vs. online gains
    # (each panel overlays Uniform vs. Biased).
    # ==================================================================
    LAST_N_AVG = 10000  # timesteps to average over, counted back from the end of each run

    def avg_gain_feedback(g, v_hist, last_n=LAST_N_AVG):
        '''Average frame.W @ (g*v) over the run's last `last_n` timesteps. g may be a
        constant (K,) vector (e.g. g_opt, broadcast across the window) or a full (K, n_steps)
        history (e.g. the online g_hist) - either way this averages the actual per-timestep
        gain feedback rather than averaging g and v separately first, which would silently
        drop any g/v correlation over the window (the two are only equivalent when g is
        constant, as in the optimal-gains case).'''
        v_window = v_hist[:, -last_n:]
        g_window = g[:, None] if g.ndim == 1 else g[:, -last_n:]
        return frame.W @ (g_window * v_window).mean(axis=1)

    gf_opt_uni     = avg_gain_feedback(g_opt,      v_uni)
    gf_opt_bias    = avg_gain_feedback(g_opt_bias, v_bias)
    gf_online_uni  = avg_gain_feedback(g_uni,      v_uni)
    gf_online_bias = avg_gain_feedback(g_bias,     v_bias)

    neuron_idx = np.arange(N_RF)

    fig4, axes4 = plt.subplots(1, 2, figsize=(12, 5))
    for ax, gf_uni, gf_bias, title in zip(
            axes4, [gf_opt_uni, gf_online_uni], [gf_opt_bias, gf_online_bias],
            ['Optimal Gains', f'Online Gains (avg, last {LAST_N_AVG} steps)']):
        ax.plot(neuron_idx, gf_uni,  'o-', color='#0b3d91', linewidth=2.5, markersize=6, label='Uniform')
        ax.plot(neuron_idx, gf_bias, 'o-', color='#b35900', linewidth=2.5, markersize=6, label='Biased')
        ax.axhline(0, color='gray', linewidth=1.0, linestyle='--', alpha=0.7)
        ax.set_title(title, fontsize=15, fontweight='bold')
        ax.set_xlabel("Neuron Index", fontsize=13, fontweight='bold')
        ax.set_ylabel("Gain Feedback", fontsize=13, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(width=2.0, length=6, labelsize=11)
        ax.legend(fontsize=10)
    fig4.suptitle("Steady-state gain feedback per neuron: optimal vs. online gains",
                  fontsize=15, fontweight='bold')
    plt.tight_layout()

    plt.show()
