"""
diagnostics_whiten.py
---------------------
Four standalone diagnostics for exploring model behavior.

1. Scale sweep      – how probe tuning curves change with the `scale` parameter
2. No normalization – activity when a is forced to 0 (normalization disabled)
3. dg/dt variants   – compare avg/N, avg²/N (default), avg²/N², avg^1.5/N
4. Random N_matrix  – replace all-ones pooling matrix with random [0, 1] entries

Run from the ORGaNICs_Whitening/ directory:
    python diagnostics_whiten.py
"""

import gc
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
import time

from tunings_whiten import V1Tunings
from stimuli_whiten import StimulusGenerator
from simulation_whiten import V1Dynamics, Frame


# ── Shared parameters ──────────────────────────────────────────────────────────
N           = 169
STREAM_LEN  = 5000    # shorter than Carandini for faster diagnostics
PROBE_RES   = 90
PROBE_STEPS = 100
N_BINS      = 13
Z_SPONT     = 0.1
AVG_WINDOW  = 2000

np.random.seed(20)


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _make_canonical_frame():
    """Load N169 frame and canonicalize so W @ W.T == (K/N)*I."""
    frame = Frame(csv_path="Frames/N169_Frame.csv")
    S = frame.W @ frame.W.T
    eigvals, eigvecs = np.linalg.eigh(S)
    S_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    N_neu, K_neu = frame.W.shape
    frame.W = np.sqrt(K_neu / N_neu) * (S_inv_sqrt @ frame.W)
    return frame


def gaussian_rectify(y, threshold=0.5, sigma=0.2, r_max=1.0):
    return 0.5 * (1 + erf((y - threshold) / (sigma * np.sqrt(2)))) * r_max


def run_probe(frame, tunings, fixed_gains, probe_angles,
              frozen_u=None, frozen_a=None, z_spont=Z_SPONT,
              scale=1.0, force_a_zero=False):
    """
    Probe tuning curves with optional overrides:
      scale        – scales the input vector after normalization
      force_a_zero – sets a=0 throughout (disables normalization)
    """
    N_neu = frame.dim
    n_probes = len(probe_angles)
    tuning_curves = np.zeros((N_neu, n_probes))
    W_yy = tunings.W_yy

    dt = 0.05; tau_y = 1.0; tau_u = 2.0; tau_a = 5.0
    beta = 1.0; sigma_c = 0.05; tuning_width = 0.5

    for i, angle in enumerate(probe_angles):
        y = z_spont * np.ones(N_neu)
        u = np.copy(frozen_u) if frozen_u is not None else np.zeros(N_neu)
        a = np.zeros(N_neu) if force_a_zero else (
            np.copy(frozen_a) if frozen_a is not None else np.zeros(N_neu))

        z_t = np.exp(tuning_width * np.cos(2 * (tunings.theta - angle)))
        z_t = (z_t / np.max(z_t)) * scale

        for _ in range(PROBE_STEPS):
            u_plus      = gaussian_rectify(u)
            y_plus      = gaussian_rectify(y)
            a_plus      = np.zeros(N_neu) if force_a_zero else gaussian_rectify(a)
            sqrt_y_plus = np.sqrt(y_plus)

            v_t           = frame.W.T @ y
            gain_feedback = (frame.W @ (fixed_gains * v_t)) if fixed_gains is not None else 0.0
            recurrent_drive = (1.0 / (1.0 + a_plus)) * (W_yy @ sqrt_y_plus)
            input_drive     = (beta * z_t) / 2 + z_spont
            pool_term       = tunings.N_matrix @ (y_plus * (u_plus ** 2))

            dy = (-y + input_drive + recurrent_drive - gain_feedback) / tau_y
            du = (-u + (sigma_c**2) + pool_term) / tau_u
            da = np.zeros_like(a) if force_a_zero else (-a + u_plus + a * u_plus) / tau_a

            y += dt * dy
            u += dt * du
            a += dt * da

        tuning_curves[:, i] = gaussian_rectify(y)

    return tuning_curves


def minmax_norm(binned, ref_binned=None):
    """Per-bin min-max normalization matching Carandini_plots.py process_pair().
    ref_binned: the reference condition (e.g. uniform / normal).
                If None, each bin is normalized against itself.
    """
    ref = binned if ref_binned is None else ref_binned
    bin_max = np.max(ref, axis=1, keepdims=True)
    bin_min = np.min(ref, axis=1, keepdims=True)
    return (binned - bin_min) / (bin_max - bin_min + 1e-9)


def get_binned_curves(tuning_curves, neuron_preferences, probe_angles, n_bins=N_BINS):
    discrete_step = np.pi / len(neuron_preferences)
    bin_edges     = np.linspace(0, np.pi, n_bins + 1) - (discrete_step / 2)
    binned        = np.zeros((n_bins, len(probe_angles)))
    idx           = np.clip(np.digitize(neuron_preferences, bin_edges) - 1, 0, n_bins - 1)
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            binned[b] = np.mean(tuning_curves[mask], axis=0)
    return binned


def get_binned_activity(rates, neuron_bin_idx, window=AVG_WINDOW, n_bins=N_BINS):
    """Average steady-state firing per orientation bin."""
    duration = 20; keep = 5
    steady   = rates[:, -window:]
    n_t      = steady.shape[1]
    time_mask = (np.arange(n_t) % duration) >= (duration - keep)
    means    = np.mean(steady[:, time_mask], axis=1)
    binned   = np.zeros(n_bins)
    for b in range(n_bins):
        mask = neuron_bin_idx == b
        if mask.any():
            binned[b] = np.mean(means[mask])
    return binned


# ── Modified V1Dynamics for dg/dt variants ─────────────────────────────────────

class V1DynamicsCustomGain(V1Dynamics):
    """V1Dynamics with a swappable dg/dt target term."""

    MODES = {
        'avg_N':    'avg / N',
        'avg2_N':   'avg² / N  (default)',
        'avg2_N2':  'avg² / N²',
        'avg15_N':  'avg^1.5 / N',
    }

    def __init__(self, v1_model, frame, dt=0.05, adaptive=True, gain_mode='avg2_N'):
        super().__init__(v1_model, frame, dt, adaptive)
        assert gain_mode in self.MODES, f"Unknown gain_mode: {gain_mode}"
        self.gain_mode = gain_mode

    def _derivatives(self, state, z_t):
        N_neu, K = self.v1.N, self.frame.K
        y   = state[0:N_neu];    u   = state[N_neu:2*N_neu]
        a   = state[2*N_neu:3*N_neu]; g = state[3*N_neu:3*N_neu+K]
        avg = state[3*N_neu+2*K:3*N_neu+2*K+1]

        u_plus      = self.gaussian_rectify(u)
        y_plus      = self.gaussian_rectify(y)
        a_plus      = self.gaussian_rectify(a)
        sqrt_y_plus = np.sqrt(y_plus)

        if self.adaptive:
            v_t           = self.frame.W.T @ y
            gain_feedback = self.frame.W @ (g * v_t)
            davg_dt       = (-avg + np.linalg.norm(y)) / self.tau_avg

            if self.gain_mode == 'avg_N':
                target = avg / N_neu
            elif self.gain_mode == 'avg2_N':
                target = (avg ** 2) / N_neu
            elif self.gain_mode == 'avg2_N2':
                target = (avg ** 2) / (N_neu ** 2)
            elif self.gain_mode == 'avg15_N':
                target = (np.abs(avg) ** 1.5) / N_neu

            dg_dt = (v_t * v_t - target) / self.tau_g
            dv_dt = (-v_t + self.frame.W.T @ y) / self.tau_v
        else:
            gain_feedback = 0.0
            dg_dt   = np.zeros(K)
            dv_dt   = np.zeros(K)
            davg_dt = np.zeros(1)

        recurrent_drive = (1.0 / (1.0 + a_plus)) * (self.v1.W_yy @ sqrt_y_plus)
        input_drive     = (self.beta * z_t) / 2
        sigma_term      = (self.sigma) ** 2
        pool_term       = self.v1.N_matrix @ (y_plus * (u_plus ** 2))

        dy_dt = (-y + input_drive + recurrent_drive - gain_feedback) / self.tau_y
        du_dt = (-u + sigma_term + pool_term) / self.tau_u
        da_dt = (-a + u_plus + a * u_plus + self.alpha * du_dt) / self.tau_a

        return np.concatenate([dy_dt, du_dt, da_dt, dg_dt, dv_dt, davg_dt])


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── Shared initialisation ─────────────────────────────────────────────────
    print("Initializing shared resources...")
    tunings   = V1Tunings(N=N)
    frame     = _make_canonical_frame()
    stim_gen  = StimulusGenerator(N=N, K=N, stream_length=STREAM_LEN)

    adaptor_idx = N // 2
    adaptor_rad = stim_gen.theta_inputs[adaptor_idx]
    adaptor_deg = adaptor_rad * 180 / np.pi

    # Bin geometry shared across all diagnostics
    discrete_step_rad = np.pi / N
    bin_edges_rad     = np.linspace(0, np.pi, N_BINS + 1) - (discrete_step_rad / 2)
    bin_centers       = (bin_edges_rad[:-1] + bin_edges_rad[1:]) / 2
    bin_centers_deg   = bin_centers * 180 / np.pi
    neuron_bin_idx    = np.clip(np.digitize(tunings.theta, bin_edges_rad) - 1, 0, N_BINS - 1)

    seq_bias = stim_gen.generate_input_ensembles(biased=True)

    probe_angles     = np.linspace(0, np.pi, PROBE_RES)
    probe_angles_deg = probe_angles * 180 / np.pi

    blue_colors = plt.cm.Blues(np.linspace(0.4, 1.0, N_BINS))

    # x-axis relative to adaptor (for probe curves)
    x_axis      = (probe_angles_deg - adaptor_deg + 90) % 180 - 90
    sort_idx    = np.argsort(x_axis)
    x_sorted    = x_axis[sort_idx]

    # x-axis relative to adaptor (for binned-activity curves)
    x_peak      = (bin_centers_deg - adaptor_deg + 90) % 180 - 90
    sort_idx_2  = np.argsort(x_peak)
    x_peak_s    = x_peak[sort_idx_2]


    # ══════════════════════════════════════════════════════════════════════════
    # DIAGNOSTIC 1: Scale sweep
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("DIAGNOSTIC 1: Probe tuning curves vs. input scale")
    print("=" * 60)
    # `scale` is defined in both stimuli_whiten.py (generate_sequence: scale=1.0,
    # generate_input_ensembles: effectively 2.5) and in Carandini_plots.py's
    # run_probe (scale=1.0). Here we sweep it to see how the shape changes.

    scales       = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    scale_colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(scales)))
    mid_bin      = N_BINS // 2

    # Cache results to avoid duplicate probes
    scale_binned = {}
    for scale in scales:
        tc = run_probe(frame, tunings, fixed_gains=None, probe_angles=probe_angles,
                       scale=scale)
        scale_binned[scale] = get_binned_curves(tc, tunings.theta, probe_angles)

    # Per-bin min-max normalize each scale against itself (no shared reference here)
    scale_norm = {s: minmax_norm(scale_binned[s]) for s in scales}

    fig1, axes1 = plt.subplots(1, 2, figsize=(13, 5))

    # Left: overlay all scales for the middle (near-adaptor) bin — normalized shapes
    for si, scale in enumerate(scales):
        axes1[0].plot(x_sorted, scale_norm[scale][mid_bin][sort_idx],
                      color=scale_colors[si], linewidth=2, label=f'scale = {scale}')
    axes1[0].set_title(f"Normalized Tuning Curve (bin {mid_bin}, near adaptor) vs Scale",
                       fontweight='bold')
    axes1[0].set_xlabel("Orientation Relative to Adaptor (°)")
    axes1[0].set_ylabel("Normalized Response")
    axes1[0].set_xlim(-90, 90)
    axes1[0].legend(fontsize='small')
    axes1[0].grid(False)

    # Right: peak of the *raw* response per bin vs scale (shows gain, not shape)
    peak_grid = np.array([np.max(scale_binned[s], axis=1) for s in scales]).T  # (N_BINS, n_scales)
    im = axes1[1].imshow(peak_grid, aspect='auto', cmap='viridis', origin='lower',
                         extent=[scales[0], scales[-1], 0, N_BINS])
    plt.colorbar(im, ax=axes1[1], label='Peak Response (raw)')
    axes1[1].set_title("Peak Response per Orientation Bin vs Scale", fontweight='bold')
    axes1[1].set_xlabel("Scale")
    axes1[1].set_ylabel("Orientation Bin")
    axes1[1].grid(False)

    fig1.suptitle("Diagnostic 1: Effect of Input Scale on Tuning Curves",
                  fontweight='bold', fontsize=13)
    plt.tight_layout()
    plt.show()


    # ══════════════════════════════════════════════════════════════════════════
    # DIAGNOSTIC 2: Activity when a = 0 (normalization disabled)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("DIAGNOSTIC 2: Activity with a = 0 (normalization off)")
    print("=" * 60)

    print("Running adaptive simulation (biased)...")
    engine_adp = V1Dynamics(tunings, frame, adaptive=True)
    rates_adp, gains_adp, u_adp, a_adp = engine_adp.run_simulation(seq_bias)
    final_gains = gains_adp[:, -1]
    final_u     = u_adp[:, -1]
    final_a     = a_adp[:, -1]
    del rates_adp, gains_adp, u_adp, a_adp; gc.collect()

    print("Running non-adaptive simulation (biased)...")
    engine_org2 = V1Dynamics(tunings, frame, adaptive=False)
    rates_org2, _, u_org2, a_org2 = engine_org2.run_simulation(seq_bias)
    final_u_org = u_org2[:, -1]
    final_a_org = a_org2[:, -1]
    del rates_org2, u_org2, a_org2; gc.collect()

    print("Probing (4 conditions)...")
    tc_adp_normal = run_probe(frame, tunings, final_gains, probe_angles,
                              frozen_u=final_u, frozen_a=final_a)
    tc_adp_a0     = run_probe(frame, tunings, final_gains, probe_angles,
                              frozen_u=final_u, frozen_a=final_a, force_a_zero=True)
    tc_org_normal = run_probe(frame, tunings, fixed_gains=None, probe_angles=probe_angles,
                              frozen_u=final_u_org, frozen_a=final_a_org)
    tc_org_a0     = run_probe(frame, tunings, fixed_gains=None, probe_angles=probe_angles,
                              frozen_u=final_u_org, frozen_a=final_a_org, force_a_zero=True)

    def _bin(tc):
        return get_binned_curves(tc, tunings.theta, probe_angles)

    b_adp_normal = _bin(tc_adp_normal)
    b_adp_a0     = _bin(tc_adp_a0)
    b_org_normal = _bin(tc_org_normal)
    b_org_a0     = _bin(tc_org_a0)

    # Per-bin min-max normalization: "normal" condition is the reference for each row,
    # mirroring how Carandini_plots.py uses the uniform response as the reference.
    n_adp_normal = minmax_norm(b_adp_normal, ref_binned=b_adp_normal)
    n_adp_a0     = minmax_norm(b_adp_a0,     ref_binned=b_adp_normal)
    n_org_normal = minmax_norm(b_org_normal,  ref_binned=b_org_normal)
    n_org_a0     = minmax_norm(b_org_a0,      ref_binned=b_org_normal)

    fig2, axes2 = plt.subplots(2, 2, figsize=(13, 9), sharey='row',
                               gridspec_kw={'hspace': 0.35, 'wspace': 0.1})

    titles = [["Adaptive — Normal", "Adaptive — a = 0 (norm off)"],
              ["Non-Adaptive — Normal", "Non-Adaptive — a = 0 (norm off)"]]
    data   = [[n_adp_normal, n_adp_a0],
              [n_org_normal, n_org_a0]]

    for r in range(2):
        for c in range(2):
            ax = axes2[r, c]
            for i in range(N_BINS):
                ax.plot(x_sorted, data[r][c][i][sort_idx],
                        color=blue_colors[i], linewidth=1.5)
            ax.set_title(titles[r][c], fontweight='bold')
            ax.set_xlim(-90, 90)
            ax.grid(False)
            if r == 1:
                ax.set_xlabel("Orientation Relative to Adaptor (°)")
        axes2[r, 0].set_ylabel("Normalized Response")

    fig2.suptitle("Diagnostic 2: Effect of Disabling Normalization (a = 0)",
                  fontweight='bold', fontsize=13)
    plt.tight_layout()
    plt.show()


    # ══════════════════════════════════════════════════════════════════════════
    # DIAGNOSTIC 3: dg/dt target formula variants
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("DIAGNOSTIC 3: dg/dt target formula variants")
    print("=" * 60)
    # Current formula: dg_dt = (v_t² - avg²/N) / tau_g
    # Variants swap the target term.

    gain_modes    = ['avg_N', 'avg2_N', 'avg2_N2', 'avg15_N']
    mode_labels   = ['avg / N', 'avg² / N  (default)', 'avg² / N²', 'avg^1.5 / N']
    mode_colors   = ['steelblue', 'mediumseagreen', 'darkorange', 'mediumpurple']

    # Non-adaptive is the same regardless of gain_mode
    print("Running non-adaptive baseline...")
    engine_org3 = V1DynamicsCustomGain(tunings, frame, adaptive=False, gain_mode='avg2_N')
    rates_org3, _, u_org3, a_org3 = engine_org3.run_simulation(seq_bias)
    peaks_org3  = get_binned_activity(rates_org3, neuron_bin_idx)
    del rates_org3, u_org3, a_org3; gc.collect()
    norm_org3 = peaks_org3 / np.mean(peaks_org3)

    fig3, ax3 = plt.subplots(1, 1, figsize=(7, 5))
    ax3.axhline(1, color='grey', linestyle='--', linewidth=1.2, zorder=1)
    ax3.plot(x_peak_s, norm_org3[sort_idx_2], 's-', color='coral',
             linewidth=2, markersize=5, label='Non-Adaptive (all modes)')

    for mode, label, color in zip(gain_modes, mode_labels, mode_colors):
        print(f"  Adaptive, gain_mode = {mode}...")
        eng = V1DynamicsCustomGain(tunings, frame, adaptive=True, gain_mode=mode)
        rates_m, _, u_m, a_m = eng.run_simulation(seq_bias)
        peaks_m  = get_binned_activity(rates_m, neuron_bin_idx)
        del rates_m, u_m, a_m; gc.collect()
        norm_m = peaks_m / np.mean(peaks_m)
        ax3.plot(x_peak_s, norm_m[sort_idx_2], 'o-', color=color,
                 linewidth=2, markersize=5, label=f'Adaptive: {label}')

    ax3.set_title("Biased Ensemble: dg/dt Target Variants", fontweight='bold')
    ax3.set_xlabel("Orientation (°)")
    ax3.set_ylabel("Response / Mean Response")
    ax3.set_xlim(-90, 90)
    ax3.legend(fontsize='small')
    ax3.grid(False)
    fig3.suptitle("Diagnostic 3: Effect of dg/dt Formula on Normalized Response",
                  fontweight='bold', fontsize=13)
    plt.tight_layout()
    plt.show()


    # ══════════════════════════════════════════════════════════════════════════
    # DIAGNOSTIC 4: Random N_matrix
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("DIAGNOSTIC 4: Random N_matrix vs all-ones")
    print("=" * 60)
    # N_matrix normally = np.ones((N, N)), which means every neuron pools from all others.
    # Here we replace it with uniform random entries in [0, 1].

    tunings_rand = V1Tunings(N=N)
    np.random.seed(42)
    tunings_rand.N_matrix = np.random.uniform(0, 1, (N, N))

    print("Running adaptive (random N_matrix)...")
    eng_rand_adp = V1Dynamics(tunings_rand, frame, adaptive=True)
    rates_ra, _, u_ra, a_ra = eng_rand_adp.run_simulation(seq_bias)
    peaks_rand_adp = get_binned_activity(rates_ra, neuron_bin_idx)
    del rates_ra, u_ra, a_ra; gc.collect()

    print("Running non-adaptive (random N_matrix)...")
    eng_rand_org = V1Dynamics(tunings_rand, frame, adaptive=False)
    rates_ro, _, u_ro, a_ro = eng_rand_org.run_simulation(seq_bias)
    peaks_rand_org = get_binned_activity(rates_ro, neuron_bin_idx)
    del rates_ro, u_ro, a_ro; gc.collect()

    print("Running adaptive (all-ones N_matrix, baseline)...")
    eng_ones_adp = V1Dynamics(tunings, frame, adaptive=True)
    rates_oa, _, u_oa, a_oa = eng_ones_adp.run_simulation(seq_bias)
    peaks_ones_adp = get_binned_activity(rates_oa, neuron_bin_idx)
    del rates_oa, u_oa, a_oa; gc.collect()

    # Non-adaptive all-ones baseline already computed above (norm_org3 uses peaks_org3)
    peaks_ones_org = peaks_org3

    norm_rand_adp = peaks_rand_adp / np.mean(peaks_rand_adp)
    norm_rand_org = peaks_rand_org / np.mean(peaks_rand_org)
    norm_ones_adp = peaks_ones_adp / np.mean(peaks_ones_adp)
    norm_ones_org = peaks_ones_org / np.mean(peaks_ones_org)

    fig4, axes4 = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    for ax, title, adp, org in zip(
        axes4,
        ["N_matrix: All Ones (baseline)", "N_matrix: Random [0, 1]"],
        [norm_ones_adp, norm_rand_adp],
        [norm_ones_org, norm_rand_org],
    ):
        ax.axhline(1, color='grey', linestyle='--', linewidth=1.2, zorder=1)
        ax.plot(x_peak_s, adp[sort_idx_2], 'o-', color='steelblue',
                linewidth=2, markersize=5, label='Adaptive')
        ax.plot(x_peak_s, org[sort_idx_2], 's-', color='coral',
                linewidth=2, markersize=5, label='Non-Adaptive')
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel("Orientation (°)")
        ax.set_xlim(-90, 90)
        ax.legend(fontsize='small')
        ax.grid(False)

    axes4[0].set_ylabel("Response / Mean Response")
    fig4.suptitle("Diagnostic 4: Effect of Random Pooling Matrix (N_matrix) — Biased Ensemble",
                  fontweight='bold', fontsize=13)
    plt.tight_layout()
    plt.show()

    print("\nAll diagnostics complete.")
 