"""
Analytic_responses_PCA.py

Local, low-rank, "online-PCA-style" adaptation model -- see step #10 of
whitening_adaptation_notes.md. Rather than factorizing the full N x N
covariance with a K~14000 overcomplete frame (the approach in
Analytic_responses.py), project onto a small number of LOCAL, TIGHT pooling
channels, compare each pool's own variance against a FIXED reference (the
uniform ensemble, a "developmental prior"), and shrink only where the current
stimulus drives a pool's variance above that reference -- never below it.

Two design decisions carried over directly from step #10:
  - Baseline = fixed developmental prior (uniform ensemble, computed once and
    frozen), not a second slow-tracked distribution.
  - The comparison is DIAGONAL (each pool judged only against its own
    baseline), not a full cross-pool generalized eigendecomposition -- a full
    M x M version could let one pool's excess leak into a distant pool's
    receptive field, reintroducing the delocalized "ripple" artifact from
    step #4.

Variance requires an actual distribution to be defined over, and a literally
sustained (constant) adaptor has none in a noiseless model. Step #10's
resolution: define it via ordinary Poisson-like trial-to-trial neural
response noise (Var ~= Mean) riding on top of the deterministic drive. Once
the comparison is diagonal and the stimulus deterministic, that reduces to a
monotonic function of mean drive -- no full incremental eigendecomposition is
needed, just a per-pool relative-variance computation.

That noise's magnitude is exposed here as `poisson_noise_scale`, so a "short
timescale" test (large scale -- a few noisy observations dominate) can be
compared against a "long timescale" test (small scale -- as if many trials'
worth of averaging had already low-pass-filtered the noise down).
"""

import numpy as np
import matplotlib.pyplot as plt
from tunings_whiten import V1Tunings
from stimuli_whiten import StimulusGenerator

sigma = 0.1       # normalization constant (matches V1Dynamics default)
N_matrix = None   # set in __main__ after V1Tunings is instantiated


def local_pool_basis(N, theta, M):
    """
    M evenly-spaced, LOCAL, TIGHT pooling vectors over a period-pi orientation
    domain -- the reduced subspace the adaptation gains below project onto
    and feed back through.

    Each pool is a raised-cosine window with half-width equal to the spacing
    between neighboring centers, so any point on the circle is covered by
    exactly its two nearest neighbors. Using cos^2(x) + cos^2(pi/2 - x) = 1
    (i.e. cos^2 + sin^2 = 1) at the shared boundary gives
    sum_k w_k(theta)^2 = 1 EXACTLY everywhere -- a tight frame by
    construction, not by fitting. Each pool's support is only 2*(pi/M) wide,
    so a probe far from a pool's center produces exactly zero response
    through that pool: locality is geometric, independent of gains.
    """
    centers = np.arange(M) * np.pi / M
    spacing = np.pi / M
    half_width = spacing
    B = np.zeros((N, M))
    for k, c in enumerate(centers):
        d = theta - c
        d = (d + np.pi / 2) % np.pi - np.pi / 2
        w = np.zeros(N)
        mask = np.abs(d) < half_width
        w[mask] = np.cos((np.pi / 2) * d[mask] / half_width)
        B[:, k] = w
    return B, centers


def get_baseline_pool_drive(stimuli, pool_basis):
    """
    Mean pooled drive per pool under the uniform ensemble -- the fixed
    "developmental prior" both timescales below compare against. Computed
    once and frozen, never re-estimated online.
    """
    return (np.asarray(stimuli) @ pool_basis).mean(axis=0)


def get_pca_adaptation_gains(stimulus, pool_basis, baseline_drive,
                              poisson_noise_scale=1.0, g_baseline=2.5,
                              Boost_max=3.0, n=2.0, x0=1.0):
    """
    Diagonal (per-pool) relative-variance adaptation gain -- the practical
    form of the "online PCA" idea in step #10, reduced (by the diagonal
    restriction plus a deterministic stimulus) to comparing each pool's own
    Poisson-noise-plus-drive variance against its fixed baseline:

        Var_baseline_k = baseline_drive_k                         (Var = Mean)
        Var_current_k  = baseline_drive_k + poisson_noise_scale * drive_k

    where drive_k = pool_basis[:, k] . stimulus is this pool's own
    deterministic response (e.g. to a sustained adaptor). `poisson_noise_scale`
    sets how much trial-to-trial noise rides on top of that deterministic
    drive -- large for a "short timescale" test (a few noisy observations
    dominate), small for a "long timescale" one (as if many trials' worth of
    averaging had already low-pass-filtered the noise down).

    Shrink-only relative to the fixed baseline: gain only rises above
    g_baseline where Var_current exceeds Var_baseline (guaranteed here, since
    poisson_noise_scale * drive_k >= 0), never below it.

    `x0` (the saturation point of the boost nonlinearity) MUST be a fixed
    constant, not derived from `excess` itself -- excess is proportional to
    poisson_noise_scale, so a data-dependent x0 (e.g. median(excess)) would
    scale right along with it and the ratio excess/x0 would cancel
    poisson_noise_scale out entirely, silently making the whole function
    invariant to the very parameter this script means to vary. (Caught by
    running this script: short and long timescales gave IDENTICAL gains
    before this was fixed.)
    """
    drive = pool_basis.T @ np.asarray(stimulus)
    excess = poisson_noise_scale * drive / baseline_drive  # == rel_var - 1, always >= 0
    boost = Boost_max * excess**n / (x0**n + excess**n)
    return g_baseline + boost


def build_adaptation_feedback_matrix(pool_basis, gains):
    """M = B @ diag(gains) @ B.T -- local by construction (see local_pool_basis)."""
    return (pool_basis * gains) @ pool_basis.T


def get_adapted_response(stimulus, pool_basis, gains):
    """Steady-state response after adaptation feedback: y = (I + M)^-1 @ stimulus."""
    N = pool_basis.shape[0]
    M = build_adaptation_feedback_matrix(pool_basis, gains)
    return np.linalg.solve(np.eye(N) + M, stimulus)


if __name__ == "__main__":

    N = 169
    # M=24 (7 neurons/pool) showed visible "pool-comb" ringing in the raw
    # per-neuron tuning curves: (I+M)^-1 is a single fixed matrix built from
    # only M evenly-spaced pools, so its own eigenstructure carries a comb at
    # the pool spacing that textures EVERY response, regardless of what's
    # being tested. Checked a sweep (M=4..169): amplitude drops monotonically
    # as M decreases (roughness 0.042 at M=24 -> 0.0056 at M=8), though it
    # never fully vanishes at small M -- it's an inherent tradeoff of using a
    # small, discrete set of local pools at all, not a bug in one parameter.
    # M=12 keeps reasonable spatial resolution (~15 deg/pool) while cutting
    # ripple amplitude more than 3x vs the original M=24.
    M_POOLS = 12

    print("Initializing...")
    tunings = V1Tunings(N=N)
    N_matrix = tunings.N_matrix

    stim_gen = StimulusGenerator(N=N, num_angles=N, stream_length=N, contrast=0.05)
    adaptor_idx = N // 2
    adaptor_rad = stim_gen.theta_inputs[adaptor_idx]

    print("Generating stimulus streams...")
    seq_uni, centers_uni = stim_gen.generate_input_ensembles(biased=False, return_angles=True, duration=1)
    stimuli_uni = list(seq_uni.T)

    pool_basis, pool_centers = local_pool_basis(N, tunings.theta, M_POOLS)
    pool_centers_deg = np.degrees(pool_centers)
    baseline_drive = get_baseline_pool_drive(stimuli_uni, pool_basis)

    # Sustained single-adaptor stimulus -- the "current" input under test.
    z_adaptor = np.exp(-(((tunings.theta - adaptor_rad + np.pi / 2) % np.pi - np.pi / 2) ** 2)
                        / (2 * stim_gen.tuning_width ** 2))
    z_adaptor = stim_gen.contrast * 15 * z_adaptor / np.max(z_adaptor)

    # Short timescale: a lot of Poisson noise (few, noisy observations dominate).
    # Long timescale: much less (as if many trials' worth of averaging had
    # already low-pass-filtered the noise down).
    NOISE_SCALE_SHORT = 5.0
    NOISE_SCALE_LONG = 0.2

    print("Computing short- and long-timescale adaptation gains...")
    g_control = 2.5 * np.ones(M_POOLS)  # flat baseline gain, no adaptation
    g_short = get_pca_adaptation_gains(z_adaptor, pool_basis, baseline_drive,
                                        poisson_noise_scale=NOISE_SCALE_SHORT)
    g_long = get_pca_adaptation_gains(z_adaptor, pool_basis, baseline_drive,
                                       poisson_noise_scale=NOISE_SCALE_LONG)

    print(f"short (noise_scale={NOISE_SCALE_SHORT}): gain range {g_short.min():.3f} - {g_short.max():.3f}")
    print(f"long  (noise_scale={NOISE_SCALE_LONG}): gain range {g_long.min():.3f} - {g_long.max():.3f}")

    fig_gains, ax_gains = plt.subplots(1, 2, figsize=(12, 4.5))
    ax_gains[0].plot(pool_centers_deg, g_short, 'o-', color='#1f77b4')
    ax_gains[0].axvline(np.degrees(adaptor_rad), color='r', linestyle='--', alpha=0.6, label='adaptor')
    ax_gains[0].set_title(f'Short timescale (noise_scale={NOISE_SCALE_SHORT})')
    ax_gains[0].set_xlabel('Pool center (deg)'); ax_gains[0].set_ylabel('Gain')
    ax_gains[0].legend()

    ax_gains[1].plot(pool_centers_deg, g_long, 'o-', color='#ff7f0e')
    ax_gains[1].axvline(np.degrees(adaptor_rad), color='r', linestyle='--', alpha=0.6, label='adaptor')
    ax_gains[1].set_title(f'Long timescale (noise_scale={NOISE_SCALE_LONG})')
    ax_gains[1].set_xlabel('Pool center (deg)'); ax_gains[1].set_ylabel('Gain')
    ax_gains[1].legend()
    plt.tight_layout(); plt.show()

    # Suppression tuning curve: response to test probes at varying offsets
    # from the adaptor, short vs. long timescale, both vs. the same unadapted
    # control.
    def gaussian_probe(center, width):
        d = tunings.theta - center
        d = (d + np.pi / 2) % np.pi - np.pi / 2
        return np.exp(-d**2 / (2 * width**2))

    offsets_deg = np.linspace(-90, 90, 37)
    resp_control, resp_short, resp_long = [], [], []
    for off_deg in offsets_deg:
        probe = gaussian_probe(adaptor_rad + np.radians(off_deg), stim_gen.tuning_width)
        y_control = get_adapted_response(probe, pool_basis, g_control)
        y_short = get_adapted_response(probe, pool_basis, g_short)
        y_long = get_adapted_response(probe, pool_basis, g_long)
        # Matched-filter readout (project response back onto the probe's own
        # shape) -- robust to the pool-boundary structure of a 24-pool basis.
        resp_control.append((probe @ y_control) / (probe @ probe))
        resp_short.append((probe @ y_short) / (probe @ probe))
        resp_long.append((probe @ y_long) / (probe @ probe))

    resp_control = np.array(resp_control)
    suppression_short = 100 * (1 - np.array(resp_short) / resp_control)
    suppression_long = 100 * (1 - np.array(resp_long) / resp_control)

    fig_supp, ax_supp = plt.subplots(figsize=(7, 4.5))
    ax_supp.plot(offsets_deg, suppression_short, label=f'short (noise_scale={NOISE_SCALE_SHORT})')
    ax_supp.plot(offsets_deg, suppression_long, label=f'long (noise_scale={NOISE_SCALE_LONG})')
    ax_supp.axhline(0, color='gray', linewidth=0.8)
    ax_supp.set_xlabel('Test orientation - adaptor (deg)')
    ax_supp.set_ylabel('% suppression relative to control')
    ax_supp.set_title('Adaptation tuning curve: short vs. long timescale')
    ax_supp.legend()
    plt.tight_layout(); plt.show()

    # ==================================================================
    # Everything below reproduces the 5 plots made by Analytic_responses.py,
    # for direct side-by-side comparison. That script's two conditions are
    # "uniform ensemble" (no adaptation) vs "biased ensemble" (adapted); this
    # script's natural analog is g_control (no adaptation, the true parallel
    # to "uniform") vs g_short/g_long (both adapted, at different timescales,
    # under the SAME sustained-adaptor input -- there's no second "biased
    # ensemble" stimulus stream here, only the sustained adaptor). Where the
    # original plots exactly 2 curves, these show all 3 (control/short/long)
    # since that's this script's natural comparison set.
    # ==================================================================
    N_BINS = 13
    DARK_GREEN = '#006400'

    # ------------------------------------------------------------------
    # (1) Gain distribution histogram -- same style as Analytic_responses.py.
    # Note the gains here are per-POOL (M=24), not per-frame-column (K~14000),
    # so the histogram is far coarser -- that's an honest consequence of the
    # local-pool basis being much lower-dimensional, not a plotting choice.
    # ------------------------------------------------------------------
    fig3, ax3 = plt.subplots(figsize=(4, 3))
    ax3.hist(g_control, bins=10, color='#888888', rwidth=0.9, label='Control', alpha=0.85)
    ax3.hist(g_short,   bins=10, color=DARK_GREEN, rwidth=0.9, label='Short',   alpha=0.70)
    ax3.hist(g_long,    bins=10, color='#228B22',  rwidth=0.9, label='Long',    alpha=0.70)
    ax3.set_xlabel("Gain Value", fontsize=14, fontweight='bold')
    ax3.set_ylabel("Count",      fontsize=14, fontweight='bold')
    ax3.set_title("Gain Distribution (M=24 pools)", fontsize=13, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(False)
    for spine in ax3.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(2.5)
    ax3.tick_params(axis='both', width=2.5, length=6, labelsize=11)
    plt.tight_layout(); plt.show()

    # ------------------------------------------------------------------
    # (2) "mu" equivalent. The original computes mu via a self-consistency
    # loop averaging responses over a whole STREAM of stimuli. There's no
    # such stream here -- the "context" under test IS the single sustained
    # adaptor -- so the analogous quantity is simply the steady-state
    # response TO that adaptor under each condition's gains.
    # ------------------------------------------------------------------
    M_control = build_adaptation_feedback_matrix(pool_basis, g_control)
    M_short   = build_adaptation_feedback_matrix(pool_basis, g_short)
    M_long    = build_adaptation_feedback_matrix(pool_basis, g_long)
    mu_control = get_adapted_response(z_adaptor, pool_basis, g_control)
    mu_short   = get_adapted_response(z_adaptor, pool_basis, g_short)
    mu_long    = get_adapted_response(z_adaptor, pool_basis, g_long)

    fig_mu, ax_mu = plt.subplots(figsize=(6, 3))
    ax_mu.plot(mu_control, label='control')
    ax_mu.plot(mu_short,   label='short')
    ax_mu.plot(mu_long,    label='long')
    ax_mu.set_xlabel("Neuron index"); ax_mu.set_ylabel("mu (response to sustained adaptor)")
    ax_mu.legend(); plt.tight_layout(); plt.show()

    # ------------------------------------------------------------------
    # (3) Gain feedback (-M @ mu), same construction as the original.
    # ------------------------------------------------------------------
    gain_feedback_control = -M_control @ mu_control
    gain_feedback_short   = -M_short   @ mu_short
    gain_feedback_long    = -M_long    @ mu_long
    fig_gf, ax_gf = plt.subplots(figsize=(6, 3))
    ax_gf.plot(gain_feedback_control, label='control')
    ax_gf.plot(gain_feedback_short,   label='short')
    ax_gf.plot(gain_feedback_long,    label='long')
    ax_gf.set_xlabel("Neuron index"); ax_gf.set_ylabel("Gain feedback (- M @ mu)")
    ax_gf.legend(); plt.tight_layout(); plt.show()

    # ------------------------------------------------------------------
    # (d)-(g) Distinct stimulus vectors, binned tuning curves, Figure 1.
    # Same construction as the original's steady-state tuning-curve section.
    # ------------------------------------------------------------------
    distinct_stimuli = []
    for angle in stim_gen.theta_inputs:
        d = stim_gen.theta_inputs - angle
        d = (d + np.pi / 2) % np.pi - np.pi / 2
        z = np.exp(-d**2 / (2 * stim_gen.tuning_width**2))
        z = stim_gen.contrast * 15 * z / np.max(z)
        distinct_stimuli.append(z)

    n_distinct = len(distinct_stimuli)
    responses_control = np.zeros((N, n_distinct))
    responses_short = np.zeros((N, n_distinct))
    responses_long = np.zeros((N, n_distinct))
    print("Computing steady-state tuning curves (control/short/long)...")
    for j, z in enumerate(distinct_stimuli):
        responses_control[:, j] = get_adapted_response(z, pool_basis, g_control)
        responses_short[:, j]   = get_adapted_response(z, pool_basis, g_short)
        responses_long[:, j]    = get_adapted_response(z, pool_basis, g_long)

    probe_angles = stim_gen.theta_inputs
    probe_angles_deg = probe_angles * 180 / np.pi

    def get_binned_curves(tuning_curves, neuron_preferences, probe_angs, n_bins=13):
        N_neurons = len(neuron_preferences)
        discrete_step = np.pi / N_neurons
        bin_edges = np.linspace(0, np.pi, n_bins + 1) - (discrete_step / 2)
        binned_response = np.zeros((n_bins, len(probe_angs)))
        neuron_bin_indices = np.digitize(neuron_preferences, bin_edges) - 1
        neuron_bin_indices = np.clip(neuron_bin_indices, 0, n_bins - 1)
        for b in range(n_bins):
            mask = neuron_bin_indices == b
            if np.any(mask):
                binned_response[b, :] = np.mean(tuning_curves[mask, :], axis=0)
        return binned_response

    binned_control = get_binned_curves(responses_control, tunings.theta, probe_angles, N_BINS)
    binned_short   = get_binned_curves(responses_short,   tunings.theta, probe_angles, N_BINS)
    binned_long    = get_binned_curves(responses_long,    tunings.theta, probe_angles, N_BINS)
    # Normalize all three against the control condition's own range, exactly
    # as the original normalizes both conditions against the uniform-ensemble
    # range (bin_max/bin_min taken from the "no adaptation" condition).
    bin_max = np.max(binned_control, axis=1, keepdims=True)
    bin_min = np.min(binned_control, axis=1, keepdims=True)
    norm_control = (binned_control - bin_min) / (bin_max - bin_min + 1e-9)
    norm_short   = (binned_short   - bin_min) / (bin_max - bin_min + 1e-9)
    norm_long    = (binned_long    - bin_min) / (bin_max - bin_min + 1e-9)

    adaptor_deg = adaptor_rad * 180 / np.pi
    uni_angles_deg = centers_uni * 180 / np.pi
    # The "current" input for short/long is the single sustained adaptor --
    # honestly represented as a spike histogram (all mass at one orientation),
    # not fabricated to look like a spread-out stream.
    adaptor_spike_deg = np.full(200, adaptor_deg)

    discrete_step_hist = 180 / N
    bins_hist = np.linspace(0, 180, N_BINS + 1) - (discrete_step_hist / 2)
    weights_uni = np.ones_like(uni_angles_deg) / len(uni_angles_deg)
    weights_adaptor = np.ones_like(adaptor_spike_deg) / len(adaptor_spike_deg)

    x_axis = (probe_angles_deg - adaptor_deg + 90) % 180 - 90
    sort_idx = np.argsort(x_axis)
    x_axis_sorted = x_axis[sort_idx]

    blue_colors = plt.cm.Blues(np.linspace(0.2, 1.0, N_BINS))

    fig, axes = plt.subplots(2, 3, figsize=(14, 6), sharey='row',
                             gridspec_kw={'height_ratios': [0.8, 1.0]})

    axes[0, 0].hist(uni_angles_deg, bins=bins_hist, weights=weights_uni, color='black', rwidth=0.9)
    axes[0, 0].set_title("Control\n(uniform ensemble, no adaptation)", fontweight='bold', fontsize=13)
    axes[0, 0].set_ylabel("Probability", fontsize=16)

    for col, title in ((1, "Short"), (2, "Long")):
        axes[0, col].hist(adaptor_spike_deg, bins=bins_hist, weights=weights_adaptor, color='black', rwidth=0.9)
        axes[0, col].set_title(f"{title}\n(sustained adaptor, same input)", fontweight='bold', fontsize=13)

    for ax in axes[0]:
        ax.set_xlim(0, 180)
        ax.tick_params(labelbottom=False)

    for i in range(N_BINS):
        axes[1, 0].plot(x_axis_sorted, norm_control[i][sort_idx], color=blue_colors[i], linewidth=2.0)
        axes[1, 1].plot(x_axis_sorted, norm_short[i][sort_idx],   color=blue_colors[i], linewidth=2.0)
        axes[1, 2].plot(x_axis_sorted, norm_long[i][sort_idx],    color=blue_colors[i], linewidth=2.0)

    axes[1, 0].set_ylabel("Analytic Response", fontsize=16)
    for c in range(3):
        ax = axes[1, c]
        ax.set_xlim(-90, 90)
        ax.grid(False)
        ax.set_xlabel("Stimulus Orientation (deg)", fontsize=14)

    plt.tight_layout(); plt.show()

    # ------------------------------------------------------------------
    # Figure 2 -- average activity per neuron. The original averages tuning
    # curves over the biased STREAM's orientation frequencies. There's no
    # such stream here, so this instead shows the unweighted mean response
    # across all test orientations -- still a meaningful, distinct summary of
    # each condition's overall population activity pattern.
    # ------------------------------------------------------------------
    avg_control = np.mean(responses_control, axis=1)
    avg_short   = np.mean(responses_short, axis=1)
    avg_long    = np.mean(responses_long, axis=1)

    # Min-max (not mean) normalization: unlike the original script's
    # stream-averaged mu (smoothly positive), these per-orientation averages
    # straddle zero (checked directly: control ranges -0.027 to 0.106, mean
    # only 0.018), so dividing by the mean amplifies noise into spikes.
    def minmax_norm(x, lo, hi):
        return (x - lo) / (hi - lo + 1e-9)

    ref_lo, ref_hi = avg_control.min(), avg_control.max()
    norm_avg_control = minmax_norm(avg_control, ref_lo, ref_hi)
    norm_avg_short    = minmax_norm(avg_short,   ref_lo, ref_hi)
    norm_avg_long     = minmax_norm(avg_long,    ref_lo, ref_hi)

    neuron_prefs_deg = tunings.theta * 180 / np.pi
    x_neuron = (neuron_prefs_deg - adaptor_deg + 90) % 180 - 90
    sort_neuron_idx = np.argsort(x_neuron)

    fig2, ax2 = plt.subplots(1, 1, figsize=(6, 4))
    # No axhline(1) reference here: unlike the original's mean-normalization
    # (where 1 = "no change from baseline"), this min-max normalization is
    # anchored to Control's own range, so Control trivially spans 0-1 by
    # construction -- the meaningful comparison is Short/Long's DEVIATION
    # from Control's shape, not their position relative to a fixed line.
    ax2.plot(x_neuron[sort_neuron_idx], norm_avg_control[sort_neuron_idx],
             color='#333333', linewidth=2.5, label='Control')
    ax2.plot(x_neuron[sort_neuron_idx], norm_avg_short[sort_neuron_idx],
             color='#800020', linewidth=2.5, label='Short')
    ax2.plot(x_neuron[sort_neuron_idx], norm_avg_long[sort_neuron_idx],
             color='#B08000', linewidth=2.5, label='Long')
    ax2.set_title("Average Activity (across all test orientations)", fontweight='bold', fontsize=16)
    ax2.set_ylabel("Normalized Response", fontweight='bold', fontsize=13)
    ax2.set_xlabel("Neuron Preference (deg)", fontweight='bold', fontsize=13)
    ax2.set_xlim(-90, 90)
    ax2.legend(loc='upper right')
    ax2.grid(False)
    for spine in ax2.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(2.5)
    plt.tight_layout(); plt.show()
