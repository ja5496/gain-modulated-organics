'''
figures.py

Standalone covariance/eigenvalue diagnostic figures for the single-RF (N_RF-neuron)
stimulus ensembles used throughout this codebase. Currently contains two figures:
plot_eigenvalue_diagnostic (two panels, described below) and plot_eigenvector_heatmaps
(a 2x2 grid visualizing the uniform ensemble's covariance EIGENVECTORS, not just its
eigenvalues, across a small Fano-factor sweep -- see that function's own docstring).

plot_eigenvalue_diagnostic has two panels:

  Panel 1 -- Normalized-covariance eigenvalue spectra for three NOISELESS (deterministic)
  stimulus ensembles: uniform / biased (single adaptor) / double-peaked. Original
  motivation (moved here from Surround_Analytic_Responses.py, then extended): which
  directions of stimulus covariance actually change between ensembles, and by how much?
  This directly bears on how many gain-modulating interneurons (Duong et al.) the
  circuit needs -- if only a handful of directions ever carry real variance (in ANY
  ensemble), that many interneurons suffice; the rest of an overcomplete frame would be
  adapting to structure that was never there. (An earlier version of this panel also
  included a "biased + Poisson variance" ensemble; removed by request so every spectrum
  here is on the same, noiseless footing -- that comparison now lives entirely in
  Panel 2.)

  Panel 2 -- The biased (single-adaptor) ensemble's spectrum swept across several Fano
  factors (FANO_VALUES, currently 0.001-0.5 -- Fano=1.0 dropped from the plot by
  request), to show how injected trial-to-trial noise progressively buries the
  deterministic rank-2 signal structure as its magnitude grows.

CRITICAL caveat inherited from the original diagnostic: with TUNING_WIDTH=0.75, every
DETERMINISTIC ensemble's covariance below is effectively RANK 2 (a raised-Gaussian
profile this wide, sampled and circularly shifted across N_RF=13 orientations, is itself
close to rank 2 -- its discrete circular Fourier transform is dominated by the DC +
first-harmonic component). Eigenvalues past rank ~2 are indistinguishable from each
ensemble's own residual-harmonic floor -- a ratio of noise to noise, not signal to
signal (the retention-threshold reference line that used to mark this floor was removed
from the plot by request; the floor is still visible directly as the point where each
deterministic curve flattens out).

Extension (2026-08-25): standardized every covariance calculation on the NORMALIZED
stimulus profile z / sqrt(SIGMA_NORM^2 + ||z||^2), not the raw stimulus. Explicit
assumptions -- flag any of these if they don't match intent:

  1. "Response covariance" = covariance of the NORMALIZED profile, not the raw stimulus.
     This is the quantity divisive normalization actually equalizes (per the PCA
     whitening diagnostic in Surround_Analytic_Responses.py: y' = z/sqrt(sigma^2+||z||^2)
     is the pre-gain-feedback normalized response), so every ensemble below is compared
     on that footing. The "raw" pipeline from the original diagnostic is dropped
     entirely -- there is now one covariance definition, used everywhere.
  2. Poisson-variance noise (Panel 2 only) is injected via add_poisson_variance: the SAME
     balanced single-adaptor category construction as the deterministic biased ensemble
     (every non-adaptor orientation shown once; the adaptor shown enough extra times to
     match -- the "equal non-adaptor representation" fix from
     Surround_Analytic_Responses.py / the whitening_adaptation_notes.md step-9
     discussion of why a *shared, fixed* trial budget across categories artificially
     deflates non-adaptor representation), replayed over N_TRIALS_POISSON independent
     stimulus presentations. For EACH presentation, independent zero-mean noise is added
     to every neuron's RAW drive value (i.e. before the sigma-based response
     normalization in normalize_profiles below), with per-neuron variance = fano * that
     neuron's own drive on that presentation: noise_i ~ N(0, fano * profile_i). This is
     the standard *Gaussian approximation* to Poisson trial variability, not literal
     Poisson sampling -- a true Poisson-distributed count is a non-negative integer with
     mean = variance = lambda and cannot itself be "centered at zero"; the mean-zero
     Gaussian version imposes that same variance-scales-with-the-mean statistic as an
     additive jitter on top of the deterministic profile (same mechanism as
     stimuli_whiten.py's add_poisson_noise, Var = poisson_fano * mean). Repeated
     presentations are required: a single deterministic pass per category (as used for
     the noise-free Panel-1 ensembles) has no trial-to-trial variability to form a
     covariance from.
  3. "Double-peaked distribution" = the same balanced-representation logic generalized
     to TWO adaptors, placed N_RF//2 index-steps apart (~90 deg on the 0-180 deg
     orientation wheel; not exactly 90 deg since N_RF=13 is odd). Each adaptor
     independently receives the same per-adaptor extra-repeat count as the single-
     adaptor case (extra reps are additive on top of one guaranteed presentation per
     non-adaptor orientation, never subtracted from a fixed budget -- see point 2), so
     the two peaks are directly comparable in strength to the single-adaptor case.
     Deterministic, no injected trial noise.
  4. Every spectrum (both panels) is plotted relative to its OWN top eigenvalue (matching
     the original diagnostic's "relative to top" convention), so shape and effective
     rank are comparable regardless of absolute variance scale.
  5. Plotting floor (both panels): eigenvalues below true numerical rank are ~0 up to
     floating-point roundoff (can land slightly negative from eigvalsh), which on a log
     axis sends the line plunging to the machine-epsilon floor (~1e-16) instead of just
     sitting below the retention threshold. Clipped to PLOT_FLOOR for DISPLAY only; the
     printed spectra are the exact, unclipped values.
'''

import os
import sys
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch, Circle
from matplotlib.colors import to_rgba
from stimuli_whiten import StimulusGenerator
from frame_whiten import Frame

N_RF              = 13
TUNING_WIDTH      = 0.75
ENSEMBLE_CONTRAST = 0.6
SIGMA_NORM        = 0.25    # matches frame_whiten.compute_uniform_target_covariance / V1Dynamics_Surround
N_TRIALS_POISSON  = 200     # independent noisy replays used to estimate a Poisson-variance ensemble's covariance (Panel 2)
FANO_VALUES       = [0.001, 0.01, 0.05, 0.1, 0.5]   # Panel 2's sweep (Fano=1.0 dropped from the plot by request)
EIGVEC_FANO_VALUES = [0.0, 0.001, 0.01, 0.5]         # plot_eigenvector_heatmaps' 2x2 sweep
SEED              = 2       # fixes every noise draw so the figure is reproducible
PLOT_FLOOR        = 1e-6    # display-only floor (see module docstring point 5)

COLORS = {
    'Uniform':                   'black',
    'Biased (single adaptor)':   '#800020',
    'Double-peaked':             '#002060',
}


def profiles_from_indices(stim_gen, indices):
    '''Gaussian tuning-curve profiles (N_RF, n_samples) for a sequence of category
    indices, using the same formula as StimulusGenerator.generate_input_ensembles /
    Surround_Analytic_Responses.py's manual biased-ensemble construction.'''
    centers = stim_gen.theta_inputs[indices]
    delta = stim_gen.theta_RF[:, None] - centers[None, :]
    delta = (delta + np.pi / 2) % np.pi - np.pi / 2
    profile = np.exp(-delta**2 / (2 * stim_gen.tuning_width**2))
    profile = stim_gen.contrast * profile / np.linalg.norm(profile, axis=0, keepdims=True)
    return profile


def build_balanced_biased_indices(stim_gen, adaptor_indices, rng):
    '''
    Balanced multi-adaptor index construction: every non-adaptor orientation appears
    exactly once; each adaptor orientation independently receives
    len(non_adaptor)//2 extra repeats -- the same per-adaptor magnitude as the
    original single-adaptor "equal non-adaptor representation" fix (see module
    docstring point 2). Generalizes cleanly from 1 adaptor to N.
    '''
    all_idx = np.arange(stim_gen.num_angles)
    non_adaptor_idx = np.setdiff1d(all_idx, adaptor_indices)
    reps_per_adaptor = len(non_adaptor_idx) // 2
    adaptor_reps = np.repeat(adaptor_indices, reps_per_adaptor)
    indices = np.concatenate([non_adaptor_idx, adaptor_reps])
    rng.shuffle(indices)
    return indices


def add_poisson_variance(profile, n_trials, fano, rng):
    '''Replay `profile` (N_RF, n_categories) over n_trials independent stimulus
    presentations. For each presentation, independent zero-mean noise is added to every
    neuron's RAW drive value -- BEFORE the sigma-based response normalization applied
    later in normalize_profiles -- with per-neuron variance = fano * that neuron's own
    drive on that presentation: noise_i ~ N(0, fano * profile_i). This is the Gaussian
    approximation to Poisson trial variability (Var = fano*mean, matching
    stimuli_whiten.py's add_poisson_noise), NOT literal discrete Poisson sampling: an
    actual Poisson count is a non-negative integer with mean = variance = lambda and
    cannot be "centered at zero" -- the mean-zero Gaussian version imposes that same
    variance-scales-with-the-mean statistic as an additive jitter on the deterministic
    profile instead. Re-imposes the same length<=1 hard cap as the rest of this
    codebase's ||z||<=1 convention. Returns (N_RF, n_categories*n_trials).'''
    reps = np.tile(profile, (1, n_trials))
    noise_std = np.sqrt(fano * np.clip(reps, 0, None))
    noisy = reps + rng.normal(0, noise_std, size=reps.shape)
    norms = np.linalg.norm(noisy, axis=0, keepdims=True)
    noisy = noisy * np.minimum(1.0, 1.0 / norms)
    return noisy


def normalize_profiles(profile):
    '''(N_RF, n_samples) raw profile -> (n_samples, N_RF) normalized profile
    z / sqrt(SIGMA_NORM^2 + ||z||^2) -- the quantity divisive normalization actually
    equalizes (see module docstring point 1).'''
    profile = profile.T                                     # (n_samples, N_RF)
    energy = np.sum(profile ** 2, axis=1, keepdims=True)
    return profile / np.sqrt(SIGMA_NORM**2 + energy)


def ensemble_covariance_and_spectrum(normalized_profile):
    '''Covariance of a normalized-profile ensemble, and its eigenvalue spectrum sorted
    descending and expressed relative to its own top eigenvalue.'''
    cov = np.cov(normalized_profile, rowvar=False)
    eigvals = np.linalg.eigvalsh(cov)[::-1]
    return cov, eigvals / eigvals[0]


def ensemble_covariance_eigh(normalized_profile):
    '''Covariance and its full eigendecomposition: Cov = E @ diag(eigvals) @ E.T, in
    whatever order np.linalg.eigh itself returns them (ascending eigenvalue) --
    deliberately UN-ranked, unlike ensemble_covariance_and_spectrum's descending "rank 1
    = top eigenvalue" convention, so plot_eigenvector_heatmaps shows E exactly as the
    decomposition produces it.'''
    cov = np.cov(normalized_profile, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    return cov, eigvals, eigvecs


def canonicalize_eigvec_signs(eigvecs):
    '''Fixes the arbitrary +/- sign ambiguity of each eigenvector column (an
    eigendecomposition determines every eigenvector only up to an overall sign) by
    flipping any column whose largest-magnitude entry is negative. Without this, two
    otherwise-identical eigenvectors computed under different Fano factors could come
    back with opposite signs purely by numerical accident, which would show up in the
    heatmap as a meaningless color inversion rather than real structural change.'''
    lead_entry = eigvecs[np.argmax(np.abs(eigvecs), axis=0), np.arange(eigvecs.shape[1])]
    flip = np.where(lead_entry < 0, -1.0, 1.0)
    return eigvecs * flip


def build_ensembles(stim_gen, rng):
    '''Constructs the three raw (pre-normalization), NOISELESS stimulus ensembles plotted
    in Panel 1 -- Poisson-variance noise is Panel-2-only (see fano_sweep_spectra).'''
    adaptor_idx = stim_gen.num_angles // 2
    # ~90 deg away on the 0-180 deg orientation wheel; not exact since N_RF is odd.
    orthogonal_idx = (adaptor_idx + stim_gen.num_angles // 2) % stim_gen.num_angles

    uniform_idx = np.arange(stim_gen.num_angles)
    rng.shuffle(uniform_idx)
    profile_uniform = profiles_from_indices(stim_gen, uniform_idx)

    biased_idx = build_balanced_biased_indices(stim_gen, [adaptor_idx], rng)
    profile_biased = profiles_from_indices(stim_gen, biased_idx)

    double_idx = build_balanced_biased_indices(stim_gen, [adaptor_idx, orthogonal_idx], rng)
    profile_double = profiles_from_indices(stim_gen, double_idx)

    return {
        'Uniform':                 profile_uniform,
        'Biased (single adaptor)': profile_biased,
        'Double-peaked':           profile_double,
    }


def fano_sweep_spectra(profile_biased, rng, fano_values=FANO_VALUES, n_trials=N_TRIALS_POISSON):
    '''Panel 2: normalized-covariance eigenvalue spectrum (relative to own top) of the
    SAME deterministic biased (single-adaptor) profile used in Panel 1, re-noised
    independently at each Fano factor in fano_values via add_poisson_variance. Returns
    an OrderedDict-like dict {fano: spectrum}, in the given order.'''
    spectra_by_fano = {}
    for fano in fano_values:
        noisy = add_poisson_variance(profile_biased, n_trials, fano, rng)
        _, spectrum = ensemble_covariance_and_spectrum(normalize_profiles(noisy))
        spectra_by_fano[fano] = spectrum
    return spectra_by_fano


def plot_eigenvalue_diagnostic():
    rng = np.random.default_rng(SEED)
    stim_gen = StimulusGenerator(N=N_RF, num_angles=N_RF, stream_length=N_RF,
                                  tuning_width=TUNING_WIDTH, contrast=ENSEMBLE_CONTRAST)

    ensembles = build_ensembles(stim_gen, rng)

    spectra = {}
    for name, profile in ensembles.items():
        _, spectra[name] = ensemble_covariance_and_spectrum(normalize_profiles(profile))

    print("Normalized-covariance eigenvalue spectra (relative to own top):")
    for name, spec in spectra.items():
        print(f"  {name:28s} {np.array2string(spec, precision=2, suppress_small=True)}")

    fano_spectra = fano_sweep_spectra(ensembles['Biased (single adaptor)'], rng)
    print("Biased-ensemble spectrum across Fano factors (relative to own top):")
    for fano, spec in fano_spectra.items():
        print(f"  Fano={fano:<5.2f} {np.array2string(spec, precision=2, suppress_small=True)}")

    fig_eig, (ax_spec, ax_fano) = plt.subplots(1, 2, figsize=(15, 6.5), sharey=True)
    rank_idx = np.arange(1, N_RF + 1)

    AXIS_WIDTH = 2.5

    # ---- Panel 1: ensemble spectra ----
    for name, spec in spectra.items():
        ax_spec.plot(rank_idx, np.clip(spec, PLOT_FLOOR, None), 'o-', color=COLORS[name],
                     linewidth=2.5, markersize=6, label=name)
    ax_spec.set_yscale('log')
    ax_spec.set_ylim(bottom=PLOT_FLOOR / 2)
    ax_spec.set_title("Eigenvalue Spectrum", fontsize=24, fontweight='bold')
    ax_spec.set_xticks([])
    ax_spec.set_ylabel("Eigenvalue", fontsize=18, fontweight='bold')
    ax_spec.tick_params(axis='y', labelsize=20, width=AXIS_WIDTH, length=8)
    ax_spec.legend(fontsize=18, frameon=False)
    ax_spec.spines['top'].set_visible(False)
    ax_spec.spines['right'].set_visible(False)
    ax_spec.spines['left'].set_linewidth(AXIS_WIDTH)
    ax_spec.spines['bottom'].set_linewidth(AXIS_WIDTH)

    # ---- Panel 2: biased-ensemble spectrum vs. Fano factor ----
    fano_colors = plt.cm.YlOrRd(np.linspace(0.35, 0.95, len(FANO_VALUES)))
    for fano, color in zip(FANO_VALUES, fano_colors):
        ax_fano.plot(rank_idx, np.clip(fano_spectra[fano], PLOT_FLOOR, None), 'o-', color=color,
                     linewidth=2.5, markersize=6, label=f"Fano = {fano:g}")
    ax_fano.set_yscale('log')
    ax_fano.set_title("Eigenvalue Spectrum", fontsize=24, fontweight='bold')
    ax_fano.set_xticks([])
    ax_fano.tick_params(axis='y', labelsize=20, width=AXIS_WIDTH, length=8)
    ax_fano.legend(fontsize=18, frameon=False)
    ax_fano.spines['top'].set_visible(False)
    ax_fano.spines['right'].set_visible(False)
    ax_fano.spines['left'].set_linewidth(AXIS_WIDTH)
    ax_fano.spines['bottom'].set_linewidth(AXIS_WIDTH)

    plt.tight_layout()
    return fig_eig


def plot_eigenvector_heatmaps():
    '''2x2 grid of heatmaps of E (Cov = E @ diag(lambda) @ E.T) for the UNIFORM
    ensemble's normalized covariance, at EIGVEC_FANO_VALUES Fano factors -- visualizes
    how injected trial noise reshapes the covariance's eigenBASIS, not just its
    eigenvalues (c.f. plot_eigenvalue_diagnostic's Panel 2). Fano=0.0 reduces to
    N_TRIALS_POISSON exact replays of the deterministic uniform profile
    (add_poisson_variance with fano=0 adds exactly zero noise), included as the
    noiseless reference case. Columns are UN-ranked: E is shown exactly in
    np.linalg.eigh's own (ascending-eigenvalue) column order, not resorted by
    eigenvalue rank (see ensemble_covariance_eigh). Eigenvector signs are canonicalized
    (see canonicalize_eigvec_signs) so panel-to-panel color changes reflect real
    structure, not an arbitrary sign flip.'''
    rng = np.random.default_rng(SEED)
    stim_gen = StimulusGenerator(N=N_RF, num_angles=N_RF, stream_length=N_RF,
                                  tuning_width=TUNING_WIDTH, contrast=ENSEMBLE_CONTRAST)
    uniform_idx = np.arange(stim_gen.num_angles)
    rng.shuffle(uniform_idx)
    profile_uniform = profiles_from_indices(stim_gen, uniform_idx)

    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    for ax, fano in zip(axes.flat, EIGVEC_FANO_VALUES):
        noisy = add_poisson_variance(profile_uniform, N_TRIALS_POISSON, fano, rng)
        _, _, eigvecs = ensemble_covariance_eigh(normalize_profiles(noisy))
        eigvecs = canonicalize_eigvec_signs(eigvecs)
        # Eigenvector entries are bounded in [-1, 1] (unit-norm columns) -- a fixed
        # vmin/vmax lets all four panels share one colorbar scale for direct comparison.
        im = ax.imshow(eigvecs, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        ax.set_title(f"Fano = {fano:g}", fontsize=16, fontweight='bold')
        ax.set_xlabel("Eigenvector index", fontsize=11, fontweight='bold')
        ax.set_ylabel("Neuron index", fontsize=11, fontweight='bold')
        ax.tick_params(labelsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(r"Eigenvectors $E$ of the Uniform-Ensemble Covariance ($\mathrm{Cov} = E\,\mathrm{diag}(\lambda)\,E^T$)",
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    return fig


def _cov_ellipse_patch(cov, n_std, color, lw, fill_alpha, zorder, center=(0.0, 0.0)):
    '''An Ellipse patch (centered at `center`) whose semi-axes are n_std * sqrt(eigenvalues
    of cov) along cov's own eigenvectors -- i.e. the n_std standard-deviation contour of a
    Gaussian with covariance `cov`. Edge is drawn fully opaque (thick border); the interior
    is the same color at low alpha (light shading) -- decoupled via separate face/edge RGBA
    rather than the patch's single `alpha`, which would dim the border too.'''
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    width, height = 2 * n_std * np.sqrt(np.clip(eigvals, 0, None))
    return Ellipse(center, width, height, angle=angle,
                    facecolor=to_rgba(color, fill_alpha), edgecolor=to_rgba(color, 1.0),
                    linewidth=lw, zorder=zorder)


def _whitening_matrix(cov, theta_t):
    '''Symmetric map M = theta_t * cov^{-1/2}: for data with sample covariance `cov`,
    the transformed sample covariance M @ cov @ M.T is EXACTLY theta_t^2 * I (not just in
    expectation) -- M is built from that same cov, so the cancellation is algebraic and
    holds regardless of sample size.'''
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.clip(eigvals, 1e-12, None)
    return theta_t * (eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T)


def _add_origin_axes(ax, lim, lw):
    '''Draws only an x-axis and a y-axis through the origin (arrow-tipped), not a bounding
    box: the two "data" spines are relocated to x=0/y=0 and thickened, the other two hidden,
    and ticks removed -- the standard geometric-diagram convention, appropriate here since
    every object in the figure (both covariance ellipses, the frame vectors) is centered on
    the origin.'''
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_linewidth(lw)
    ax.spines['bottom'].set_linewidth(lw)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    # Arrowheads at the positive ends of each axis (standard recipe: place a triangular
    # marker at axis-fraction 1.0 along one axis, data-coordinate 0 along the other).
    ax.plot(1, 0, marker=(3, 0, -90), markersize=11, color='black',
            transform=ax.get_yaxis_transform(), clip_on=False)
    ax.plot(0, 1, marker=(3, 0, 0), markersize=11, color='black',
            transform=ax.get_xaxis_transform(), clip_on=False)


def plot_covariance_whitening_frame(n_points=300, theta_t=1.0, std_major=2.0, std_minor=1.4,
                                     rotation_deg=35.0, n_std=1.5, seed=0, save_path=None):
    '''
    Figure 1 -- geometric intuition for a covariance transformation: an overcomplete 2D
    frame (Mercedes frame, K=3 vectors w_1,w_2,w_3 for N=2, per frame_whiten.Frame) whitens
    an anisotropic data cloud down to an isotropic one of radius theta_t.

    Construction:
      - `data`: n_points samples from a mean-zero 2D Gaussian with covariance
        R(rotation_deg) diag(std_major^2, std_minor^2) R(rotation_deg)^T. std_minor > theta_t
        by construction, so the variance along EVERY direction (minimized over directions at
        exactly std_minor, the ellipse's own minor semi-axis) exceeds theta_t -- the premise
        the figure is meant to show.
      - The whitening map M = theta_t * C^{-1/2} is built from `data`'s own SAMPLE covariance
        C (see _whitening_matrix), so the transformed cloud's sample covariance is EXACTLY
        theta_t^2 * I for this exact finite sample, not just approximately after averaging --
        the drawn n_std-circle is therefore an exact fit to the transformed scatter, matching
        the drawn n_std-ellipse's exact fit to the original scatter.
      - The Mercedes frame (frame_whiten.Frame(dim=2, frame_type='mercedes_tight'), an exact
        unit-norm TIGHT frame -- see that module's docstring) supplies w_1, w_2, w_3, drawn as
        arrows from the origin reaching just past the ORIGINAL ellipse's boundary (length tied
        to std_major, not to theta_t): at the theta_t/std_minor gap used here, an arrow scaled
        to the transformed circle's small radius left too little angular separation between
        adjacent 120-degree-apart tips for the w_i labels to sit without overlapping.

    Colors: light blue for the original (non-transformed) cloud/ellipse, maroon (matching
    this module's own COLORS['Biased (single adaptor)']) for the transformed cloud/circle.
    Axes: only the x- and y-axis through the origin, thick, no box, no ticks, no grid, no
    title, no axis labels -- see _add_origin_axes.
    '''
    ORIG_COLOR  = '#5B9BD5'   # light blue
    TRANS_COLOR = '#800020'   # maroon (matches COLORS['Biased (single adaptor)'] above)
    FRAME_COLOR = '#404040'   # dark gray -- distinct from the pure-black axes, neutral vs. both data colors
    AXIS_LW     = 2.5

    # ---- Data: anisotropic cloud with variance > theta_t along every direction ----
    rng = np.random.default_rng(seed)
    theta = np.radians(rotation_deg)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    assert std_minor > theta_t, "std_minor must exceed theta_t (every-direction variance > theta_t)"
    data = rng.standard_normal((n_points, 2)) * np.array([std_major, std_minor])
    data = data @ R.T

    cov = np.cov(data, rowvar=False)
    M = _whitening_matrix(cov, theta_t)
    data_t = data @ M.T   # sample covariance of data_t is EXACTLY theta_t^2 * I (see _whitening_matrix)

    # ---- Mercedes frame (dim=2 -> K=3 vectors, exact unit-norm tight frame) ----
    np.random.seed(seed)   # Frame's greedy seeding uses the global RNG, not an injectable one
    frame = Frame(dim=2, frame_type='mercedes_tight')
    W = frame.W.copy()   # (2, 3)
    # A frame spans the same lines under w_i -> -w_i (sign is arbitrary per atom), so
    # flipping w_3 through the origin changes nothing about the whitening transform above --
    # purely cosmetic, to lay the three arrows out like the Mercedes tri-star logo instead of
    # two of them bunching together on one side.
    W[:, 2] *= -1
    frame_len = 1.08 * n_std * std_major   # arrow tips sit just outside the original ellipse

    # ---- Figure ----
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.set_aspect('equal', adjustable='box')

    ax.scatter(data[:, 0], data[:, 1], s=32, color=ORIG_COLOR, alpha=1.0,
               edgecolors='none', zorder=2)
    ax.scatter(data_t[:, 0], data_t[:, 1], s=32, color=TRANS_COLOR, alpha=1.0,
               edgecolors='none', zorder=4)

    ax.add_patch(_cov_ellipse_patch(cov, n_std, ORIG_COLOR, lw=3.5, fill_alpha=0.15, zorder=3))
    ax.add_patch(_cov_ellipse_patch(np.cov(data_t, rowvar=False), n_std, TRANS_COLOR,
                                     lw=3.5, fill_alpha=0.20, zorder=5))

    # ---- Frame vectors w_1, w_2, w_3 ----
    label_pad = 1.16
    for i in range(W.shape[1]):
        vx, vy = W[0, i] * frame_len, W[1, i] * frame_len
        ax.add_patch(FancyArrowPatch((0, 0), (vx, vy), arrowstyle='-|>', mutation_scale=22,
                                      linewidth=3.0, color=FRAME_COLOR, zorder=6))
        lx, ly = W[0, i] * frame_len * label_pad, W[1, i] * frame_len * label_pad
        ax.text(lx, ly, rf'$\mathbf{{w_{i + 1}}}$', fontsize=24, fontweight='bold',
                color=FRAME_COLOR, ha='center', va='center', zorder=7)

    # ---- Axes / limits ----
    # Cropped to content (97th-percentile scatter radius, the ellipse, the frame + its
    # labels) rather than the scatter's absolute max, so a rare far-tail point doesn't
    # dictate the whole figure's scale and shrink everything else down to it.
    all_pts = np.vstack([data, data_t])
    radii = np.linalg.norm(all_pts, axis=1)
    content_extent = max(np.percentile(radii, 97), frame_len * label_pad)
    lim = 1.15 * content_extent
    _add_origin_axes(ax, lim, AXIS_LW)

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def _circle_patch(center, radius, color, lw, fill_alpha, zorder):
    '''A Circle patch with a fully-opaque border and a lightly-shaded interior of the SAME
    color -- same face/edge-alpha decoupling trick as _cov_ellipse_patch, so the thick
    border reads at full color while the interior stays light.'''
    return Circle(center, radius, facecolor=to_rgba(color, fill_alpha),
                  edgecolor=to_rgba(color, 1.0), linewidth=lw, zorder=zorder)


def _crf_surround_centers(radius, n_surround, gap):
    '''Shared layout math for the cRF + surround flower (see plot_crf_surround_schematic's
    docstring for the tangent-packing derivation): surround centers sit on a ring of radius
    2*radius + gap, evenly spaced (360/n_surround apart, first one straight up). Factored out
    so every figure built on this flower (e.g. plot_crf_surround_normalization_schematic)
    uses IDENTICAL geometry to plot_crf_surround_schematic, rather than a second copy that
    could drift out of sync with it.'''
    ring_r = 2 * radius + gap
    angles = 90 + np.arange(n_surround) * (360.0 / n_surround)
    surround_centers = [(ring_r * np.cos(np.radians(a)), ring_r * np.sin(np.radians(a)))
                         for a in angles]
    return surround_centers, ring_r


def plot_crf_surround_schematic(radius=1.0, n_surround=6, gap=0.18, save_path=None):
    '''
    Figure -- schematic of the classical receptive field (cRF) model used throughout this
    codebase: one central cRF surrounded by n_surround (default 6) same-size surround RFs.

    Layout: all n_surround+1 circles share one radius. Surround centers sit on a ring of
    radius 2*radius + gap, evenly spaced (360/n_surround apart, first one straight up) --
    starting from the "seven equal circles" packing (ring radius 2*radius, where every
    surround circle is EXACTLY tangent to the cRF circle AND to its two ring neighbors: for
    n_surround=6, two circles of radius r whose centers are both at distance 2r from the
    origin and 60 degrees apart are themselves exactly 2r apart center-to-center) and then
    inflating the ring radius by `gap`. At 60-degree spacing the chord between ring
    neighbors equals the ring radius itself (2*r*sin(30 deg) = r), so this single `gap` term
    opens an IDENTICAL, exact gap both between the cRF and every surround circle and between
    each pair of neighboring surround circles -- not two separately-tuned spacings.

    Colors: cRF dark blue (matches this module's COLORS['Double-peaked']); surround light
    blue (matches plot_covariance_whitening_frame's ORIG_COLOR) -- reusing both across the
    module's figures rather than inventing new ones. A white "+" marks the center of the
    cRF (the fixation point / center of gaze), and both the cRF and one representative
    surround circle carry a text label (all six surrounds are identical by construction, so
    labeling one is labeling all of them). Pure schematic -- no axes, ticks, or grid.
    '''
    CRF_COLOR  = '#002060'   # dark blue (COLORS['Double-peaked'])
    SURR_COLOR = '#5B9BD5'   # light blue (plot_covariance_whitening_frame's ORIG_COLOR)
    BORDER_LW  = 5.0

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')

    surround_centers, ring_r = _crf_surround_centers(radius, n_surround, gap)

    for c in surround_centers:
        ax.add_patch(_circle_patch(c, radius, SURR_COLOR, lw=BORDER_LW, fill_alpha=0.35, zorder=2))
    # cRF fill is near-solid (not the same light 0.35 alpha as the surrounds): at low alpha
    # over white, the dark-navy CRF_COLOR washes out to a gray-lavender instead of reading
    # as "dark blue" -- it needs much less mixing with the white background to hold its color.
    ax.add_patch(_circle_patch((0, 0), radius, CRF_COLOR, lw=BORDER_LW, fill_alpha=0.88, zorder=3))

    # ---- fixation-point "+" at the very center of the cRF ----
    arm = 0.28 * radius
    ax.plot([-arm, arm], [0, 0], color='white', linewidth=3.5, solid_capstyle='round', zorder=4)
    ax.plot([0, 0], [-arm, arm], color='white', linewidth=3.5, solid_capstyle='round', zorder=4)

    # ---- labels ----
    ax.text(0, -0.55 * radius, 'cRF', color='white', fontsize=20, fontweight='bold',
            ha='center', va='center', zorder=5)
    ax.text(surround_centers[0][0], surround_centers[0][1], 'Surround', color=CRF_COLOR,
            fontsize=19, fontweight='bold', ha='center', va='center', zorder=5)

    lim = 1.15 * (ring_r + radius)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def plot_crf_surround_normalization_schematic(radius=1.0, n_surround=6, gap=0.18, save_path=None):
    '''
    Figure -- extends plot_crf_surround_schematic's cRF+surround flower (SAME geometry, via
    _crf_surround_centers) to show WHERE two different mechanisms of this codebase's model
    act:
      - Normalization POOLS activity across every RF in the flower (cRF + all n_surround
        surrounds): drawn as curved gray arrows from each of the 7 circles converging into
        one shared hub ("N") below the flower, captioned "pooled normalization" -- one
        shared computation, fed by all seven.
      - Gain adaptation is LOCAL to each RF: drawn as an identical small amber "g" badge at
        the same relative position inside every one of the 7 circles, captioned once
        ("local gain") since all seven are mechanistically identical -- independent per RF,
        not shared, in visual contrast to the single shared pooling hub.

    Pooling-arrow geometry: each arrow curves (matplotlib arc3 connectionstyle), bowed away
    from the flower's own center (sign of the curvature set by which side of the hub's
    vertical axis the source circle sits on), so it routes AROUND the flower rather than
    through it -- a STRAIGHT line from the top surround circle to a hub centered on the
    vertical axis below would otherwise cut directly through both the cRF and the bottom
    surround circle, since all three sit exactly on that same axis.
    '''
    CRF_COLOR  = '#002060'   # dark blue (COLORS['Double-peaked'])
    SURR_COLOR = '#5B9BD5'   # light blue (plot_covariance_whitening_frame's ORIG_COLOR)
    POOL_COLOR = '#595959'   # neutral gray -- pooling/normalization is the "shared" mechanism
    GAIN_COLOR = '#B8860B'   # amber -- gain is the "local" mechanism: a third, distinct hue
                              # from both the RF-identity blues and POOL_COLOR's gray
    BORDER_LW  = 5.0

    fig, ax = plt.subplots(figsize=(7.5, 8.5))
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')

    surround_centers, ring_r = _crf_surround_centers(radius, n_surround, gap)
    all_centers = [(0.0, 0.0)] + surround_centers   # cRF first, then the n_surround surrounds

    # ---- pooling hub, well below the flower ----
    # hub_gap (1.3*radius, vs. plot_crf_surround_schematic-style spacing of ~0.18*radius) is
    # deliberately generous: the arrows below need room to bow around the bottom surround
    # circle rather than crossing its face (see docstring) -- a tight gap left no room for
    # that curvature to resolve before converging.
    hub_r = 0.55 * radius
    hub_y = -(ring_r + radius) - 1.3 * radius - hub_r
    hub_center = (0.0, hub_y)

    # Arrows drawn first (low zorder) so each one appears to emerge from under its RF's
    # edge. Two things keep the 7 arrows from tangling into each other or crossing the
    # bottom surround circle: (1) curvature bows every arrow OUTWARD, away from x=0 (the
    # cRF and bottom-surround circle sit exactly ON x=0, so they're both forced to one
    # consistent side rather than left at rad=0, which would send them straight through
    # each other); (2) landing points fan out along the hub's rim by source x-position
    # instead of all 7 pinching into one point directly above the hub.
    for cx, cy in all_centers:
        direction = np.array([0.0 - cx, hub_y - cy])
        direction = direction / np.linalg.norm(direction)
        start = (cx + radius * direction[0], cy + radius * direction[1])
        rad = -0.55 if cx < -1e-6 else 0.55
        land_deg = 90 + np.clip(cx / (ring_r + radius), -1, 1) * 60
        end = (hub_center[0] + hub_r * np.cos(np.radians(land_deg)),
               hub_center[1] + hub_r * np.sin(np.radians(land_deg)))
        ax.add_patch(FancyArrowPatch(start, end, connectionstyle=f'arc3,rad={rad}',
                                      arrowstyle='-|>', mutation_scale=16, linewidth=2.0,
                                      color=POOL_COLOR, alpha=0.85, zorder=1))

    ax.add_patch(_circle_patch(hub_center, hub_r, POOL_COLOR, lw=BORDER_LW * 0.7,
                                fill_alpha=0.9, zorder=5))
    ax.text(*hub_center, 'N', color='white', fontsize=20, fontweight='bold',
            ha='center', va='center', zorder=6)
    ax.text(hub_center[0], hub_center[1] - hub_r * 1.5, 'pooled normalization',
            color=POOL_COLOR, fontsize=15, fontweight='bold', ha='center', va='top', zorder=6)

    # ---- the flower itself (identical to plot_crf_surround_schematic) ----
    for c in surround_centers:
        ax.add_patch(_circle_patch(c, radius, SURR_COLOR, lw=BORDER_LW, fill_alpha=0.35, zorder=2))
    ax.add_patch(_circle_patch((0, 0), radius, CRF_COLOR, lw=BORDER_LW, fill_alpha=0.88, zorder=3))

    arm = 0.28 * radius
    ax.plot([-arm, arm], [0, 0], color='white', linewidth=3.5, solid_capstyle='round', zorder=4)
    ax.plot([0, 0], [-arm, arm], color='white', linewidth=3.5, solid_capstyle='round', zorder=4)
    ax.text(0, -0.55 * radius, 'cRF', color='white', fontsize=20, fontweight='bold',
            ha='center', va='center', zorder=4)
    ax.text(surround_centers[0][0], surround_centers[0][1], 'Surround', color=CRF_COLOR,
            fontsize=19, fontweight='bold', ha='center', va='center', zorder=4)

    # ---- local-gain badges: one IDENTICAL small "g" marker per RF, same relative offset in
    # every circle, reading as "the same independent process, repeated locally" -- contrast
    # with the single shared pooling hub above. ----
    badge_r = 0.20 * radius
    badge_dx = 0.66 * radius * np.cos(np.radians(135))
    badge_dy = 0.66 * radius * np.sin(np.radians(135))
    for cx, cy in all_centers:
        bx, by = cx + badge_dx, cy + badge_dy
        ax.add_patch(_circle_patch((bx, by), badge_r, GAIN_COLOR, lw=2.0, fill_alpha=0.95, zorder=6))
        ax.text(bx, by, 'g', color='white', fontsize=13, fontweight='bold',
                ha='center', va='center', zorder=7)

    # Caption the badge on the top surround circle (leader line into the open margin) --
    # all seven badges are identical, so labeling one labels the mechanism.
    top_cx, top_cy = surround_centers[0]
    bx, by = top_cx + badge_dx, top_cy + badge_dy
    ax.annotate('local gain\n(independent per RF)', xy=(bx, by),
                xytext=(bx - 0.9 * radius, by + 0.55 * radius),
                color=GAIN_COLOR, fontsize=13, fontweight='bold', ha='right', va='center',
                arrowprops=dict(arrowstyle='-', color=GAIN_COLOR, linewidth=1.5), zorder=7)

    lim_x = 1.3 * (ring_r + radius)
    lim_y_top = 1.2 * (ring_r + radius)
    lim_y_bottom = hub_y - hub_r - 0.9 * radius
    ax.set_xlim(-lim_x, lim_x)
    ax.set_ylim(lim_y_bottom, lim_y_top)

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def _add_first_quadrant_axes(ax, xlim, ylim, lw, xmin=0.0):
    '''Draws only the positive x- and y-axis (plain lines, no arrowheads), anchored at the
    (xmin, 0) CORNER of the plot rather than the CENTER -- companion to _add_origin_axes,
    which instead relocates the spines to a centered zero (appropriate when content
    straddles the origin). Here content lives entirely in the first quadrant, so the default
    left/bottom spine position (already sitting at the axes' own left/bottom edge) is
    exactly right; only the top/right spines are hidden and ticks removed. `xmin` defaults
    to 0 (the true origin) but can be set to a small positive value for a log-x panel, where
    an x-axis literally starting at 0 is undefined.'''
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_linewidth(lw)
    ax.spines['bottom'].set_linewidth(lw)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(xmin, xlim)
    ax.set_ylim(0, ylim)


def _ellipse_radius_at_angle(a, b, psi, phi):
    '''Distance from an ellipse's own center to its boundary along the ABSOLUTE-frame
    direction phi (radians), for an ellipse with local semi-axes a (along its own x'-axis),
    b (along its own y'-axis), rotated by psi (radians) from the lab x-axis. Standard
    polar-radius-of-a-rotated-ellipse result: rotate phi into the ellipse's own local frame
    (phi - psi) and evaluate the textbook axis-aligned polar form r(theta) =
    1/sqrt((cos(theta)/a)^2 + (sin(theta)/b)^2) there. Used to anchor an arrow exactly on
    the large ellipse's boundary along whatever direction it is drawn in, regardless of the
    ellipse's own tilt.'''
    local = phi - psi
    return 1.0 / np.sqrt((np.cos(local) / a) ** 2 + (np.sin(local) / b) ** 2)


class _AxesTextMeasurer:
    '''Places mpl Text objects on `ax` in ax.transAxes coordinates and measures their
    ACTUAL RENDERED extent (via the canvas renderer, converted back into axes-fraction
    units) -- the workaround this figure's equation needs for two things mpl mathtext
    cannot do in one string: (1) inline color changes (no \\color primitive), and
    (2) auto-sized enclosing parentheses around a piece built from separately-placed Text
    objects (mathtext's \\left(\\right) auto-sizing only sees glyphs *inside its own single
    mathtext expression*). Must be constructed AFTER the figure's final layout is fixed
    (e.g. after fig.tight_layout()), since every measurement is taken at the axes' current
    pixel size -- a layout change afterward would leave the fraction-unit placements here
    stale.'''

    def __init__(self, fig, ax):
        self.fig, self.ax = fig, ax
        self.inv = ax.transAxes.inverted()
        fig.canvas.draw()
        self.renderer = fig.canvas.get_renderer()

    def place(self, text, x, y, color, fontsize, fontweight='bold', ha='left', va='center'):
        return self.ax.text(x, y, text, fontsize=fontsize, color=color, fontweight=fontweight,
                             ha=ha, va=va, transform=self.ax.transAxes)

    def bbox(self, text_artist):
        '''(x0, y0, x1, y1) of an already-placed Text artist, in ax.transAxes coordinates.'''
        self.fig.canvas.draw()
        bb = text_artist.get_window_extent(renderer=self.renderer)
        x0, y0 = self.inv.transform((bb.x0, bb.y0))
        x1, y1 = self.inv.transform((bb.x1, bb.y1))
        return x0, y0, x1, y1

    def measure(self, text, fontsize, fontweight='bold'):
        '''Width and height (axes-fraction) `text` would render at, without leaving it in
        the figure -- place off-screen, measure, remove.'''
        t = self.place(text, -10.0, -10.0, 'black', fontsize, fontweight=fontweight)
        x0, y0, x1, y1 = self.bbox(t)
        t.remove()
        return x1 - x0, y1 - y0


def generate_ellipse_norm_schamatic(save_path=None):
    '''
    Manuscript schematic (single-layer V1 model, conceptual only -- no explicit equations
    drawn from Pal et al. or Morone et al., per this figure's scope): depicts
    $\\mathcal{F}_{V1}$(LGN Drive) as the SUM of two mechanisms this codebase models,
    Adaptation (gain-modulated covariance whitening, Duong et al.) and Normalization
    (divisive normalization, Heeger 2020 / Pal et al.), reproducing the hand sketch
    supplied with the request.

    Layout, left to right, one row of 4 GridSpec columns (a blank text column, the
    Adaptation panel, a blank "+" column, the Normalization panel):

      1. "$\\mathcal{f}_{V1}$(LGN / Drive) =" in large bold text -- $\\mathcal{f}_{V1}$
         (lowercase, curly), the parentheses, and "=" in black at EQ_FONTSIZE (mathtext's
         \\mathcal gives the "curly" function-name styling without requiring a system LaTeX
         install via usetex); "LGN" stacked directly above "Drive" in burgundy at
         WORD_FONTSIZE, the pair horizontally centered between the parentheses (which are
         rendered at a separately-computed, larger font size so the glyphs are tall enough
         to visually enclose both stacked lines). Built with _AxesTextMeasurer since mpl
         mathtext supports neither inline color changes nor auto-sized parentheses around
         text assembled from separate pieces.
      2. Adaptation panel: a large, anisotropic covariance ellipse (burgundy, thick border,
         opaque fill) -- the RAW / overcomplete-frame response covariance -- enclosing a
         smaller, CONCENTRIC circle (navy, thick border, opaque fill) -- the same
         covariance after gain-modulated whitening. Both are constructed to sit entirely in
         the first quadrant (see the ex/ey bounding-radius margin construction below) and
         share one center, since whitening reshapes covariance without shifting the mean.
         Three short black arrows, evenly spaced 120 deg apart (90/210/330 deg), run from
         the ELLIPSE's own boundary to the CIRCLE's boundary along the same radial
         direction (_ellipse_radius_at_angle), schematically depicting the gain
         transformation collapsing the excess, non-isotropic variance -- left unlabeled,
         matching the source sketch. Panel titled "Adaptation" only (no axis labels).
      3. A large bold "+".
      4. Normalization panel: a thick green Naka-Rushton-style saturating curve
         (response = x^n / (x^n + c50^n)) against a LOG-scaled "contrast" x-axis -- the
         classical divisive-normalization contrast-response nonlinearity, plotted against
         log contrast per the standard vision-science display convention (e.g. Heeger 2020 /
         Kohn et al. 2015), which is also what turns the Naka-Rushton hyperbolic-saturating
         form into the familiar sigmoid. Curve shape is purely qualitative -- no parameters
         here are fit to or claimed to reproduce any specific equation from Heeger 2020 or
         Pal et al. Panel titled "Normalization" (same font size/weight as "Adaptation")
         with x-axis label "contrast" (large, bold, smaller than the titles); no y-axis
         label, matching the source sketch.

    Both panels sit in equal-width gridspec columns of the same row, so their drawn BOXES
    are identical in size/placement regardless of what's inside ("axes should match... in
    terms of length and placement" per the original request). The Adaptation panel uses
    `aspect='equal', adjustable='datalim'` (equal aspect is substantively required there,
    since a "whitened" covariance is circular only if x- and y-data units are drawn at the
    same physical scale; `adjustable='datalim'` keeps its box the same pixel size as its
    cell -- rather than shrinking/letterboxing the box itself, as `adjustable='box'` would --
    so its title stays aligned with the Normalization panel's). The Normalization panel is
    left at the default `aspect='auto'`: with a log x-axis and a linear y-axis, "equal
    aspect" has no single meaningful physical-unit correspondence to enforce. No overall
    figure title, per request.
    '''
    PASTEL_BLUE = '#8FC1E3'   # large ellipse + "LGN Drive" label
    NAVY        = '#002060'   # matches this module's COLORS['Double-peaked']
    CHERRY_RED  = '#D2042D'   # normalization curve
    AXIS_LW  = 2.5

    # ---- Adaptation-panel geometry: anisotropic ellipse + concentric circle, both fully
    # inside the first quadrant by construction ----
    a, b, psi_deg = 3.2, 1.7, 35.0   # ellipse semi-major, semi-minor, tilt (deg)
    psi = np.radians(psi_deg)
    # Half-width/height of the rotated ellipse's own axis-aligned bounding box (standard
    # rotated-ellipse bounding-box formula) -- used to place its center far enough from the
    # origin that the WHOLE ellipse (not just its center) clears both axes.
    ex = np.sqrt((a * np.cos(psi)) ** 2 + (b * np.sin(psi)) ** 2)
    ey = np.sqrt((a * np.sin(psi)) ** 2 + (b * np.cos(psi)) ** 2)
    margin = 1.25
    center = (ex * margin, ey * margin)
    xlim = center[0] + ex * margin
    ylim = center[1] + ey * margin

    circle_r = 0.84 * b   # strictly less than b (the ellipse's minimum radius, at any
                           # angle), so the circle stays enclosed for EVERY arrow angle, not
                           # just the three chosen below -- pulled back a bit from an earlier
                           # 0.93 so the two shorter arrows keep a visible tail/shaft, not
                           # just an arrowhead, while the circle still sits close to the
                           # ellipse boundary
    # Arrows anchored to the ellipse's OWN major-axis direction (psi_deg), not fixed compass
    # angles: the ellipse's radius r(phi) is 180-deg periodic, so any three 120-deg-apart
    # angles sample it at {0, +60, -60} deg from wherever the first one sits (mod 180) --
    # anchoring at the major axis (r=a, the global max) guarantees the longest possible first
    # arrow and two equal, still-clearly-visible arrows at +-60 deg from it. Fixed compass
    # angles (e.g. 90/210/330) have no such guarantee: for THIS ellipse's tilt they landed
    # close to the minor-axis direction (where r -> b -> circle_r as circle_r grows), which
    # collapsed two of the three arrows to near-zero length once circle_r was widened.
    arrow_angles_deg = [psi_deg, psi_deg + 120, psi_deg + 240]

    fig = plt.figure(figsize=(13.5, 4.2))
    gs = fig.add_gridspec(1, 4, width_ratios=[2.7, 2.0, 0.35, 2.0], wspace=0.1)
    ax_text  = fig.add_subplot(gs[0, 0]); ax_text.axis('off')
    ax_adapt = fig.add_subplot(gs[0, 1])
    ax_plus  = fig.add_subplot(gs[0, 2]); ax_plus.axis('off')
    ax_norm  = fig.add_subplot(gs[0, 3])

    # ---- Adaptation panel: ellipse -> circle ----
    big_ellipse = Ellipse(center, 2 * a, 2 * b, angle=psi_deg,
                           facecolor=to_rgba(PASTEL_BLUE, 0.35), edgecolor=to_rgba(PASTEL_BLUE, 1.0),
                           linewidth=4.5, zorder=2)
    ax_adapt.add_patch(big_ellipse)

    small_circle = Circle(center, circle_r, facecolor=to_rgba(NAVY, 0.32),
                           edgecolor=to_rgba(NAVY, 1.0), linewidth=4.5, zorder=3)
    ax_adapt.add_patch(small_circle)

    # Arrow tails start this far OUTSIDE the ellipse's own boundary (not exactly on it), so
    # each arrow visibly originates outside the large ellipse and crosses its border on the
    # way to the circle. The FIRST angle (psi_deg, the major-axis direction -- the longest
    # arrow, top-right in the rendered figure) gets a smaller extension than the other two:
    # at its already-greater length, the same 0.12*b used for the shorter pair read as
    # starting too far out.
    TAIL_EXTENSIONS = [0.05 * b, 0.12 * b, 0.12 * b]
    for ang_deg, tail_ext in zip(arrow_angles_deg, TAIL_EXTENSIONS):
        phi = np.radians(ang_deg)
        r_e = _ellipse_radius_at_angle(a, b, psi, phi)
        u = np.array([np.cos(phi), np.sin(phi)])
        start = tuple(np.array(center) + (r_e + tail_ext) * u)
        end = tuple(np.array(center) + circle_r * u)
        ax_adapt.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=16,
                                            linewidth=2.2, color='black', zorder=4))

    # adjustable='datalim' (not 'box'): keeps this panel's drawn BOX the same pixel size as
    # its gridspec cell (matching the Normalization panel's box exactly, so their titles
    # align and neither panel is letterboxed with blank margin) by extending the data
    # limits outward as needed to hit equal aspect, rather than shrinking the box inward.
    ax_adapt.set_aspect('equal', adjustable='datalim')
    _add_first_quadrant_axes(ax_adapt, xlim, ylim, AXIS_LW)
    ax_adapt.set_title('Adaptation', fontsize=24, fontweight='bold')

    # ---- Normalization panel: generic saturating contrast-response curve, log-x ----
    # Contrast-response functions are conventionally plotted against LOG contrast (the
    # standard display convention in the vision-science literature this schematic serves --
    # e.g. Heeger 2020, Kohn et al. 2015): a Naka-Rushton-style saturating curve against log
    # contrast reads as the classic sigmoid, flat near threshold and saturating at high
    # contrast. x_min/x_max/c50 are schematic (no axis numbers are shown -- ticks are off),
    # chosen only so the curve's rise sits centered in the panel. This panel therefore no
    # longer shares a numeric xlim with the Adaptation panel or forces aspect='equal' (a
    # log-x axis has no single linear "data unit" to equate against a linear y-axis) -- it
    # still matches that panel's BOX size/placement exactly, since both sit in equal-width
    # gridspec columns of the same row.
    x_min, x_curve_max, x_axis_max = 0.02, 8.0, 12.0   # curve saturates well before
                                                        # x_axis_max, leaving right-edge margin
    x_curve = np.logspace(np.log10(x_min), np.log10(x_curve_max), 300)
    c50, n_exp = 0.6, 1.6   # n_exp (the Naka-Rushton exponent) sets the sigmoid's steepness
                             # on this log-x axis -- lowered from an earlier 2.5 for a more
                             # gradual transition
    y_curve = (x_curve ** n_exp) / (x_curve ** n_exp + c50 ** n_exp)   # in [0, 1)
    ax_norm.plot(x_curve, y_curve, color=CHERRY_RED, linewidth=4.5, zorder=2)

    ax_norm.set_xscale('log')
    _add_first_quadrant_axes(ax_norm, x_axis_max, 1.0 / 0.92, AXIS_LW, xmin=x_min)
    ax_norm.minorticks_off()   # log-scale axes auto-add minor ticks; set_xticks([]) alone
                                # only clears the majors, leaving stray tick marks visible
    ax_norm.set_title('Normalization', fontsize=24, fontweight='bold')
    ax_norm.set_xlabel('contrast', fontsize=18, fontweight='bold')

    # ---- "+" between the two panels ----
    ax_plus.text(0.5, 0.5, '+', fontsize=48, fontweight='bold', color='black',
                 ha='center', va='center', transform=ax_plus.transAxes)

    # ---- layout fixed BEFORE measuring mixed-color text widths (see
    # _draw_mixed_color_text docstring) ----
    fig.tight_layout()

    EQ_FONTSIZE = 40     # parentheses (base size before auto-sizing), "="
    F_FONTSIZE = 56      # f_V1 -- larger than EQ_FONTSIZE, sized on its own
    WORD_FONTSIZE = 27   # "LGN" / "Drive" -- unchanged from the original single-line size
    m = _AxesTextMeasurer(fig, ax_text)

    # "LGN" over "Drive": vertical half-gap set from one line's own measured height, so the
    # two lines sit snugly stacked regardless of font/DPI changes.
    word_w0, word_h = m.measure('Drive', WORD_FONTSIZE)
    dy = 0.55 * word_h
    stack_h = word_h + 2 * dy   # tight vertical extent of the two-line block itself

    # Auto-size the parentheses: parenthesis-glyph height scales ~linearly with fontsize, so
    # measure one test glyph at EQ_FONTSIZE and rescale to the padded block height below.
    # VPAD inflates the target height beyond the bare stack_h so the words sit clear of the
    # parentheses' own curvature near the top/bottom (a paren glyph narrows as it curves in,
    # so sizing it to the exact bare text height let the curve visually pinch into the text).
    _, paren_h_at_eq = m.measure('(', EQ_FONTSIZE)
    VPAD = 1.15
    paren_fontsize = EQ_FONTSIZE * (stack_h * VPAD / paren_h_at_eq)

    # Horizontal clearance between the parentheses and the word block -- proportional to
    # word height so it scales sensibly with font size -- so neither word ever touches a
    # paren glyph regardless of how wide the (independently vertically-scaled) parens end up.
    pad_x = 0.35 * word_h

    cur_x = 0.0
    # Plain mathtext italic (no \mathcal wrapper): mpl's \mathcal only defines proper
    # glyphs for UPPERCASE letters in the default 'cm' fontset, so \mathcal{f} was silently
    # falling back to a plain-looking lowercase f. Default math-italic "f" is the standard
    # generic-function glyph (matches LaTeX's own $f$) and already has the curled/hooked
    # descender at the bottom that \mathcal{f} was missing.
    t_f = m.place(r'$f_{V1}$', cur_x, 0.5, 'black', F_FONTSIZE)
    _, _, cur_x, _ = m.bbox(t_f)

    t_open = m.place('(', cur_x, 0.5, 'black', paren_fontsize, fontweight='normal')
    _, _, cur_x, _ = m.bbox(t_open)
    cur_x += pad_x

    word_w = max(word_w0, m.measure('LGN', WORD_FONTSIZE)[0])
    center_x = cur_x + word_w / 2
    m.place('LGN', center_x, 0.5 + dy, PASTEL_BLUE, WORD_FONTSIZE, ha='center')
    m.place('Drive', center_x, 0.5 - dy, PASTEL_BLUE, WORD_FONTSIZE, ha='center')
    cur_x = cur_x + word_w + pad_x

    t_close = m.place(')', cur_x, 0.5, 'black', paren_fontsize, fontweight='normal')
    _, _, cur_x, _ = m.bbox(t_close)

    m.place(' =', cur_x, 0.5, 'black', EQ_FONTSIZE)

    if save_path is None:
        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  'ellipse_norm_schematic.png')
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure to {save_path}")
    return fig


if __name__ == "__main__":
    #plot_eigenvalue_diagnostic()
    #plot_eigenvector_heatmaps()
    #plot_covariance_whitening_frame()
    #plot_crf_surround_schematic()
    #plot_crf_surround_normalization_schematic()
    generate_ellipse_norm_schamatic()
    plt.show()
