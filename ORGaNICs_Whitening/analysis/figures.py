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
from stimuli_whiten import StimulusGenerator

N_RF              = 13
TUNING_WIDTH      = 0.75
ENSEMBLE_CONTRAST = 0.6
SIGMA_NORM        = 0.25    # matches frame_whiten.compute_uniform_target_covariance / V1Dynamics_Surround
N_TRIALS_POISSON  = 200     # independent noisy replays used to estimate a Poisson-variance ensemble's covariance (Panel 2)
FANO_VALUES       = [0.001, 0.01, 0.05, 0.1, 0.5]   # Panel 2's sweep (Fano=1.0 dropped from the plot by request)
EIGVEC_FANO_VALUES = [0.0, 0.001, 0.01, 0.5]         # plot_eigenvector_heatmaps' 2x2 sweep
SEED              = 0       # fixes every noise draw so the figure is reproducible
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


if __name__ == "__main__":
    plot_eigenvalue_diagnostic()
    plot_eigenvector_heatmaps()
    plt.show()
