import numpy as np
import matplotlib.pyplot as plt
from typing import Literal

'''
---- stimuli_whiten.py ----
Generates synthetic responses to orientation gratings using raised cosine functions.
These responses are fed into our V1 dynamics as the input layer.

'''

class StimulusGenerator:
    def __init__(self, N=60, num_angles = 26, stream_length = 10920, tuning_width = 0.75, Ensemble=False, contrast=1.0, N_RF = 13, N_SETS = 7):
        self.N = N # Number of primary neurons
        self.num_angles = num_angles # Number of distinct input orientations
        self.stream_length = stream_length # Total length of the input stream
        self.tuning_width = tuning_width # Width of raised cosine input
        self.contrast = contrast
        self.N_RF = N_RF
        self.N_SETS = N_SETS

        # Preferred orientations of the stimuli from 0 to pi
        self.theta_tunings = np.linspace(0, np.pi, N, endpoint=False)
        self.theta_inputs = np.linspace(0, np.pi, num_angles, endpoint=False)
        # Preferred orientations of the N_RF receptive-field neurons. Kept separate from
        # theta_inputs: num_angles sets the resolution of the discrete stimulus identities
        # drawn at each timestep, while theta_RF is the (independent) set of tuning centers
        # the RF neurons project that continuous stimulus angle onto to form their drive.
        self.theta_RF = np.linspace(0, np.pi, N_RF, endpoint=False)

    def generate_input_ensembles(self, biased=False, mean_center=False,
                                 von_mises=False, von_mises_center=0.0,
                                 von_mises_kappa=4.0, return_angles=False, duration=20,
                                 add_poisson_noise=False, poisson_fano=1.0):
        '''
        Generate uniform or biased ensemble of input profiles
        centered at random orientations.

        Returns:
            np.ndarray: Shape ( num_angles{number of distinct stimuli} , stream_length )
            If return_angles=True, returns (profiles, centers) where centers is
            the per-timestep stimulus angle array of shape (stream_length,).
        '''
        if von_mises:
            num_inputs = int(self.stream_length / duration)
            mu = np.deg2rad(von_mises_center)
            centers_raw = np.random.vonmises(mu, von_mises_kappa, num_inputs)
            centers_raw = ((centers_raw % np.pi) + np.pi) % np.pi  # wrap to [0, π)
            centers = np.repeat(centers_raw, duration)  # shape: (stream_length,)

            delta_theta = self.theta_inputs[:, np.newaxis] - centers[np.newaxis, :]
            delta_theta = (delta_theta + np.pi/2) % np.pi - np.pi/2
            profiles = np.exp(-delta_theta**2 / (2 * self.tuning_width**2)) #+ 0.3
            scale = 15 # COEFFICIENT OF ~15 ACHIEVES CORRECT SATURATION FOR CONTRAST OF 1
            profiles = self.contrast * scale * profiles / np.max(profiles)
            if mean_center:
                profiles -= profiles.mean(axis=1, keepdims=True)
            if return_angles:
                return profiles, centers
            return profiles

        # Generate the indices of all the distinct stimuli
        base_indices = np.arange(self.num_angles)
        
        # Append it on itself until it reaches self.stream_length
        num_inputs = int(self.stream_length / duration) # number of stimuli shown 
        n_full  = num_inputs // self.num_angles
        n_extra = num_inputs % self.num_angles
        full_indices  = np.tile(base_indices, n_full)
        extra_indices = np.random.choice(base_indices, size=n_extra, replace=False)
        indices = np.concatenate([full_indices, extra_indices])

        # Optionally overwrite roughly 33% of the indices with the adaptor index
        if biased:
            one_third_split = len(indices) // 3 # Calculate the index representing the first third
            adaptor_idx = self.num_angles // 2 # Define the adaptor index
            indices[:one_third_split] = adaptor_idx # Apply the mask to the first third of the array

        # Randomly shuffle the indices array in-place
        np.random.shuffle(indices) 

        # Adding the duration of the inputs in so it doesn't flash a new one every time step. 
        indices = np.repeat(indices, duration)

        # Convert indices to actual orientation centers; shape: (stream_length,)
        centers = self.theta_inputs[indices]

        # Generate stimulus curves using broadcasting - matrix of shape (K_stimuli, stream_length).
        delta_theta = self.theta_inputs[:, np.newaxis] - centers[np.newaxis, :]
        delta_theta = (delta_theta + np.pi/2) % np.pi - np.pi/2  # wrap to [-π/2, π/2]
        #profiles = np.exp(self.tuning_width * np.cos(2 * delta_theta)) # RAISED COSINE PROFILE
        profiles = np.exp(-delta_theta**2 / (2 * self.tuning_width**2)) #+ 0.3 # GAUSSIAN PROFILE
        
        # 5. Normalize, scale, then (optionally) mean-center across the ensemble
        scale = 1 # COEFFICIENT OF ~15 ACHIEVES CORRECT SATURATION FOR CONTRAST OF 1
        profiles = self.contrast * scale * profiles / np.linalg.norm(profiles, keepdims=True, axis=0)
        
        if add_poisson_noise:
            noise_std = np.sqrt(poisson_fano * np.clip(profiles, 0, None))
            profiles = profiles + np.random.normal(0, noise_std)
            norms = np.linalg.norm(profiles, axis=0, keepdims=True)
            profiles = profiles / np.max(norms) # Scale so maximum length of input is 1

        if mean_center:
            profiles -= profiles.mean(axis=1, keepdims=True)

        if return_angles:
            return profiles, centers
        return profiles
    
    def generate_surround_ensembles(self, adapt_location: Literal['adapt CRF only', 'adapt surround only', 'adapt CRF and surround'],
                                 biased=False, return_angles=False, mean_center=False,
                                 duration=20, add_poisson_noise=False, poisson_fano=1.0):
        '''
        Generate uniform or biased ensemble of raised cosine input profiles
        centered at random orientations, projected onto the N_RF receptive-field
        neurons' tuning curves to form their drive.

        Args:
            poisson_fano (float): Fano factor (Var/mean) of the injected noise, only used
                when add_poisson_noise=True. 1.0 (default) is true Poisson noise. Scale this
                up/down to make neurons noisier/quieter relative to their firing rate. Note that
                the noise is hard-capped afterward (see below) - any column pushed past length 1
                is rescaled back down to exactly 1, so large poisson_fano narrows the noise
                distribution's effective spread rather than letting it blow past the model's
                ||z|| <= 1 assumption.

        Returns:
            np.ndarray: Shape ( N_RF * N_SETS , stream_length )
            If return_angles=True, returns (profiles, centers) where centers is
            the per-timestep stimulus angle array of shape (stream_length,).
        '''

        # Generate the indices of all the distinct stimuli
        base_indices = np.arange(self.num_angles)
        
        # Append it on itself until it reaches self.stream_length
        num_inputs = int(self.stream_length / duration) # number of stimuli shown 
        n_full  = num_inputs // self.num_angles
        n_extra = num_inputs % self.num_angles
        full_indices  = np.tile(base_indices, n_full)
        extra_indices = np.random.choice(base_indices, size=n_extra, replace=False)
        indices = np.concatenate([full_indices, extra_indices])

        # Optionally overwrite roughly 33% of the indices with the adaptor index
        if biased:
            one_third_split = len(indices) // 3 # Calculate the index representing the first third
            adaptor_idx = self.num_angles // 2 # Define the adaptor index
            indices[:one_third_split] = adaptor_idx # Apply the mask to the first third of the array

        # Randomly shuffle the indices array in-place
        np.random.shuffle(indices) 

        # Adding the duration of the inputs in so it doesn't flash a new one every time step. 
        indices = np.repeat(indices, duration)

        # Convert indices to actual orientation centers; shape: (stream_length,)
        centers = self.theta_inputs[indices]

        # Project the (num_angles-resolution) stimulus angle at each timestep onto the
        # N_RF receptive-field neurons' own tuning curves - matrix of shape (N_RF, stream_length).
        # num_angles only controls how finely the discrete stimulus identity is sampled above;
        # it has no bearing on how many neurons receive the resulting drive. Left raw
        # (unnormalized) here - normalization happens below, after extension to the full
        # population, so it sees the true full-population input vector (see note below).
        delta_theta = self.theta_RF[:, np.newaxis] - centers[np.newaxis, :]
        delta_theta = (delta_theta + np.pi/2) % np.pi - np.pi/2  # wrap to [-π/2, π/2]
        #profile = np.exp(self.tuning_width * np.cos(2 * delta_theta)) # RAISED COSINE PROFILE
        profile = np.exp(-delta_theta**2 / (2 * self.tuning_width**2)) #+ 0.3 # GAUSSIAN PROFILE

        # Extend each individual (still-raw) profile (i.e. each column/timestep of the stream)
        # from N_RF neurons to the full N_RF * N_SETS population, placing the driven profile
        # in the CRF and/or surround slots and a flat baseline elsewhere. baseline is
        # broadcast across the full stream length so this applies per-timestep. This happens
        # BEFORE normalization (unlike before) so that the L2-normalize-to-unit-length step
        # below acts on the true full-population input vector b*z, not just the driven
        # N_RF-neuron sub-block - the ORGaNICs papers' fixed-point derivation assumes ||z|| is
        # O(1) for the whole population, and normalizing only the driven sub-block left the
        # full vector's length scaling with sqrt(N_SETS) once baseline was concatenated in.
        baseline = np.full((self.N_RF, profile.shape[1]), 0.10)
        match adapt_location:
            case 'adapt CRF only':
                full_profile = np.concatenate([profile] + [baseline] * (self.N_SETS - 1), axis=0)
            case 'adapt surround only':
                full_profile = np.concatenate([baseline] + [profile] * (self.N_SETS - 1), axis=0)
            case 'adapt CRF and surround':
                full_profile = np.concatenate([profile] * (self.N_SETS), axis=0)

        # 5. Normalize the full population vector to unit length, then scale by contrast.
        scale = 1 # COEFFICIENT OF ~15 ACHIEVES CORRECT SATURATION FOR CONTRAST OF 1
        profiles = self.contrast * scale * full_profile / np.linalg.norm(full_profile, keepdims=True, axis=0)

        if add_poisson_noise:
            # Gaussian approximation to Poisson noise, applied independently at every
            # (neuron, timestep) - including within a single held-constant presentation, so a
            # "constant" stimulus still produces trial-by-trial variability in the drive.
            # Mean is unchanged (E[profiles + noise] = profiles); Var = poisson_fano * profiles,
            # i.e. variance proportional to the instantaneous drive, matching true Poisson
            # statistics (Var = mean) at poisson_fano=1.0.
            #
            # This is deliberately NOT implemented as a rescaled discrete Poisson draw
            # (k * Poisson(profiles/k)): that trick keeps Var = k*profiles correct on average,
            # but raising k to make the noise louder also shrinks the underlying Poisson rate
            # (profiles/k), which makes the process increasingly bursty/quantized (rare,
            # huge-magnitude spikes) rather than smoothly louder. The Gaussian form here scales
            # cleanly to any poisson_fano - turn it up as far as needed to make rate-proportional
            # noise the dominant contributor to variance, with no burstiness ceiling.
            noise_std = np.sqrt(poisson_fano * np.clip(profiles, 0, None))
            profiles = profiles + np.random.normal(0, noise_std)

            # Hard cap: noise can push a column's norm well past the length-1 ceiling the
            # normalization step above was trying to enforce (confirmed in stimuli_whiten.py's
            # __main__ diagnostic - poisson_fano=5.0 alone reached column norms of ~8.8), which is
            # almost certainly what tips V1Dynamics_Surround's RK4 integration into runaway - the
            # whole model is tuned/tested around ||z|| <= 1. Only rescale columns that actually
            # exceed 1 (at realistic poisson_fano this ends up being ~every column, not a rare
            # correction - verified this doesn't distort the covariance structure though: the
            # per-column norm barely varies with which orientation is shown, since it's dominated
            # by the 6 orientation-blind surround blocks, so the divisor is nearly constant across
            # columns anyway). Chosen over a single global rescale (divide every column by the
            # stream's largest norm) because that alternative is dominated by one rare outlier draw
            # - it would crush the *typical* column well below length 1, and gets worse the longer
            # the stream runs (more chances to draw a bigger outlier).
            norms = np.linalg.norm(profiles, axis=0, keepdims=True)
            profiles = profiles * np.minimum(1.0, 1.0 / norms)

        if mean_center:
            profiles -= profiles.mean(axis=1, keepdims=True)

        if return_angles:
            return profiles, centers
        

        return profiles


    def generate_contrast_stream(self, peak_ln_contrast, contrast_sigma=1.0,
                                 return_metadata=False, **kwargs):
        '''
        Generates a stimulus stream scaled by contrasts drawn from a truncated
        log-normal distribution bounded between e^-3 and 1 (i.e., ln(contrast) ∈ [-3, 0]).

        Args:
            peak_ln_contrast (float): ln(contrast) at which the distribution peaks (mode in log-space).
            contrast_sigma (float): Standard deviation of the underlying normal distribution.
            return_metadata (bool): If True, return (stream, angles_per_pres, contrasts_per_pres)
                where angles_per_pres and contrasts_per_pres are per-stimulus-presentation arrays.
            **kwargs: Arguments to pass to generate_input_ensembles.

        Returns:
            np.ndarray or tuple: Contrast-scaled stream, or 3-tuple if return_metadata=True.
        '''
        # 1. Generate base normalized profiles from existing function
        if return_metadata:
            profiles, centers = self.generate_input_ensembles(return_angles=True, **kwargs)
        else:
            profiles = self.generate_input_ensembles(**kwargs)

        # 2. Determine number of distinct stimulus presentations from the actual
        #    profile length (uniform path truncates to complete cycles, so
        #    profiles.shape[1] may be shorter than self.stream_length)
        duration = 20  # Matches the duration in generate_input_ensembles
        num_inputs = profiles.shape[1] // duration

        # 3. peak_ln_contrast is the mean of the underlying normal (mode in log-space)
        mu = peak_ln_contrast

        # 4. Fast rejection sampling for strict truncation in [1e-3, 1]
        contrasts = np.empty(num_inputs)
        mask = np.ones(num_inputs, dtype=bool)

        while mask.any():
            # Draw samples only for the indices that still need valid values
            samples = np.random.lognormal(mean=mu, sigma=contrast_sigma, size=mask.sum())
            valid = (samples >= np.exp(-3)) & (samples <= 1.0)

            # Assign valid samples and update the mask
            contrasts[np.where(mask)[0][valid]] = samples[valid]
            mask[np.where(mask)[0][valid]] = False

        # 5. Expand contrast array to match the temporal stream length
        contrast_stream = np.repeat(contrasts, duration)

        # 6. Scale the normalized profiles via NumPy broadcasting
        stream = profiles * contrast_stream
        if return_metadata:
            angles_per_pres = centers[::duration]   # one angle per presentation
            return stream, angles_per_pres, contrasts
        return stream


    def plot_covariance_matrices(self):
        '''Plot heatmaps of the covariance matrix for both uniform and biased ensembles.'''
        uni  = self.generate_input_ensembles(biased=False)
        bias = self.generate_input_ensembles(biased=True)
        cov_uni  = np.cov(uni,  rowvar=True)
        cov_bias = np.cov(bias, rowvar=True)

        ticks = np.linspace(0, self.num_angles - 1, 5).astype(int)
        tick_labels = [f"{int(self.theta_inputs[t] * 180 / np.pi)}°" for t in ticks]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for ax, mat, title in zip(axes, [cov_uni, cov_bias],
                                  ['Uniform Ensemble', 'Biased Ensemble']):
            im = ax.imshow(mat, cmap='viridis', aspect='auto')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(title, fontsize=20, fontweight='bold', pad=12)
            ax.set_xlabel('Orientation', fontsize=18, fontweight='bold')
            ax.set_ylabel('Orientation', fontsize=18, fontweight='bold')
            ax.set_xticks(ticks); ax.set_xticklabels(tick_labels, fontsize=14)
            ax.set_yticks(ticks); ax.set_yticklabels(tick_labels, fontsize=14)
            for spine in ax.spines.values():
                spine.set_linewidth(2.5)
            ax.tick_params(width=2.5, length=6)

        plt.tight_layout()
        plt.show()

    def plot_surround_covariance_matrices(self, adapt_location='adapt CRF only', add_poisson_noise=True, poisson_fano=5.0):
        '''Plot heatmaps of the covariance matrix restricted to the first N_RF rows
        (the classical receptive field neurons) of the population generated by
        generate_surround_ensembles, for both uniform and biased ensembles.

        Defaults to add_poisson_noise=True, poisson_fano=30.0: at poisson_fano=1.0 (true
        Poisson) the rate-proportional noise is dominated by the deterministic
        condition-to-condition swings in the drive and the diagonal still shows a spurious
        secondary peak at the neuron orthogonal to the adaptor; poisson_fano~30 is large
        enough for the noise to dominate and reveal a clean, monotonic falloff in variance
        with distance from the adaptor. Pass add_poisson_noise=False to see the raw,
        noise-free covariance structure instead.
        '''
        uni  = self.generate_surround_ensembles(adapt_location, biased=False, add_poisson_noise=add_poisson_noise, poisson_fano=poisson_fano)
        bias = self.generate_surround_ensembles(adapt_location, biased=True, add_poisson_noise=add_poisson_noise, poisson_fano=poisson_fano)

        uni_rf  = uni[:self.N_RF]
        bias_rf = bias[:self.N_RF]

        cov_uni  = np.cov(uni_rf,  rowvar=True)
        cov_bias = np.cov(bias_rf, rowvar=True)

        ticks = np.linspace(0, self.N_RF - 1, min(5, self.N_RF)).astype(int)
        tick_labels = [str(t) for t in ticks]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for ax, mat, title in zip(axes, [cov_uni, cov_bias],
                                  ['Uniform Ensemble (CRF)', 'Biased Ensemble (CRF)']):
            im = ax.imshow(mat, cmap='viridis', aspect='auto')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(title, fontsize=20, fontweight='bold', pad=12)
            ax.set_xlabel('CRF neuron index', fontsize=18, fontweight='bold')
            ax.set_ylabel('CRF neuron index', fontsize=18, fontweight='bold')
            ax.set_xticks(ticks); ax.set_xticklabels(tick_labels, fontsize=14)
            ax.set_yticks(ticks); ax.set_yticklabels(tick_labels, fontsize=14)
            for spine in ax.spines.values():
                spine.set_linewidth(2.5)
            ax.tick_params(width=2.5, length=6)

        plt.tight_layout()
        plt.show()

    def plot_tuning_curves(self):
        '''Visualize the tuning curve for each stimulus orientation.'''
        fig, ax = plt.subplots(figsize=(10, 6))

        theta_fine = np.linspace(0, np.pi, 500)
        theta_fine_deg = theta_fine * 180 / np.pi
        colors = plt.cm.Reds(np.linspace(0.2, 1.0, self.num_angles))

        for i in range(self.num_angles):
            delta_theta = theta_fine - self.theta_inputs[i]
            delta_theta = (delta_theta + np.pi/2) % np.pi - np.pi/2
            profile = np.exp(-delta_theta**2 / (2 * self.tuning_width**2)) #+ 0.3
            scale = 1 # COEFFICIENT OF ~15 ACHIEVES CORRECT SATURATION FOR CONTRAST OF 1
            profile = scale * profile / np.max(profile)
            ax.plot(theta_fine_deg, profile, color=colors[i], alpha=0.7, linewidth=1.2)

        ax.set_xlabel("Orientation (deg)")
        ax.set_ylabel("Response (normalized)")
        ax.set_title(f"Tuning Curves ({self.num_angles} orientations)")
        ax.set_xlim([0, 180])
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_von_mises_distributions(self, von_mises_kappa=4.0, num_samples=5000):
        '''Plot KDE curves for von Mises @ 0°, von Mises @ 90°, and uniform.'''
        from scipy.stats import gaussian_kde
        n = num_samples

        # Uniform centers
        base = np.arange(self.num_angles)
        reps = n // self.num_angles
        idx = np.tile(base, reps)
        np.random.shuffle(idx)
        centers_uniform = np.rad2deg(self.theta_inputs[idx])

        # Von Mises @ 0°
        raw0 = np.random.vonmises(0.0, von_mises_kappa, n)
        centers_vm0 = np.rad2deg(((raw0 % np.pi) + np.pi) % np.pi)

        # Von Mises @ 90°
        raw90 = np.random.vonmises(np.deg2rad(90), von_mises_kappa, n)
        centers_vm90 = np.rad2deg(((raw90 % np.pi) + np.pi) % np.pi)

        theta_deg = np.linspace(0, 180, 500)

        # Circular KDE: augment with ±180° shifted copies so the KDE wraps
        # correctly at both edges; multiply by 3 to restore density normalization.
        def circular_kde(data):
            aug = np.concatenate([data - 180, data, data + 180])
            return gaussian_kde(aug)(theta_deg) * 3

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(theta_deg, circular_kde(centers_uniform),
                color='#CC5500',   lw=3, label='Uniform')
        ax.plot(theta_deg, circular_kde(centers_vm0),
                color='#36454F',   lw=3, label='Von Mises, center=0°')
        ax.plot(theta_deg, circular_kde(centers_vm90),
                color='#228B22',   lw=3, label='Von Mises, center=90°')

        ax.set_xlabel('Stimulus orientation (degrees)', fontsize=16, fontweight='bold')
        ax.set_ylabel('Probability density', fontsize=16, fontweight='bold')
        ax.set_title('Stimulus center distributions', fontsize=16, fontweight='bold')
        ax.set_xlim(0, 180)
        ax.tick_params(labelsize=13)
        ax.legend(fontsize=13)
        plt.tight_layout()
        plt.show()

    def plot_contrast_distributions(self, peak_ln_contrasts=(0, -1.5, -3),
                                     contrast_sigma=1.0,
                                     titles=('High Contrast', 'Medium Contrast', 'Low Contrast')):
        '''Plot probability vs ln(contrast) for three truncated log-normal distributions overlaid.'''
        from scipy.stats import truncnorm
        colors = ('black', 'red', 'green')
        fig, ax = plt.subplots(figsize=(7, 5))
        ln_lo = -3  # lower truncation bound in log-space (~-6.9)
        ln_hi = 0.0           # upper truncation bound in log-space
        x_pad = 1.5           # how far past each boundary to extend the x axis
        x_full = np.linspace(ln_lo - x_pad, ln_hi + x_pad, 2000)
        in_range = (x_full >= ln_lo) & (x_full <= ln_hi)

        for peak_ln, title, color in zip(peak_ln_contrasts, titles, colors):
            mu_ln = peak_ln
            pdf_raw = (np.exp(-(x_full - mu_ln)**2 / (2 * contrast_sigma**2))
                       / (contrast_sigma * np.sqrt(2 * np.pi)))
            # Normalize so the truncated area integrates to 1
            Z = np.trapz(pdf_raw[in_range], x_full[in_range])
            pdf_norm = pdf_raw / Z

            # Geometric mean: exp(E[Y]) where Y is the truncated normal
            a_std = (ln_lo - mu_ln) / contrast_sigma
            b_std = (ln_hi - mu_ln) / contrast_sigma
            dist = truncnorm(a_std, b_std, loc=mu_ln, scale=contrast_sigma)
            geom_mean = np.exp(dist.mean())

            label = f"Geom. mean = {geom_mean:.3f}"
            # Faded dashed tails outside the allowed range
            ax.plot(x_full, np.where(~in_range, pdf_norm, np.nan),
                    lw=1.5, color=color, ls='--', alpha=0.35)
            # Solid filled curve within the allowed range
            ax.plot(x_full, np.where(in_range, pdf_norm, np.nan),
                    lw=2, color=color, label=label)
            ax.fill_between(x_full, np.where(in_range, pdf_norm, 0),
                            alpha=0.2, color=color)

        # Shade excluded regions and mark truncation boundaries
        ax.axvspan(ln_lo - x_pad, ln_lo, color='gray', alpha=0.08)
        ax.axvspan(ln_hi, ln_hi + x_pad, color='gray', alpha=0.08)
        ax.axvline(ln_lo, color='gray', lw=1.5, ls=':', alpha=0.7)
        ax.axvline(ln_hi, color='gray', lw=1.5, ls=':', alpha=0.7)

        ax.set_xlabel(r'$\ln(\mathrm{contrast})$', fontsize=16, fontweight='bold')
        ax.set_ylabel(r'$P(\mathrm{contrast})$', fontsize=16, fontweight='bold')
        ax.set_xlim(ln_lo, ln_hi)
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['left', 'bottom']].set_color('gray')
        ax.tick_params(colors='gray', labelsize=13)
        ax.legend(fontsize=11, loc='upper left')
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    stim_gen = StimulusGenerator()
    #stim_gen.plot_covariance_matrices()
    #stim_gen.plot_tuning_curves()
    #stim_gen.plot_von_mises_distributions()
    #stim_gen.plot_contrast_distributions()
    #stim_gen.plot_surround_covariance_matrices()

    # ==========================================================================
    # Poisson-noise divergence diagnostic: generate_surround_ensembles normalizes
    # each column to ||z||=contrast BEFORE add_poisson_noise perturbs it, so the
    # noisy stream can end up with columns longer than 1 - which is almost
    # certainly what's tipping V1Dynamics_Surround's RK4 integration into runaway
    # (the whole model is tuned/tested around ||z|| <= 1). Testing two fixes here
    # without touching the functions yet:
    #   1. Hard cap - only rescale columns whose norm exceeds 1, down to exactly 1;
    #      columns already <= 1 are left untouched.
    #   2. Global rescale - divide the ENTIRE stream by its single largest column
    #      norm, so every column ends up <= 1 but relative scaling between
    #      columns/timesteps (i.e. the actual noise structure) is preserved.
    # ==========================================================================
    ADAPT_LOCATION = 'adapt CRF only'
    POISSON_FANO   = 5.0

    def hard_cap(stream, max_len=1.0):
        '''Rescale only the columns whose norm exceeds max_len, down to exactly max_len.'''
        norms = np.linalg.norm(stream, axis=0, keepdims=True)
        scale = np.minimum(1.0, max_len / norms)
        return stream * scale

    def global_rescale(stream, max_len=1.0):
        '''Divide every column by the single largest column norm in the whole stream.'''
        max_norm = np.linalg.norm(stream, axis=0).max()
        return stream * (max_len / max_norm)

    uni_raw  = stim_gen.generate_surround_ensembles(ADAPT_LOCATION, biased=False, add_poisson_noise=True, poisson_fano=POISSON_FANO)
    bias_raw = stim_gen.generate_surround_ensembles(ADAPT_LOCATION, biased=True,  add_poisson_noise=True, poisson_fano=POISSON_FANO)

    print(f"Raw (noisy) max column norm - uniform: {np.linalg.norm(uni_raw, axis=0).max():.3f}, "
          f"biased: {np.linalg.norm(bias_raw, axis=0).max():.3f}  (fixed-point derivation assumes <= 1)")

    fixes = {
        'Hard cap (clip norm > 1)':     hard_cap,
        'Global rescale (/ max norm)':  global_rescale,
    }

    for fix_name, fix_fn in fixes.items():
        uni  = fix_fn(uni_raw)
        bias = fix_fn(bias_raw)
        print(f"  {fix_name} - max column norm after fix: uniform={np.linalg.norm(uni, axis=0).max():.3f}, "
              f"biased={np.linalg.norm(bias, axis=0).max():.3f}")

        uni_rf  = uni[:stim_gen.N_RF]
        bias_rf = bias[:stim_gen.N_RF]

        cov_uni  = np.cov(uni_rf,  rowvar=True)
        cov_bias = np.cov(bias_rf, rowvar=True)

        ticks = np.linspace(0, stim_gen.N_RF - 1, min(5, stim_gen.N_RF)).astype(int)
        tick_labels = [str(t) for t in ticks]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for ax, mat, title in zip(axes, [cov_uni, cov_bias],
                                  [f'Uniform Ensemble (CRF)\n{fix_name}', f'Biased Ensemble (CRF)\n{fix_name}']):
            im = ax.imshow(mat, cmap='viridis', aspect='auto')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(title, fontsize=16, fontweight='bold', pad=12)
            ax.set_xlabel('CRF neuron index', fontsize=14, fontweight='bold')
            ax.set_ylabel('CRF neuron index', fontsize=14, fontweight='bold')
            ax.set_xticks(ticks); ax.set_xticklabels(tick_labels, fontsize=12)
            ax.set_yticks(ticks); ax.set_yticklabels(tick_labels, fontsize=12)
            for spine in ax.spines.values():
                spine.set_linewidth(2.5)
            ax.tick_params(width=2.5, length=6)

        plt.tight_layout()

    plt.show()
    
