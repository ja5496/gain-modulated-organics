import numpy as np
import matplotlib.pyplot as plt

'''
---- stimuli_whiten.py ----
Generates synthetic responses to orientation gratings using raised cosine functions.
These responses are fed into our V1 dynamics as the input layer.

'''

class StimulusGenerator:
    def __init__(self, N=60, num_angles = 26, stream_length = 10920, tuning_width = 0.75, Ensemble=False):
        self.N = N # Number of primary neurons
        self.num_angles = num_angles # Number of distinct input orientations
        self.stream_length = stream_length # Total length of the input stream
        self.tuning_width = tuning_width # Width of raised cosine input

        # Preferred orientations of the stimuli from 0 to pi
        self.theta_tunings = np.linspace(0, np.pi, N, endpoint=False)
        self.theta_inputs = np.linspace(0, np.pi, num_angles, endpoint=False)

    def generate_input_ensembles(self, biased=False, mean_center=False):
        '''
        Generate uniform or biased ensemble of raised cosine input profiles  
        centered at random orientations. 
        
        Returns:
            np.ndarray: Shape ( num_angles{number of distinct stimuli} , stream_length )
        '''
        # Generate the indices of all the distinct stimuli
        base_indices = np.arange(self.num_angles)
        
        # Append it on itself until it reaches self.stream_length
        duration = 20 # Stimuli are flashed for a period of (duration * dt).  
        num_inputs = int(self.stream_length / duration) # number of stimuli shown 
        repeats = num_inputs // self.num_angles  # floor: only complete cycles, so every orientation appears exactly repeats times
        indices = np.tile(base_indices, repeats)

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
        profiles = np.exp(-delta_theta**2 / (2 * self.tuning_width**2)) + 0.3 # GAUSSIAN PROFILE
        
        # 5. Normalize, scale, then mean-center each time step
        profiles = profiles / np.max(profiles)
        if mean_center:
            profiles -= profiles.mean(axis=0, keepdims=True)

        return profiles

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

    def plot_tuning_curves(self):
        '''Visualize the tuning curve for each stimulus orientation.'''
        fig, ax = plt.subplots(figsize=(10, 6))

        theta_fine = np.linspace(0, np.pi, 500)
        theta_fine_deg = theta_fine * 180 / np.pi
        colors = plt.cm.Reds(np.linspace(0.2, 1.0, self.num_angles))

        for i in range(self.num_angles):
            delta_theta = theta_fine - self.theta_inputs[i]
            delta_theta = (delta_theta + np.pi/2) % np.pi - np.pi/2
            profile = np.exp(-delta_theta**2 / (2 * self.tuning_width**2)) + 0.3
            profile = profile / np.max(profile)
            ax.plot(theta_fine_deg, profile, color=colors[i], alpha=0.7, linewidth=1.2)

        ax.set_xlabel("Orientation (deg)")
        ax.set_ylabel("Response (normalized)")
        ax.set_title(f"Tuning Curves ({self.num_angles} orientations)")
        ax.set_xlim([0, 180])
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    stim_gen = StimulusGenerator()
    stim_gen.plot_covariance_matrices()
    stim_gen.plot_tuning_curves()
    
