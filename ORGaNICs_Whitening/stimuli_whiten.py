import numpy as np
import matplotlib.pyplot as plt

'''
---- stimuli_whiten.py ----
Generates V1-tuned responses to LGN outputs using raised cosine functions. 
Now supports additive white noise to simulate broad-spectrum suppression effects.
'''

class StimulusGenerator:
    def __init__(self, N=60, K=200, stream_length = 8000, tuning_width = 1, Ensemble=False):
        self.N = N # Number of primary neurons
        self.K = K # Number of distinct input orientations
        self.stream_length = stream_length
        self.tuning_width = tuning_width
        # Preferred orientations from 0 to pi
        
        self.theta_tunings = np.linspace(0, np.pi, N, endpoint=False)
        self.theta_inputs = np.linspace(0, np.pi, K, endpoint=False)

    def generate_sequence(self, regimes):
        '''
        Generates a sequence of neural inputs from LGN based on a list of regimes.
        
        Args:
            regimes (list of dict): Each dict corresponds to a time block and can contain:
                - 'orientation': (float) Stimulus orientation in radians
                - 'contrast': (float) Stimulus contrast magnitude
                - 'n_steps': (int) Duration of the block in time steps
                - 'noise_level': (float, optional) Std dev of additive Gaussian white noise
        '''
        seq = []
        for r in regimes:
            # 1. Generate the base tuning profile (Von Mises / Raised Cosine)
            # This represents the "signal" drive to the population
            profile = np.exp(self.tuning_width * np.cos(2*(self.theta_inputs - r['orientation'])))
            
            # Normalize and scale by contrast
            scale = 2.5 # Scales the input vector by a fixed amount after normalization (not counting contrast)
            profile = scale * profile / np.max(profile) * r['contrast']
            
            # 2. Tile across time: Shape becomes (N_neurons, n_steps)
            block = np.tile(profile, (r['n_steps'], 1)).T
            
            # 3. Add White Noise
            noise_level = r.get('noise_level', 0.0)
            if noise_level > 0:
                # Generate noise for every neuron at every time step independently
                noise = np.random.normal(loc=0.0, scale=noise_level, size=block.shape)
                block = block + noise
                
                # Ensure drive doesn't go below zero (standard for firing rates/energy)
                block = np.maximum(0, block)
            seq.append(block)
            
        return np.hstack(seq)
    
    def generate_input_ensembles(self, biased=False):
        '''
        Generate uniform or biased ensemble of input profiles (Von Mises) 
        centered at random orientations. 
        
        Returns:
            np.ndarray: Shape ( K{number of distinct stimuli} , stream_length )
        '''
        # Generate the indices of all the distinct stimuli
        base_indices = np.arange(self.K)
        
        # Append it on itself until it reaches self.stream_length
        duration = 20 # Stimuli are flashed for a period of (duration * dt).  
        num_inputs = int(self.stream_length / duration) # actual number of stimuli shown (instead of time steps)
        repeats = int(np.ceil(num_inputs / self.K))
        indices = np.tile(base_indices, repeats)[:self.stream_length] 

        # Optionally overwrite roughly 33% of the indices with the adaptor index
        if biased:
            one_third_split = len(indices) // 3 # Calculate the index representing the first third
            adaptor_idx = self.K // 2 # Define the adaptor index
            indices[:one_third_split] = adaptor_idx # Apply the mask to the first third of the array

        # Randomly shuffle the indices array in-place
        np.random.shuffle(indices) 

        # Now add the duration of the inputs in so it doesn't flash a new one every time step. 
        indices = np.repeat(indices, duration)

        print(len(indices), self.stream_length)

        # Convert indices to actual orientation centers; shape: (stream_length,)
        centers = self.theta_inputs[indices]

        # Generate stimulus curves using broadcasting - matrix of shape (K_stimuli, stream_length).
        delta_theta = self.theta_inputs[:, np.newaxis] - centers[np.newaxis, :]
        profiles = np.exp(self.tuning_width * np.cos(2 * delta_theta))
        
        # 5. Normalize and Scale (Matching your "generate_sequence" style)
        # Normalize to 0-1 range 
        profiles = 2.5*profiles / np.max(profiles)
        
        return profiles


    def plot_tuning_curves(self):
        '''Visualize the tuning curve for each neuron as shifted raised cosines.'''
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Fine-grained x-axis for smooth curves
        theta_fine = np.linspace(0, np.pi, 500)
        theta_fine_deg = theta_fine * 180 / np.pi
        
        # Red color gradient from light to dark
        colors = plt.cm.Reds(np.linspace(0.2, 1.0, self.N))
        
        for i in range(self.N):
            # Tuning curve for neuron i (preferred orientation = self.theta[i])
            profile = np.exp(2.5 * np.cos(2*(theta_fine - self.theta[i])))
            profile = profile / np.max(profile)
            ax.plot(theta_fine_deg, profile, color=colors[i], alpha=0.7, linewidth=1.2)
        
        ax.set_xlabel("Orientation (deg)")
        ax.set_ylabel("Response (normalized)")
        ax.set_title(f"Tuning Curves for {self.N} Neurons")
        ax.set_xlim([0, 180])
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":

    # --- Example Usage ---
    stim_gen = StimulusGenerator(N=60)
    stim_gen.generate_input_ensembles(biased=True)
    
    # Define a sequence: 
    # 1. Clean stimulus
    # 2. Noisy stimulus (same orientation)
    regimes = [
        {
            'orientation': np.pi/2, # 90 degrees
            'contrast': 1.0, 
            'n_steps': 50,
            'noise_level': 0.0      # Clean
        },
        {
            'orientation': np.pi/2, 
            'contrast': 1.0, 
            'n_steps': 50,
            'noise_level': 0.1      # Added White Noise
        }
    ]

    # Generate data
    data = stim_gen.generate_sequence(regimes)
    
    # --- Quick Visualization of the Output Matrix ---
    plt.figure(figsize=(12, 6))
    plt.imshow(data, aspect='auto', cmap='hot', origin='lower')
    plt.colorbar(label='Input Drive')
    plt.xlabel('Time Step')
    plt.ylabel('Neuron Index (Preferred Orientation)')
    plt.title('V1 Input Drive: Clean vs. Noisy Stimulus')
    plt.axvline(x=50, color='white', linestyle='--', linewidth=2, label='Noise Onset')
    plt.legend()
    plt.show()