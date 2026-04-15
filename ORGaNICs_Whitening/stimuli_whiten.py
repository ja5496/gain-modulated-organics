import numpy as np
import matplotlib.pyplot as plt

'''
---- stimuli_whiten.py ----
Generates synthetic responses to orientation gratings using raised cosine functions.
These responses are fed into our V1 dynamics as the input layer.

'''

class StimulusGenerator:
    def __init__(self, N=60, num_angles = 169, stream_length = 10140, tuning_width = 0.3, Ensemble=False):
        self.N = N # Number of primary neurons
        self.num_angles = num_angles # Number of distinct input orientations
        self.stream_length = stream_length # Total length of the input stream
        self.tuning_width = tuning_width # Width of raised cosine input

        # Preferred orientations of the stimuli from 0 to pi
        self.theta_tunings = np.linspace(0, np.pi, N, endpoint=False)
        self.theta_inputs = np.linspace(0, np.pi, num_angles, endpoint=False)

    def generate_input_ensembles(self, biased=False):
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
        repeats = int(np.ceil(num_inputs / self.num_angles)) # amount one stimuli should be repeated in uniform ensemble
        indices = np.tile(base_indices, repeats)[:self.stream_length] 

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
        profiles = np.exp(self.tuning_width * np.cos(2 * delta_theta))
        
<<<<<<< HEAD
        # 5. Normalize and Scale (Matching your "generate_sequence" style)
        # Normalize to 0-1 range 
        profiles = 2.5*profiles / np.max(profiles)
        
        #profiles -= profiles.mean(axis=0, keepdims=True) # Centers the inputs so the stimuli have mean = 0
=======
        # 5. Normalize, scale, then mean-center each time step
        profiles =  profiles / np.max(profiles)
        #profiles -= profiles.mean(axis=0, keepdims=True) # Centers the inputs so the stimuli have mean = 0

>>>>>>> target_practice
        return profiles

    def plot_tuning_curves(self):
        '''Visualize the tuning curve for each neuron as raised cosines.'''
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Fine-grained x-axis for smooth curves
        theta_fine = np.linspace(0, np.pi, 169)
        theta_fine_deg = theta_fine * 180 / np.pi
        
        # Red color gradient from light to dark
        colors = plt.cm.Reds(np.linspace(0.2, 1.0, 169))
        
        for i in range(169):
            # Tuning curve for neuron i (preferred orientation = self.theta[i])
            profile = np.exp(self.tuning_width * np.cos(2*(theta_fine - self.theta_inputs[i])))
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
    stim_gen = StimulusGenerator()
    stim_gen.generate_input_ensembles(biased=True)
    stim_gen.plot_tuning_curves()
    
