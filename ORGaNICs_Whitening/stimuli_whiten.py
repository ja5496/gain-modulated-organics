import numpy as np
import matplotlib.pyplot as plt

'''
---- stimuli_whiten.py ----
Generates V1-tuned responses to LGN outputs using raised cosine functions. 
Now supports additive white noise to simulate broad-spectrum suppression effects.
'''

class StimulusGenerator:
    def __init__(self, N=60):
        self.N = N
        # Preferred orientations from 0 to pi
        self.theta = np.linspace(0, np.pi, N, endpoint=False)

    def generate_sequence(self, regimes):
        '''
        Generates a sequence of neural responses based on a list of regimes.
        
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
            profile = np.exp(6.0 * np.cos(2*(self.theta - r['orientation'])))
            
            # Normalize and scale by contrast
            # Added a 2 so that 0.5 contrast roughly corresponds to turning point of gains
            profile = 2 * profile / np.max(profile) * r['contrast']
            
            # 2. Tile across time: Shape becomes (N_neurons, n_steps)
            block = np.tile(profile, (r['n_steps'], 1)).T
            
            # 3. Add White Noise
            # "Theoretically... suppress tunings besides the peak" 
            # This works because noise increases the denominator in divisive normalization
            noise_level = r.get('noise_level', 0.0)
            if noise_level > 0:
                # Generate noise for every neuron at every time step independently
                noise = np.random.normal(loc=0.0, scale=noise_level, size=block.shape)
                block = block + noise
                
                # Rectification: Ensure drive doesn't go below zero (standard for firing rates/energy)
                block = np.maximum(0, block)

            seq.append(block)
            
        return np.hstack(seq)

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
            profile = np.exp(6.0 * np.cos(2*(theta_fine - self.theta[i])))
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