import numpy as np
import matplotlib.pyplot as plt

'''
---- stimuli_whiten.py ----
Generates V1-tuned responses to LGN outputs using raised cosine functions. 

'''

class StimulusGenerator:
    def __init__(self, N=60):
        self.N = N
        self.theta = np.linspace(0, np.pi, N, endpoint=False)

    def generate_sequence(self, regimes):
        seq = []
        for r in regimes:
            profile = np.exp(6.0 * np.cos(2*(self.theta - r['orientation'])))
            profile = 2*profile / np.max(profile) * r['contrast'] # Added a 2 so that 0.5 contrast would correspond to the turning point of gains
            seq.append(np.tile(profile, (r['n_steps'], 1)).T)
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
    stim_gen = StimulusGenerator(N=60)
    stim_gen.plot_tuning_curves()