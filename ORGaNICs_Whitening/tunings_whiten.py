import numpy as np
import matplotlib.pyplot as plt

'''
---- tunings_whiten.py ----
This script generates orientation tuning curves for the primary neurons used by 
simulation_whiten.py. Curves are raised cosine functions. 

'''

class V1Tunings: # Artificial tuning curves of each neuron
    def __init__(self, N=60, sigma_exc=0.15, sigma_inh=0.3, A_exc=1.0, A_inh=0.4): 
        self.N = N
        self.theta = np.linspace(0, np.pi, N, endpoint=False)
        self.W_yy = self._make_recurrent_weights(sigma_exc, sigma_inh, A_exc, A_inh)
        self.N_matrix = np.ones((N, N))

    def _make_recurrent_weights(self, sigma_exc, sigma_inh, A_exc, A_inh):
        '''
        Creates Wyy that excites close-by and inhibits further away (all ORGaNICs papers
        have Wyy of this form). Local excitation (narrow Gaussian) minus surround inhibition
        (wide Gaussian).

        '''

        W = np.zeros((self.N, self.N))
        for i in range(self.N):
            d = np.abs(self.theta - self.theta[i])
            d = np.minimum(d, np.pi - d)  # wrap for circular stimulus space
            exc = A_exc * np.exp(-d**2 / (2 * sigma_exc**2))
            inh = A_inh * np.exp(-d**2 / (2 * sigma_inh**2))
            W[i, :] = exc - inh

                    # Normalize so maximum eigenvalue is 1
        max_eig = np.max(np.real(np.linalg.eigvals(W)))
        return W / max_eig

    def plot_recurrent_weights(self):
        ''' Visualize the recurrent weight matrix as a heatmap.'''
        fig, ax = plt.subplots(figsize=(7, 6))
        
        theta_deg = self.theta * 180 / np.pi
        
        im = ax.imshow(self.W_yy, aspect='auto', cmap='viridis', origin='upper',
                       extent=[0, self.N-1, theta_deg[-1], theta_deg[0]])
        
        ax.set_xlabel("Neuron Index")
        ax.set_ylabel("Preferred Orientation (deg)")
        ax.set_title("Recurrent Weight Matrix (W_yy)")
        plt.colorbar(im, ax=ax, label="Weight", fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    tunings = V1Tunings(N=60, sigma_exc=0.15, sigma_inh=0.4, A_exc=1.0, A_inh=0.4)
    print(f"W_yy shape: {tunings.W_yy.shape}")
    print(f"W_yy min: {tunings.W_yy.min():.3f}, max: {tunings.W_yy.max():.3f}")
    tunings.plot_recurrent_weights()