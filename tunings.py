import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def circular_dist(theta1, theta2, period=np.pi):
    """
    Computes the shortest distance between two angles on a circle.
    """
    diff = np.abs(theta1 - theta2)
    return np.minimum(diff, period - diff)

def von_mises(theta, center, kappa, period=np.pi):
    """
    Computes a Von Mises function (circular Gaussian).
    """
    # Scale theta to 2pi for standard Von Mises math, then scale back
    scale_factor = 2 * np.pi / period
    return np.exp(kappa * np.cos(scale_factor * (theta - center)))

class V1Tunings:
    def __init__(self, N=100, kappa_exc=10.0, kappa_norm=2.0, max_rate=100.0):
        """
        Initializes V1 Orientation Tuning weights.
        """
        self.N = N
        self.theta = np.linspace(0, np.pi, N, endpoint=False)
        self.kappa_exc = kappa_exc
        self.kappa_norm = kappa_norm
        
        # Initialize Matrices
        self.W_rec = self._init_recurrent_weights()
        self.W_norm = self._init_normalization_weights()
        
        # Initialize Input Gain (b0)
        self.b0 = np.ones(N) * 0.2  

    def _init_recurrent_weights(self):
        W = np.zeros((self.N, self.N))
        for i in range(self.N):
            W[i, :] = von_mises(self.theta, self.theta[i], self.kappa_exc)
        W = W / np.max(W) 
        return W

    def _init_normalization_weights(self):
        W = np.zeros((self.N, self.N))
        for i in range(self.N):
            W[i, :] = von_mises(self.theta, self.theta[i], self.kappa_norm)
        W = W / np.max(W)
        return W

    def get_input_drive(self, stim_theta, contrast):
        tuning_profile = von_mises(self.theta, stim_theta, self.kappa_exc)
        tuning_profile = tuning_profile / np.max(tuning_profile)
        return contrast * tuning_profile

# --- Visualization Logic ---
if __name__ == "__main__":
    
    # 1. Initialize Model
    # Using N=180 to map 1:1 with degrees for intuitive plotting
    v1 = V1Tunings(N=50, kappa_exc=15.0, kappa_norm=4.0)
    
    # Setup the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 2. Setup Colormap (HSV is cyclic, perfect for orientation)
    # We create a color for each neuron based on its preferred angle
    colors = cm.hsv(v1.theta / np.pi) 

    # 3. Plot 'All' Tunings
    # We iterate through every neuron and plot its recurrent weight profile
    # (This effectively visualizes the tuning curve of each neuron)
    print("Plotting tuning curves...")
    for i in range(v1.N):
        ax.plot(np.degrees(v1.theta), 
                v1.W_rec[i, :], 
                color=colors[i], 
                alpha=0.6, 
                linewidth=1)

    # Formatting
    ax.set_title(f"Population Tuning Curves (N={v1.N})", fontsize=14)
    ax.set_xlabel("Orientation (Degrees)", fontsize=12)
    ax.set_ylabel("Response / Weight Strength", fontsize=12)
    ax.set_xlim(0, 180)
    ax.grid(True, alpha=0.3)
    
    # Add a colorbar to act as a legend
    sm = cm.ScalarMappable(cmap=cm.hsv, norm=plt.Normalize(0, 180))
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Preferred Orientation (Deg)')

    plt.tight_layout()
    plt.show()

    # Optional: Plot the matrix view to verify the "Ring" structure
    plt.figure(figsize=(6, 5))
    plt.imshow(v1.W_rec, aspect='auto', cmap='viridis', origin='lower',
               extent=[0, 180, 0, 180])
    plt.colorbar(label='Connection Strength')
    plt.title("Recurrent Weight Matrix (W_rec)")
    plt.xlabel("Input Neuron Pref (Deg)")
    plt.ylabel("Target Neuron Pref (Deg)")
    plt.show()