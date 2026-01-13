import numpy as np
import matplotlib.pyplot as plt

def von_mises(theta, center, kappa, period=np.pi):
    """
    Helper: Computes the circular bell curve for the stimulus profile.
    """
    scale_factor = 2 * np.pi / period
    return np.exp(kappa * np.cos(scale_factor * (theta - center)))

class StimulusGenerator:
    def __init__(self, N=180, default_kappa=6.0):
        """
        Handles the creation of stimulus sequences (regimes).
        """
        self.N = N
        self.theta = np.linspace(0, np.pi, N, endpoint=False)
        self.kappa = default_kappa

    def make_grating(self, orientation, contrast):
        """Creates a static 1D input drive for a single oriented grating."""
        profile = von_mises(self.theta, orientation, self.kappa)
        if np.max(profile) > 0:
            profile = profile / np.max(profile)
        return profile * contrast

    def make_plaid(self, theta1, contrast1, theta2, contrast2):
        """Creates a 'Plaid' or 'Mask' stimulus by summing two gratings."""
        g1 = self.make_grating(theta1, contrast1)
        g2 = self.make_grating(theta2, contrast2)
        return g1 + g2

    def generate_sequence(self, regimes):
        """
        Creates a continuous string of stimulus vectors based on a list of regimes.
        
        Parameters:
            regimes : A list of dictionaries. Each dictionary defines a block of stimuli.
                      Example structure:
                      {
                        'n_steps': 100,           # How long this regime lasts
                        'contrast': 1.0,          # Base contrast
                        'orientation': np.pi/2,   # Scalar (fixed) or List (changing)
                        'mask_contrast': 0.0      # Optional: Add a mask
                        'mask_orientation': 0.0   # Optional: Mask angle
                      }
        
        Returns:
            stimulus_stream : Matrix of shape (N_neurons, Total_Steps)
        """
        full_sequence = []

        for i, reg in enumerate(regimes):
            n_steps = reg.get('n_steps', 10)
            contrast = reg.get('contrast', 0.5)
            mask_contrast = reg.get('mask_contrast', 0.0)
            mask_ori = reg.get('mask_orientation', 0.0)
            
            # Handle Orientation: Can be a single fixed value OR a list of values
            ori_input = reg.get('orientation', np.pi/2)
            
            # If scalar, repeat it for n_steps
            if np.isscalar(ori_input):
                orientations = np.ones(n_steps) * ori_input
            else:
                orientations = np.array(ori_input)
                if len(orientations) != n_steps:
                    raise ValueError(f"Regime {i}: Orientation list length must match n_steps.")

            # Generate vectors for this regime
            for t in range(n_steps):
                # 1. Base Stimulus
                step_drive = self.make_grating(orientations[t], contrast)
                
                # 2. Add Mask (if requested)
                if mask_contrast > 0:
                    mask_drive = self.make_grating(mask_ori, mask_contrast)
                    step_drive = step_drive + mask_drive
                
                full_sequence.append(step_drive)

        # Convert list of vectors to (N, Total_Steps) matrix
        return np.array(full_sequence).T

# --- Usage Example: Adaptation Protocol ---
if __name__ == "__main__":
    
    # 1. Initialize
    stim_gen = StimulusGenerator(N=180, default_kappa=8.0)
    
    # 2. Define the Experiment "Script"
    # Goal: Adapt to high contrast, then test response to low contrast
    experimental_script = [
        # Regime 1: Adaptation Phase (High Contrast, Fixed 90 deg)
        {
            'n_steps': 50,
            'contrast': 1.0,
            'orientation': np.pi/2  # 90 degrees
        },
        # Regime 2: Test Phase (Low Contrast, Fixed 90 deg)
        # We expect the adapted neurons to respond weakly here
        {
            'n_steps': 50,
            'contrast': 0.2,
            'orientation': np.pi/2
        },
        # Regime 3: Recovery/Control (Switch Orientation to 0 deg)
        # The neurons at 0 deg should be fresh (unadapted)
        {
            'n_steps': 50,
            'contrast': 0.2,
            'orientation': 0.0 
        }
    ]

    # 3. Generate the "String of Vectors"
    stim_stream = stim_gen.generate_sequence(experimental_script)
    
    print(f"Generated stimulus stream with shape: {stim_stream.shape}") 
    # Shape should be (180, 150) -> 180 neurons x 150 total steps

    # 4. Visualize
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot as a heatmap (Space x Discrete Steps)
    im = ax.imshow(stim_stream, aspect='auto', origin='lower', cmap='binary',
                   extent=[0, stim_stream.shape[1], 0, 180])
    
    # Add regime dividers
    current_step = 0
    for reg in experimental_script:
        current_step += reg['n_steps']
        ax.axvline(current_step, color='red', linestyle='--', alpha=0.5)

    ax.set_title("Stimulus Input Stream (z)")
    ax.set_xlabel("Discrete Step (n)")
    ax.set_ylabel("Neuron Pref (Deg)")
    plt.colorbar(im, label="Input Drive Magnitude")
    
    # Label the regimes
    plt.text(25, 170, "Adapt (High C)", color='red', ha='center', fontweight='bold')
    plt.text(75, 170, "Test (Low C)", color='red', ha='center', fontweight='bold')
    plt.text(125, 170, "New Ori (Low C)", color='red', ha='center', fontweight='bold')

    plt.tight_layout()
    plt.show()
