import numpy as np
import matplotlib.pyplot as plt
from tunings import V1Tunings
from stimuli import StimulusGenerator

class V1Dynamics:
    def __init__(self, v1_model, dt=1.0, tau_v=20.0, tau_adapt=2000.0):
        """
        Initializes the dynamic simulation.
        
        Parameters:
            v1_model  : Instance of V1Tunings class (contains weights).
            dt        : Simulation time step (ms).
            tau_v     : Time constant of neural activity (fast, e.g., 20ms).
            tau_adapt : Time constant of adaptation (slow, e.g., 2000ms).
        """
        self.v1 = v1_model
        self.dt = dt
        self.tau_v = tau_v
        self.tau_adapt = tau_adapt
        
        # --- Adaptation Parameters ---
        # alpha: Strength of LOCAL fatigue (specific to active neurons)
        # beta : Strength of GLOBAL homeostasis (scales with total population)
        self.alpha = 0.05 
        self.beta = 0.02
        self.target_b0 = 1.0  # The resting value for gain

    def nonlinearity(self, v):
        """
        Power-law nonlinearity (Supralinear).
        Essential for normalization in recurrent networks (SSN).
        """
        # Rectify (max(0, v)) then square
        return np.maximum(v, 0) ** 2.0

    def run_simulation(self, stimulus_stream):
        """
        Runs the time-step evolution of the network.
        
        Parameters:
            stimulus_stream : Matrix (N_neurons x N_steps) from stimuli.py
        
        Returns:
            rates_history : Firing rates over time.
            b0_history    : Adaptation gain over time.
        """
        N, n_steps = stimulus_stream.shape
        
        # Initialize State Variables
        v = np.zeros(N)          # Membrane potential / Drive
        b0 = self.v1.b0.copy()   # Input gain (start with initial vector)
        
        # Storage for history (to plot later)
        rates_history = np.zeros((N, n_steps))
        b0_history = np.zeros((N, n_steps))
        
        print(f"Starting simulation for {n_steps} steps...")
        
        for t in range(n_steps):
            
            # --- 1. Compute Neural Dynamics (Fast) ---
            
            # Current Firing Rate
            r = self.nonlinearity(v)
            
            # Inputs
            # Feedforward: Gain (b0) * Stimulus (z)
            input_drive = b0 * stimulus_stream[:, t]
            
            # Recurrent Excitation: W_rec * r
            exc_drive = self.v1.W_rec @ r
            
            # Normalization/Inhibition: W_norm * r
            inh_drive = self.v1.W_norm @ r
            
            # Differential Equation for V (Euler Integration)
            # dv/dt = (-v + Input + Exc - Inh) / tau
            dv = (-v + input_drive + exc_drive - inh_drive) / self.tau_v
            v = v + dv * self.dt
            
            # --- 2. Compute Adaptation Dynamics (Slow) ---
            
            # Local Term: The neuron's own firing rate
            local_fatigue = r 
            
            # Global Term: The average activity of the population
            global_activity = np.mean(r)
            
            # Differential Equation for b0 (Input Gain)
            # db0/dt = (Rest - b0 - Local - Global) / tau_adapt
            # We use 'target_b0' as the spring pulling it back to baseline
            db0 = ( (self.target_b0 - b0) 
                   - (self.alpha * local_fatigue) 
                   - (self.beta * global_activity) ) / self.tau_adapt
                   
            b0 = b0 + db0 * self.dt
            
            # Clip b0 to ensure it doesn't go negative (biological constraint)
            b0 = np.maximum(b0, 0.0)
            
            # --- 3. Store Data ---
            rates_history[:, t] = r
            b0_history[:, t] = b0
            
        return rates_history, b0_history

# --- Main Execution Block ---
if __name__ == "__main__":
    
    # 1. Setup Model & Inputs
    print("Initializing Model...")
    tunings = V1Tunings(N=180, kappa_exc=15.0, kappa_norm=4.0)
    stim_gen = StimulusGenerator(N=180, default_kappa=8.0)
    
    # 2. Define Experiment Script
    # "Adapt High" -> "Test Low" -> "Recovery"
    regimes = [
        {'n_steps': 2000, 'contrast': 1.0, 'orientation': np.pi/2}, # Adapt 90 deg
        {'n_steps': 1000, 'contrast': 0.2, 'orientation': np.pi/2}, # Test 90 deg
        {'n_steps': 1000, 'contrast': 0.2, 'orientation': 0.0},     # Test 0 deg
    ]
    
    # Generate the movie (1 step = 1 ms roughly, depending on dt)
    stim_stream = stim_gen.generate_sequence(regimes)
    
    # 3. Run Dynamics
    engine = V1Dynamics(tunings, dt=1.0, tau_v=20.0, tau_adapt=500.0) 
    # Note: tau_adapt is set fast (500ms) here just so you can see it happen quickly in the plot.
    # In reality, this might be 5000ms+ (seconds).
    
    rates, gains = engine.run_simulation(stim_stream)
    
    # 4. Visualization
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # Plot A: Stimulus Input (Visual check)
    axes[0].imshow(stim_stream, aspect='auto', cmap='binary', origin='lower')
    axes[0].set_ylabel("Neuron Pref")
    axes[0].set_title("Input Drive (z)")
    
    # Plot B: Firing Rates (Output)
    im_r = axes[1].imshow(rates, aspect='auto', cmap='inferno', origin='lower', vmax=np.percentile(rates, 99))
    axes[1].set_ylabel("Neuron Pref")
    axes[1].set_title("Population Firing Rates (Hz)")
    plt.colorbar(im_r, ax=axes[1], label="Rate")
    
    # Plot C: Adaptation Gains (b0) - The "Internal State"
    # We use a diverging colormap (coolwarm) centered at 1.0 to show increase/decrease
    im_b = axes[2].imshow(gains, aspect='auto', cmap='coolwarm', origin='lower', vmin=0.5, vmax=1.5)
    axes[2].set_ylabel("Neuron Pref")
    axes[2].set_xlabel("Time Steps")
    axes[2].set_title("Input Gains ($b_0$) | Blue = Adapted/Fatigued")
    plt.colorbar(im_b, ax=axes[2], label="Gain")
    
    # Add vertical lines for regime changes
    t = 0
    for reg in regimes:
        t += reg['n_steps']
        for ax in axes:
            ax.axvline(t, color='white', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()
    
    # 5. Diagnostic: Plot center neuron trace
    plt.figure(figsize=(10, 4))
    center_idx = 90
    plt.plot(rates[center_idx, :], label="Firing Rate", color='orange')
    plt.plot(gains[center_idx, :] * 10, label="Gain ($b_0$) x10", color='blue')
    plt.title(f"Trace for Neuron {center_idx} (Preferred Orientation)")
    plt.legend()
    plt.xlabel("Time")
    plt.grid(True, alpha=0.3)
    plt.show()
