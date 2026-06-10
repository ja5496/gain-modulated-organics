'''

Script to check the normalization (plots a contrast response function).

Compute network steady state response to varying contrasts in the following way:
    1. Create a normalized (0 to 1) stimulus profile
    2. Create an array of contrasts that span from 0 to 1
    3. Probe the population response magnitude at each contrast and plot it
    4. Do so for gains = 0 to isolate ORGaNICs and tune normalization
    5. Adapt the full network (including gains) to a uniform ensemble of orientations at
        high contrast and compare to gains = 0 plot
    6. Both plots calculate the semi-saturation constant (the contrast at which the
        half-maximum is achieved), and display on the plot.


'''
import numpy as np
import matplotlib.pyplot as plt
from tunings_whiten import V1Tunings
from simulation_whiten import Frame, V1Dynamics
from stimuli_whiten import StimulusGenerator

# --- Parameters ---
N             = 169
STREAM_LENGTH = 5460
PROBE_STEPS   = 5
N_CONTRASTS   = 15
FRAME_PATH    = "Frames/N169_Frame.csv"


# ----- STEP 1: Define input profile -----

def profile(theta, input_width=0.75, center_angle=None):
    if center_angle is None:
        center_angle = np.pi / 2
    stim = np.exp(-(theta - center_angle) ** 2 / (2 * input_width ** 2))
    scale = 15 # COEFFICIENT OF ~15 ACHIEVES CORRECT SATURATION FOR CONTRAST OF 1
    return scale * (stim - stim.min()) / (stim.max() - stim.min())

# ----- STEP 2: Probe function -----

def probe_normalization(z_profile, dynamics, fixed_gains, n_steps=PROBE_STEPS):
    """
    Run n_steps of ORGaNICs with fixed gains and return ||y_ss||.
    gain_feedback is computed dynamically each step as W @ (g * W.T @ y),
    which matches the steady-state whitening objective while keeping g fixed.
    """

    y = np.zeros(N)
    u = np.zeros(N)
    a = np.zeros(N)
    v = np.zeros(dynamics.frame.K)
    tau_v = 100

    sigma_term = (dynamics.sigma / 2) ** 2
    dt = dynamics.dt

    def derivs(y_, u_, a_, v_):
        u_plus = dynamics.gaussian_rectify(u_)
        y_plus = dynamics.gaussian_rectify(y_)
        a_plus = dynamics.gaussian_rectify(a_)
        sq_y_plus   = np.sqrt(y_plus)

        gain_fb = dynamics.frame.W @ (fixed_gains * v)
        rec     = (1.0 / (1.0 + a_plus)) * (dynamics.v1.W_yy @ sq_y_plus)
        pool_t  = dynamics.v1.N_matrix @ (y_plus * (u_plus ** 2))

        dy = (-y_ + z_profile / 2 + rec - gain_fb) / dynamics.tau_y
        du = (-u_ + sigma_term + pool_t)            / dynamics.tau_u
        da = (-a_ + u_plus + a_ * u_plus)           / dynamics.tau_a
        dv = (-v_ + dynamics.frame.W.T @ y_) / tau_v
        return dy, du, da, dv

    for _ in range(n_steps):
        dy1, du1, da1, dv1 = derivs(y, u, a, v)
        dy2, du2, da2, dv2 = derivs(y + 0.5*dt*dy1, u + 0.5*dt*du1, a + 0.5*dt*da1, v + 0.5*dt*dv1)
        dy3, du3, da3, dv3 = derivs(y + 0.5*dt*dy2, u + 0.5*dt*du2, a + 0.5*dt*da2, v + 0.5*dt*dv2)
        dy4, du4, da4, dv4 = derivs(y +     dt*dy3, u +     dt*du3, a +     dt*da3, v +     dt*dv3)

        y += (dt / 6.0) * (dy1 + 2*dy2 + 2*dy3 + dy4)
        u += (dt / 6.0) * (du1 + 2*du2 + 2*du3 + du4)
        a += (dt / 6.0) * (da1 + 2*da2 + 2*da3 + da4)
        v += (dt / 6.0) * (dv1 + 2*dv2 + 2*dv3 + dv4)

    firing_rates = dynamics.gaussian_rectify(y)
    return np.linalg.norm(firing_rates)


# ----- Helper: semi-saturation constant -----

def compute_c50(contrasts, responses):
    max_r = np.max(responses)
    if max_r == 0:
        return np.nan
    half_max = max_r / 2
    for i in range(len(responses)):
        if responses[i] >= half_max:
            if i == 0:
                return contrasts[0]
            c1, c2 = contrasts[i - 1], contrasts[i]
            r1, r2 = responses[i - 1], responses[i]
            return c1 + (half_max - r1) * (c2 - c1) / (r2 - r1)
    return contrasts[-1]


# ----- STEP 3: Run dynamics and sweep contrasts -----

if __name__ == "__main__":

    print("Initializing network...")
    tunings  = V1Tunings(N=N)
    frame    = Frame(csv_path=FRAME_PATH)
    stim_gen = StimulusGenerator(N=N, num_angles=N, stream_length=STREAM_LENGTH)

    # Normalized stimulus profile centered at pi/2
    stim_profile = profile(tunings.theta) 

    # Adapt full network to uniform ensemble at contrast = 1
    print("Adapting to uniform ensemble...")
    uniform_stream = stim_gen.generate_input_ensembles(biased=False)
    engine = V1Dynamics(tunings, frame, adaptive=True, input_adaptive=False)
    _, gains_hist, _, _, _, _, _ = engine.run_simulation(uniform_stream)
    uniform_gains = gains_hist[:, -1]

    # Contrast sweep (log-spaced so points are evenly distributed on log axis)
    contrasts   = np.logspace(-2, 0, N_CONTRASTS)
    zero_gains  = np.zeros(frame.K)

    print("Probing with g = 0...")
    responses_zero = np.array([
        probe_normalization(c * stim_profile, engine, zero_gains)
        for c in contrasts
    ])

    print("Probing with uniform ensemble gains...")
    responses_uniform = np.array([
        probe_normalization(c * stim_profile, engine, uniform_gains)
        for c in contrasts
    ])

    c50_zero    = compute_c50(contrasts, responses_zero)
    c50_uniform = compute_c50(contrasts, responses_uniform)
    print(f"C50 (g = 0):             {c50_zero:.3f}")
    print(f"C50 (uniform ensemble):  {c50_uniform:.3f}")


    # ----- STEP 4: Plot -----

    burgundy = '#800020'
    navy     = '#002060'

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(contrasts, responses_zero,    color=burgundy, linewidth=3.0,
            label=f'Gains = 0  (C₅₀ = {c50_zero:.2f})')
    ax.plot(contrasts, responses_uniform, color=navy,     linewidth=3.0,
            label=f'Uniform Ensemble Gains  (C₅₀ = {c50_uniform:.2f})')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for side in ['left', 'bottom']:
        ax.spines[side].set_linewidth(2.5)
        ax.spines[side].set_color('black')

    ax.set_xscale('log')
    ax.tick_params(colors='black', width=2, length=6, labelsize=13)
    ax.set_xlabel("Contrast",  fontsize=18, fontweight='bold', color='black')
    ax.set_ylabel("Response",  fontsize=18, fontweight='bold', color='black')
    ax.legend(loc='upper left', fontsize=12, frameon=False)
    ax.grid(False)

    plt.tight_layout()
    plt.show()
