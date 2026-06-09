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

''' Psuedo-code:

 ----- STEP 1 - Define input -----
 def profile(input_width, center_angle):
    theta = np.linspace(0 to pi)
    stimulus = np.exp(- (theta - center_angle) ** 2 / (2 * input_width**2))
    normalize stimulus from 0 to 1
    return stimulus 

    
    
 ----- STEP 2 - Probe Function -----
def probe_normalization(stimulus_profile, fixed_gains, ):
    1) Take dynamics from simulation_whiten.py (but without adapting gains)
    2) For a given stimulus profile, calculate it's steady state response (run 100 time steps)
    3) Return the euclidean norm of the steady-state y vector



----- STEP 3 - Run dynamics for fixed gains  -----
from stimuli_whiten get uniform ensemble from StimulusGenerator.generate_input_ensembles(biased=False)
    --> stimulus stream should be 5460 inputs long 

Simulate response to uniform ensemble at contrast = 1, get the fixed gains

contrasts = np.linspace(0,1,15)

for each set of gains {zero gains or uniform ensemble gains}: 
    for contrast in contrasts:
        (a) run 100 time steps of simulation to get steady state y (gains fixed)
        (b) record the magnitude (euclidean norm) of the response vector in an array
        (c) one array for each set of gains


----- STEP 4 - Plot -----

- Axes: black, only x and y (not a box), thick, x-label: "Contrast" (Large font), y-label: "Response" (Large font).
- No gridlines
- Legend in the top left: "g = 0" for a thick burgundy curve, "uniform ensemble gains" for a thick navy blue curve. 
    - include both semi-saturation constants in the legend.
- No title. 
        

'''



