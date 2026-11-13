# Methods notes

Notes for the Methods section, one section per analysis. Claims about prior work are marked
[CITATION NEEDED].

## Settling time of the normalization circuit

Script: `analysis/normalization_settling.py`. Figures: `figures/fig_norm_timecourse.py`,
`figures/fig_norm_discrepancy.py`, `figures/fig_norm_tau_sweep.py`. Outputs:
`results/normalization_settling/` (`normalization_settling.npz`, `params.json` with git commit,
seed and date).

### Equations

The normalization subsystem of the cRF + surround model (`V1Dynamics_Surround`) was isolated by
holding all interneuron gains at zero ($\boldsymbol{g} = 0$, $\tau_g \to \infty$), which removes the
gain-feedback term of Eq. dy. The remaining dynamics are single-layer ORGaNICs:

$$
\tau_y\frac{d\boldsymbol{y}}{dt} = -\boldsymbol{y} + \beta\,\boldsymbol{z} + \left(\frac{1}{1+\boldsymbol{a}^+}\right)\boldsymbol{W}_{yy}\sqrt{\boldsymbol{y}^+}
$$
$$
\tau_u \frac{d\boldsymbol{u}}{dt} = -\boldsymbol{u} + \left(\frac{\sigma}{2}\right)^2 + \boldsymbol{N}\left(\boldsymbol{y}^+ \odot (\boldsymbol{u}^+)^{2}\right)
$$
$$
\tau_a \frac{d\boldsymbol{a}}{dt} = -\boldsymbol{a} + \boldsymbol{u}^+ + \boldsymbol{a}\odot \boldsymbol{u}^+
$$

with $\boldsymbol{y}^+ = \lfloor\boldsymbol{y}\rfloor^2$, $\boldsymbol{u}^+ = \lfloor\boldsymbol{u}\rfloor^{1/2}$,
$\boldsymbol{a}^+ = \lfloor\boldsymbol{a}\rfloor$, $\boldsymbol{W}_{yy} = \boldsymbol{I}$ and
$\boldsymbol{N}$ the all-ones matrix over all $N_\text{tot} = 78$ neurons (one cRF and five surround
sets of 13). For nonnegative drive the fixed point is exact:

$$
u^* = \frac{\sigma^2 + \sum_j z_j^2}{4}, \qquad a^* = \frac{\sqrt{u^*}}{1-\sqrt{u^*}}, \qquad
\boldsymbol{y}^{*2} = \frac{\boldsymbol{z}^2}{\sigma^2 + \boldsymbol{N}\boldsymbol{z}^2}
$$

(Eq. y_steady with $\boldsymbol{z}' = \boldsymbol{z}$).

### Stimulus and protocol

A full-field oriented grating (orientation $\pi/2$, contrast 1, Gaussian orientation profile of
width 0.75 rad over each set's 13 preferred orientations, the same in all six sets, divided by the
fixed input bound so that $\sum_j z_j^2 = 0.999$) was switched on at $t = 0$. The network started
from the blank-stimulus steady state ($\boldsymbol{y} = 0$, $u = \sigma^2/4$,
$a = \sqrt{u}/(1-\sqrt{u})$), held for 5 ms before onset.

Settling time of population $x \in \{y, u, a\}$: the first time after which
$\max_i |x_i(t) - x_i^*| / \max_i |x_i^*| \le 0.01$ for the rest of the window. For the firing
rates (Fig. 2) the discrepancy is
$\|\lfloor\boldsymbol{y}(t)\rfloor^2 - \boldsymbol{y}^{*2}\|_2 / \|\boldsymbol{y}^{*2}\|_2$.

### Parameters

| Symbol | Value | Note |
|---|---|---|
| $\beta$ | 0.5 | |
| $\sigma$ | 0.15 | |
| $\tau_y$ | 0.4 ms | fixed throughout |
| $\tau_u$ | 30 ms (original); 5.3 ms (tuned); sweep 0.47–60 ms | half-octave grid, 15 values |
| $\tau_a$ | 0.2 ms (original and tuned); sweep 0.05–3.2 ms | half-octave grid, 13 values |
| $\tau_g$ | $\infty$ | adaptation off |
| $N_\text{RF}$, $N_\text{sets}$ | 13, 6 | frame `N13_mercedes_K182_Frame.csv` (unused, $g=0$) |

The time constants in `V1Dynamics_Surround` ($\tau_y = 0.4$, $\tau_u = 30$, $\tau_a = 0.2$ ms) differ
from those in `docs/equations.md` (0.2, 15, 0.1 ms); this analysis uses the code values.

### Integration

Fixed-step classical RK4 (`V1Dynamics_Surround.run_simulation`), stimulus held constant within
each step, $\Delta t = \min(0.1, \min(\tau_y, \tau_u, \tau_a)/4)$ ms (0.05 ms for the original and
tuned parameters; 0.0125 ms at the stiffest grid corner). Windows: 300 ms after onset for the two
time courses, 60 ms for each sweep point.

### Verification

- Fixed-point residual of the right-hand side at the analytic fixed point: $1.1\times10^{-15}$.
- Linear stability: finite-difference Jacobian of the reduced $(\boldsymbol{y}, \boldsymbol{u}, \boldsymbol{a})$
  system at the fixed point. For the original and tuned parameters all eigenvalues have negative
  real part; the slowest is exactly $-1/\tau_u$ ($-0.0333$ and $-0.189\ \text{ms}^{-1}$). Four sweep
  points ($\tau_u = 0.47$ ms, $\tau_a \ge 1.13$ ms) are unstable.
- Perturbation: ±5 % random perturbations of $y$, $u$, $a$ at the fixed point decay to a relative
  error of $3\times10^{-6}$ within $10\max(\tau)$.
- Time-step convergence: settling times change by < 0.03 ms when $\Delta t$ is divided by 4
  (original, tuned and stiffest grid corner).
- Gains stayed exactly zero in every run (asserted).
- Gain-adaptation and target-feasibility checks: not applicable (adaptation off).

### Results

- Original time constants: settling times 110.7 ms (y), 129.0 ms (u), 129.6 ms (a); firing rates
  within 1 % of the normalized state after 130.5 ms.
- 64 of 195 grid points settle within 30 ms with a stable fixed point. The robust region is
  $\tau_a \le 0.2$ ms with $\tau_u \le 7.5$ ms, narrowing for slower $a$ ($\tau_a = 0.28$:
  $\tau_u \in [0.94, 7.5]$; 0.4: $[1.33, 10.6]$; 0.57: $[1.88, 10.6]$ ms); no point with
  $\tau_a \ge 0.8$ ms qualifies.
- Tuned parameters ($\tau_u = 5.3$ ms, $\tau_a = 0.2$ ms): 18.6 ms (y), 20.0 ms (u), 20.5 ms (a);
  rates within 1 % after 21.3 ms. At contrasts 0.125, 0.25 and 0.5 the slowest population settles
  in 26.2, 27.8 and 22.4 ms.
- Settling is not monotonic in $(\tau_u, \tau_a)$: isolated points outside the robust region (for
  example $\tau_u = 30$, $\tau_a = 2.26$ ms, 31 ms) settle quickly because the onset transient
  happens to excite the slow $-1/\tau_u$ mode weakly. These are specific to the stimulus.
