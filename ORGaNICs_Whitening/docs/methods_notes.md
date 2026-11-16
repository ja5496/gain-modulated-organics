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
$\max_i |x_i(t) - x_i^*| / \max_i |x_i^*| \le 0.01$ for the rest of the window, provided the band
is held for at least 20 ms before the window ends (otherwise "not settled"). For the firing
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
  real part. Four sweep points ($\tau_u = 0.47$ ms, $\tau_a \ge 1.13$ ms) are unstable.
- Settling mode: the full Jacobian's slowest eigenvalue, exactly $-1/\tau_u$, belongs to the
  $u_i - u_j$ difference modes, which a spatially uniform onset does not excite. The settling time
  is set instead by the slowest eigenvalue on the invariant subspace with uniform $u$ and $a$:
  $-0.0347\ \text{ms}^{-1}$ (original; $\ln 100 / 0.0347 = 133$ ms vs. 129.6 ms observed) and
  $-0.266\ \text{ms}^{-1}$ (tuned; 17.3 ms vs. 20.5 ms). In 46 of the 64 feasible sweep points this
  mode is a complex pair (damped ringing); at the tuned point it is real.
- Perturbation: ±5 % random perturbations of $y$, $u$, $a$ at the fixed point decay to a relative
  error of $3\times10^{-6}$ within $10\max(\tau)$.
- Time-step convergence: settling times change by < 0.04 ms (below one step) when $\Delta t$ is
  divided by 4 (original, tuned and stiffest grid corner). An independent check against an
  implicit Radau integrator (rtol $10^{-10}$) agreed to within one $\Delta t$.
- Independent re-implementation (78 neurons, per-neuron $u$ and $a$, no $v$/$\mu$ in the state)
  matched the model run to $3\times10^{-17}$, and the feasible set was unchanged when re-scored
  with a 400 ms window.
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
- Settling is not monotonic in $(\tau_u, \tau_a)$: isolated points (for example $\tau_u = 30$,
  $\tau_a = 2.26$ ms, 31 ms; $\tau_u = 7.5$, $\tau_a = 0.2$ ms, 11.5 ms vs. about 20 ms for its
  neighbours) settle quickly because the onset excites the slowest symmetric mode with a small
  amplitude or at a favourable phase. These pockets are specific to the stimulus, so the tuned
  point was required to have all eight grid neighbours feasible.

## Model revision: gain feedback through v, variance statistic on c (V1Dynamics_Surround)

Code: `simulation.py` (`V1Dynamics_Surround._derivatives`), mirrored in
`analysis/response_tools.py` (`frozen_derivatives`). Tests: `tests/test_simulation_surround.py`.

### Equations implemented

$$
\begin{aligned}
\tau_y \frac{d\boldsymbol{y}}{dt} &= -\boldsymbol{y} + \beta\,\boldsymbol{z} + \left(\frac{1}{1+\boldsymbol{a}^+}\right)\boldsymbol{W}_{yy}\sqrt{\boldsymbol{y}^+} - \boldsymbol{u}^+ \odot \boldsymbol{W}_{yv}\boldsymbol{v}\\
\tau_u \frac{d\boldsymbol{u}}{dt} &= -\boldsymbol{u} + \left(\frac{\sigma}{2}\right)^2 + \boldsymbol{N}\left(\boldsymbol{y}^+ \odot (\boldsymbol{u}^+)^2\right)\\
\tau_a \frac{d\boldsymbol{a}}{dt} &= -\boldsymbol{a} + \boldsymbol{u}^+ + \boldsymbol{a}\odot\boldsymbol{u}^+\\
\tau_v \frac{d\boldsymbol{v}}{dt} &= -\boldsymbol{v} + \boldsymbol{g}\odot\boldsymbol{c}, \qquad \boldsymbol{c} = \boldsymbol{W}_{yv}^\top\sqrt{\boldsymbol{y}^+}\\
\tau_g \frac{d\boldsymbol{g}}{dt} &= (\boldsymbol{c}-\boldsymbol{\mu})^2 - \boldsymbol{\theta}_t\\
\tau_\mu \frac{d\boldsymbol{\mu}}{dt} &= -\boldsymbol{\mu} + \boldsymbol{c}
\end{aligned}
$$

$\boldsymbol{v}$, $\boldsymbol{g}$, $\boldsymbol{\mu}$, $\boldsymbol{c} \in \mathbb{R}^K$ are local to each receptive field
(one cRF copy and one copy shared by the identical surround sets); $\boldsymbol{\mu}$ is now
$K$-dimensional. Gains are clamped, $\boldsymbol{g} \leftarrow \max(\boldsymbol{g}, 0)$, after every step
(`gains_nonneg=True`, as in all analysis scripts).

Previous form (commit d63ab08): $\tau_v \dot{\boldsymbol{v}} = -\boldsymbol{v} + \boldsymbol{W}_{yv}^\top\boldsymbol{y}$,
feedback $\boldsymbol{u}^+ \odot \boldsymbol{W}_{yv}(\boldsymbol{g}\odot\boldsymbol{v})$,
$\tau_g \dot{\boldsymbol{g}} = (\boldsymbol{v} - \boldsymbol{W}_{yv}^\top\boldsymbol{\mu})^2 - \boldsymbol{\theta}_t$,
$\tau_\mu\dot{\boldsymbol{\mu}} = -\boldsymbol{\mu} + \boldsymbol{y}$ ($\boldsymbol{\mu}\in\mathbb{R}^{N}$).
Before d63ab08 the feedback was gated by $\boldsymbol{a}^+/(1+\boldsymbol{a}^+)$ instead of $\boldsymbol{u}^+$; the
two agree at the fixed point because Eq. da gives $a/(1+a) = \sqrt{u}$ there.

### Parameters

$\beta = 0.5$, $\sigma = 0.15$; $\tau_y = 0.4$, $\tau_u = 5$, $\tau_a = 0.2$, $\tau_v = 30$, $\tau_g = 500$,
$\tau_\mu = 2000$ ms (code values; `docs/equations.md` lists different ones). $N_\text{RF} = 13$,
$N_\text{sets} = 6$, frame `N13_mercedes_K182_Frame.csv` ($K = 182$). $\boldsymbol{\theta}_t =
\operatorname{diag}(\boldsymbol{W}_{yv}^\top \boldsymbol{C}\boldsymbol{W}_{yv})$ with $\boldsymbol{C}$ = 0.15 × the uniform
low-contrast target covariance (mean $9.5\times10^{-5}$, range $9.7\times10^{-7}$–$3.9\times10^{-4}$).

### Integration

Fixed-step RK4, $\Delta t = 0.1$ ms (`run_simulation`), stimulus held constant within a step, gain
clamp after each step.

### Comparison with the previous form

With gains frozen and $\boldsymbol{y} \ge 0$, the new $\boldsymbol{v}$ equals $\boldsymbol{g}\odot$ the old one at
all times (same linear filter, constant $\boldsymbol{g}$), so the principal-neuron trajectories are
identical. Simulated from the blank steady state with 9 gain sets (zero; gains adapted by either
form to a grating or an ensemble; 3× those) and 4 stimuli, 400 ms: $\max_t |\boldsymbol{y}_\text{new} -
\boldsymbol{y}_\text{old}| / \max|\boldsymbol{y}^*| \le 3\times10^{-15}$ in the 31 cases with $\boldsymbol{z}' \ge 0$.
In the 5 cases with negative $\boldsymbol{z}'$ (min $-0.012$; gains adapted at 90° probed at 45°) the
analytic fixed point does not apply, both forms converge ($|\dot{\boldsymbol{y}}| \sim 10^{-16}$) and
their fixed points differ by up to $2\times10^{-3}$, because the old form fed raw (negative)
$\boldsymbol{y}$ of the surround into $\boldsymbol{v}$.

Settling (all neurons within 1 % of $\boldsymbol{y}^*$, held $\ge 20$ ms): 18–27 ms with $\boldsymbol{g} = 0$;
36–81 ms with adapted gains, identical in both forms and limited by $\tau_v$ (worst case over 6
cases: 81, 41, 33, 23, 19 ms for $\tau_v$ = 30, 15, 10, 5, 2 ms).

Adaptation from a zero state (script settings): sustained grating (4.2 s), adapted gains
correlate $r = 1.000$ between forms (new/old median ratio 1.06). Uniform ensemble, 30 ms per
stimulus (8.4 s): the old statistic $(\boldsymbol{v} - \boldsymbol{W}_{yv}^\top\boldsymbol{\mu})^2$ averaged $13\,\theta_t$
(median) in the second half while $(\boldsymbol{c} - \boldsymbol{\mu})^2$ averaged $41\,\theta_t$: the
$\tau_v = 30$ ms filter removed about two thirds of the variance. Under the new form gains are
larger (median ratio 2.3, $r = 0.88$). In neither form had the gains converged by 8.4 s (mean gain
still rising 6–8 % over the last 0.84 s).

### Verification

- Right-hand side equals an independent per-neuron implementation of the six equations at random
  states (rtol $10^{-12}$); analytic fixed point with frozen nonzero gains is a zero of the
  right-hand side ($<10^{-12}$); under constant input from $\boldsymbol{\mu} = \boldsymbol{c}^*$, gains decrease at
  rate $\theta_t/\tau_g$ and stay at 0 (`tests/test_simulation_surround.py`).
- $\boldsymbol{g} \ge 0$ at every step of all adaptation runs.
- Independent verification (model-verifier, separate implementation): right-hand side matches a
  full model with every surround set explicit to $\le 1.7\times10^{-15}$ (N_SETS 1, 2, 6; random
  $\boldsymbol{W}_{yy}$ and $\boldsymbol{N}$ too). Frozen-gain fixed point is an exact root (residual
  $1.4\times10^{-16}$); fast-subsystem Jacobian (1326-dim) has all Re $\lambda < 0$, slowest
  $-0.032\ \text{ms}^{-1}$ (set by $\tau_v$); 20 % perturbations decay to $3\times10^{-8}$ in 300 ms.
- Gain adaptation, N = 2, K = 3 (shortened $\tau_g = 40$ ms, $\tau_\mu = 1$ s, 60 s run, feasible
  target): $\langle(c-\mu)^2\rangle/\theta_t - 1 = +1.6, -0.3, -8.5\,\%$ (SE about 7.5 %); response
  covariance within 6.2 % (relative Frobenius) of the target, sampling SE about 8 %.
- Gain adaptation, N_RF = 13, K = 182: not converged within the runs tried (36 s, shortened
  $\tau_g$); the target is feasible in the quasi-static limit (180/182 directions within 5.5 %).
  Not yet verified for the production model.
- Caveat: in a saturated-normalization N = 2 test, all gains collapsed to 0 even though the target
  was achievable, because raising a gain raised the variance there; the gain rule can fail to find
  feasible targets in that regime.
