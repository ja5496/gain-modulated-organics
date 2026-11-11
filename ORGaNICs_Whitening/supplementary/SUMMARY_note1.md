# Summary for Jake: Section 2.1 (not part of the SI)

**I stopped after Section 2.1, as ground rule 1 requires: the old and new equations give a
meaningfully different headline result.** The difference does not come from either of the two
changes you listed. It comes from a third difference between the code and Eqs. 1-6 that the
instructions do not list. Sections 2.2-2.4, the LaTeX document and the bibliography are not started.

## What I need from you

1. **Which gain equation is "new"?** The code drives the gains with `(v - W^T mu)^2`, where `v` is
   `c` low-pass filtered at tau_v = 30 ms. Eq. 5 of the instructions uses `(c - mu)^2`. Options:
   - (a) Eqs. 1-6 exactly as written (variance from `c`). Adaptation is about twice as strong at
     the same time point, and the suppression at the adaptor is nearly gone at 8.3 s.
   - (b) New gate and new gain placement, but keep the code's filtered variance signal. This
     reproduces the codebase results to within 0.001 and needs one extra filtered copy of `c`
     (already implemented as `variance_signal='filtered'`).
2. **How long should adaptation last, and with which target?** The gains do not converge (details
   below), so every adapted-state result depends on the adaptation time. I need either a duration
   to use throughout, or a larger target.
3. **The manuscript and `Adaptive_ORGaNICs_refs.bib` are not in the repository.** I need both for
   the write-up (your wording, Eqs. 10-11 and 22, Figs. 3c and 3e, and the reference list).
4. **Benucci et al. (2013).** The session's network policy blocks nature.com and
   pmc.ncbi.nlm.nih.gov, so I could not read the paper. Either allow one of those hosts in the
   environment's network settings or upload the PDF.

## Findings

Protocol for all adapted-state numbers: small adaptors (cRF only), 32 ms presentations, 40 ms
surround delay, gains active for both ensembles from zero, gains frozen, large test gratings,
8 repeats per ensemble. "Ratio" is a neuron's peak response after biased adaptation divided by its
peak after uniform adaptation.

**The two listed changes do nothing measurable.**
- Gate `u+` instead of `a+/(1+a+)`: the two gates never differ by more than 5.4e-4 (0.27 %),
  because `a` follows `u` closely (its effective time constant tau_a/(1-u+) is about 0.25 ms).
  Responses differ by at most 2e-4 after a stimulus switch; the ratio changes by at most 0.0001.
  `u+` was at most 0.41 at every steady state I tested, so `a` does not slow down or diverge. The
  input norm is at most 1 by construction, which bounds the steady-state `u+` at 0.51. I did not
  log `u+` during the adaptation runs.
- Gain inside the `v` equation: exact for frozen gains (difference 1.7e-16). With adapting gains the
  ratio changes by at most 0.0005 and the gains by 0.3 %.
- The fixed point is unchanged, as you expected.

**The unlisted change is the whole difference (Fig. S3).**
- Ratio at the adaptor after 8.3 s: **0.931 +/- 0.010 (old) vs 0.990 +/- 0.011 (new)**. Largest
  change across neurons 0.072, about five times the s.e.m.
- Switching only the variance signal reproduces the new result (0.989). Rectifying `c` changes the
  ratio by at most 0.005.
- Mean gain at 8.3 s: 0.081 (old) vs 0.154 (new).
- It is not a simple rescaling. For independent 32 ms presentations the filter passes a fraction
  kappa = 0.385 of the variance of `c`. Dividing theta_t and tau_g by kappa in the new equations
  brings the mean gain back to the old level (0.072) but not the suppression at the adaptor (0.986).

**The circuit has not settled when the next stimulus arrives (Figs. S1, S2).** `a` is fast
(0.25 ms), `y` rises in about 2 ms, but `u` and `v` need 30 ms. After a new stimulus `y` overshoots
(peak at 8 ms) and is still 13 % above its steady state at 32 ms; `u` has covered 64 % of its
change. After a switch between stimuli with adapted gains, the squared deviation from the new
fixed point is still 2.7 % of its initial value at 32 ms. So responses read at the end of a
presentation are not steady-state responses. This holds equally for all four combinations of the
listed changes.

## Things that contradict or complicate the manuscript or the instructions

- **The gains do not converge.** Old equations: mean gain still growing 1.1 %/s at 40 s. New
  equations at 160 s: still growing 0.3 %/s, variances still about 1.4 times the target, and peak
  responses down to 0.8 % of the unadapted level. The target (0.15 x the low-contrast covariance) is
  far below the response variance at contrast 0.8, and the gain rate is set by that small variance.
  The 8.4 s used in `adapt_ensembles.py` reads out a transient: uniform-adapted peaks are 24 % of
  unadapted at 8.3 s and 10 % at 40 s (old equations).
- **The suppression at the adaptor fades with adaptation time.** Old equations: ratio 0.87 at
  1.9 s, 0.93 at 8.3 s, 0.99 at 40 s. New equations: 0.92, 0.99, 1.03. At long times the biased
  ensemble gives no suppression at the adaptor and facilitation of 10-25 % at the flanks.
- **Most of the early suppression comes from the start-up of the running mean.** `mu` starts at
  zero with tau_mu = 2 s, so for the first seconds the gains integrate the squared mean response,
  which is largest for adaptor-tuned neurons. With tau_mu = 200 ms the ratio at the adaptor at
  8.3 s is 0.996 instead of 0.923 (old equations, 4 repeats) and the mean gain is 0.034 instead of
  0.082. This is a test with one alternative value, not a full account.
- **There is no untuned suppression relative to the uniform-adapted state.** Far from the adaptor
  the ratio is 1.07-1.18 (facilitation). This matches what you reported finding.
- **The profile is not symmetric about the adaptor** (e.g. 0.970 at -14 deg vs 0.923 at +14 deg),
  by several s.e.m. The stimuli are symmetric, so I suspect the frame; I have not tested this.
- **Code differs from the instructions in more than the gain equation:**
  - Time constants: tau_y 0.4, tau_a 0.2, tau_u 30, tau_v 30, tau_g 500, tau_mu 2000 ms in the
    code, against tau_a 0.1 and tau_u 15 ms quoted in the instructions. I used the code's values.
  - `c = W^T y` without rectification, and `mu` tracks `y` (equivalent to tracking `c`).
  - Presentations last 30 ms in the scripts; I used 32 ms.
  - The target covariance is multiplied by 0.15 inside `V1Dynamics_Surround`.
  - tau_v / tau_g = 6 % in the code, not below 1 %.
- **Surround gains are not zero after small-stimulus adaptation** (mean 0.009, about 10 % of the cRF
  gains). The surround's input is constant but its responses vary through the shared normalization
  pool. `adapt_ensembles.py` sets them to zero before the readout; I kept them.
- **The analytic fixed point is inexact when some z' < 0.** With zero gains it matches the simulated
  steady state to 1e-10. With gains from a 4 s adaptation a few components of z' are slightly
  negative and the formula is off by up to 2 % of the peak response, in neurons that barely respond.
  Peak responses are unaffected (analytic and simulated readouts agree to 2e-5).

## Checks

- Regression: `simulation_variants.py` with codebase options reproduces `V1Dynamics_Surround` to
  1e-16 (responses, gains, v, mu, u, a) with the surround delay.
- Time step: halving dt from 0.1 to 0.05 ms changes adapted gains by 3e-6 and responses by 8e-5.
- Independent verification of `simulation_variants.py` (a separate reviewer agent, tests in
  `../tests/test_simulation_variants.py`): **verified with caveats; 49 tests pass, 4 fail.**
  - The deterministic equations match an independently written right-hand side to 1.8e-15 for all
    64 option combinations, and the old model to 1e-15 in four regression scenarios. Everything in
    Section 2.1 uses only this deterministic part.
  - The reviewer found four bugs in the noise options, which no result here uses. I fixed them:
    response noise silently forced a rectified `c`; Poisson input noise was skipped after a stimulus
    change when the presentation was not a whole number of bins; Poisson input noise was shared
    across a batch; and a helper crashed for white response noise.
  - The 4 failing tests are left failing on purpose, because they are real limits:
    1. With the surround delay, halving dt changes responses by up to 3e-3 in the reviewer's
       harsher scenario (8e-5 with my stimuli). The delayed signal is sampled once per step, as
       in `simulation_whiten.py`, so the scheme is first order in dt and the effective delay is
       half a step (0.05 ms) shorter than stated.
    2. With white input noise, the variance of the *recorded* `y` depends on dt (s.d. 13 % too
       large at dt = 0.1 ms in a linear test). What the gains integrate does not. This must be
       handled before the noise figure of Section 2.3.
  - Also flagged, not yet addressed: Poisson response noise feeds the square root of a count-based
    rate to downstream terms, which is strongly biased low at low counts.
  - Not covered by the tests: adapted variance matching theta_t under stochastic input, target
    feasibility, and regression against a baseline ORGaNICs figure (CLAUDE.md checks 6, 7, 9).

## Draft blurbs for the three figures

**Fig. S1, time scales.** A small grating is switched on from a blank screen, with adaptation
turned off. Each panel shows one population against time on a logarithmic axis (thin lines: the 13
cRF neurons; thick line: their mean); the fourth panel overlays the two gating signals. The figure
shows how long each population takes to settle. The principal neurons respond within a few
milliseconds and then overshoot, because the normalization signal `u` builds up over tens of
milliseconds; `a` follows `u` with no visible lag, so the old and new gates coincide.

**Fig. S2, settling time.** The circuit, with adapted gains held fixed, sits at its steady state
for one orientation and the stimulus then switches to another. Panel a plots the squared distance
of the responses from the new steady state against time (logarithmic vertical axis) for the four
combinations of old and new gate and gain placement; the dashed line marks the next stimulus at
32 ms. Panel b plots how far each variant departs from the codebase response. The four curves are
indistinguishable: neither change alters how fast the circuit settles.

**Fig. S3, tuning curves after biased adaptation.** The circuit adapts to the uniform and to the
biased ensemble, and tuning curves are then measured with gains frozen. Panel a overlays tuning
curves for the old and new equations (every second neuron), each divided by the neuron's peak
after uniform adaptation; panel b plots that peak ratio against preferred orientation; panel c
shows the change in the ratio caused by each change on its own; panels d and e follow the mean gain
and the ratio at the adaptor over adaptation time. The gate and the gain placement leave the result
unchanged. Taking the variance from `c` instead of the filtered `v` roughly doubles the gains and
removes most of the suppression at the adaptor.

## Not done or not verified

- Sections 2.2-2.4, the LaTeX document, the bibliography.
- Figures were inspected at one-third size plus one full-resolution crop; a final full-size pass is
  still owed before delivery.
- The manuscript's time constants (tau_a 0.1, tau_u 15 ms) were not simulated.
- The cause of the asymmetry of the profile, and whether the filtered and unfiltered variance
  signals differ because of onset transients, were not tested.
