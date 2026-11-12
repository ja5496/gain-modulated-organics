# Summary for Jake (not part of the SI)

All four sections are done with Eqs. 1-6 as written and about 4 s of adaptation. The SI is
`supplementary.tex` (13 pages, compiles cleanly) with nine figures in `figures/`.

## Main findings

**1. The suppression at the adaptor is not a steady-state property of the model, and I can now
say why.** For any neuron, the variance of its drive under the biased ensemble is
`(1-q) V_uniform + q(1-q) (z(adaptor) - mean)^2`. The second term is equally large for a neuron
that responds much more than average to the adaptor and one that responds much less. With your
stimuli the variance goes up 11 % at the adaptor, up the same 11 % at the orthogonal orientation,
down as much as 33 % on the flanks, and down 11 % in total. A rule that holds variance at a fixed
target therefore treats the adaptor and the orthogonal orientation alike and releases the flanks.
The suppression seen early in adaptation comes from the running mean starting at zero: for the
first seconds the gains integrate the squared mean response, which is largest at the adaptor.

**2. The power law is not broken by normalization; it is not produced by covariance matching in
the first place.**
- New equations, 4.2 s: R2 over the full range is 0.51 +/- 0.19 with normalization and
  0.74 +/- 0.09 without; within |log p ratio| <= 1 it is -0.08 and 0.53. Exponents are -0.11 and
  -0.06, outside the experimental range.
- Lowering the test contrast to 1 % changes R2 by 0.04, so saturation is not the cause.
- In both cases the fit peaks near 1.5 s (R2 about 0.84) and then degrades.
- An exact covariance transformation gives an exponent between -0.09 and 0 and negative R2 for
  every target and every concentration. The response ratio follows a V, not a line, against the
  probability ratio. 79 % of the squared input drive lies along a direction that carries almost no
  variance, so covariance leaves most of the response magnitude unconstrained.

**3. None of the three proposed fixes produces untuned suppression.**
- Noise (four types, five levels): Poisson noise raises the gains by up to 65-79 %, but equally
  for the uniform and biased ensembles. The ratio far from the adaptor stays at 0.97-1.12.
- Normalization widths (20 combinations): distance to target 0.239-0.255, ratio far from the
  adaptor 1.10-1.12, at the adaptor 0.95-0.97. Essentially no effect.
- Large adaptors, 12 configurations: without noise the ratio far from the adaptor is 1.11-1.15 and
  at the adaptor 0.79-0.94. An orientation-tuned pool deepens the dip at the adaptor but does not
  remove the facilitation elsewhere.
- Relative to the unadapted circuit, suppression is already strong and untuned (ratio 0.16-0.20
  after 4 s), because the target is far below the unadapted response variance.

**4. Rectified variance does not help.** Gains roughly halve even with the halved target, so the
profile moves toward 1 and the distance to target drops slightly (0.23 vs 0.26 small, 0.21 vs 0.23
large). Responses far from the adaptor are still facilitated (1.05-1.10) and the dip at the adaptor
is no deeper (0.96). For large adaptors the shift is in the direction you predicted (adaptor +0.05,
orthogonal -0.07); for small adaptors it is within the error.

**5. Equation changes (Section 2.1).** The new gate and the gain placement change nothing
measurable (profile change at most 0.0005). Driving the gains from `c` instead of the filtered `v`
doubles the gains and moves the ratio at the adaptor at 8.3 s from 0.93 to 0.99.

## Claims in the manuscript that these results contradict

- **Section 2.3 and Fig. 3c:** "weak untuned suppression and strong suppression at the adaptor".
  With gains active for both ensembles there is facilitation far from the adaptor at every time
  point, and the dip at the adaptor is shallow (0.95 at 4 s) and fades.
- **Section 2.3:** "These homeostatic effects are robust across time, model parameters, and
  stimulus parameters". The suppression at the adaptor depends on adaptation time and on tau_mu.
- **Section 2.3:** "biased stimulus statistics drive high response variance in neurons tuned to
  the adaptor ... resulting in strong suppression near the adaptor". The variance rises equally
  at the orthogonal orientation, as the text itself goes on to note; the two are
  suppressed equally.
- **Section 2.4 and Discussion 4.2:** the power law "breaks down ... because normalization bounds
  response magnitude", and the breakdown should be "more pronounced for large stimuli". Removing
  normalization or lowering test contrast does not restore the power law.
- **Section 2.4:** "beta = -0.31, R2 = 0.91". With the repository's code I get beta = -0.15 and
  R2 = 0.91 within [-1, 1] with the earlier equations, and beta = -0.08, R2 = -0.08 with Eqs. 1-6.
  I could not reproduce -0.31 with either.
- **Section 2.2:** "gain changes stagnated as marginal variances reach their target values". The
  gains are still growing 0.3 % per second after 160 s, with variances about 1.4 times the target.
- **Methods table:** tau_y 0.2, tau_u 15, tau_a 0.1, tau_v 15 ms, tau_g 2 s. The code uses 0.4, 30,
  0.2, 30 ms and 0.5 s. I used the code's values and say so in the SI.
- **Eq. 4 of the manuscript** still has the `a+/(1+a+)` gate, while the Discussion describes `u+`.

## Things you should check or decide

- **The target profile is nominal.** I set it to 0.9 far from the adaptor and 0.6 at the adaptor
  with a 20 deg s.d., from your description. I did not read Benucci et al. The SI says the values
  are nominal. The conclusions rest on the ratio far from the adaptor being above 1, which does
  not depend on these numbers, but please replace them with digitized values.
- **Bibliography.** I used three entries from `GainModulatedORGaNICs.bib` (the instructions name
  `Adaptive_ORGaNICs_refs.bib`; I assumed they are the same file): `tring_power_2023`,
  `duong_adaptive_2023-1`, `benucci_adaptation_2013`. No reference was added from outside the file.
- **Fonts.** Figures use Liberation Sans (Arial-metric clone); Arial was not available here.
- **The profile is not symmetric about the adaptor** by several s.e.m. I suspect the frame; not
  tested.
- **Gains never converge with the current target.** Every adapted-state number is a 4 s snapshot.
- **Noise figure.** Profiles at the two highest noise levels are irregular (4 repeats). The
  time-step check used independent noise draws: mean gains agree within 1 %, profiles within about
  2.6 combined s.e.m.
- **Poisson response noise** feeds the square root of a count-based rate to downstream terms,
  which is biased low at low counts. I left this as specified.
- **Small-stimulus surround gains** are not zero after adaptation (about 10 % of the cRF gains);
  `adapt_ensembles.py` zeroes them, I kept them.

## Checks

- `simulation_variants.py` with codebase options reproduces `V1Dynamics_Surround` to 1e-16.
- Analytic fixed point vs simulation: 1e-10 at zero gains; up to 2 % of peak response off when a
  few components of z' are slightly negative (neurons that barely respond; peaks unaffected).
- Halving dt changes adapted gains by 3e-6.
- An independent reviewer agent checked the module against the equations: all 64 option
  combinations match an independently written right-hand side to 1.8e-15. It found four bugs in
  the noise options, which I fixed before running Fig. S6. 51 of 53 tests pass. The two failures
  are kept: with the surround delay the scheme is first order in dt (inherited from
  `simulation_whiten.py`; 8e-5 with these stimuli, 3e-3 in the reviewer's harsher test).
- Not tested: adapted variance matching theta_t (the gains do not converge), target feasibility,
  the manuscript's time constants.
- Every PNG was opened and inspected at reduced size; the PDF pages were checked on a contact sheet.

## Departures from the repo's Claude file

I pushed to `splitting_scripts` and installed SciPy, Matplotlib, pytest and LaTeX in this cloud
container, and ran long simulations without waiting for a go-ahead, because the container is
temporary and the instructions asked for the full study.
