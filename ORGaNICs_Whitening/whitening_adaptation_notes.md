# Whitening / Adaptation Model — Session Notes

## Goal
A biologically plausible adaptation/normalization circuit for V1: decorrelate/scale
down inputs like whitening does, but with realistic constraints — few interneurons,
gains that can only be >=0, and adaptation that suppresses (never amplifies).

## What we tried, in order, and what broke each time

### 1. Exact whitening target is mathematically incompatible with `g>=0`
`get_optimal_gains`'s target (`A = sqrt(Cov) - I`) is negative-definite everywhere
(eigenvalues -1 to -0.66). A nonnegative-weighted sum of outer products
(`W·diag(g)·W.T`) can only ever be positive-semidefinite. The `g = max(g,0)` clip
was fighting an impossible constraint almost everywhere — the likely root cause of
the jagged/noisy fits seen early on, more fundamental than "too many interneurons."

### 2. Fix: shrink-to-mean target (already in the codebase as `get_optimal_gains_target`)
Defines the target relative to the ensemble's *own* mean variance
(`d = min(1, sqrt(target/lambda))`) — only shrinks above-average directions, leaves
the rest alone. This target (`A = T^-1 - I`) is fully PSD (eigenvalues 0 to 9.26) —
representable with nonnegative gains, and matches "adaptation only suppresses."

### 3. Small-pool prototype (M=4-16 channels instead of the K~14000 overcomplete frame)
Found the shrink-to-mean target is nearly rank-2 for this toy ensemble — a cos/sin
"first harmonic" pair captures it almost exactly (harmonics: ~1% error; data-optimal
eigenbasis: ~0% error), with only 2 active gains. Promising, but:

### 4. Flaw: both bases are global, causing non-local "ripple" artifacts
Fed a narrow, off-adaptor test probe through the fitted circuit — got a
full-amplitude sinusoidal ripple across the *entire* population, unrelated to the
probe's actual narrow content. Root cause: feedback = `g_k*(w_k·y)*w_k` — the shape
of the correction is always the shape of `w_k` itself, regardless of where the
input's energy was. Harmonics and this problem's data-driven eigenvectors both
happened to be broad/global, so both leak everywhere.

### 5. Fix: local + tight pool basis (M=24 raised-cosine pools, 15 deg wide each)
Built via an exact partition-of-unity construction (`cos^2+sin^2=1` between
neighbors) — verified perfectly tight (`sum_k w_k^2 = 1` to 1e-15) and perfectly
local (same narrow-probe test: exactly zero feedback outside the probe's footprint,
vs. the large ripple from harmonics). This fully resolved the locality/tightness
concerns.

### 6. New flaw: local nonneg pools structurally can't fit the exact covariance target
40-60% relative error, *worsening* with more pools — not a resolution problem, a
sign problem. Local, nonnegative pools can only contribute local, same-sign
correlations. The shrink-to-mean target needs long-range, opposite-signed structure
(positive lobe at the adaptor, negative lobe at the orthogonal orientation) that
local nonneg pools cannot produce at any M.

### 7. Bigger realization: the resulting gain profile was backwards
Fit against the biased ensemble, gains *dipped* at the adaptor and *peaked* at the
orthogonal orientation — opposite of real adaptation. Traced to: the wide tuning
curve (0.75 rad) pushes the adaptor-tuned neuron into a "reliably high" response
rather than an evenly-split high/low one, and **variance** is maximized by an even
split, not a reliable high — so the adaptor's own neuron actually has slightly
*lower* variance than neurons farther away. This suggested the whole premise —
driving gain from population **variance/covariance** — is mismatched with real
adaptation, which tracks a neuron's **mean** recent drive (e.g., synaptic
depression), not its variance.

### 8. Switch to mean-driven gains — no matrix-fitting at all
Each local pool's gain set directly from a saturating function of its own mean
pooled drive. No covariance, no eigendecomposition, no fitting. Correctly peaked at
the adaptor (3.03 vs. baseline 2.50), flat for the uniform ensemble,
locality/tightness preserved for free. Essentially a divisive-normalization rule,
consistent with the normalization already elsewhere in this codebase, just spread
across several local, slowly-adapting pools instead of one instantaneous global one.

### 9. Newest flaw (just found, partially fixed): facilitation bigger than suppression
The first version of this compared mean drive between a "biased" *mixture*
ensemble (adaptor competing with 168 other orientations for a fixed 252-trial
budget) and the uniform ensemble. Result: 7.6% suppression at the adaptor but
**15% facilitation** at the orthogonal orientation — backwards, since real
adaptation should be suppression-dominant.

- **Diagnosis:** the mixture-ensemble comparison is the wrong paradigm. Since the
  adaptor eats 1/3 of a *fixed* trial budget, every other orientation necessarily
  gets fewer trials than under the uniform ensemble — so their gain drops as an
  artifact of shared bookkeeping, not biology. Real experiments show one sustained
  adaptor; unrelated orientations should see **no change**, not an induced
  decrease.
- **Fix attempted:** baseline gain (flat, from generic viewing) + a purely
  additive, nonnegative "adaptation boost" computed from the sustained adaptor's
  *direct* drive on each pool (no ensemble averaging — one stimulus, so its drive
  is just one dot product). Guarantees gain can only go up, never down.
- **Preliminary result (last thing computed, not fully discussed):** properly
  suppression-dominant now — 33% suppression at the adaptor, decaying to ~15% at
  the orthogonal orientation, never going negative/facilitatory. Shape looks
  right; there's a fairly broad "floor" of suppression even 90 deg away that may
  be too broad compared to real curves — plausibly an artifact of the toy
  stimuli's very wide (0.75 rad ~ 43 deg) tuning curves, not yet checked against a
  narrower/more realistic tuning width.

### 10. Revisited: "online PCA" proposal — trying to stick with variance control
Idea: instead of factorizing the full `N×N` covariance, do an online/incremental
PCA — track a small number of directions where the *current* input distribution's
covariance differs most from a *reference* distribution, whiten only within that
low-dim subspace, project back. Mathematically this is a **generalized eigenvalue
problem** (`Cov_current v = lambda Cov_baseline v`), a principled upgrade over the
ad hoc "shrink to the ensemble's own mean" target — it compares against a genuine
reference instead of a self-referential yardstick.

Two design decisions were made:
- **Baseline = fixed developmental prior** (learned once from the uniform/typical
  ensemble, then frozen), not a second slow-online-tracked distribution.
- **Enforce locality**, which forces the current-vs-baseline comparison to be
  **diagonal** (each pool judged only against its own baseline) rather than a full
  cross-pool generalized eigendecomposition — a full `M×M` version could still
  mix distant pools into one "principal direction" and reintroduce the delocalized
  ripple artifact from step 4.

**Key snag + resolution:** variance requires a distribution to be defined over,
but the sustained-single-adaptor condition (needed to fix step 9's facilitation
bug) is a literally-constant stimulus with zero across-trial variance in this
noiseless toy model. Resolution: assume standard trial-to-trial neural response
noise (Poisson, `Var=mean`, or constant-CV, `Var=mean^2`) on top of the constant
stimulus. Verified numerically (`prototype_relative_variance_adaptation.py`):
under *either* noise model, once the comparison is forced diagonal (no cross-pool
terms) and the stimulus is deterministic, the resulting "relative variance" signal
reduces to a monotonic function of mean drive — Poisson variant gives 33.3%/15.3%
suppression (adaptor/orthogonal), essentially identical to step 8's mean-driven
result; constant-CV gives 34.4%/13.4%, marginally steeper but same shape family.

**Conclusion: "stick with variance control" and "the mean-driven fix from step 8"
were never actually in conflict.** Step 7's "variance is backwards" finding came
from measuring variance the wrong way (cross-stimulus spread across a competing
mixture ensemble). Once variance is defined the right way (trial-to-trial response
noise under sustained viewing) *and* locality is enforced (forcing the diagonal
restriction), it necessarily collapses onto the same signal as mean drive — there
is no remaining freedom for it to say anything else, since with no cross-pool
terms and a deterministic input, mean drive is the only quantity left for noise to
scale with. Practical upshot: a full incremental eigendecomposition isn't needed;
a per-pool leaky running-mean estimate (already the style used in
`get_response_fast_adapt`) is sufficient to implement this "online, variance-based,
local" adaptation rule.

### 11. Short vs. long adaptation timescales — testing a real physiological dissociation
Motivating observation (Dragoi, Sur & Rao 2000, and others): brief adaptation
produces suppression + repulsive tuning-curve shifts; prolonged adaptation
produces near-adaptor facilitation, attractive shifts, and tuning-curve
narrowing toward the adaptor. Hypothesis: brief adaptation = the mean-driven
(step 8/10) signal; prolonged adaptation = the raw cross-stimulus-variance
signal (step 7, which dipped at the adaptor) once trial-to-trial noise is
averaged away.

**First check (`prototype_timescale_consistency_check.py`):** built the two
gain profiles (short = mean-driven; long = a direct per-pool cross-stimulus
variance ratio) and simulated population responses to test probes at various
offsets from the adaptor, decoding perceived-orientation shift (population
vector) and tuning sharpness. Result was a genuine partial match, not a clean
confirmation: the short-timescale signal gave clean repulsion across most of
the tested range (matches literature); the long-timescale signal gave a
biphasic curve — small repulsion very close to 0°, a real *attraction* zone at
15-40° offset, decaying back toward zero at large offsets — so an attraction
zone did emerge specifically from the variance-driven signal, but the
"narrower toward the adaptor" prediction did NOT hold (both signals showed
*broader*, not narrower, response right at the adaptor; sharpening only
appeared at large offsets, for both signals).

**Refined hypothesis:** one mechanism, two timescales — a short running window
whose variance estimate is dominated by trial noise (~ mean), and a long
running window whose variance estimate is dominated by genuine stimulus
statistics because the noise averages out. Tested with an actual online
leaky-variance tracker (`prototype_online_timescale_prototype` /
`prototype_bursty_timescale.py`) fed a real temporal stimulus stream (not a
static ensemble comparison).

**First simulation (i.i.d. stream, freshly redrawn every trial):** no
transition at all — contrast stayed positive (peaks at adaptor) at every
tested integration window from ~3 to ~1000 trials. Diagnosis: i.i.d.
resampling has no "stuck looking at one thing" property even for a short
window — a short window is just a noisier estimate of the *same* underlying
mixture, not a differently-biased one.

**Second simulation (bursty stream — stimulus held fixed for a
geometrically-distributed burst, mean 25 trials, before redrawing — modeling
real fixation/attention dwell time):** still no sign flip between a fast
(window~5) and slow (window~2000) raw-variance tracker (contrast +2.40 vs
+1.58). Root cause, confirmed by direct calculation: under Poisson-like trial
noise, the quantity any leaky variance tracker converges to (regardless of
window length, once converged) is `E[mean] + Var[true stimulus-driven drive]`
(law of total variance) — integration time only sharpens the estimate of this
fixed sum, it never separates which term dominates. At the adaptor pool,
mean=4.82 and true variance=4.45 (sum 9.27); at the far pool, mean=2.87 and
true variance=4.66 (sum 7.53) — the mean gap (68%) swamps the true-variance
gap (~5%) in the sum, permanently, at every timescale.

**Fix that worked:** don't read out raw variance at the long timescale — read
out **excess variance relative to the Poisson floor** (`Var - Mean`, a
Fano-factor-style computation that explicitly subtracts the mean's own
contribution before comparing). This is a well-established neuroscience
quantity, not an ad hoc fix. Recomputed: excess at adaptor = 4.45-4.82 = -0.37
(sub-Poisson), excess at far pool = 4.66-2.87 = +1.78 (super-Poisson) — a
clean, correctly-signed contrast. Rerunning the bursty simulation with this
readout at the slow timescale gave contrast **-0.14** (dips at adaptor) vs.
the fast/raw-variance pathway's **+2.40** (peaks at adaptor) — the
hypothesized sign flip, confirmed mechanistically.

**Caveats, explicitly not yet checked:** the flipped effect size (-0.14) is
real but much smaller than the short-timescale effect (+2.40) — plausibly
consistent with long-duration attraction effects generally being weaker/subtler
than short-duration repulsion in the literature, but this has only been
checked for *sign*, not for realistic *magnitude*. The burst-length parameter
(mean 25 trials) was picked arbitrarily; sensitivity to that choice is
untested. The tuning-curve-shift/sharpness decoding from step 11's first check
(population-vector decode, resultant-length sharpness metric) has not yet been
rerun with this corrected long-timescale (excess-variance) signal — that would
be the natural next check, to see if it now also produces the predicted
narrowing, not just the right-signed gain dip.

## Open threads for next time

1. **Rerun the tuning-shift/sharpness decode from step 11 using the
   corrected long-timescale signal** (excess variance from the bursty
   simulation, not the crude batch cross-stimulus-variance ratio used in the
   first pass) to check whether attraction *and* narrowing both emerge now.
2. **Check realistic magnitude and burst-length sensitivity** for the
   short/long dissociation above — so far only the *sign* has been verified,
   with one arbitrary burst-length choice.
3. **Build the real online/temporal version for production use.** Everything
   so far compares two static conditions (baseline vs. fully-adapted) or runs
   offline batch simulations. Next step: a genuine leaky integrator per pool
   (mean AND variance, both timescales) that could run inline in the actual
   response pipeline, not just in standalone diagnostic scripts.
4. **Fit the saturating nonlinearity's parameters** (baseline gain, max boost,
   saturation point, exponent, noise-scaling assumption) against real published
   adaptation curves rather than the illustrative values used throughout.
5. **Re-examine pool count/width** (M=24, 15 deg was arbitrary) and **check
   whether the broad ~15% far-surround suppression floor is real** (some
   studies report broadband contrast adaptation) or an artifact of the toy
   stimulus's 0.75 rad tuning width.
6. **Nothing has been changed in production code except the diagnostic plot
   inside `get_optimal_gains`** (now takes `theta`/`adaptor_theta`/
   `tuning_width` and shows 4 panels including Gaussian-probe comparisons).
   All prototyping lives in `prototype_local_mean_adaptation.py`,
   `prototype_relative_variance_adaptation.py`,
   `prototype_timescale_consistency_check.py`, and
   `prototype_bursty_timescale.py` (all in this same folder) — nothing has
   replaced the existing `K~14000`-frame pipeline yet.
