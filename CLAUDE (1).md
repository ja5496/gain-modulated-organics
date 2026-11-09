# Project: [repo name] — joint normalization and adaptation in ORGaNICs

## About me and this project
I'm a PhD student in physics and computational neuroscience (Martiniani and Heeger labs, NYU). This repo implements a dynamical firing-rate model of V1 that combines ORGaNICs normalization with the Duong et al. adaptive whitening framework, so that responses are both normalized and covariance-transformed. I'm comfortable with Python and the math; explain numerical and modeling choices, not basic syntax.

## Repo layout
[To be updated from the actual repo.]
- `src/[package]/`: model code (importable package)
- `scripts/`: simulation and analysis entry points
- `tests/`: pytest tests
- `figures/`: one script per figure, plus `paper.mplstyle`
- `docs/equations.md`: source of truth for model equations and their context
- `docs/methods_notes.md`: running methods summary for my writing agent
- `refs/`: PDFs of papers being implemented
- `data/raw/`: never modify
- `results/`: simulation outputs (gitignored)

## Environment
- Python [3.x], environment: `[conda activate env-name]`. Run everything inside it.
- Stack: NumPy, SciPy, Matplotlib. Don't add dependencies without asking.
- Tests: `python -m pytest tests/ -q`

## The model
Full equations and derivations are in `docs/equations.md` (Eqs. dy, du, da, dv, dg, dmu, y_steady, z_prime). Key facts:
- Variables: principal neurons y; normalization interneurons u, a; variance interneurons v; interneuron gains g; running mean μ. y, u, a ∈ ℝᴺ; v, g, μ ∈ ℝᴷ.
- Rectification exponents differ by variable: y⁺ = ⌊y⌋², u⁺ = ⌊u⌋^(1/2), a⁺ = ⌊a⌋. So √(y⁺) = ⌊y⌋ (not √y) and (u⁺)² = ⌊u⌋.
- c = W_yvᵀ √(y⁺) feeds Eqs. dv, dg and dmu. Compute it once per step from the current y.
- **Gains are clamped nonnegative:** after every update, g ← max(g, 0).
- Constants: β = 1/2, σ = 0.15. Time constants: τ_y = 0.2 ms, τ_u = 15 ms, τ_a = 0.1 ms, τ_v = 15 ms, τ_g = 2 s, τ_μ = 2 s. Use one time unit (s or ms) throughout.
- Columns of W_yv are fixed unit vectors forming an overcomplete frame. Setting g = 0 recovers single-layer ORGaNICs.

## Working rules
**Editing.** Edit code freely, but keep changes scoped to the task; no unrelated refactoring. Work on a branch, never directly on main. Commit in small steps with messages that say what changed and why. Never push.

**Running code.** This runs on my laptop.
- Tests and quick checks (under ~2 minutes): go ahead.
- Anything longer: first run a smoke test with reduced settings (shorter duration, smaller N, coarser parameter grid). Then tell me the full command, number of timesteps, estimated runtime and memory, and what it will save, and wait for my go-ahead.
- Don't use all CPU cores without asking.

## Implementing models from equations
1. Equations come from `docs/equations.md` or a paper in `refs/`. Before writing code, restate the equations you're implementing and give the mapping from each symbol to its code name.
2. If anything is ambiguous (sign conventions, which terms are rectified, time constants, indexing), ask. Don't guess.
3. Tag each term in the code with its equation label, e.g. `# Eq. dy: recurrent drive`.
4. Keep the model definition (right-hand side), the integrator, and the analysis in separate functions.
5. The system is stiff (τ_a = 0.1 ms vs. τ_g = 2 s, a ratio of ~2×10⁴). For time-varying input, use fixed-step integration (exponential Euler for leaky variables, forward Euler for g followed by the nonnegativity clamp) with dt ≤ τ_a/10 unless a convergence test justifies larger. For constant-input fixed-point tests with g frozen, `solve_ivp` with an implicit method (Radau, BDF, or LSODA) is fine. If existing code uses a different scheme, keep it and make sure the convergence check covers it.

## Verification (required before reporting any result)
New or changed model code isn't done until it's verified. Use the `model-verifier` subagent.

1. **Fast-subsystem fixed point.** Freeze g and μ, constant z, W_yy = I. Integrate to steady state and compare element-wise with:
   - z′ = (I + W_yv D(g) W_yvᵀ)⁻¹ z
   - y* = z′ / √(σ² + N z′²)
   - u* = (σ² + N z′²) / 4
   - a*/(1 + a*) = √u*
   - v* = D(g) W_yvᵀ y*
   Agreement to [1e-6 relative]; confirm with root-finding on the RHS (`scipy.optimize.root`). Test g = 0 (single-layer ORGaNICs, z′ = z) and several nonzero g.
2. **Validity conditions.** Assert u* < 1 (otherwise Eq. da has no stable fixed point). Flag any negative component of z′, since rectification makes the fixed-point formula invalid there.
3. **W_yy ≠ I** (if used): the formula is approximate. Report the deviation; don't assert agreement.
4. **Stability.** Jacobian eigenvalues of the fast subsystem at the fixed point have negative real parts; perturb the simulation and confirm it returns.
5. **Gain clamp.** Assert g ≥ 0 at every step. Under constant input, c → μ, so g decreases at rate θ_t/τ_g until it reaches 0 and stays there; z′ → z. This is expected.
6. **Gain adaptation.** Stochastic input with known covariance: after convergence, the time average of (c − μ)² matches θ_t along each column of W_yv whose gain is positive, within a tolerance that accounts for the finite averaging window (τ_μ = 2 s). For N = 2, K = 3, also check the full covariance of y. Report g trajectories, not just endpoints.
7. **Target feasibility.** Normalization bounds y, so variance along each frame direction is bounded. If θ_t exceeds what's achievable, that gain pins at 0 instead of matching the target. Report any pinned gains and whether the target was feasible.
8. **Convergence.** Halving dt changes results by less than [tolerance].
9. **Regression.** [Baseline figure from the original ORGaNICs paper] still reproduces.

Report which checks passed, which failed, and which were skipped and why. Never loosen a tolerance, skip a test, or edit a test to make it pass. A failing check is a result; report it.

## Simulations and analysis
- Save each run's parameters (dict or YAML) alongside its output, with the git commit hash, random seed, and date.
- Set random seeds explicitly.
- Save raw outputs (`.npz`) separately from figures, so figures can be redrawn without rerunning.
- When summarizing results, give numbers with uncertainty where it applies. Flag anything surprising instead of explaining it away.

## Figures
- Use `figures/paper.mplstyle`. Target journal: [journal]; widths [single-column, double-column].
- One script per figure (`figures/fig_<name>.py`) that loads saved results and writes a vector PDF plus a PNG preview.
- Axis labels with units. Colorblind-safe palette, with the same color for the same condition across all figures [define mapping].
- No panel titles in paper figures; that information goes in captions.

## Debugging and refactoring
- Reproduce a bug with a minimal failing test before fixing it.
- Refactors must leave numerical outputs unchanged (within float tolerance). Run the regression tests before and after.

## Handoff to my writing agent
After finishing a model or analysis, update `docs/methods_notes.md` with the equations implemented (LaTeX), a parameter table, the integration method and tolerances, and the verification checks run. I paste this into my writing agent for the Methods section.
Mark any claim about prior work [CITATION NEEDED]. Never cite papers from memory; my literature agent handles sources.
