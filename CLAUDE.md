# Project: [repo name] — joint normalization and adaptation in ORGaNICs

## About me and this project
I'm a PhD student in physics and computational neuroscience (Martiniani and Heeger labs, NYU). This repo implements a dynamical firing-rate model of V1 that combines ORGaNICs normalization with the Duong et al. adaptive whitening framework, so that responses are both normalized and covariance-transformed. I'm comfortable with Python and the math; explain numerical and modeling choices, not basic syntax.

## Repo layout
All code lives in `ORGaNICs_Whitening/`. It is not an installable package: modules are flat files, and scripts make them importable with `sys.path.insert(0, REPO_ROOT)`, where `REPO_ROOT` is `ORGaNICs_Whitening/`. Active development is on the `splitting_scripts` branch. That branch is mid-cleanup: much of the code is deprecated and slated for removal, so check before building on an older module.

Model code (top level of `ORGaNICs_Whitening/`):
- `simulation_whiten.py`: `Frame`, `V1Dynamics` (single RF) and `V1Dynamics_Surround` (cRF plus surround sets, adaptation local to each RF)
- `frame_whiten.py`: builds and saves overcomplete frames W_yv and target covariances (also defines its own `Frame` class)
- `stimuli_whiten.py`: `StimulusGenerator`, synthetic LGN responses to oriented gratings
- `tunings_whiten.py`: `V1Tunings`, recurrent weights and tuning curves for the primary neurons

Scripts:
- `precompute_adapted_gains.py`: runs adaptation and writes steady-state gains to `data/optimal_gains/`
- `analysis/`: simulation, analysis and figure scripts, mixed together. Each has a `__main__` block, and most end with `plt.show()` rather than saving.
  - Experiment replications: `Carandini_plots.py`, `Dario_plots.py`, `Kohn_distinct_plots.py`
  - Adaptation runs: `adapt_single_stimulus.py`, `adapt_ensembles.py`
  - Analytic or self-consistent responses: `Analytic_responses.py`, `Surround_Analytic_Responses.py`
  - Diagnostics: `convergence_tests.py`, `diagnostics_whiten.py`, `normalization.py`
  - `figures.py`: schematic and covariance-diagnostic figures. Writes PNGs into `analysis/`, e.g. `crf_surround_schematic.png` and `ellipse_norm_schematic.png`.
  - `response_tools.py`: shared helpers for probing `V1Dynamics_Surround` with gains held fixed

Tests: `tests/` holds pytest tests; it's currently empty apart from `__init__.py` (see Environment).

Data (`data/`, tracked in git):
- `frames/`: frame matrices W_yv as CSV (e.g. `N13_mercedes_K182_Frame.csv`)
- `target_covs/`: target covariance matrices as CSV
- `optimal_gains/`: precomputed adapted gains (CSV) plus `meta.json`

Docs (in `ORGaNICs_Whitening/docs/`):
- `equations.md`: model equations and fixed-point derivation, transcribed from the manuscript Methods
- `whitening_adaptation_notes.md`

There is no separate figures directory, results directory or `.mplstyle` file yet.

## Environment
- Python 3.11, environment: `conda activate gain_modulation`. Run everything inside it.
- Dependencies are not specified anywhere: there is no `requirements.txt`, `environment.yml` or `pyproject.toml`. The code imports NumPy, SciPy, Matplotlib and tqdm. Don't add dependencies without asking.
- Run scripts from inside `ORGaNICs_Whitening/` (e.g. `python analysis/Dario_plots.py`). Some defaults, including `V1Dynamics_Surround`'s `target_covariance_path`, are relative paths such as `data/...`, so they only resolve from that directory.
- Tests: pytest, in `ORGaNICs_Whitening/tests/` (no tests yet). From the repo root, run `conda run -n gain_modulation --cwd ORGaNICs_Whitening python -m pytest tests/ -q`.

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
5. The system is stiff (τ_a = 0.1 ms vs. τ_g = 2 s, a ratio of ~2×10⁴). For time-varying input, use fixed-step integration (exponential Euler for leaky variables, forward Euler for g followed by the nonnegativity clamp) with dt ≤ τ_a/10 unless a convergence test justifies larger. For constant-input fixed-point tests with g frozen, `solve_ivp` with an implicit method (Radau, BDF, or LSODA) is fine. If existing code uses a different scheme, keep it.

## Verification (required before reporting any result)
New or changed model code isn't done until it's verified. Use the `model-verifier` subagent.


1. **Validity conditions.** Flag any negative component of z′, since rectification makes the fixed-point formula invalid there.
2. **Stability.** Jacobian eigenvalues of the fast subsystem at the fixed point have negative real parts; perturb the simulation and confirm it returns.
3. **Gain clamp.** Assert g ≥ 0 at every step. Under constant input, c → μ, so g decreases at rate θ_t/τ_g until it reaches 0 and stays there; z′ → z. This is expected.
4. **Gain adaptation.** Stochastic input with known covariance: after convergence, the time average of (c − μ)² matches θ_t along each column of W_yv whose gain is positive, within a tolerance that accounts for the finite averaging window (τ_μ = 2 s). For N = 2, K = 3, also check the full covariance of y. Report g trajectories, not just endpoints.
5. **Target feasibility.** Normalization bounds y, so variance along each frame direction is bounded. If θ_t exceeds what's achievable, that gain pins at 0 instead of matching the target. Report any pinned gains and whether the target was feasible.

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
- Refactors must leave numerical outputs unchanged (within float tolerance). Check this by running the affected scripts before and after with the same seeds and comparing their outputs.

## Handoff to my writing agent
After finishing a model or analysis, update `docs/methods_notes.md` with the equations implemented (LaTeX), a parameter table, the integration method and tolerances, and the verification checks run. I paste this into my writing agent for the Methods section.
Mark any claim about prior work [CITATION NEEDED]. Never cite papers from memory; my literature agent handles sources.
