# Project: [repo name] — gain modulation / adaptation in ORGaNICs

## About me and this project
I'm a PhD student in physics and computational neuroscience (Martiniani and Heeger labs, NYU). This repo implements ORGaNICs and extends it with [a gain-modulation / adaptation mechanism]. I'm comfortable with Python and the math; explain numerical and modeling choices, not basic syntax.

## Repo layout
[Edit to match the real layout.]
- `src/[package]/`: model code (importable package)
- `scripts/`: simulation and analysis entry points
- `tests/`: pytest tests
- `figures/`: one script per figure, plus `paper.mplstyle`
- `docs/equations.md`: source of truth for model equations (LaTeX, numbered)
- `docs/methods_notes.md`: running methods summary for my writing agent
- `refs/`: PDFs of papers being implemented
- `data/raw/`: never modify
- `results/`: simulation outputs (gitignored)

## Environment
- Python [3.x], environment: `[conda activate env-name]`. Run everything inside it.
- Stack: NumPy, SciPy, Matplotlib. Don't add dependencies without asking.
- Tests: `python -m pytest tests/ -q`

## Working rules
**Editing.** Edit code freely, but keep changes scoped to the task; no unrelated refactoring. Work on a branch, never directly on main. Commit in small steps with messages that say what changed and why. Never push.

**Running code.** This runs on my laptop.
- Tests and quick checks (under ~2 minutes): go ahead.
- Anything longer: first run a smoke test with reduced settings (fewer timesteps, smaller N, coarser parameter grid). Then tell me the full command, estimated runtime and memory, and what it will save, and wait for my go-ahead.
- Don't use all CPU cores without asking.

## Implementing models from equations
1. Equations come from `docs/equations.md` or a paper in `refs/`. Before writing code, restate the equations you're implementing and give the mapping from each symbol to its code name.
2. If anything is ambiguous (sign conventions, which terms are rectified, time constants, indexing), ask. Don't guess.
3. Tag each term in the code with its equation number, e.g. `# Eq. 3: recurrent drive`.
4. Keep the model definition (right-hand side), the integrator, and the analysis in separate functions.
5. Default to `scipy.integrate.solve_ivp` [or our existing integrator] and state the method and tolerances. For fixed-step integration, justify dt relative to the smallest time constant.

## Verification (required before reporting any result)
New or changed model code isn't done until it's verified. Use the `model-verifier` subagent. Required checks:
- **Analytic limits:** match known closed-form results. [List yours, e.g. the steady-state response in the regime where it reduces to the normalization equation; behavior when the gain/adaptation term is switched off.]
- **Fixed points:** steady state from integration agrees with root-finding on the RHS (`scipy.optimize.root`).
- **Stability:** Jacobian eigenvalues at the fixed point are consistent with the simulated dynamics.
- **Convergence:** halving dt or tightening tolerances changes results by less than [1e-6 relative].
- **Regression:** the baseline model still reproduces [a specific figure from the original ORGaNICs paper].

Report which checks passed, which failed, and which were skipped and why. Never loosen a tolerance, skip a test, or edit a test just to make it pass. A failing check is a result; report it.

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
