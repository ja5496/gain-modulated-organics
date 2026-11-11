# Supplementary investigation: code, data and figures

Status: **Section 2.1 (Supplementary Note 1) only.** Section 2.1 found a meaningful difference
between the codebase equations and the revised equations, so work stopped there as instructed
(ground rule 1). See `SUMMARY_note1.md`.

Nothing in the existing model code is changed. Every alternative lives behind an option in
`../simulation_variants.py` (`EquationOptions`, `NoiseOptions`); with the default options it
reproduces `simulation_whiten.V1Dynamics_Surround` to 1e-16.

## How to run

From this folder, with NumPy, SciPy and Matplotlib installed:

```bash
python sanity_checks.py                       # about 5 min on 4 cores
python figS1_timescales.py                    # seconds
python figS2_settling.py                      # about 1 min
python figS3_tuning_old_vs_new.py             # redraws from data/; add --recompute to rerun (about 25 min on 4 cores)
python extra_checks_note1.py                  # prints from data/; --tau-mu (2 min), --long (12 min) to rerun
```

Each figure script loads its saved output from `data/` and redraws the PNG. `--recompute` reruns
the simulation. All random sequences use fixed seeds (listed below).

## Figures

| Figure | Script | Data | What is simulated |
|---|---|---|---|
| `figures/FigS1_timescales.png` | `figS1_timescales.py` | `data/FigS1_timescales.npz` | Blank screen to one small grating at t = 0, zero gains, 50 ms |
| `figures/FigS2_settling.png` | `figS2_settling.py` | `data/FigS2_settling.npz` | Stimulus switch with frozen adapted gains, four gate x gain-placement combinations, 100 ms |
| `figures/FigS3_tuning_old_vs_new.png` | `figS3_tuning_old_vs_new.py` | `data/FigS3_tuning_old_vs_new.npz` | Adaptation to uniform and biased ensembles for seven equation variants, 40 s, 8 repeats each |

Numbers quoted in the summary are printed by each script and kept in `data/*_report.log`,
`data/sanity_checks.log` and `data/Note1_extra_checks.log`.

## Parameters

Shared by all figures (set in `common.py`; taken from `analysis/adapt_ensembles.py` and
`simulation_whiten.py`, which are the reference):

| Parameter | Value |
|---|---|
| Population | 13 neurons per receptive field, 1 cRF + 5 surround sets (78 neurons) |
| Frame | `data/frames/N13_mercedes_K182_Frame.csv` (K = 182) |
| Target | 0.15 x `data/target_covs/uniform_target_covariance_low_c.csv`; theta_t = diag(W^T C W) |
| Time constants (ms) | tau_y 0.4, tau_a 0.2, tau_u 30, tau_v 30, tau_g 500, tau_mu 2000 |
| sigma, beta | 0.15, 0.5 |
| Normalization weights | all ones; W_yy = identity |
| Surround to cRF delay | 40 ms; the delayed signal is held at its blank-screen value until it arrives |
| Stimuli | Gaussian input profile, width 0.75 rad, contrast 0.8; 13 orientations; adaptor = orientation 6 (83.1 deg) |
| Presentation | 32 ms (`adapt_ensembles.py` uses 30 ms) |
| Integration | RK4, fixed step; gains clamped at zero after every step |

Per figure:

| Figure | dt (ms) | Seeds | Other |
|---|---|---|---|
| S1 | 0.01 | none | small grating at the preferred orientation of cRF neuron 6; starts from the blank-screen steady state |
| S2 | 0.05 | 200 (adapting sequence) | gains after 260 presentations of the biased ensemble (codebase equations); switch from neuron 6's to neuron 10's preferred orientation, small gratings |
| S3 | 0.1 | 100-107 (uniform), 200-207 (biased) | small adaptors; 1250 presentations; readout after 260 (8.32 s) with large test gratings every 2 deg; full dynamics settled for 600 ms for panels a-c, analytic fixed point for panels d-e (they agree to 2e-5) |

Equation variants in Fig. S3 (`common.VARIANTS`, `figS3_tuning_old_vs_new.model_for`):

| Name | Feedback gate | Gain multiplies | Gains driven by | c |
|---|---|---|---|---|
| old (codebase) | a+/(1+a+) | feedback, in the y equation | (v - W^T mu)^2, v = c low-passed at tau_v | W^T y |
| gating | u+ | as old | as old | as old |
| placement | as old | v equation | as old | as old |
| variance | as old | as old | (c - mu)^2 | as old |
| rectify_c | as old | as old | as old | W^T max(y, 0) |
| new (Eqs. 1-6) | u+ | v equation | (c - mu)^2 | W^T max(y, 0) |
| new_matched | as new, with theta_t and tau_g divided by kappa = 0.385 | | | |

## Other files

- `common.py`: protocol helpers (seeded ensembles, adaptation with the surround delay, frozen-gain readout).
- `plotstyle.py`: figure style. Final sizes 165 mm wide at 600 dpi; Okabe-Ito colours with fixed
  meanings (blue = old equations, vermillion = new equations). Arial and Helvetica are not installed
  on the machine that made these figures, so the Arial-metric clone Liberation Sans was used.
- `sanity_checks.py`: regression against the existing class, analytic fixed point, time step.
- `extra_checks_note1.py`: 160 s adaptation (do the gains converge?) and the running-mean test.
- `../tests/test_simulation_variants.py`: pytest checks of `simulation_variants.py`.
