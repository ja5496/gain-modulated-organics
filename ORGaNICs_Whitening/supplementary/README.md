# Supplementary investigation: code, data and figures

- `supplementary.tex`: the stand-alone Supplementary Information (compiles with pdfLaTeX;
  `supplementary.pdf` is the compiled copy). Upload it with the nine PNGs in `figures/`, either
  in a `figures/` folder or next to the `.tex` file.
- `SUMMARY.md`: plain summary of the findings for Jake (not part of the SI).
- `../simulation_variants.py`: every model alternative behind an option. Nothing in the existing
  model code is changed; with the default options it reproduces
  `simulation_whiten.V1Dynamics_Surround` to 1e-16.

## How to run

From this folder, with NumPy, SciPy and Matplotlib installed. Use one BLAS thread per process;
the scripts start up to four processes themselves.

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
```

```bash
python figS1_timescales.py
```

Each figure script loads its saved output from `data/` and redraws the PNG; add `--recompute`
to rerun the simulation. Figs. S8 and S9 read the outputs of earlier figures, so recompute in
the order S6, S7, S8, S9. All random sequences use fixed seeds.

## Figures

| Figure | Script | Data | Run time (4 cores) |
|---|---|---|---|
| `FigS1_timescales.png` | `figS1_timescales.py` | `data/FigS1_timescales.npz` | seconds |
| `FigS2_settling.png` | `figS2_settling.py` | `data/FigS2_settling.npz` | 1 min |
| `FigS3_tuning_old_vs_new.png` | `figS3_tuning_old_vs_new.py` | `data/FigS3_tuning_old_vs_new.npz` | 25 min |
| `FigS4_power_law.png` | `figS4_power_law.py` | `data/FigS4_power_law.npz` | 3 min |
| `FigS5_covariance_transform.png` | `figS5_covariance_transform.py` | `data/FigS5_covariance_transform.npz` | seconds |
| `FigS6_noise.png` | `figS6_noise.py` | `data/FigS6_noise.npz` | 8 min |
| `FigS7_normalization_widths.png` | `figS7_normalization_widths.py` | `data/FigS7_normalization_widths.npz` | 7 min |
| `FigS8_large_stimuli.png` | `figS8_large_stimuli.py` | `data/FigS8_large_stimuli.npz` | 2 min |
| `FigS9_rectified_variance.png` | `figS9_rectified_variance.py` | `data/FigS9_rectified_variance.npz` | 2 min |

The numbers quoted in the SI and the summary are printed by each script and kept in
`data/FigS*_report.log`, `data/sanity_checks.log` and `data/Note1_extra_checks.log`.

## Parameters shared by all figures

Set in `common.py`; taken from `analysis/adapt_ensembles.py` and `simulation_whiten.py`.

| Parameter | Value |
|---|---|
| Population | 13 neurons per receptive field, 1 cRF + 5 surround sets (78 neurons) |
| Frame | `data/frames/N13_mercedes_K182_Frame.csv` (K = 182) |
| Target | 0.15 x `uniform_target_covariance_low_c.csv` (S4, S5: `..._mid_c.csv`, as in `Dario_plots.py`); theta_t = diag(W^T C W) |
| Time constants (ms) | tau_y 0.4, tau_a 0.2, tau_u 30, tau_v 30, tau_g 500, tau_mu 2000 |
| sigma, beta | 0.15, 0.5 |
| Normalization weights | all ones unless stated; W_yy = identity |
| Surround to cRF delay | 40 ms (S4: none, as in `Dario_plots.py`) |
| Stimuli | Gaussian input profile, width 0.75 rad, contrast 0.8; 13 orientations; adaptor = orientation 6 (83.1 deg), 38 % of the biased ensemble |
| Presentation | 32 ms (S4: 30 ms) |
| Integration | RK4, dt = 0.1 ms unless stated; gains clamped at zero after every step |

## Parameters per figure

| Figure | Equations | Adaptation | Seeds | Readout and other settings |
|---|---|---|---|---|
| S1 | codebase (gains zero) | none | none | dt 0.01 ms; small grating at neuron 6's preference, from the blank-screen steady state |
| S2 | four gate x gain-placement combinations | 8.32 s, codebase, biased | 200 | dt 0.05 ms; gains frozen; switch from neuron 6's to neuron 10's preferred orientation |
| S3 | seven variants (table below) | 40 s, small adaptors | 100-107 uniform, 200-207 biased | frozen gains after 8.32 s; large test gratings every 2 deg; full dynamics (a-c), analytic fixed point (d-e) |
| S4 | new, with and without normalization; codebase for reference | 4.2 s, three von Mises environments (30/90/150 deg, kappa 6/3/1.5), small adaptors | 300-309, 400-409, 500-509 | frozen gains; large test gratings at 13 orientations; analytic fixed point; test contrast 1-100 % |
| S5 | none (linear algebra) | none | none | ridge 1e-6 x mean eigenvalue; targets: circuit, identity, uniform input covariance; kappa x 0.25-4 |
| S6 | new + noise | 4 s, small adaptors | 100-103, 200-203; noise generators 1-4 | frozen gains; large test gratings at 26 orientations; full dynamics with noise, rates averaged over 150 ms; levels in the script header |
| S7 | new, Gaussian normalization weights | 4 s, small adaptors | 100-103, 200-203 | sigma_s 0.5, 1, 2, 5, inf; sigma_theta 15, 30, 60, inf; frozen gains; full dynamics, 600 ms |
| S8 | new | 4 s, large adaptors | 100-107, 200-207; noise generator 5 | in-stream readout over 2-4 s; 12 configurations (weights x noise x surround tuning width) |
| S9 | new, standard vs rectified variance (theta_t / 2) | 4 s, small and large adaptors | 100-107, 200-207 | small: frozen gains, analytic fixed point; large: in-stream |

Equation variants in Fig. S3 (`common.VARIANTS`):

| Name | Feedback gate | Gain multiplies | Gains driven by | c |
|---|---|---|---|---|
| old (codebase) | a+/(1+a+) | feedback, in the y equation | (v - W^T mu)^2, v = c low-passed at tau_v | W^T y |
| gating | u+ | as old | as old | as old |
| placement | as old | v equation | as old | as old |
| variance | as old | as old | (c - mu)^2 | as old |
| rectify_c | as old | as old | as old | W^T max(y, 0) |
| new (Eqs. 1-6) | u+ | v equation | (c - mu)^2 | W^T max(y, 0) |
| new_matched | as new, with theta_t and tau_g divided by kappa = 0.385 | | | |

Target suppression profile (Figs. S6-S9, `common.target_profile`): 0.9 - 0.3 exp(-d^2 / 2 w^2),
w = 20 deg. These are nominal values that encode the reported pattern; they were not digitized
from the original paper. Similarity metric: r.m.s. difference between a profile and the target.

Normalization weights (Fig. S7, `common.normalization_weights`): cRF at the origin, five surround
RFs on a ring one cRF diameter away; rescaled so that the cRF neuron tuned to an adapting grating
receives the same pooled drive as with all-ones weights.

Noise (Fig. S6, `simulation_variants.NoiseOptions`): white noise is quoted as the s.d. after
1 ms low-pass filtering, as a fraction of the peak unadapted input or response; Poisson noise uses
1 ms bins and a rate scale in spikes/s at the peak unadapted input or firing rate.

## Other files

- `common.py`: protocol helpers. `plotstyle.py`: figure style (165 mm wide at 600 dpi, Okabe-Ito
  colours with fixed meanings). Arial and Helvetica are not installed on the machine that made
  the figures, so the Arial-metric clone Liberation Sans was used; the scripts pick Arial
  automatically where it exists.
- `sanity_checks.py`: regression against the existing class, analytic fixed point, time step.
- `extra_checks_note1.py`: 160 s adaptation and the running-mean test.
- `../tests/test_simulation_variants.py`: 53 pytest checks of `simulation_variants.py` (51 pass;
  the 2 failures are documented in `SUMMARY.md`).
- `Adaptive_ORGaNICs_Manucript (1).pdf`, `GainModulatedORGaNICs.bib`: context supplied by Jake.
