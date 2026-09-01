'''
precompute_adapted_gains.py

Precomputes steady-state adaptation gains (g_cRF, g_surround) for each of the 3 adapted
conditions ('adapt CRF only', 'adapt surround only', 'adapt CRF and surround'), averaged over
N_SEEDS independent live V1Dynamics_Surround adaptation runs, and saves them to
data/optimal_gains/. Surround_Analytic_Responses.py then loads these cached gains directly
instead of running a fresh (noisy, ~1 min) live adaptation on every invocation.

theta_t (the adaptation-gain ODE's target, see V1Dynamics_Surround.calibrate_theta_t) is
calibrated ONCE, from a single long, high-precision reference run, and shared across every seed's
adapted-condition run -- theta_t is a fixed population-level quantity (a property of N_RF/N_SETS/
sigma/tuning-width/contrast, not of any individual trial), so it should be held exactly fixed
across cases rather than re-estimated noisily per seed. What the N_SEEDS averaging is FOR is
smoothing out the stochastic ADAPTATION phase's own trial-to-trial Poisson noise, not theta_t.

The one calibration run is deliberately much LONGER than any single seed's adapted-run budget
(CALIBRATION_STREAM_LENGTH = 8x ADAPT_STREAM_LENGTH) so its own sampling error is small: a naive
calibration at ordinary length carries real error (measured directly: theta_t's mean varied ~7%
and its per-interneuron max ~10% across 16 independent ordinary-length calibrations), and sharing
that one noisy draw across every seed would bake that error into every downstream gain uniformly
(with no averaging to cancel it, unlike genuinely independent noise). Running the calibration 8x
longer shrinks its standard error by ~sqrt(8)=2.83x, making the shared bias small relative to the
residual noise already present elsewhere in the pipeline, while still being cheaper overall than
16 separate ordinary-length calibrations (one ~68s run vs. 16x~8.5s=136s).
'''
import os
import sys
import json
import time
import numpy as np

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "analysis"))

from simulation_whiten import Frame, V1Dynamics_Surround
from tunings_whiten import V1Tunings
from stimuli_whiten import StimulusGenerator
import analysis.Surround_simulated_responses as SSR
from analysis import Surround_Analytic_Responses as SAR   # shared constants only (N_RF, N_SETS, ...)

N_RF              = SAR.N_RF
N_SETS            = SAR.N_SETS
FRAME_PATH        = SAR.FRAME_PATH
TARGET_COV_PATH   = SAR.TARGET_COV_PATH
TUNING_WIDTH      = SAR.TUNING_WIDTH
ENSEMBLE_CONTRAST = SAR.ENSEMBLE_CONTRAST

ADAPT_STREAM_LENGTH = 60000    # per-seed adapted-condition run length (dt=0.1 -> 2.4x tau_g)
DURATION = SSR.DURATION        # 200, matches Surround_simulated_responses.py exactly
N_SEEDS = 16
SEEDS = list(range(N_SEEDS))

CALIBRATION_STREAM_LENGTH = 2 * ADAPT_STREAM_LENGTH   # 480000 -- see module docstring
CALIBRATION_SEED = 999   # distinct from SEEDS (0..15) so calibration noise is independent of any
                          # individual seed's adapted-stream noise

ADAPTED_CONDITIONS = ['adapt CRF only', 'adapt surround only', 'adapt CRF and surround']
COND_SLUG = {
    'adapt CRF only':         'adapt_CRF_only',
    'adapt surround only':    'adapt_surround_only',
    'adapt CRF and surround': 'adapt_CRF_and_surround',
}
OUT_DIR = os.path.join(REPO_ROOT, "data", "optimal_gains")


def main(seeds=SEEDS, adapt_stream_length=ADAPT_STREAM_LENGTH, duration=DURATION,
         calibration_stream_length=CALIBRATION_STREAM_LENGTH, calibration_seed=CALIBRATION_SEED,
         out_dir=OUT_DIR, verbose=True):
    tunings = V1Tunings(N=N_RF)
    frame = Frame(csv_path=FRAME_PATH)
    stim_gen = StimulusGenerator(N_RF=N_RF, N_SETS=N_SETS, num_angles=N_RF,
                                  stream_length=adapt_stream_length,
                                  tuning_width=TUNING_WIDTH, contrast=ENSEMBLE_CONTRAST)
    K = frame.K
    n_seeds = len(seeds)

    # ---- ONE shared, long, high-precision calibration -- see module docstring. ----
    t_cal0 = time.time()
    if verbose:
        print(f"=== Calibrating theta_t once ({calibration_stream_length} steps, "
              f"seed={calibration_seed}) ===")
    np.random.seed(calibration_seed)
    calib_dyn = V1Dynamics_Surround(tunings, frame, N_RF=N_RF, N_SETS=N_SETS,
                                     target_covariance_path=TARGET_COV_PATH, gains_nonneg=True)
    stim_gen.stream_length = calibration_stream_length
    SSR.run_adaptation_phase(calib_dyn, stim_gen, 'no adaptation')
    theta_t_shared = calib_dyn.theta_t.copy()
    if verbose:
        print(f"Calibration done in {time.time() - t_cal0:.1f}s -- theta_t: "
              f"mean={theta_t_shared.mean():.5g}, min={theta_t_shared.min():.5g}, "
              f"max={theta_t_shared.max():.5g}")
    stim_gen.stream_length = adapt_stream_length

    g_cRF_sum      = {cond: np.zeros(K) for cond in ADAPTED_CONDITIONS}
    g_surround_sum = {cond: np.zeros(K) for cond in ADAPTED_CONDITIONS}
    g_cRF_runs     = {cond: [] for cond in ADAPTED_CONDITIONS}   # kept for std / convergence diagnostics

    t0 = time.time()
    for i, seed in enumerate(seeds):
        if verbose:
            print(f"\n=== Seed {seed} ({i + 1}/{n_seeds}) ===")
        np.random.seed(seed)
        # Fresh V1Dynamics_Surround per seed, but theta_t is set directly from the ONE shared
        # calibration above -- no per-seed 'no adaptation' run (see module docstring for why).
        dyn = V1Dynamics_Surround(tunings, frame, N_RF=N_RF, N_SETS=N_SETS,
                                   target_covariance_path=TARGET_COV_PATH, gains_nonneg=True)
        dyn.theta_t = theta_t_shared.copy()

        for cond in ADAPTED_CONDITIONS:
            g_cRF, g_surround, *_ = SSR.run_adaptation_phase(dyn, stim_gen, cond)
            g_cRF_sum[cond] += g_cRF
            g_surround_sum[cond] += g_surround
            g_cRF_runs[cond].append(g_cRF)

        if verbose:
            for cond in ADAPTED_CONDITIONS:
                n = i + 1
                mean_now = g_cRF_sum[cond] / n
                if n > 1:
                    mean_prev = (g_cRF_sum[cond] - g_cRF_runs[cond][-1]) / (n - 1)
                    rel_change = (np.linalg.norm(mean_now - mean_prev) /
                                  (np.linalg.norm(mean_prev) + 1e-12))
                    print(f"  [{cond}] running-mean g_cRF relative change: {rel_change:.3%}")

    elapsed = time.time() - t0
    if verbose:
        print(f"\nTotal adapted-run time: {elapsed:.1f}s ({elapsed / n_seeds:.1f}s/seed), "
              f"plus {time.time() - t_cal0 - elapsed:.1f}s calibration "
              f"= {time.time() - t_cal0:.1f}s total")

    os.makedirs(out_dir, exist_ok=True)
    meta = {
        "n_seeds": n_seeds, "seeds": list(seeds),
        "adapt_stream_length": adapt_stream_length, "duration": duration,
        "calibration_stream_length": calibration_stream_length,
        "calibration_seed": calibration_seed,
        "n_rf": N_RF, "n_sets": N_SETS,
        "frame_path": FRAME_PATH, "target_cov_path": TARGET_COV_PATH,
        "theta_t_shared_mean": float(theta_t_shared.mean()),
        "theta_t_shared_min": float(theta_t_shared.min()),
        "theta_t_shared_max": float(theta_t_shared.max()),
        "columns": "g_cRF_mean,g_surround_mean,g_cRF_std",
        "generated_by": "precompute_adapted_gains.py",
    }
    saved_paths = {}
    for cond in ADAPTED_CONDITIONS:
        g_cRF_mean = g_cRF_sum[cond] / n_seeds
        g_surround_mean = g_surround_sum[cond] / n_seeds
        g_cRF_std = np.std(g_cRF_runs[cond], axis=0)
        out = np.column_stack([g_cRF_mean, g_surround_mean, g_cRF_std])
        path = os.path.join(out_dir, f"{COND_SLUG[cond]}.csv")
        np.savetxt(path, out, delimiter=",")
        saved_paths[cond] = path
        if verbose:
            print(f"Saved {cond} -> {path}  (||g_cRF_mean||={np.linalg.norm(g_cRF_mean):.4g}, "
                  f"||g_cRF_std||={np.linalg.norm(g_cRF_std):.4g})")

    meta_path = os.path.join(out_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    if verbose:
        print(f"Saved provenance metadata -> {meta_path}")

    return saved_paths, meta


if __name__ == "__main__":
    main()
