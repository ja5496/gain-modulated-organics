'''
common.py

Shared set-up for the supplementary investigation: paths, the model configuration used by
analysis/adapt_ensembles.py (frame, target covariance, contrasts, surround delay), stimulus
helpers with explicit random seeds, the adaptation protocol and the frozen-gain readout.

All times are in ms. Everything here calls simulation_variants.AdaptiveORGaNICs; the existing
model code (simulation_whiten.py) is not modified.
'''

import contextlib
import io
import os
import subprocess
import sys
from dataclasses import replace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, REPO_ROOT)

import simulation_variants as sv                      # noqa: E402
from stimuli_whiten import StimulusGenerator          # noqa: E402

DATA_DIR = os.path.join(HERE, 'data')
FIG_DIR = os.path.join(HERE, 'figures')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# ---- Configuration of analysis/adapt_ensembles.py (the code is the reference) ----
N_RF, N_SETS = 13, 6
N_TOT = N_RF * N_SETS
FRAME_PATH = os.path.join(REPO_ROOT, 'data/frames/N13_mercedes_K182_Frame.csv')
TARGET_COV_PATH = os.path.join(REPO_ROOT, 'data/target_covs/uniform_target_covariance_low_c.csv')
TARGET_SCALE = 0.15            # V1Dynamics_Surround multiplies the stored target covariance by 0.15
DT = 0.1                       # ms
SURROUND_DELAY = 40.0          # ms
BASELINE_SETTLE = 500.0        # ms under the blank screen before adapting
ENSEMBLE_CONTRAST = 0.8
PROBE_CONTRAST = 0.8
TUNING_WIDTH = 0.75            # rad, width of the Gaussian input profile
PRES_MS = 32.0                 # presentation duration (Benucci et al.; adapt_ensembles.py uses 30)
ADAPTOR_IDX = N_RF // 2        # index of the adaptor orientation (as in stimuli_whiten.py)

SMALL = 'adapt CRF only'           # stimulus covers the cRF only
LARGE = 'adapt CRF and surround'   # stimulus covers cRF and surround

# The equation variants compared in Supplementary Note 1
VARIANTS = {
    'old':         sv.CODEBASE,
    'new':         sv.NEW,
    'gating':      replace(sv.CODEBASE, gating='u'),
    'placement':   replace(sv.CODEBASE, gain_placement='interneuron'),
    'variance':    replace(sv.CODEBASE, variance_signal='instant'),
    'rectify_c':   replace(sv.CODEBASE, rectify_c=True),
}


def load_frame():
    return np.loadtxt(FRAME_PATH, delimiter=',')


def target_covariance():
    return TARGET_SCALE * np.loadtxt(TARGET_COV_PATH, delimiter=',')


def make_model(options=sv.CODEBASE, dt=DT, **kwargs):
    W = load_frame()
    theta_t = kwargs.pop('theta_t', None)
    if theta_t is None:
        theta_t = sv.theta_t_from_target(W, target_covariance())
    return sv.AdaptiveORGaNICs(W, theta_t, N_RF=N_RF, N_SETS=N_SETS, options=options, dt=dt, **kwargs)


def stimulus_generator(contrast=ENSEMBLE_CONTRAST, n_pres=N_RF):
    return StimulusGenerator(N_RF=N_RF, N_SETS=N_SETS, num_angles=N_RF, stream_length=n_pres,
                             tuning_width=TUNING_WIDTH, contrast=contrast)


def pref_deg():
    '''Preferred orientation of each cRF neuron (deg).'''
    return np.degrees(np.linspace(0, np.pi, N_RF, endpoint=False))


def rel_to_adaptor(deg):
    adaptor = pref_deg()[ADAPTOR_IDX]
    return (np.asarray(deg) - adaptor + 90.0) % 180.0 - 90.0


def ensemble_presentations(n_pres, biased, seed, location=SMALL, contrast=ENSEMBLE_CONTRAST, **kwargs):
    '''
    One input drive per presentation, drawn by StimulusGenerator.generate_surround_ensembles
    (uniform or biased ensemble; kwargs e.g. von_mises=True) with a fixed seed.
    Returns (z (n_pres, N_TOT), stimulus orientation of each presentation in rad).
    '''
    gen = stimulus_generator(contrast, n_pres)
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        z, centers = gen.generate_surround_ensembles(location, biased=biased, duration=1,
                                                     return_angles=True, **kwargs)
    finally:
        np.random.set_state(state)
    return z.T.copy(), centers


def oriented(theta, contrast=PROBE_CONTRAST, location=LARGE):
    '''Input drive of gratings at orientations theta (rad): (n, N_TOT).'''
    gen = stimulus_generator(1.0)
    return np.stack([gen.oriented_drive(t, contrast, location) for t in np.atleast_1d(theta)])


def blank():
    '''Blank screen: the fixed baseline drive everywhere, (1, N_TOT).'''
    return oriented(0.0, 0.0, 'no adaptation')


def blank_pool(model):
    '''Each neuron's settled contribution to the normalization pool under the blank screen with
    zero gains: holds the delayed surround signal until the first real delayed sample exists
    (response_tools.settled_surround_msg).'''
    quiet_model = model.with_options()                 # same circuit without noise, one row
    quiet_model.noise = sv.NoiseOptions()
    rows = model.N_matrix.shape[0] if model.N_matrix.ndim == 3 else 1   # one row per normalization matrix
    res = quiet_model.settle(np.repeat(blank(), rows, axis=0), 0.0, 0.0, settle_ms=BASELINE_SETTLE)
    return quiet_model.q_of(res['state'])


def adapt(model, z_pres, pres_ms=PRES_MS, delay=SURROUND_DELAY, rng=None, **kwargs):
    '''Adapts from a zero state to the presentations z_pres ((B,) n_pres, N_TOT), each held for
    pres_ms, with the surround -> cRF delay. Returns simulate()'s dict.'''
    steps = int(round(pres_ms / model.dt))
    assert np.isclose(steps * model.dt, pres_ms)
    q0 = blank_pool(model) if delay > 0 else None
    model.set_surround_delay(delay)
    try:
        return model.simulate(z_pres, steps, q_baseline=q0, rng=rng, **kwargs)
    finally:
        model.set_surround_delay(0.0)


def tuning_curves(model, g_cRF, g_surround, angles, contrast=PROBE_CONTRAST, location=LARGE,
                  method='analytic', settle_ms=600.0):
    '''
    Frozen-gain steady-state tuning curves of the cRF neurons: firing rates (rectified y squared)
    to gratings at `angles` (rad). Returns (N_RF, n_angles).
    method='analytic': closed-form fixed point (all-ones normalization weights only).
    method='dynamics': integrates the full fast dynamics to steady state.
    '''
    z = oriented(angles, contrast, location)
    if method == 'analytic':
        y = model.analytic_fixed_point(z, g_cRF, g_surround)['y']
    else:
        y = model.settle(z, g_cRF, g_surround, settle_ms=settle_ms)['y_end'][:, 0]
    return (np.maximum(y[:, :N_RF], 0.0) ** 2).T


def git_commit():
    try:
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], cwd=REPO_ROOT,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return 'unknown'


def save_data(name, **arrays):
    '''Saves a figure's raw output to data/<name>.npz with the git commit.'''
    path = os.path.join(DATA_DIR, name + '.npz')
    np.savez_compressed(path, git_commit=git_commit(), **arrays)
    return path


def load_data(name):
    path = os.path.join(DATA_DIR, name + '.npz')
    return np.load(path, allow_pickle=True) if os.path.exists(path) else None


@contextlib.contextmanager
def quiet():
    with contextlib.redirect_stdout(io.StringIO()):
        yield


# ---------------------------------------------------------------- Supplementary Notes 2-4
ADAPT_PRES = 125               # 125 presentations x 32 ms = 4.0 s of adaptation (Notes 3 and 4)

# Target suppression profile (Notes 3 and 4). The original data were not available, so these are
# NOMINAL values that encode the reported pattern: slight untuned suppression (ratio TARGET_FAR
# far from the adaptor) and strong suppression at the adaptor (ratio TARGET_FAR - TARGET_DEPTH),
# with a Gaussian dip of s.d. TARGET_WIDTH degrees.
TARGET_FAR, TARGET_DEPTH, TARGET_WIDTH = 0.9, 0.3, 20.0


def target_profile(rel_deg):
    return TARGET_FAR - TARGET_DEPTH * np.exp(-0.5 * (np.asarray(rel_deg) / TARGET_WIDTH) ** 2)


def profile_distance(profile, rel_deg=None):
    """Similarity metric: r.m.s. difference between a suppression profile (one value per cRF
    neuron) and the target profile. 0 = identical; smaller is closer to experiment."""
    rel = rel_to_adaptor(pref_deg()) if rel_deg is None else rel_deg
    return float(np.sqrt(np.mean((np.asarray(profile) - target_profile(rel)) ** 2)))


def profile_features(profile):
    """(ratio far from the adaptor, |rel. preference| >= 55 deg; ratio at the adaptor; depth = far - adaptor)."""
    rel = rel_to_adaptor(pref_deg())
    far = float(np.mean(np.asarray(profile)[np.abs(rel) >= 55]))
    at = float(np.asarray(profile)[ADAPTOR_IDX])
    return far, at, far - at


def uniform_and_biased(n_pres, n_seeds, location=SMALL, contrast=ENSEMBLE_CONTRAST):
    """Presentations of the uniform (seeds 100..) and biased (seeds 200..) ensembles:
    (2 n_seeds, n_pres, N_TOT), uniform first; and the stimulus orientations (rad)."""
    out = [ensemble_presentations(n_pres, b, s0 + i, location, contrast)
           for b, s0 in ((False, 100), (True, 200)) for i in range(n_seeds)]
    return np.stack([o[0] for o in out]), np.stack([o[1] for o in out])


def time_averaged_rates(model, z, g, settle_ms=150.0, average_ms=150.0, sample_ms=2.0, rng=None):
    """Frozen-gain firing rates max(y,0)^2 of all neurons, averaged over time after settling
    (for circuits with noise). z: (B, N_TOT); g: (B, 2K). Returns (B, N_TOT)."""
    steps = int(round(sample_ms / model.dt))
    n_settle, n_avg = int(round(settle_ms / sample_ms)), int(round(average_ms / sample_ms))
    z = np.atleast_2d(z)
    state = model.zero_state(z.shape[0])
    state[:, model.sl['g']] = g
    probe = model.with_options()
    probe.set_surround_delay(0.0)
    res = probe.simulate(np.repeat(z[:, None, :], n_settle + n_avg, axis=1), steps, state0=state, adapt=False, rng=rng)
    return (np.maximum(res['y_end'][:, n_settle:], 0.0) ** 2).mean(axis=1)


def instream_tuning(y_end, centers, window=None):
    """In-stream tuning curves: firing rate at the last step of each presentation, averaged over
    the presentations of each orientation within window = (first, last) presentation index
    (default: second half). y_end (n_pres, N); centers (n_pres,) rad. Returns (N, N_RF)."""
    n_pres = y_end.shape[0]
    lo, hi = window if window is not None else (n_pres // 2, n_pres)
    idx = np.round(np.asarray(centers[lo:hi]) * N_RF / np.pi).astype(int) % N_RF
    rate = np.maximum(y_end[lo:hi], 0.0) ** 2
    return np.stack([rate[idx == k].mean(axis=0) if np.any(idx == k) else np.full(rate.shape[1], np.nan)
                     for k in range(N_RF)], axis=1)


def normalization_weights(sigma_s, sigma_theta_deg):
    """
    Gaussian normalization weights N_ij ~ exp(-d_ij^2 / 2 sigma_s^2) exp(-dtheta_ij^2 / 2 sigma_theta^2).
    Geometry: the cRF at the origin and the 5 surround RFs evenly spaced on a ring one cRF
    diameter away (abutting the cRF); d_ij is the distance between RF centres in cRF diameters,
    dtheta_ij the circular difference between preferred orientations (period 180 deg).
    np.inf for either width gives a flat profile along that dimension.
    Rescaling: N is multiplied by one number so that the cRF neuron tuned to an adapting (small,
    contrast 0.8) grating receives the same pooled drive, sum_j N_ij z_j^2, as with all-ones
    weights. Its unadapted response to that grating is then identical to the all-ones case.
    """
    ang = 2 * np.pi * np.arange(N_SETS - 1) / (N_SETS - 1)
    centres = np.vstack([[0.0, 0.0], np.c_[np.cos(ang), np.sin(ang)]])
    d = np.linalg.norm(centres[:, None] - centres[None], axis=2)
    pref = pref_deg()
    dth = (pref[:, None] - pref[None] + 90.0) % 180.0 - 90.0
    w_space = np.ones_like(d) if np.isinf(sigma_s) else np.exp(-d ** 2 / (2 * sigma_s ** 2))
    w_ori = np.ones_like(dth) if np.isinf(sigma_theta_deg) else np.exp(-dth ** 2 / (2 * sigma_theta_deg ** 2))
    N = np.kron(w_space, w_ori)
    z = oriented(np.radians(pref[ADAPTOR_IDX]), ENSEMBLE_CONTRAST, SMALL)[0]
    return N * np.sum(z ** 2) / (N[ADAPTOR_IDX] @ z ** 2)


def unadapted_response(z, N, sigma=0.15):
    """Closed-form unadapted (zero-gain) steady state for any normalization weights:
    y_i = z_i / sqrt(sigma^2 + sum_j N_ij z_j^2). z: (n, N_TOT)."""
    return z / np.sqrt(sigma ** 2 + (z ** 2) @ N.T)
