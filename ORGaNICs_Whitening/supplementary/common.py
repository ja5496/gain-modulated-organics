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
    res = model.settle(blank(), 0.0, 0.0, settle_ms=BASELINE_SETTLE)
    return model.q_of(res['state'])


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
