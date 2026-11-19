'''
Tests for simulation.V1Dynamics_Surround against its equations (docs/equations.md, with the
u+ gating of the gain feedback):

    Eq. dy:  tau_y dy/dt = -y + beta z + W_yy sqrt(y+) / (1 + a+) - u+ * W_yv v
    Eq. du:  tau_u du/dt = -u + (sigma/2)^2 + N (y+ * (u+)^2)
    Eq. da:  tau_a da/dt = -a + u+ + a * u+
    Eq. dv:  tau_v dv/dt = -v + g * c,          c = W_yv^T sqrt(y+)
    Eq. dg:  tau_g dg/dt = (c - mu)^2 - theta_t
    Eq. dmu: tau_mu dmu/dt = -mu + c

with y+ = floor(y)^2, u+ = floor(u)^(1/2), a+ = floor(a). v, g, mu and c are local to each RF;
the model keeps one cRF copy and one representative surround copy (u, a likewise).

1. The right-hand side matches an independent per-neuron implementation at random states.
2. With gains frozen, the analytic fixed point (Eq. y_steady, with v* = g * c*) is a zero of the
   right-hand side.
3. With gains_nonneg, g >= 0 after every step; under constant input c -> mu, so g decays to 0.
'''

import contextlib
import io
import os
import sys

import numpy as np
from scipy.linalg import block_diag

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

import simulation as sim        # noqa: E402
from tunings import V1Tunings   # noqa: E402
from stimuli import StimulusGenerator   # noqa: E402

N_RF, N_SETS = 13, 6
FRAME = os.path.join(REPO, 'data/frames/N13_mercedes_K182_Frame.csv')
TCOV = os.path.join(REPO, 'data/target_covs/uniform_target_covariance_low_c.csv')


def make_model(**kw):
    with contextlib.redirect_stdout(io.StringIO()):
        dyn = sim.V1Dynamics_Surround(V1Tunings(N=N_RF), sim.Frame(FRAME), N_RF=N_RF,
                                      N_SETS=N_SETS, target_covariance_path=TCOV, **kw)
    dyn.calibrate_theta_t(C_zz_uniform=dyn.uniform_target_covariance, uniform_target=True,
                          verbose=False)
    return dyn


def reference_rhs(dyn, s, z):
    '''Per-neuron loop over the six equations; u, a expanded to every neuron of every set and
    v, g, mu, c handled per RF (sets 1..N_SETS-1 share the surround copy).'''
    p = dyn.unpack_state(s)
    W, K = dyn.frame.W, dyn.frame.K
    N_TOT = N_RF * N_SETS
    y = p['y']
    u = np.concatenate([p['u_cRF']] + [p['u_surround']] * (N_SETS - 1))
    a = np.concatenate([p['a_cRF']] + [p['a_surround']] * (N_SETS - 1))
    yp = np.maximum(y, 0) ** 2
    up = np.maximum(u, 0) ** 0.5
    ap = np.maximum(a, 0)
    rf = {'cRF': 0, 'surround': 1}
    c = {k: W.T @ np.sqrt(yp[i * N_RF:(i + 1) * N_RF]) for k, i in rf.items()}
    v_of_set = [p['v_cRF']] + [p['v_surround']] * (N_SETS - 1)

    dy = np.zeros(N_TOT)
    du = np.zeros(N_TOT)
    da = np.zeros(N_TOT)
    for n in range(N_TOT):
        set_idx, i = divmod(n, N_RF)
        rec = sum(dyn.W_yy[n, m] * np.sqrt(yp[m]) for m in range(N_TOT)) / (1 + ap[n])
        fb = up[n] * sum(W[i, k] * v_of_set[set_idx][k] for k in range(K))
        dy[n] = (-y[n] + dyn.beta * z[n] + rec - fb) / dyn.tau_y
        pool = sum(dyn.N_matrix[n, m] * yp[m] * up[m] ** 2 for m in range(N_TOT))
        du[n] = (-u[n] + (dyn.sigma / 2) ** 2 + pool) / dyn.tau_u
        da[n] = (-a[n] + up[n] + a[n] * up[n]) / dyn.tau_a
    out = dict(y=dy, u_cRF=du[:N_RF], u_surround=du[N_RF:2 * N_RF],
               a_cRF=da[:N_RF], a_surround=da[N_RF:2 * N_RF])
    for k in rf:
        out[f'v_{k}'] = (-p[f'v_{k}'] + p[f'g_{k}'] * c[k]) / dyn.tau_v
        out[f'g_{k}'] = ((c[k] - p[f'mu_{k}']) ** 2 - dyn.theta_t) / dyn.tau_g
        out[f'mu_{k}'] = (-p[f'mu_{k}'] + c[k]) / dyn.tau_mu
    return out


def test_rhs_matches_equations():
    dyn = make_model()
    rng = np.random.default_rng(0)
    sg = StimulusGenerator(N=N_RF, N_RF=N_RF, N_SETS=N_SETS)
    for trial in range(3):
        s = rng.uniform(-0.2, 1.0, dyn.state_size)   # includes negative entries: rectification
        z = sg.oriented_drive(rng.uniform(0, np.pi), rng.uniform(0.1, 1), 'adapt CRF only')
        # The surround copy convention is exact only if every surround set has the same state.
        got = dyn.unpack_state(dyn._derivatives(s, z, 0.0))
        ref = reference_rhs(dyn, s, z)
        for k in ('u_cRF', 'u_surround', 'a_cRF', 'a_surround', 'v_cRF', 'v_surround',
                  'g_cRF', 'g_surround', 'mu_cRF', 'mu_surround'):
            np.testing.assert_allclose(got[k], ref[k], rtol=1e-12, atol=1e-12, err_msg=k)
        # y: the model pools surround-set u from the representative copy; compare every neuron.
        np.testing.assert_allclose(got['y'], ref['y'], rtol=1e-12, atol=1e-12, err_msg='y')


def test_frozen_gain_fixed_point_is_analytic():
    '''Eq. y_steady with z' = (I + W D(g) W^T)^-1 z per RF; u* = (sigma^2 + sum z'^2)/4,
    a* = sqrt(u*)/(1 - sqrt(u*)), v* = g * W^T y*. Requires z' >= 0 (checked).'''
    dyn = make_model()
    dyn.tau_g = np.inf
    rng = np.random.default_rng(1)
    K, W = dyn.frame.K, dyn.frame.W
    sg = StimulusGenerator(N=N_RF, N_RF=N_RF, N_SETS=N_SETS)
    z = sg.oriented_drive(np.pi / 3, 1.0, 'adapt CRF and surround')
    g_c, g_s = rng.uniform(0, 0.05, K), rng.uniform(0, 0.05, K)
    M = block_diag(W @ np.diag(g_c) @ W.T, *[W @ np.diag(g_s) @ W.T] * (N_SETS - 1))
    zp = np.linalg.solve(np.eye(z.size) + M, z)
    assert zp.min() > 0, "z' has negative entries: fixed-point formula invalid"
    norm_sq = dyn.sigma ** 2 + np.sum(zp ** 2)
    y = zp / np.sqrt(norm_sq)
    u = norm_sq / 4
    a = np.sqrt(u) / (1 - np.sqrt(u))
    s = np.zeros(dyn.state_size)
    s[:z.size] = y
    s[dyn._off_u_cRF:dyn._off_u_cRF + 2 * N_RF] = u
    s[dyn._off_a_cRF:dyn._off_a_cRF + 2 * N_RF] = a
    s[dyn._off_g_cRF:dyn._off_g_cRF + K] = g_c
    s[dyn._off_g_surround:dyn._off_g_surround + K] = g_s
    s[dyn._off_v_cRF:dyn._off_v_cRF + K] = g_c * (W.T @ y[:N_RF])
    s[dyn._off_v_surround:dyn._off_v_surround + K] = g_s * (W.T @ y[N_RF:2 * N_RF])
    d = dyn.unpack_state(dyn._derivatives(s, z, 0.0))
    for k in ('y', 'u_cRF', 'u_surround', 'a_cRF', 'a_surround', 'v_cRF', 'v_surround', 'g_cRF'):
        assert np.max(np.abs(d[k])) < 1e-12, (k, np.max(np.abs(d[k])))


def test_gain_clamp_and_decay_under_constant_input():
    '''Constant input from the g = 0 fixed point with mu = c* and small positive gains g0:
    (c - mu)^2 stays ~0, so each g decreases at rate theta_t/tau_g, reaches 0 and stays there
    (gains_nonneg clamp). g0_k = theta_k * 100 ms / tau_g, so every gain hits 0 near 100 ms.'''
    dyn = make_model(gains_nonneg=True)
    dyn.tau_g, dyn.tau_mu = 5.0, 2.0
    K, W = dyn.frame.K, dyn.frame.W
    sg = StimulusGenerator(N=N_RF, N_RF=N_RF, N_SETS=N_SETS)
    z = sg.oriented_drive(np.pi / 2, 1.0, 'adapt CRF and surround')
    norm_sq = dyn.sigma ** 2 + np.sum(z ** 2)
    y = z / np.sqrt(norm_sq)
    u = norm_sq / 4
    g0 = dyn.theta_t * 100.0 / dyn.tau_g
    s = np.zeros(dyn.state_size)
    s[:z.size] = y
    s[dyn._off_u_cRF:dyn._off_u_cRF + 2 * N_RF] = u
    s[dyn._off_a_cRF:dyn._off_a_cRF + 2 * N_RF] = np.sqrt(u) / (1 - np.sqrt(u))
    for rf, sl in (('cRF', slice(0, N_RF)), ('surround', slice(N_RF, 2 * N_RF))):
        off = getattr(dyn, f'_off_g_{rf}')
        s[off:off + K] = g0
        off = getattr(dyn, f'_off_v_{rf}')
        s[off:off + K] = g0 * (W.T @ y[sl])
        off = getattr(dyn, f'_off_mu_{rf}')
        s[off:off + K] = W.T @ y[sl]
    n = 2000   # 200 ms
    with contextlib.redirect_stdout(io.StringIO()):
        out = dyn.run_simulation(np.tile(z[:, None], n), initial_state=s)
    g_c, g_s, mu_c = out[5], out[6], out[9]
    t = np.arange(1, n + 1) * dyn.dt
    assert mu_c.shape == (K, n)
    assert g_c.min() >= 0 and g_s.min() >= 0
    # Early on, g follows g0 - theta_t t / tau_g (to within 1 % of g0).
    i = np.argmin(np.abs(t - 50.0))
    np.testing.assert_allclose(g_c[:, i], g0 - dyn.theta_t * t[i] / dyn.tau_g, atol=0.01 * g0.max())
    assert np.all(g_c[:, -1] == 0) and np.all(g_s[:, -1] == 0)
