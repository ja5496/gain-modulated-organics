'''
sanity_checks.py

Ground-rule checks run before any experiment (results printed and saved to data/sanity_checks.npz):

1. Regression: with the codebase options, simulation_variants reproduces
   simulation_whiten.V1Dynamics_Surround on an adaptation run with the surround delay.
2. Fixed point: for all-ones normalization weights and frozen gains, the simulated steady state
   of every gating x gain-placement combination matches the analytic fixed point
   (Eqs. 10-11 of the manuscript), for zero and for adapted gains, small and large stimuli.
3. Time step: halving dt does not change frozen-gain trajectories or an adaptation run.

Run:  python sanity_checks.py        (about 5 minutes on 4 cores)
'''

import sys
from dataclasses import replace
from multiprocessing import Pool

import numpy as np

import common as C
import simulation_variants as sv

COMBOS = {'gating a, gain in y eq. (codebase)': sv.CODEBASE,
          'gating u, gain in y eq.':            replace(sv.CODEBASE, gating='u'),
          'gating a, gain in v eq.':            replace(sv.CODEBASE, gain_placement='interneuron'),
          'gating u, gain in v eq. (Eqs. 1-6)': sv.NEW}
ADAPT_PRES = 130            # 130 x 32 ms = 4.16 s adaptation to obtain non-zero test gains
SEED = 11


def regression():
    sys.path.insert(0, C.REPO_ROOT + '/analysis')
    import simulation_whiten as sw
    from tunings_whiten import V1Tunings
    from response_tools import settled_surround_msg
    sw.tqdm = lambda x, **k: x
    n_pres, steps = 40, 320
    z, _ = C.ensemble_presentations(n_pres, True, SEED, C.LARGE)
    with C.quiet():
        frame = sw.Frame(C.FRAME_PATH)
        dyn = sw.V1Dynamics_Surround(V1Tunings(N=C.N_RF), frame, dt=C.DT, N_RF=C.N_RF, N_SETS=C.N_SETS,
                                     target_covariance_path=C.TARGET_COV_PATH, gains_nonneg=True)
        dyn.calibrate_theta_t(C_zz_uniform=dyn.uniform_target_covariance, uniform_target=True)
        msg = settled_surround_msg(dyn, C.blank()[0], int(C.BASELINE_SETTLE / C.DT))
        dyn.set_surround_delay(C.SURROUND_DELAY)
        out = dyn.run_simulation(np.repeat(z.T, steps, axis=1), surround_msg_baseline=msg)
    res = C.adapt(C.make_model(sv.CODEBASE), z)
    old_y = out[0][:, steps - 1::steps].T
    err_y = np.abs(np.maximum(res['y_end'][0], 0) - old_y).max()
    err_g = np.abs(res['g_end'][0, :, :dyn.frame.K] - out[5][:, steps - 1::steps].T).max()
    print(f'1. Regression vs V1Dynamics_Surround ({n_pres} presentations, delay {C.SURROUND_DELAY:g} ms): '
          f'max |dy| = {err_y:.1e}, max |dg| = {err_g:.1e}')
    return dict(regression_err_y=err_y, regression_err_g=err_g)


def adapted_gains(name):
    '''Gains after a short adaptation to the biased ensemble (small stimuli), per variant.'''
    z, _ = C.ensemble_presentations(ADAPT_PRES, True, SEED, C.SMALL)
    res = C.adapt(C.make_model(COMBOS[name]), z)
    return name, res['g_end'][0, -1]


def fixed_point(gains):
    print('2. Simulated steady state vs analytic fixed point (frozen gains, all-ones N, 600 ms settle)')
    angles = np.linspace(0, np.pi, C.N_RF, endpoint=False)
    out = {}
    worst = 0.0
    for name, opt in COMBOS.items():
        m = C.make_model(opt)
        K = m.K
        for g_label, g in (('g = 0', np.zeros(2 * K)), ('adapted g', gains[name])):
            for loc_label, loc in (('small', C.SMALL), ('large', C.LARGE)):
                z = C.oriented(angles, C.PROBE_CONTRAST, loc)
                fp = m.analytic_fixed_point(z, g[:K], g[K:])
                res = m.settle(z, g[:K], g[K:], settle_ms=600.0)
                st = m.unpack(res['state'])
                rel = lambda x, ref: np.abs(x - ref).max() / np.abs(ref).max()
                e_y, e_u, e_a = rel(st['y'], fp['y']), rel(st['u'], fp['u']), rel(st['a'], fp['a'])
                zmin = fp['z_prime'].min()
                # Fast-subsystem residual at the simulated state (should vanish at a fixed point)
                d = m.derivatives(res['state'], z, None, False, {})
                resid = np.abs(d[:, :3 * m.N_TOT]).max()
                worst = max(worst, e_y, e_u, e_a)
                print(f'   {name:36s} {g_label:10s} {loc_label:5s}: rel. err y {e_y:.1e}, u {e_u:.1e}, '
                      f'a {e_a:.1e} | max u+ {np.sqrt(st["u"].max()):.3f} | min z\' {zmin:+.1e} | '
                      f'|dy,du,da/dt| {resid:.1e}')
                out[f'fp_{name}_{g_label}_{loc_label}'] = np.array([e_y, e_u, e_a, np.sqrt(st['u'].max()), zmin])
    print(f'   worst relative error: {worst:.1e}')
    return out


def dt_trajectory(gains):
    '''Frozen-gain response to a stimulus switch at dt = 0.1, 0.05, 0.025 ms.'''
    print('3a. Time step, frozen gains: orientation switch, y(t) sampled every 0.1 ms over 100 ms')
    out = {}
    pref = np.linspace(0, np.pi, C.N_RF, endpoint=False)
    z0, z1 = C.oriented(pref[6], location=C.SMALL), C.oriented(pref[10], location=C.SMALL)
    for name in ('gating a, gain in y eq. (codebase)', 'gating u, gain in v eq. (Eqs. 1-6)'):
        g = gains[name]
        trajs = {}
        for dt in (0.1, 0.05, 0.025):
            m = C.make_model(COMBOS[name], dt=dt)
            start = m.settle(z0, g[:m.K], g[m.K:], settle_ms=600.0)['state']
            tr = m.settle(z1, g[:m.K], g[m.K:], settle_ms=100.0, state0=start, record_traces=True)['traces']['y'][0]
            trajs[dt] = tr[int(round(0.1 / dt)) - 1::int(round(0.1 / dt))]
        scale = np.abs(trajs[0.025]).max()
        e1 = np.abs(trajs[0.1] - trajs[0.025]).max() / scale
        e2 = np.abs(trajs[0.05] - trajs[0.025]).max() / scale
        print(f'   {name:36s}: max |y_dt - y_0.025| / max|y| = {e1:.1e} (dt 0.1), {e2:.1e} (dt 0.05)')
        out[f'dt_traj_{name}'] = np.array([e1, e2])
    return out


def dt_adaptation(args):
    name, dt = args
    z = np.stack([C.ensemble_presentations(ADAPT_PRES, b, SEED + b, C.SMALL)[0] for b in (False, True)])
    res = C.adapt(C.make_model(COMBOS[name], dt=dt), z)
    return name, dt, res['g_end'][:, -1], res['y_end']


if __name__ == '__main__':
    saved = regression()
    with Pool(4) as pool:
        gains = dict(pool.map(adapted_gains, list(COMBOS)))
        saved.update(fixed_point(gains))
        saved.update(dt_trajectory(gains))
        print(f'3b. Time step, adaptation run ({ADAPT_PRES} presentations x {C.PRES_MS:g} ms, uniform and biased, '
              f'delay {C.SURROUND_DELAY:g} ms): dt = 0.1 vs 0.05 ms')
        names = ('gating a, gain in y eq. (codebase)', 'gating u, gain in v eq. (Eqs. 1-6)')
        runs = pool.map(dt_adaptation, [(n, dt) for n in names for dt in (0.1, 0.05)])
    by = {(n, dt): (g, y) for n, dt, g, y in runs}
    for n in names:
        g1, y1 = by[(n, 0.1)]
        g2, y2 = by[(n, 0.05)]
        eg = np.abs(g1 - g2).max() / np.abs(g2).max()
        ey = np.abs(y1 - y2).max() / np.abs(y2).max()
        print(f'   {n:36s}: rel. change final gains {eg:.1e}, responses at presentation ends {ey:.1e}')
        saved[f'dt_adapt_{n}'] = np.array([eg, ey])
    C.save_data('sanity_checks', **saved)
