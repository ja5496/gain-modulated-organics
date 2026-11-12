'''
extra_checks_note1.py  ->  data/Note1_long_run.npz, data/Note1_tau_mu.npz (no figure)

Two follow-up checks behind statements in the summary of Supplementary Note 1. Same protocol as
figS3_tuning_old_vs_new.py (small adaptors, 32 ms presentations, surround delay 40 ms, frozen-gain
readout with large test gratings, analytic fixed point).

1. long_run: do the gains converge? New equations, uniform (seeds 100, 101) and biased (seeds
   200, 201) ensembles, 5000 presentations (160 s), gains recorded every 25 presentations.
2. tau_mu: is the suppression at the adaptor produced by the start-up of the running mean mu
   (mu starts at 0, tau_mu = 2000 ms)? Old and new equations with tau_mu = 2000 ms and 200 ms,
   4 repeats per ensemble (seeds 100-103, 200-203), 260 presentations (8.32 s).

Run:  python extra_checks_note1.py [--long] [--tau-mu]     (--long: about 12 minutes)
'''

import sys
from multiprocessing import Pool

import numpy as np

import common as C
import simulation_variants as sv

K = 182
REL_ANGLES = np.arange(-90.0, 90.0 + 1e-9, 2.0)


def peaks(g):
    m = C.make_model(sv.CODEBASE)
    angles = np.radians(C.pref_deg()[C.ADAPTOR_IDX] + REL_ANGLES)
    return C.tuning_curves(m, g[:K], g[K:], angles).max(axis=1)


def ensembles(n_pres, n_seeds):
    return np.stack([C.ensemble_presentations(n_pres, False, 100 + i, C.SMALL)[0] for i in range(n_seeds)]
                    + [C.ensemble_presentations(n_pres, True, 200 + i, C.SMALL)[0] for i in range(n_seeds)])


def long_run(n_pres=5000, rec=25, n_seeds=2):
    z = ensembles(n_pres, n_seeds)
    res = C.adapt(C.make_model(sv.NEW), z, record_g_every_pres=rec)
    W = C.load_frame()
    theta = sv.theta_t_from_target(W, C.target_covariance())
    c_end = np.maximum(res['y_end'][:, -520:, :C.N_RF], 0) @ W          # c at the ends of the last 520 presentations
    return dict(t=(np.arange(res['g_end'].shape[1]) + 1) * rec * C.PRES_MS / 1000.0,
                g=res['g_end'].astype(np.float32), var_over_theta=c_end.var(axis=1) / theta, n_seeds=n_seeds)


def report_long(d):
    n = int(d['n_seeds'])
    g, t = d['g'].astype(float), d['t']
    p0 = peaks(np.zeros(2 * K))
    order = np.argsort(C.rel_to_adaptor(C.pref_deg()))
    print('1. Long run, new equations: t (s) | mean cRF gain uniform, biased | growth (%/s) | peak biased/uniform '
          '(by preferred orientation rel. adaptor) | uniform-adapted / unadapted peak')
    prev = None
    for ck in (9, 24, 49, 99, 149, 199):
        gu, gb = g[:n, ck, :K].mean(), g[n:, ck, :K].mean()
        pu = np.mean([peaks(x) for x in g[:n, ck]], axis=0)
        pb = np.mean([peaks(x) for x in g[n:, ck]], axis=0)
        rate = '' if prev is None else f'{100 * (gb - prev[1]) / gb / (t[ck] - prev[0]):.2f}'
        print(f'   {t[ck]:6.1f} | {gu:.3f} {gb:.3f} | {rate:>5s} | {np.round((pb / pu)[order], 2)} | {np.mean(pu / p0):.4f}')
        prev = (t[ck], gb)
    v = d['var_over_theta']
    print(f'   variance of c at presentation ends / theta_t over the last 16.6 s: median {np.median(v[:n]):.2f} (uniform), '
          f'{np.median(v[n:]):.2f} (biased); gains > 1e-3: {(g[:, -1, :K] > 1e-3).sum(axis=1)} of {K}')


def tau_mu_run(args):
    name, tau_mu, n_seeds = args
    res = C.adapt(C.make_model(C.VARIANTS[name], params=sv.ModelParams(tau_mu=tau_mu)), ensembles(260, n_seeds),
                  record_g_every_pres=20)
    return res['g_end'].astype(np.float32)


def tau_mu(n_seeds=4):
    cases = [('old', 2000.0), ('old', 200.0), ('new', 2000.0), ('new', 200.0)]
    with Pool(4) as pool:
        gains = pool.map(tau_mu_run, [(n, t, n_seeds) for n, t in cases])
    return dict(cases=np.array([f'{n}, tau_mu = {t:g} ms' for n, t in cases]), g=np.stack(gains),
                t=(np.arange(13) + 1) * 20 * C.PRES_MS / 1000.0, n_seeds=n_seeds)


def report_tau_mu(d):
    n = int(d['n_seeds'])
    order = np.argsort(C.rel_to_adaptor(C.pref_deg()))
    print('2. Running-mean start-up: peak biased/uniform by preferred orientation rel. adaptor')
    for case, g in zip(d['cases'], d['g'].astype(float)):
        for ck in (2, 12):
            pk = np.array([peaks(x) for x in g[:, ck]])
            prof = (pk[n:] / pk[:n].mean(axis=0)).mean(axis=0)
            print(f'   {case:24s} t = {d["t"][ck]:4.1f} s, mean gain {g[n:, ck, :K].mean():.3f}: {np.round(prof[order], 3)}')


if __name__ == '__main__':
    if '--long' in sys.argv:
        C.save_data('Note1_long_run', **long_run())
    if '--tau-mu' in sys.argv:
        C.save_data('Note1_tau_mu', **tau_mu())
    for name, rep in (('Note1_long_run', report_long), ('Note1_tau_mu', report_tau_mu)):
        d = C.load_data(name)
        if d is not None:
            rep(d)
