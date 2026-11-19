'''
noise_sweep.py

Supplementary Note 4 (exploratory): how noise changes the effects of the three adaptation rules of
alternative_rules.py under the single-adaptor protocol (adapt_single_stimulus.py, Fig. 1).

Sweep: noise type (white, Poisson-like) x location (gain pathway, response, input) x level.
  white    s + sigma xi,          sigma = rho * s_ref
  poisson  s + sqrt(F s+) xi,     F = rho^2 * s_ref
so that at the reference signal s_ref the noise s.d. is rho * s_ref for both types. s_ref is the
unadapted signal of the adaptor-tuned cRF neuron under the cRF adaptor (contrast 1): its input
drive z for location 'input', its fixed-point response r = floor(y) for 'gain' and 'response'.
Noise is resampled every NOISE_BIN_MS (see alternative_rules.GainRuleDynamics). Probe responses
are noise-free fixed points with the adapted gains frozen.

Metrics (cRF population mean firing rate, relative to unadapted):
  response_gain   rate at contrast 1 after the cRF adaptor
  facilitation    rate at contrast 1 after the surround adaptor
  c50_ratio       c50 after / before the cRF + surround adaptor, c50 = contrast at which the rate
                  reaches half the unadapted rate at contrast 1
  shift_cRF, shift_surround   flank-neuron tuning shifts (deg) after the cRF / surround adaptor

Run from ORGaNICs_Whitening/:
    python supplementary/noise_sweep.py             # sweep (3 worker processes), save, plot
    python supplementary/noise_sweep.py --plot      # replot from the saved results
    python supplementary/noise_sweep.py --smoke     # 2 short jobs, not saved
'''

import argparse
import datetime
import itertools
import json
import os
import subprocess
import sys

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, REPO_ROOT, os.path.join(REPO_ROOT, 'analysis')]

import alternative_rules as A                                       # noqa: E402
import adapt_single_stimulus as ASS                                 # noqa: E402
import plotstyle as ps                                              # noqa: E402
import matplotlib.pyplot as plt                                     # noqa: E402
from response_tools import (get_response_offline, offline_gain_operator,  # noqa: E402
                            curve_peak_deg, to_steps)
from stimuli import StimulusGenerator                               # noqa: E402

KINDS = ['white', 'poisson']
LOCS = ['gain', 'response', 'input']
LOC_LABEL = {'gain': 'Gain pathway', 'response': 'Response', 'input': 'Input drive'}
RHOS = [0.1, 0.25, 0.5, 1.0]
NOISE_BIN_MS = 1.0
N_WORKERS = 3
SEED = 0
FINE_C = np.logspace(-2, 0, 200)
METRICS = ['response_gain', 'c50_ratio', 'facilitation', 'shift_cRF', 'shift_surround']
METRIC_LABEL = {'response_gain': 'cRF adapted:\nrate at $c = 1$',
                'c50_ratio': 'cRF + surround:\n$c_{50}$ ratio',
                'facilitation': 'Surround adapted:\nrate at $c = 1$',
                'shift_cRF': 'cRF adapted:\nflank shift (deg)',
                'shift_surround': 'Surround adapted:\nflank shift (deg)'}

RESULTS = os.path.join(HERE, 'results', 'noise_sweep.npz')
FIG_BASE = os.path.join(HERE, 'figures', 'fig_noise_sweep')


def stim_gen():
    return StimulusGenerator(N_RF=ASS.N_RF, N_SETS=ASS.N_SETS, num_angles=ASS.N_RF,
                             tuning_width=ASS.TUNING_WIDTH)


def reference_signals():
    sg = stim_gen()
    dyn = A.build('variance', sg)
    z = sg.oriented_drive(ASS.ADAPTOR_THETA, ASS.ADAPTOR_CONTRAST, 'adapt CRF only')
    y = get_response_offline(dyn, z, np.zeros((z.size, z.size)), rectify=False)
    return dict(input=float(z[:ASS.N_RF].max()), gain=float(y[:ASS.N_RF].max()),
                response=float(y[:ASS.N_RF].max()))


def c50(curve, half):
    i = np.argmax(curve >= half)
    if curve[i] < half:
        return np.nan
    if i == 0:
        return FINE_C[0]
    x0, x1 = np.log(FINE_C[i - 1]), np.log(FINE_C[i])
    return float(np.exp(x0 + (half - curve[i - 1]) / (curve[i] - curve[i - 1]) * (x1 - x0)))


def job(args):
    rule, kind, loc, rho, s_ref, adapt_steps = args
    np.random.seed(SEED)
    if kind == 'none':
        A.NOISE = None
    else:
        amp = rho * s_ref if kind == 'white' else rho ** 2 * s_ref
        A.NOISE = dict(kind=kind, loc=loc, amp=amp, bin_steps=to_steps(NOISE_BIN_MS, ASS.DT))
    sg = stim_gen()
    out = A.run_rule(rule, sg, adapt_steps)
    dyn = A.build(rule, sg)   # for the probes (noise-free fixed point)
    g = {c: (out[f'{rule}/g_cRF_final/{c.replace(" ", "_")}'],
             out[f'{rule}/g_surround_final/{c.replace(" ", "_")}']) for c in ASS.CONDITIONS}
    P = np.stack([sg.oriented_drive(ASS.ADAPTOR_THETA, c, 'adapt CRF and surround') for c in FINE_C], 1)
    rate = {c: get_response_offline(dyn, P, offline_gain_operator(dyn, *g[c]))[:ASS.N_RF].mean(0)
            for c in ASS.CONDITIONS}
    ref = rate['no adaptation']
    ang = np.degrees(np.linspace(0, np.pi, ASS.N_PROBES, endpoint=False))
    ctrl = curve_peak_deg(out[f'{rule}/flank/no_adaptation'], ang)
    shift = lambda c: ((curve_peak_deg(out[f'{rule}/flank/{c.replace(" ", "_")}'], ang) - ctrl + 90) % 180) - 90
    return dict(rule=rule, kind=kind, loc=loc, rho=rho,
                response_gain=rate['adapt CRF only'][-1] / ref[-1],
                facilitation=rate['adapt surround only'][-1] / ref[-1],
                c50_ratio=c50(rate['adapt CRF and surround'], ref[-1] / 2) / c50(ref, ref[-1] / 2),
                shift_cRF=shift('adapt CRF only'), shift_surround=shift('adapt surround only'),
                g_max_cRF=float(g['adapt CRF only'][0].max()))


def run(smoke=False):
    from multiprocessing import Pool
    s_ref = reference_signals()
    adapt_steps = to_steps(200.0 if smoke else 4200.0, ASS.DT)
    jobs = [(r, 'none', 'none', 0.0, 0.0, adapt_steps) for r in A.RULES]
    jobs += [(r, k, l, rho, s_ref[l], adapt_steps)
             for r, k, l, rho in itertools.product(A.RULES, KINDS, LOCS, RHOS)]
    if smoke:
        jobs = [jobs[0], (A.RULES[1], 'poisson', 'input', 0.5, s_ref['input'], adapt_steps)]
    print(f'{len(jobs)} jobs on {N_WORKERS} workers; s_ref = {s_ref}')
    with Pool(N_WORKERS) as pool:
        rows = pool.map(job, jobs, chunksize=1)
    if smoke:
        for r in rows:
            print(r)
        return
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    np.savez(RESULTS, rows=json.dumps(rows), s_ref=json.dumps(s_ref))
    params = dict(seed=SEED, adapt_ms=4200.0, kinds=KINDS, locs=LOCS, rhos=RHOS,
                  noise_bin_ms=NOISE_BIN_MS, s_ref=s_ref,
                  git_commit=subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT,
                                            capture_output=True, text=True).stdout.strip(),
                  date=datetime.datetime.now().isoformat(timespec='seconds'))
    with open(RESULTS.replace('.npz', '_params.json'), 'w') as f:
        json.dump(params, f, indent=2)
    print('saved', RESULTS)


def plot():
    d = np.load(RESULTS)
    rows = json.loads(str(d['rows']))
    ps.apply()
    loc_color = {'gain': ps.OKABE_ITO['blue'], 'response': ps.OKABE_ITO['vermillion'],
                 'input': ps.OKABE_ITO['green']}
    kind_ls = {'white': '-', 'poisson': (0, (3, 2))}
    fig, axes = plt.subplots(len(METRICS), 3, figsize=(ps.FULL_WIDTH, 210 * ps.MM), sharex=True)
    fig.subplots_adjust(left=0.13, right=0.98, top=0.93, bottom=0.07, wspace=0.35, hspace=0.3)
    x = [0.0] + RHOS
    for j, rule in enumerate(A.RULES):
        base = next(r for r in rows if r['rule'] == rule and r['kind'] == 'none')
        axes[0, j].set_title(A.RULE_LABEL[rule], fontsize=10, fontweight='bold', pad=8)
        for i, m in enumerate(METRICS):
            ax = axes[i, j]
            for kind, loc in itertools.product(KINDS, LOCS):
                sel = sorted((r for r in rows if r['rule'] == rule and r['kind'] == kind and r['loc'] == loc),
                             key=lambda r: r['rho'])
                y = [base[m]] + [r[m] for r in sel]
                ax.plot(x, y, color=loc_color[loc], ls=kind_ls[kind], lw=1.1, marker='o', ms=2.5,
                        label=f'{LOC_LABEL[loc]}, {kind}' if (i == 0 and j == 0) else None)
            ps.ref_line(ax, 0.0 if m.startswith('shift') else 1.0, axis='h')
            if j == 0:
                ax.set_ylabel(METRIC_LABEL[m], fontsize=8)
            if i == len(METRICS) - 1:
                ax.set_xlabel(r'Noise s.d. / signal, $\rho$')
        for i in range(len(METRICS)):
            pass
    for i in range(len(METRICS)):
        lo = min(a.get_ylim()[0] for a in axes[i]); hi = max(a.get_ylim()[1] for a in axes[i])
        for a in axes[i]:
            a.set_ylim(lo, hi)
    fig.legend(loc='upper center', ncol=3, fontsize=7, bbox_to_anchor=(0.55, 1.0), frameon=False)
    os.makedirs(os.path.dirname(FIG_BASE), exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{FIG_BASE}.{ext}', dpi=ps.DPI)
    print('saved', FIG_BASE + '.{pdf,png}')
    # text summary
    for rule in A.RULES:
        print(f'\n{A.RULE_LABEL[rule]}')
        for r in sorted((r for r in rows if r['rule'] == rule), key=lambda r: (r['kind'], r['loc'], r['rho'])):
            print(f"  {r['kind']:8s} {r['loc']:9s} rho={r['rho']:<5} " +
                  ' '.join(f"{m}={r[m]:+.3f}" for m in METRICS) + f"  gmax={r['g_max_cRF']:.3f}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--plot', action='store_true')
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    if a.smoke:
        run(smoke=True)
    else:
        if not a.plot:
            run()
        plot()
