'''
alternative_rules.py

Supplementary Note 4 (draft): alternative adaptation rules in the same circuit. Each rule is run
through the protocol of analysis/adapt_single_stimulus.py (Fig. 1 of the main text) with
everything else held the same.

Rules (columns of the figure):
  covariance   Covariance homeostasis (the model of the main text): frame N13_mercedes_K182,
               tau_g dg/dt = (c - mu)^2 - theta_t, theta_t = diag(W^T Sigma_target W).
  variance     Variance normalization: identity frame (K = N_RF = 13, interneuron k reads and
               feeds back onto principal neuron k only); gain dynamics unchanged, so
               theta_t = diag(Sigma_target).
  first_order  First-order homeostasis: identity frame; tau_g dg/dt = c - theta_1 (no mean
               subtraction), with c = W^T sqrt(y+) = floor(y) and theta_1 the mean, over
               orientations, of the unadapted (g = 0) fixed-point response y of the cRF neurons to
               a large grating at contrast 0.8. Feedback is unchanged (u+ * W v, v -> g * c).

In every case the gains are clamped to g >= 0, and the probe responses are computed from the
analytical fixed point (Eq. S17), which holds for any feedback of the form W D(g) W^T.

Rows of the figure (Fig. 1b,c of the main text, from adapt_single_stimulus.py):
  1  contrast-response functions of the cRF population (l2 norm of the 13 firing rates) at the
     adaptor orientation
  2  tuning curves of the flank neuron (69.2 deg, 20.8 deg from the 90 deg adaptor)
The gain trajectories and membrane-potential curves are still saved in the .npz.

Run from ORGaNICs_Whitening/:
    python supplementary/alternative_rules.py            # simulate, save .npz, plot
    python supplementary/alternative_rules.py --plot     # replot from the saved .npz
    python supplementary/alternative_rules.py --smoke    # short adaptation (smoke test, not saved)
'''

import argparse
import datetime
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, 'analysis'))
sys.path.insert(0, HERE)

import plotstyle as ps                                                # noqa: E402  (Agg backend)
import matplotlib.pyplot as plt                                       # noqa: E402

from simulation import Frame, V1Dynamics_Surround                     # noqa: E402
from stimuli import StimulusGenerator                                 # noqa: E402
from tunings import V1Tunings                                         # noqa: E402
from response_tools import (offline_gain_operator, get_response_offline,  # noqa: E402
                            curve_peak_deg, to_steps)
import adapt_single_stimulus as ASS                                   # noqa: E402

RULES = ['covariance', 'variance', 'first_order']
RULE_LABEL = {'covariance': 'Covariance homeostasis',
              'variance': 'Variance normalization',
              'first_order': 'First-order homeostasis'}
FIRST_ORDER_CONTRAST = 0.8      # contrast of the large grating that sets theta_1
N_TARGET_ANGLES = 180           # orientations averaged to obtain theta_1
GAIN_SUBSAMPLE = 10             # store every 10th step of the gain trajectories

RESULTS = os.path.join(HERE, 'results', 'alternative_rules.npz')
FIG_BASE = os.path.join(HERE, 'figures', 'fig_alternative_rules')


class FirstOrderDynamics(V1Dynamics_Surround):
    '''V1Dynamics_Surround with the gain equation replaced by first-order homeostasis:
    tau_g dg/dt = c - theta_1 (cRF and the representative surround set). Every other equation,
    including the u+-gated feedback W v and tau_v dv/dt = -v + g c, is the parent's.'''

    theta_first = None   # (K,) first-order targets, set before running

    def _derivatives(self, state, z_t, surround_msg_delayed):
        d = super()._derivatives(state, z_t, surround_msg_delayed)
        K, N_RF = self.frame.K, self.N_RF
        sqrt_y_plus = np.maximum(state[:N_RF * self.N_SETS], 0.0)          # sqrt(y+) = floor(y)
        c_cRF = self.frame.W.T @ sqrt_y_plus[:N_RF]
        c_surround = (self.frame.W.T @ sqrt_y_plus[N_RF:2 * N_RF] if self.N_SETS >= 2
                      else np.zeros(K))
        # Eq. dg replaced: first-order homeostasis (no mean subtraction, no squaring)
        d[self._off_g_cRF:self._off_g_cRF + K] = (c_cRF - self.theta_first) / self.tau_g
        d[self._off_g_surround:self._off_g_surround + K] = (c_surround - self.theta_first) / self.tau_g
        return d


def identity_frame(n):
    return SimpleNamespace(W=np.eye(n), dim=n, K=n)


def build(rule, stim_gen):
    tunings = V1Tunings(N=ASS.N_RF)
    if rule == 'covariance':
        frame, cls = Frame(csv_path=ASS.FRAME_PATH), V1Dynamics_Surround
    elif rule == 'variance':
        frame, cls = identity_frame(ASS.N_RF), V1Dynamics_Surround
    else:
        frame, cls = identity_frame(ASS.N_RF), FirstOrderDynamics
    dyn = cls(tunings, frame, dt=ASS.DT, N_RF=ASS.N_RF, N_SETS=ASS.N_SETS,
              target_covariance_path=ASS.TARGET_COV_PATH, gains_nonneg=True)
    dyn.calibrate_theta_t(C_zz_uniform=dyn.uniform_target_covariance, uniform_target=True,
                          verbose=False)
    if rule == 'first_order':
        # Mean unadapted (g = 0) fixed-point response of the cRF neurons to a large grating at
        # contrast 0.8, averaged over orientations: one target shared by every interneuron.
        M0 = np.zeros((dyn.N_RF * dyn.N_SETS,) * 2)
        angles = np.linspace(0, np.pi, N_TARGET_ANGLES, endpoint=False)
        probes = np.stack([stim_gen.oriented_drive(a, FIRST_ORDER_CONTRAST, 'adapt CRF and surround')
                           for a in angles], axis=1)
        y0 = get_response_offline(dyn, probes, M0, rectify=False)[:dyn.N_RF]
        dyn.theta_first = np.full(frame.K, np.maximum(y0, 0).mean())
        dyn.theta_t = dyn.theta_first.copy()   # only used by run_adaptation_phase's printout
    return dyn


def run_rule(rule, stim_gen, adapt_steps):
    dyn = build(rule, stim_gen)
    ASS.ADAPTATION_STEPS = adapt_steps
    ASS.SIM_HISTORY.clear()
    print(f'\n=== {RULE_LABEL[rule]} (K = {dyn.frame.K}) ===')
    frozen = {cond: ASS.run_adaptation_phase(dyn, stim_gen, cond) for cond in ASS.CONDITIONS}
    assert all(np.all(np.asarray(v[:2]) >= 0) for v in frozen.values())

    crf = slice(0, ASS.N_RF)
    full = lambda th, c: stim_gen.oriented_drive(th, c, 'adapt CRF and surround')
    M = {cond: offline_gain_operator(dyn, *frozen[cond][:2]) for cond in ASS.CONDITIONS}
    zmin = {}

    def respond(cond, probes, rectify=True):
        zp = np.linalg.solve(np.eye(probes.shape[0]) + M[cond], probes)
        zmin[cond] = min(zmin.get(cond, np.inf), zp[crf].min())
        return get_response_offline(dyn, probes, M[cond], rectify=rectify)

    # Row 2: contrast-response functions (l2 norm of the cRF population) at the adaptor orientation
    P_c = np.stack([full(ASS.ADAPTOR_THETA, c) for c in ASS.CRF_CONTRASTS], axis=1)
    crf_curves = {cond: np.linalg.norm(respond(cond, P_c)[crf], axis=0) for cond in ASS.CONDITIONS}

    # Row 3: flank-neuron tuning curves
    angles = np.linspace(0, np.pi, ASS.N_PROBES, endpoint=False)
    P_t = np.stack([full(a, ASS.PROBE_CONTRAST) for a in angles], axis=1)
    adaptor_idx = stim_gen.num_angles // 2
    flank_idx = (adaptor_idx - 1) % ASS.N_RF
    flank = {cond: respond(cond, P_t)[crf][flank_idx] for cond in ASS.CONDITIONS}

    # Row 4: membrane potential of the adaptor-tuned cRF neuron vs contrast
    vm = {cond: respond(cond, P_c, rectify=False)[crf][adaptor_idx] for cond in ASS.CONDITIONS}

    g_hist = ASS.SIM_HISTORY['adapt CRF and surround'][:, ::GAIN_SUBSAMPLE]
    out = {}
    for cond in ASS.CONDITIONS:
        key = cond.replace(' ', '_')
        out[f'{rule}/crf/{key}'] = crf_curves[cond]
        out[f'{rule}/flank/{key}'] = flank[cond]
        out[f'{rule}/vm/{key}'] = vm[cond]
        out[f'{rule}/g_cRF_final/{key}'] = frozen[cond][0]
        out[f'{rule}/g_surround_final/{key}'] = frozen[cond][1]
        out[f'{rule}/zprime_min/{key}'] = zmin[cond]
    out[f'{rule}/g_hist'] = g_hist
    out[f'{rule}/theta'] = dyn.theta_first if rule == 'first_order' else dyn.theta_t
    out[f'{rule}/flank_idx'] = flank_idx
    return out


def simulate(smoke=False):
    np.random.seed(0)
    stim_gen = StimulusGenerator(N_RF=ASS.N_RF, N_SETS=ASS.N_SETS, num_angles=ASS.N_RF,
                                 tuning_width=ASS.TUNING_WIDTH)
    adapt_ms = 200.0 if smoke else 4200.0
    adapt_steps = to_steps(adapt_ms, ASS.DT, 'adapt_ms')
    out = {}
    for rule in RULES:
        out.update(run_rule(rule, stim_gen, adapt_steps))
    out['adapt_ms'] = adapt_ms
    out['crf_contrasts'] = ASS.CRF_CONTRASTS
    out['probe_angles_deg'] = np.degrees(np.linspace(0, np.pi, ASS.N_PROBES, endpoint=False))
    out['gain_t_ms'] = np.arange(0, adapt_steps, GAIN_SUBSAMPLE) * ASS.DT + ASS.DT
    if smoke:
        report(out)
        return
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    np.savez(RESULTS, **out)
    params = dict(seed=0, dt_ms=ASS.DT, adapt_ms=adapt_ms, adaptor_theta_deg=np.degrees(ASS.ADAPTOR_THETA),
                  adaptor_contrast=ASS.ADAPTOR_CONTRAST, probe_contrast=ASS.PROBE_CONTRAST,
                  target_cov=os.path.basename(ASS.TARGET_COV_PATH), frame=os.path.basename(ASS.FRAME_PATH),
                  first_order_contrast=FIRST_ORDER_CONTRAST, rules=RULE_LABEL,
                  git_commit=subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT,
                                            capture_output=True, text=True).stdout.strip(),
                  date=datetime.datetime.now().isoformat(timespec='seconds'))
    with open(RESULTS.replace('.npz', '_params.json'), 'w') as f:
        json.dump(params, f, indent=2)
    print('saved', RESULTS)


def report(d):
    ang = d['probe_angles_deg']
    for rule in RULES:
        print(f"\n{RULE_LABEL[rule]}: theta = {np.unique(np.round(d[f'{rule}/theta'], 6))[:4]}")
        ctrl = curve_peak_deg(d[f'{rule}/flank/no_adaptation'], ang)
        for cond in ASS.CONDITIONS:
            key = cond.replace(' ', '_')
            g = d[f'{rule}/g_cRF_final/{key}']
            gs = d[f'{rule}/g_surround_final/{key}']
            shift = ((curve_peak_deg(d[f'{rule}/flank/{key}'], ang) - ctrl + 90) % 180) - 90
            r = d[f'{rule}/crf/{key}'] / d[f'{rule}/crf/no_adaptation'].max()
            print(f"  {cond:24s} g_cRF>1e-3: {np.sum(g > 1e-3):3d}/{g.size} (max {g.max():.3g}), "
                  f"g_surr max {gs.max():.3g} | CRF at c=1: {r[-1]:.3f}, at c=0.1: "
                  f"{r[np.argmin(abs(d['crf_contrasts'] - 0.1))]:.3f} | flank shift {shift:+.2f} deg | "
                  f"min z' {float(d[f'{rule}/zprime_min/{key}']):+.3g}")


def plot():
    d = dict(np.load(RESULTS))
    report(d)
    ps.apply()
    C, ang = d['crf_contrasts'], d['probe_angles_deg']
    adaptor_deg = np.degrees(ASS.ADAPTOR_THETA)
    colors = dict(ASS.CONDITION_COLOR)
    colors['adapt CRF only'] = '#E6B800'      # darker yellow so it reads on white

    fig, axes = plt.subplots(2, 3, figsize=(ps.FULL_WIDTH, 100 * ps.MM))
    fig.subplots_adjust(left=0.1, right=0.98, top=0.9, bottom=0.11, wspace=0.42, hspace=0.55)

    for j, rule in enumerate(RULES):
        ax_c, ax_t = axes[:, j]
        ax_c.set_title(RULE_LABEL[rule], fontsize=10, fontweight='bold', pad=10)

        norm = d[f'{rule}/crf/no_adaptation'].max()
        for cond in ASS.CONDITIONS:
            ax_c.plot(C, d[f'{rule}/crf/{cond.replace(" ", "_")}'] / norm, color=colors[cond], lw=1.25,
                      label=ASS.CONDITION_LABEL[cond])
        ax_c.set_xscale('log')
        ax_c.set_xlabel('Contrast')
        ax_c.set_ylim(bottom=0)

        flank_conds = ['no adaptation', 'adapt surround only', 'adapt CRF only']
        ctrl = curve_peak_deg(d[f'{rule}/flank/no_adaptation'], ang)
        fnorm = d[f'{rule}/flank/no_adaptation'].max()
        for cond in flank_conds:
            curve = d[f'{rule}/flank/{cond.replace(" ", "_")}']
            shift = ((curve_peak_deg(curve, ang) - ctrl + 90) % 180) - 90
            lab = None if cond == 'no adaptation' else fr'$\Delta\theta$ = {shift:+.1f}°'
            ax_t.plot(ang, curve / fnorm, color=colors[cond], lw=1.25, label=lab)
        ps.ref_line(ax_t, ctrl, axis='v')
        ax_t.annotate('', xy=(adaptor_deg, 0.80), xytext=(adaptor_deg, 0.97),
                      xycoords=('data', 'axes fraction'), textcoords=('data', 'axes fraction'),
                      arrowprops=dict(arrowstyle='-|>', color='black', lw=1.0, mutation_scale=8))
        ax_t.set_xlim(0, 180)
        ax_t.set_xticks([0, 90, 180])
        ax_t.set_xlabel('Orientation (deg)')
        ax_t.set_ylim(bottom=0)
        ax_t.legend(loc='upper right', fontsize=7, handlelength=1.0, bbox_to_anchor=(1.05, 1.0))

        if j == 0:
            ax_c.set_ylabel('Norm. response')
            ax_t.set_ylabel('Norm. response')

    # shared y ranges across columns
    for row in (0, 1):
        hi = max(ax.get_ylim()[1] for ax in axes[row])
        for ax in axes[row]:
            ax.set_ylim(0, hi)
    axes[0, 0].legend(loc='upper left', fontsize=7, handlelength=1.2)

    ps.label_panels(fig, axes.T.ravel())
    os.makedirs(os.path.dirname(FIG_BASE), exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{FIG_BASE}.{ext}', dpi=ps.DPI)
    print('saved', FIG_BASE + '.{pdf,png}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--plot', action='store_true')
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    if a.smoke:
        simulate(smoke=True)
    else:
        if not a.plot:
            simulate()
        plot()
