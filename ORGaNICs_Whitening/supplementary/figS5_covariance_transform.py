'''
figS5_covariance_transform.py  ->  figures/FigS5_covariance_transform.png, data/FigS5_covariance_transform.npz

Does an exact covariance transformation alone (no normalization, no circuit) give a power law?

For each environment X (von Mises orientation distributions of figS4_power_law.py) the covariance
Sigma_X of the cRF input drive z (small grating, contrast 0.8) is computed exactly on a 1-deg
orientation grid. The unique symmetric positive-definite M_X with M_X Sigma_X M_X = Sigma_t is

    M_X = Sigma_X^(-1/2) (Sigma_X^(1/2) Sigma_t Sigma_X^(1/2))^(1/2) Sigma_X^(-1/2),

with a ridge eps * mean eigenvalue added to Sigma_X. Responses are M_X z(s) at the 13 preferred
orientations; firing rate = max(y,0)^2; magnitude = l2 norm of the rates (as in Fig. S4).

Targets: (i) the circuit's target (0.15 x uniform_target_covariance_mid_c), (ii) identity
(whitening), (iii) the covariance of the input drive under a uniform ensemble.
Inputs: the three environments with all concentrations kappa multiplied by 0.25 ... 4.
No random numbers. Run:  python figS5_covariance_transform.py [--recompute]
'''

import sys

import numpy as np

import common as C
import plotstyle as ps
from figS4_power_law import ENVS, PAIRS, PROBE_ANGLES, MID_TARGET, density, fit_power_law

NAME = 'FigS5_covariance_transform'
EPS = 1e-6                                    # ridge, relative to the mean eigenvalue of Sigma_X
EPS_SWEEP = [1e-10, 1e-8, 1e-6, 1e-4, 1e-2]
KAPPA_SCALES = [0.25, 0.5, 1.0, 2.0, 4.0]
GRID = np.radians(np.arange(0.0, 180.0, 1.0))
Z_GRID = C.oriented(GRID, C.ENSEMBLE_CONTRAST, C.SMALL)[:, :C.N_RF]          # (180, 13)
Z_PROBE = C.oriented(PROBE_ANGLES, C.ENSEMBLE_CONTRAST, C.SMALL)[:, :C.N_RF]  # (13, 13)


def sqrtm_psd(A):
    w, V = np.linalg.eigh((A + A.T) / 2)
    return (V * np.sqrt(np.clip(w, 0, None))) @ V.T


def input_covariance(p):
    '''Mean and covariance of the input drive when orientations have density p on GRID.'''
    w = p / p.sum()
    mean = w @ Z_GRID
    return mean, (Z_GRID - mean).T @ ((Z_GRID - mean) * w[:, None])


def exact_transform(sigma_x, sigma_t, eps=EPS):
    s = sigma_x + eps * np.trace(sigma_x) / len(sigma_x) * np.eye(len(sigma_x))
    root = sqrtm_psd(s)
    inv_root = np.linalg.inv(root)
    return inv_root @ sqrtm_psd(root @ sigma_t @ root) @ inv_root


def subspace_transform(sigma_x, sigma_t, rel_tol=1e-6):
    '''Variant for rank-deficient inputs: the exact transform inside the subspace where the
    input has variance (eigenvalues > rel_tol x largest), and the identity outside it, i.e.
    directions the ensemble does not constrain are left untouched (as the circuit does with
    zero gains).'''
    w, V = np.linalg.eigh(sigma_x)
    U = V[:, w > rel_tol * w[-1]]
    sx, st = U.T @ sigma_x @ U, U.T @ sigma_t @ U
    root = sqrtm_psd(sx)
    inv_root = np.linalg.inv(root)
    m_sub = inv_root @ sqrtm_psd(root @ st @ root) @ inv_root
    return U @ m_sub @ U.T + (np.eye(len(sigma_x)) - U @ U.T), U.shape[1]


def targets():
    _, uniform_in = input_covariance(np.ones(len(GRID)))
    return {'circuit target': MID_TARGET, 'identity (whitening)': np.eye(C.N_RF), 'uniform input covariance': uniform_in}


def experiment(sigma_t, kappa_scale=1.0, eps=EPS, convention='rate', subspace=False):
    '''Pooled log10 probability ratios and log10 response ratios, and the fits.'''
    prob, mag, extra = {}, {}, {}
    for env, (center, kappa) in ENVS.items():
        p_grid = density(GRID, center, kappa * kappa_scale)
        mean, sigma_x = input_covariance(p_grid)
        M = subspace_transform(sigma_x, sigma_t)[0] if subspace else exact_transform(sigma_x, sigma_t, eps)
        y = Z_PROBE @ M.T
        rate = np.maximum(y, 0) ** 2 if convention == 'rate' else np.abs(y)
        mag[env] = np.linalg.norm(rate, axis=1)
        prob[env] = density(PROBE_ANGLES, center, kappa * kappa_scale)
        extra[env] = dict(sigma_x=sigma_x, M=M, cond=np.linalg.cond(sigma_x), ymin=y.min(),
                          residual=np.abs(M @ (sigma_x + eps * np.trace(sigma_x) / len(sigma_x) * np.eye(len(sigma_x))) @ M - sigma_t).max())
    x = np.concatenate([np.log10(prob[a] / prob[b]) for a, b in PAIRS])
    y = np.concatenate([np.log10(mag[a] / mag[b]) for a, b in PAIRS])
    keep = np.abs(x) <= 1
    return dict(x=x, y=y, full=fit_power_law(x, y), inside=fit_power_law(x[keep], y[keep]), extra=extra)


def simulate():
    out = dict(target_names=np.array(list(targets())), kappa_scales=np.array(KAPPA_SCALES), eps_sweep=np.array(EPS_SWEEP), eps=EPS)
    for i, (name, sigma_t) in enumerate(targets().items()):
        e = experiment(sigma_t)
        out[f'x_{i}'], out[f'y_{i}'] = e['x'], e['y']
        out[f'fit_{i}'] = np.array([*e['full'], *e['inside']])
        out[f'sigma_t_{i}'] = sigma_t
        out[f'kappa_fit_{i}'] = np.array([[*r['full'], *r['inside'], np.abs(r['x']).max()] for r in
                                          (experiment(sigma_t, kappa_scale=k) for k in KAPPA_SCALES)])
        out[f'eps_fit_{i}'] = np.array([[*r['full'], *r['inside']] for r in (experiment(sigma_t, eps=ep) for ep in EPS_SWEEP)])
        out[f'amp_fit_{i}'] = np.array(experiment(sigma_t, convention='amplitude')['full'])
        sub = experiment(sigma_t, subspace=True)
        out[f'sub_fit_{i}'] = np.array([*sub['full'], *sub['inside']])
        out[f'corr_{i}'] = np.corrcoef(e['x'], e['y'])[0, 1]
        if i == 0:
            for env in ENVS:
                out[f'sigma_x_{env}'] = e['extra'][env]['sigma_x']
                out[f'M_{env}'] = e['extra'][env]['M']
            out['cond'] = np.array([e['extra'][env]['cond'] for env in ENVS])
            out['residual'] = np.array([e['extra'][env]['residual'] for env in ENVS])
            out['ymin'] = np.array([e['extra'][env]['ymin'] for env in ENVS])
            out['rank'] = np.array([subspace_transform(e['extra'][env]['sigma_x'], sigma_t)[1] for env in ENVS])
            out['spectrum_A'] = np.linalg.eigvalsh(e['extra']['A']['sigma_x'])[::-1]
            ones = np.ones(C.N_RF) / np.sqrt(C.N_RF)
            out['dc_fraction'] = np.array([np.mean((Z_PROBE @ ones) ** 2 / np.sum(Z_PROBE ** 2, axis=1))])
            out['dc_variance_fraction'] = np.array([ones @ e['extra'][env]['sigma_x'] @ ones / np.trace(e['extra'][env]['sigma_x']) for env in ENVS])
    return out


def plot(d):
    import matplotlib.pyplot as plt
    ps.apply()
    OK = ps.OKABE_ITO
    colors = [OK['black'], OK['purple'], OK['sky']]
    names = list(d['target_names'])
    fig = plt.figure(figsize=(ps.FULL_WIDTH, 122 * ps.MM))
    ax_x = fig.add_axes([0.1, 0.61, 0.19, 0.3])
    ax_t = fig.add_axes([0.325, 0.61, 0.19, 0.3])
    ax_s = fig.add_axes([0.685, 0.61, 0.3, 0.3])
    ax_p = fig.add_axes([0.095, 0.1, 0.4, 0.34])
    ax_k = fig.add_axes([0.635, 0.1, 0.35, 0.34])

    ticks = [0, 6, 12]
    labels = ['0', '83', '166']
    for ax, mat, title in ((ax_x, d['sigma_x_A'], 'Input covariance (A)'), (ax_t, d['sigma_t_0'], 'Target covariance')):
        im = ax.imshow(mat / np.abs(mat).max(), cmap=ps.SEQUENTIAL, vmin=-1, vmax=1)
        ax.set_xticks(ticks, labels=labels)
        ax.set_yticks(ticks, labels=labels)
        ax.set_title(title, fontsize=9, pad=5)
        for sp in ax.spines.values():
            sp.set_visible(True)
    ax_x.set_ylabel('Preferred\norientation (deg)')
    fig.text(0.2975, 0.555, 'Preferred orientation (deg)', ha='center', va='top', fontsize=10)
    ax_t.set_yticklabels([])
    cax = fig.add_axes([0.527, 0.655, 0.01, 0.21])
    cb = fig.colorbar(im, cax=cax, ticks=[-1, 0, 1])
    cb.ax.tick_params(labelsize=8, length=2)

    # c: responses to each probe before and after the exact transform (environment A vs C)
    pref = np.degrees(PROBE_ANGLES)
    for env, color in (('A', ps.COLORS['biased']), ('C', ps.COLORS['uniform'])):
        mag = np.linalg.norm(np.maximum(Z_PROBE @ d[f'M_{env}'].T, 0) ** 2, axis=1)
        ax_s.plot(pref, mag / mag.mean(), color=color, lw=ps.LW_MAIN, marker='o', ms=3.5, label=f'environment {env}')
        ax_s.axvline(ENVS[env][0], color=color, lw=0.8, ls=(0, (4, 3)))
    ax_s.set_xlim(0, 166.2)
    ax_s.set_xticks([0, 45, 90, 135])
    ax_s.set_ylim(0, 1.3)
    ax_s.set_yticks([0, 0.5, 1.0])
    ax_s.set_xlabel('Test orientation (deg)')
    ax_s.set_ylabel('Response magnitude\n(mean = 1)')
    ax_s.legend(loc='lower center', fontsize=8, ncol=2, columnspacing=1.0, handlelength=1.4, bbox_to_anchor=(0.5, 0.02))
    ax_s.text(ENVS['A'][0] + 3, 0.42, 'most frequent\nin A', fontsize=8, color=ps.COLORS['biased'], va='center')

    # d: log-log ratios under the exact transform for each target
    ps.ref_line(ax_p, 0.0)
    ps.ref_line(ax_p, 0.0, axis='v')
    for i, name in enumerate(names):
        x, y = d[f'x_{i}'], d[f'y_{i}']
        beta, r2 = d[f'fit_{i}'][:2]
        ax_p.scatter(x, y, s=14, color=colors[i], linewidths=0, zorder=3, alpha=0.85)
        xs = np.array([x.min(), x.max()])
        ax_p.plot(xs, beta * xs, color=colors[i], lw=ps.LW_MAIN, label=rf'{name}: $\beta$ = {beta:.2f}, $R^2$ = {r2:.1f}'.replace('-', '−'))
    ax_p.axvspan(-1, 1, color=ps.COLORS['reference'], alpha=0.1, lw=0, zorder=0)
    ax_p.set_xlabel(r'$\log_{10}\,[p_X/p_Y]$')
    ax_p.set_ylabel(r'$\log_{10}\,[r_X/r_Y]$')
    ax_p.set_ylim(-0.07, 0.42)
    ax_p.set_yticks([0, 0.1, 0.2])
    ax_p.legend(loc='upper left', fontsize=8, handlelength=1.4, bbox_to_anchor=(0.0, 1.04), labelspacing=0.2)

    # e: exponent versus how peaked the environments are
    ks = d['kappa_scales']
    ax_k.axhspan(-0.4, -0.15, color=ps.COLORS['reference'], alpha=0.22, lw=0, zorder=0)
    ax_k.text(1.0, -0.275, 'experiment', fontsize=8, color='#4D4D4D', ha='center', va='center')
    ps.ref_line(ax_k, 0.0)
    for i, name in enumerate(names):
        ax_k.plot(ks, d[f'kappa_fit_{i}'][:, 0], color=colors[i], lw=ps.LW_MAIN, marker='o', ms=3.5)
    ax_k.set_xscale('log')
    ax_k.set_xticks(ks, labels=[f'{k:g}' for k in ks])
    ax_k.minorticks_off()
    ax_k.set_ylim(-0.45, 0.08)
    ax_k.set_yticks([-0.4, -0.2, 0])
    ax_k.set_xlabel(r'Concentration $\kappa$ of all environments ($\times$)')
    ax_k.set_ylabel(r'Exponent $\beta$')
    ps.label_panels(fig, [ax_x, ax_t, ax_s, ax_p, ax_k])
    return ps.save(fig, NAME)


def report(d):
    print(f'  ridge eps = {float(d["eps"]):g} x mean eigenvalue; condition number of Sigma_X (A, B, C): '
          + ', '.join(f'{c:.1e}' for c in d['cond']) + f'; max |M Sigma M - Sigma_t|: {d["residual"].max():.1e}; '
          f'min component of M z: {d["ymin"].min():+.2e}')
    sp = d['spectrum_A']
    print(f'  eigenvalues of Sigma_A / largest: ' + ', '.join(f'{v:.1e}' for v in sp[:5] / sp[0]) + f'; numerical rank (A, B, C): {d["rank"]}')
    print(f'  fraction of the squared norm of the raw probe drive along the uniform (all-neurons) direction: {float(d["dc_fraction"][0]):.2f}; '
          f'fraction of the input variance along it (A, B, C): ' + ', '.join(f'{v:.1e}' for v in d['dc_variance_fraction']))
    for i, name in enumerate(d['target_names']):
        f = d[f'fit_{i}']
        print(f'  target = {name}: full range beta = {f[0]:.3f}, R2 = {f[1]:.3f}; |x| <= 1: beta = {f[2]:.3f}, R2 = {f[3]:.3f}; '
              f'amplitude convention (|y| instead of y^2): beta = {d[f"amp_fit_{i}"][0]:.3f}, R2 = {d[f"amp_fit_{i}"][1]:.3f}')
        print(f'     Pearson correlation of log r ratio with log p ratio: {float(d[f"corr_{i}"]):+.3f}; subspace variant (identity outside '
              f'the input subspace): beta = {d[f"sub_fit_{i}"][0]:.3f}, R2 = {d[f"sub_fit_{i}"][1]:.3f}; |x| <= 1: beta = {d[f"sub_fit_{i}"][2]:.3f}, R2 = {d[f"sub_fit_{i}"][3]:.3f}')
        print('     ridge sweep (eps: beta, R2 full): ' + '; '.join(f'{e:g}: {r[0]:.3f}, {r[1]:.3f}' for e, r in zip(d['eps_sweep'], d[f'eps_fit_{i}'])))
        print('     kappa scale (x: beta, R2 full | beta, R2 inside | max |log p ratio|): ' + '; '.join(
            f'{k:g}: {r[0]:.3f}, {r[1]:.3f} | {r[2]:.3f}, {r[3]:.3f} | {r[4]:.2f}' for k, r in zip(d['kappa_scales'], d[f'kappa_fit_{i}'])))


if __name__ == '__main__':
    d = None if '--recompute' in sys.argv else C.load_data(NAME)
    if d is None:
        C.save_data(NAME, **simulate())
        d = C.load_data(NAME)
    report(d)
    print('saved', plot(d))
