'''
plotstyle.py

One style for every supplementary figure. Figures are drawn at their final printed size
(165 mm full width, 80 mm half width) and exported as PNG at 600 dpi, so that they are included
in LaTeX without rescaling.

Fonts: Arial, then Helvetica; if neither is installed, a metric-compatible clone (Liberation Sans
for Arial, TeX Gyre Heros for Helvetica), then DejaVu Sans. Math text uses the same sans-serif face.
Sizes at final size: axis labels 10 pt, tick labels 9 pt, legends 9 pt, panel letters 12 pt bold.
'''

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

MM = 1.0 / 25.4
FULL_WIDTH = 165 * MM
HALF_WIDTH = 80 * MM
DPI = 600

# ---- Okabe-Ito palette, one meaning per colour in every figure ----
OKABE_ITO = dict(black='#000000', orange='#E69F00', sky='#56B4E9', green='#009E73',
                 yellow='#F0E442', blue='#0072B2', vermillion='#D55E00', purple='#CC79A7')
COLORS = dict(
    old='#0072B2',          # codebase equations (and the a neurons, which gate the old feedback)
    new='#D55E00',          # revised equations (and the u neurons, which gate the new feedback)
    principal='#000000',    # principal neurons y
    norm_on='#009E73',      # normalization on
    norm_off='#E69F00',     # normalization off
    uniform='#56B4E9',      # uniform ensemble
    biased='#CC79A7',       # biased ensemble
    reference='#7F7F7F',    # reference lines, unadapted state
)
SEQUENTIAL = 'viridis'

LW_MAIN = 1.75
LW_THIN = 0.75
ALPHA_THIN = 0.3


def _pick_font():
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in ('Arial', 'Helvetica', 'Liberation Sans', 'TeX Gyre Heros', 'DejaVu Sans'):
        if name in installed:
            return name
    return 'sans-serif'


FONT = _pick_font()


def apply():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': [FONT, 'DejaVu Sans'],
        'mathtext.fontset': 'custom',
        'mathtext.rm': FONT,
        'mathtext.it': f'{FONT}:italic',
        'mathtext.bf': f'{FONT}:bold',
        'mathtext.sf': FONT,
        'mathtext.fallback': 'stixsans',
        'font.size': 9,
        'axes.labelsize': 10,
        'axes.titlesize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'legend.frameon': False,
        'legend.handlelength': 1.6,
        'legend.handletextpad': 0.5,
        'legend.borderaxespad': 0.2,
        'legend.labelspacing': 0.3,
        'axes.linewidth': 0.8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.labelpad': 3,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.minor.width': 0.6,
        'ytick.minor.width': 0.6,
        'xtick.major.size': 3.5,
        'ytick.major.size': 3.5,
        'xtick.minor.size': 2,
        'ytick.minor.size': 2,
        'lines.linewidth': LW_MAIN,
        'lines.solid_capstyle': 'round',
        'savefig.dpi': DPI,
        'figure.dpi': 150,
        'pdf.fonttype': 42,
    })


def panel_label(ax, letter, x=-0.02, y=1.04, **kwargs):
    '''Bold lowercase panel letter (12 pt) at the top-left corner of ax, outside the axes.
    x is in axes fraction measured from the left edge of the y tick labels when dx is given.'''
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=12, fontweight='bold',
            va='bottom', ha='right', **kwargs)


def label_panels(fig, axes, letters='abcdefghijkl', pad_pt=2.0, top_pt=4.0):
    '''Places panel letters after layout: each letter sits just above its axes, left-aligned
    with the left edge of the axes' tick labels / y label.'''
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    for ax, letter in zip(axes, letters):
        tight = ax.get_tightbbox(renderer)
        pos = ax.get_window_extent(renderer)
        x_fig, _ = inv.transform((tight.x0, 0))
        _, y_fig = inv.transform((0, pos.y1 + top_pt * fig.dpi / 72.0))
        fig.text(x_fig, y_fig, letter, fontsize=12, fontweight='bold', va='bottom', ha='left')


def ref_line(ax, value=1.0, axis='h', **kwargs):
    '''Dashed reference line (e.g. ratio = 1).'''
    style = dict(color=COLORS['reference'], lw=0.8, ls=(0, (4, 3)), zorder=0)
    style.update(kwargs)
    return ax.axhline(value, **style) if axis == 'h' else ax.axvline(value, **style)


def save(fig, name):
    '''Saves figures/<name>.png at 600 dpi at exactly the figure's size (no tight cropping, so
    the printed width is the designed width).'''
    from common import FIG_DIR
    path = os.path.join(FIG_DIR, name + '.png')
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path
