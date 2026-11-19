'''
_style.py - shared helpers for the paper figure scripts in figures/.

Loads paper.mplstyle, defines figure widths and the condition -> colour mapping (Okabe-Ito,
the same colour for the same condition in every figure), and writes each figure as a vector
PDF plus a PNG preview.
'''

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FIG_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(FIG_DIR)
plt.style.use(os.path.join(FIG_DIR, 'paper.mplstyle'))

MM = 1.0 / 25.4
SINGLE_COL = 89 * MM
DOUBLE_COL = 183 * MM

# Condition -> colour (Okabe-Ito; validated for CVD separation).
COLORS = dict(
    original='#0072B2',     # time constants as in V1Dynamics_Surround
    tuned='#D55E00',        # time constants that settle within the deadline
    reference='#7F7F7F',    # analytic steady state, tolerance lines
    window='#EBEBEB',       # shading of the settling deadline window
)


def panel_letters(axes, letters='abcdefgh', x=-0.2):
    '''Bold lowercase panel letters (8 pt) at the top left of each axes. Drawn as left-aligned
    axes titles so constrained_layout reserves room for them; x is in axes fraction.'''
    for ax, letter in zip(axes, letters):
        ax.set_title(letter, loc='left', x=x, fontsize=8, fontweight='bold', pad=4)


def save(fig, out_dir, name):
    '''Writes <out_dir>/<name>.pdf (vector) and <name>.png (600 dpi preview) at the designed size.'''
    os.makedirs(out_dir, exist_ok=True)
    paths = [os.path.join(out_dir, f'{name}.{ext}') for ext in ('pdf', 'png')]
    for p in paths:
        fig.savefig(p)
    plt.close(fig)
    return paths
