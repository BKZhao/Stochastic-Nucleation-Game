"""Unified publication figure style — Nature/PRX minimalist house style.

Single source of truth for ALL main figures. Supersedes the previous two
conflicting modules (experiments/plot_style.py and code/plotstyle.py).

Design rules (enforced here, not per-script):
  - sans-serif, compact print sizes, panel letters bold lowercase.
  - open frame: left + bottom spine only; ticks outward; recessive gray.
  - NO panel titles (captions carry that); NO rainbow colormaps.
  - magnitude  -> ONE sequential hue (perceptually ordered, CVD-safe).
  - signed     -> diverging two-hue + neutral-gray midpoint at the zero.
  - categorical phases -> fixed, CVD-safe discrete colors (never cycled).
  - vector PDF + review PNG; fonttype 42 (editable text).
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# ----------------------------------------------------------------------
# print geometry (Elsevier / CSF double column ~ 7.0 in; single ~ 3.4 in)
# ----------------------------------------------------------------------
WIDTH_SINGLE = 3.40
WIDTH_ONEHALF = 5.20
WIDTH_DOUBLE = 7.02

# square-panel geometry (user house rule: square, not rectangular)
# a single square panel ~ single-column; a row of two squares ~ double-column.
WIDTH_SQ = 3.30          # side of one square panel (single-column)
WIDTH_SQ_PAIR = 6.60     # a row of two square panels (double-column)

# ----------------------------------------------------------------------
# ink + neutral tokens (cool-biased grays, not pure gray)
# ----------------------------------------------------------------------
INK = "#16191f"        # primary text / strong lines
INK2 = "#4a515c"       # secondary text
MUTED = "#8a919c"      # axis lines, ticks, grid, guide lines
FAINT = "#c7ccd3"      # faint fills / minor grid

# ----------------------------------------------------------------------
# categorical (Wong 2011, CVD-safe) — semantic assignment, fixed order
# ----------------------------------------------------------------------
BLUE = "#0F5AA6"       # prosperity / cooperators / negative-coupling
RED = "#C0392B"        # desert / defectors / collapse
GREEN = "#2E8B6F"      # rescue / recovery
AMBER = "#E08A1E"      # cascade / warning
PURPLE = "#8E5FA8"     # secondary series
TEAL = "#3E8FA6"       # secondary series

# fixed categorical order (never cycled; a 7th series folds into "other")
CATEGORICAL = [BLUE, RED, GREEN, AMBER, PURPLE, TEAL]

# CSF-style saturated series palette for a PARAMETER SWEEP (one hue per value),
# ordered dark->light like the reference paper's tau=0..1 series. CVD-imperfect
# by nature (a rainbow sweep), so ALWAYS pair with a right-side value legend.
CSF_SERIES = [
    "#111111", "#d81e1e", "#1a9e3b", "#1f5fd0", "#17b0c4",
    "#c026c0", "#8a6d1e", "#e07b1a", "#7a3fd0", "#7f7f7f",
]


def series_colors(n):
    """n saturated series colors for a parameter sweep (interpolated if n>base)."""
    import matplotlib as _mpl
    if n <= len(CSF_SERIES):
        return CSF_SERIES[:n]
    cm = _mpl.colors.LinearSegmentedColormap.from_list("csfser", CSF_SERIES)
    return [cm(i / (n - 1)) for i in range(n)]


def series_legend(ax, handles, labels, title=None, loc="right"):
    """Right-side boxed vertical legend (CSF house style)."""
    leg = ax.legend(handles, labels, title=title, loc="center left",
                    bbox_to_anchor=(1.02, 0.5), frameon=True, fancybox=False,
                    edgecolor=MUTED, framealpha=1.0, borderpad=0.5,
                    handlelength=1.6, labelspacing=0.35, fontsize=6.8)
    leg.get_frame().set_linewidth(0.6)
    if title:
        leg.get_title().set_fontsize(7.2)
    return leg

# ----------------------------------------------------------------------
# sequential (magnitude) — single cool hue, light -> dark, CVD-safe.
# hand-tuned navy->steel->pale so it is perceptually monotone and NOT viridis.
# ----------------------------------------------------------------------
def _make_sequential():
    stops = [
        (0.00, "#f3f6fa"),
        (0.30, "#a9c6df"),
        (0.60, "#4b8cbf"),
        (0.85, "#1f5a95"),
        (1.00, "#0d2f57"),
    ]
    return mcolors.LinearSegmentedColormap.from_list("pub_seq", stops)

SEQ = _make_sequential()

# ----------------------------------------------------------------------
# diverging (signed, zero-centered) — blue (neg) / gray / red (pos).
# neutral gray midpoint; NO hue at the center (dataviz non-negotiable).
# ----------------------------------------------------------------------
def _make_diverging():
    stops = [
        (0.00, "#0F5AA6"),
        (0.25, "#6FA3CE"),
        (0.50, "#EDEDED"),
        (0.75, "#D98C82"),
        (1.00, "#C0392B"),
    ]
    return mcolors.LinearSegmentedColormap.from_list("pub_div", stops)

DIV = _make_diverging()

# ----------------------------------------------------------------------
# categorical phase colormap (discrete) — order chosen per figure.
# default: desert / prosperity / bistable-mixed
# ----------------------------------------------------------------------
PHASE_COLORS = [RED, BLUE, GREEN]
PHASE_CMAP = mcolors.ListedColormap(PHASE_COLORS)

# multilayer outcome (0 joint desert, 1 contained, 2 joint active, 3 mixed)
OUTCOME_COLORS = [RED, AMBER, GREEN, FAINT]
OUTCOME_CMAP = mcolors.ListedColormap(OUTCOME_COLORS)
OUTCOME_LABELS = ["joint desert", "contained", "joint active", "mixed"]


def apply():
    """Install the global rcParams for the house style."""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "TeX Gyre Heros",
                            "DejaVu Sans"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "sans",
        "mathtext.it": "sans:italic",
        "mathtext.default": "regular",
        "font.size": 8.0,
        "axes.labelsize": 8.5,
        "axes.titlesize": 8.5,
        "axes.titlepad": 3.0,
        "axes.labelcolor": INK,
        "axes.edgecolor": MUTED,
        "axes.linewidth": 0.7,
        "text.color": INK,
        # CSF house style: four-side frame, ticks INWARD, top+right ticks on
        "axes.spines.top": True,
        "axes.spines.right": True,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.labelcolor": INK,
        "ytick.labelcolor": INK,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 2.6,
        "ytick.major.size": 2.6,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,
        "legend.fontsize": 7.0,
        "legend.frameon": False,
        "legend.handlelength": 1.3,
        "legend.handletextpad": 0.5,
        "legend.columnspacing": 1.0,
        "legend.labelspacing": 0.3,
        "lines.linewidth": 1.4,
        "lines.markersize": 3.6,
        "lines.markeredgewidth": 0.7,
        "axes.grid": False,
        "grid.color": FAINT,
        "grid.linewidth": 0.5,
        "figure.dpi": 150,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": True,
    })


def apply_pre():
    """Journal-width physics panels with readable ticks and distinct line styles."""
    apply()
    mpl.rcParams.update({
        "font.size": 9.0, "axes.labelsize": 10.0, "axes.titlesize": 9.0,
        "xtick.labelsize": 9.0, "ytick.labelsize": 9.0,
        "legend.fontsize": 9.0, "lines.markersize": 4.5,
        "axes.linewidth": .8, "xtick.major.size": 3.0, "ytick.major.size": 3.0,
    })


def apply_aaai():
    """ML-conference (AAAI/NeurIPS) figure style: larger fonts for small
    two-column figures, open frame, ticks inward. Perceptually uniform
    colormaps (SEQ_A / DIV_A) are used instead of the custom navy ramp."""
    apply()
    mpl.rcParams.update({
        "font.size": 11.0,
        "axes.labelsize": 12.0,
        "axes.titlesize": 12.0,
        "xtick.labelsize": 10.0,
        "ytick.labelsize": 10.0,
        "legend.fontsize": 9.5,
        "lines.linewidth": 1.8,
        "lines.markersize": 4.5,
        "axes.linewidth": 0.9,
        "xtick.major.width": 0.9,
        "ytick.major.width": 0.9,
        "xtick.major.size": 3.2,
        "ytick.major.size": 3.2,
    })


def compact_layout(fig, *, w_pad=0.003, h_pad=0.003,
                   wspace=0.005, hspace=0.005):
    """Tighten constrained-layout padding for dense journal multi-panel plots."""
    for ax in fig.axes:
        ax.xaxis.labelpad = 2.0
        ax.yaxis.labelpad = 2.0
        ax.tick_params(pad=2.0)
    fig.get_layout_engine().set(
        w_pad=w_pad,
        h_pad=h_pad,
        wspace=wspace,
        hspace=hspace,
    )


# perceptually-uniform, polished colormaps for the ML-conference style.
# magma = sequential magnitude (black->purple->orange->yellow), coolwarm =
# smooth diverging for signed/zero-centred fields.
# AAAI figures reuse the original custom house palette (user preference):
SEQ_A = SEQ          # navy -> cyan -> pale sequential
DIV_A = DIV          # blue -> gray -> red diverging
MAG_A = plt.get_cmap("cividis")


def panel_label(ax, letter, x=-0.10, y=1.02, **kw):
    """Bold lowercase panel label in axes coordinates, formatted as ``(a)``."""
    artist = ax.text(
        x, y, f"({letter})", transform=ax.transAxes, ha="left", va="bottom",
        fontsize=9.5, fontweight="bold", color=INK, **kw,
    )
    artist.set_in_layout(False)
    panel_labels = getattr(ax.figure, "_panel_labels", [])
    panel_labels.append(artist)
    ax.figure._panel_labels = panel_labels
    return artist


def add_colorbar(fig, mappable, ax, label=None, ticks=None,
                 fraction=0.046, pad=0.03):
    """Slim recessive colorbar. constrained_layout-compatible (no set_position).

    Pass a single Axes or a list of Axes to attach to (for a shared bar).
    """
    cb = fig.colorbar(mappable, ax=ax, fraction=fraction, pad=pad,
                      aspect=26)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor(MUTED)
    cb.ax.tick_params(width=0.7, length=2.4, color=MUTED, labelsize=mpl.rcParams["ytick.labelsize"],
                      labelcolor=INK2)
    if ticks is not None:
        cb.set_ticks(ticks)
    if label:
        cb.set_label(label, fontsize=mpl.rcParams["axes.labelsize"], color=INK, labelpad=2)
    return cb


def label_box(ax, x, y, text, color=None, fontsize=6.2, ha="left",
              va="bottom", box=True, **kw):
    """In-axes annotation with an optional semi-opaque white backing box so text
    never becomes illegible over data. Unified across all figures."""
    color = INK if color is None else color
    bbox = dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.72) if box else None
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=fontsize,
            color=color, bbox=bbox, zorder=20, **kw)


def clean_axes(ax):
    """CSF four-side frame with inward ticks (line panels)."""
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(True)
        ax.spines[s].set_color(INK)
        ax.spines[s].set_linewidth(0.7)
    ax.tick_params(which="both", direction="in", top=True, right=True,
                   width=0.7, length=2.6, color=INK)


def heatmap_frame(ax):
    """For imshow/pcolormesh panels: thin full box (a heatmap needs closure)."""
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(True)
        ax.spines[s].set_color(MUTED)
        ax.spines[s].set_linewidth(0.7)
    ax.tick_params(width=0.7, length=2.4, color=MUTED)


def save(fig, stem, figure_dir):
    """Export vector PDF + review PNG; return (pdf_path, png_path)."""
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    stem_path = figure_dir / Path(stem).name
    pdf_path = stem_path.with_suffix(".pdf")
    png_path = stem_path.with_suffix(".png")
    save_kwargs = {}
    extra_artists = [
        *getattr(fig, "_panel_labels", []),
        *fig.legends,
    ]
    if extra_artists:
        save_kwargs["bbox_extra_artists"] = extra_artists
    fig.savefig(pdf_path, **save_kwargs)
    fig.savefig(png_path, dpi=400, **save_kwargs)
    plt.close(fig)
    return pdf_path, png_path
