"""Unified figure style for paper figures.

Usage at the top of any plotting script:

    from core.figure_style import apply_paper_style, Palette
    apply_paper_style()
    ...
    ax.plot(t, y, color=Palette.pos, label="...")

Calling `apply_paper_style()` once mutates `matplotlib.rcParams` for the
remainder of the Python process. All subsequent `plt.subplots()` and
`fig.savefig()` calls inherit the unified style.
"""

from __future__ import annotations

from dataclasses import dataclass
from matplotlib import rcParams


@dataclass(frozen=True)
class Palette:
    """Paper-consistent colour palette.

    Two-class semantic colours (used in the E1, E2, E3a plots):
      - pos / neg: the +1 vs -1 branches
      - avg:      mixture / posterior-average / total
      - ref:      closed-form theoretical reference lines

    Coupling colours (E3a binary, E3b, E3 plots) -- keep these stable across
    every plot so a reader can identify a coupling by colour:
      - C0 independent       : grey (baseline)
      - C1 Euclidean OT      : blue
      - C2 condition/branch  : green
      - C3 semantic-cost OT  : warm gradient from light to dark with lambda
      - C4 geometry-only OT  : purple
    """
    # two-class semantic
    pos: str = "#e87a23"      # K = +1 / branch A
    neg: str = "#3471a4"      # K = -1 / branch B
    avg: str = "#1a1a1a"      # mixture / posterior average / total
    ref: str = "#c0392b"      # critical-SNR / theoretical reference
    light: str = "#bbbbbb"    # secondary

    # coupling colours
    C0: str = "#7f7f7f"       # independent baseline (grey)
    C1: str = "#3471a4"       # Euclidean OT (blue)
    C2: str = "#3aa455"       # condition / branch aware (green)
    C4: str = "#8c4ab5"       # geometry-only OT (purple)
    # C3 lambda gradient (light to dark warm)
    C3_grad: tuple = (
        "#fee08b",            # lambda ~ 0.5 or 1 (lightest)
        "#fdae6b",
        "#f47f43",
        "#d94801",
        "#8c2d04",            # lambda ~ 30 or 100 (darkest)
    )


def coupling_color(name: str, lam_index: int | None = None) -> str:
    """Return the canonical colour for a coupling name.

    `name` is the canonical coupling identifier ("C0_independent",
    "C1_euclidean_ot", "C2_branch_aware", "C2_condition_aware_random",
    "C3_semOT_lam<X>", "C4_geometry_only_ot").  `lam_index` is the position
    of the lambda in the sweep (0 = smallest, len-1 = largest).
    """
    p = Palette()
    if name.startswith("C0"):
        return p.C0
    if name.startswith("C1"):
        return p.C1
    if name.startswith("C2"):
        return p.C2
    if name.startswith("C4"):
        return p.C4
    if name.startswith("C3"):
        if lam_index is None:
            return p.C3_grad[2]  # middle of gradient
        n = len(p.C3_grad)
        idx = max(0, min(n - 1, lam_index))
        return p.C3_grad[idx]
    return p.avg


def apply_paper_style() -> None:
    """Mutate matplotlib rcParams to a unified paper style.

    Idempotent.  Call once at the top of a plotting script.
    """
    rcParams["font.family"] = "serif"
    rcParams["font.size"] = 10
    rcParams["mathtext.fontset"] = "cm"

    # Embed fonts as Type 42 (TrueType) rather than the matplotlib default
    # Type 3 (PostScript bitmap). Type 42 yields selectable / searchable PDF
    # text and is required by some venues (ACM / IEEE camera-ready).
    rcParams["pdf.fonttype"] = 42
    rcParams["ps.fonttype"] = 42

    rcParams["axes.titlesize"] = 9
    rcParams["axes.labelsize"] = 9
    rcParams["xtick.labelsize"] = 9
    rcParams["ytick.labelsize"] = 9
    rcParams["legend.fontsize"] = 8.5
    rcParams["legend.frameon"] = True
    rcParams["legend.framealpha"] = 0.92
    rcParams["legend.edgecolor"] = "#999999"

    rcParams["axes.grid"] = True
    rcParams["grid.alpha"] = 0.22
    rcParams["grid.linewidth"] = 0.7
    rcParams["axes.spines.top"] = False
    rcParams["axes.spines.right"] = False
    rcParams["axes.linewidth"] = 0.9

    rcParams["lines.linewidth"] = 1.5
    rcParams["lines.markersize"] = 4

    rcParams["figure.dpi"] = 130
    rcParams["savefig.dpi"] = 200
    rcParams["savefig.bbox"] = "tight"
    rcParams["savefig.facecolor"] = "white"


__all__ = ["apply_paper_style", "Palette", "coupling_color"]
