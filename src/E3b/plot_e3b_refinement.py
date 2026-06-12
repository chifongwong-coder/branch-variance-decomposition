"""v7 appendix figure: branch-refinement sanity check.

Two panels showing the refinement gap on two illustrative
couplings (C2 vs C1) + one bar chart summarising max_t Delta
across all 10 couplings with seed-SD error bars.

Reads results/e3b_unified_v7.json (produced by e3b_branch_refinement.py
full). Outputs figures/e3b_refinement.{png,pdf}.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.figure_style import apply_paper_style, Palette
apply_paper_style()

HERE = Path(__file__).resolve().parent
JSON_PATH = HERE.parents[1] / "results" / "e3b_unified_v7.json"
OUT_PNG = HERE.parents[1] / "figures" / "e3b_refinement.png"

# Couplings highlighted in the story panels.
PANEL_COUPLINGS = ("C2_coarse_random", "C1_hungarian")

# Couplings shown in the summary bar (in display order).
BAR_ORDER = (
    "C0_independent",
    "C1_hungarian",
    "C2_coarse_random",
    "C3_lam0_5_coarse",
    "C3_lam1_coarse",
    "C3_lam2_coarse",
    "C3_lam5_coarse",
    "C3_lam10_coarse",
    "C3_lam30_coarse",
    "C3_inf_coarse",
)


def load_runs():
    d = json.load(open(JSON_PATH))
    config = d["config"]
    t_grid = np.array(config["t_grid"], dtype=np.float64)
    metrics = ("A_v", "A_within_K", "A_between_K",
               "A_within_Kprime", "A_between_Kprime", "Delta_diff")
    by_cpl = {}
    for r in d["runs"]:
        c = r["coupling"]
        if c not in by_cpl:
            by_cpl[c] = {m: [] for m in metrics}
        for m in metrics:
            by_cpl[c][m].append(np.asarray(r[m], dtype=np.float64))
    for c in by_cpl:
        for m in metrics:
            by_cpl[c][m] = np.stack(by_cpl[c][m], axis=0)
    return config, t_grid, by_cpl


def coupling_display(name: str) -> str:
    return {
        "C0_independent":   r"$C0$ indep",
        "C1_hungarian":     r"$C1$ Eucl OT",
        "C2_coarse_random": r"$C2$ coarse-rand",
        "C3_lam0_5_coarse": r"$C3@\lambda{=}0.5$",
        "C3_lam1_coarse":   r"$C3@\lambda{=}1$",
        "C3_lam2_coarse":   r"$C3@\lambda{=}2$",
        "C3_lam5_coarse":   r"$C3@\lambda{=}5$",
        "C3_lam10_coarse":  r"$C3@\lambda{=}10$",
        "C3_lam30_coarse":  r"$C3@\lambda{=}30$",
        "C3_inf_coarse":    r"$C3^\infty$",
    }.get(name, name)


def coupling_color_local(name: str) -> str:
    if name == "C0_independent":    return Palette.C0
    if name == "C1_hungarian":      return Palette.C1
    if name == "C2_coarse_random":  return Palette.C2
    if name == "C3_inf_coarse":     return "#8c2d04"
    if name == "C3_lam0_5_coarse":  return Palette.C3_grad[0]
    if name == "C3_lam1_coarse":    return Palette.C3_grad[1]
    if name == "C3_lam2_coarse":    return Palette.C3_grad[2]
    if name == "C3_lam5_coarse":    return Palette.C3_grad[3]
    if name == "C3_lam10_coarse":   return Palette.C3_grad[4]
    if name == "C3_lam30_coarse":   return "#5c1d03"
    return "#888"


def plot_panel(ax, t_grid, stats, title):
    """One coupling panel: A_between^K (solid), A_between^K' (dashed),
    Delta shaded between."""
    Ab_K = stats["A_between_K"].mean(axis=0)
    Ab_Kp = stats["A_between_Kprime"].mean(axis=0)
    Delta = stats["Delta_diff"].mean(axis=0)
    sd_K = stats["A_between_K"].std(axis=0, ddof=1)
    sd_Kp = stats["A_between_Kprime"].std(axis=0, ddof=1)

    ax.fill_between(t_grid, Ab_K, Ab_Kp, alpha=0.20, color="#7a3a1a",
                    label=r"$\Delta = \mathcal{A}_{\rm between}^{K'} - \mathcal{A}_{\rm between}^{K}$")
    ax.plot(t_grid, Ab_K,  color=Palette.neg, lw=2.0, ls="-",
            label=r"$\mathcal{A}_{\rm between}^{K}$")
    ax.fill_between(t_grid, Ab_K - sd_K, Ab_K + sd_K, color=Palette.neg, alpha=0.10, lw=0)
    ax.plot(t_grid, Ab_Kp, color=Palette.pos, lw=2.0, ls="--",
            label=r"$\mathcal{A}_{\rm between}^{K'}$")
    ax.fill_between(t_grid, Ab_Kp - sd_Kp, Ab_Kp + sd_Kp, color=Palette.pos, alpha=0.10, lw=0)
    ax.plot(t_grid, Delta, color=Palette.avg, lw=1.4, ls=":",
            label=r"$\Delta(t)$")

    ax.set_xlabel(r"$t$")
    ax.set_ylabel("(velocity-variance units)")
    ax.set_title(title, fontsize=8)
    ax.grid(True, alpha=0.22, lw=0.7)
    ax.legend(loc="best", fontsize=6, framealpha=0.92)


def main():
    config, t_grid, by_cpl = load_runs()

    apply_paper_style()
    plt.rcParams.update({"axes.labelsize": 8, "xtick.labelsize": 8,
                         "ytick.labelsize": 8, "axes.titlesize": 8})
    fig = plt.figure(figsize=(8.5, 4.9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0],
                          hspace=0.32, wspace=0.22)
    ax_c2 = fig.add_subplot(gs[0, 0])
    ax_c1 = fig.add_subplot(gs[0, 1])
    ax_bar = fig.add_subplot(gs[1, :])

    # Top row: two illustrative couplings
    plot_panel(ax_c2, t_grid, by_cpl["C2_coarse_random"],
               r"(a) $C2$ coarse-class random (large refinement gap)")
    plot_panel(ax_c1, t_grid, by_cpl["C1_hungarian"],
               r"(b) $C1$ Euclidean OT (small refinement gap)")

    # Bottom: summary bar of max_t Delta per coupling
    labels, means, sds, colors = [], [], [], []
    for c in BAR_ORDER:
        stats = by_cpl[c]
        # max_t Delta per seed, then mean / SD over seeds
        per_seed_max = stats["Delta_diff"].max(axis=1)   # (seeds,)
        labels.append(coupling_display(c))
        means.append(float(per_seed_max.mean()))
        sds.append(float(per_seed_max.std(ddof=1)) if per_seed_max.size > 1 else 0.0)
        colors.append(coupling_color_local(c))

    x = np.arange(len(labels))
    ax_bar.bar(x, means, yerr=sds, capsize=4, color=colors,
               edgecolor="#444", linewidth=0.8)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax_bar.set_ylabel(r"$\max_t \Delta(t)$")
    ax_bar.set_title("(c) Refinement gap by coupling (mean $\\pm$ SD over "
                     f"{len(config['seeds'])} seeds)", fontsize=8)
    ax_bar.grid(True, axis="y", alpha=0.22, lw=0.7)

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
