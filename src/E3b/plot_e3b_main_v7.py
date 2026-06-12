"""Main E3b figure: A_v, A_within, A_between, R_switch vs t for all 10
couplings, with seed-SD bands.

Reads results/e3b_unified_v7.json (produced by e3b_branch_refinement.py full).
Outputs figures/e3b_main.{png,pdf}.

Uses only the coarse-K decomposition (A_within_K, A_between_K). Fine-K'
columns are used by plot_e3b_refinement.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.figure_style import apply_paper_style, Palette, coupling_color
apply_paper_style()

HERE = Path(__file__).resolve().parent
JSON_PATH = HERE.parents[1] / "results" / "e3b_unified_v7.json"
OUT_PNG = HERE.parents[1] / "figures" / "e3b_main.png"


def load_runs():
    d = json.load(open(JSON_PATH))
    config = d["config"]
    t_grid = np.array(config["t_grid"], dtype=np.float64)
    coupling_order = config["couplings"]
    seeds = config["seeds"]

    # group: for each coupling -> (n_seeds, n_t) arrays for each metric
    metrics = ("A_v", "A_within_K", "A_between_K")
    by_cpl = {c: {m: [] for m in metrics} for c in coupling_order}
    for r in d["runs"]:
        c = r["coupling"]
        for m in metrics:
            by_cpl[c][m].append(np.asarray(r[m], dtype=np.float64))
    for c in coupling_order:
        for m in metrics:
            by_cpl[c][m] = np.stack(by_cpl[c][m], axis=0)  # (seeds, t)
        # R_switch = A_between / (A_v + eps)
        eps = 1e-8
        by_cpl[c]["R_switch_K"] = (by_cpl[c]["A_between_K"]
                                   / np.maximum(by_cpl[c]["A_v"], eps))
    return config, t_grid, coupling_order, seeds, by_cpl


def coupling_display(name: str) -> str:
    """Pretty label for the 10-coupling list."""
    mapping = {
        "C0_independent":   r"$C_0$ indep",
        "C1_hungarian":     r"$C_1$ Euclidean OT",
        "C2_coarse_random": r"$C_2$ coarse-random",
        "C3_lam0_5_coarse": r"$C_3@\lambda{=}0.5$",
        "C3_lam1_coarse":   r"$C_3@\lambda{=}1$",
        "C3_lam2_coarse":   r"$C_3@\lambda{=}2$",
        "C3_lam5_coarse":   r"$C_3@\lambda{=}5$",
        "C3_lam10_coarse":  r"$C_3@\lambda{=}10$",
        "C3_lam30_coarse":  r"$C_3@\lambda{=}30$",
        "C3_inf_coarse":    r"$C_3^\infty$ blocked OT",
    }
    return mapping.get(name, name)


def coupling_style(name: str):
    """(color, linestyle, linewidth, marker) per coupling."""
    if name == "C0_independent":
        return Palette.C0, "-", 1.6, None
    if name == "C1_hungarian":
        return Palette.C1, "-", 1.8, "o"
    if name == "C2_coarse_random":
        return Palette.C2, "-", 1.8, "s"
    if name == "C3_inf_coarse":
        return "#8c2d04", "--", 2.0, None  # darkest warm, dashed for endpoint
    if name == "C3_lam0_5_coarse":
        return Palette.C3_grad[0], "-", 1.4, None
    if name == "C3_lam1_coarse":
        return Palette.C3_grad[1], "-", 1.4, None
    if name == "C3_lam2_coarse":
        return Palette.C3_grad[2], "-", 1.4, None
    if name == "C3_lam5_coarse":
        return Palette.C3_grad[3], "-", 1.4, None
    if name == "C3_lam10_coarse":
        return Palette.C3_grad[4], "-", 1.4, None
    if name == "C3_lam30_coarse":
        return "#5c1d03", "-", 1.6, None
    return "#888", "-", 1.0, None


def main():
    config, t_grid, coupling_order, seeds, by_cpl = load_runs()

    apply_paper_style()
    plt.rcParams.update({"axes.labelsize": 14, "xtick.labelsize": 14,
                         "ytick.labelsize": 14, "axes.titlesize": 14})
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.8), sharex=True)
    ax_av, ax_aw, ax_ab, ax_rs = axes.flat

    panels = [
        (ax_av, "A_v",        r"(a) $\mathcal{A}_v(t)$"),
        (ax_aw, "A_within_K", r"(b) $\mathcal{A}_{\rm within}^{K}(t)$"),
        (ax_ab, "A_between_K",r"(c) $\mathcal{A}_{\rm between}^{K}(t)$"),
        (ax_rs, "R_switch_K", r"(d) $\mathcal{R}_{\rm switch}^{K}(t)$"),
    ]

    for c in coupling_order:
        color, ls, lw, marker = coupling_style(c)
        for ax, metric, _ in panels:
            arr = by_cpl[c][metric]                # (seeds, t)
            mean = arr.mean(axis=0)
            sd = arr.std(axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros_like(mean)
            ax.plot(t_grid, mean, color=color, ls=ls, lw=lw,
                    marker=marker, ms=3, mew=0,
                    label=coupling_display(c))
            ax.fill_between(t_grid, mean - sd, mean + sd,
                            color=color, alpha=0.12, lw=0)

    for ax, _, title in panels:
        ax.set_title(title, fontsize=14)
        ax.grid(True, alpha=0.22, lw=0.7)

    ax_av.set_ylabel("(velocity-variance units)")
    ax_ab.set_ylabel("(velocity-variance units)")
    ax_ab.set_xlabel(r"$t$")
    ax_rs.set_xlabel(r"$t$")
    ax_rs.set_ylim(-0.02, 1.02)

    # reserve the top strip for the figure legend so it does not cover the
    # (a)/(b) panel titles of the top row
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    handles, labels = ax_av.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, fontsize=10,
               frameon=True, framealpha=0.92, bbox_to_anchor=(0.5, 1.0))
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
