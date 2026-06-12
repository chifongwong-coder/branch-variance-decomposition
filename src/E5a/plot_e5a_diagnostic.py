"""E5a plot script 1 of 3: A1 real proxy diagnostic.

Reads results/e5a_v7.json and produces a 2x2 grid of
A_v(t), A_within^L(t), A_between^L(t), R_switch^L(t) for L in
{oracle, CLIP_kmeans, weakRP_kmeans}, with seed-SD bands.

Output: figures/e5a_proxy_diagnostic.{png,pdf}
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
JSON_PATH = HERE.parents[1] / "results" / "e5a_v7.json"
OUT_PNG = HERE.parents[1] / "figures" / "e5a_proxy_diagnostic.png"


LABEL_DISPLAY = {
    "oracle": r"$K_{\rm oracle}$ (CIFAR-10 class)",
    "CLIP":   r"$K_{\rm proxy}^{\rm CLIP}$ (k-means on CLIP features)",
    "weakRP": r"$K_{\rm proxy}^{\rm weakRP}$ (k-means on rand-proj-16)",
}

LABEL_COLOR = {
    "oracle":  Palette.avg,           # black-ish
    "CLIP":    Palette.pos,           # orange
    "weakRP":  Palette.neg,           # blue
}

LABEL_LS = {"oracle": "-", "CLIP": "--", "weakRP": ":"}


def load():
    d = json.load(open(JSON_PATH))
    t_grid = np.array(d["config"]["t_grid"])
    seeds = d["config"]["seeds"]
    metrics = {}
    for key in ("A_v",
                "A_within_oracle", "A_between_oracle",
                "A_within_CLIP",   "A_between_CLIP",
                "A_within_weakRP", "A_between_weakRP",
                "support_size_mean",
                "support_size_median", "frac_full_support", "frac_single_support"):
        arr = np.zeros((len(seeds), len(t_grid)))
        for r in d["runs"]:
            si = seeds.index(r["seed"])
            ti = int(np.argmin(np.abs(t_grid - r["t"])))
            arr[si, ti] = r[key]
        metrics[key] = arr
    return d["config"], t_grid, seeds, metrics


def plot_curve(ax, t_grid, arr, color, ls, label):
    mean = arr.mean(axis=0)
    sd = arr.std(axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros_like(mean)
    ax.plot(t_grid, mean, color=color, ls=ls, lw=2.0, label=label)
    ax.fill_between(t_grid, mean - sd, mean + sd, color=color, alpha=0.18, lw=0)


def main():
    config, t_grid, seeds, m = load()

    apply_paper_style()
    plt.rcParams.update({"axes.labelsize": 15, "xtick.labelsize": 15,
                         "ytick.labelsize": 15, "axes.titlesize": 15})
    # 2x2 layout:
    #   top row (the headline pair): A_between curves + R_switch curves
    #   bottom row (the context pair): A_v (flat, scale-matched explanation)
    #                                  + |\mathcal{K}_i| (branch commitment signal)
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0), sharex=True)
    ax_ab, ax_rs, ax_av, ax_si = axes.flat

    # A_between per label
    for lbl, color in LABEL_COLOR.items():
        plot_curve(ax_ab, t_grid, m[f"A_between_{lbl}"], color, LABEL_LS[lbl],
                   LABEL_DISPLAY[lbl])
    ax_ab.set_title(r"(a) $\mathcal{A}_{\rm between}^{L}(t)$",
                    fontsize=15)
    ax_ab.set_ylabel("(velocity-variance units)")

    # R_switch = A_between / A_v per label
    eps = 1e-12  # matches paper §3 epsilon_0 convention
    for lbl, color in LABEL_COLOR.items():
        rs = m[f"A_between_{lbl}"] / np.maximum(m["A_v"], eps)
        plot_curve(ax_rs, t_grid, rs, color, LABEL_LS[lbl], LABEL_DISPLAY[lbl])
    ax_rs.set_title(r"(b) $\mathcal{R}_{\rm switch}^{L}(t)$ = $\mathcal{A}_{\rm between}^{L}/\mathcal{A}_v$",
                    fontsize=15)
    ax_rs.set_ylabel("ratio")
    ax_rs.set_ylim(-0.005, max(0.25, ax_rs.get_ylim()[1]))

    # A_v (label-independent), included to show the scale-matched flatness
    plot_curve(ax_av, t_grid, m["A_v"], Palette.avg, "-", r"$\mathcal{A}_v$")
    ax_av.set_title(r"(c) $\mathcal{A}_v(t)$",
                    fontsize=15)
    ax_av.set_ylabel("(velocity-variance units)")
    ax_av.set_xlabel(r"$t$")

    # Support-size |\mathcal{K}_i| = number of distinct oracle classes per kNN neighborhood
    plot_curve(ax_si, t_grid, m["support_size_mean"], Palette.ref, "-",
               r"$|\mathcal{K}_i|$ mean")
    # also overlay frac_full_support and frac_single_support on a secondary axis
    ax_si.set_title(r"(d) $|\mathcal{K}_i|(t)$",
                    fontsize=15)
    ax_si.set_ylabel(r"$|\mathcal{K}_i|$ mean (max = $|\mathcal{K}| = 10$)")
    ax_si.set_xlabel(r"$t$")
    ax_si.set_ylim(-0.3, 10.5)
    ax_si.axhline(10, color="gray", ls=":", lw=1, alpha=0.6)
    ax_si.axhline(1, color="gray", ls=":", lw=1, alpha=0.6)
    ax_si2 = ax_si.twinx()
    plot_curve(ax_si2, t_grid, m["frac_full_support"], Palette.pos, ":",
               r"$\Pr(|\mathcal{K}_i| = 10)$")
    plot_curve(ax_si2, t_grid, m["frac_single_support"], Palette.neg, ":",
               r"$\Pr(|\mathcal{K}_i| = 1)$")
    ax_si2.set_ylabel("fraction of anchors")
    ax_si2.set_ylim(-0.02, 1.02)
    h1, l1 = ax_si.get_legend_handles_labels()
    h2, l2 = ax_si2.get_legend_handles_labels()
    ax_si.legend(h1 + h2, l1 + l2, loc="center right", fontsize=9, framealpha=0.92)

    for ax in axes.flat:
        ax.grid(True, alpha=0.22, lw=0.7)

    # main legend lives on the A_between panel (shared label set with R_switch)
    ax_ab.legend(loc="upper right", fontsize=9, framealpha=0.92)

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
