"""Build a single comparison figure summarising the P2 Sinkhorn baseline
against the Hungarian phase 2 reference.

Reads:
  results/e3_metrics_phase2.json  (Hungarian C1 / C3@10 / C4)
  results/e3_metrics_sinkhorn.json (Sinkhorn C1 / C3@10 / C4 x eps)

Writes:
  figures/e3_sinkhorn_vs_hungarian.png

Two panels:
  Left  : Pareto plot, T_full (raw Euclidean transport) vs.
          A_v_sem^S(t=1/2 | C). Hungarian shown as filled diamonds;
          Sinkhorn shown as smaller circles connected by a dashed
          line in increasing-eps order to make the entropic spread
          visible.
  Right : Mismatch rate as a function of eps for each base coupling,
          with the Hungarian reference as a horizontal dashed line.

Run:
  python3 build_sinkhorn_comparison_figure.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.figure_style import apply_paper_style, Palette
apply_paper_style()

HERE = Path(__file__).resolve().parent
RES = HERE.parents[1] / "results"
FIG = HERE.parents[1] / "figures"

HUNG_JSON = RES / "e3_metrics_phase2.json"
SINK_JSON = RES / "e3_metrics_sinkhorn.json"
OUT_PNG = FIG / "e3_sinkhorn_vs_hungarian.png"

# Map "base" coupling -> Hungarian phase 2 key (paper-active rows)
HUNG_KEY = {
    "C1": "C1_euclidean_ot",
    "C3@10": "C3_semOT_lam10",
    "C4": "C4_geometry_only_ot",
}
# Map "base" coupling -> Sinkhorn prefix
SINK_PREFIX = {
    "C1":    "C1_sinkhorn_eps",
    "C3@10": "C3_lam10_sinkhorn_eps",
    "C4":    "C4_sinkhorn_eps",
}
COLOR = {
    "C1":    Palette.neg,         # blue
    "C3@10": "#d68a3c",           # warm orange (mid of C3 gradient)
    "C4":    "#7a4eaf",           # purple, matching paper palette
}
LABEL = {
    "C1":    "C1",
    "C3@10": r"C3 $\lambda{=}10$",
    "C4":    "C4",
}
EPS_LIST = [0.03, 0.1, 0.3]


def aggregate_phase2(data, key):
    runs = [r for r in data["runs"].values() if r["name"] == key]
    pm = [r["pair_metrics"] for r in runs]
    mismatch = [p["mismatch_rate"] for p in pm]
    t_full = [p["transport_full"] for p in pm]
    # t=0.5 slice is index 9 in linspace(0.05, 0.95, 19)
    av_sem = [r["metrics_per_t"][9]["cond_avg"]["80"]["sem"]["A_v_norm"]
              for r in runs]
    return {
        "mismatch_mean": float(np.mean(mismatch)),
        "mismatch_sd":   float(np.std(mismatch, ddof=1) if len(mismatch) > 1 else 0.0),
        "tfull_mean":    float(np.mean(t_full)),
        "tfull_sd":      float(np.std(t_full, ddof=1) if len(t_full) > 1 else 0.0),
        "avsem_mean":    float(np.mean(av_sem)),
        "avsem_sd":      float(np.std(av_sem, ddof=1) if len(av_sem) > 1 else 0.0),
    }


def aggregate_sinkhorn(data, prefix, eps):
    # Returns None if this (coupling, eps) row is absent. main() assumes every
    # base x eps combination in EPS_LIST is present in the JSON (the full
    # released Sinkhorn run); a partial run leaves a None that the plotting
    # loops will dereference, so regenerate all rows before building the figure.
    key = f"{prefix}{eps:g}"
    runs = [r for r in data["runs"].values() if r["name"] == key]
    if not runs:
        return None
    return aggregate_phase2(data, key)


def main():
    apply_paper_style()
    plt.rcParams.update({"axes.labelsize": 15, "xtick.labelsize": 15,
                         "ytick.labelsize": 15, "axes.titlesize": 15})
    with open(HUNG_JSON) as f:
        hung = json.load(f)
    with open(SINK_JSON) as f:
        sink = json.load(f)

    hung_agg = {base: aggregate_phase2(hung, k) for base, k in HUNG_KEY.items()}
    sink_agg = {
        base: {eps: aggregate_sinkhorn(sink, SINK_PREFIX[base], eps)
               for eps in EPS_LIST}
        for base in HUNG_KEY
    }

    fig, (ax_pareto, ax_bar) = plt.subplots(1, 2, figsize=(12.5, 4.6))

    # ============================================================
    # Left panel: Pareto T_full vs A_v_sem
    # ============================================================
    for base in HUNG_KEY:
        col = COLOR[base]
        # Sinkhorn trajectory across eps
        xs = [sink_agg[base][e]["tfull_mean"] for e in EPS_LIST]
        ys = [sink_agg[base][e]["avsem_mean"] for e in EPS_LIST]
        xerr = [sink_agg[base][e]["tfull_sd"] for e in EPS_LIST]
        yerr = [sink_agg[base][e]["avsem_sd"] for e in EPS_LIST]
        ax_pareto.plot(xs, ys, "--", color=col, alpha=0.6, lw=1.0, zorder=1)
        for i, e in enumerate(EPS_LIST):
            ax_pareto.errorbar(xs[i], ys[i], xerr=xerr[i], yerr=yerr[i],
                                fmt="o", color=col, ms=6 + 2 * i,
                                alpha=0.55 + 0.2 * i, capsize=2, lw=1.0,
                                markeredgewidth=0.6, markeredgecolor="white",
                                zorder=2,
                                label=f"{LABEL[base]}, $\\varepsilon{{=}}{e:g}$")
        # Hungarian reference: filled diamond
        h = hung_agg[base]
        ax_pareto.errorbar(h["tfull_mean"], h["avsem_mean"],
                            xerr=h["tfull_sd"], yerr=h["avsem_sd"],
                            fmt="D", color=col, ms=10,
                            markeredgecolor="black", markeredgewidth=0.7,
                            capsize=2, zorder=3,
                            label=f"{LABEL[base]} Hung.")
    ax_pareto.set_xlabel(r"$T_{\rm full}$ (raw Euclidean transport)")
    ax_pareto.set_ylabel(r"$\widetilde{\mathcal{A}}_v^S(t{=}1/2\,|\,C)$")
    ax_pareto.set_title("(a) Pareto: transport vs semantic ambiguity (k=80)")
    ax_pareto.grid(True, alpha=0.22)
    # legend moved to top strip — collected after all series are added

    # ============================================================
    # Right panel: mismatch by eps with Hungarian baseline
    # ============================================================
    x = np.arange(len(EPS_LIST))
    width = 0.27
    offsets = {"C1": -width, "C3@10": 0.0, "C4": +width}
    for base in HUNG_KEY:
        col = COLOR[base]
        means = [sink_agg[base][e]["mismatch_mean"] for e in EPS_LIST]
        sds = [sink_agg[base][e]["mismatch_sd"] for e in EPS_LIST]
        ax_bar.bar(x + offsets[base], means, width, yerr=sds,
                    capsize=2, color=col, alpha=0.9,
                    edgecolor="white", linewidth=0.5,
                    label=f"{LABEL[base]} Sinkhorn")
        # Hungarian reference: short horizontal line spanning the cluster
        h_mm = hung_agg[base]["mismatch_mean"]
        for xi in x:
            ax_bar.hlines(h_mm, xi + offsets[base] - width / 2,
                           xi + offsets[base] + width / 2,
                           color=col, linestyles=(0, (3, 1.5)),
                           linewidth=1.3, zorder=4)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([f"$\\varepsilon{{=}}{e:g}$" for e in EPS_LIST])
    ax_bar.set_xlabel("Sinkhorn entropy regularization")
    ax_bar.set_ylabel(r"mismatch rate  $\Pr(K_X \neq C)$")
    ax_bar.set_title("(b) Mismatch vs $\\varepsilon$  (dashed = Hungarian)")
    ax_bar.set_ylim(-0.02, 0.62)
    ax_bar.grid(True, alpha=0.22, axis="y")
    ax_bar.legend(loc="upper left", fontsize=9, framealpha=0.85)

    handles, labels = ax_pareto.get_legend_handles_labels()
    fig.tight_layout()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.01),
               ncol=4, fontsize=9, framealpha=0.85, borderpad=0.5, columnspacing=0.9)
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
