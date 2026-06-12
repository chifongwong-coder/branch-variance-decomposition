"""E3a plotter: reads results/e3a_metrics_<TAG>.json (produced by
e3a_coupling_comparison.py) and draws the four-panel coupling-comparison figure
for the binary geometry-aligned positive control. Run and plot are separated so
the figure regenerates from saved data with no recompute and carries no
experiment-number title.

Usage:
    python plot_e3a.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.figure_style import apply_paper_style

HERE = Path(__file__).resolve().parent
RES = HERE.parents[1] / "results"
FIG = HERE.parents[1] / "figures"

TAG = "default"          # JSON tag plotted into the paper figure
OUTPUT_STEM = "e3a_curves"
K = "80"                 # kNN bandwidth column (string key after JSON round-trip)


def display_name(name):
    """Map an internal coupling name to a clean legend label."""
    if name.startswith("C0"):
        return r"$C_0$"
    if name.startswith("C1"):
        return r"$C_1$"
    if name.startswith("C2"):
        return r"$C_2$"
    if name.startswith("C3_semOT_lam"):
        return rf"$C_3@\lambda={name.split('lam')[1]}$"
    return name


def main():
    with open(RES / f"e3a_metrics_{TAG}.json") as f:
        d = json.load(f)
    cfg = d["config"]
    runs = d["runs"]
    c0_oracle = d.get("c0_oracle", {})

    by_name = {}
    for run in runs.values():
        by_name.setdefault(run["name"], []).append(run)
    # Drop the C3@lambda=0 sanity run (it only verifies C3 reduces to C1).
    names = [n for n in by_name if "sanity" not in n]

    apply_paper_style()
    plt.rcParams.update({"axes.labelsize": 14, "xtick.labelsize": 14,
                         "ytick.labelsize": 14, "axes.titlesize": 14})
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.2))
    colour_map = plt.get_cmap("tab10")
    for ni, name in enumerate(names):
        seeds_data = by_name[name]
        t_arr = np.array([m["t"] for m in seeds_data[0]["metrics"]])

        def stack(metric):
            return np.array([[mt["per_k"][K][metric] for mt in run["metrics"]]
                             for run in seeds_data])  # (n_seeds, n_t)

        av_norm = stack("A_v_norm").mean(0)
        H_norm = stack("H_KY").mean(0) / np.log(2.0)
        r_sw = stack("R_switch").mean(0)
        not_R_v2 = stack("one_minus_R_v2").mean(0)

        c = colour_map(ni % 10)
        axes[0, 0].plot(t_arr, av_norm, "o-", lw=1.5, ms=4, color=c,
                        alpha=0.85, label=display_name(name))
        axes[0, 1].plot(t_arr, H_norm, "o-", lw=1.5, ms=4, color=c,
                        alpha=0.85, label=display_name(name))
        axes[1, 0].plot(t_arr, r_sw, "o-", lw=1.5, ms=4, color=c,
                        alpha=0.85, label=display_name(name))
        axes[1, 1].plot(t_arr, not_R_v2, "o-", lw=1.5, ms=4, color=c,
                        alpha=0.85, label=display_name(name))

    # closed-form C0 R_switch reference (binary target only).
    if c0_oracle:
        items = sorted((float(t), v) for t, v in c0_oracle.items())
        cf_t = [t for t, _ in items]
        cf_r = [v["R_switch"] for _, v in items]
        axes[1, 0].plot(cf_t, cf_r, "k:", lw=2, alpha=0.7,
                        label="C0 closed-form (oracle)")

    titles = [r"Normalized $\mathcal{A}_v(t) = \mathcal{A}_v/E\|U\|^2$",
              r"$H_K(t)/\log 2$",
              r"$R_{\rm switch}(t) = \mathcal{A}_{\rm between}/\mathcal{A}_v$",
              r"$1 - R_v^2(t)$"]
    for ax, title in zip(axes.flat, titles):
        ax.set_xlabel("t")
        ax.set_title(title, fontsize=14)
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    FIG.mkdir(exist_ok=True)
    out = FIG / f"{OUTPUT_STEM}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
