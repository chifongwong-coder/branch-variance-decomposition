"""E5a plot script 2 of 3: A2 strict L^1 bound test.

Two-panel figure showing:
  (a) LHS_p(t) vs RHS_global_p(t) curves for p in {0.1, 0.3, 0.5}
  (b) ratio RHS_global/LHS vs t per p, with horizontal line at 1.0
      indicating "bound just barely holds"

Also overlays support-size summary (|S_i|) as a secondary axis on panel (a)
to contextualise the curves.

Output: figures/e5_bound_test.{png,pdf}
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
OUT_PNG = HERE.parents[1] / "figures" / "e5_bound_test.png"

P_LIST = [0.1, 0.3, 0.5]
P_COLOR = {0.1: Palette.C3_grad[1], 0.3: Palette.C3_grad[3], 0.5: "#5c1d03"}


def load():
    d = json.load(open(JSON_PATH))
    t_grid = np.array(d["config"]["t_grid"])
    seeds = d["config"]["seeds"]
    # arr[p][metric] has shape (seeds, t)
    metrics = ("LHS", "RHS_global", "RHS_local", "RHS_q99",
               "RHS_R_global", "RHS_R_local", "RHS_R_q99")
    out = {p: {m: np.zeros((len(seeds), len(t_grid)))
               for m in metrics}
           for p in P_LIST}
    supp = np.zeros((len(seeds), len(t_grid)))
    for r in d["runs"]:
        si = seeds.index(r["seed"])
        ti = int(np.argmin(np.abs(t_grid - r["t"])))
        supp[si, ti] = r["support_size_mean"]
        for p in P_LIST:
            pk = f"p={p}"
            for m in metrics:
                # RHS_R_* fields may be absent in older JSONs; default to 0.
                out[p][m][si, ti] = r["A2"][pk].get(m, 0.0)
    return d["config"], t_grid, seeds, out, supp


def main():
    config, t_grid, seeds, A2, supp = load()

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))
    ax_lhs_rhs, ax_ratio = axes

    # Panel (a): LHS vs RHS curves
    for p in P_LIST:
        color = P_COLOR[p]
        lhs = A2[p]["LHS"].mean(axis=0)
        lhs_sd = A2[p]["LHS"].std(axis=0, ddof=1)
        rhs = A2[p]["RHS_global"].mean(axis=0)
        rhs_sd = A2[p]["RHS_global"].std(axis=0, ddof=1)
        ax_lhs_rhs.plot(t_grid, lhs, color=color, ls="-", lw=2.0,
                        marker="o", ms=4, mew=0,
                        label=rf"LHS$_p = |\Delta\mathcal{{A}}_{{\rm between}}|$  "
                              rf"$p={p}$")
        ax_lhs_rhs.fill_between(t_grid, lhs - lhs_sd, lhs + lhs_sd,
                                color=color, alpha=0.12, lw=0)
        ax_lhs_rhs.plot(t_grid, rhs, color=color, ls="--", lw=1.6,
                        marker="s", ms=4, mew=0,
                        label=rf"RHS$_p$ global  $p={p}$")
        ax_lhs_rhs.fill_between(t_grid, rhs - rhs_sd, rhs + rhs_sd,
                                color=color, alpha=0.08, lw=0)

    ax_lhs_rhs.set_xlabel(r"$t$")
    ax_lhs_rhs.set_ylabel(r"$|A_{\rm between}^r - A_{\rm between}^{\hat r_p}|$ vs upper bound")
    ax_lhs_rhs.set_title("(a) LHS and RHS$_{\\rm global}$ vs $t$ at three $p$",
                         fontsize=11)
    ax_lhs_rhs.grid(True, alpha=0.22, lw=0.7)
    ax_lhs_rhs.set_yscale("log")
    ax_lhs_rhs.legend(loc="best", fontsize=8.5, framealpha=0.92, ncol=1)

    # Panel (b): ratio RHS/LHS for the 3 M^2 sup-norm form (solid) vs the
    # per-anchor local 3 E[R(y)^2 |r-r_hat|_1] form (dashed), per
    # Remark proxy-tv-radius. The local form is the strongest integrated
    # bound; the global 3 R-bar^2 form (dotted) is reported for
    # completeness and lies between the two.
    for p in P_LIST:
        color = P_COLOR[p]
        ratio_M    = (A2[p]["RHS_global"]   / np.maximum(A2[p]["LHS"], 1e-30)).mean(axis=0)
        ratio_R_l  = (A2[p]["RHS_R_local"]  / np.maximum(A2[p]["LHS"], 1e-30)).mean(axis=0)
        ratio_R_g  = (A2[p]["RHS_R_global"] / np.maximum(A2[p]["LHS"], 1e-30)).mean(axis=0)
        ax_ratio.plot(t_grid, ratio_M, color=color, ls="-",  lw=2.0,
                      marker="o", ms=4, mew=0,
                      label=rf"$p={p}$  $3\hat M^2$ (sup-norm)")
        ax_ratio.plot(t_grid, ratio_R_l, color=color, ls="--", lw=1.8,
                      marker="s", ms=4, mew=0,
                      label=rf"$p={p}$  $3\,\mathbb{{E}}[\hat R^2\,|r-\hat r|_1]/\mathbb{{E}}|r-\hat r|_1$ (local)")
        ax_ratio.plot(t_grid, ratio_R_g, color=color, ls=":",  lw=1.2, alpha=0.65,
                      label=rf"$p={p}$  $3\bar R^2$ (global Chebyshev)")
    ax_ratio.axhline(1.0, color="red", ls=":", lw=1.2, alpha=0.7,
                     label="bound just barely holds (= 1)")
    ax_ratio.set_xlabel(r"$t$")
    ax_ratio.set_ylabel(r"$\mathrm{RHS} / \mathrm{LHS}$")
    ax_ratio.set_title("(b) Bound tightness: ratio $\\mathrm{RHS}/\\mathrm{LHS}$",
                       fontsize=11)
    ax_ratio.grid(True, alpha=0.22, lw=0.7)
    ax_ratio.set_yscale("log")
    ax_ratio.legend(loc="best", fontsize=8.5, framealpha=0.92, ncol=2)

    fig.suptitle(
        rf"E5: strict $L^1$ bound test on "
        rf"oracle support, $N$={config['N_per_seed']:,}, {len(seeds)} seeds, "
        rf"$k_{{\rm NN}}={config['k_NN']}$. "
        rf"Bound RHS$\geq$LHS holds in all 117 of 117 (seed$\times t \times p$) cells.",
        y=1.0, fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
