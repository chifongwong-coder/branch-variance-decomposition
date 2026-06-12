"""E4 plotter: reads results/e4_metrics_<TAG>.json (produced by
e4_fm_mlp_fixed_t.py) and draws the capacity-ladder figure: final eval loss,
approximation error, and the irreducible lower bound versus MLP width, per t,
with the closed-form floor. Run and plot are separated so the figure regenerates
from saved data without retraining and carries no experiment-number title.

Usage:
    python plot_e4.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RES = HERE.parents[1] / "results"
FIG = HERE.parents[1] / "figures"

TAG = "conservative"     # JSON tag plotted into the paper figure


def main():
    with open(RES / f"e4_metrics_{TAG}.json") as f:
        d = json.load(f)
    cfg = d["config"]
    runs = d["runs"]
    av_cf = {float(k): v for k, v in d["av_closed_form"].items()}
    T_GRID = cfg["T_GRID"]
    HIDDEN_LADDER = cfg["HIDDEN_LADDER"]

    fig, axes = plt.subplots(1, len(T_GRID), figsize=(16, 4),
                             sharey=False, squeeze=False)
    axes = axes[0]
    for ti, t in enumerate(T_GRID):
        ax = axes[ti]
        L_list, approx_list, irred_list = [], [], []
        for hidden in HIDDEN_LADDER:
            sub = [r for r in runs if r["t"] == t and r["hidden"] == hidden]
            L_list.append([r["L_eval_final"] for r in sub])
            approx_list.append([r["approx_eval_final"] for r in sub])
            irred_list.append([r["irred_eval_final"] for r in sub])
        L_arr = np.array(L_list)              # (n_hidden, n_seeds)
        approx_arr = np.array(approx_list)
        irred_arr = np.array(irred_list)
        x = np.array(HIDDEN_LADDER)
        ax.errorbar(x, L_arr.mean(1), yerr=L_arr.std(1, ddof=1),
                    fmt="o-", capsize=4, lw=2, label="eval loss")
        ax.errorbar(x, approx_arr.mean(1), yerr=approx_arr.std(1, ddof=1),
                    fmt="s--", capsize=4, lw=2, label=r"$\|v_\theta - v^*\|^2$")
        ax.errorbar(x, irred_arr.mean(1), yerr=irred_arr.std(1, ddof=1),
                    fmt="^:", capsize=4, lw=2, label=r"irred lower bound")
        ax.axhline(av_cf[t], color="black", ls=":", alpha=0.6,
                   label=fr"$\mathcal{{A}}_v(t)$ CF: {av_cf[t]:.3f}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("hidden width")
        if ti == 0:
            ax.set_ylabel("loss")
        ax.set_title(f"t = {t}", fontsize=10)
        if ti == len(T_GRID) - 1:
            ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    FIG.mkdir(exist_ok=True)
    out = FIG / f"e4_capacity_ladder_{TAG}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
