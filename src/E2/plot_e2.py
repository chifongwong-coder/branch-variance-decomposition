"""E2 plotter: reads results/e2_metrics.json (produced by e2_crossing_velocity.py)
and draws the four-panel crossing-path oracle figure. Run and plot are separated
so the figure can be regenerated from the saved data without re-running the
experiment, and so the figure carries no experiment-number title.

Usage:
    python plot_e2.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RES = HERE.parents[1] / "results"
FIG = HERE.parents[1] / "figures"


def main():
    with open(RES / "e2_metrics.json") as f:
        d = json.load(f)
    cfg = d["config"]
    curves = d["curves"]
    bw = d["bandwidth_at_t0.5"]
    av_cf_05 = d["av_cf_at_t0.5"]
    sp = d["sample_positions"]
    knn_k = cfg["knn_k"]

    t_arr = np.array([r["t"] for r in curves])
    av_hat = np.array([r["av_hat"] for r in curves])
    av_norm_hat = np.array([r["av_norm_hat"] for r in curves])
    R_v2_hat = np.array([r["R_v2_hat"] for r in curves])
    within_cf = np.array([r["within_cf"] for r in curves])
    between_cf = np.array([r["between_cf"] for r in curves])
    between_emp = np.array([r["between_emp"] for r in curves])
    r_switch_emp = np.array([r["r_switch_emp"] for r in curves])
    r_switch_cf = np.array([r["r_switch_cf"] for r in curves])
    H_cf = np.array([r["H_KY_cf"] for r in curves])

    plt.rcParams.update({"axes.labelsize": 14, "xtick.labelsize": 14,
                         "ytick.labelsize": 14, "axes.titlesize": 14})
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # Panel (a): sample positions at a few t (from saved positions).
    ax = axes[0, 0]
    for t_show in sp["t_show"]:
        y = np.array(sp["y"][f"{t_show}"])
        ax.scatter(y[:, 0], y[:, 1], s=4, alpha=0.5, label=f"t={t_show}")
    ax.set_aspect("equal")
    ax.set_title(r"(a) Sample positions $Y_t$ at endpoints and $t=0.5$")
    ax.set_xlabel("y0"); ax.set_ylabel("y1")
    ax.legend(loc="best", fontsize=9)

    # Panel (b): within + between (closed form, stacked) + empirical cross-check.
    ax = axes[0, 1]
    ax.fill_between(t_arr, 0, within_cf, alpha=0.55, color="tab:orange",
                    label=r"within-branch")
    ax.fill_between(t_arr, within_cf, within_cf + between_cf, alpha=0.55,
                    color="tab:blue", label=r"between-branch")
    ax.plot(t_arr, av_hat, "o-", color="black", lw=1.6,
            label=fr"empirical kNN total")
    ax.plot(t_arr, between_emp, "s--", color="darkblue", lw=1.2, alpha=0.85,
            label="empirical between")
    ax.set_xlabel("t"); ax.set_ylabel(r"$\mathcal{A}_v(t)$")
    ax.set_title(r"(b) Velocity ambiguity decomposition (within + between)")
    ax.legend(loc="best", fontsize=8); ax.grid(True, alpha=0.3)

    # Panel (c): normalized A_v + 1 - R_v^2 + H(K|Y_t)/log 2 + R_switch.
    ax = axes[1, 0]
    ax.plot(t_arr, av_norm_hat, "o-", lw=2, label=r"$\mathcal{A}_v / E\|U\|^2$")
    ax.plot(t_arr, 1 - R_v2_hat, "s--", lw=1.6, label=r"$1 - R_v^2$")
    ax.plot(t_arr, H_cf / np.log(2.0), "^:", lw=1.6,
            label=r"$H(K|Y_t)/\log 2$")
    ax.plot(t_arr, r_switch_emp, "v-", lw=2, color="tab:purple",
            label=r"$R_{\rm switch}$ (between/total)")
    ax.plot(t_arr, r_switch_cf, "--", color="tab:purple", lw=1.0, alpha=0.6,
            label=r"$R_{\rm switch}$ closed form")
    ax.set_xlabel("t"); ax.set_ylabel("normalized")
    ax.set_title("(c) Normalized ambiguity, $1{-}R_v^2$, $H$, $R_{\\rm switch}$")
    ax.legend(loc="best", fontsize=8); ax.grid(True, alpha=0.3)

    # Panel (d): bandwidth sensitivity at t=0.5, 3-seed mean +/- sample SD over
    # the full k-sweep (k=80 included), raw vs k/(k-1) finite-sample corrected.
    with open(RES / "e2_seed_stability_3seed.json") as f:
        ss = json.load(f)
    agg = ss["aggregate_across_seeds"]
    cf05 = ss["config"]["closed_form_av_t05"]
    ax = axes[1, 1]
    ks = [r["k"] for r in agg]
    raw_m = [r["raw_mean"] for r in agg]
    raw_sd = [r["raw_sd"] for r in agg]
    cor_m = [r["corrected_mean"] for r in agg]
    cor_sd = [r["corrected_sd"] for r in agg]
    ax.errorbar(ks, raw_m, yerr=raw_sd, fmt="o-", lw=2, capsize=3,
                color="tab:blue", label=r"raw $\widehat{\mathcal{A}}_v$")
    ax.errorbar(ks, cor_m, yerr=cor_sd, fmt="s-", lw=2, capsize=3,
                color="tab:green", label=r"$k/(k{-}1)$ corrected")
    ax.axhline(cf05, color="red", ls="--",
               label=f"closed form {cf05:.3f}")
    ax.set_xscale("log")
    ax.set_xlabel("k (kNN neighbors)")
    ax.set_ylabel(r"$\mathcal{A}_v(t{=}0.5)$")
    ax.set_title("(d) Bandwidth sensitivity at t=0.5")
    ax.legend(loc="lower right", fontsize=9)

    fig.tight_layout()
    FIG.mkdir(exist_ok=True)
    out = FIG / "e2_crossing_oracle.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
