"""E7: plot the per-t branch-variance decomposition profile across couplings.

Reads results/e7_diag_<coupling>_seed<seed>.json for the couplings/seeds
present and produces:
  figures/e7_bvd_profile.png   per-t A_v / A_within / A_between / R_switch

Couplings without a diagnostics JSON are skipped. Aggregates mean +/- SD over
whatever seeds are present.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.figure_style import apply_paper_style  # noqa: E402
apply_paper_style()

HERE = Path(__file__).resolve().parent
RES = HERE.parents[1] / "results"
FIG = HERE.parents[1] / "figures"
FIG.mkdir(parents=True, exist_ok=True)

COUPLINGS = ["c0", "c1", "c3", "c3inf"]
LABEL = {"c0": "C0 independent", "c1": "C1 Euclidean OT",
         "c3": "C3 semantic-cost OT", "c3inf": "C3-inf class-conditional OT"}
COLOR = {"c0": "tab:gray", "c1": "tab:blue",
         "c3": "tab:orange", "c3inf": "tab:green"}


def load_all():
    """Load the canonical (final-checkpoint) diagnostics per coupling. The
    step-tagged saturation files (e7_diag_<cp>_seed<s>_step<N>.json) are skipped
    so the per-coupling mean/SD aggregates over seeds, not over training steps."""
    runs = defaultdict(list)
    for j in sorted(RES.glob("e7_diag_*.json")):
        if "_step" in j.stem:
            continue
        d = json.load(open(j))
        runs[d["coupling"]].append(d)
    return runs


def plot_bvd_profile(runs):
    metrics = ["A_v", "A_within", "A_between", "R_switch"]
    titles = [r"$A_v(t)$", r"$A_{\rm within}(t)$",
              r"$A_{\rm between}(t)$", r"$R_{\rm switch}(t)$"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, m, ttl in zip(axes.flat, metrics, titles):
        for cp in COUPLINGS:
            if cp not in runs:
                continue
            t = np.array(runs[cp][0]["bvd"]["t_grid"])
            ys = np.array([r["bvd"][m] for r in runs[cp]])  # (n_seed, n_t)
            mean = ys.mean(0)
            ax.plot(t, mean, "o-", ms=3, color=COLOR[cp], label=LABEL[cp])
            if ys.shape[0] > 1:
                sd = ys.std(0, ddof=1)
                ax.fill_between(t, mean - sd, mean + sd, color=COLOR[cp], alpha=0.2)
        ax.set_xlabel("t")
        ax.set_title(ttl)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=8.5,
               bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("E7 CIFAR-10 FM: branch-variance decomposition by coupling "
                 "(CLIP feature space)", y=0.95)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(FIG / "e7_bvd_profile.png", dpi=200)
    plt.close(fig)
    print(f"  wrote {FIG / 'e7_bvd_profile.png'}")


def main():
    runs = load_all()
    if not runs:
        print("no e7_diag_*.json found in results/; run the diagnostics first.")
        return
    print(f"loaded couplings: {sorted(runs)} "
          f"({sum(len(v) for v in runs.values())} runs)")
    # report A_v / A_within alongside A_between (their joint
    # coincidence across couplings is the artifact fingerprint).
    print(f"\n{'coupling':8} {'maxA_v':>8} {'maxA_within':>12} {'maxA_between':>13} "
          f"{'transport':>10}")
    for cp in COUPLINGS:
        if cp not in runs:
            continue
        r = runs[cp][0]
        b = r["bvd"]
        print(f"{cp:8} {max(b['A_v']):>8.4f} {max(b['A_within']):>12.4f} "
              f"{max(b['A_between']):>13.4f} {r['mean_transport_cost']:>10.1f}")
    print()
    plot_bvd_profile(runs)


if __name__ == "__main__":
    main()
