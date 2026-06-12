"""Build a single composite figure that summarises E4-S1 (under-training
control) + E4-S2 (endpoint sanity) in two panels.

Reads metrics JSONs produced by e4_fm_mlp_fixed_t.py for the three tags:
  - conservative      (hidden in {64,128,256}, STEPS=5000, full t grid 0.2-0.8)
  - undertraining_h256 (hidden=256, STEPS=15000, full t grid 0.2-0.8)
  - endpoints         (hidden in {64,128,256}, STEPS=5000, t in {0.05, 0.95})
plus the oracle decomposition JSON for the closed-form A_v(t) curve:
  - e4_s3_metrics.json
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

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
OUT = Path(__file__).resolve().parents[2] / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def load(name):
    with open(RES / name) as f:
        return json.load(f)


def mean_over_seeds(runs, t, hidden, key):
    vals = [r[key] for r in runs if r["t"] == t and r["hidden"] == hidden]
    return float(np.mean(vals)), float(np.std(vals, ddof=1))


def build():
    cons = load("e4_metrics_conservative.json")
    long = load("e4_metrics_undertraining_h256.json")
    ends = load("e4_metrics_endpoints.json")
    s3 = load("e4_s3_metrics.json")

    # ------------------------------------------------------------------
    # Panel A data: S1 under-training control
    # ------------------------------------------------------------------
    t_main = [0.2, 0.4, 0.5, 0.6, 0.8]
    # approx error: hidden=64 @ 5k, hidden=256 @ 5k, hidden=256 @ 15k
    approx_h64_5k = [mean_over_seeds(cons["runs"], t, 64, "approx_eval_final")
                     for t in t_main]
    approx_h256_5k = [mean_over_seeds(cons["runs"], t, 256, "approx_eval_final")
                      for t in t_main]
    approx_h256_15k = [mean_over_seeds(long["runs"], t, 256, "approx_eval_final")
                       for t in t_main]

    # ------------------------------------------------------------------
    # Panel B data: closed-form A_v(t) + L_eval markers across full t grid
    # ------------------------------------------------------------------
    cf_t = np.array([r["t"] for r in s3["curves"]])
    cf_av = np.array([r["a_v"] for r in s3["curves"]])

    # L_eval at hidden=64 only (closest to the saturation point)
    L_main = [mean_over_seeds(cons["runs"], t, 64, "L_eval_final")
              for t in t_main]
    L_ends = [mean_over_seeds(ends["runs"], t, 64, "L_eval_final")
              for t in [0.05, 0.95]]

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ----- Panel A: S1 grouped bar chart of approx error -----
    ax = axes[0]
    x = np.arange(len(t_main))
    bw = 0.27
    a64_means = [m for m, _ in approx_h64_5k]
    a64_stds = [s for _, s in approx_h64_5k]
    a256_5k_means = [m for m, _ in approx_h256_5k]
    a256_5k_stds = [s for _, s in approx_h256_5k]
    a256_15k_means = [m for m, _ in approx_h256_15k]
    a256_15k_stds = [s for _, s in approx_h256_15k]

    ax.bar(x - bw, a64_means, bw, yerr=a64_stds, capsize=3,
           color="tab:green", label="h=64, 5k steps")
    ax.bar(x, a256_5k_means, bw, yerr=a256_5k_stds, capsize=3,
           color="tab:orange", label="h=256, 5k steps")
    ax.bar(x + bw, a256_15k_means, bw, yerr=a256_15k_stds, capsize=3,
           color="tab:blue", label="h=256, 15k steps")
    ax.set_xticks(x)
    ax.set_xticklabels([f"t={t}" for t in t_main])
    ax.set_yscale("log")
    ax.set_ylabel(r"approximation error $\,\|\,v_\theta - v^*\|^2$  (log)")
    ax.set_title("(a) under-training control "
                 r"(larger MLP, 3$\times$ training steps)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3, which="both", axis="y")

    # ----- Panel B: A_v(t) + L_eval markers, full t grid -----
    ax = axes[1]
    ax.plot(cf_t, cf_av, "k-", lw=2,
            label=r"closed-form $\mathcal{A}_v(t)$ (oracle)")
    # L_eval markers at conservative t values (hidden=64)
    Lm_means = [m for m, _ in L_main]
    Lm_stds = [s for _, s in L_main]
    Le_means = [m for m, _ in L_ends]
    Le_stds = [s for _, s in L_ends]
    ax.errorbar(t_main, Lm_means, yerr=Lm_stds, fmt="o", color="tab:blue",
                ms=8, capsize=4,
                label=r"$L_{\rm eval}$ at h=64 (conservative t)")
    ax.errorbar([0.05, 0.95], Le_means, yerr=Le_stds, fmt="D", color="tab:red",
                ms=8, capsize=4,
                label=r"$L_{\rm eval}$ at h=64 (endpoints)")
    ax.set_xlabel("t")
    ax.set_ylabel(r"loss / $\mathcal{A}_v(t)$")
    ax.set_title(r"(b) endpoint sanity + saturation of $L_{\rm eval}$ "
                 r"to $\mathcal{A}_v(t)$ across full t-grid")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = OUT / "e4_supplement.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out}")

    # Quick numerical readout
    print("\nS1 under-training bar chart (means):")
    for i, t in enumerate(t_main):
        print(f"  t={t}: h64@5k={a64_means[i]:.4f}  "
              f"h256@5k={a256_5k_means[i]:.4f}  "
              f"h256@15k={a256_15k_means[i]:.4f}  "
              f"ratio(5k/15k)={a256_5k_means[i]/max(a256_15k_means[i],1e-9):.2f}x")
    print("\nS2 endpoint markers (h=64):")
    for t, m in zip([0.05, 0.95], L_ends):
        cf_at_t = float(np.interp(t, cf_t, cf_av))
        print(f"  t={t}: L_eval={m[0]:.4f}  CF A_v(t)~={cf_at_t:.4f}  "
              f"diff_pct={(m[0]-cf_at_t)/cf_at_t*100:+.2f}%")


if __name__ == "__main__":
    build()
