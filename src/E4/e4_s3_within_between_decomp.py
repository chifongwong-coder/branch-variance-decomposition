"""E4-S3: Oracle within/between decomposition of A_v(t) for the E4 setting.

No training. Pure closed-form + Monte Carlo over the marginal of Y_t.

Setting (matches E4 / E0)
  X | K=k ~ N(k m, s^2),  K in {-1,+1} equal prior, m=2.0, s=0.25.
  Z ~ N(0,1)  (independent coupling).
  Y_t = (1-t) Z + t X,  U = X - Z.

Closed form
  tau^2(t)        = (1-t)^2 + t^2 s^2
  Var(U|Y_t, K)   = s^2 / tau^2(t)            (per dim, indep of y, k)
  Var_K|Y(E[U|Y_t, K])   = m^2 (1-t)^2 / tau^4(t) * sech^2(t m Y_t / tau^2)
  A_within(t)     = s^2 / tau^2(t)
  A_between(t)    = m^2 (1-t)^2 / tau^4(t) * E_{Y_t}[sech^2(t m Y_t / tau^2)]
  A_v(t)          = A_within(t) + A_between(t)
  R_switch(t)     = A_between(t) / A_v(t)

Outputs
  figures/e4_s3_decomp.png
  results/e4_s3_metrics.json
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

M = 2.0
S = 0.25
S2 = S ** 2

T_GRID = np.concatenate([
    np.array([0.01, 0.025]),
    np.linspace(0.05, 0.95, 19),
    np.array([0.975, 0.99]),
])
N_MC = 1_000_000
SEED = 7


def tau2(t):
    return (1.0 - t) ** 2 + (t ** 2) * S2


def decomp_at_t(t, n_mc=N_MC, seed=SEED):
    tt = tau2(t)
    within = S2 / tt
    rng = np.random.default_rng(seed + int(10000 * t))
    n = n_mc
    k = rng.choice([-1, 1], size=n)
    z = rng.standard_normal(n)
    x = k * M + S * rng.standard_normal(n)
    y = (1.0 - t) * z + t * x
    sech2 = 1.0 / np.cosh(t * M * y / tt) ** 2
    between = ((1.0 - t) ** 2 * (M ** 2) / (tt ** 2)) * sech2.mean()
    a_v = within + between
    r_switch = between / max(a_v, 1e-12)
    # bootstrap SE on between (the noisy part)
    b_se = float(
        (((1.0 - t) ** 2 * (M ** 2) / (tt ** 2))
         * (sech2.std(ddof=1) / np.sqrt(n))))
    return {
        "t": float(t),
        "tau2": float(tt),
        "within": float(within),
        "between": float(between),
        "between_se": b_se,
        "a_v": float(a_v),
        "r_switch": float(r_switch),
    }


def run():
    rows = [decomp_at_t(float(t)) for t in T_GRID]

    out_dir = Path(__file__).resolve().parents[2]
    fig_dir = out_dir / "figures"
    res_dir = out_dir / "results"
    fig_dir.mkdir(exist_ok=True)
    res_dir.mkdir(exist_ok=True)

    t_arr = np.array([r["t"] for r in rows])
    within = np.array([r["within"] for r in rows])
    between = np.array([r["between"] for r in rows])
    a_v = np.array([r["a_v"] for r in rows])
    r_switch = np.array([r["r_switch"] for r in rows])

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # Panel 1: stacked decomposition
    ax = axes[0]
    ax.fill_between(t_arr, 0.0, within,
                    alpha=0.6, color="tab:orange",
                    label=r"$\mathcal{A}_{\rm within}(t) = s^2/\tau^2(t)$")
    ax.fill_between(t_arr, within, within + between,
                    alpha=0.6, color="tab:blue",
                    label=r"$\mathcal{A}_{\rm between}(t)$")
    ax.plot(t_arr, a_v, "ko-", lw=1.2, ms=3,
            label=r"$\mathcal{A}_v(t) = $ total")
    ax.set_xlabel("t")
    ax.set_ylabel(r"$\mathcal{A}_v(t)$")
    ax.set_title(
        rf"(a) Oracle within / between decomposition "
        rf"(1D binary Gaussian, m={M}, s={S})")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Panel 2 (R_switch(t)): relative contribution of between
    ax = axes[1]
    ax.plot(t_arr, r_switch, "v-", lw=2, color="tab:purple",
            label=r"$R_{\rm switch}(t) = \mathcal{A}_{\rm between}/\mathcal{A}_v$")
    ax.axhline(0.5, color="gray", ls=":", alpha=0.5)
    ax.set_xlabel("t")
    ax.set_ylabel(r"$R_{\rm switch}(t)$")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(r"(b) Semantic switching ratio $R_{\rm switch}(t)$")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(fig_dir / "e4_s3_decomp.png", dpi=130)
    plt.close(fig)

    # Key indicators
    idx_peak_av = int(np.argmax(a_v))
    idx_min_av = int(np.argmin(a_v))
    idx_peak_rsw = int(np.argmax(r_switch))
    summary = {
        "config": {"M": M, "S": S, "T_GRID": t_arr.tolist(),
                   "N_MC": N_MC, "SEED": SEED},
        "curves": rows,
        "key_indicators": {
            "A_v(0+)": float(a_v[0]),
            "A_v(1-)": float(a_v[-1]),
            "argmax_t_A_v": float(t_arr[idx_peak_av]),
            "max_A_v": float(a_v[idx_peak_av]),
            "argmin_t_A_v": float(t_arr[idx_min_av]),
            "min_A_v": float(a_v[idx_min_av]),
            "argmax_t_R_switch": float(t_arr[idx_peak_rsw]),
            "max_R_switch": float(r_switch[idx_peak_rsw]),
            "R_switch_at_endpoints": [float(r_switch[0]), float(r_switch[-1])],
        },
    }
    with open(res_dir / "e4_s3_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("E4-S3 oracle within/between decomposition")
    print("-" * 60)
    print(f"  config: m={M}, s={S}, N_MC={N_MC}, T_grid={len(t_arr)} points")
    print(f"  A_v range  : {a_v.min():.4f} (t={t_arr[idx_min_av]:.3f}) "
          f"-> {a_v.max():.4f} (t={t_arr[idx_peak_av]:.3f})")
    print(f"  A_v at 0+/1- (t={t_arr[0]:.3f}/{t_arr[-1]:.3f}): "
          f"{a_v[0]:.4f} / {a_v[-1]:.4f}")
    print(f"  R_switch range: {r_switch.min():.4f} (t={t_arr[idx_min_av]:.3f}) "
          f"-> {r_switch.max():.4f} (t={t_arr[idx_peak_rsw]:.3f})")
    print(f"  R_switch endpoints: {r_switch[0]:.4f} / {r_switch[-1]:.4f}")
    print("  Selected t values:")
    for r in rows[::3]:
        print(f"    t={r['t']:.3f}  within={r['within']:.4f}  "
              f"between={r['between']:.4f}  A_v={r['a_v']:.4f}  "
              f"R_switch={r['r_switch']:.4f}")
    print(f"  figures -> {fig_dir / 'e4_s3_decomp.png'}")
    print(f"  metrics -> {res_dir / 'e4_s3_metrics.json'}")


if __name__ == "__main__":
    run()
