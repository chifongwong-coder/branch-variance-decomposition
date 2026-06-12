"""C3 assignment-margin diagnostic.

Tests the assignment-margin theory:

  The C3 phase boundary at fixed lambda is controlled NOT by total
  geometry cost (D_G * sigma_s^2), but by the assignment margin

      m_i = d_i^same - d_i^diff

  where d_i^same = min_{j : K_X[j] = C[i]} ||Z[i] - X[j]||^2 and
        d_i^diff = min_{j : K_X[j] != C[i]} ||Z[i] - X[j]||^2.

  Negative m_i means geometry-only OT already prefers the matched target.
  Positive m_i means semantic penalty must overcome a margin of size m_i
  to flip the assignment to the matched (same-condition) target.

  Prediction:  fraction(m_i > lambda)  ~  C3@lambda mismatch rate
  (heuristic; Hungarian assignment is global, not row-wise nearest).

For each P4a cell at the 5 paper-canonical seeds, this script re-samples
(Z, X, C, K_X) via e3.sample_data (paper-canonical defaults), computes
the row-wise margin m_i, reports per-cell scalars, and produces the c3-margin figure
comparing fraction(m_i > lambda) to the measured C3@lambda mismatch from
the existing P4a JSON.

Output:
  results/c3_margin_analysis.json
  figures/phase_diagram_c3_margin.png
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "4")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.figure_style import apply_paper_style, Palette  # noqa: E402
import E3.e3_coupling_comparison as e3  # noqa: E402

apply_paper_style()
PAL = Palette()


# ---------------------------------------------------------------------------
# Config (paper-canonical defaults; --stage selection not needed here)
# ---------------------------------------------------------------------------

N_PAIRS = 10_000   # match P4a / P1 grid
SEEDS = [0, 1, 2, 3, 4]
LAMBDAS = [1.0, 10.0, 100.0]
SIGMA_S_DEFAULT = 1.0

# P4a cells (same 5)
CELLS = [(8, 0.1), (8, 2.0), (128, 0.1), (128, 2.0), (32, 0.5)]


# ---------------------------------------------------------------------------
# Row-wise margin computation
# ---------------------------------------------------------------------------

def compute_row_margins(Z: np.ndarray, X: np.ndarray,
                        C: np.ndarray, K_X: np.ndarray) -> np.ndarray:
    """For each source i, compute m_i = d_i^same - d_i^diff where
    d_i^same = min_j ||Z[i] - X[j]||^2 with K_X[j] = C[i]
    d_i^diff = min_j ||Z[i] - X[j]||^2 with K_X[j] != C[i]

    Returns m of shape (N,).
    """
    N = len(Z)
    m = np.zeros(N, dtype=np.float64)
    # Vectorize per-condition: for each C value k, compute pairwise dist
    # to all X with K_X = k and to all X with K_X != k.
    for k in (-1, 1):
        src_idx = np.where(C == k)[0]
        same_idx = np.where(K_X == k)[0]
        diff_idx = np.where(K_X != k)[0]
        if len(src_idx) == 0 or len(same_idx) == 0 or len(diff_idx) == 0:
            continue
        Z_src = Z[src_idx]               # (n_src, D)
        X_same = X[same_idx]             # (n_same, D)
        X_diff = X[diff_idx]             # (n_diff, D)
        # Compute pairwise squared distances in chunks to keep memory low
        chunk = 1024
        d_same = np.empty(len(src_idx), dtype=np.float64)
        d_diff = np.empty(len(src_idx), dtype=np.float64)
        for start in range(0, len(src_idx), chunk):
            end = min(start + chunk, len(src_idx))
            Zc = Z_src[start:end]        # (c, D)
            # ||Zc - X_same||^2 = ||Zc||^2 + ||X_same||^2 - 2 Zc @ X_same.T
            dd_same = ((Zc[:, None, :] - X_same[None, :, :]) ** 2).sum(-1)
            dd_diff = ((Zc[:, None, :] - X_diff[None, :, :]) ** 2).sum(-1)
            d_same[start:end] = dd_same.min(axis=1)
            d_diff[start:end] = dd_diff.min(axis=1)
        m[src_idx] = d_same - d_diff
    return m


# ---------------------------------------------------------------------------
# Per-cell driver (one cell, one seed)
# ---------------------------------------------------------------------------

def run_one_cell_one_seed(D_G: int, A: float, sigma_s: float,
                          seed: int, N: int) -> dict:
    """Re-sample (Z, X, C, K_X) at the given config and compute margin stats."""
    # Mutate e3 module globals (matches e_phase_diagram.py pattern)
    e3.D_G = int(D_G)
    e3.A = float(A)
    e3.SIGMA_S = float(sigma_s)
    e3.D = e3.D_G + 1
    e3.N_PAIRS = N

    Z, X, C, K_X = e3.sample_data(N, seed)
    m = compute_row_margins(Z.astype(np.float64), X.astype(np.float64),
                            C, K_X)

    return {
        "seed": int(seed),
        "n": int(len(m)),
        "median": float(np.median(m)),
        "mean": float(m.mean()),
        "q25": float(np.quantile(m, 0.25)),
        "q75": float(np.quantile(m, 0.75)),
        "q90": float(np.quantile(m, 0.90)),
        "q95": float(np.quantile(m, 0.95)),
        "fraction_pos": float((m > 0).mean()),  # frac with m_i > 0 (any penalty needed)
        "fraction_above_lambda": {
            str(int(lam)): float((m > lam).mean()) for lam in LAMBDAS
        },
        "raw_quantiles_for_hist": {
            "p1":  float(np.quantile(m, 0.01)),
            "p5":  float(np.quantile(m, 0.05)),
            "p99": float(np.quantile(m, 0.99)),
        },
    }


# ---------------------------------------------------------------------------
# Plot the c3-margin figure
# ---------------------------------------------------------------------------

def plot_c3_margin(margin_results: dict, p4a_path: Path, out_dir: Path,
              prefix: str = "phase_diagram"):
    """Two-panel c3-margin figure:
    Left:  fraction(m_i > lambda) vs lambda per cell (5 cells x 3 lambdas)
    Right: scatter of fraction(m_i > lambda) vs measured C3@lambda mismatch.
    """
    # Load P4a JSON for measured mismatch
    with open(p4a_path) as f:
        p4a = json.load(f)

    # Build (D_G, A) -> measured C3@lambda mismatch (mean over seeds)
    measured = {}
    for cell in p4a["cells"]:
        key = (cell["D_G"], cell["A"])
        by_c = {}
        for r in cell["runs"]:
            by_c.setdefault(r["coupling"], []).append(r)
        per_lam = {}
        for lam in LAMBDAS:
            tag = f"C3@{int(lam)}"
            if tag in by_c:
                mis = np.mean([r["pair_metrics"]["mismatch_rate"]
                               for r in by_c[tag]])
                per_lam[int(lam)] = float(mis)
        measured[key] = per_lam

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.5))

    # ---- Left: fraction(m_i > lambda) curves per cell ----
    cell_colors = plt.get_cmap("tab10")
    for ci, ((D_G, A), per_seed) in enumerate(margin_results.items()):
        lsnr = np.log10(A * A / (D_G * SIGMA_S_DEFAULT ** 2))
        # Aggregate over seeds
        frac_per_lam = []
        frac_sd_per_lam = []
        for lam in LAMBDAS:
            vals = [s["fraction_above_lambda"][str(int(lam))]
                    for s in per_seed]
            frac_per_lam.append(np.mean(vals))
            frac_sd_per_lam.append(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
        axL.errorbar(LAMBDAS, frac_per_lam, yerr=frac_sd_per_lam,
                     fmt="o-", ms=6, capsize=2.5,
                     color=cell_colors(ci),
                     label=fr"$D_G={D_G}, a={A:g}$ (lsnr$={lsnr:+.2f}$)",
                     lw=1.4)
    axL.set_xscale("log")
    axL.set_xlabel(r"$\lambda$")
    axL.set_ylabel(r"fraction $(m_i > \lambda)$ over source pool")
    axL.set_title(r"(a) Fraction of sources with margin exceeding $\lambda$")
    axL.set_ylim(-0.02, 0.55)
    axL.axhline(0.5, color="gray", ls=":", lw=0.7, alpha=0.4)
    axL.legend(loc="best", fontsize=7)

    # ---- Right: scatter of fraction vs measured mismatch ----
    all_pred = []
    all_meas = []
    for ci, ((D_G, A), per_seed) in enumerate(margin_results.items()):
        if (D_G, A) not in measured:
            continue
        for lam in LAMBDAS:
            vals = [s["fraction_above_lambda"][str(int(lam))]
                    for s in per_seed]
            pred = float(np.mean(vals))
            meas = measured[(D_G, A)].get(int(lam), float("nan"))
            axR.scatter(pred, meas, s=60, color=cell_colors(ci),
                        alpha=0.85,
                        label=(fr"$D_G={D_G}, a={A:g}$"
                               if lam == LAMBDAS[0] else None),
                        edgecolor="black", linewidth=0.4)
            axR.annotate(fr"$\lambda={int(lam)}$",
                         xy=(pred, meas), xytext=(4, 4),
                         textcoords="offset points",
                         fontsize=6, alpha=0.7)
            all_pred.append(pred)
            all_meas.append(meas)
    # y = x reference
    if all_pred:
        lo = -0.02
        hi = max(max(all_pred), max(all_meas)) * 1.10
        ref = np.linspace(lo, hi, 50)
        axR.plot(ref, ref, ls="--", color="gray", lw=1.0,
                 label=r"$y = x$ (perfect proxy)")
    # quantify the relationship
    arr_p = np.array(all_pred); arr_m = np.array(all_meas)
    mask = np.isfinite(arr_p) & np.isfinite(arr_m)
    if mask.sum() >= 3:
        rho = np.corrcoef(arr_p[mask], arr_m[mask])[0, 1]
        axR.text(0.04, 0.96, fr"Pearson $\rho = {rho:+.3f}$",
                 transform=axR.transAxes, fontsize=8, va="top",
                 bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="white", edgecolor="gray", lw=0.5))
    axR.set_xlabel(r"fraction $(m_i > \lambda)$ (row-wise proxy)")
    axR.set_ylabel(r"measured mismatch at $C_3@\lambda$")
    axR.set_title(r"(b) Row-wise margin proxy vs measured Hungarian mismatch")
    axR.legend(loc="best", fontsize=7)

    fig.tight_layout()
    out = out_dir / f"{prefix}_c3_margin.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    here = Path(__file__).resolve().parents[2]
    out_dir = here / "figures"
    out_dir.mkdir(exist_ok=True)
    res_path = here / "results" / "c3_margin_analysis.json"
    p4a_path = here / "results" / "phase_diagram_lambda_sweep.json"

    print(f"C3 assignment-margin diagnostic")
    print(f"  cells: {CELLS}")
    print(f"  seeds: {SEEDS}")
    print(f"  N_PAIRS: {N_PAIRS}")
    print(f"  lambdas tested: {LAMBDAS}")
    print(f"  sigma_s: {SIGMA_S_DEFAULT}")
    print("-" * 78)

    margin_results = {}
    t0 = time.time()
    for D_G, A in CELLS:
        lsnr = np.log10(A * A / (D_G * SIGMA_S_DEFAULT ** 2))
        print(f"\n--- cell (D_G={D_G}, a={A}) lsnr={lsnr:+.2f}")
        per_seed = []
        for seed in SEEDS:
            t1 = time.time()
            stats = run_one_cell_one_seed(D_G, A, SIGMA_S_DEFAULT,
                                           seed, N_PAIRS)
            per_seed.append(stats)
            wall = time.time() - t1
            print(f"  seed={seed}  median(m)={stats['median']:+.3f}  "
                  f"frac(>1)={stats['fraction_above_lambda']['1']:.3f}  "
                  f"frac(>10)={stats['fraction_above_lambda']['10']:.3f}  "
                  f"frac(>100)={stats['fraction_above_lambda']['100']:.3f}  "
                  f"({wall:.1f}s)")
        margin_results[(D_G, A)] = per_seed
    print(f"\nTotal wall: {time.time() - t0:.1f}s")

    # Save JSON
    payload = {
        "config": {
            "cells": CELLS,
            "seeds": SEEDS,
            "lambdas": LAMBDAS,
            "N_PAIRS": N_PAIRS,
            "sigma_s": SIGMA_S_DEFAULT,
        },
        "per_cell": {
            f"DG{D_G}_a{A:g}": per_seed
            for (D_G, A), per_seed in margin_results.items()
        },
    }
    with open(res_path, "w") as f:
        json.dump(payload, f, indent=2, default=float)
    print(f"wrote {res_path}")

    # Plot
    plot_c3_margin(margin_results, p4a_path, out_dir)


if __name__ == "__main__":
    main()
