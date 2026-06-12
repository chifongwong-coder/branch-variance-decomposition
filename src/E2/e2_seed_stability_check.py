"""E2 seed-stability check (3 seeds). Supplements e2_crossing_velocity.py.

The headline E2 verification (biased plug-in 2.0150, closed-form 2.04,
post-correction 0.025% relative error) is computed at a single seed
(SEED=0), so the 0.025% residual has no quoted uncertainty.

This sister script runs the same bandwidth sweep at t=0.5 across SEEDS=(0,1,2)
and reports per-seed raw A_v plus mean/SD, both raw and after the k/(k-1)
finite-sample correction. It DOES NOT touch e2_crossing_velocity.py, the
existing e2_metrics.json, or any other E2 artifact.

Output: results/e2_seed_stability_3seed.json (distinct filename, no
collision with the canonical e2_metrics.json).

Usage:
    python3 e2_seed_stability_check.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

# Reuse the canonical E2 config + helpers without touching the source file.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from E2.e2_crossing_velocity import (  # noqa: E402
    KNN_K,             # headline bandwidth = 80; not in the canonical sweep
    KNN_K_SWEEP,       # bandwidth-sensitivity grid (10, 30, 50, 100, 200, 400)
    SIGMA_B,
    sample_pairs,
    interpolate,
    velocity,
    closed_form_av,
)

SEEDS = (0, 1, 2)
T_CENTER = 0.5
# Include the headline k=80 so the canonical 2.0150 number's seed-SD is directly reported.
EVAL_KS = tuple(sorted(set(list(KNN_K_SWEEP) + [KNN_K])))
OUT_JSON = Path(__file__).resolve().parents[2] / "results" / "e2_seed_stability_3seed.json"


def knn_local_av_only(y, u, k):
    """Biased 1/k local trace-cov estimate at every empirical query point.

    Matches the convention used in e2_crossing_velocity.knn_local_av_and_explained
    so the seed=0 row of this script reproduces the canonical e2_metrics.json
    bandwidth_sweep values exactly.
    """
    tree = cKDTree(y)
    _, idxs = tree.query(y, k=k)
    U = u[idxs]                       # (N, k, d)
    mu = U.mean(axis=1, keepdims=True)
    diff = U - mu
    trace_cov = (diff ** 2).sum(axis=(1, 2)) / k
    return float(trace_cov.mean())


def main():
    av_cf_t05, _, _, _ = closed_form_av(T_CENTER)
    print(f"closed-form A_v(t=1/2)  = {av_cf_t05:.6f}")
    print(f"sigma_b                 = {SIGMA_B}")
    print(f"seeds                   = {SEEDS}")
    print(f"bandwidth sweep         = {EVAL_KS}")
    print()

    # per-seed: list of bandwidth dicts
    per_seed_rows = []
    print(f"{'seed':>5} {'k':>5} {'raw':>10} {'corrected':>10} {'rel_err_%':>10}")
    print("-" * 50)
    for seed in SEEDS:
        z, x, _ = sample_pairs(seed=seed)
        y_c = interpolate(z, x, T_CENTER)
        u_c = velocity(z, x)
        bandwidth_rows = []
        for k in EVAL_KS:
            raw = knn_local_av_only(y_c, u_c, k=k)
            corr = raw * k / (k - 1)
            rel_err_pct = 100.0 * (corr - av_cf_t05) / av_cf_t05
            bandwidth_rows.append({
                "k": int(k),
                "raw_av_hat": raw,
                "corrected_av_hat": corr,
                "rel_err_vs_cf_pct": rel_err_pct,
            })
            print(f"{seed:>5d} {k:>5d} {raw:>10.6f} {corr:>10.6f} {rel_err_pct:>+10.4f}")
        per_seed_rows.append({
            "seed": int(seed),
            "bandwidths": bandwidth_rows,
        })

    print()
    print("seed-aggregate (mean +/- sample SD across seeds, per k):")
    print(f"{'k':>5} {'raw_mean':>10} {'raw_sd':>10} "
          f"{'corr_mean':>10} {'corr_sd':>10} {'relerr_mean_%':>14} {'relerr_sd_%':>12}")
    aggregate = []
    for ki, k in enumerate(EVAL_KS):
        raws = np.array([per_seed_rows[si]["bandwidths"][ki]["raw_av_hat"]
                         for si in range(len(SEEDS))])
        corrs = np.array([per_seed_rows[si]["bandwidths"][ki]["corrected_av_hat"]
                          for si in range(len(SEEDS))])
        relerrs = np.array([per_seed_rows[si]["bandwidths"][ki]["rel_err_vs_cf_pct"]
                            for si in range(len(SEEDS))])
        raw_mean, raw_sd = float(raws.mean()), float(raws.std(ddof=1))
        corr_mean, corr_sd = float(corrs.mean()), float(corrs.std(ddof=1))
        relerr_mean, relerr_sd = float(relerrs.mean()), float(relerrs.std(ddof=1))
        aggregate.append({
            "k": int(k),
            "raw_mean": raw_mean,
            "raw_sd": raw_sd,
            "corrected_mean": corr_mean,
            "corrected_sd": corr_sd,
            "rel_err_mean_pct": relerr_mean,
            "rel_err_sd_pct": relerr_sd,
        })
        print(f"{k:>5d} {raw_mean:>10.6f} {raw_sd:>10.6f} "
              f"{corr_mean:>10.6f} {corr_sd:>10.6f} "
              f"{relerr_mean:>+14.4f} {relerr_sd:>12.4f}")

    payload = {
        "purpose": (
            "Seed-stability addendum to e2_crossing_velocity.py / e2_metrics.json. "
            "Does NOT replace the canonical "
            "E2 result; provides the missing seed-SD bar on the 0.025% claim."
        ),
        "config": {
            "seeds": list(SEEDS),
            "t_center": T_CENTER,
            "sigma_b": SIGMA_B,
            "n_per_branch": 60_000,
            "n_total": 120_000,
            "knn_k_sweep": list(EVAL_KS),
            "closed_form_av_t05": av_cf_t05,
        },
        "per_seed": per_seed_rows,
        "aggregate_across_seeds": aggregate,
    }
    OUT_JSON.parent.mkdir(exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
