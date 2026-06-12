"""Phase diagram driver for (D_G, a, sigma_s) sweep.

Reuses the E3 primitives in
`e3_coupling_comparison.py` by mutating its module-level constants per
cell (D_G, A, SIGMA_S, D, N_PAIRS, K_LIST, K_MAX). Each cell is
independent; mutation is single-threaded so there is no race.

Stages
  P0  smoke         4 scientific-anchor cells, 1 seed, t=0.5            ~5 min
  P1  main          5x5=25 cell grid, 5 seeds, t=0.5                    ~5-6 h
  P2  sentinel      5 t-pts on 5 corner+headline cells (post-P1)        ~30 min
  P3  refinement    4-6 cells around mismatch=0.25 (cond. on P1)        ~2 h
  P4a corner lam    5 cells x lambda {1,10,100}                         ~10 min
  P4b sigma_s       4 cells x sigma_s {0.5, 2.0}                        ~30 min

Couplings per cell (4 essential): C0 independent, C1 Euclidean OT,
C3 semantic-cost OT @lambda=10, C4 geometry-only OT.

Paper-canonical defaults are hard-coded; --stage selects the run, no other
behavior is env-controlled.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Thread caps: must be set before numpy/scipy/sklearn import.
# ---------------------------------------------------------------------------
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_k, "4")

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np

import E3.e3_coupling_comparison as e3

try:
    import psutil
    _HAVE_PSUTIL = True
except ImportError:
    _HAVE_PSUTIL = False


# ---------------------------------------------------------------------------
# Paper-canonical settings (fixed for reproducibility)
# ---------------------------------------------------------------------------

N_PAIRS = 10_000          # halved from E3
K = 80                    # paper-canonical bandwidth, biased 1/k
BATCH_SIZE_OT = 1024
SIGMA_S_DEFAULT = 1.0
LAMBDA_C3 = 10.0

# Main grid axes
D_G_GRID = [8, 16, 32, 64, 128]
A_GRID = [0.1, 0.3, 0.5, 1.0, 2.0]

# Couplings (4 essential)
COUPLING_TAGS = ["C0", "C1", "C3@10", "C4"]

# Sentinel t-profile (Stage P2): 5 t-pts on 5 cells
SENTINEL_T = [0.1, 0.3, 0.5, 0.7, 0.9]
SENTINEL_CELLS = [
    (32, 0.5),    # headline
    (8,  0.1),    # low-SNR corner
    (8,  2.0),    # high-SNR/low-D corner
    (128, 0.1),   # low-SNR/high-D corner
    (128, 2.0),   # high-a/high-D corner
]

# Stage P0 smoke cells: 2 scientific anchors + 1 headline + 1 marginal corner
SMOKE_CELLS = [
    (16, 1.0, "high_snr_reachable"),    # expect mismatch ~ 0 for C1
    (64, 0.3, "low_snr_failure"),       # expect mismatch -> 0.5 for C1
    (32, 0.5, "headline"),              # E3 canonical cell
    (8,  0.1, "marginal_sanity"),       # low-D low-a corner
]

# Stage P4a corner lambda sweep cells (+ values)
P4A_LAMBDAS = [1.0, 10.0, 100.0]
P4A_CELLS = [
    (8,   0.1),
    (8,   2.0),
    (128, 0.1),
    (128, 2.0),
    (32,  0.5),
]

# Stage P4b sigma_s sanity
P4B_CELLS = [
    (8,   0.1, 0.5),
    (8,   0.1, 2.0),
    (128, 2.0, 0.5),
    (128, 2.0, 2.0),
]


# ---------------------------------------------------------------------------
# Cell-level config
# ---------------------------------------------------------------------------

def configure_cell(D_G: int, A: float, sigma_s: float):
    """Mutate e3 module globals for a single cell. Cells run serially,
    so the mutation is safe."""
    e3.D_G = int(D_G)
    e3.A = float(A)
    e3.SIGMA_S = float(sigma_s)
    e3.D = e3.D_G + 1
    e3.N_PAIRS = N_PAIRS
    e3.K_LIST = [K]
    e3.K_MAX = K
    e3.BATCH_SIZE_OT = BATCH_SIZE_OT


def make_coupling(tag: str, seed: int, cell_idx: int):
    """Returns a callable (Z, X, C, K_X, rng) -> (Zp, Xp, Cp, KXp)."""
    if tag == "C0":
        return e3.couple_independent
    if tag == "C1":
        return e3.couple_euclidean_ot
    if tag == "C4":
        return e3.couple_geometry_only_ot
    if tag.startswith("C3@"):
        lam = float(tag[len("C3@"):])
        return (lambda Z, X, C, K_X, rng, _lam=lam:
                e3.couple_semantic_cost_ot(Z, X, C, K_X, rng, _lam))
    raise ValueError(f"unknown coupling tag {tag!r}")


def _rss_mb() -> float:
    if _HAVE_PSUTIL:
        return psutil.Process().memory_info().rss / (1024 ** 2)
    return float("nan")


# ---------------------------------------------------------------------------
# Single cell runner
# ---------------------------------------------------------------------------

def run_cell(D_G: int, A: float, sigma_s: float,
             seeds: list[int], coupling_tags: list[str],
             t_values: list[float],
             cell_label: str = "") -> dict:
    """Run all (coupling x seed x t) combinations for one phase-diagram cell.

    Returns a dict with cell config, per-coupling/seed/t metrics, and timing.
    """
    configure_cell(D_G, A, sigma_s)
    lsnr = float(np.log10(A * A / (D_G * sigma_s * sigma_s)))
    print(f"\n--- cell (D_G={D_G}, a={A}, sigma_s={sigma_s}) "
          f"log10(a^2/(D_G*sigma_s^2)) = {lsnr:+.3f}  [{cell_label}]")

    cell = {
        "D_G": int(D_G), "A": float(A), "sigma_s": float(sigma_s),
        "lsnr": lsnr, "label": cell_label,
        "N_PAIRS": N_PAIRS, "K": K,
        "seeds": list(seeds), "couplings": list(coupling_tags),
        "t_values": list(t_values),
        "runs": [],
    }

    t_cell_start = time.time()
    for seed in seeds:
        Z, X, C, K_X = e3.sample_data(N_PAIRS, seed)
        for ci, tag in enumerate(coupling_tags):
            t_run = time.time()
            rng_c = np.random.default_rng(seed * 1000 + ci + 7)
            fn = make_coupling(tag, seed, ci)
            Zp, Xp, Cp, KXp = fn(Z, X, C, K_X, rng_c)
            pm = e3.pair_metrics(Zp, Xp, Cp, KXp)

            per_t = []
            for t in t_values:
                tm = e3.estimate_one_t(Zp, Xp, Cp, KXp, float(t))
                per_t.append(tm)
                gc.collect()
            wall = time.time() - t_run

            # Headline scalar extraction at t closest to 0.5, k=K
            mid_idx = int(np.argmin(np.abs(np.array(t_values) - 0.5)))
            mid_tm = per_t[mid_idx]
            cond_avg = mid_tm.get("cond_avg", {})
            if K in cond_avg:
                sem_h = cond_avg[K]["sem"]
            else:
                sem_h = {"A_v_norm": float("nan"), "R_switch": float("nan")}

            cell["runs"].append({
                "seed": int(seed),
                "coupling": tag,
                "wall_s": wall,
                "pair_metrics": pm,
                "metrics_per_t": per_t,
            })
            print(f"   seed={seed} {tag:6s} mismatch={pm['mismatch_rate']:.3f} "
                  f"A_v_sem(t=0.5|C)={sem_h.get('A_v_norm', float('nan')):.3f} "
                  f"R_sw(t=0.5|C)={sem_h.get('R_switch', float('nan')):.3f} "
                  f"({wall:.1f}s)")

    cell["wall_s"] = time.time() - t_cell_start
    cell["rss_mb_after"] = _rss_mb()
    return cell


# ---------------------------------------------------------------------------
# Stage drivers
# ---------------------------------------------------------------------------

def stage_smoke(out_dir: Path) -> dict:
    """Stage P0: 4 anchor cells, 1 seed, t=0.5. Sanity check before P1."""
    print(f"=== Stage P0 SMOKE ({len(SMOKE_CELLS)} cells, 1 seed, t=0.5) ===")
    t_global = time.time()
    cells = []
    for (D_G, A, label) in SMOKE_CELLS:
        cell = run_cell(D_G, A, SIGMA_S_DEFAULT,
                        seeds=[0],
                        coupling_tags=COUPLING_TAGS,
                        t_values=[0.5],
                        cell_label=label)
        cells.append(cell)
    elapsed = time.time() - t_global
    print(f"\nP0 SMOKE complete in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    return {
        "stage": "P0_smoke",
        "config": {
            "N_PAIRS": N_PAIRS, "K": K, "lambda_C3": LAMBDA_C3,
            "coupling_tags": COUPLING_TAGS,
            "smoke_cells": [(d, a, lbl) for (d, a, lbl) in SMOKE_CELLS],
        },
        "wall_s": elapsed,
        "cells": cells,
    }


def stage_main(out_dir: Path, seeds: list[int]) -> dict:
    """Stage P1: full 5x5 grid, t=0.5."""
    cells_to_run = [(d, a) for d in D_G_GRID for a in A_GRID]
    print(f"=== Stage P1 MAIN ({len(cells_to_run)} cells, "
          f"{len(seeds)} seeds, t=0.5) ===")
    t_global = time.time()
    cells = []
    for (D_G, A) in cells_to_run:
        cell = run_cell(D_G, A, SIGMA_S_DEFAULT,
                        seeds=list(seeds),
                        coupling_tags=COUPLING_TAGS,
                        t_values=[0.5],
                        cell_label="main")
        cells.append(cell)
    elapsed = time.time() - t_global
    print(f"\nP1 MAIN complete in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    return {
        "stage": "P1_main",
        "config": {
            "N_PAIRS": N_PAIRS, "K": K, "lambda_C3": LAMBDA_C3,
            "coupling_tags": COUPLING_TAGS,
            "D_G_grid": D_G_GRID, "A_grid": A_GRID, "seeds": list(seeds),
        },
        "wall_s": elapsed,
        "cells": cells,
    }


def stage_sentinel(out_dir: Path, seeds: list[int]) -> dict:
    """Stage P2: 5 t-pts on 5 corner cells; verifies t=0.5 representativeness."""
    print(f"=== Stage P2 SENTINEL ({len(SENTINEL_CELLS)} cells, "
          f"{len(seeds)} seeds, {len(SENTINEL_T)} t-pts) ===")
    t_global = time.time()
    cells = []
    for (D_G, A) in SENTINEL_CELLS:
        cell = run_cell(D_G, A, SIGMA_S_DEFAULT,
                        seeds=list(seeds),
                        coupling_tags=COUPLING_TAGS,
                        t_values=list(SENTINEL_T),
                        cell_label="sentinel")
        cells.append(cell)
    elapsed = time.time() - t_global
    print(f"\nP2 SENTINEL complete in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    return {
        "stage": "P2_sentinel",
        "config": {
            "N_PAIRS": N_PAIRS, "K": K, "lambda_C3": LAMBDA_C3,
            "coupling_tags": COUPLING_TAGS,
            "sentinel_cells": SENTINEL_CELLS, "t_grid": SENTINEL_T,
            "seeds": list(seeds),
        },
        "wall_s": elapsed,
        "cells": cells,
    }


def stage_p4a_lambda(out_dir: Path, seeds: list[int]) -> dict:
    """Stage P4a: corner-cell lambda sweep for C3."""
    lam_tags = [f"C3@{lam:g}" for lam in P4A_LAMBDAS]
    coupling_tags = ["C0", "C1"] + lam_tags + ["C4"]
    print(f"=== Stage P4a LAMBDA-SWEEP ({len(P4A_CELLS)} cells, "
          f"{len(seeds)} seeds, lambdas={P4A_LAMBDAS}) ===")
    t_global = time.time()
    cells = []
    for (D_G, A) in P4A_CELLS:
        cell = run_cell(D_G, A, SIGMA_S_DEFAULT,
                        seeds=list(seeds),
                        coupling_tags=coupling_tags,
                        t_values=[0.5],
                        cell_label="p4a_lambda_sweep")
        cells.append(cell)
    elapsed = time.time() - t_global
    print(f"\nP4a LAMBDA complete in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    return {
        "stage": "P4a_lambda",
        "config": {
            "N_PAIRS": N_PAIRS, "K": K,
            "coupling_tags": coupling_tags,
            "cells": P4A_CELLS, "lambdas": P4A_LAMBDAS,
            "seeds": list(seeds),
        },
        "wall_s": elapsed,
        "cells": cells,
    }


def stage_p4b_sigma(out_dir: Path, seeds: list[int]) -> dict:
    """Stage P4b: sigma_s sanity check."""
    print(f"=== Stage P4b SIGMA-S ({len(P4B_CELLS)} cells, "
          f"{len(seeds)} seeds) ===")
    t_global = time.time()
    cells = []
    for (D_G, A, sigma_s) in P4B_CELLS:
        cell = run_cell(D_G, A, sigma_s,
                        seeds=list(seeds),
                        coupling_tags=COUPLING_TAGS,
                        t_values=[0.5],
                        cell_label="p4b_sigma_sanity")
        cells.append(cell)
    elapsed = time.time() - t_global
    print(f"\nP4b SIGMA-S complete in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    return {
        "stage": "P4b_sigma",
        "config": {
            "N_PAIRS": N_PAIRS, "K": K, "lambda_C3": LAMBDA_C3,
            "coupling_tags": COUPLING_TAGS,
            "cells": P4B_CELLS, "seeds": list(seeds),
        },
        "wall_s": elapsed,
        "cells": cells,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True,
                        choices=["P0", "P1", "P2", "P4a", "P4b"],
                        help="experiment stage to run")
    parser.add_argument("--seeds", type=str, default=None,
                        help="comma-separated seed list (default: stage default)")
    parser.add_argument("--out-tag", type=str, default=None,
                        help="suffix for output JSON filename")
    args = parser.parse_args()

    out_dir = Path(__file__).resolve().parents[2] / "results"
    out_dir.mkdir(exist_ok=True)

    # Defaults per stage
    default_seeds = {
        "P0": [0],
        "P1": [0, 1, 2, 3, 4],
        "P2": [0, 1, 2, 3, 4],
        "P4a": [0, 1, 2, 3, 4],
        "P4b": [0, 1, 2, 3, 4],
    }[args.stage]
    seeds = (default_seeds if args.seeds is None
             else [int(s) for s in args.seeds.split(",")])

    print(f"phase_diagram driver  stage={args.stage}  seeds={seeds}")
    print(f"  N_PAIRS={N_PAIRS}  K={K}  lambda_C3={LAMBDA_C3}")
    print(f"  OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}")
    print(f"  startup RSS={_rss_mb():.0f} MB")
    print("-" * 78)

    if args.stage == "P0":
        result = stage_smoke(out_dir)
    elif args.stage == "P1":
        result = stage_main(out_dir, seeds)
    elif args.stage == "P2":
        result = stage_sentinel(out_dir, seeds)
    elif args.stage == "P4a":
        result = stage_p4a_lambda(out_dir, seeds)
    elif args.stage == "P4b":
        result = stage_p4b_sigma(out_dir, seeds)
    else:
        raise ValueError(args.stage)

    # Output names are descriptive (no internal stage code in the filename);
    # --stage keeps its short code as an internal CLI selector only.
    stage_out = {
        "P0": "phase_diagram_smoke.json",
        "P1": "phase_diagram_heatmap.json",
        "P2": "phase_diagram_t_sentinel.json",
        "P4a": "phase_diagram_lambda_sweep.json",
        "P4b": "phase_diagram_sigma_sanity.json",
    }
    if args.out_tag:
        out_path = out_dir / f"phase_diagram_{args.out_tag}.json"
    else:
        out_path = out_dir / stage_out[args.stage]
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=float)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
