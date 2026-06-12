"""Aggregate the E5b recovery-vs-bandwidth store into the figure-ready summary.

Reads results/expE5b_recovery_bandwidth.json (per-seed recovery in the ambiguity
window, t <= 1/2, written by expE5b_recovery_bandwidth.py) and writes
results/e5_recovery_vs_cardinality.json, the summary read by
src/E5/plot_e5_realdata_validation.py panel (a): proxy recovery at matched
neighbors-per-branch (20 and 40) for K = 2, 3, 10.

Every K is read the same way, from the per-seed store; there are no hardcoded
values. Populate the store by running expE5b_recovery_bandwidth.py for each K
(the run is deterministic and reproduces the same recovery values).

Run:  python3 src/E5b/make_e5_recovery_aggregate.py
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STORE = HERE.parents[1] / "results" / "expE5b_recovery_bandwidth.json"
OUT = HERE.parents[1] / "results" / "e5_recovery_vs_cardinality.json"

# (K, [k at 20/branch, k at 40/branch])
K_TO_KS = {2: [40, 80], 3: [60, 120], 10: [200, 400]}


def from_store(store, K, ks):
    sk = store.get(str(K), {})
    means, stds = [], []
    for k in ks:
        vals = [sk[s][str(k)]["rec_ambig"] for s in sk
                if str(k) in sk[s] and sk[s][str(k)]["rec_ambig"] is not None]
        if not vals:
            return None
        means.append(round(float(np.mean(vals)), 4))
        stds.append(round(float(np.std(vals, ddof=1)), 4) if len(vals) > 1 else None)
    return means, stds


def main():
    store = json.load(open(STORE)) if STORE.exists() else {}
    rec_mean, rec_std = {}, {}
    for K, ks in K_TO_KS.items():
        got = from_store(store, K, ks)
        if got is None:
            raise SystemExit(
                f"K={K} not in {STORE.name}; run "
                f"expE5b_recovery_bandwidth.py --K {K} --ks {','.join(map(str, ks))}")
        rec_mean[str(K)], rec_std[str(K)] = got
    out = {
        "neighbors_per_branch": [20, 40],
        "K": [2, 3, 10],
        "k": {str(K): ks for K, ks in K_TO_KS.items()},
        "recovery_mean": rec_mean,
        "recovery_std": rec_std,
        "window": "ambiguity (t <= 1/2)",
        "seeds": 3,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=2)
    print(f"wrote {OUT}")
    for K in out["K"]:
        print(f"  K={K}: {rec_mean[str(K)]}  (k={out['k'][str(K)]})")


if __name__ == "__main__":
    main()
