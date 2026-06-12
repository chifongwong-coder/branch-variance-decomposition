"""E5b: unsupervised-proxy recovery vs kNN bandwidth, per branch cardinality K.

The de-biased estimator subtracts a cardinality null whose magnitude scales as
K/k (a random K-partition of the k neighbors leaves ~k/K per branch), so resolving
more branches needs proportionally more neighbors: k must grow with K. This driver
measures, for K = 2, 3, 10, the proxy recovery of the oracle de-biased between-branch
signal as the bandwidth k varies, both over the full t-grid and over the ambiguity
window t <= 0.5 (where the between-branch signal is present). Holding ~20 to 40
neighbors per branch, recovery stays near one at every K; a single fixed bandwidth
instead depresses the recovery at fine K and drives the full-grid K=10 reading below
the null at k=80.

CPU only, cached CLIP features (no model, no CLIP forward). Reuses the shared chunked
abetween_debiased machinery (core/expE_common).

RESUMABLE: every (K, seed) result is saved incrementally to
results/expE5b_recovery_bandwidth.json the moment it is computed; re-running loads
what exists and only computes missing seeds. Going from 3 to 5 seeds runs only
seeds 3, 4.

Run:  python3 src/E5b/expE5b_recovery_bandwidth.py [--K 3] [--seeds N | --seed-list 0,1,2] [--ks 24,60,120]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from core import expE_common as C                                       # noqa: E402

FEAT = HERE.parents[1] / "results" / "cifar10_clip_features.npz"
OUT = HERE.parents[1] / "results" / "expE5b_recovery_bandwidth.json"
N = 8000
N_NULL = 6
T_GRID = [round(float(x), 3) for x in np.linspace(0.05, 0.95, 13)]
AMBIG = 0.5
VEHICLES = {0, 1, 8, 9}
K3_MAP = {0: 0, 1: 0, 8: 0, 9: 0, 2: 1, 6: 1, 3: 2, 4: 2, 5: 2, 7: 2}


def coarsen(cls, K):
    if K == 2:
        return np.array([0 if c in VEHICLES else 1 for c in cls], dtype=np.int64)
    if K == 3:
        return np.array([K3_MAP[c] for c in cls], dtype=np.int64)
    return np.asarray(cls, dtype=np.int64)   # K=10: identity (full class set)


def knn_idx(Y, k):
    Y = np.ascontiguousarray(Y, dtype=np.float32)
    try:
        import faiss
        index = faiss.IndexFlatL2(Y.shape[1]); index.add(Y)
        return index.search(Y, k)[1]
    except Exception:
        return C.knn_indices(Y, k)


def load_store():
    if OUT.exists():
        return json.load(open(OUT))
    return {}


def save_store(store):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".json.tmp")
    json.dump(store, open(tmp, "w"), indent=2)
    tmp.replace(OUT)                                   # atomic


def compute_seed(feats, labels, K, K_NN_LIST, s):
    """Per-seed: recovery (full grid and ambiguity window) and deb_oracle at each k."""
    n_avail, dim = feats.shape
    kmax = max(K_NN_LIST); msk = np.array(T_GRID) <= AMBIG
    rng = np.random.default_rng(s)
    sub = rng.permutation(n_avail)[:N]
    f = feats[sub]; cls = labels[sub]
    oracle = coarsen(cls, K)
    proxy = KMeans(n_clusters=K, n_init=10, random_state=0).fit_predict(
        f.astype(np.float32)).astype(np.int64)
    zf = rng.standard_normal((N, dim)) / np.sqrt(dim)
    U = f - zf
    per = {k: {"or": [], "px": []} for k in K_NN_LIST}
    for t in T_GRID:
        Yt = (1 - t) * zf + t * f
        nn_full = knn_idx(Yt, kmax)
        for k in K_NN_LIST:
            nn = nn_full[:, :k]
            sd = s * 100000 + k                        # same null for oracle & proxy
            d_or = C.abetween_debiased(U, nn, oracle, n_null=N_NULL,
                                       n_classes=K, chunk=300, seed=sd)["A_between_deb"]
            d_px = C.abetween_debiased(U, nn, proxy, n_null=N_NULL,
                                       n_classes=K, chunk=300, seed=sd)["A_between_deb"]
            per[k]["or"].append(d_or); per[k]["px"].append(d_px)
    res = {}
    for k in K_NN_LIST:
        o = np.array(per[k]["or"]); p = np.array(per[k]["px"])
        res[str(k)] = {
            "rec_full": float(p.mean() / o.mean()) if o.mean() > 1e-4 else None,
            "rec_ambig": float(p[msk].mean() / o[msk].mean()) if o[msk].mean() > 1e-4 else None,
            "deb_or_full": float(o.mean()),
            "deb_or_ambig": float(o[msk].mean()),
        }
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed-list", default=None, dest="seed_list")
    ap.add_argument("--ks", default="80,200,400")
    ap.add_argument("--K", type=int, default=3)
    args = ap.parse_args()
    K = args.K
    K_NN_LIST = [int(x) for x in args.ks.split(",")]
    seeds = ([int(x) for x in args.seed_list.split(",")] if args.seed_list
             else list(range(args.seeds)))

    d = np.load(FEAT)
    feats = d["features_l2"].astype(np.float64); labels = d["labels"].astype(np.int64)
    print(f"K={K}  N={N}  seeds={seeds}  k-list={K_NN_LIST}  ({len(labels)}x{feats.shape[1]})",
          flush=True)

    store = load_store()
    sk = store.setdefault(str(K), {})
    for s in seeds:
        have = sk.get(str(s))
        if have and all(str(k) in have for k in K_NN_LIST):
            print(f"  seed {s}: loaded from cache (skip)", flush=True)
            continue
        res = compute_seed(feats, labels, K, K_NN_LIST, s)
        sk.setdefault(str(s), {}).update(res)
        save_store(store)                              # incremental, after every seed
        print(f"  seed {s} done and saved", flush=True)

    # aggregate over all seeds present for this K at the requested k-list
    def agg(field, k):
        vals = [sk[str(s)][str(k)][field] for s in seeds
                if str(s) in sk and str(k) in sk[str(s)] and sk[str(s)][str(k)][field] is not None]
        return float(np.mean(vals)) if vals else float("nan")
    print(f"\nK={K} proxy recovery vs bandwidth (mean over {len(seeds)} seeds; {OUT.name}):")
    print(f"{'k':>5} {'nb/branch':>9} {'recov(full)':>12} {'recov(t<=.5)':>13} {'deb_or(full)':>13} {'deb_or(t<=.5)':>14}")
    for k in K_NN_LIST:
        print(f"{k:>5} {k // K:>9} {agg('rec_full', k):>12.3f} {agg('rec_ambig', k):>13.3f} "
              f"{agg('deb_or_full', k):>+13.4f} {agg('deb_or_ambig', k):>+14.4f}")


if __name__ == "__main__":
    main()
