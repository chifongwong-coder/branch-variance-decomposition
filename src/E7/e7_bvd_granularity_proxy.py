"""E7 no-train check: branch-variable granularity + proxy-K
recovery for the canonical BVD.

Implements three ideas from the validation discussion, all model-free, on cached
real CIFAR-10 CLIP features (the c0 coupling, i.e. real-feature velocities):

  1. K is a free granularity. Report A_between(t) at K=1 (degenerate, must be 0),
     K=2 (animals vs vehicles, a nested coarsening of K=10), and K=10 (classes).
     The decomposition is exact for any K and A_between is weakly increasing under
     refinement (K=1 <= K=2 <= K=10), which this checks.
  2. K=1 degenerate boundary: A_between = 0, A_within = A_v.
  3. proxy-K recovery: cluster the CLIP features (KMeans, k=2 and k=10) and use the
     clusters as a label-free branch variable. Report what fraction of the
     oracle-K A_between the proxy-K recovers. The success bar from the discussion
     is 1/3 to 1/2.

Run:  python3 e7_bvd_granularity_proxy.py
Writes results/e7_bvd_granularity_proxy.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from core.branch_decomp import decompose                          # noqa: E402
from E7.e7_bvd_sensitivity_check import _load_aligned, N, DPIX  # noqa: E402
from E7.e7_cifar_fm_diagnostics import T_GRID, K_NN             # noqa: E402

RES = HERE.parents[1] / "results"
# CIFAR-10: 0 airplane,1 automobile,2 bird,3 cat,4 deer,5 dog,6 frog,7 horse,8 ship,9 truck
VEHICLES = {0, 1, 8, 9}
ANIMALS = {2, 3, 4, 5, 6, 7}


def a_between_for_labels(feat, kx, labels_full, rng_seed=777):
    """Canonical A_between(t) on the c0 (identity) coupling for a given branch
    labeling `labels_full` (length N). Mirrors the bvd_curves inner loop."""
    rng = np.random.default_rng(rng_seed)
    dim = feat.shape[1]
    fp = feat                                                # c0: col = identity
    Zf = (rng.standard_normal(fp.shape) / np.sqrt(dim)).astype(np.float32)
    U = (fp - Zf).astype(np.float64)
    out = []
    for t in T_GRID:
        Yt = ((1 - t) * Zf + t * fp).astype(np.float32)
        nn = NearestNeighbors(n_neighbors=K_NN, algorithm="brute").fit(Yt).kneighbors(Yt)[1]
        s = decompose(U[nn], labels_full[nn])["aggregated"]
        out.append((s["A_v"], s["A_within"], s["A_between"]))
    return np.array(out)                                     # (T, 3): A_v, A_within, A_between


def main():
    feat_all, lab_all, _ = _load_aligned()
    rng = np.random.default_rng(2025)
    sub = rng.permutation(feat_all.shape[0])[:N]
    feat = feat_all[sub].astype(np.float32)
    kx = lab_all[sub].astype(np.int64)

    # oracle K at three granularities
    K1 = np.zeros(N, dtype=np.int64)
    K2 = np.array([0 if c in VEHICLES else 1 for c in kx], dtype=np.int64)  # vehicle/animal
    K10 = kx

    # proxy K via KMeans on CLIP features (label-free)
    proxy2 = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(feat).astype(np.int64)
    proxy10 = KMeans(n_clusters=10, n_init=10, random_state=0).fit_predict(feat).astype(np.int64)

    curves = {
        "oracle_K1": a_between_for_labels(feat, kx, K1),
        "oracle_K2": a_between_for_labels(feat, kx, K2),
        "oracle_K10": a_between_for_labels(feat, kx, K10),
        "proxy_K2": a_between_for_labels(feat, kx, proxy2),
        "proxy_K10": a_between_for_labels(feat, kx, proxy10),
    }

    def ab(name):  # max-t A_between (and mean) for a curve
        arr = curves[name][:, 2]
        return float(arr.max()), float(arr.mean())

    print(f"N={N}  K_NN={K_NN}  |t-grid|={len(T_GRID)}  (c0 real-feature velocities)\n")
    print(f"{'branch variable':18} {'maxA_between':>13} {'meanA_between':>14}")
    for name in ("oracle_K1", "oracle_K2", "oracle_K10", "proxy_K2", "proxy_K10"):
        mx, mn = ab(name)
        print(f"{name:18} {mx:>13.4f} {mn:>14.4f}")

    # refinement monotonicity: K1 <= K2 <= K10 (max-t)
    m1, m2, m10 = ab("oracle_K1")[0], ab("oracle_K2")[0], ab("oracle_K10")[0]
    mono = (m1 <= m2 + 1e-9) and (m2 <= m10 + 1e-9)
    print(f"\nrefinement monotonicity K1<=K2<=K10: {m1:.4f} <= {m2:.4f} <= {m10:.4f}  "
          f"[{'OK' if mono else 'FAIL'}]")
    print(f"K=1 degenerate A_between ~ 0: {m1:.2e}  [{'OK' if abs(m1) < 1e-9 else 'FAIL'}]")

    # proxy recovery fraction (mean over t, matched granularity)
    rec2 = curves["proxy_K2"][:, 2].mean() / max(curves["oracle_K2"][:, 2].mean(), 1e-12)
    rec10 = curves["proxy_K10"][:, 2].mean() / max(curves["oracle_K10"][:, 2].mean(), 1e-12)
    print(f"\nproxy-K recovery (mean-t A_between proxy/oracle):")
    print(f"  K=2  (animals/vehicles vs KMeans-2):  {rec2:.2f}  "
          f"[{'>=1/3' if rec2 >= 1/3 else '<1/3'}]")
    print(f"  K=10 (classes vs KMeans-10):          {rec10:.2f}  "
          f"[{'>=1/3' if rec10 >= 1/3 else '<1/3'}]")

    out = {"N": N, "K_NN": K_NN, "t_grid": list(T_GRID),
           "curves": {k: v.tolist() for k, v in curves.items()},
           "refinement_monotonic": bool(mono),
           "recovery_K2": float(rec2), "recovery_K10": float(rec10)}
    with open(RES / "e7_bvd_granularity_proxy.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {RES / 'e7_bvd_granularity_proxy.json'}")


if __name__ == "__main__":
    main()
