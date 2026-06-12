"""E5a Stage 1b: fit two k-means proxies + quality metrics + Hungarian
relabel to CIFAR-10 oracle classes.

Input:  results/cifar10_clip_features.npz  (from Stage 1a)
Output:
  results/cifar10_branch_labels.npz
    K_oracle          (50000,) int64  CIFAR-10 class index
    K_proxy_CLIP      (50000,) int64  Hungarian-relabeled k-means(CLIP-l2, K=10)
    K_proxy_weakRP    (50000,) int64  Hungarian-relabeled k-means(rand-proj(CLIP, 16), K=10)
    rand_proj_matrix  (512, 16) float32  the random projection matrix used
  results/cifar10_proxy_quality.json
    per-proxy quality metrics: Hungarian acc, NMI, ARI, cluster_size_entropy,
    confusion matrix (10x10), per-class accuracy, raw -> relabeled mapping

Idempotent (skip if outputs exist; FORCE=1 to overwrite).
Runtime: ~30-60 s on Mac CPU under threadpool_limits(1).
"""

from __future__ import annotations

import functools, json, os, subprocess, sys, time
from pathlib import Path

for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(var, "4")

print = functools.partial(print, flush=True)

_self_pid = os.getpid()
_others = [int(p) for p in subprocess.run(
    ["pgrep", "-f", "e5a_prepare_branches.py"],
    capture_output=True, text=True).stdout.split() if int(p) != _self_pid]
if _others:
    print(f"REFUSED TO START: another instance running at PID(s) {_others}.",
          file=sys.stderr)
    sys.exit(2)

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import (
    normalized_mutual_info_score, adjusted_rand_score, confusion_matrix,
)
from scipy.optimize import linear_sum_assignment
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
RES = HERE.parents[1] / "results"
IN_PATH = RES / "cifar10_clip_features.npz"
OUT_LABELS = RES / "cifar10_branch_labels.npz"
OUT_QUAL = RES / "cifar10_proxy_quality.json"
K_CLASSES = 10
RP_DIM = 16            # weak-proxy random projection target dim
KMEANS_N_INIT = 10
RANDOM_STATE = 0
FORCE = bool(int(os.environ.get("FORCE", "0")))


def hungarian_relabel(pred: np.ndarray, true: np.ndarray, K: int = 10):
    """Maximum-weight bipartite matching of predicted cluster ids to true
    classes; returns relabeled pred + matched-accuracy + perm map."""
    conf = np.zeros((K, K), dtype=np.int64)
    for p, t in zip(pred, true):
        conf[p, t] += 1
    row, col = linear_sum_assignment(-conf)        # maximise matches
    perm = np.zeros(K, dtype=np.int64)
    for r, c in zip(row, col):
        perm[r] = c
    relabeled = perm[pred]
    return relabeled, perm


def cluster_size_entropy(labels: np.ndarray, K: int) -> float:
    """Shannon entropy (nats) of the empirical cluster-size distribution."""
    counts = np.bincount(labels, minlength=K).astype(np.float64)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def per_class_accuracy(relabeled: np.ndarray, true: np.ndarray, K: int):
    """For each true class c, the fraction of samples that the relabeled
    proxy puts in c. Returns list of K floats."""
    return [float(((relabeled == c) & (true == c)).sum() / (true == c).sum())
            for c in range(K)]


def evaluate_proxy(name: str, raw_pred: np.ndarray, oracle: np.ndarray, K: int):
    relabeled, perm = hungarian_relabel(raw_pred, oracle, K=K)
    cm = confusion_matrix(oracle, relabeled, labels=list(range(K))).tolist()
    quality = {
        "name": name,
        "hungarian_acc": float((relabeled == oracle).mean()),
        "nmi": float(normalized_mutual_info_score(oracle, raw_pred)),
        "ari": float(adjusted_rand_score(oracle, raw_pred)),
        "cluster_size_entropy_raw":      cluster_size_entropy(raw_pred, K),
        "cluster_size_entropy_relabeled": cluster_size_entropy(relabeled, K),
        "max_entropy_log_K":              float(np.log(K)),
        "per_class_acc": per_class_accuracy(relabeled, oracle, K),
        "perm_pred_to_oracle": perm.tolist(),
        "confusion_matrix":   cm,
    }
    return relabeled, quality


def main():
    if not IN_PATH.exists():
        print(f"FATAL: {IN_PATH} not found. Run e5a_extract_clip_features.py first.",
              file=sys.stderr)
        sys.exit(2)

    if OUT_LABELS.exists() and OUT_QUAL.exists() and not FORCE:
        print(f"=== E5a Stage 1b: cached outputs detected ===")
        z = np.load(OUT_LABELS)
        with open(OUT_QUAL) as f:
            q = json.load(f)
        print(f"  {OUT_LABELS.name}: keys={list(z.files)}")
        print(f"  {OUT_QUAL.name}: proxies={list(q.keys())}")
        for k, v in q.items():
            print(f"    {k}: hungarian_acc={v['hungarian_acc']:.3f}  "
                  f"NMI={v['nmi']:.3f}  ARI={v['ari']:.3f}")
        print("\nSet FORCE=1 to recompute. Skipping.")
        return

    print(f"=== E5a Stage 1b: prepare branch labels ===")
    t0 = time.time()

    print(f"Loading features from {IN_PATH}...")
    data = np.load(IN_PATH)
    X_l2 = data["features_l2"]                            # (50000, 512) float32
    oracle = data["labels"].astype(np.int64)              # (50000,) int64
    print(f"  X_l2 shape: {X_l2.shape}  ({time.time() - t0:.1f}s)")

    rng = np.random.default_rng(RANDOM_STATE)
    quality = {}

    with threadpool_limits(limits=1):
        # ----- proxy 1: k-means on L2-normalised CLIP features
        t1 = time.time()
        print(f"\nFitting k-means(CLIP_l2, K={K_CLASSES}, n_init={KMEANS_N_INIT}) "
              f"under threadpool_limits(1)...")
        km_clip = KMeans(n_clusters=K_CLASSES, n_init=KMEANS_N_INIT,
                          random_state=RANDOM_STATE).fit(X_l2)
        raw_pred_clip = km_clip.labels_.astype(np.int64)
        print(f"  done in {time.time() - t1:.1f}s")
        K_proxy_CLIP, q_clip = evaluate_proxy("CLIP_kmeans", raw_pred_clip, oracle, K_CLASSES)
        quality["CLIP_kmeans"] = q_clip
        print(f"  hungarian_acc={q_clip['hungarian_acc']:.3f}  "
              f"NMI={q_clip['nmi']:.3f}  ARI={q_clip['ari']:.3f}  "
              f"cluster_entropy={q_clip['cluster_size_entropy_raw']:.3f} "
              f"(max log K = {q_clip['max_entropy_log_K']:.3f})")
        print(f"  per-class acc: {[f'{a:.2f}' for a in q_clip['per_class_acc']]}")

        # proxy 2 (weak): random projection to 16-D, then k-means
        t1 = time.time()
        print(f"\nFitting k-means(rand-proj(CLIP, {RP_DIM}-D), K={K_CLASSES})...")
        d_in = X_l2.shape[1]
        R = rng.standard_normal((d_in, RP_DIM)).astype(np.float32) / np.sqrt(d_in)
        X_rp = X_l2 @ R                                  # (50000, 16)
        km_rp = KMeans(n_clusters=K_CLASSES, n_init=KMEANS_N_INIT,
                        random_state=RANDOM_STATE).fit(X_rp)
        raw_pred_rp = km_rp.labels_.astype(np.int64)
        print(f"  done in {time.time() - t1:.1f}s")
        K_proxy_weakRP, q_rp = evaluate_proxy("weakRP_kmeans", raw_pred_rp,
                                               oracle, K_CLASSES)
        quality["weakRP_kmeans"] = q_rp
        print(f"  hungarian_acc={q_rp['hungarian_acc']:.3f}  "
              f"NMI={q_rp['nmi']:.3f}  ARI={q_rp['ari']:.3f}  "
              f"cluster_entropy={q_rp['cluster_size_entropy_raw']:.3f}")
        print(f"  per-class acc: {[f'{a:.2f}' for a in q_rp['per_class_acc']]}")

    # ----- save outputs
    print(f"\nSaving labels -> {OUT_LABELS}")
    np.savez_compressed(OUT_LABELS,
                        K_oracle=oracle,
                        K_proxy_CLIP=K_proxy_CLIP.astype(np.int64),
                        K_proxy_weakRP=K_proxy_weakRP.astype(np.int64),
                        rand_proj_matrix=R)
    print(f"  size: {OUT_LABELS.stat().st_size / 1e6:.2f} MB")

    print(f"Saving quality metrics -> {OUT_QUAL}")
    with open(OUT_QUAL, "w") as f:
        json.dump(quality, f, indent=2)
    print(f"  size: {OUT_QUAL.stat().st_size / 1024:.1f} kB")

    print(f"\nStage 1b DONE in {time.time() - t0:.1f}s total.")


if __name__ == "__main__":
    main()
