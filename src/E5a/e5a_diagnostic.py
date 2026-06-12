"""E5a Stage 2 + 3: A1 real proxy diagnostic + A2 strict L^1 bound test.

Single overnight-shaped run that produces BOTH:
  (A1) the real-feature branch-decomposition diagnostic with three label
       choices (oracle / CLIP k-means proxy / weak-RP k-means proxy);
  (A2) the strict observed-support perturbation theorem test of
       Proposition `proxy-tv-bound` using oracle labels (same kNN graph).

Inputs: results/cifar10_clip_features.npz, cifar10_branch_labels.npz
        (from Stage 1).
Output: results/e5a_v7.json (all metrics, indexed by mode/seed/t).

Configuration:
  N                = 10,000  class-stratified, 1000 per class per seed
  seeds            = {0, 1, 2}
  t-grid           = 13 points in [0.05, 0.95]
  k_NN bandwidth   = 80
  source Z         = N(0, I_d / d) scale-matched to L2-normalised X
  perturbation p   = {0.1, 0.3, 0.5}  observed-support
  Mhat^2           = max over (anchor i, k in S_i) ||m_{i,k}||^2

Estimator contract identical to E3b: single global kNN graph per (seed, t)
reused for every label / perturbation; biased 1/k local covariance.

Runtime estimate: ~15-25 min on M3 Pro CPU (FAISS exact kNN + decompose
at d=512 dominate).
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
    ["pgrep", "-f", "e5a_diagnostic.py"],
    capture_output=True, text=True).stdout.split() if int(p) != _self_pid]
if _others:
    print(f"REFUSED TO START: another instance at PID(s) {_others}.",
          file=sys.stderr); sys.exit(2)

import numpy as np
import faiss

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from E5a.e5a_fast import decompose_fast as decompose
from E5a.e5a_fast import a2_observed_support_fast

HERE = Path(__file__).resolve().parent
RES = HERE.parents[1] / "results"
FEAT_PATH = RES / "cifar10_clip_features.npz"
LABEL_PATH = RES / "cifar10_branch_labels.npz"
OUT_PATH = RES / "e5a_v7.json"

# Paper-canonical defaults (hard-coded, not environment-configurable)
N_PER_SEED = 10_000
SEEDS = [0, 1, 2]
T_GRID = np.linspace(0.05, 0.95, 13).tolist()
K_KNN = 80
K_CLASSES = 10
P_PERTURB = [0.1, 0.3, 0.5]
KNN_WORKERS = 4
FAISS_THREADS = 4
faiss.omp_set_num_threads(FAISS_THREADS)


# ---------------------------------------------------------------------------
# class-stratified sampler (1000 per class per seed,
# without replacement)
# ---------------------------------------------------------------------------

def stratified_subset(labels: np.ndarray, n_per_class: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = []
    for c in range(K_CLASSES):
        class_idx = np.where(labels == c)[0]
        if len(class_idx) < n_per_class:
            raise ValueError(f"Class {c} has only {len(class_idx)} samples, need {n_per_class}")
        idx.append(rng.choice(class_idx, size=n_per_class, replace=False))
    idx = np.concatenate(idx)
    rng.shuffle(idx)
    return idx


# ---------------------------------------------------------------------------
# Support-size statistics for oracle pipeline (used by A2 and quality reporting)
# ---------------------------------------------------------------------------

def support_stats(K_neighbors: np.ndarray, K: int = K_CLASSES) -> dict:
    N, k = K_neighbors.shape
    sizes = np.zeros(N, dtype=np.int64)
    for c in range(K):
        sizes += (K_neighbors == c).any(axis=1).astype(np.int64)
    return {
        "mean":   float(sizes.mean()),
        "median": float(np.median(sizes)),
        "min":    int(sizes.min()),
        "max":    int(sizes.max()),
        "frac_full":   float((sizes == K).mean()),
        "frac_single": float((sizes == 1).mean()),
    }


# ---------------------------------------------------------------------------
# A2 strict observed-support perturbation test of Proposition proxy-tv-bound.
# Same per-anchor support S_i = {k : n_{i,k} > 0} used in LHS and RHS.
# ---------------------------------------------------------------------------

def a2_observed_support(U_n, K_n, p_list, K=K_CLASSES):
    """Thin wrapper around e5a_fast.a2_observed_support_fast (kept for
    name stability in run_one_t). The slow numpy-loop reference lives below
    and is dead code retained for documentation / future bisection."""
    return a2_observed_support_fast(U_n, K_n, p_list, K=K)


def _a2_observed_support_slow_reference(U_n: np.ndarray, K_n: np.ndarray, p_list: list[float],
                                         K: int = K_CLASSES) -> dict:
    """Returns dict keyed by p with LHS / RHS_global / RHS_local / RHS_q99 / ratio.

    Math (per anchor i):
        S_i = {k : n_{i,k} > 0}
        r_{i,k}    = n_{i,k} / k_NN
        m_{i,k}    = mean of U_j over neighbors j with K_j = k
        u_{S_i,k}  = 1/|S_i| if k in S_i else 0
        r_hat_{p,i,k} = (1 - p) r_{i,k} + p u_{S_i,k}
        bar_m_q_i  = sum_k q_{i,k} m_{i,k}   for q in {r, r_hat_p}
        B_q_i      = sum_{k in S_i} q_{i,k} ||m_{i,k} - bar_m_q_i||^2
        LHS_p,i    = |B_r_i - B_{r_hat_p}_i|
        ||r - r_hat_p||_1,i = sum_{k in S_i} |r_{i,k} - r_hat_{p,i,k}|
    Global:
        LHS_p             = E_i LHS_p,i
        RHS_global,p      = 3 * Mhat_global^2 * E_i ||r - r_hat_p||_1,i
        RHS_local,p       = 3 * E_i [Mhat_i^2 * ||r - r_hat_p||_1,i]
        RHS_q99,p         = 3 * Mhat_q99^2 * E_i ||r - r_hat_p||_1,i  (heuristic)
    """
    N, knn, d = U_n.shape

    # counts n_{i,k}: (N, K)
    counts = np.zeros((N, K), dtype=np.float64)
    for c in range(K):
        counts[:, c] = (K_n == c).sum(axis=1)
    # local posterior r
    r = counts / knn
    in_S = (counts > 0)
    safe_n = np.where(counts > 0, counts, 1.0)

    # local group means m_{i,k} for k in S_i (zero elsewhere)
    m = np.zeros((N, K, d), dtype=np.float64)
    for c in range(K):
        mask = (K_n == c).astype(np.float64)
        sum_c = (mask[..., None] * U_n).sum(axis=1)
        m[:, c, :] = sum_c / safe_n[:, c, None]
    m = m * in_S[..., None].astype(np.float64)

    # Mhat^2 statistics from m (only positions in S_i)
    m_norm_sq = (m ** 2).sum(axis=-1)               # (N, K)
    pos_mask = in_S.astype(bool)
    nonzero_norm_sq = m_norm_sq[pos_mask]            # 1-D over (i, k in S_i)
    M_hat_sq_global = float(nonzero_norm_sq.max())
    M_hat_sq_q99    = float(np.quantile(nonzero_norm_sq, 0.99))
    M_i_sq = m_norm_sq.max(axis=1)                   # (N,) per anchor

    # observed-support uniform u_{S_i}
    S_size = in_S.sum(axis=1).astype(np.float64)     # (N,)
    safe_S = np.where(S_size > 0, S_size, 1.0)
    u = in_S.astype(np.float64) / safe_S[:, None]

    # Centroid-based per-anchor Chebyshev-radius upper bound (Remark rem:proxy-tv-radius).
    # See e5a_fast.a2_observed_support_fast for the rationale; mirrored here for
    # the slow-path reference.
    c_centroid = (m * in_S[..., None].astype(np.float64)).sum(axis=1) / safe_S[:, None]
    diff_c = (m - c_centroid[:, None, :])
    diff_c_norm_sq = (diff_c ** 2).sum(axis=-1)
    diff_c_masked = np.where(in_S, diff_c_norm_sq, -1.0)
    R_i_sq = diff_c_masked.max(axis=1)               # (N,)
    R_hat_sq_global = float(R_i_sq.max())
    R_hat_sq_q99 = float(np.quantile(R_i_sq, 0.99))

    # Reference B_r and its per-anchor mean direction bar_m_r
    bar_m_r = (r[..., None] * m).sum(axis=1)         # (N, d)
    diff_r = (m - bar_m_r[:, None, :]) * in_S[..., None].astype(np.float64)
    sq_r = (diff_r ** 2).sum(axis=-1)                # (N, K)
    B_r = (r * sq_r).sum(axis=1)                     # (N,)

    out = {}
    for p in p_list:
        r_hat = (1.0 - p) * r + p * u
        bar_m_h = (r_hat[..., None] * m).sum(axis=1)
        diff_h = (m - bar_m_h[:, None, :]) * in_S[..., None].astype(np.float64)
        sq_h = (diff_h ** 2).sum(axis=-1)
        B_h = (r_hat * sq_h).sum(axis=1)

        LHS_per = np.abs(B_r - B_h)
        L1_per = np.abs(r - r_hat).sum(axis=1)

        LHS = float(LHS_per.mean())
        L1_mean = float(L1_per.mean())
        RHS_global = 3.0 * M_hat_sq_global * L1_mean
        RHS_local  = 3.0 * float((M_i_sq * L1_per).mean())
        RHS_q99    = 3.0 * M_hat_sq_q99 * L1_mean
        RHS_R_global = 3.0 * R_hat_sq_global * L1_mean
        RHS_R_local  = 3.0 * float((R_i_sq * L1_per).mean())
        RHS_R_q99    = 3.0 * R_hat_sq_q99 * L1_mean

        out[f"p={p}"] = {
            "p": float(p),
            "LHS":              LHS,
            "RHS_global":       RHS_global,
            "RHS_local":        RHS_local,
            "RHS_q99":          RHS_q99,
            "RHS_R_global":     RHS_R_global,
            "RHS_R_local":      RHS_R_local,
            "RHS_R_q99":        RHS_R_q99,
            "ratio_global":     RHS_global / max(LHS, 1e-30),
            "ratio_local":      RHS_local  / max(LHS, 1e-30),
            "ratio_q99":        RHS_q99    / max(LHS, 1e-30),
            "ratio_R_global":   RHS_R_global / max(LHS, 1e-30),
            "ratio_R_local":    RHS_R_local  / max(LHS, 1e-30),
            "ratio_R_q99":      RHS_R_q99    / max(LHS, 1e-30),
            "L1_mean":          L1_mean,
            "B_r_mean":         float(B_r.mean()),
            "B_h_mean":         float(B_h.mean()),
        }

    out["M_hat_sq_global"] = M_hat_sq_global
    out["M_hat_sq_q99"]    = M_hat_sq_q99
    out["R_hat_sq_global"] = R_hat_sq_global
    out["R_hat_sq_q99"]    = R_hat_sq_q99
    return out


# ---------------------------------------------------------------------------
# Per-(seed, t) pipeline
# ---------------------------------------------------------------------------

def run_one_t(X_feat: np.ndarray, K_oracle: np.ndarray, K_CLIP: np.ndarray,
              K_weakRP: np.ndarray, idx_subset: np.ndarray, t: float,
              seed: int, k_nn: int) -> dict:
    d = X_feat.shape[1]
    rng = np.random.default_rng(seed * 1000 + int(round(t * 1000)))

    X = X_feat[idx_subset].astype(np.float32)
    KO = K_oracle[idx_subset].astype(np.int64)
    KC = K_CLIP[idx_subset].astype(np.int64)
    KR = K_weakRP[idx_subset].astype(np.int64)

    # scale-matched Z ~ N(0, I/d)
    Z = rng.standard_normal((len(idx_subset), d)).astype(np.float32) / np.sqrt(d)
    Y = ((1.0 - t) * Z + t * X).astype(np.float32)
    U = (X - Z).astype(np.float64)

    # one global kNN graph
    index = faiss.IndexFlatL2(d)
    index.add(Y)
    _, knn_idx = index.search(Y, k_nn)
    U_n = U[knn_idx]                                    # (N, k, d) float64

    # A1: decompose under 3 labels on the same neighborhoods
    s_oracle = decompose(U_n, KO[knn_idx])
    s_clip   = decompose(U_n, KC[knn_idx])
    s_rp     = decompose(U_n, KR[knn_idx])

    # support stats from oracle pipeline (independent of A2)
    supp = support_stats(KO[knn_idx], K=K_CLASSES)

    # A2: observed-support perturbation, oracle labels only
    a2 = a2_observed_support(U_n, KO[knn_idx], P_PERTURB, K=K_CLASSES)

    Av_inv_max = max(
        abs(s_oracle["aggregated"]["A_v"] - s_clip["aggregated"]["A_v"]),
        abs(s_oracle["aggregated"]["A_v"] - s_rp["aggregated"]["A_v"]),
    )

    return {
        "t": float(t),
        "seed": int(seed),
        # three label sources: oracle, CLIP, weak random projection
        "A_v":                s_oracle["aggregated"]["A_v"],
        "A_within_oracle":    s_oracle["aggregated"]["A_within"],
        "A_between_oracle":   s_oracle["aggregated"]["A_between"],
        "A_within_CLIP":      s_clip["aggregated"]["A_within"],
        "A_between_CLIP":     s_clip["aggregated"]["A_between"],
        "A_within_weakRP":    s_rp["aggregated"]["A_within"],
        "A_between_weakRP":   s_rp["aggregated"]["A_between"],
        # invariants
        "id_resid_oracle":    s_oracle["aggregated"]["identity_residual_max_abs"],
        "id_resid_CLIP":      s_clip["aggregated"]["identity_residual_max_abs"],
        "id_resid_weakRP":    s_rp["aggregated"]["identity_residual_max_abs"],
        "Av_inv_max":         float(Av_inv_max),
        # support stats
        "support_size_mean":   supp["mean"],
        "support_size_median": supp["median"],
        "support_size_min":    supp["min"],
        "support_size_max":    supp["max"],
        "frac_full_support":   supp["frac_full"],
        "frac_single_support": supp["frac_single"],
        # A2 perturbation results
        "A2": a2,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    if not FEAT_PATH.exists() or not LABEL_PATH.exists():
        print(f"FATAL: missing prerequisite files. Run Stages 1a/1b first.",
              file=sys.stderr); sys.exit(2)

    print("=== E5a Stage 2/3: A1 diagnostic + A2 bound test ===")
    t0 = time.time()

    print(f"Loading features and labels...")
    data = np.load(FEAT_PATH)
    X_l2 = data["features_l2"]                          # (50000, 512) float32
    lab = np.load(LABEL_PATH)
    K_oracle = lab["K_oracle"]
    K_CLIP   = lab["K_proxy_CLIP"]
    K_weakRP = lab["K_proxy_weakRP"]
    print(f"  X_l2: {X_l2.shape}  K_oracle: {K_oracle.shape}  "
          f"({time.time() - t0:.1f}s)")

    print(f"\nConfig: N={N_PER_SEED} per seed, seeds={SEEDS}, "
          f"|t-grid|={len(T_GRID)}, k_NN={K_KNN}, p={P_PERTURB}")
    n_cells = len(SEEDS) * len(T_GRID)
    print(f"Total (seed, t) cells: {n_cells}\n")

    runs = []
    for seed in SEEDS:
        idx = stratified_subset(K_oracle, N_PER_SEED // K_CLASSES, seed=seed)
        # sanity: class counts
        per_class = [int((K_oracle[idx] == c).sum()) for c in range(K_CLASSES)]
        print(f"seed={seed}: stratified subset {N_PER_SEED}; per-class counts {per_class}")

        for t_idx, t in enumerate(T_GRID):
            t_start = time.time()
            m = run_one_t(X_l2, K_oracle, K_CLIP, K_weakRP, idx, t, seed, K_KNN)
            dur = time.time() - t_start
            runs.append(m)
            print(f"  t={t:.2f} [{t_idx+1}/{len(T_GRID)}]  dur={dur:5.1f}s  "
                  f"A_v={m['A_v']:.5f}  "
                  f"A_btw oracle={m['A_between_oracle']:.5f} "
                  f"CLIP={m['A_between_CLIP']:.5f} "
                  f"RP={m['A_between_weakRP']:.5f}  "
                  f"|S_i|mean={m['support_size_mean']:.2f}  "
                  f"id_resid={max(m['id_resid_oracle'], m['id_resid_CLIP'], m['id_resid_weakRP']):.2e}  "
                  f"Av_inv={m['Av_inv_max']:.2e}")

    elapsed_total = time.time() - t0
    print(f"\nAll {len(runs)} cells done in {elapsed_total / 60:.1f} min.")

    config = {
        "N_per_seed":   N_PER_SEED,
        "seeds":        SEEDS,
        "t_grid":       T_GRID,
        "k_NN":         K_KNN,
        "K_classes":    K_CLASSES,
        "p_perturb":    P_PERTURB,
        "feature_extractor": "CLIP ViT-B/32 (use_safetensors=True)",
        "feature_normalisation": "L2 (||X||=1)",
        "source_Z_dist": "N(0, I_d/d) scale-matched",
        "distance":     "L2 on Y_t",
        "coupling":     "C0 independent",
        "estimator":    "single global FAISS L2 kNN graph; biased 1/k local covariance",
        "kNN_workers":  KNN_WORKERS,
        "FAISS_threads": FAISS_THREADS,
        "elapsed_seconds": elapsed_total,
    }
    print(f"Saving {OUT_PATH}...")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump({"config": config, "runs": runs}, f, indent=2)
    print(f"  size: {OUT_PATH.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
