"""E3: Conditional Semantic Mismatch Toy.

Geometry-semantics mismatch setting where Euclidean OT is expected to ignore
the semantic coordinate because it is "drowned out" by a high-dimensional
nuisance geometry coordinate.

Setup

  Source         Z = [Z_G, Z_S] ~ N(0, I_{d_g+1}),  d_g = 32, full dim D = 33
  Condition      C ∈ {-1, +1}, exactly balanced, independent of Z (external)
  Target         G ~ N(0, I_{d_g})
                 K_X ∈ {-1, +1}, exactly balanced
                 S = a * K_X + sigma_s * eps,  eps ~ N(0, 1)
                 X = concat(G, S),  total dim D = d_g + 1 = 33
  Interpolant    Y_t = (1-t) Z + t X,  U = X - Z

Couplings (every coupling preserves Z marginal exactly; X gets re-paired)
  C0  independent                 (Z, X) random pairing baseline
  C1  Euclidean OT                cost_ij = ||Z_i - X_j||^2 on full 33-D
  C2  condition-aware random      pair (Z_i with C_i=k) to (X_j with K_X_j=k)
                                  within each k-group, random order
  C3  semantic-cost OT            cost_ij = ||Z-X||^2 + lambda * 1[C != K_X]
                                  lambda grid {1, 3, 10, 30, 100}
  C4  geometry-only OT            cost_ij = ||Z_G - X_G||^2 (first 32 dims only),
                                  pair full state, metrics on full / geom / sem

kNN backend: sklearn.neighbors.NearestNeighbors(algorithm="auto").  In 33-D,
this picks BLAS-based "brute" which is ~9x faster than scipy.cKDTree.

Metrics per (coupling, seed, t) (full save list in JSON):
  Pair-level:        mismatch_rate, transport_full / _geom / _sem
  Unconditional kNN  full / geom / sem subspaces, per k in K_LIST
                     each with A_v_raw, A_v_norm, R_v^2, 1-R_v^2, H_KY,
                     A_within, A_between, R_switch
  Conditional kNN    per C in {-1, +1}, then weighted average,
                     same 8 metrics per subspace per k
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Thread caps: must be set before numpy/scipy/sklearn import.
# ---------------------------------------------------------------------------
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_k, "4")

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import linear_sum_assignment

try:
    import psutil
    _HAVE_PSUTIL = True
except ImportError:
    _HAVE_PSUTIL = False


# ---------------------------------------------------------------------------
# Config (env-var overridable)
# ---------------------------------------------------------------------------

D_G = 32
A = 0.5
SIGMA_S = 1.0
N_PAIRS = 20000
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # 10 seeds for headline phase2_10seeds
LAMBDAS = [1.0, 3.0, 10.0, 30.0, 100.0]
T_GRID = np.linspace(0.05, 0.95, 19)
K_LIST = [30, 50, 80, 120, 200]
K_MAX = max(K_LIST)
BATCH_SIZE_OT = 1024
INCLUDE_C4 = True
RUN_C3_LAM0_SANITY = False
C3INF_ONLY = False
INCLUDE_C3INF = True   # paper headline tab:e3 includes C3-infty row
SINKHORN_ONLY = False
INCLUDE_SINKHORN = False
SINKHORN_EPS = [0.03, 0.1, 0.3]
SINKHORN_LAMBDA = 10.0
SINKHORN_SAMPLE_MODE = "per_source"
SINKHORN_RESCALE = True
SLEEP_BETWEEN_COUPLINGS = 0.0
TAG = os.environ.get("E3C_TAG", "default")
DTYPE = np.float32                 # memory-efficient; metrics still accurate
EPS = 1e-12

# Threading is machine-dependent, not paper-dependent; keep env-overridable.
KNN_N_JOBS = int(os.environ.get("E3C_KNN_JOBS", "4"))

D = D_G + 1                        # total state dimension


# ---------------------------------------------------------------------------
# Process resource monitor
# ---------------------------------------------------------------------------

def _monitor_start():
    if not _HAVE_PSUTIL:
        return None
    p = psutil.Process()
    return {"t": time.time(), "cpu": p.cpu_times(), "p": p}


def _monitor_end(mon):
    if mon is None:
        return float("nan"), float("nan")
    p = mon["p"]
    dt = max(time.time() - mon["t"], 1e-9)
    new_cpu = p.cpu_times()
    cpu_total = (new_cpu.user - mon["cpu"].user
                 + new_cpu.system - mon["cpu"].system)
    cpu_pct = (cpu_total / dt) * 100.0
    rss = p.memory_info().rss / (1024 ** 2)
    return cpu_pct, rss


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------

def sample_data(N, seed):
    """E3 source/target/condition sampler.
    Returns:
      Z   (N, D)  float32   source samples
      X   (N, D)  float32   target samples
      C   (N,)    int64     external condition, exactly N/2 in {-1, +1}
      K_X (N,)    int64     target semantic label, exactly N/2 in {-1, +1}
    All marginals preserved: Z ~ N(0, I_D), C ⊥ Z, K_X ⊥ G.
    """
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((N, D)).astype(DTYPE)
    # External balanced condition, shuffled independently of Z
    C = np.concatenate([np.full(N // 2, -1, dtype=np.int64),
                        np.full(N - N // 2, 1, dtype=np.int64)])
    rng.shuffle(C)
    # Stratified balanced target labels, independent of C
    K_X = np.concatenate([np.full(N // 2, -1, dtype=np.int64),
                          np.full(N - N // 2, 1, dtype=np.int64)])
    rng.shuffle(K_X)
    G = rng.standard_normal((N, D_G)).astype(DTYPE)
    S = (A * K_X + SIGMA_S * rng.standard_normal(N)).astype(DTYPE)
    X = np.concatenate([G, S[:, None]], axis=1)
    return Z, X, C, K_X


# ---------------------------------------------------------------------------
# Couplings (each returns (Z_out, X_out, C_out, K_X_out))
# ---------------------------------------------------------------------------

def couple_independent(Z, X, C, K_X, rng):
    return Z, X, C, K_X


def _hungarian_pair_generic(Z, X, C, K_X, rng, batch_size, cost_fn):
    """Minibatch Hungarian with user-provided cost_fn(Z_b, X_b, C_b, KX_b).
    Re-pairs X (and its K_X label) within each batch; Z and C stay aligned."""
    n = len(Z)
    perm = rng.permutation(n)
    X_p = X.copy()
    K_X_p = K_X.copy()
    for b in range(0, n, batch_size):
        idx = perm[b: b + batch_size]
        Z_b = Z[idx]
        X_b = X[idx]
        C_b = C[idx]
        KX_b = K_X[idx]
        cost = cost_fn(Z_b, X_b, C_b, KX_b)
        # ensure float64 for linear_sum_assignment (it expects double)
        if cost.dtype != np.float64:
            cost = cost.astype(np.float64)
        row, col = linear_sum_assignment(cost)
        X_p[idx[row]] = X_b[col]
        K_X_p[idx[row]] = KX_b[col]
    return Z, X_p, C, K_X_p


def couple_euclidean_ot(Z, X, C, K_X, rng):
    """C1: full Euclidean OT on 33-D state."""
    def cost_fn(Zb, Xb, Cb, KXb):
        return ((Zb[:, None, :] - Xb[None, :, :]) ** 2).sum(-1)
    return _hungarian_pair_generic(Z, X, C, K_X, rng, BATCH_SIZE_OT, cost_fn)


def couple_semantic_cost_ot(Z, X, C, K_X, rng, lambda_sem):
    """C3: ||Z-X||^2 + lambda * 1[C != K_X]."""
    def cost_fn(Zb, Xb, Cb, KXb):
        c = ((Zb[:, None, :] - Xb[None, :, :]) ** 2).sum(-1)
        if lambda_sem > 0.0:
            mis = (Cb[:, None] != KXb[None, :]).astype(np.float64)
            c = c + lambda_sem * mis
        return c
    return _hungarian_pair_generic(Z, X, C, K_X, rng, BATCH_SIZE_OT, cost_fn)


def couple_geometry_only_ot(Z, X, C, K_X, rng):
    """C4: cost only on first d_g dims (nuisance geometry); pair full state."""
    def cost_fn(Zb, Xb, Cb, KXb):
        return ((Zb[:, None, :D_G] - Xb[None, :, :D_G]) ** 2).sum(-1)
    return _hungarian_pair_generic(Z, X, C, K_X, rng, BATCH_SIZE_OT, cost_fn)


def _class_balanced_batches(src_neg, src_pos, tgt_neg, tgt_pos, block_size):
    """Yield (src_neg, src_pos, tgt_neg, tgt_pos) index batches of size
    up to block_size each, including a tail batch with the remainder.
    All four arrays must be pre-shuffled by the caller."""
    n_per_class = min(len(src_neg), len(src_pos), len(tgt_neg), len(tgt_pos))
    start = 0
    while start < n_per_class:
        end = min(start + block_size, n_per_class)
        yield (src_neg[start:end], src_pos[start:end],
               tgt_neg[start:end], tgt_pos[start:end])
        start = end


def couple_c3_blocked_ot(Z, X, C, K_X, rng):
    """C3-infty: hard condition-blocked Hungarian OT on full-state Euclidean
    cost. Pairs source (C=k) with target (K_X=k) inside each k-block,
    using class-balanced batching plus a tail batch so all N samples are
    consumed. Returns paired (Z, X, C, K_X) with C == K_X enforced."""
    src_neg = np.where(C == -1)[0]
    src_pos = np.where(C == +1)[0]
    tgt_neg = np.where(K_X == -1)[0]
    tgt_pos = np.where(K_X == +1)[0]
    rng.shuffle(src_neg)
    rng.shuffle(src_pos)
    rng.shuffle(tgt_neg)
    rng.shuffle(tgt_pos)
    block_size = BATCH_SIZE_OT // 2  # 512 when BATCH_SIZE_OT == 1024
    parts_Z, parts_X, parts_C, parts_KX = [], [], [], []
    for sneg, spos, tneg, tpos in _class_balanced_batches(
            src_neg, src_pos, tgt_neg, tgt_pos, block_size):
        for sidx, tidx in [(sneg, tneg), (spos, tpos)]:
            Z_b = Z[sidx]
            X_b = X[tidx]
            cost = ((Z_b[:, None, :] - X_b[None, :, :]) ** 2).sum(-1)
            if cost.dtype != np.float64:
                cost = cost.astype(np.float64)
            row, col = linear_sum_assignment(cost)
            parts_Z.append(Z_b[row])
            parts_X.append(X_b[col])
            parts_C.append(C[sidx[row]])
            parts_KX.append(K_X[tidx[col]])
    Zp = np.concatenate(parts_Z)
    Xp = np.concatenate(parts_X)
    Cp = np.concatenate(parts_C)
    KXp = np.concatenate(parts_KX)
    assert (Cp == KXp).all(), \
        "C3-infty produced pairs with C != K_X; block construction bug"
    return Zp, Xp, Cp, KXp


def _sinkhorn_pair_generic(Z, X, C, K_X, rng, batch_size, cost_fn, eps,
                            sample_mode="plan"):
    """Per-batch entropic OT (Sinkhorn) + sample pairs from the plan.

    Each batch of size B computes a B x B Sinkhorn plan P with uniform
    marginals and regularization eps, then yields B paired tuples drawn
    from P. Two sampling modes:

      "plan" : sample B (i, j) joint pairs from P with replacement;
               source indices can repeat or be dropped.

      "per_source" : for each source index i, sample one target j from
               P[i,:] / row_sum_i; preserves source marginal exactly
               (Tong-2024-style rounding).
    """
    import ot
    n = len(Z)
    perm = rng.permutation(n)
    Zp = np.empty_like(Z)
    Xp = np.empty_like(X)
    Cp = np.empty_like(C)
    KXp = np.empty_like(K_X)
    for b in range(0, n, batch_size):
        idx = perm[b: b + batch_size]
        B = len(idx)
        Z_b = Z[idx]
        X_b = X[idx]
        C_b = C[idx]
        KX_b = K_X[idx]
        cost = cost_fn(Z_b, X_b, C_b, KX_b).astype(np.float64)
        if SINKHORN_RESCALE:
            scale = float(np.median(cost))
            if scale > EPS:
                cost = cost / scale
        a = np.ones(B) / B
        c = np.ones(B) / B
        P = ot.sinkhorn(a, c, cost, reg=float(eps),
                        numItermax=2000, stopThr=1e-7)
        if sample_mode == "plan":
            Pflat = P.flatten()
            Pflat = Pflat / max(Pflat.sum(), EPS)
            flat_idx = rng.choice(B * B, size=B, replace=True, p=Pflat)
            rows = flat_idx // B
            cols = flat_idx % B
        elif sample_mode == "per_source":
            rows = np.arange(B)
            row_sums = P.sum(axis=1, keepdims=True)
            row_probs = P / np.maximum(row_sums, EPS)
            cols = np.empty(B, dtype=np.int64)
            for i in range(B):
                cols[i] = rng.choice(B, p=row_probs[i])
        else:
            raise ValueError(f"unknown sample_mode={sample_mode!r}")
        Zp[b: b + B] = Z_b[rows]
        Xp[b: b + B] = X_b[cols]
        Cp[b: b + B] = C_b[rows]
        KXp[b: b + B] = KX_b[cols]
    return Zp, Xp, Cp, KXp


def couple_sinkhorn_euclidean(Z, X, C, K_X, rng, eps, sample_mode="plan"):
    """C1-Sinkhorn: entropic OT with full-state squared Euclidean cost."""
    def cost_fn(Zb, Xb, Cb, KXb):
        return ((Zb[:, None, :] - Xb[None, :, :]) ** 2).sum(-1)
    return _sinkhorn_pair_generic(Z, X, C, K_X, rng, BATCH_SIZE_OT,
                                  cost_fn, eps, sample_mode)


def couple_sinkhorn_semantic(Z, X, C, K_X, rng, eps, lambda_sem,
                              sample_mode="plan"):
    """C3-Sinkhorn at given lambda: entropic OT with semantic-cost penalty."""
    def cost_fn(Zb, Xb, Cb, KXb):
        c = ((Zb[:, None, :] - Xb[None, :, :]) ** 2).sum(-1)
        if lambda_sem > 0.0:
            mis = (Cb[:, None] != KXb[None, :]).astype(np.float64)
            c = c + lambda_sem * mis
        return c
    return _sinkhorn_pair_generic(Z, X, C, K_X, rng, BATCH_SIZE_OT,
                                  cost_fn, eps, sample_mode)


def couple_sinkhorn_geometry(Z, X, C, K_X, rng, eps, sample_mode="plan"):
    """C4-Sinkhorn: entropic OT on first D_G dims only; pair full state."""
    def cost_fn(Zb, Xb, Cb, KXb):
        return ((Zb[:, None, :D_G] - Xb[None, :, :D_G]) ** 2).sum(-1)
    return _sinkhorn_pair_generic(Z, X, C, K_X, rng, BATCH_SIZE_OT,
                                  cost_fn, eps, sample_mode)


def couple_condition_aware_random(Z, X, C, K_X, rng):
    """C2: pair within (C=k, K_X=k) groups, random order. Trim to balanced."""
    parts_Z, parts_X, parts_C, parts_KX = [], [], [], []
    for c in (-1, 1):
        z_idx = np.where(C == c)[0]
        x_idx = np.where(K_X == c)[0]
        n_match = min(len(z_idx), len(x_idx))
        z_sel = rng.permutation(z_idx)[:n_match]
        x_sel = rng.permutation(x_idx)[:n_match]
        parts_Z.append(Z[z_sel])
        parts_X.append(X[x_sel])
        parts_C.append(C[z_sel])
        parts_KX.append(K_X[x_sel])
    return (np.concatenate(parts_Z), np.concatenate(parts_X),
            np.concatenate(parts_C), np.concatenate(parts_KX))


# ---------------------------------------------------------------------------
# kNN + metrics
# ---------------------------------------------------------------------------

def knn_query(Y, k_max, n_jobs):
    """sklearn auto-mode kNN.  In 33-D, picks brute = BLAS matmul, ~9x faster
    than scipy cKDTree.  Returns indices (n_q, k_max).  Build is cheap; the
    cost is the kneighbors call."""
    nn = NearestNeighbors(n_neighbors=k_max, algorithm="auto", n_jobs=n_jobs)
    nn.fit(Y)
    return nn.kneighbors(Y, return_distance=False)


def _subspace_metrics(U_n, U_all_sub, shared):
    """Subspace-specific portion of the metric computation.
    Re-uses K-derived quantities already computed in `shared`.

    U_n        (n, k, d_sub):  neighborhood velocity (view OK)
    U_all_sub  (n, d_sub):     full sample for E||U_sub||^2
    shared     dict with k, mask_pos, mask_neg, n_pos, n_neg,
               safe_pos, safe_neg, H_KY
    """
    k = shared["k"]
    mask_pos = shared["mask_pos"]
    mask_neg = shared["mask_neg"]
    n_pos = shared["n_pos"]
    n_neg = shared["n_neg"]
    safe_pos = shared["safe_pos"]
    safe_neg = shared["safe_neg"]
    H_KY = shared["H_KY"]

    # ---- A_v_raw on this subspace ----
    mu_total = U_n.mean(axis=1, keepdims=True)            # (n, 1, d_sub)
    diff = U_n - mu_total
    A_v_raw = float(((diff ** 2).sum(axis=(1, 2)) / k).mean())
    mu_sq = (mu_total[:, 0, :] ** 2).sum(axis=-1)          # (n,)
    E_U_sq = float((U_all_sub ** 2).sum(axis=-1).mean())
    A_v_norm = A_v_raw / max(E_U_sq, EPS)
    R_v2 = float(mu_sq.mean()) / max(E_U_sq, EPS)

    # ---- within / between decomposition (biased 1/k so identity holds) ----
    sum_pos = (mask_pos[..., None] * U_n).sum(axis=1)
    sum_neg = (mask_neg[..., None] * U_n).sum(axis=1)
    overall = (sum_pos + sum_neg) / k
    mean_pos = np.where(n_pos[:, None] > 0,
                        sum_pos / safe_pos[:, None], overall)
    mean_neg = np.where(n_neg[:, None] > 0,
                        sum_neg / safe_neg[:, None], overall)
    diff_pos = (U_n - mean_pos[:, None, :]) * mask_pos[..., None]
    diff_neg = (U_n - mean_neg[:, None, :]) * mask_neg[..., None]
    ss_within = ((diff_pos ** 2).sum(axis=(1, 2))
                 + (diff_neg ** 2).sum(axis=(1, 2)))
    A_within = float((ss_within / k).mean())
    d_pos = mean_pos - overall
    d_neg = mean_neg - overall
    A_between = float(((n_pos * (d_pos ** 2).sum(axis=-1)
                        + n_neg * (d_neg ** 2).sum(axis=-1)) / k).mean())
    R_switch = A_between / max(A_within + A_between, EPS)

    return {
        "A_v_raw": A_v_raw,
        "A_v_norm": A_v_norm,
        "R_v2": R_v2,
        "one_minus_R_v2": 1.0 - R_v2,
        "H_KY": H_KY,
        "A_within": A_within,
        "A_between": A_between,
        "R_switch": R_switch,
    }


def _compute_per_k_for_subspaces(idxs, U_full, K_X_full, k_list):
    """Given kNN indices and a U sample, compute per-k metrics on
    full / geom / sem velocity subspaces.

    Optimization: K-derived quantities (mask_pos, n_pos, safe_pos, H_KY, ...)
    are computed once per k and shared across the three subspaces.  Memory
    peak: U_n_full = (n, k_max, D) materialised once.
    """
    k_max = max(k_list)
    U_n_full = U_full[idxs]                # (n, k_max, D); one copy
    K_n_full = K_X_full[idxs]              # (n, k_max)
    U_n_G = U_n_full[..., :D_G]            # view
    U_n_S = U_n_full[..., D_G:]            # view  (n, k_max, 1)
    U_all_G = U_full[:, :D_G]
    U_all_S = U_full[:, D_G:]

    per_k = {}
    for k in k_list:
        K_n_k = K_n_full[:, :k]
        # ---- K-derived shared quantities (computed once per k) ----
        mask_pos = (K_n_k == 1).astype(np.float32)
        mask_neg = 1.0 - mask_pos
        n_pos = mask_pos.sum(axis=1)
        n_neg = float(k) - n_pos
        safe_pos = np.maximum(n_pos, 1.0)
        safe_neg = np.maximum(n_neg, 1.0)
        # H(K | Y_t) using f_pos
        f_pos = n_pos / k
        H_per_q = -(np.where(f_pos > 0,
                             f_pos * np.log(np.clip(f_pos, EPS, 1.0)), 0.0)
                    + np.where(f_pos < 1,
                               (1.0 - f_pos)
                               * np.log(np.clip(1.0 - f_pos, EPS, 1.0)), 0.0))
        H_KY = float(H_per_q.mean())

        shared = {
            "k": k,
            "mask_pos": mask_pos, "mask_neg": mask_neg,
            "n_pos": n_pos, "n_neg": n_neg,
            "safe_pos": safe_pos, "safe_neg": safe_neg,
            "H_KY": H_KY,
        }

        per_k[k] = {
            "full": _subspace_metrics(U_n_full[:, :k], U_full, shared),
            "geom": _subspace_metrics(U_n_G[:, :k], U_all_G, shared),
            "sem": _subspace_metrics(U_n_S[:, :k], U_all_S, shared),
        }
    # Release the largest array explicitly to help GC between t-points
    del U_n_full
    return per_k


def estimate_one_t(Z, X, C, K_X, t):
    """For a single t, compute unconditional + conditional (per-C) metrics
    on full / geom / sem subspaces across all k in K_LIST."""
    Y = (1.0 - t) * Z + t * X
    U = X - Z

    out = {"t": float(t), "uncond": None, "cond_per_c": {}, "cond_avg": {}}

    # ---- Unconditional: kNN over full sample (N, 33) ----
    idxs = knn_query(Y, K_MAX, KNN_N_JOBS)
    out["uncond"] = _compute_per_k_for_subspaces(idxs, U, K_X, K_LIST)
    del idxs

    # ---- Conditional: per-C kNN within each C group ----
    ns_per_c = {}
    for c in (-1, 1):
        mask = (C == c)
        n_c = int(mask.sum())
        ns_per_c[c] = n_c
        if n_c < K_MAX:
            out["cond_per_c"][c] = {"n": n_c, "skipped": True}
            continue
        idxs_c = knn_query(Y[mask], K_MAX, KNN_N_JOBS)
        out["cond_per_c"][c] = {
            "n": n_c,
            "per_k": _compute_per_k_for_subspaces(idxs_c, U[mask],
                                                  K_X[mask], K_LIST),
        }
        del idxs_c

    # ---- Weighted-average conditional metrics over C ----
    n_total = sum(v["n"] for v in out["cond_per_c"].values()
                  if "skipped" not in v)
    if n_total > 0:
        cond_avg = {}
        for k in K_LIST:
            cond_avg[k] = {"full": {}, "geom": {}, "sem": {}}
            sample = out["cond_per_c"][next(c for c in (-1, 1)
                                            if "skipped" not in
                                            out["cond_per_c"][c])
                                       ]["per_k"][k]
            metric_keys = sample["full"].keys()
            for sub in ("full", "geom", "sem"):
                for mk in metric_keys:
                    total = 0.0
                    for c in (-1, 1):
                        cd = out["cond_per_c"][c]
                        if "skipped" in cd:
                            continue
                        total += (cd["n"] / n_total) * cd["per_k"][k][sub][mk]
                    cond_avg[k][sub][mk] = float(total)
        out["cond_avg"] = cond_avg

    return out


# ---------------------------------------------------------------------------
# Pair-level metrics
# ---------------------------------------------------------------------------

def pair_metrics(Z, X, C, K_X):
    diff = X - Z
    return {
        "mismatch_rate": float((K_X != C).mean()),
        "transport_full": float((diff ** 2).sum(-1).mean()),
        "transport_geom": float((diff[:, :D_G] ** 2).sum(-1).mean()),
        "transport_sem": float((diff[:, D_G:] ** 2).sum(-1).mean()),
        "n_pairs": int(len(Z)),
        "balance_C_plus": float((C == 1).mean()),
        "balance_KX_plus": float((K_X == 1).mean()),
    }


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------

def _rss_mb():
    if _HAVE_PSUTIL:
        return psutil.Process().memory_info().rss / (1024 ** 2)
    return float("nan")


def build_couplings():
    c3inf_entry = ("C3inf_blocked_ot", couple_c3_blocked_ot,
                   {"lambda": float("inf"),
                    "note": "hard condition-blocked Hungarian"})
    sinkhorn_entries = []
    for eps in SINKHORN_EPS:
        eps_tag = f"eps{eps:g}"
        sinkhorn_entries.append((
            f"C1_sinkhorn_{eps_tag}",
            (lambda Z, X, C, K_X, rng, _eps=eps:
             couple_sinkhorn_euclidean(Z, X, C, K_X, rng, _eps,
                                       SINKHORN_SAMPLE_MODE)),
            {"sinkhorn_eps": eps, "base": "C1",
             "sample_mode": SINKHORN_SAMPLE_MODE},
        ))
        sinkhorn_entries.append((
            f"C3_lam{SINKHORN_LAMBDA:g}_sinkhorn_{eps_tag}",
            (lambda Z, X, C, K_X, rng, _eps=eps, _lam=SINKHORN_LAMBDA:
             couple_sinkhorn_semantic(Z, X, C, K_X, rng, _eps, _lam,
                                      SINKHORN_SAMPLE_MODE)),
            {"sinkhorn_eps": eps, "lambda": SINKHORN_LAMBDA, "base": "C3",
             "sample_mode": SINKHORN_SAMPLE_MODE},
        ))
        sinkhorn_entries.append((
            f"C4_sinkhorn_{eps_tag}",
            (lambda Z, X, C, K_X, rng, _eps=eps:
             couple_sinkhorn_geometry(Z, X, C, K_X, rng, _eps,
                                      SINKHORN_SAMPLE_MODE)),
            {"sinkhorn_eps": eps, "base": "C4",
             "sample_mode": SINKHORN_SAMPLE_MODE},
        ))
    if C3INF_ONLY:
        return [c3inf_entry]
    if SINKHORN_ONLY:
        return sinkhorn_entries
    couplings = [
        ("C0_independent", couple_independent, {}),
        ("C1_euclidean_ot", couple_euclidean_ot, {}),
        ("C2_condition_aware_random", couple_condition_aware_random, {}),
    ]
    for lam in LAMBDAS:
        name = f"C3_semOT_lam{lam:g}"
        couplings.append((
            name,
            (lambda Z, X, C, K_X, rng, _lam=lam:
             couple_semantic_cost_ot(Z, X, C, K_X, rng, _lam)),
            {"lambda": lam},
        ))
    if INCLUDE_C4:
        couplings.append(("C4_geometry_only_ot",
                          couple_geometry_only_ot, {}))
    if INCLUDE_C3INF:
        couplings.append(c3inf_entry)
    if INCLUDE_SINKHORN:
        couplings.extend(sinkhorn_entries)
    if RUN_C3_LAM0_SANITY:
        couplings.append(("C3_semOT_lam0_eq_C1_sanity",
                          (lambda Z, X, C, K_X, rng:
                           couple_semantic_cost_ot(Z, X, C, K_X, rng, 0.0)),
                          {"lambda": 0.0, "note": "should equal C1"}))
    return couplings


def run():
    out_dir = Path(__file__).resolve().parents[2]
    fig_dir = out_dir / "figures"
    res_dir = out_dir / "results"
    fig_dir.mkdir(exist_ok=True)
    res_dir.mkdir(exist_ok=True)

    print(f"E3 conditional semantic mismatch  [tag={TAG}]")
    print(f"  D_G={D_G}, full D={D}, a={A}, sigma_s={SIGMA_S}")
    print(f"  N_PAIRS={N_PAIRS}, seeds={SEEDS}")
    print(f"  T_GRID={len(T_GRID)} pts in [{T_GRID[0]:.2f}, {T_GRID[-1]:.2f}]")
    print(f"  K_LIST={K_LIST}  (single sklearn kneighbors at k_max={K_MAX})")
    print(f"  LAMBDAS={LAMBDAS}  include_C4={INCLUDE_C4}")
    print(f"  BATCH_SIZE_OT={BATCH_SIZE_OT}  KNN_N_JOBS={KNN_N_JOBS}")
    print(f"  CPU threads cap (OMP)={os.environ.get('OMP_NUM_THREADS')}")
    print(f"  startup RSS={_rss_mb():.0f}MB")
    print("-" * 78)

    couplings = build_couplings()
    print(f"Couplings ({len(couplings)}): {[c[0] for c in couplings]}")
    print("-" * 78)

    results = {
        "config": {
            "D_G": D_G, "D": D, "A": A, "SIGMA_S": SIGMA_S,
            "N_PAIRS": N_PAIRS, "SEEDS": SEEDS, "LAMBDAS": LAMBDAS,
            "T_GRID": T_GRID.tolist(), "K_LIST": K_LIST,
            "BATCH_SIZE_OT": BATCH_SIZE_OT,
            "INCLUDE_C4": INCLUDE_C4,
            "RUN_C3_LAM0_SANITY": RUN_C3_LAM0_SANITY,
        },
        "runs": {},
    }

    t0_global = time.time()
    for seed in SEEDS:
        print(f"\n=== seed {seed} ===")
        Z, X, C, K_X = sample_data(N_PAIRS, seed)
        print(f"  Z marginal: mean={Z.mean(0).mean():.3f}, "
              f"std={Z.std(0).mean():.3f}  (should be ~0, ~1)")
        print(f"  C balance: +1 {(C == 1).mean():.3f}, "
              f"K_X balance: +1 {(K_X == 1).mean():.3f}")
        print(f"  pair-up before coupling: P(K_X==C)={(K_X == C).mean():.3f}")
        for ci, (name, fn, meta) in enumerate(couplings):
            mon = _monitor_start()
            t_start = time.time()
            rng_c = np.random.default_rng(seed * 1000 + ci + 7)
            Zp, Xp, Cp, KXp = fn(Z, X, C, K_X, rng_c)
            pm = pair_metrics(Zp, Xp, Cp, KXp)

            metrics_per_t = []
            t_inner_start = time.time()
            for ti, t in enumerate(T_GRID):
                tm = estimate_one_t(Zp, Xp, Cp, KXp, float(t))
                metrics_per_t.append(tm)
                if (ti + 1) % 5 == 0 or ti == len(T_GRID) - 1:
                    dt_inner = time.time() - t_inner_start
                    print(f"      [{name}] t {ti+1}/{len(T_GRID)} done  "
                          f"(elapsed {dt_inner:.1f}s,  rss {_rss_mb():.0f}MB)",
                          flush=True)
                    gc.collect()
            elapsed = time.time() - t_start
            cpu_pct, rss_mb = _monitor_end(mon)
            key = f"{name}_seed{seed}"
            results["runs"][key] = {
                "name": name,
                "seed": seed,
                "meta": meta,
                "pair_metrics": pm,
                "metrics_per_t": metrics_per_t,
                "wall_time_s": elapsed,
                "cpu_percent_during": cpu_pct,
                "rss_mb_after": rss_mb,
            }
            mid = len(T_GRID) // 2
            # Headline scalars at t=0.5, k=80
            uncond_full_mid = metrics_per_t[mid]["uncond"][80]["full"]
            cond_sem_mid = (metrics_per_t[mid]["cond_avg"]
                            .get(80, {"sem": {"A_v_norm": float("nan"),
                                              "H_KY": float("nan"),
                                              "R_switch": float("nan")}})
                            ["sem"])
            print(f"  [{ci+1:2d}/{len(couplings)}] {name:30s}"
                  f"  N={pm['n_pairs']:6d}  "
                  f"mismatch={pm['mismatch_rate']:.3f}  "
                  f"trans(full/geom/sem)={pm['transport_full']:.2f}/"
                  f"{pm['transport_geom']:.2f}/{pm['transport_sem']:.2f}  "
                  f"A_v_sem_norm(0.5|C)={cond_sem_mid['A_v_norm']:.3f}  "
                  f"H(K|Y,C)/log2={cond_sem_mid['H_KY']/np.log(2):.3f}  "
                  f"({elapsed:.1f}s  cpu={cpu_pct:.0f}%  rss={rss_mb:.0f}MB)")
            is_last = (ci == len(couplings) - 1 and seed == SEEDS[-1])
            if SLEEP_BETWEEN_COUPLINGS > 0 and not is_last:
                print(f"      ... sleep {SLEEP_BETWEEN_COUPLINGS:.0f}s")
                time.sleep(SLEEP_BETWEEN_COUPLINGS)

    total = time.time() - t0_global
    print(f"\nTotal wall time: {total:.1f}s ({total / 60:.1f} min)")

    out_json = res_dir / f"e3_metrics_{TAG}.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"Wrote {out_json}")

    make_figures(results, fig_dir)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def make_figures(results, fig_dir):
    runs = results["runs"]
    by_name = {}
    for key, r in runs.items():
        by_name.setdefault(r["name"], []).append(r)

    names = list(by_name.keys())
    cmap = plt.get_cmap("tab10")
    name_color = {n: cmap(i % 10) for i, n in enumerate(names)}

    def metric_over_t(run, kind, k, subspace, mk):
        """Extract metric scalar over t-grid.
        kind in {"uncond", "cond_avg"}; subspace in {"full","geom","sem"}.
        """
        out = []
        for tm in run["metrics_per_t"]:
            if kind == "uncond":
                out.append(tm["uncond"][k][subspace][mk])
            elif kind == "cond_avg":
                cd = tm.get("cond_avg", {})
                if k in cd:
                    out.append(cd[k][subspace][mk])
                else:
                    out.append(float("nan"))
        return np.array(out)

    def average_over_seeds(name, *args, **kwargs):
        stacks = [metric_over_t(r, *args, **kwargs) for r in by_name[name]]
        return np.mean(stacks, axis=0), np.std(stacks, axis=0, ddof=1) \
            if len(stacks) > 1 else (np.mean(stacks, axis=0), None)

    # =====================================================================
    # Figure 1 (main): 4-panel summary
    #   Panel 1: mismatch_rate by coupling
    #   Panel 2: max_t A_v_sem_norm(t | C) by coupling
    #   Panel 3: max_t H(K_X|Y_t,C)/log2 by coupling
    #   Panel 4: transport_cost_full vs max_t A_v_sem_norm(t|C) (Pareto)
    # =====================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: mismatch_rate
    ax = axes[0, 0]
    names_sorted = sorted(names, key=lambda n: by_name[n][0]["meta"].get(
        "lambda", -1 if "C0" in n else (-0.5 if "C1" in n else (
            -0.25 if "C2" in n else (-0.1 if "C4" in n else 0)))))
    mismatch = [np.mean([r["pair_metrics"]["mismatch_rate"]
                         for r in by_name[n]]) for n in names_sorted]
    ax.bar(range(len(names_sorted)), mismatch,
           color=[name_color[n] for n in names_sorted])
    ax.set_xticks(range(len(names_sorted)))
    ax.set_xticklabels(names_sorted, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel(r"$\Pr(K_X \neq C)$ after coupling")
    ax.set_title(r"(a) mismatch rate by coupling")
    ax.axhline(0.5, color="gray", ls=":", alpha=0.5, label="random = 0.5")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 2: max_t A_v_sem_norm(t | C)
    ax = axes[0, 1]
    max_av_sem = []
    for n in names_sorted:
        vals = []
        for r in by_name[n]:
            arr = metric_over_t(r, "cond_avg", 80, "sem", "A_v_norm")
            vals.append(np.nanmax(arr))
        max_av_sem.append(np.mean(vals))
    ax.bar(range(len(names_sorted)), max_av_sem,
           color=[name_color[n] for n in names_sorted])
    ax.set_xticks(range(len(names_sorted)))
    ax.set_xticklabels(names_sorted, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel(r"$\max_t\,\mathcal{A}_v^S(t|C) / E\|U_S\|^2$")
    ax.set_title(r"(b) max-t semantic-only conditional ambiguity")
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 3: max_t H(K_X|Y_t,C) / log 2
    ax = axes[1, 0]
    max_H = []
    for n in names_sorted:
        vals = []
        for r in by_name[n]:
            arr = metric_over_t(r, "cond_avg", 80, "sem", "H_KY") / np.log(2.0)
            vals.append(np.nanmax(arr))
        max_H.append(np.mean(vals))
    ax.bar(range(len(names_sorted)), max_H,
           color=[name_color[n] for n in names_sorted])
    ax.set_xticks(range(len(names_sorted)))
    ax.set_xticklabels(names_sorted, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel(r"$\max_t\,H(K_X|Y_t, C)/\log 2$")
    ax.set_title("(c) max-t conditional branch entropy")
    ax.axhline(1.0, color="gray", ls=":", alpha=0.5, label="log 2 (max)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 4 (Pareto): transport_cost_full vs max_t A_v_sem_norm(t|C)
    ax = axes[1, 1]
    for n in names_sorted:
        tcs = [r["pair_metrics"]["transport_full"] for r in by_name[n]]
        max_avs = []
        for r in by_name[n]:
            arr = metric_over_t(r, "cond_avg", 80, "sem", "A_v_norm")
            max_avs.append(np.nanmax(arr))
        ax.errorbar(np.mean(tcs), np.mean(max_avs),
                    xerr=(np.std(tcs, ddof=1) if len(tcs) > 1 else 0),
                    yerr=(np.std(max_avs, ddof=1) if len(max_avs) > 1 else 0),
                    fmt="o", ms=9, capsize=4,
                    color=name_color[n], label=n)
    ax.set_xlabel(r"$T_{\rm full}$ (raw Euclidean transport)")
    ax.set_ylabel(r"$\max_t\,\mathcal{A}_v^S(t|C) / E\|U_S\|^2$")
    ax.set_title("(d) Pareto: geometric cost vs semantic ambiguity")
    ax.legend(loc="best", fontsize=7)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"E3 main ({TAG}): N={N_PAIRS}, d_g={D_G}, a={A}, "
                 f"sigma_s={SIGMA_S}",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(fig_dir / f"e3_main_{TAG}.png", dpi=120)
    plt.close(fig)

    # =====================================================================
    # Figure 2 (curves): 4-panel t-curves at k=80
    #   Panel 1: A_v_full_norm(t)  cond_avg
    #   Panel 2: A_v_sem_norm(t)   cond_avg
    #   Panel 3: H(K_X|Y_t,C)/log2 cond_avg
    #   Panel 4: R_switch_sem(t)   cond_avg
    # =====================================================================
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
    t_arr = np.array([tm["t"] for tm in next(iter(runs.values()))["metrics_per_t"]])

    def plot_curve(ax, kind, k, subspace, mk, scale=1.0):
        for n in names_sorted:
            stacks = [metric_over_t(r, kind, k, subspace, mk) * scale
                      for r in by_name[n]]
            mean_curve = np.nanmean(stacks, axis=0)
            ax.plot(t_arr, mean_curve, "o-", lw=1.5, ms=4,
                    color=name_color[n], alpha=0.85, label=n)

    ax = axes2[0, 0]
    plot_curve(ax, "cond_avg", 80, "full", "A_v_norm")
    ax.set_xlabel("t")
    ax.set_ylabel(r"$\mathcal{A}_v(t|C) / E\|U\|^2$")
    ax.set_title(r"(a) full-state conditional $\widetilde{\mathcal{A}}_v(t\,|\,C)$")
    ax.legend(loc="best", fontsize=7)
    ax.grid(True, alpha=0.3)

    ax = axes2[0, 1]
    plot_curve(ax, "cond_avg", 80, "sem", "A_v_norm")
    ax.set_xlabel("t")
    ax.set_ylabel(r"$\mathcal{A}_v^S(t|C) / E\|U_S\|^2$")
    ax.set_title(r"(b) semantic-only conditional $\widetilde{\mathcal{A}}_v^S(t\,|\,C)$")
    ax.legend(loc="best", fontsize=7)
    ax.grid(True, alpha=0.3)

    ax = axes2[1, 0]
    plot_curve(ax, "cond_avg", 80, "sem", "H_KY", scale=1.0 / np.log(2.0))
    ax.set_xlabel("t")
    ax.set_ylabel(r"$H(K_X|Y_t,C)/\log 2$ (sem subspace)")
    ax.set_title("(c) semantic-subspace conditional branch entropy")
    ax.axhline(1.0, color="gray", ls=":", alpha=0.5)
    ax.legend(loc="best", fontsize=7)
    ax.grid(True, alpha=0.3)

    ax = axes2[1, 1]
    plot_curve(ax, "cond_avg", 80, "sem", "R_switch")
    ax.set_xlabel("t")
    ax.set_ylabel(r"$R_{\rm switch}(t|C)$")
    ax.set_title(r"(d) semantic-only $R_{\rm switch}(t\,|\,C)$")
    ax.legend(loc="best", fontsize=7)
    ax.grid(True, alpha=0.3)

    fig2.suptitle(f"E3 curves ({TAG}): conditional t-profiles at k=80",
                  fontsize=12)
    fig2.tight_layout()
    fig2.savefig(fig_dir / f"e3_curves_{TAG}.png", dpi=120)
    plt.close(fig2)

    print(f"Wrote {fig_dir / f'e3_main_{TAG}.png'}")
    print(f"Wrote {fig_dir / f'e3_curves_{TAG}.png'}")


if __name__ == "__main__":
    run()
