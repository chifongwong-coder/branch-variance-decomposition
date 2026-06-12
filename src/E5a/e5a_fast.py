"""E5a-specific fast estimator helpers.

decompose_fast is now canonical in branch_decomp.py; this module re-exports
it for back-compat with callers that imported `from e5a_fast import
decompose_fast`. The unique-to-E5a piece is a2_observed_support_fast
(observed-support theorem-bound test), which lives only here because it
implements Proposition `proxy-tv-bound`-specific logic not reused by E3b.
"""

from __future__ import annotations

import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.branch_decomp import decompose_fast  # canonical, re-exported here


def a2_observed_support_fast(U_neighbors: np.ndarray,
                              K_neighbors: np.ndarray,
                              p_list: list[float],
                              K: int = 10) -> dict:
    """Observed-support perturbation test, vectorised.

    For each anchor i:
      S_i           = {k : n_{i,k} > 0}
      r_{i,k}       = n_{i,k} / k_nn
      m_{i,k}       = mean of U_j over j with K_j = k (for k in S_i)
      u_{S_i,k}     = 1/|S_i| if k in S_i else 0
      r_hat_{p,i,k} = (1 - p) r_{i,k} + p u_{S_i,k}

    Avoids (N, k, d) intermediates by:
      - vectorising group means with einsum
      - using parallel-axis for sum of squared deviations
    """
    U_neighbors = np.asarray(U_neighbors, dtype=np.float64)
    K_neighbors = np.asarray(K_neighbors)
    N, knn, d = U_neighbors.shape

    # Per-class one-hot mask stack: (N, K, knn). 10k * 10 * 80 = 8 MB at K=10.
    masks = np.stack([(K_neighbors == c).astype(np.float64) for c in range(K)],
                      axis=1)                                       # (N, K, knn)
    counts = masks.sum(axis=-1)                                     # (N, K) = n_{i,k}
    in_S = (counts > 0)                                              # (N, K)
    safe_n = np.where(counts > 0, counts, 1.0)

    # m_{i,k} for all k: vectorised group means.
    # sum_per_class[i, c, d] = sum_j masks[i, c, j] * U_n[i, j, d]
    sum_per_class = np.einsum("icj,ijd->icd", masks, U_neighbors,
                                optimize=True)                       # (N, K, d)
    m = sum_per_class / safe_n[..., None]                            # (N, K, d)
    m = m * in_S[..., None].astype(np.float64)                       # zero outside S_i

    # M_hat^2 (max ||m_{i,k}||^2 over (i, k in S_i))
    m_norm_sq = (m ** 2).sum(axis=-1)                                # (N, K)
    nonzero = m_norm_sq[in_S]                                        # 1-D
    M_hat_sq_global = float(nonzero.max())
    M_hat_sq_q99 = float(np.quantile(nonzero, 0.99))
    M_i_sq = m_norm_sq.max(axis=1)                                   # (N,)

    # local posteriors r and observed-support uniform u
    r = counts / knn                                                 # (N, K)
    S_size = in_S.sum(axis=1).astype(np.float64)
    safe_S = np.where(S_size > 0, S_size, 1.0)
    u = in_S.astype(np.float64) / safe_S[:, None]

    # Centroid-based per-anchor Chebyshev-radius upper bound (Remark rem:proxy-tv-radius)
    # R_i^2 = max_{k in S_i} ||m_{i,k} - c_i||^2 where c_i = (1/|S_i|) sum_{k in S_i} m_{i,k}.
    # This is an upper bound on the true Chebyshev radius (centroid is one valid choice of c),
    # so 3 R_i^2 is a valid translation-invariant tightening of 3 M_{i,k}^2 (with R_i <= M_i).
    c_centroid = (m * in_S[..., None].astype(np.float64)).sum(axis=1) / safe_S[:, None]  # (N, d)
    diff_c = (m - c_centroid[:, None, :])                            # (N, K, d)
    diff_c_norm_sq = (diff_c ** 2).sum(axis=-1)                      # (N, K)
    # Only k in S_i contribute; outside S_i set to -1 so max ignores them.
    diff_c_masked = np.where(in_S, diff_c_norm_sq, -1.0)
    R_i_sq = diff_c_masked.max(axis=1)                               # (N,) per-anchor R_centroid^2
    R_hat_sq_global = float(R_i_sq.max())
    R_hat_sq_q99 = float(np.quantile(R_i_sq, 0.99))

    # Compute reference B_r:
    # B_r_i = sum_k r_{i,k} ||m_{i,k} - bar_m_r_i||^2
    # Use parallel-axis style: B_r_i = sum_k r_{i,k} ||m_{i,k}||^2 - ||bar_m_r_i||^2
    # because bar_m_r_i = sum_k r_{i,k} m_{i,k} and r_{i,k} sums to 1.
    # Proof:  sum r_k ||m_k - bar_m||^2
    #       = sum r_k (||m_k||^2 - 2 m_k·bar_m + ||bar_m||^2)
    #       = sum r_k ||m_k||^2 - 2 bar_m·(sum r_k m_k) + ||bar_m||^2 sum r_k
    #       = sum r_k ||m_k||^2 - 2 ||bar_m||^2 + ||bar_m||^2
    #       = sum r_k ||m_k||^2 - ||bar_m||^2.
    bar_m_r = (r[..., None] * m).sum(axis=1)                         # (N, d)
    bar_m_r_sq = (bar_m_r ** 2).sum(axis=-1)                         # (N,)
    B_r = (r * m_norm_sq).sum(axis=1) - bar_m_r_sq                   # (N,)

    out = {
        "M_hat_sq_global": M_hat_sq_global,
        "M_hat_sq_q99":    M_hat_sq_q99,
        "R_hat_sq_global": R_hat_sq_global,
        "R_hat_sq_q99":    R_hat_sq_q99,
    }
    for p in p_list:
        r_hat = (1.0 - p) * r + p * u
        bar_m_h = (r_hat[..., None] * m).sum(axis=1)                 # (N, d)
        bar_m_h_sq = (bar_m_h ** 2).sum(axis=-1)
        B_h = (r_hat * m_norm_sq).sum(axis=1) - bar_m_h_sq           # (N,)

        LHS_per = np.abs(B_r - B_h)
        L1_per = np.abs(r - r_hat).sum(axis=1)

        LHS = float(LHS_per.mean())
        L1_mean = float(L1_per.mean())
        # 3 M^2 base bound (Proposition prop:proxy-tv-bound, sup-norm form)
        RHS_global = 3.0 * M_hat_sq_global * L1_mean
        RHS_local  = 3.0 * float((M_i_sq * L1_per).mean())
        RHS_q99    = 3.0 * M_hat_sq_q99 * L1_mean
        # 3 R^2 translation-invariant tightening (Remark rem:proxy-tv-radius)
        RHS_R_global = 3.0 * R_hat_sq_global * L1_mean
        RHS_R_local  = 3.0 * float((R_i_sq * L1_per).mean())
        RHS_R_q99    = 3.0 * R_hat_sq_q99 * L1_mean

        out[f"p={p}"] = {
            "p":           float(p),
            "LHS":         LHS,
            "RHS_global":  RHS_global,
            "RHS_local":   RHS_local,
            "RHS_q99":     RHS_q99,
            "RHS_R_global": RHS_R_global,
            "RHS_R_local":  RHS_R_local,
            "RHS_R_q99":    RHS_R_q99,
            "ratio_global": RHS_global / max(LHS, 1e-30),
            "ratio_local":  RHS_local  / max(LHS, 1e-30),
            "ratio_q99":    RHS_q99    / max(LHS, 1e-30),
            "ratio_R_global": RHS_R_global / max(LHS, 1e-30),
            "ratio_R_local":  RHS_R_local  / max(LHS, 1e-30),
            "ratio_R_q99":    RHS_R_q99    / max(LHS, 1e-30),
            "L1_mean":     L1_mean,
            "B_r_mean":    float(B_r.mean()),
            "B_h_mean":    float(B_h.mean()),
        }
    return out
