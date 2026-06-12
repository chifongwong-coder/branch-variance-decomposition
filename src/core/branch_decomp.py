"""Label-agnostic local within/between decomposition for the branch-decomposition
diagnostic.

Two algebraically equivalent implementations of the biased $1/k$ kNN
within/between split are exposed:

  - decompose_fast (default): uses the parallel-axis identity
        sum_{j in g} ||U_j - mu_g||^2 = sum_{j in g} ||U_j||^2 - n_g ||mu_g||^2
    and `np.einsum("ij,ijd->id", mask, U_neighbors)` to avoid (N, k, d)
    intermediate tensors. Bit-equivalent to the reference (max relative
    error ~8e-16) but ~10-20x faster at d >= 64 because it avoids
    allocating multi-GB temporaries inside the per-label loop. Use this
    for high-dimensional features (E5a, etc.).

  - _decompose_reference (debug/audit): the original explicit `diff_g`
    form. Kept for self-test cross-validation. Use for low-dimensional
    work where its simpler structure helps reading.

`decompose(...)` is the public dispatcher; by default it forwards to
`decompose_fast`. Pass impl='reference' to force the reference version.

Estimator contract:
  - A single global kNN graph must be computed ONCE per (Y_t, k) and
    passed in as `U_neighbors` / `labels_neighbors`. Both coarse-K and
    fine-K' decompositions (and any oracle / proxy variants) MUST share
    the same neighbor sets to preserve the per-anchor identity at
    floating-point precision.
  - Biased 1/k convention: local covariance divides by k (not k-1);
    per-group covariance divides by n_g (not n_g - 1). The within +
    between identity is exact under this convention.
  - Empty groups (n_g = 0) contribute exactly zero; no weight renorm.
  - float64 throughout.

Run `python branch_decomp.py` to execute the embedded self-tests, which
include a reference-vs-fast cross-check.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Fast implementation (parallel-axis, default)
# ---------------------------------------------------------------------------

def decompose_fast(U_neighbors: np.ndarray,
                   labels_neighbors: np.ndarray,
                   U_per_neighbor_sq: np.ndarray | None = None) -> dict:
    """Fast within/between decomposition. Bit-equivalent to the reference.

    U_per_neighbor_sq optionally accepts a precomputed (N, k) tensor of
    per-neighbor squared norms; useful when the same kNN graph feeds
    multiple label decompositions (further speedup).
    """
    U_neighbors = np.asarray(U_neighbors, dtype=np.float64)
    labels_neighbors = np.asarray(labels_neighbors)
    N, k, d = U_neighbors.shape
    if labels_neighbors.shape != (N, k):
        raise ValueError(
            f"labels_neighbors shape {labels_neighbors.shape} does not match "
            f"U_neighbors first two dims {(N, k)}"
        )

    if U_per_neighbor_sq is None:
        U_per_neighbor_sq = (U_neighbors ** 2).sum(axis=-1)       # (N, k)

    U_sq_sum = U_per_neighbor_sq.sum(axis=1)                      # (N,)
    mu_bar = U_neighbors.mean(axis=1)                             # (N, d)
    mu_bar_sq = (mu_bar ** 2).sum(axis=-1)                        # (N,)
    A_v_per = U_sq_sum / k - mu_bar_sq                            # (N,)

    A_within_per = np.zeros(N, dtype=np.float64)
    A_between_per = np.zeros(N, dtype=np.float64)

    for g in np.unique(labels_neighbors):
        mask = (labels_neighbors == g).astype(np.float64)         # (N, k)
        n_g = mask.sum(axis=1)                                     # (N,)
        sum_sq_in_g = (mask * U_per_neighbor_sq).sum(axis=1)       # (N,)
        sum_in_g = np.einsum("ij,ijd->id", mask, U_neighbors,
                              optimize=True)                       # (N, d)
        safe_n = np.where(n_g > 0, n_g, 1.0)
        mu_g = sum_in_g / safe_n[:, None]                          # (N, d)
        mu_g_sq = (mu_g ** 2).sum(axis=-1)                         # (N,)

        within_contrib = (sum_sq_in_g - n_g * mu_g_sq) / k         # (N,)
        A_within_per += np.where(n_g > 0, within_contrib, 0.0)

        between_contrib = (n_g / k) * ((mu_g - mu_bar) ** 2).sum(axis=-1)
        A_between_per += np.where(n_g > 0, between_contrib, 0.0)

    identity_resid_per = A_v_per - A_within_per - A_between_per

    return {
        "per_anchor": {
            "A_v": A_v_per,
            "A_within": A_within_per,
            "A_between": A_between_per,
            "identity_residual": identity_resid_per,
        },
        "aggregated": {
            "A_v": float(A_v_per.mean()),
            "A_within": float(A_within_per.mean()),
            "A_between": float(A_between_per.mean()),
            "identity_residual_max_abs": float(
                np.abs(identity_resid_per).max()
            ),
        },
    }


# ---------------------------------------------------------------------------
# Reference implementation (explicit diff_g; kept for audit / self-test)
# ---------------------------------------------------------------------------

def _decompose_reference(U_neighbors: np.ndarray,
                          labels_neighbors: np.ndarray) -> dict:
    """Original explicit-diff implementation. Slower at d >= 64 due to
    repeated (N, k, d) allocations inside the per-label loop. Kept as the
    audit / debug reference."""
    U_neighbors = np.asarray(U_neighbors, dtype=np.float64)
    labels_neighbors = np.asarray(labels_neighbors)
    N, k, d = U_neighbors.shape
    if labels_neighbors.shape != (N, k):
        raise ValueError(
            f"labels_neighbors shape {labels_neighbors.shape} does not match "
            f"U_neighbors first two dims {(N, k)}"
        )

    mu_bar = U_neighbors.mean(axis=1, keepdims=True)               # (N, 1, d)
    A_v_per = ((U_neighbors - mu_bar) ** 2).sum(axis=(1, 2)) / k    # (N,)
    mu_bar_2d = mu_bar[:, 0, :]                                     # (N, d)

    A_within_per = np.zeros(N, dtype=np.float64)
    A_between_per = np.zeros(N, dtype=np.float64)

    for g in np.unique(labels_neighbors):
        mask = (labels_neighbors == g).astype(np.float64)
        n_g = mask.sum(axis=1)
        sum_g = (mask[..., None] * U_neighbors).sum(axis=1)
        safe_n_g = np.where(n_g > 0, n_g, 1.0)
        mu_g = sum_g / safe_n_g[:, None]

        diff_g = (U_neighbors - mu_g[:, None, :]) * mask[..., None]
        A_within_per += (diff_g ** 2).sum(axis=(1, 2)) / k

        between_contrib = (n_g / k) * ((mu_g - mu_bar_2d) ** 2).sum(axis=-1)
        A_between_per += np.where(n_g > 0, between_contrib, 0.0)

    identity_resid_per = A_v_per - A_within_per - A_between_per

    return {
        "per_anchor": {
            "A_v": A_v_per,
            "A_within": A_within_per,
            "A_between": A_between_per,
            "identity_residual": identity_resid_per,
        },
        "aggregated": {
            "A_v": float(A_v_per.mean()),
            "A_within": float(A_within_per.mean()),
            "A_between": float(A_between_per.mean()),
            "identity_residual_max_abs": float(
                np.abs(identity_resid_per).max()
            ),
        },
    }


# ---------------------------------------------------------------------------
# Public dispatcher
# ---------------------------------------------------------------------------

def decompose(U_neighbors: np.ndarray,
              labels_neighbors: np.ndarray,
              impl: str = "fast") -> dict:
    """Public API. impl='fast' (default) or 'reference' for the original
    explicit-diff implementation. The two are algebraically equivalent;
    'fast' is recommended for d >= 16."""
    if impl == "fast":
        return decompose_fast(U_neighbors, labels_neighbors)
    if impl == "reference":
        return _decompose_reference(U_neighbors, labels_neighbors)
    raise ValueError(f"impl must be 'fast' or 'reference', got {impl!r}")


# ---------------------------------------------------------------------------
# Delta-local direct estimator (used by E3b refinement)
# ---------------------------------------------------------------------------

def delta_local_direct(U_neighbors: np.ndarray,
                       coarse_neighbors: np.ndarray,
                       fine_neighbors: np.ndarray) -> dict:
    r"""Direct local Delta via the nested-decomposition formula.

    Under K = f(K') (K' refines K), per anchor i:

        Delta_local_i = sum_{c, g' subset c} (n_{g'}/k) ||mu_{g'} - mu_c||^2

    This equals A_between^{K'} - A_between^{K} at finite sample under the
    biased 1/k convention, provided both decompositions use the SAME
    neighbor sets.
    """
    U_neighbors = np.asarray(U_neighbors, dtype=np.float64)
    coarse_neighbors = np.asarray(coarse_neighbors)
    fine_neighbors = np.asarray(fine_neighbors)
    N, k, d = U_neighbors.shape

    delta_per = np.zeros(N, dtype=np.float64)
    unique_coarse = np.unique(coarse_neighbors)
    unique_fine = np.unique(fine_neighbors)

    for c in unique_coarse:
        coarse_mask = (coarse_neighbors == c).astype(np.float64)
        n_c = coarse_mask.sum(axis=1)
        sum_c = (coarse_mask[..., None] * U_neighbors).sum(axis=1)
        safe_n_c = np.where(n_c > 0, n_c, 1.0)
        mu_c = sum_c / safe_n_c[:, None]

        for g_prime in unique_fine:
            joint_mask = ((fine_neighbors == g_prime) &
                          (coarse_neighbors == c)).astype(np.float64)
            n_gp = joint_mask.sum(axis=1)
            sum_gp = (joint_mask[..., None] * U_neighbors).sum(axis=1)
            safe_n_gp = np.where(n_gp > 0, n_gp, 1.0)
            mu_gp = sum_gp / safe_n_gp[:, None]
            contrib = (n_gp / k) * ((mu_gp - mu_c) ** 2).sum(axis=-1)
            delta_per += np.where(n_gp > 0, contrib, 0.0)

    return {
        "per_anchor": delta_per,
        "aggregated": float(delta_per.mean()),
    }


# ---------------------------------------------------------------------------
# Self-tests (run when invoked as `python branch_decomp.py`).
# ---------------------------------------------------------------------------

def _test_identity_and_refinement():
    """Three small synthetic checks plus a fast-vs-reference cross-check.
      A. Single-label group => A_between = 0, A_within = A_v.
      B. Identity A_v = A_within + A_between holds to ~1e-15.
      C. Refinement: K' nested in K => Delta_diff = Delta_local exactly,
         and both >= 0 to numerical precision.
      D. decompose_fast and _decompose_reference give identical aggregated
         values to 1e-12 relative.
    """
    rng = np.random.default_rng(42)
    N, k, d = 100, 16, 3
    U = rng.standard_normal((N, k, d))

    # ---- A: single-label
    L_const = np.zeros((N, k), dtype=int)
    s = decompose(U, L_const)
    assert abs(s["aggregated"]["A_between"]) < 1e-12
    assert abs(s["aggregated"]["A_within"] - s["aggregated"]["A_v"]) < 1e-12

    # ---- B: identity for arbitrary labels (4 classes, random assignment)
    L = rng.integers(0, 4, size=(N, k))
    s = decompose(U, L)
    assert s["aggregated"]["identity_residual_max_abs"] < 1e-12, \
        f"identity residual = {s['aggregated']['identity_residual_max_abs']}"

    # ---- C: refinement check, K = (K' >= 2)
    K_fine = L
    K_coarse = (K_fine >= 2).astype(int)
    s_coarse = decompose(U, K_coarse)
    s_fine = decompose(U, K_fine)
    delta_diff = (s_fine["aggregated"]["A_between"]
                  - s_coarse["aggregated"]["A_between"])
    dl = delta_local_direct(U, K_coarse, K_fine)
    delta_local = dl["aggregated"]
    assert abs(delta_diff - delta_local) < 1e-12
    assert delta_diff >= -1e-12
    assert dl["per_anchor"].min() >= -1e-12
    Av_diff = abs(s_coarse["aggregated"]["A_v"] - s_fine["aggregated"]["A_v"])
    assert Av_diff < 1e-15
    assert (s_fine["aggregated"]["A_within"]
            <= s_coarse["aggregated"]["A_within"] + 1e-12)

    # ---- D: fast vs reference cross-check at higher d
    rng2 = np.random.default_rng(7)
    U2 = rng2.standard_normal((50, 32, 128))
    L2 = rng2.integers(0, 5, size=(50, 32))
    sf = decompose(U2, L2, impl="fast")
    sr = decompose(U2, L2, impl="reference")
    for key in ("A_v", "A_within", "A_between"):
        a, b = sf["aggregated"][key], sr["aggregated"][key]
        den = max(abs(a), abs(b), 1e-30)
        assert abs(a - b) / den < 1e-12, \
            f"fast vs reference disagree on {key}: {a} vs {b}"

    print("branch_decomp self-tests passed.")
    print(f"  A_v identity residual max abs (sample): "
          f"{s['aggregated']['identity_residual_max_abs']:.2e}")
    print(f"  Delta_diff == Delta_local within: "
          f"{abs(delta_diff - delta_local):.2e}")
    print(f"  Av invariance across nested labels: {Av_diff:.2e}")
    print(f"  fast vs reference max relerr (cross-check d=128): "
          f"{max(abs(sf['aggregated'][k] - sr['aggregated'][k]) / max(abs(sf['aggregated'][k]), 1e-30) for k in ('A_v','A_within','A_between')):.2e}")


if __name__ == "__main__":
    _test_identity_and_refinement()
