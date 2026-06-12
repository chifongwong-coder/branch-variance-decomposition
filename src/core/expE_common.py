"""Shared helpers for the supplementary experiments (expE7a..expE5b).

These reuse the verified core estimator ``branch_decomp.decompose_fast`` and a
plain brute-force kNN graph. Everything here is CPU/numpy except where a caller
passes already-computed features. Kept deliberately small and self-contained so
the overnight E-suite has as few moving parts as possible.
"""
from __future__ import annotations

import os

# Thermal / thread caps. Import-time so any script importing this is capped.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "NUMEXPR_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from core.branch_decomp import decompose_fast  # noqa: E402

RES = HERE.parents[1] / "results"
RES.mkdir(exist_ok=True)


def knn_indices(Y, k):
    """(N,k) neighbor index array from a brute-force L2 kNN graph on Y (N,d)."""
    from sklearn.neighbors import NearestNeighbors
    Y = np.ascontiguousarray(Y, dtype=np.float32)
    nn = NearestNeighbors(n_neighbors=k, algorithm="brute").fit(Y)
    return nn.kneighbors(Y, return_distance=False)


def abetween_chunked(U, nn, labels, chunk=500):
    """A_between (and A_v, A_within) over a fixed kNN graph, chunked over
    anchors to cap the (chunk,k,d) float64 transient. Returns the aggregated
    means over all anchors.

    U      (N,d)   velocity vectors
    nn     (N,k)   neighbor indices into U
    labels (N,)    branch label of each row of U
    """
    U = np.ascontiguousarray(U, dtype=np.float64)
    labels = np.asarray(labels)
    N, k = nn.shape
    av = np.empty(N); aw = np.empty(N); ab = np.empty(N)
    for b in range(0, N, chunk):
        sl = slice(b, min(b + chunk, N))
        idx = nn[sl]                       # (c,k)
        Un = U[idx]                        # (c,k,d) float64
        Ln = labels[idx]                   # (c,k)
        s = decompose_fast(Un, Ln)
        pa = s["per_anchor"]
        av[sl] = pa["A_v"]; aw[sl] = pa["A_within"]; ab[sl] = pa["A_between"]
    return {"A_v": float(av.mean()), "A_within": float(aw.mean()),
            "A_between": float(ab.mean())}


def abetween_debiased(U, nn, labels, n_null=6, n_classes=None, chunk=500, seed=0):
    """De-biased A_between: A_between(labels) - mean over n_null balanced random
    partitions of the same cardinality. Mirrors the E5 cardinality null."""
    rng = np.random.default_rng(seed)
    real = abetween_chunked(U, nn, labels, chunk)["A_between"]
    N = len(labels)
    K = int(n_classes if n_classes is not None else len(np.unique(labels)))
    # Balanced K-way random partition (equal-size groups), matching the E5
    # cardinality null, rather than an unbalanced integer draw.
    base = np.tile(np.arange(K), N // K + 1)[:N]
    nulls = []
    for _ in range(n_null):
        rand_lab = rng.permutation(base)
        nulls.append(abetween_chunked(U, nn, rand_lab, chunk)["A_between"])
    null_mean = float(np.mean(nulls))
    return {"A_between_raw": real, "A_between_null": null_mean,
            "A_between_deb": real - null_mean}


def av_subspace_conditional(Y, U_sub, C, k):
    """Conditional A_v on a velocity subspace, biased-1/k, kNN within each C
    group, then C-weighted (the paper's A_v^S(t|C) on the 'sem' subspace).

    Y     (N,Dfull)  conditioning state (full state, not the subspace)
    U_sub (N,dsub)   velocity restricted to the subspace
    C     (N,)       external condition
    Returns (A_v_raw, A_v_norm) where norm divides by E||U_sub||^2.
    """
    U_sub = np.ascontiguousarray(U_sub, dtype=np.float64)
    if U_sub.ndim == 1:
        U_sub = U_sub[:, None]
    E_U_sq = float((U_sub ** 2).sum(-1).mean())
    raw_acc, n_acc = 0.0, 0
    for c in np.unique(C):
        m = np.where(C == c)[0]
        if len(m) < k + 1:
            continue
        nn = knn_indices(Y[m], k)          # indices into the group
        Un = U_sub[m][nn]                  # (nc,k,dsub)
        mu = Un.mean(axis=1, keepdims=True)
        av_per = ((Un - mu) ** 2).sum(axis=(1, 2)) / k   # biased 1/k
        raw_acc += float(av_per.sum()); n_acc += len(av_per)
    A_v_raw = raw_acc / max(n_acc, 1)
    return A_v_raw, A_v_raw / max(E_U_sq, 1e-12)


def cosine_ratio(a, b):
    """cosine and mean ratio b/a (componentwise) of two vectors over a mask."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
    with np.errstate(divide="ignore", invalid="ignore"):
        r = b / a
    ratio = float(np.nanmean(r[np.isfinite(r)]))
    return cos, ratio


def save_json(name, obj):
    import json
    p = RES / name
    with open(p, "w") as f:
        json.dump(obj, f, indent=2)
    return p
