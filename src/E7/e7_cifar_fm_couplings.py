"""E7: minibatch coupling rules for the CIFAR-10 Flow-Matching sweep.

Each rule takes a minibatch of noise sources Z, real targets X with class
labels KX, and a per-source desired class C (the external condition, sampled
to match the target class histogram so within-class matching is feasible). It
returns a permutation `col` such that source i is paired with target X[col[i]],
together with the class label used to condition the model on that pair.

These mirror the E3 couplings (independent / Euclidean OT / semantic-cost OT /
class-conditional blocked OT) but operate per training minibatch on flattened
pixels, which is the standard minibatch-OT-CFM setting.

All assignments use the exact Hungarian solver (scipy linear_sum_assignment) on
the B x B (or per-class block) squared-Euclidean cost; the training batch B = 128
keeps this well under a millisecond per batch.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def _pairwise_sq(Zf: np.ndarray, Xf: np.ndarray) -> np.ndarray:
    """(B, B) squared Euclidean cost between flattened sources and targets."""
    z2 = (Zf * Zf).sum(1)[:, None]
    x2 = (Xf * Xf).sum(1)[None, :]
    return (z2 + x2 - 2.0 * Zf @ Xf.T).astype(np.float64)


def couple_independent(Zf, Xf, KX, C):
    """C0: identity pairing (Z and X are already i.i.d. samples). Condition on
    the paired target's true class."""
    B = Zf.shape[0]
    col = np.arange(B)
    return col, KX[col]


def couple_euclidean_ot(Zf, Xf, KX, C):
    """C1: minibatch Euclidean OT (condition-blind). Condition on the paired
    target's class, which the geometry-only cost does not control."""
    cost = _pairwise_sq(Zf, Xf)
    _, col = linear_sum_assignment(cost)
    return col, KX[col]


def couple_semantic_cost_ot(Zf, Xf, KX, C, lambda_sem):
    """C3@lambda: Euclidean cost plus a class-mismatch penalty between the
    source's desired class C and the target class KX. Condition on C."""
    cost = _pairwise_sq(Zf, Xf)
    cost = cost + lambda_sem * (C[:, None] != KX[None, :]).astype(np.float64)
    _, col = linear_sum_assignment(cost)
    return col, C


def couple_class_conditional_ot(Zf, Xf, KX, C):
    """C3-inf: blocked OT solved independently within each class. Source i
    (desired class C[i]) is matched only to targets of the same class. Requires
    the C histogram to match the KX histogram (the driver guarantees this by
    setting C to a permutation of KX). Condition on C."""
    B = Zf.shape[0]
    col = np.full(B, -1, dtype=np.int64)
    classes = np.unique(KX)
    for k in classes:
        src = np.where(C == k)[0]
        tgt = np.where(KX == k)[0]
        # equal counts by construction; guard if a class is absent in one side
        m = min(len(src), len(tgt))
        if m == 0:
            continue
        sub = _pairwise_sq(Zf[src[:m]], Xf[tgt[:m]])
        r, c = linear_sum_assignment(sub)
        col[src[r]] = tgt[c]
    # any unmatched (ragged histogram) fall back to identity
    unmatched = np.where(col < 0)[0]
    if len(unmatched):
        free = np.setdiff1d(np.arange(B), col[col >= 0])
        col[unmatched] = free[: len(unmatched)]
    return col, C


def build_coupling(name, lambda_sem=10.0):
    """Return a coupling fn (Zf, Xf, KX, C) -> (col, cond_label) by name."""
    name = name.lower()
    if name in ("c0", "independent"):
        return couple_independent
    if name in ("c1", "euclidean_ot"):
        return couple_euclidean_ot
    if name in ("c3", "semantic_cost"):
        return lambda Zf, Xf, KX, C: couple_semantic_cost_ot(Zf, Xf, KX, C, lambda_sem)
    if name in ("c3inf", "class_conditional"):
        return couple_class_conditional_ot
    raise ValueError(f"unknown coupling {name!r}")


def _self_test():
    rng = np.random.default_rng(0)
    B, d, K = 64, 3 * 32 * 32, 10
    Xf = rng.standard_normal((B, d)).astype(np.float32)
    Zf = rng.standard_normal((B, d)).astype(np.float32)
    KX = rng.integers(0, K, B)
    C = KX[rng.permutation(B)]  # same histogram as KX
    for name in ("c0", "c1", "c3", "c3inf"):
        fn = build_coupling(name, lambda_sem=10.0)
        col, cond = fn(Zf, Xf, KX, C)
        assert col.shape == (B,) and cond.shape == (B,)
        assert len(np.unique(col)) == B, f"{name}: not a permutation"
        if name == "c3inf":
            assert np.all(KX[col] == cond), "c3inf must pair same-class"
        print(f"  {name}: permutation OK, mean same-class pair = "
              f"{(KX[col] == cond).mean():.3f}")
    print("e7_cifar_fm_couplings self-test passed.")


if __name__ == "__main__":
    _self_test()
