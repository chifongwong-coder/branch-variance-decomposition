"""E7 pairing-sensitivity + positive control for the
feature-space BVD diagnostic.

Locks in the finding as a guarded,
reproducible test: the feature-space BVD used by e7_cifar_fm_diagnostics.bvd_curves
is near-invariant to the training coupling for c0/c1/c3inf, because for those
three the decomposition label cond == the paired target's true class
(e7_cifar_fm_couplings.py:35,43, and c3inf's same-class blocking), so any
same-class-preserving pairing yields the same labeled feature multiset and the
row-permutation-invariant decompose() returns the same curve. Only c3 (label C
decoupled from the paired class) can separate; it is the positive control that
proves the instrument is not simply dead.

This script reuses the EXACT bvd_curves computation path: pixel-space coupling on
real CIFAR images, CLIP-feature decomposition with the run's K_NN / T_GRID. It
computes the cross-coupling gap (signal) against the within-coupling
seed-resampling gap (noise floor), and asserts:
  - c0/c1/c3inf cross gap  <  seed-noise floor   (coupling-blind, as found)
  - c3 vs c0 gap           >  seed-noise floor    (positive control passes)

Run:  python3 e7_bvd_sensitivity_check.py
Exit code 0 on pass, 1 on fail (suitable for a smoke test / CI guard).
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from core.branch_decomp import decompose                       # noqa: E402
from E7.e7_cifar_fm_couplings import build_coupling           # noqa: E402
from E7.e7_cifar_fm_diagnostics import T_GRID, K_NN           # authoritative constants
from E7.e7_cifar_fm_train import load_cifar                   # noqa: E402

RES = HERE.parents[1] / "results"
N = 2500                       # subsample; kept modest so Hungarian stays fast
DPIX = 3 * 32 * 32


def _load_aligned():
    """CLIP features (cached) aligned with their CIFAR pixel images and labels."""
    npz = np.load(RES / "cifar10_clip_features.npz")
    feat_all = npz["features_l2"].astype(np.float32)
    lab_all = npz["labels"].astype(np.int64)
    X, Y = load_cifar(train=True)                          # pixels in [-1,1], in order
    Y = Y.numpy()
    assert feat_all.shape[0] == Y.shape[0] == lab_all.shape[0], "row count mismatch"
    assert (Y == lab_all).all(), "cached features are not aligned with CIFAR label order"
    return feat_all, lab_all, X.reshape(X.shape[0], -1).numpy()


def a_between_curve(coupling_name, feat_all, lab_all, pix_all, seed, lam=10.0):
    """Mirror e7_cifar_fm_diagnostics.bvd_curves and return A_between(t)."""
    rng = np.random.default_rng(seed)
    sub = rng.permutation(feat_all.shape[0])[:N]
    feat = feat_all[sub]
    kx = lab_all[sub]
    Xf = pix_all[sub]                                      # real pixels for the coupling cost
    dim = feat.shape[1]
    coupling = build_coupling(coupling_name, lambda_sem=lam)
    zp = rng.standard_normal((N, DPIX)).astype(np.float32)
    c = kx[rng.permutation(N)]
    col, cond = coupling(zp, Xf, kx, c)
    fp = feat[col]
    Zf = (rng.standard_normal(fp.shape) / np.sqrt(dim)).astype(np.float32)
    U = (fp - Zf).astype(np.float64)
    out = []
    for t in T_GRID:
        Yt = ((1 - t) * Zf + t * fp).astype(np.float32)
        nn = NearestNeighbors(n_neighbors=K_NN, algorithm="brute").fit(Yt).kneighbors(Yt)[1]
        out.append(decompose(U[nn], cond[nn])["aggregated"]["A_between"])
    return np.array(out)


def _curve_from_pairing(feat, col, cond, rng):
    """A_between(t) for an arbitrary (col, cond) on a fixed feature subsample.
    Mirrors the bvd_curves inner loop; lets us drive controls (identity, random
    permutation, label-decoupled) through the exact estimator path."""
    fp = feat[col]
    dim = feat.shape[1]
    Zf = (rng.standard_normal(fp.shape) / np.sqrt(dim)).astype(np.float32)
    U = (fp - Zf).astype(np.float64)
    out = []
    for t in T_GRID:
        Yt = ((1 - t) * Zf + t * fp).astype(np.float32)
        nn = NearestNeighbors(n_neighbors=K_NN, algorithm="brute").fit(Yt).kneighbors(Yt)[1]
        out.append(decompose(U[nn], cond[nn])["aggregated"]["A_between"])
    return np.array(out)


def controls(feat_all, lab_all, floor):
    """0.6: identity / random-permutation / label-decoupled controls on one fixed
    subsample. Confirms the estimator is blind iff the label tracks the paired
    feature, and responds when the label is decoupled."""
    rng = np.random.default_rng(2024)
    sub = rng.permutation(feat_all.shape[0])[:N]
    feat = feat_all[sub]
    kx = lab_all[sub].astype(np.int64)

    ident_col = np.arange(N)
    rand_col = rng.permutation(N)                              # label FOLLOWS the paired feature
    decoupled_cond = kx[rng.permutation(N)]                    # label INDEPENDENT of the feature

    base = _curve_from_pairing(feat, ident_col, kx[ident_col], np.random.default_rng(7))
    rand_follow = _curve_from_pairing(feat, rand_col, kx[rand_col], np.random.default_rng(7))
    sorted_col = np.argsort(kx, kind="stable")                 # group by class; label still follows
    sorted_follow = _curve_from_pairing(feat, sorted_col, kx[sorted_col], np.random.default_rng(7))
    decoupled = _curve_from_pairing(feat, ident_col, decoupled_cond, np.random.default_rng(7))

    rows = [
        ("identity (label follows)", np.abs(base - base).max(), "blind"),
        ("random perm (label follows)", np.abs(rand_follow - base).max(), "blind"),
        ("sorted-by-class (label follows)", np.abs(sorted_follow - base).max(), "blind"),
        ("label-decoupled (control)", np.abs(decoupled - base).max(), "SEPARATES"),
    ]
    print("\n0.6 controls (gap vs identity baseline; floor "
          f"= {floor:.5f}):")
    ok = True
    for name, gap, expect in rows:
        sep = gap > 1.5 * floor
        got = "SEPARATES" if sep else "blind"
        good = (got == expect)
        ok = ok and good
        print(f"  {name:34} gap={gap:.5f}  ratio={gap / floor:5.2f}  "
              f"expect={expect:9} got={got:9} {'OK' if good else 'FAIL'}")
    return ok


def identity_gate(feat_all, lab_all):
    """0.7: estimator sanity. The A_v = A_within + A_between identity residual must
    be ~0, and A_v must be invariant to a relabeling that only changes the
    partition (labels move between/within, not the total)."""
    rng = np.random.default_rng(55)
    sub = rng.permutation(feat_all.shape[0])[:N]
    feat = feat_all[sub]
    kx = lab_all[sub].astype(np.int64)
    col = np.arange(N)
    fp = feat[col]
    dim = feat.shape[1]
    Zf = (rng.standard_normal(fp.shape) / np.sqrt(dim)).astype(np.float32)
    U = (fp - Zf).astype(np.float64)
    t = 0.5
    Yt = ((1 - t) * Zf + t * fp).astype(np.float32)
    nn = NearestNeighbors(n_neighbors=K_NN, algorithm="brute").fit(Yt).kneighbors(Yt)[1]
    s_true = decompose(U[nn], kx[col][nn])
    relabel = kx[rng.permutation(N)]                          # partition changes, U unchanged
    s_relab = decompose(U[nn], relabel[nn])
    resid = s_true["aggregated"]["identity_residual_max_abs"]
    av_true = s_true["aggregated"]["A_v"]
    av_relab = s_relab["aggregated"]["A_v"]
    av_invar = abs(av_true - av_relab)
    print("\n0.7 estimator identity gate (t=0.5):")
    print(f"  identity residual max|A_v - A_within - A_between| = {resid:.2e}")
    print(f"  A_v invariance to relabeling |A_v - A_v'|         = {av_invar:.2e}")
    ok = resid < 1e-6 and av_invar < 1e-9
    print(f"  [{'OK' if ok else 'FAIL'}] residual < 1e-6 and A_v relabel-invariant < 1e-9")
    return ok


def main():
    feat_all, lab_all, pix_all = _load_aligned()

    # Three couplings whose label tracks the paired class (expected coincident).
    same_seed = 777
    curves = {cp: a_between_curve(cp, feat_all, lab_all, pix_all, same_seed)
              for cp in ("c0", "c1", "c3inf")}
    pairs = [("c0", "c1"), ("c0", "c3inf"), ("c1", "c3inf")]
    cross_gap = max(np.abs(curves[a] - curves[b]).max() for a, b in pairs)

    # Within-c0 seed-to-seed gap = the noise floor of the estimator.
    c0_seeds = [a_between_curve("c0", feat_all, lab_all, pix_all, s)
                for s in (777, 778, 779, 780)]
    seed_floor = max(np.abs(c0_seeds[i] - c0_seeds[j]).max()
                     for i in range(len(c0_seeds)) for j in range(i + 1, len(c0_seeds)))

    # Positive control: c3 decouples the label from the paired class.
    c3 = a_between_curve("c3", feat_all, lab_all, pix_all, same_seed, lam=10.0)
    c3_gap = np.abs(c3 - curves["c0"]).max()

    print(f"N={N}  K_NN={K_NN}  |t-grid|={len(T_GRID)}")
    print(f"cross-coupling max gap (c0/c1/c3inf, same seed) = {cross_gap:.5f}")
    print(f"within-c0 seed-to-seed max gap (noise floor)    = {seed_floor:.5f}")
    print(f"c3 vs c0 max gap (positive control)             = {c3_gap:.5f}")
    print(f"ratios: cross/floor = {cross_gap / seed_floor:.2f}   c3/floor = {c3_gap / seed_floor:.2f}")

    blind = cross_gap < seed_floor
    control = c3_gap > seed_floor
    print(f"\n[{'PASS' if blind else 'FAIL'}] c0/c1/c3inf are coupling-blind "
          f"(cross gap {cross_gap:.5f} {'<' if blind else '>='} floor {seed_floor:.5f})")
    print(f"[{'PASS' if control else 'FAIL'}] positive control: c3 separates "
          f"(c3 gap {c3_gap:.5f} {'>' if control else '<='} floor {seed_floor:.5f})")

    controls_ok = controls(feat_all, lab_all, seed_floor)
    gate_ok = identity_gate(feat_all, lab_all)

    ok = blind and control and controls_ok and gate_ok
    print(f"\n{'OK: ' if ok else 'GUARD TRIPPED: '}"
          + ("feature-space BVD is coupling-blind for label-tracks-class couplings, "
             "the instrument still sees label-decoupled pairings (c3 and the "
             "decoupled control), and the A_v identity holds; the finding is "
             "reproduced and the estimator is sane."
             if ok else
             "the sensitivity structure changed; investigate before trusting the diagnostic."))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
