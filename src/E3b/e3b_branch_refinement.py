"""E3b unified experiment: main coupling-comparison + branch-refinement
sanity check.

Single run that produces BOTH:
  (a) The main E3b coupling-comparison figure (3-seed N=20k with seed-SD
      bands; paper section 6.4).
  (b) The appendix "Branch refinement sanity check" data (fine vs
      coarse label decomposition on the same paired paths and kNN
      neighborhoods).

This is possible because both figures use the same data path
(`sample_data_4mode`), same paired (Z, X) sets (under each coupling), and
same global kNN graph; the only additional cost for (b) over (a) is a
second `decompose` call with the fine label. The estimator design contract
is a single global kNN graph, biased 1/k, with a numerical-precision
identity check.

Modes of operation:

    python e3b_branch_refinement.py smoke
        Step 0 smoke test: N=5_000, 1 seed (0), couplings = {C0, C2},
        single t = 0.5. Verifies estimator self-consistency before the
        full run. Wall time: ~1 minute on Mac CPU.

    python e3b_branch_refinement.py full
        Full unified run: N=20_000, seeds = {0, 1, 2}, 19 t-values in
        [0.05, 0.95], 10 couplings covering the existing E3b lambda sweep
        plus the lambda=30 and C3_infinity endpoints needed for the
        refinement appendix figure.
        Writes results/e3b_unified_v7.json.
        Estimated wall time: ~1-1.5 hours on Mac CPU.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# CPU thread caps (must be set before numpy import; mirrors
# e3a_coupling_comparison.py's discipline).
for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(var, "4")

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.branch_decomp import decompose, delta_local_direct

# ---------------------------------------------------------------------------
# Paper-canonical defaults (hard-coded, no env-var overrides).
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE.parents[1] / "results"

# 4-mode XOR target geometry (E3b convention; paper §6.4).
M_SEP = 2.0          # mode separation distance from origin along each axis
S_WIDTH = 0.25       # Gaussian width of each mode

# Coupling / estimator hyperparameters.
N_FULL = 20_000      # source-target pairs per (coupling, seed); matches E3
N_SMOKE = 5_000
K_KNN = 80           # kNN bandwidth, paper §6 headline
BATCH_SIZE_OT = 1024  # Hungarian minibatch size, paper §6.4
T_GRID_FULL = np.linspace(0.05, 0.95, 19).tolist()
T_GRID_SMOKE = [0.5]
SEEDS_FULL = [0, 1, 2]
SEEDS_SMOKE = [0]
KNN_WORKERS = 4

# Couplings to run in full mode. Each entry is (label, coupling_fn).
# C3 lambda sweep extends the existing E3b list (0.5, 1, 2, 5, 10) with
# lambda=30 to bracket the refinement appendix's C2 vs C3@30 comparison;
# C3_inf is the hard-blocked (lambda -> inf) endpoint.
LAMBDAS_C3 = (0.5, 1.0, 2.0, 5.0, 10.0, 30.0)

# Numerical tolerances (float64).
ATOL_REL_NUMERICAL = 1e-8   # |residual| < ATOL * max(1, |A_v|)

# ---------------------------------------------------------------------------
# Data sampling (4-mode XOR target; tracks mode_idx as the fine label).
# ---------------------------------------------------------------------------

# Mode centers and their XOR coarse-class assignment.
# modes[i] is the centre of mode i; K_X_per_mode[i] is the XOR coarse class.
MODES = np.array([[ M_SEP,  M_SEP],
                  [-M_SEP, -M_SEP],
                  [-M_SEP,  M_SEP],
                  [ M_SEP, -M_SEP]], dtype=np.float64)
# Main diagonal (modes 0 and 1) -> +1; anti-diagonal (modes 2 and 3) -> -1.
K_X_PER_MODE = np.array([1, 1, -1, -1], dtype=np.int64)


def sample_data_4mode(N: int, seed: int):
    """Sample (Z, X, K_Z, K_X, mode_idx) for the E3b 4-mode XOR target.

    Mirrors the TARGET_TYPE='4mode_diag' branch of
    `e3a_coupling_comparison.py::sample_data`, but also returns the
    fine mode index. K_Z is computed from sign(Z_0) * sign(Z_1); K_X is
    the XOR coarse class.
    """
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((N, 2)).astype(np.float64)
    sign_prod = np.sign(Z[:, 0]) * np.sign(Z[:, 1])
    K_Z = np.where(sign_prod >= 0, 1, -1).astype(np.int64)

    q = N // 4
    rem = N - 4 * q
    mode_idx = np.concatenate([
        np.full(q,        0, dtype=np.int64),
        np.full(q,        1, dtype=np.int64),
        np.full(q,        2, dtype=np.int64),
        np.full(q + rem,  3, dtype=np.int64),
    ])
    rng.shuffle(mode_idx)
    X = (MODES[mode_idx]
         + S_WIDTH * rng.standard_normal((N, 2)).astype(np.float64))
    K_X = K_X_PER_MODE[mode_idx]
    return Z, X, K_Z, K_X, mode_idx


# ---------------------------------------------------------------------------
# Couplings (all propagate mode_idx alongside X / K_X under any permutation).
# ---------------------------------------------------------------------------
# Common signature: (Z, X, K_Z, K_X, mode_idx, rng, **kwargs)
#                    -> (Z_out, X_out, K_Z_out, K_X_out, mode_idx_out)
# Z is never re-indexed; only X-side variables move.

def couple_c0_independent(Z, X, K_Z, K_X, mode_idx, rng):
    return Z, X, K_Z, K_X, mode_idx


def _minibatch_hungarian(Z, X, K_Z, K_X, mode_idx, rng,
                         batch_size, lambda_sem):
    """Apply minibatch Hungarian with optional XOR-class mismatch penalty.

    Permutes X, K_X, mode_idx in lockstep so the diagnostic label channels
    follow the assignment. Z, K_Z untouched.
    """
    n = len(Z)
    perm = rng.permutation(n)
    X_p = X.copy()
    K_X_p = K_X.copy()
    mode_idx_p = mode_idx.copy()
    for b in range(0, n, batch_size):
        idx = perm[b: b + batch_size]
        Z_b = Z[idx]
        X_b = X[idx]
        K_Z_b = K_Z[idx]
        K_X_b = K_X[idx]
        mode_idx_b = mode_idx[idx]
        cost = ((Z_b[:, None, :] - X_b[None, :, :]) ** 2).sum(-1)
        if lambda_sem > 0.0:
            mis = (K_Z_b[:, None] != K_X_b[None, :]).astype(np.float64)
            cost = cost + lambda_sem * mis
        row, col = linear_sum_assignment(cost)
        X_p[idx[row]]        = X_b[col]
        K_X_p[idx[row]]      = K_X_b[col]
        mode_idx_p[idx[row]] = mode_idx_b[col]
    return Z, X_p, K_Z, K_X_p, mode_idx_p


def couple_c1_hungarian(Z, X, K_Z, K_X, mode_idx, rng):
    return _minibatch_hungarian(Z, X, K_Z, K_X, mode_idx, rng,
                                BATCH_SIZE_OT, lambda_sem=0.0)


def couple_c3_semantic(Z, X, K_Z, K_X, mode_idx, rng, lambda_sem):
    return _minibatch_hungarian(Z, X, K_Z, K_X, mode_idx, rng,
                                BATCH_SIZE_OT, lambda_sem=lambda_sem)


def couple_c2_coarse_random(Z, X, K_Z, K_X, mode_idx, rng):
    """Random pairing within each coarse XOR class.

    Trims to the balanced per-class count: n_match = min(n_+, n_-) in each
    class. After this, K_Z and K_X are equal pairwise.
    """
    parts_Z, parts_X, parts_KZ, parts_KX, parts_mi = [], [], [], [], []
    for k in (-1, 1):
        z_idx = np.where(K_Z == k)[0]
        x_idx = np.where(K_X == k)[0]
        n_match = min(len(z_idx), len(x_idx))
        z_sel = rng.permutation(z_idx)[:n_match]
        x_sel = rng.permutation(x_idx)[:n_match]
        parts_Z.append(Z[z_sel])
        parts_X.append(X[x_sel])
        parts_KZ.append(K_Z[z_sel])
        parts_KX.append(K_X[x_sel])
        parts_mi.append(mode_idx[x_sel])
    return (np.concatenate(parts_Z),
            np.concatenate(parts_X),
            np.concatenate(parts_KZ),
            np.concatenate(parts_KX),
            np.concatenate(parts_mi))


def couple_c3_infinity(Z, X, K_Z, K_X, mode_idx, rng):
    """Hard-blocked OT (lambda -> infinity limit of C3): minibatch Hungarian
    is solved separately within each coarse class. Cross-class pairs have
    infinite cost and are therefore excluded.

    Implementation: split data by K_Z / K_X classes, run independent
    Hungarian on each block, then concatenate. Mirrors C3@lambda=inf
    semantics from paper Table 1.
    """
    parts_Z, parts_X, parts_KZ, parts_KX, parts_mi = [], [], [], [], []
    for k in (-1, 1):
        z_idx = np.where(K_Z == k)[0]
        x_idx = np.where(K_X == k)[0]
        n_match = min(len(z_idx), len(x_idx))
        z_sel = rng.permutation(z_idx)[:n_match]
        x_sel = rng.permutation(x_idx)[:n_match]
        Z_b = Z[z_sel]
        X_b = X[x_sel]
        K_Z_b = K_Z[z_sel]
        K_X_b = K_X[x_sel]
        mi_b = mode_idx[x_sel]
        # within-class minibatch Hungarian
        n_b = len(Z_b)
        perm_b = rng.permutation(n_b)
        X_p = X_b.copy()
        K_X_p = K_X_b.copy()
        mi_p = mi_b.copy()
        for bb in range(0, n_b, BATCH_SIZE_OT):
            idx_b = perm_b[bb: bb + BATCH_SIZE_OT]
            Zi = Z_b[idx_b]
            Xi = X_b[idx_b]
            cost = ((Zi[:, None, :] - Xi[None, :, :]) ** 2).sum(-1)
            row, col = linear_sum_assignment(cost)
            X_p[idx_b[row]]   = Xi[col]
            K_X_p[idx_b[row]] = K_X_b[idx_b][col]
            mi_p[idx_b[row]]  = mi_b[idx_b][col]
        parts_Z.append(Z_b)
        parts_X.append(X_p)
        parts_KZ.append(K_Z_b)
        parts_KX.append(K_X_p)
        parts_mi.append(mi_p)
    return (np.concatenate(parts_Z),
            np.concatenate(parts_X),
            np.concatenate(parts_KZ),
            np.concatenate(parts_KX),
            np.concatenate(parts_mi))


def _make_couplings_full():
    """Assemble the 10-coupling list: C0, C1, C2-coarse, the 6 C3 lambda
    values in LAMBDAS_C3, and C3-inf-coarse."""
    couplings = [
        ("C0_independent",   lambda Z, X, KZ, KX, MI, rng: couple_c0_independent(Z, X, KZ, KX, MI, rng)),
        ("C1_hungarian",     lambda Z, X, KZ, KX, MI, rng: couple_c1_hungarian(Z, X, KZ, KX, MI, rng)),
        ("C2_coarse_random", lambda Z, X, KZ, KX, MI, rng: couple_c2_coarse_random(Z, X, KZ, KX, MI, rng)),
    ]
    for lam in LAMBDAS_C3:
        # capture lam by default-arg to avoid late-binding inside lambda
        couplings.append((
            f"C3_lam{lam:g}_coarse".replace(".", "_"),
            (lambda Z, X, KZ, KX, MI, rng, _lam=lam:
                couple_c3_semantic(Z, X, KZ, KX, MI, rng, lambda_sem=_lam)),
        ))
    couplings.append(
        ("C3_inf_coarse", lambda Z, X, KZ, KX, MI, rng: couple_c3_infinity(Z, X, KZ, KX, MI, rng))
    )
    return couplings


COUPLINGS_FULL = _make_couplings_full()

COUPLINGS_SMOKE = [
    ("C0_independent",   lambda Z, X, KZ, KX, MI, rng: couple_c0_independent(Z, X, KZ, KX, MI, rng)),
    ("C2_coarse_random", lambda Z, X, KZ, KX, MI, rng: couple_c2_coarse_random(Z, X, KZ, KX, MI, rng)),
]


# ---------------------------------------------------------------------------
# Pipeline: for a single (coupling, seed, t), compute decomposition metrics.
# ---------------------------------------------------------------------------

def run_one_t(Z, X, K_X, mode_idx, t: float, k: int):
    """Evaluate the full diagnostic at a single t.

    Builds Y_t = (1-t) Z + t X, U = X - Z, queries the single global kNN
    over Y_t, then runs `decompose` twice (coarse vs fine) on the same
    neighbor sets and the `delta_local_direct` cross-check.

    Returns a flat dict of scalars suitable for JSON serialization.
    """
    Y = (1.0 - t) * Z + t * X
    U = X - Z
    tree = cKDTree(Y)
    # nearest neighbors include the point itself at index 0; we keep it
    # to be consistent with paper §6's existing kNN convention.
    _, knn_idx = tree.query(Y, k=k, workers=KNN_WORKERS)

    U_n = U[knn_idx]                # (N, k, d)
    K_n = K_X[knn_idx]              # (N, k)
    Mi_n = mode_idx[knn_idx]        # (N, k)

    s_coarse = decompose(U_n, K_n)
    s_fine   = decompose(U_n, Mi_n)

    delta_local = delta_local_direct(U_n, K_n, Mi_n)
    delta_diff = (s_fine["aggregated"]["A_between"]
                  - s_coarse["aggregated"]["A_between"])

    Av_c = s_coarse["aggregated"]["A_v"]
    Av_f = s_fine["aggregated"]["A_v"]
    A_v_norm = max(1.0, abs(Av_c))

    return {
        "A_v":                Av_c,
        "A_within_K":         s_coarse["aggregated"]["A_within"],
        "A_between_K":        s_coarse["aggregated"]["A_between"],
        "A_within_Kprime":    s_fine["aggregated"]["A_within"],
        "A_between_Kprime":   s_fine["aggregated"]["A_between"],
        "Delta_diff":         float(delta_diff),
        "Delta_local":        delta_local["aggregated"],
        "Delta_resid":        float(delta_diff - delta_local["aggregated"]),
        "identity_resid_K":      s_coarse["aggregated"]["identity_residual_max_abs"],
        "identity_resid_Kprime": s_fine["aggregated"]["identity_residual_max_abs"],
        "Av_invariance":      float(Av_c - Av_f),
        "A_v_norm":           A_v_norm,
    }


# ---------------------------------------------------------------------------
# Drivers.
# ---------------------------------------------------------------------------

def _check_smoke(metrics: dict, label: str):
    """Hard checks for Step 0 smoke. Raise on violation."""
    fails = []
    tol = ATOL_REL_NUMERICAL * metrics["A_v_norm"]
    if abs(metrics["identity_resid_K"]) > tol:
        fails.append(f"identity_resid_K = {metrics['identity_resid_K']:.3e} > {tol:.3e}")
    if abs(metrics["identity_resid_Kprime"]) > tol:
        fails.append(f"identity_resid_Kprime = {metrics['identity_resid_Kprime']:.3e} > {tol:.3e}")
    if abs(metrics["Av_invariance"]) > 1e-10:
        fails.append(f"Av_invariance = {metrics['Av_invariance']:.3e} > 1e-10")
    if abs(metrics["Delta_resid"]) > tol:
        fails.append(f"Delta_resid = {metrics['Delta_resid']:.3e} > {tol:.3e}")
    if metrics["Delta_diff"] < -tol:
        fails.append(f"Delta_diff = {metrics['Delta_diff']:.3e} < -{tol:.3e}")
    if fails:
        raise RuntimeError(f"[{label}] SMOKE CHECK FAILED:\n  " + "\n  ".join(fails))


def run_smoke():
    print("=== E3b branch-refinement SMOKE ===")
    print(f"  N={N_SMOKE}  seed=0  k={K_KNN}  t={T_GRID_SMOKE}")
    Z, X, K_Z, K_X, mode_idx = sample_data_4mode(N_SMOKE, seed=0)
    rng = np.random.default_rng(0)

    for label, coupling_fn in COUPLINGS_SMOKE:
        t0 = time.time()
        Zc, Xc, KZc, KXc, MIc = coupling_fn(Z, X, K_Z, K_X, mode_idx, rng)
        for t in T_GRID_SMOKE:
            metrics = run_one_t(Zc, Xc, KXc, MIc, t, K_KNN)
            _check_smoke(metrics, f"{label} t={t}")
            print(f"  [{label} t={t}]  A_v={metrics['A_v']:.4f}  "
                  f"A_between_K={metrics['A_between_K']:.4f}  "
                  f"A_between_K'={metrics['A_between_Kprime']:.4f}  "
                  f"Delta={metrics['Delta_diff']:.4f}  "
                  f"id_resid={metrics['identity_resid_K']:.2e}  "
                  f"Av_inv={metrics['Av_invariance']:.2e}  "
                  f"Delta_resid={metrics['Delta_resid']:.2e}")
        print(f"  [{label}] done in {time.time() - t0:.1f}s")

    print("SMOKE PASSED. Estimator self-consistency confirmed.")


def run_full():
    print("=== E3b branch-refinement FULL ===")
    print(f"  N={N_FULL}  seeds={SEEDS_FULL}  k={K_KNN}  |t-grid|={len(T_GRID_FULL)}")
    print(f"  couplings={[c[0] for c in COUPLINGS_FULL]}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "e3b_unified_v7.json"

    config = {
        "target_type": "4mode_diag",
        "N": N_FULL,
        "seeds": SEEDS_FULL,
        "k": K_KNN,
        "t_grid": T_GRID_FULL,
        "couplings": [c[0] for c in COUPLINGS_FULL],
        "M_SEP": M_SEP,
        "S_WIDTH": S_WIDTH,
        "BATCH_SIZE_OT": BATCH_SIZE_OT,
        "labels": {
            "coarse": "K_X XOR superclass {-1, +1}",
            "fine":   "mode_idx {0, 1, 2, 3}",
            "mapping": {"0": 1, "1": 1, "2": -1, "3": -1},
        },
        "estimator": "single global kNN graph; biased 1/k local covariance",
        "lambdas_C3": list(LAMBDAS_C3),
        "knn_workers": KNN_WORKERS,
    }
    runs = []
    t_start = time.time()

    for seed in SEEDS_FULL:
        Z, X, K_Z, K_X, mode_idx = sample_data_4mode(N_FULL, seed=seed)
        for ci, (label, coupling_fn) in enumerate(COUPLINGS_FULL):
            t0 = time.time()
            # Deterministic per-(seed, coupling) RNG stream. The coupling index
            # ci replaces hash(label): Python salts str hashes per process unless
            # PYTHONHASHSEED is pinned, which would make the random-pairing
            # couplings (C0, C2) non-reproducible across runs. Mirrors the E3
            # driver convention (seed * <stride> + coupling index).
            rng = np.random.default_rng(int(1e6) + seed * 100 + ci)
            Zc, Xc, KZc, KXc, MIc = coupling_fn(Z, X, K_Z, K_X, mode_idx, rng)
            per_t = {key: [] for key in (
                "A_v", "A_within_K", "A_between_K",
                "A_within_Kprime", "A_between_Kprime",
                "Delta_diff", "Delta_local", "Delta_resid",
                "identity_resid_K", "identity_resid_Kprime",
                "Av_invariance",
            )}
            for t in T_GRID_FULL:
                m = run_one_t(Zc, Xc, KXc, MIc, t, K_KNN)
                for key in per_t:
                    per_t[key].append(m[key])
            elapsed = time.time() - t0
            print(f"  seed={seed} {label:24s}  done in {elapsed:6.1f}s  "
                  f"max|Delta_resid|={max(abs(v) for v in per_t['Delta_resid']):.2e}  "
                  f"min(Delta)={min(per_t['Delta_diff']):.3e}  "
                  f"max(A_between_K')={max(per_t['A_between_Kprime']):.3f}")
            runs.append({"seed": seed, "coupling": label, **per_t})

    payload = {"config": config, "runs": runs}
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {out_path}")
    print(f"Total wall time: {(time.time() - t_start) / 60:.1f} min")


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ("smoke", "full"):
        print("Usage: python e3b_branch_refinement.py {smoke|full}")
        sys.exit(2)
    if sys.argv[1] == "smoke":
        run_smoke()
    else:
        run_full()


if __name__ == "__main__":
    main()
