"""E3a: Coupling comparison for Flow Matching path ambiguity.

CPU-only (NumPy + SciPy + cKDTree + Hungarian).  Thread cap set up-front to
keep the laptop cool and responsive.

Setup
  Source       Z ~ N(0, I_2)
  Source label K_Z = sign(Z_0)          (deterministic from Z; preserves marginal)
  Target       K_X in {-1, +1}, equal prior;  X | K_X=k ~ N(k * m * e_0, s^2 I_2)
  Interpolant  Y_t = (1-t) Z + t X,  U = X - Z

Couplings (after pairing, all preserve the source/target marginals)
  C0 independent
  C1 minibatch Hungarian Euclidean OT
  C2 within-branch random pairing (uses K_Z = sign(Z_0); trim to balance)
  C3 semantic-cost OT: C_ij = ||Z_i - X_j||^2 + lambda * 1[K_Z_i != K_X_j]
     lambda in {0.5, 1, 2, 5, 10}.  lambda=0 reproduces C1 (sanity-only).

Estimator: kNN local covariance, one query at k_max then slice for k_list
  (saves 5x cost vs separate queries).

Closed-form C0 oracle (for sanity only)
  tau^2(t)    = (1-t)^2 + t^2 s^2
  A_within(t) = 2 s^2 / tau^2(t)
  A_between(t)= m^2 (1-t)^2 / tau^4(t) * E[sech^2(t m Y_{t,0}/tau^2(t))]
  A_v(t)      = A_within + A_between
  R_switch(t) = A_between / A_v
  (C1/C2/C3 have no Gaussian closed form: C2 is truncated Gaussian, OT depends
   on the optimisation result.)

Env vars (override defaults)
  E3_TAG          run tag (output suffix)
  E3_S            target component std s (default 0.25)
  E3_N            number of paired samples N (default 100_000)
  E3_SEEDS        comma list, e.g. "0,1,2"
  E3_LAMBDAS      comma list of C3 lambdas (default "0.5,1,2,5,10")
  E3_BATCH_OT     OT batch size (default 1024)
  E3_KNN_WORKERS  cKDTree.query workers (default 4)
  E3_RUN_C1_SANITY  1 = also run C3@lambda=0 (sanity), 0 = skip
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Thread caps: must be set before numpy/scipy are imported.
# ---------------------------------------------------------------------------
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_k, "4")

import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment

try:
    import psutil
    _HAVE_PSUTIL = True
except ImportError:
    _HAVE_PSUTIL = False


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

M = 2.0
S = 0.25
N_PAIRS = 50000
T_GRID = np.linspace(0.05, 0.95, 19)
K_LIST = [30, 50, 80, 120, 200]
K_MAX = max(K_LIST)
SEEDS = [0]
LAMBDAS = [0.5, 1.0, 2.0, 5.0, 10.0]
BATCH_SIZE_OT = 1024
RUN_C1_SANITY = True
SLEEP_BETWEEN_COUPLINGS = 30.0
TARGET_TYPE = "binary"  # "binary" | "4mode_diag"; switch in-source when running E3b
TAG = "default"
EPS = 1e-12

# Threading is machine-dependent, not paper-dependent; keep env-overridable.
KNN_WORKERS = int(os.environ.get("E3_KNN_WORKERS", "4"))


def _monitor_start():
    """Start a CPU-usage measurement window.  Pair with _monitor_end()."""
    if not _HAVE_PSUTIL:
        return None
    p = psutil.Process()
    return {"t": time.time(), "cpu": p.cpu_times(), "p": p}


def _monitor_end(mon):
    """Return (cpu_percent_during_window, rss_mb).
    cpu_percent can exceed 100% with multithreaded workloads (sum of all cores)."""
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
# Closed-form C0 oracle (sanity reference; matches the binary-Gaussian + N(0,I)
# independent coupling, generalised to 2D).
# ---------------------------------------------------------------------------

def tau2(t):
    return (1.0 - t) ** 2 + (t ** 2) * (S ** 2)


def c0_oracle_av(t, n_mc=400_000, seed=10_000):
    """Closed-form C0 oracle.  Only available for TARGET_TYPE='binary'."""
    if TARGET_TYPE != "binary":
        return None
    tt = tau2(t)
    within = 2.0 * (S ** 2) / tt
    rng = np.random.default_rng(seed + int(1000 * t))
    k = rng.choice([-1, 1], size=n_mc)
    z0 = rng.standard_normal(n_mc)
    x0 = k * M + S * rng.standard_normal(n_mc)
    y0 = (1.0 - t) * z0 + t * x0
    sech2 = 1.0 / np.cosh(t * M * y0 / tt) ** 2
    between = ((1.0 - t) ** 2 * (M ** 2) / (tt ** 2)) * sech2.mean()
    a_v = within + between
    return {
        "A_within": float(within),
        "A_between": float(between),
        "A_v": float(a_v),
        "R_switch": float(between / max(a_v, EPS)),
    }


# ---------------------------------------------------------------------------
# Samplers
# ---------------------------------------------------------------------------

def sample_data(N, seed):
    """Returns (Z, X, K_Z, K_X) per E3a setting.

    TARGET_TYPE='binary' (E3a): X is binary Gaussian mixture on x-axis at +-m.
        K_Z = sign(Z_0), K_X in {-1, +1}.

    TARGET_TYPE='4mode_diag' (E3b mismatch toy): 4 modes at (+-m, +-m) with
        diagonal XOR semantic labels.
        K_X = +1 for the main diagonal pair (+m,+m), (-m,-m); -1 for anti-diagonal.
        K_Z = +1 if sign(Z_0)*sign(Z_1) >= 0 (UR or LL quadrant); -1 otherwise.
    """
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((N, 2))
    if TARGET_TYPE == "binary":
        K_Z = np.where(Z[:, 0] >= 0, 1, -1).astype(np.int64)
        K_X = np.concatenate([np.full(N // 2, -1, dtype=np.int64),
                              np.full(N - N // 2, 1, dtype=np.int64)])
        rng.shuffle(K_X)
        e0 = np.array([1.0, 0.0])
        X = K_X[:, None] * M * e0[None, :] + S * rng.standard_normal((N, 2))
    elif TARGET_TYPE == "4mode_diag":
        sign_prod = np.sign(Z[:, 0]) * np.sign(Z[:, 1])
        K_Z = np.where(sign_prod >= 0, 1, -1).astype(np.int64)
        # 4 modes balanced; main-diagonal = K_X = +1, anti-diag = -1
        modes = np.array([[M, M], [-M, -M], [-M, M], [M, -M]])
        K_X_per_mode = np.array([1, 1, -1, -1], dtype=np.int64)
        q = N // 4
        rem = N - 4 * q
        mode_idx = np.concatenate([
            np.full(q, 0, dtype=np.int64),
            np.full(q, 1, dtype=np.int64),
            np.full(q, 2, dtype=np.int64),
            np.full(q + rem, 3, dtype=np.int64),
        ])
        rng.shuffle(mode_idx)
        X = modes[mode_idx] + S * rng.standard_normal((N, 2))
        K_X = K_X_per_mode[mode_idx]
    else:
        raise ValueError(f"Unknown TARGET_TYPE: {TARGET_TYPE!r}")
    return Z, X, K_Z, K_X


# ---------------------------------------------------------------------------
# Couplings (each returns (Z_out, X_out, K_Z_out, K_X_out))
# ---------------------------------------------------------------------------

def couple_independent(Z, X, K_Z, K_X, rng):
    return Z, X, K_Z, K_X


def _hungarian_pair(Z, X, K_Z, K_X, rng, batch_size, lambda_sem):
    """Re-pair Z and X within each minibatch by Hungarian assignment.
    With lambda_sem > 0 the cost has a branch-mismatch penalty."""
    n = len(Z)
    perm = rng.permutation(n)
    X_p = X.copy()
    K_X_p = K_X.copy()
    for b in range(0, n, batch_size):
        idx = perm[b: b + batch_size]
        Z_b = Z[idx]
        X_b = X[idx]
        K_Z_b = K_Z[idx]
        K_X_b = K_X[idx]
        cost = ((Z_b[:, None, :] - X_b[None, :, :]) ** 2).sum(-1)
        if lambda_sem > 0.0:
            mis = (K_Z_b[:, None] != K_X_b[None, :]).astype(np.float64)
            cost = cost + lambda_sem * mis
        row, col = linear_sum_assignment(cost)
        X_p[idx[row]] = X_b[col]
        K_X_p[idx[row]] = K_X_b[col]
    return Z, X_p, K_Z, K_X_p


def couple_hungarian(Z, X, K_Z, K_X, rng):
    return _hungarian_pair(Z, X, K_Z, K_X, rng, BATCH_SIZE_OT, 0.0)


def couple_semantic_cost_ot(Z, X, K_Z, K_X, rng, lambda_sem):
    return _hungarian_pair(Z, X, K_Z, K_X, rng, BATCH_SIZE_OT, lambda_sem)


def couple_branch_aware(Z, X, K_Z, K_X, rng):
    """Within-branch random pairing, trim to balanced count per branch."""
    parts_Z, parts_X, parts_KZ, parts_KX = [], [], [], []
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
    return (np.concatenate(parts_Z), np.concatenate(parts_X),
            np.concatenate(parts_KZ), np.concatenate(parts_KX))


# ---------------------------------------------------------------------------
# kNN local covariance estimator (single query at k_max, slice for each k)
# ---------------------------------------------------------------------------

def compute_metrics(U_n, K_n, U_all, k, E_U_sq):
    """Local cov + within/between decomposition for a single k value.

    Inputs:
      U_n     (n, k, d)
      K_n     (n, k):  entries in {-1, +1}
      U_all   (n, d):  full sample (for E||U||^2)
      k       int:     number of neighbours used
      E_U_sq  float
    Returns: dict of scalar metrics.
    """
    # Total Tr Cov(U | kNN), biased (1/k) so within + between adds up exactly.
    mu_total = U_n.mean(axis=1, keepdims=True)            # (n, 1, d)
    diff = U_n - mu_total
    A_v_per_q = (diff ** 2).sum(axis=(1, 2)) / k          # (n,)
    A_v_raw = float(A_v_per_q.mean())

    mu_sq = (mu_total[:, 0, :] ** 2).sum(axis=-1)          # (n,) = ||mean||^2
    R_v2 = float(mu_sq.mean() / max(E_U_sq, EPS))

    A_v_norm = A_v_raw / max(E_U_sq, EPS)

    # H(K_X | Y_t) ~ E_q H( empirical posterior over kNN labels )
    f_pos = (K_n == 1).mean(axis=1)
    H_per_q = -(f_pos * np.log(np.clip(f_pos, EPS, 1.0))
                + (1.0 - f_pos) * np.log(np.clip(1.0 - f_pos, EPS, 1.0)))
    H_KY = float(H_per_q.mean())

    # within / between decomposition (biased 1/k)
    mask_pos = (K_n == 1).astype(np.float64)               # (n, k)
    mask_neg = 1.0 - mask_pos
    n_pos = mask_pos.sum(axis=1)
    n_neg = mask_neg.sum(axis=1)
    sum_pos = (mask_pos[..., None] * U_n).sum(axis=1)      # (n, d)
    sum_neg = (mask_neg[..., None] * U_n).sum(axis=1)
    overall = (sum_pos + sum_neg) / k
    safe_pos = np.maximum(n_pos, 1)
    safe_neg = np.maximum(n_neg, 1)
    mean_pos = np.where(n_pos[:, None] > 0,
                        sum_pos / safe_pos[:, None], overall)
    mean_neg = np.where(n_neg[:, None] > 0,
                        sum_neg / safe_neg[:, None], overall)

    diff_pos = (U_n - mean_pos[:, None, :]) * mask_pos[..., None]
    diff_neg = (U_n - mean_neg[:, None, :]) * mask_neg[..., None]
    ss_within = ((diff_pos ** 2).sum(axis=(1, 2))
                 + (diff_neg ** 2).sum(axis=(1, 2)))
    A_within_per_q = ss_within / k

    d_pos = mean_pos - overall
    d_neg = mean_neg - overall
    A_between_per_q = (n_pos * (d_pos ** 2).sum(axis=-1)
                       + n_neg * (d_neg ** 2).sum(axis=-1)) / k

    A_within = float(A_within_per_q.mean())
    A_between = float(A_between_per_q.mean())
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


def estimate_curves(Z, X, K_X, t_grid, k_list=K_LIST):
    """Return per-t dict of per-k metrics.  Single cKDTree.query at k_max."""
    k_max = max(k_list)
    U = X - Z
    E_U_sq = float((U ** 2).sum(axis=-1).mean())
    per_t = []
    for t in t_grid:
        Y = (1.0 - t) * Z + t * X
        tree = cKDTree(Y)
        _, idxs = tree.query(Y, k=k_max, workers=KNN_WORKERS)
        U_full = U[idxs]                             # (n, k_max, d)
        K_full = K_X[idxs]                           # (n, k_max)
        per_k = {}
        for k in k_list:
            per_k[k] = compute_metrics(U_full[:, :k, :], K_full[:, :k],
                                       U, k, E_U_sq)
        per_t.append({"t": float(t), "per_k": per_k})
    return per_t, E_U_sq


# ---------------------------------------------------------------------------
# Auxiliary metrics
# ---------------------------------------------------------------------------

def transport_cost(Z, X):
    return float(((X - Z) ** 2).sum(axis=-1).mean())


def branch_coverage_error(K_X):
    return float(abs((K_X == 1).mean() - 0.5))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_couplings():
    couplings = [
        ("C0_independent", couple_independent, {}),
        ("C1_OT", couple_hungarian, {}),
        ("C2_branch_aware", couple_branch_aware, {}),
    ]
    for lam in LAMBDAS:
        name = f"C3_semOT_lam{lam:g}"
        # capture lam by default arg
        couplings.append((name,
                          (lambda Z, X, K_Z, K_X, rng, _lam=lam:
                              couple_semantic_cost_ot(Z, X, K_Z, K_X, rng, _lam)),
                          {"lambda": lam}))
    if RUN_C1_SANITY:
        couplings.append(
            ("C3_semOT_lam0_sanity_eq_C1",
             (lambda Z, X, K_Z, K_X, rng: couple_semantic_cost_ot(
                 Z, X, K_Z, K_X, rng, 0.0)),
             {"lambda": 0.0, "note": "should equal C1_OT"}))
    return couplings


def run():
    out_dir = Path(__file__).resolve().parents[2]
    res_dir = out_dir / "results"
    res_dir.mkdir(exist_ok=True)

    print(f"E3a coupling comparison  [tag={TAG}]")
    print(f"  TARGET_TYPE={TARGET_TYPE}")
    print(f"  M={M}  S={S}  N_PAIRS={N_PAIRS}")
    print(f"  T_GRID={len(T_GRID)} pts in [{T_GRID[0]:.2f}, {T_GRID[-1]:.2f}]")
    print(f"  K_LIST={K_LIST}  (single query at k_max={K_MAX})")
    print(f"  SEEDS={SEEDS}  LAMBDAS={LAMBDAS}")
    print(f"  BATCH_SIZE_OT={BATCH_SIZE_OT}  KNN_WORKERS={KNN_WORKERS}")
    print(f"  SLEEP_BETWEEN_COUPLINGS={SLEEP_BETWEEN_COUPLINGS}s")
    print(f"  Threads (OMP)={os.environ.get('OMP_NUM_THREADS')}")
    print("-" * 70)

    # Closed-form C0 oracle (binary target only)
    if TARGET_TYPE == "binary":
        print("Computing closed-form C0 oracle (binary target)...")
        c0_oracle = {float(t): c0_oracle_av(float(t)) for t in T_GRID}
    else:
        print(f"No closed-form oracle for TARGET_TYPE={TARGET_TYPE}.")
        c0_oracle = {}

    couplings = build_couplings()
    print(f"Couplings to run: {[c[0] for c in couplings]}")
    print("-" * 70)

    results = {
        "config": {
            "TARGET_TYPE": TARGET_TYPE,
            "M": M, "S": S, "N_PAIRS": N_PAIRS,
            "T_GRID": T_GRID.tolist(), "K_LIST": K_LIST,
            "SEEDS": SEEDS, "LAMBDAS": LAMBDAS,
            "BATCH_SIZE_OT": BATCH_SIZE_OT,
            "KNN_WORKERS": KNN_WORKERS,
            "RUN_C1_SANITY": RUN_C1_SANITY,
            "SLEEP_BETWEEN_COUPLINGS": SLEEP_BETWEEN_COUPLINGS,
        },
        "c0_oracle": {str(t): v for t, v in c0_oracle.items()},
        "runs": {},
    }

    t0_global = time.time()
    n_couplings = len(couplings)
    for seed in SEEDS:
        print(f"\n=== seed {seed} ===")
        Z, X, K_Z, K_X = sample_data(N_PAIRS, seed)
        print(f"  K_Z balance: +1 {(K_Z == 1).mean():.3f}, "
              f"-1 {(K_Z == -1).mean():.3f}")
        print(f"  K_X balance: +1 {(K_X == 1).mean():.3f}, "
              f"-1 {(K_X == -1).mean():.3f}")
        for ci, (name, fn, meta) in enumerate(couplings):
            mon = _monitor_start()
            t_start = time.time()
            rng_c = np.random.default_rng(seed * 1000 + 7 + ci)
            Zp, Xp, KZp, KXp = fn(Z, X, K_Z, K_X, rng_c)
            t_cost = transport_cost(Zp, Xp)
            cov_err = branch_coverage_error(KXp)
            n_after = len(Zp)
            metrics, E_U_sq = estimate_curves(Zp, Xp, KXp, T_GRID)
            elapsed = time.time() - t_start
            cpu_pct, rss_mb = _monitor_end(mon)
            key = f"{name}_seed{seed}"
            results["runs"][key] = {
                "name": name,
                "seed": seed,
                "meta": meta,
                "n_pairs": n_after,
                "transport_cost": t_cost,
                "branch_coverage_err": cov_err,
                "E_U_sq": E_U_sq,
                "metrics": metrics,
                "wall_time_s": elapsed,
                "cpu_percent_during": cpu_pct,
                "rss_mb_after": rss_mb,
            }
            mid = len(T_GRID) // 2
            print(f"  [{ci+1:2d}/{n_couplings}] {name:30s}"
                  f"  N={n_after:6d}  cost={t_cost:.3f}  "
                  f"cov_err={cov_err:.3f}  "
                  f"A_v(0.5)@k80={metrics[mid]['per_k'][80]['A_v_raw']:.3f}  "
                  f"R_sw(0.5)={metrics[mid]['per_k'][80]['R_switch']:.3f}  "
                  f"({elapsed:.1f}s)  cpu={cpu_pct:.0f}%  rss={rss_mb:.0f}MB")
            is_last = (ci == n_couplings - 1
                       and seed == SEEDS[-1])
            if SLEEP_BETWEEN_COUPLINGS > 0 and not is_last:
                print(f"      ... cooling sleep {SLEEP_BETWEEN_COUPLINGS:.0f}s")
                time.sleep(SLEEP_BETWEEN_COUPLINGS)

    total = time.time() - t0_global
    print(f"\nTotal wall time: {total:.1f}s ({total/60:.1f} min)")

    out_json = res_dir / f"e3a_metrics_{TAG}.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"Wrote {out_json}")

    # Plotting moved to plot_e3a.py (run/plot split for the public repo).
    return results


if __name__ == "__main__":
    run()
