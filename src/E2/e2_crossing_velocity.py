"""E2: Finite-mixture crossing-path velocity ambiguity (Flow Matching oracle).

Goal
----
Validate the Flow Matching analogue of E1: when multiple source-target paths
pass through the same intermediate region, the optimal single-valued velocity
field is the posterior average, and the irreducible regression error is the
conditional velocity variance.

Setup (2D, two probabilistic branches, equal prior 1/2)
  Branch A:  Z_A ~ N((-1, 0), sigma_b^2 I),  X_A ~ N((+1, 0), sigma_b^2 I)
             U_A = X_A - Z_A,  E[U_A] = (2, 0)
  Branch B:  Z_B ~ N((0, -1), sigma_b^2 I),  X_B ~ N((0, +1), sigma_b^2 I)
             U_B = X_B - Z_B,  E[U_B] = (0, 2)

Linear interpolant Y_t = (1-t) Z + t X.  Both branches concentrate near the
origin at t=0.5 (paths cross), so the conditional velocity distribution there
is multi-modal with directions (2,0) and (0,2).

Closed-form (small sigma_b limit; full derivation in §1 of the script)
  sigma_y^2(t) = ((1-t)^2 + t^2) sigma_b^2
  posterior P(K=A | Y_t = y, t) = sigmoid((2t-1)(y0 - y1) / sigma_y^2(t))
  E[r_A r_B](t)  computed once via Monte Carlo over Y_t marginal
  A_v_approx(t)  = 2 sigma_b^2 d  +  ||U_A - U_B||^2 E[r_A r_B](t)

Metrics on a t-grid (with kNN local-cov estimator)
  - raw  A_v(t)  =  E_Y Tr Cov(U_t | Y_t, T=t)
  - normalized   A_v(t) / E||U_t||^2
  - explained    R_v^2(t) = E||E[U_t|Y_t]||^2 / E||U_t||^2
  - branch       H(K | Y_t)  (closed form, sanity)
  - per-branch occupancy and posterior masses

Robustness
  - kNN bandwidth sweep k in {10, 30, 50, 100, 200, 400} at t = 0.5
  - deterministic 4-point sanity verifying Cov trace -> (1/4) ||U_A - U_B||^2
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.figure_style import apply_paper_style, Palette
apply_paper_style()

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------

SIGMA_B = 0.1
N_PER_BRANCH = 60_000
T_GRID = np.linspace(0.05, 0.95, 19)
KNN_K = 80
KNN_K_SWEEP = (10, 30, 50, 100, 200, 400)
SEED = 0

U_A_mean = np.array([2.0, 0.0])
U_B_mean = np.array([0.0, 2.0])
ZA_mean = np.array([-1.0, 0.0])
XA_mean = np.array([1.0, 0.0])
ZB_mean = np.array([0.0, -1.0])
XB_mean = np.array([0.0, 1.0])

# ----------------------------------------------------------------------------
# Sampling
# ----------------------------------------------------------------------------

def sample_pairs(n_per_branch=N_PER_BRANCH, sigma_b=SIGMA_B, seed=SEED):
    rng = np.random.default_rng(seed)
    n = n_per_branch
    # Branch A
    z_a = ZA_mean + sigma_b * rng.standard_normal((n, 2))
    x_a = XA_mean + sigma_b * rng.standard_normal((n, 2))
    # Branch B
    z_b = ZB_mean + sigma_b * rng.standard_normal((n, 2))
    x_b = XB_mean + sigma_b * rng.standard_normal((n, 2))
    z = np.concatenate([z_a, z_b], axis=0)
    x = np.concatenate([x_a, x_b], axis=0)
    k = np.concatenate([np.zeros(n, dtype=np.int64),
                        np.ones(n, dtype=np.int64)], axis=0)
    perm = rng.permutation(2 * n)
    return z[perm], x[perm], k[perm]


def interpolate(z, x, t):
    return (1.0 - t) * z + t * x


def velocity(z, x):
    return x - z


# ----------------------------------------------------------------------------
# Closed-form approximations (small sigma_b limit)
# ----------------------------------------------------------------------------

def posterior_rA_closed(y, t, sigma_b=SIGMA_B):
    """P(K=A | Y_t=y, t) for the 2-branch Gaussian-Gaussian-isotropic setup.

    Derivation: log p(y|K=A) - log p(y|K=B) = 2(2t-1)(y0 - y1) / (2 sigma_y^2)
                = (2t-1)(y0 - y1) / sigma_y^2.
    """
    sigma_y2 = ((1.0 - t) ** 2 + t ** 2) * sigma_b ** 2
    arg = (2.0 * t - 1.0) * (y[..., 0] - y[..., 1]) / sigma_y2
    return 1.0 / (1.0 + np.exp(-arg))


def expected_rA_rB(t, sigma_b=SIGMA_B, n_mc=400_000, seed=11):
    """E_Y[r_A(Y_t) r_B(Y_t)] via Monte Carlo over the true marginal of Y_t."""
    rng = np.random.default_rng(seed + int(1000 * t))
    n = n_mc // 2
    z_a = ZA_mean + sigma_b * rng.standard_normal((n, 2))
    x_a = XA_mean + sigma_b * rng.standard_normal((n, 2))
    z_b = ZB_mean + sigma_b * rng.standard_normal((n, 2))
    x_b = XB_mean + sigma_b * rng.standard_normal((n, 2))
    y_a = (1.0 - t) * z_a + t * x_a
    y_b = (1.0 - t) * z_b + t * x_b
    y = np.concatenate([y_a, y_b], axis=0)
    rA = posterior_rA_closed(y, t, sigma_b)
    return float((rA * (1.0 - rA)).mean())


def closed_form_av(t, sigma_b=SIGMA_B):
    """A_v(t) under small-sigma_b approximation.

    Within-branch (per dim): Var(U_t | Y_t, K=k) = sigma_b^2 / ((1-t)^2 + t^2)
      derivation: Z, X ~ N(.,sigma_b^2 I) independent given K, and the
      bivariate-normal regression of U=X-Z on Y_t=(1-t)Z+tX yields
      Var(U|Y,K)/dim = sigma_b^2 (1-t)^2/tau_y^2 + sigma_b^2 t^2/tau_y^2
                       + 2 t(1-t) sigma_b^2/tau_y^2
                     = sigma_b^2 ((1-t)+t)^2 / tau_y^2 = sigma_b^2 / tau_y^2,
      where tau_y^2 = (1-t)^2 + t^2.

    Between-branch (tight-branch limit): trace contribution at y is
      r_A(1-r_A) ||U_A_mean - U_B_mean||^2.
    """
    d = 2
    e_rrA = expected_rA_rB(t, sigma_b)
    tau_y2 = (1.0 - t) ** 2 + t ** 2
    within_per_dim = sigma_b ** 2 / tau_y2
    within = d * within_per_dim
    between = float(np.sum((U_A_mean - U_B_mean) ** 2)) * e_rrA
    return within + between, within, between, e_rrA


def closed_form_branch_entropy(t, sigma_b=SIGMA_B, n_mc=400_000, seed=22):
    """E_Y H(K | Y_t) via the closed-form posterior."""
    rng = np.random.default_rng(seed + int(1000 * t))
    n = n_mc // 2
    z_a = ZA_mean + sigma_b * rng.standard_normal((n, 2))
    x_a = XA_mean + sigma_b * rng.standard_normal((n, 2))
    z_b = ZB_mean + sigma_b * rng.standard_normal((n, 2))
    x_b = XB_mean + sigma_b * rng.standard_normal((n, 2))
    y_a = (1.0 - t) * z_a + t * x_a
    y_b = (1.0 - t) * z_b + t * x_b
    y = np.concatenate([y_a, y_b], axis=0)
    rA = posterior_rA_closed(y, t, sigma_b)
    eps = 1e-12
    H = -(rA * np.log(np.clip(rA, eps, 1.0))
          + (1.0 - rA) * np.log(np.clip(1.0 - rA, eps, 1.0)))
    return float(H.mean())


# ----------------------------------------------------------------------------
# kNN local conditional covariance estimator
# ----------------------------------------------------------------------------

def knn_local_av_and_explained(y, u, k=KNN_K, n_query=None, seed=33):
    """Estimate:
       A_v_hat   = (1/n_query) sum_q Tr  Cov(U | Y in N_k(q))    (raw ambiguity)
       Eu_norm2  = (1/n_query) sum_q  ||(1/k) sum_{i in N_k(q)} u_i||^2
                 (proxy for E_Y ||E[U|Y]||^2)

    Both via the same kNN over the empirical sample Y.
    """
    tree = cKDTree(y)
    n = y.shape[0]
    if n_query is None or n_query >= n:
        qidx = np.arange(n)
    else:
        rng = np.random.default_rng(seed)
        qidx = rng.choice(n, size=n_query, replace=False)
    q = y[qidx]
    _, idxs = tree.query(q, k=k)
    U = u[idxs]                                   # (Q, k, d)
    mu = U.mean(axis=1, keepdims=True)            # (Q, 1, d)
    diff = U - mu
    # biased 1/k local trace (matches the within/between split convention
    # used by knn_local_within_between, so total = within + between exactly)
    trace_cov = (diff ** 2).sum(axis=(1, 2)) / k
    # local conditional mean squared norm
    mu_sq = (mu[:, 0, :] ** 2).sum(axis=-1)
    return float(trace_cov.mean()), float(mu_sq.mean())


def knn_local_within_between(y, u, k_labels, k=KNN_K):
    """Law-of-total-covariance decomposition of Tr Cov(U | Y in N_k(q)).

    Splits each kNN neighborhood by branch label and reports:
      A_within   ~  E_Y E[ Tr Cov(U | Y, K) ]
      A_between  ~  E_Y Tr Cov_K( E[U | Y, K] | Y )
    so that A_within + A_between = Tr Cov(U | Y) (with biased 1/k estimator).

    Returns means over all queries.
    """
    tree = cKDTree(y)
    _, idxs = tree.query(y, k=k)             # (N, k)
    U_n = u[idxs]                            # (N, k, d)
    K_n = k_labels[idxs]                     # (N, k)

    mask_A = (K_n == 0).astype(np.float64)   # (N, k)
    mask_B = 1.0 - mask_A
    n_A = mask_A.sum(axis=1)                 # (N,)
    n_B = mask_B.sum(axis=1)

    sum_A = (mask_A[..., None] * U_n).sum(axis=1)  # (N, d)
    sum_B = (mask_B[..., None] * U_n).sum(axis=1)  # (N, d)
    overall = (sum_A + sum_B) / k                  # (N, d)

    safe_nA = np.maximum(n_A, 1)
    safe_nB = np.maximum(n_B, 1)
    mean_A = np.where(n_A[:, None] > 0, sum_A / safe_nA[:, None], overall)
    mean_B = np.where(n_B[:, None] > 0, sum_B / safe_nB[:, None], overall)

    # within-branch sum-of-squares
    diff_A = (U_n - mean_A[:, None, :]) * mask_A[..., None]
    diff_B = (U_n - mean_B[:, None, :]) * mask_B[..., None]
    ss_within = (diff_A ** 2).sum(axis=(1, 2)) + (diff_B ** 2).sum(axis=(1, 2))
    within_per_q = ss_within / k

    # between-branch: weighted Var of branch means
    dA = mean_A - overall
    dB = mean_B - overall
    between_per_q = (n_A * (dA ** 2).sum(axis=-1)
                     + n_B * (dB ** 2).sum(axis=-1)) / k

    return float(within_per_q.mean()), float(between_per_q.mean())


# ----------------------------------------------------------------------------
# Main run
# ----------------------------------------------------------------------------

def run():
    rng = np.random.default_rng(SEED)
    z, x, k_labels = sample_pairs()

    out_dir = Path(__file__).resolve().parents[2]
    res_dir = out_dir / "results"
    res_dir.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # 0. Deterministic 4-point sanity (cross at origin at t=0.5).
    #    Verify Tr Cov(U | Y=(0,0)) -> (1/4) ||U_A - U_B||^2 = 2 with 4 paths.
    # ------------------------------------------------------------------
    det_U = np.array([[2.0, 0.0],
                      [2.0, 0.0],
                      [0.0, 2.0],
                      [0.0, 2.0]])
    det_cov = np.cov(det_U.T, ddof=0)
    det_trace = float(np.trace(det_cov))
    det_target = 0.25 * float(np.sum((U_A_mean - U_B_mean) ** 2))
    print(f"  [sanity] det 4-path Tr Cov = {det_trace:.6f}  "
          f"(target {det_target:.6f})")

    # ------------------------------------------------------------------
    # 1. Curves on the t-grid: empirical A_v / normalized / R_v^2 / H(K|Y_t)
    #    + closed-form comparison.
    # ------------------------------------------------------------------
    rows = []
    for t in T_GRID:
        y = interpolate(z, x, t)
        u = velocity(z, x)
        # empirical via kNN
        av_hat, mu_sq_hat = knn_local_av_and_explained(y, u, k=KNN_K)
        # decomposed empirical via branch labels
        within_emp, between_emp = knn_local_within_between(
            y, u, k_labels, k=KNN_K)
        e_u2 = float((u ** 2).sum(axis=1).mean())
        r_switch_emp = between_emp / max(within_emp + between_emp, 1e-12)
        # closed-form approx
        av_cf, within_cf, between_cf, e_rrA = closed_form_av(t)
        H_cf = closed_form_branch_entropy(t)
        r_switch_cf = between_cf / max(av_cf, 1e-12)
        rows.append({
            "t": float(t),
            "av_hat": av_hat,
            "av_norm_hat": av_hat / e_u2,
            "R_v2_hat": mu_sq_hat / e_u2,
            "E_u2": e_u2,
            "within_emp": within_emp,
            "between_emp": between_emp,
            "r_switch_emp": r_switch_emp,
            "av_cf": av_cf,
            "within_cf": within_cf,
            "between_cf": between_cf,
            "r_switch_cf": r_switch_cf,
            "e_rArB_cf": e_rrA,
            "H_KY_cf": H_cf,
        })

    t_arr = np.array([r["t"] for r in rows])
    av_hat = np.array([r["av_hat"] for r in rows])
    av_norm_hat = np.array([r["av_norm_hat"] for r in rows])
    R_v2_hat = np.array([r["R_v2_hat"] for r in rows])
    E_u2 = np.array([r["E_u2"] for r in rows])
    av_cf = np.array([r["av_cf"] for r in rows])
    within_cf_arr = np.array([r["within_cf"] for r in rows])
    between_cf_arr = np.array([r["between_cf"] for r in rows])
    within_emp_arr = np.array([r["within_emp"] for r in rows])
    between_emp_arr = np.array([r["between_emp"] for r in rows])
    r_switch_emp_arr = np.array([r["r_switch_emp"] for r in rows])
    r_switch_cf_arr = np.array([r["r_switch_cf"] for r in rows])
    H_cf = np.array([r["H_KY_cf"] for r in rows])

    # ------------------------------------------------------------------
    # 2. Bandwidth sensitivity at t=0.5
    # ------------------------------------------------------------------
    t_center = 0.5
    y_center = interpolate(z, x, t_center)
    u_center = velocity(z, x)
    bandwidth_rows = []
    for k in KNN_K_SWEEP:
        av_k, mu_sq_k = knn_local_av_and_explained(y_center, u_center, k=k)
        e_u2_c = float((u_center ** 2).sum(axis=1).mean())
        bandwidth_rows.append({
            "k": int(k),
            "av_hat": av_k,
            "av_norm_hat": av_k / e_u2_c,
            "R_v2_hat": mu_sq_k / e_u2_c,
        })
    av_cf_05, _, _, _ = closed_form_av(t_center)

    # ------------------------------------------------------------------
    # 3. Per-branch occupancy at t=0.5 (closed-form posterior)
    # ------------------------------------------------------------------
    rA_05 = posterior_rA_closed(y_center, t_center)
    branch_a_mask = k_labels == 0
    occ_summary = {
        "fraction_branch_A_samples": float(branch_a_mask.mean()),
        "mean_rA_at_t0.5": float(rA_05.mean()),
        "mean_rA_for_true_A": float(rA_05[branch_a_mask].mean()),
        "mean_rA_for_true_B": float(rA_05[~branch_a_mask].mean()),
        "note": "At t=0.5 both branches concentrate near origin and the "
                "closed-form posterior gives r_A=0.5 everywhere "
                "regardless of y.",
    }

    # ------------------------------------------------------------------
    # 4. Panel-(a) sample positions, serialized so the plot script
    #    (plot_e2.py) can draw the figure from this JSON alone, no recompute.
    # ------------------------------------------------------------------
    nshow = 200
    idx = np.random.default_rng(1).choice(z.shape[0], size=nshow, replace=False)
    sample_positions = {
        "n_show": nshow,
        "t_show": [0.0, 0.5, 1.0],
        "y": {f"{t_show}": interpolate(z[idx], x[idx], t_show).tolist()
              for t_show in [0.0, 0.5, 1.0]},
    }

    # ------------------------------------------------------------------
    # 5. Save metrics
    # ------------------------------------------------------------------
    summary = {
        "config": {
            "sigma_b": SIGMA_B,
            "n_per_branch": N_PER_BRANCH,
            "t_grid": T_GRID.tolist(),
            "knn_k": KNN_K,
            "knn_k_sweep": list(KNN_K_SWEEP),
            "seed": SEED,
            "U_A_mean": U_A_mean.tolist(),
            "U_B_mean": U_B_mean.tolist(),
            "branch_centers": {
                "Z_A": ZA_mean.tolist(), "X_A": XA_mean.tolist(),
                "Z_B": ZB_mean.tolist(), "X_B": XB_mean.tolist(),
            },
        },
        "deterministic_4_point_sanity": {
            "trace_cov": det_trace,
            "target_quarter_norm_sq": det_target,
            "abs_err": abs(det_trace - det_target),
        },
        "curves": rows,
        "bandwidth_at_t0.5": bandwidth_rows,
        "av_cf_at_t0.5": av_cf_05,
        "sample_positions": sample_positions,
        "occupancy_at_t0.5": occ_summary,
        "key_indicators": {
            "argmax_av_t": float(t_arr[int(np.argmax(av_hat))]),
            "av_peak_value": float(av_hat.max()),
            "av_baseline_at_endpoints": [float(av_hat[0]), float(av_hat[-1])],
            "av_peak_to_baseline_ratio": float(
                av_hat.max() / max(av_hat[[0, -1]].mean(), 1e-12)),
        },
    }
    with open(res_dir / "e2_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ------------------------------------------------------------------
    # 6. Console report
    # ------------------------------------------------------------------
    print("E2 crossing-path velocity ambiguity")
    print("-" * 64)
    print(f"  config: sigma_b={SIGMA_B}, N={2*N_PER_BRANCH}, k={KNN_K}")
    print("  deterministic sanity: "
          f"trace={det_trace:.6f}, target={det_target:.3f}")
    print(f"  argmax A_v(t) at t = "
          f"{summary['key_indicators']['argmax_av_t']:.3f}")
    print(f"  peak A_v = {av_hat.max():.4f}, "
          f"endpoints A_v ~= {av_hat[0]:.4f} / {av_hat[-1]:.4f}")
    print(f"  closed-form at t=0.5: {av_cf_05:.4f}, "
          f"empirical at t=0.5: {av_hat[len(av_hat)//2]:.4f}")
    print(f"  H(K|Y_t)/log2 at t=0.5: {H_cf[len(H_cf)//2] / np.log(2.0):.4f}")
    mid = len(t_arr) // 2
    print(f"  R_switch (empirical) at t=0.5: "
          f"{r_switch_emp_arr[mid]:.4f}, closed form: "
          f"{r_switch_cf_arr[mid]:.4f}")
    print(f"  R_switch (empirical) at t=0.05/0.95: "
          f"{r_switch_emp_arr[0]:.4f} / {r_switch_emp_arr[-1]:.4f}")
    print(f"  within/between (empirical, at t=0.5): "
          f"{within_emp_arr[mid]:.4f} / {between_emp_arr[mid]:.4f}")
    print(f"  within/between (closed form, at t=0.5): "
          f"{within_cf_arr[mid]:.4f} / {between_cf_arr[mid]:.4f}")
    print(f"  bandwidth sweep k -> A_v at t=0.5:")
    for r in bandwidth_rows:
        print(f"    k={r['k']:4d}  raw={r['av_hat']:.4f}  "
              f"norm={r['av_norm_hat']:.4f}  R_v^2={r['R_v2_hat']:.4f}")
    print(f"  occupancy: {occ_summary}")


if __name__ == "__main__":
    run()
