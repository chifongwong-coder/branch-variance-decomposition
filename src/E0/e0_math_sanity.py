"""E0: Math and implementation sanity tests for the path ambiguity toy suite.

Validates analytic identities against finite-difference / Monte Carlo so that
later experiments build on a verified oracle:

  T1 Posterior responsibilities sum to 1.
  T2 Mixture score = posterior-weighted average of component scores.
  T3 Analytic d/dx score agrees with finite difference.
  T4 Score Jacobian decomposition: total = sum_k r_k J_k + Cov_K|x(s_K).
  T5 Discrete 2-path equal-weight construction: cov trace = 1/4 ||U1-U2||^2.
  T6 Closed-form Flow Matching v*(y,t) = E[U_t|Y_t=y,T=t] for binary Gaussian
     target + linear interpolant + independent N(0,1) coupling agrees with
     local-bin Monte Carlo estimate.
  T7 Closed-form Flow Matching ambiguity
     A_v(t) = within + between
            = d s^2 / tau^2_t + (m^2 (1-t)^2 / tau^4_t) E[sech^2(t m Y_t/tau^2_t)]
     agrees with local-bin Monte Carlo estimate of E_Y Tr Cov(U_t|Y_t).

Run:
  python3 e0_math_sanity.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SEED = 0
rng = np.random.default_rng(SEED)
EPS = 1e-12

# ----------------------------------------------------------------------------
# Analytic helpers: 1D binary Gaussian mixture
# p(x) = 0.5 N(x; +m, eta2) + 0.5 N(x; -m, eta2)
# ----------------------------------------------------------------------------

def mixture_score_1d(x, m, eta2):
    a = m * x / eta2
    return (m * np.tanh(a) - x) / eta2

def mixture_score_deriv_1d(x, m, eta2):
    a = m * x / eta2
    sech2 = 1.0 / np.cosh(a) ** 2
    return -1.0 / eta2 + (m ** 2) * sech2 / (eta2 ** 2)

def responsibilities_1d(x, m, eta2):
    """Return P(K|x) for K in {-1, +1} as columns."""
    a = 2.0 * m * x / eta2
    r_plus = 1.0 / (1.0 + np.exp(-a))
    return np.stack([1.0 - r_plus, r_plus], axis=-1)

def component_scores_1d(x, m, eta2):
    """Return s_k(x) for k in {-1, +1} as columns."""
    s_minus = (-m - x) / eta2
    s_plus = (m - x) / eta2
    return np.stack([s_minus, s_plus], axis=-1)

# ----------------------------------------------------------------------------
# Analytic helpers: Flow Matching, target = binary Gaussian, source = N(0,1),
# independent coupling, linear interpolant Y_t = (1-t) Z + t X, U_t = X - Z.
# tau^2_t = (1-t)^2 + t^2 s^2.
# ----------------------------------------------------------------------------

def fm_tau2(t, s2):
    return (1.0 - t) ** 2 + (t ** 2) * s2

def fm_vstar_1d(y, t, m, s2):
    tau2 = fm_tau2(t, s2)
    return (np.tanh(t * m * y / tau2) * m * (1.0 - t)
            + y * (t * s2 - (1.0 - t))) / tau2

def fm_within_var(t, s2):
    """Per-dim Var(U_t | Y_t, K) = s^2 / tau^2_t (constant in y, k)."""
    return s2 / fm_tau2(t, s2)

def fm_av_closed_form_1d(t, m, s2, n_mc=400_000, seed=1):
    """A_v(t) for 1D binary Gaussian target."""
    tau2 = fm_tau2(t, s2)
    within = fm_within_var(t, s2)
    rl = np.random.default_rng(seed)
    k = rl.choice([-1, 1], size=n_mc)
    z = rl.standard_normal(n_mc)
    x = k * m + np.sqrt(s2) * rl.standard_normal(n_mc)
    y = (1.0 - t) * z + t * x
    sech2 = 1.0 / np.cosh(t * m * y / tau2) ** 2
    between = ((1.0 - t) ** 2 * (m ** 2) / (tau2 ** 2)) * sech2.mean()
    return within + between, within, between

def fm_sample_pairs(n, m, s2, seed=2):
    rl = np.random.default_rng(seed)
    k = rl.choice([-1, 1], size=n)
    z = rl.standard_normal(n)
    x = k * m + np.sqrt(s2) * rl.standard_normal(n)
    return z, x, k

# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------

def t1_responsibilities_sum_to_one():
    x = rng.normal(0.0, 2.0, size=500)
    r = responsibilities_1d(x, m=1.5, eta2=0.5)
    err = float(np.max(np.abs(r.sum(axis=-1) - 1.0)))
    assert err < 1e-12, err
    return err

def t2_mixture_score_eq_posterior_average():
    m, eta2 = 1.5, 0.4
    x = rng.normal(0.0, 2.0, size=500)
    s_full = mixture_score_1d(x, m, eta2)
    r = responsibilities_1d(x, m, eta2)
    s_k = component_scores_1d(x, m, eta2)
    s_avg = (r * s_k).sum(axis=-1)
    err = float(np.max(np.abs(s_avg - s_full)))
    assert err < 1e-12, err
    return err

def t3_score_derivative_vs_finite_difference():
    m, eta2 = 1.5, 0.6
    x = rng.normal(0.0, 2.0, size=500)
    h = 1e-4
    fd = (mixture_score_1d(x + h, m, eta2)
          - mixture_score_1d(x - h, m, eta2)) / (2.0 * h)
    an = mixture_score_deriv_1d(x, m, eta2)
    err = float(np.max(np.abs(fd - an)))
    assert err < 1e-5, err
    return err

def t4_jacobian_decomposition_1d():
    """1D: J_k = -1/eta2 for both components (isotropic).
       J_s = sum_k r_k J_k + Var_K|x(s_K) = -1/eta2 + r+(1-r+)(2m/eta2)^2.
    """
    m, eta2 = 1.2, 0.5
    x = rng.normal(0.0, 2.0, size=500)
    total = mixture_score_deriv_1d(x, m, eta2)
    within = -1.0 / eta2  # scalar, identical for both branches
    r = responsibilities_1d(x, m, eta2)
    s_k = component_scores_1d(x, m, eta2)
    s_mean = (r * s_k).sum(axis=-1, keepdims=True)
    switching = (r * (s_k - s_mean) ** 2).sum(axis=-1)
    rhs = within + switching
    err = float(np.max(np.abs(total - rhs)))
    assert err < 1e-12, err
    return err

def t5_finite_two_path_equal_weight_cov():
    U1 = np.array([2.0, 0.0])
    U2 = np.array([0.0, 2.0])
    U = np.stack([U1, U2])
    mu = U.mean(axis=0)
    cov = (U - mu).T @ (U - mu) / 2.0
    target = 0.25 * float(np.sum((U1 - U2) ** 2))
    err = float(abs(float(np.trace(cov)) - target))
    assert err < 1e-12, err
    return err

def t6_fm_vstar_closed_form_vs_mc():
    m, s2 = 2.0, 0.25 ** 2
    t = 0.5
    n = 1_000_000
    z, x, _ = fm_sample_pairs(n, m, s2, seed=11)
    y = (1.0 - t) * z + t * x
    u = x - z
    grid = np.array([-1.5, -0.8, -0.2, 0.2, 0.8, 1.5])
    half_width = 0.04
    diffs = []
    for y_q in grid:
        mask = np.abs(y - y_q) < half_width
        if mask.sum() < 500:
            continue
        mc_mean = float(u[mask].mean())
        cf_mean = float(fm_vstar_1d(y_q, t, m, s2))
        diffs.append((y_q, mc_mean, cf_mean, abs(mc_mean - cf_mean)))
    err = max(d[3] for d in diffs)
    assert err < 0.05, f"max |MC - closed-form| = {err}; {diffs}"
    return {"max_abs_err": err, "per_point": diffs}

def t6b_fm_vstar_closed_form_vs_mc_multi_t():
    """Same test at several t values: stress-test the closed form for v*.

    Statistical sanity rather than absolute tolerance: in regions where
    local conditional variance Var(U|Y_t) is large (e.g. between modes),
    the MC estimate of E[U|Y_t=y] has standard error sigma_U/sqrt(N_local).
    We require |MC - CF| < 4 * SE_MC, which is a robust signal-to-noise check
    for the closed form being correct.
    """
    m, s2 = 2.0, 0.25 ** 2
    n = 2_000_000
    out = {}
    failures = []
    for t in [0.2, 0.4, 0.6, 0.8]:
        z, x, _ = fm_sample_pairs(n, m, s2, seed=20 + int(10 * t))
        y = (1.0 - t) * z + t * x
        u = x - z
        grid = np.array([-1.5, -0.5, 0.0, 0.5, 1.5])
        hw = 0.06
        per_t = []
        for y_q in grid:
            mask = np.abs(y - y_q) < hw
            c = int(mask.sum())
            if c < 1000:
                continue
            u_local = u[mask]
            y_local = y[mask]
            mc_mean = float(u_local.mean())
            # Bin-averaged closed form: avoid bin-averaging bias from
            # nonlinear v*(y) inside the window by averaging CF at every
            # sample's actual y value within the bin.
            cf_bin_avg = float(fm_vstar_1d(y_local, t, m, s2).mean())
            cf_center = float(fm_vstar_1d(y_q, t, m, s2))
            se = float(u_local.std(ddof=1) / np.sqrt(c))
            err = abs(mc_mean - cf_bin_avg)
            per_t.append({
                "y": float(y_q), "n": c,
                "mc": mc_mean,
                "cf_bin_avg": cf_bin_avg, "cf_center": cf_center,
                "err": err, "se": se,
                "z_score": err / max(se, 1e-9),
            })
            if err > 4.0 * se:
                failures.append((t, y_q, err, se))
        out[f"t={t}"] = per_t
    assert not failures, f"v* multi-t z>4 failures: {failures}"
    return out

def t7_fm_av_closed_form_vs_local_estimator():
    m, s2 = 2.0, 0.25 ** 2
    t = 0.5
    n = 2_000_000
    z, x, _ = fm_sample_pairs(n, m, s2, seed=31)
    y = (1.0 - t) * z + t * x
    u = x - z
    # Local-bin estimator over equal-mass bins on y.
    edges = np.quantile(y, np.linspace(0.0, 1.0, 81))
    idx = np.clip(np.digitize(y, edges[1:-1]), 0, 79)
    n_bins = 80
    total = 0.0
    count = 0
    for b in range(n_bins):
        sel = idx == b
        c = int(sel.sum())
        if c < 200:
            continue
        local_var = float(u[sel].var(ddof=0))
        total += local_var * c
        count += c
    av_mc = total / count
    av_cf, within, between = fm_av_closed_form_1d(
        t, m, s2, n_mc=600_000, seed=32)
    rel_err = abs(av_mc - av_cf) / av_cf
    assert rel_err < 0.05, f"A_v rel_err={rel_err} mc={av_mc} cf={av_cf}"
    return {
        "rel_err": float(rel_err),
        "av_mc": float(av_mc),
        "av_cf": float(av_cf),
        "within_cf": float(within),
        "between_cf": float(between),
    }

# ----------------------------------------------------------------------------

def run_all():
    cases = [
        ("T1 responsibilities sum to 1", t1_responsibilities_sum_to_one),
        ("T2 mixture score = posterior average", t2_mixture_score_eq_posterior_average),
        ("T3 score derivative vs FD", t3_score_derivative_vs_finite_difference),
        ("T4 Jacobian decomposition (1D binary)", t4_jacobian_decomposition_1d),
        ("T5 finite 2-path equal-weight cov", t5_finite_two_path_equal_weight_cov),
        ("T6 v*(y, t=0.5) closed form vs MC", t6_fm_vstar_closed_form_vs_mc),
        ("T6b v* closed form vs MC, multi t", t6b_fm_vstar_closed_form_vs_mc_multi_t),
        ("T7 A_v(t=0.5) closed form vs MC", t7_fm_av_closed_form_vs_local_estimator),
    ]
    results = {}
    failed = 0
    print("E0: math and implementation sanity tests")
    print("-" * 72)
    for name, fn in cases:
        try:
            r = fn()
            results[name] = r
            if isinstance(r, dict):
                rep = r.get("rel_err", r.get("max_abs_err", "ok"))
            else:
                rep = r
            print(f"  PASS  {name:48s}  result={rep}")
        except AssertionError as e:  # noqa: PERF203
            failed += 1
            results[name] = {"failed": str(e)}
            print(f"  FAIL  {name:48s}  {e}")
    print("-" * 72)
    print(f"  {len(cases) - failed}/{len(cases)} passed")

    out_dir = Path(__file__).resolve().parents[2] / "results"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "e0_results.json", "w") as f:
        json.dump(results, f, indent=2, default=lambda o: str(o))
    print(f"  wrote {out_dir / 'e0_results.json'}")
    return failed

if __name__ == "__main__":
    raise SystemExit(run_all())
