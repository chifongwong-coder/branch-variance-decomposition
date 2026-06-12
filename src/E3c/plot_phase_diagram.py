"""Plot script for the (D_G, a, sigma_s) phase diagram.

Reads results/phase_diagram_heatmap.json (Stage P1, 5x5 grid x 4 couplings
x 5 seeds at t=0.5, k=80, biased 1/k) and produces:

  C1 mismatch heatmap on (log2 D_G, log10 a)
  C3@10 A_v_sem heatmap (same axes)
  collapse plot: scalar vs lsnr, with logistic fit on C1
             - C1 includes BOTH primary (mismatch) and A_v_sem readouts
               (because the mismatch curve is uniformly ~0.5 on the
                grid; A_v_sem is the variable that actually moves)
  paired diagnostic: A_v_sem(C1) - A_v_sem(C4) vs lsnr
  backup-axis collapse: 3 alternatives
             - sqrt(D_G):  log10(a^2 / sqrt(D_G) sigma_s^2)
             - sem-only:   log10(a^2 / sigma_s^2)
             - residual vs log D_G and log a
  semantic endpoint correlation rho_S = Corr(Z_S, X_S) per
             coupling (closed-form from transport_sem; zero-cost). Tests
             the expert-derived mechanism: C1 induces rho_S > 0 by using
             the S coordinate as a tie-breaker; C4 ignores S so rho_S ~ 0.
             Right panel: A_v_sem vs rho_S scatter (monotonic mechanism).

Also runs the section 5.1 identity sanity checks and the section 10.4
"C1 collapses to C4 in low-SNR" check; prints PASS/FAIL summary.

Outputs PNG to figures/.

Usage:
    python plot_phase_diagram.py [--input PATH] [--out-prefix STR]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# Headless backend for batch plotting; matches other plot_*.py scripts.
import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.figure_style import apply_paper_style, Palette  # noqa: E402

apply_paper_style()
PAL = Palette()

# ---------------------------------------------------------------------------
# Coupling palette (this experiment uses 4 couplings; legend stays consistent)
# ---------------------------------------------------------------------------
COUPLING_COLOR = {
    "C0":    PAL.C0,
    "C1":    PAL.C1,
    "C3@10": PAL.C3_grad[3],
    "C4":    PAL.C4,
}
COUPLING_LABEL = {
    "C0":    r"$C_0$ independent",
    "C1":    r"$C_1$ Euclidean OT",
    "C3@10": r"$C_3@\lambda{=}10$",
    "C4":    r"$C_4$ geometry-only",
}


# ---------------------------------------------------------------------------
# JSON -> per-cell scalar table
# ---------------------------------------------------------------------------

def load_cells(json_path: Path) -> dict:
    """Returns the raw payload from a phase_diagram_*.json file."""
    with open(json_path) as f:
        return json.load(f)


def extract_scalars(payload: dict, K: int = 80, t_target: float = 0.5
                    ) -> dict:
    """Per (D_G, a, sigma_s, coupling) extract mean / SD over seeds of:
       mismatch_rate, transport_full, transport_geom, transport_sem,
       A_v_sem (cond_avg @ K at t closest to t_target), R_switch (same).

    Returns a flat dict keyed by (D_G, a, sigma_s, coupling) -> dict of
    {mean, sd, seed_values}.
    """
    out = {}
    for cell in payload["cells"]:
        D_G = int(cell["D_G"])
        A = float(cell["A"])
        sigma_s = float(cell["sigma_s"])
        lsnr = float(cell["lsnr"])
        t_grid = cell["t_values"]
        ti = int(np.argmin(np.abs(np.asarray(t_grid) - t_target)))

        per_coupling = {}
        for run in cell["runs"]:
            tag = run["coupling"]
            pm = run["pair_metrics"]
            mt = run["metrics_per_t"][ti]
            ca = mt.get("cond_avg", {})
            if str(K) in ca:
                ca_K = ca[str(K)]
            elif K in ca:
                ca_K = ca[K]
            else:
                ca_K = None
            # rho_S = Corr(Z_S, X_S) derived from transport_sem (no rerun needed)
            # Z_S ~ N(0, 1) so Var(Z_S) = 1
            # X_S = a*K_X + sigma_s*eps with K_X in {-1,+1} balanced; Var(X_S) = a^2 + sigma_s^2
            # transport_sem = E[(X_S - Z_S)^2] = Var(X_S) + Var(Z_S) - 2 Cov
            # => Cov = (1 + a^2 + sigma_s^2 - transport_sem) / 2
            # => rho_S = Cov / sqrt(Var(X_S) * Var(Z_S)) = Cov / sqrt(a^2 + sigma_s^2)
            var_xs = A * A + sigma_s * sigma_s
            cov_zs_xs = 0.5 * (1.0 + var_xs - pm["transport_sem"])
            rho_S = cov_zs_xs / np.sqrt(var_xs) if var_xs > 0 else float("nan")

            # Gaussian-predicted A_v^S at t=1/2 under joint-Gaussian approx:
            #   q = a^2 + sigma_s^2  (= Var(X_S))
            #   sigma_Z^2 = 1
            #   c = rho_S * sqrt(q)
            #   Var(U_S|Y_{1/2}) = 4q(1 - rho_S^2) / (1 + q + 2*rho_S*sqrt(q))
            q_var = var_xs
            denom = 1.0 + q_var + 2.0 * rho_S * np.sqrt(q_var)
            if denom > 1e-12:
                A_v_gauss = 4.0 * q_var * (1.0 - rho_S * rho_S) / denom
            else:
                A_v_gauss = float("nan")

            scal = {
                "mismatch": pm["mismatch_rate"],
                "transport_full": pm["transport_full"],
                "transport_geom": pm["transport_geom"],
                "transport_sem": pm["transport_sem"],
                "rho_S": float(rho_S),
                "A_v_sem": (ca_K["sem"]["A_v_norm"] if ca_K else float("nan")),
                "A_v_sem_raw": (ca_K["sem"]["A_v_raw"] if ca_K else float("nan")),
                "A_v_gauss_pred": float(A_v_gauss),
                "R_switch_sem": (ca_K["sem"]["R_switch"] if ca_K else float("nan")),
                "A_v_full": (ca_K["full"]["A_v_norm"] if ca_K else float("nan")),
            }
            per_coupling.setdefault(tag, []).append(scal)

        agg = {}
        for tag, runs in per_coupling.items():
            keys = runs[0].keys()
            agg[tag] = {}
            for k in keys:
                vals = np.array([r[k] for r in runs], dtype=np.float64)
                agg[tag][k + "_mean"] = float(np.mean(vals))
                agg[tag][k + "_sd"] = float(np.std(vals, ddof=1)
                                            if len(vals) > 1 else 0.0)
                agg[tag][k + "_vals"] = vals.tolist()
        out[(D_G, A, sigma_s)] = {
            "lsnr": lsnr,
            "couplings": agg,
            "n_seeds": len(per_coupling.get(next(iter(per_coupling)), [])),
        }
    return out


# ---------------------------------------------------------------------------
# Section 5.1 identity sanity + section 10.4 C1-vs-C4 collapse checks
# ---------------------------------------------------------------------------

def identity_sanity(scalars: dict, tol: float = 0.02) -> list[str]:
    """Plan section 5.1. Returns list of WARN messages (empty = all pass)."""
    msgs = []
    for (D_G, A, sigma_s), entry in scalars.items():
        cup = entry["couplings"]
        # C0 mismatch ~ 0.5
        c0_mis = cup.get("C0", {}).get("mismatch_mean", float("nan"))
        if abs(c0_mis - 0.5) > tol:
            msgs.append(f"WARN cell(D_G={D_G}, a={A}): C0 mismatch="
                        f"{c0_mis:.4f}, expected 0.5+/-{tol}")
        # C4 mismatch ~ 0.5
        c4_mis = cup.get("C4", {}).get("mismatch_mean", float("nan"))
        if abs(c4_mis - 0.5) > tol:
            msgs.append(f"WARN cell(D_G={D_G}, a={A}): C4 mismatch="
                        f"{c4_mis:.4f}, expected 0.5+/-{tol}")
        # transport_full(C1) < transport_full(C0)
        c0_tf = cup.get("C0", {}).get("transport_full_mean", float("nan"))
        c1_tf = cup.get("C1", {}).get("transport_full_mean", float("nan"))
        if c1_tf > c0_tf:
            msgs.append(f"WARN cell(D_G={D_G}, a={A}): transport_full(C1)="
                        f"{c1_tf:.3f} > transport_full(C0)={c0_tf:.3f}")
    return msgs


def c1_c4_collapse(scalars: dict, lsnr_threshold: float = -3.0) -> list[str]:
    """Plan section 10.4 check. Reports cells with lsnr < threshold whose
    |Delta A_v_sem| > 2 * seed_SD (= "C1 and C4 do not match in low-SNR
    regime")."""
    msgs = []
    for (D_G, A, sigma_s), entry in scalars.items():
        if entry["lsnr"] >= lsnr_threshold:
            continue
        cup = entry["couplings"]
        c1 = cup.get("C1", {})
        c4 = cup.get("C4", {})
        d_av = c1.get("A_v_sem_mean", float("nan")) - c4.get("A_v_sem_mean", float("nan"))
        # use the larger SD of the two as the band
        sd = max(c1.get("A_v_sem_sd", 0.0), c4.get("A_v_sem_sd", 0.0))
        if abs(d_av) > 2 * sd and sd > 0:
            msgs.append(f"cell(D_G={D_G}, a={A}) lsnr={entry['lsnr']:.2f}: "
                        f"A_v_sem(C1)-A_v_sem(C4)={d_av:+.3f}, 2*SD={2*sd:.3f}"
                        f"  (C1 not equivalent to C4 even at low SNR)")
    return msgs


# ---------------------------------------------------------------------------
# Logistic-fit helper
# ---------------------------------------------------------------------------

def _logistic(x, lo, hi, mu, s):
    """y = lo + (hi - lo) / (1 + exp((x - mu)/s)).  Used on C1 mismatch
    (and on A_v_sem after rescaling): at x << mu (low SNR), y -> hi;
    at x >> mu, y -> lo."""
    return lo + (hi - lo) / (1.0 + np.exp((x - mu) / s))


def fit_logistic(x, y, fix_endpoints: tuple | None = None) -> dict:
    """Fit y = lo + (hi - lo) sigma((mu - x)/s) to (x, y) data.
    If fix_endpoints is (lo, hi), only fit (mu, s)."""
    try:
        if fix_endpoints is not None:
            lo, hi = fix_endpoints

            def f(xx, mu, s):
                return _logistic(xx, lo, hi, mu, s)

            popt, pcov = curve_fit(f, x, y, p0=[float(np.median(x)), 0.5],
                                   maxfev=5000)
            mu, s = popt
            yhat = f(np.asarray(x), mu, s)
            ok = True
        else:
            def f(xx, lo, hi, mu, s):
                return _logistic(xx, lo, hi, mu, s)

            popt, pcov = curve_fit(
                f, x, y, p0=[float(np.min(y)), float(np.max(y)),
                             float(np.median(x)), 0.5], maxfev=5000)
            lo, hi, mu, s = popt
            yhat = f(np.asarray(x), lo, hi, mu, s)
            ok = True
    except Exception as e:
        return {"ok": False, "error": str(e)}
    rms = float(np.sqrt(np.mean((np.asarray(y) - yhat) ** 2)))
    return {"ok": ok, "lo": float(lo), "hi": float(hi),
            "mu": float(mu), "s": float(s), "rms": rms}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _grid_arrays(scalars: dict, D_G_list: list[int], A_list: list[float],
                 coupling: str, key: str) -> tuple[np.ndarray, np.ndarray]:
    """Returns (Z_mean, Z_sd) shaped (n_a, n_D) for heatmap plotting.
    Rows are a values (log10 a, ascending in plot), cols are D_G (log2)."""
    n_d = len(D_G_list)
    n_a = len(A_list)
    Z_mean = np.full((n_a, n_d), np.nan)
    Z_sd = np.full((n_a, n_d), np.nan)
    for ia, A in enumerate(A_list):
        for id_, D_G in enumerate(D_G_list):
            entry = scalars.get((D_G, A, 1.0))
            if entry is None:
                continue
            cup = entry["couplings"].get(coupling)
            if cup is None:
                continue
            Z_mean[ia, id_] = cup.get(key + "_mean", float("nan"))
            Z_sd[ia, id_] = cup.get(key + "_sd", float("nan"))
    return Z_mean, Z_sd


def plot_heatmap(ax, Z_mean, Z_sd, D_G_list, A_list, title, cbar_label,
                 cmap="viridis", lsnr_contour: float | None = None,
                 sigma_s: float = 1.0):
    """Single heatmap with per-cell annotation `mean / sd`."""
    n_a, n_d = Z_mean.shape
    im = ax.imshow(Z_mean, origin="lower", aspect="auto", cmap=cmap,
                   extent=(-0.5, n_d - 0.5, -0.5, n_a - 0.5))
    ax.set_xticks(range(n_d))
    ax.set_xticklabels([fr"$2^{{{int(np.log2(d))}}}$" for d in D_G_list])
    ax.set_yticks(range(n_a))
    ax.set_yticklabels([f"{a:g}" for a in A_list])
    ax.set_xlabel(r"$D_G$")
    ax.set_ylabel(r"$a$")
    ax.set_title(title)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)
    # annotate each cell
    for ia in range(n_a):
        for id_ in range(n_d):
            m = Z_mean[ia, id_]
            sd = Z_sd[ia, id_]
            if np.isnan(m):
                continue
            ax.text(id_, ia, f"{m:.2f}\n±{sd:.2f}",
                    ha="center", va="center", fontsize=6.5,
                    color=("white" if m < 0.4 * np.nanmax(Z_mean) +
                           0.6 * np.nanmin(Z_mean) else "black"))
    # lsnr contour
    if lsnr_contour is not None:
        # build lsnr surface on cell-centered grid
        lsnr = np.array([[np.log10(A * A / (D_G * sigma_s ** 2))
                          for D_G in D_G_list] for A in A_list])
        xs = np.arange(n_d)
        ys = np.arange(n_a)
        XX, YY = np.meshgrid(xs, ys)
        cs = ax.contour(XX, YY, lsnr, levels=[lsnr_contour],
                        colors="black", linewidths=1.4, linestyles="--")
        ax.clabel(cs, fmt=fr"$\log_{{10}}\mathrm{{SNR}}={lsnr_contour:g}$",
                  fontsize=8)


def plot_mismatch_avsem(scalars: dict, D_G_list: list[int], A_list: list[float],
                   out_dir: Path, prefix: str):
    """C1 mismatch heatmap (left) + C3@10 A_v_sem heatmap (right)."""
    # C1 mismatch heatmap
    Z_mean_A, Z_sd_A = _grid_arrays(scalars, D_G_list, A_list, "C1", "mismatch")
    fig, ax = plt.subplots(figsize=(7, 5))
    plot_heatmap(ax, Z_mean_A, Z_sd_A, D_G_list, A_list,
                 r"$C_1$ mismatch on $(D_G, a)$ grid (5 seeds)",
                 r"$\Pr(K_X \neq C)$ after $C_1$",
                 cmap="magma", lsnr_contour=-2.0)
    fig.tight_layout()
    out = out_dir / f"{prefix}_C1_mismatch.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")

    # C3@10 A_v_sem heatmap
    Z_mean_B, Z_sd_B = _grid_arrays(scalars, D_G_list, A_list, "C3@10",
                                    "A_v_sem")
    fig, ax = plt.subplots(figsize=(7, 5))
    plot_heatmap(ax, Z_mean_B, Z_sd_B, D_G_list, A_list,
                 r"$C_3@\lambda{=}10$ semantic-only "
                 r"$\widetilde{\mathcal{A}}_v^S(\frac{1}{2}|C)$",
                 r"$\widetilde{\mathcal{A}}_v^S(0.5\,|\,C)$",
                 cmap="viridis", lsnr_contour=-2.0)
    fig.tight_layout()
    out = out_dir / f"{prefix}_C3lam10_AvSem.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def plot_collapse(scalars: dict, out_dir: Path, prefix: str):
    """Collapse plot.
    Left panel: mismatch vs lsnr per coupling (with logistic-fit attempt on C1).
    Right panel: A_v_sem vs lsnr per coupling (with logistic-fit on C1).
    """
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.5))

    # pre-collect (lsnr, mean, sd) per coupling
    couplings = ["C0", "C1", "C3@10", "C4"]
    bag = {c: {"lsnr": [], "mis_mean": [], "mis_sd": [],
               "av_mean": [], "av_sd": []} for c in couplings}
    for (D_G, A, sigma_s), entry in scalars.items():
        lsnr = entry["lsnr"]
        for c in couplings:
            cup = entry["couplings"].get(c)
            if cup is None:
                continue
            bag[c]["lsnr"].append(lsnr)
            bag[c]["mis_mean"].append(cup["mismatch_mean"])
            bag[c]["mis_sd"].append(cup["mismatch_sd"])
            bag[c]["av_mean"].append(cup["A_v_sem_mean"])
            bag[c]["av_sd"].append(cup["A_v_sem_sd"])

    # ---- Left: mismatch vs lsnr ----
    for c in couplings:
        x = np.array(bag[c]["lsnr"])
        y = np.array(bag[c]["mis_mean"])
        e = np.array(bag[c]["mis_sd"])
        axL.errorbar(x, y, yerr=e, fmt="o", ms=5.5, capsize=2.5,
                     color=COUPLING_COLOR[c], label=COUPLING_LABEL[c],
                     lw=1.0)
    # try logistic fit on C1
    xc1 = np.array(bag["C1"]["lsnr"])
    yc1 = np.array(bag["C1"]["mis_mean"])
    fit = fit_logistic(xc1, yc1, fix_endpoints=(0.0, 0.5))
    if fit["ok"]:
        xgrid = np.linspace(xc1.min(), xc1.max(), 100)
        yfit = _logistic(xgrid, 0.0, 0.5, fit["mu"], fit["s"])
        axL.plot(xgrid, yfit, ls="--", color=COUPLING_COLOR["C1"], lw=1.2,
                 alpha=0.7,
                 label=fr"$C_1$ logistic fit "
                       fr"$\hat\mu={fit['mu']:.2f}$, $\hat s={fit['s']:.2f}$,"
                       fr" rms={fit['rms']:.3f}")
    else:
        axL.text(0.04, 0.04, f"$C_1$ logistic fit failed:\n{fit['error']}",
                 transform=axL.transAxes, fontsize=7, color="gray")
    axL.axhline(0.5, color="gray", ls=":", lw=1, alpha=0.7,
                label="random = 0.5")
    axL.axhline(0.0, color="lightgray", ls=":", lw=1, alpha=0.7)
    axL.set_xlabel(r"$\log_{10}(a^2 / D_G \sigma_s^2)$")
    axL.set_ylabel(r"$\Pr(K_X \neq C)$ after coupling")
    axL.set_title("(a) Collapse on mismatch rate")
    axL.legend(loc="upper right", fontsize=7.5)

    # ---- Right: A_v_sem(C1) vs lsnr ----
    for c in couplings:
        x = np.array(bag[c]["lsnr"])
        y = np.array(bag[c]["av_mean"])
        e = np.array(bag[c]["av_sd"])
        axR.errorbar(x, y, yerr=e, fmt="s", ms=5.5, capsize=2.5,
                     color=COUPLING_COLOR[c], label=COUPLING_LABEL[c],
                     lw=1.0)
    xc1 = np.array(bag["C1"]["lsnr"])
    yc1 = np.array(bag["C1"]["av_mean"])
    fit2 = fit_logistic(xc1, yc1)
    if fit2["ok"]:
        xgrid = np.linspace(xc1.min(), xc1.max(), 100)
        yfit = _logistic(xgrid, fit2["lo"], fit2["hi"], fit2["mu"], fit2["s"])
        axR.plot(xgrid, yfit, ls="--", color=COUPLING_COLOR["C1"], lw=1.2,
                 alpha=0.7,
                 label=fr"$C_1$ logistic fit "
                       fr"$\hat\mu={fit2['mu']:.2f}$, $\hat s={fit2['s']:.2f}$,"
                       fr" rms={fit2['rms']:.3f}")
    axR.set_xlabel(r"$\log_{10}(a^2 / D_G \sigma_s^2)$")
    axR.set_ylabel(r"$\widetilde{\mathcal{A}}_v^S(\frac{1}{2}|C)$")
    axR.set_title("(b) Collapse on semantic-only ambiguity")
    axR.legend(loc="upper right", fontsize=7.5)

    fig.tight_layout()
    out = out_dir / f"{prefix}_collapse.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")
    return {"mismatch_fit": fit, "A_v_sem_fit": fit2}


def plot_paired_diagnostic(scalars: dict, out_dir: Path, prefix: str):
    """Paired Delta A_v_sem (C1 - C4) vs lsnr. Tests section 10.4."""
    rows = []
    for (D_G, A, sigma_s), entry in scalars.items():
        c1 = entry["couplings"].get("C1", {})
        c4 = entry["couplings"].get("C4", {})
        d = c1.get("A_v_sem_mean", np.nan) - c4.get("A_v_sem_mean", np.nan)
        sd = np.sqrt(c1.get("A_v_sem_sd", 0) ** 2 + c4.get("A_v_sem_sd", 0) ** 2)
        rows.append((entry["lsnr"], d, sd, D_G, A))
    rows.sort(key=lambda r: r[0])
    x = np.array([r[0] for r in rows])
    y = np.array([r[1] for r in rows])
    sd = np.array([r[2] for r in rows])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(x, y, yerr=sd, fmt="o-", ms=5.5, capsize=2.5,
                color=PAL.ref, lw=1.2)
    ax.axhline(0.0, color="gray", ls=":", lw=1)
    ax.set_xlabel(r"$\log_{10}(a^2 / D_G \sigma_s^2)$")
    ax.set_ylabel(r"$\widetilde{\mathcal{A}}_v^S(C_1) - \widetilde{\mathcal{A}}_v^S(C_4)$")
    ax.set_title(r"Paired diagnostic: $C_1$ vs $C_4$ on semantic ambiguity")
    fig.tight_layout()
    out = out_dir / f"{prefix}_C1minusC4.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def plot_backup_axes(scalars: dict, out_dir: Path, prefix: str,
                          primary_fit: dict | None = None):
    """Backup-axes 4-panel diagnostic.
       (E-1) collapse on sqrt(D_G) axis
       (E-2) collapse on semantic-only axis a^2 / sigma_s^2
       (E-3) primary-fit residual vs log_2 D_G
       (E-4) primary-fit residual vs log_10 a
    """
    rows = []
    for (D_G, A, sigma_s), entry in scalars.items():
        c1 = entry["couplings"].get("C1", {})
        av = c1.get("A_v_sem_mean", np.nan)
        sd = c1.get("A_v_sem_sd", 0.0)
        rows.append((D_G, A, sigma_s, entry["lsnr"], av, sd))
    arr = np.array(rows, dtype=object)
    D_G_arr = np.array([r[0] for r in rows], dtype=float)
    A_arr = np.array([r[1] for r in rows], dtype=float)
    sigma_arr = np.array([r[2] for r in rows], dtype=float)
    lsnr_arr = np.array([r[3] for r in rows], dtype=float)
    av_arr = np.array([r[4] for r in rows], dtype=float)
    sd_arr = np.array([r[5] for r in rows], dtype=float)

    lsnr_sqrt = np.log10(A_arr ** 2 / (np.sqrt(D_G_arr) * sigma_arr ** 2))
    lsnr_sem = np.log10(A_arr ** 2 / sigma_arr ** 2)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9.5))

    def scatter_with_fit(ax, x, y, sd, title, xlabel):
        ax.errorbar(x, y, yerr=sd, fmt="o", ms=5.5, capsize=2.5,
                    color=PAL.C1, lw=1.0)
        fit = fit_logistic(x, y)
        if fit["ok"]:
            xg = np.linspace(x.min(), x.max(), 100)
            ax.plot(xg, _logistic(xg, fit["lo"], fit["hi"],
                                  fit["mu"], fit["s"]),
                    ls="--", color=PAL.ref, lw=1.2,
                    label=fr"fit $\hat\mu={fit['mu']:.2f}$ rms={fit['rms']:.3f}")
            ax.legend(loc="upper right", fontsize=7.5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"$\widetilde{\mathcal{A}}_v^S(C_1)$")
        ax.set_title(title)

    scatter_with_fit(axes[0, 0], lsnr_sqrt, av_arr, sd_arr,
                     r"(E-1) Backup axis: $\sqrt{D_G}$ scaling",
                     r"$\log_{10}(a^2 / \sqrt{D_G} \sigma_s^2)$")
    scatter_with_fit(axes[0, 1], lsnr_sem, av_arr, sd_arr,
                     r"(E-2) Backup axis: semantic-only SNR",
                     r"$\log_{10}(a^2 / \sigma_s^2)$")

    # Residual panels w.r.t. primary fit on lsnr
    if primary_fit is None or not primary_fit.get("ok", False):
        for ax in (axes[1, 0], axes[1, 1]):
            ax.text(0.5, 0.5,
                    "primary fit unavailable\n(skipping residual panels)",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=10, color="gray")
            ax.set_xticks([])
            ax.set_yticks([])
    else:
        f = primary_fit
        yhat = _logistic(lsnr_arr, f["lo"], f["hi"], f["mu"], f["s"])
        resid = av_arr - yhat
        ax = axes[1, 0]
        ax.errorbar(np.log2(D_G_arr), resid, yerr=sd_arr, fmt="o", ms=5.5,
                    capsize=2.5, color=PAL.neg, lw=1.0)
        ax.axhline(0, color="gray", ls=":", lw=1)
        ax.set_xlabel(r"$\log_2 D_G$")
        ax.set_ylabel(r"residual of $\widetilde{\mathcal{A}}_v^S(C_1)$")
        ax.set_title(r"Primary-fit residual vs $\log_2 D_G$")
        ax = axes[1, 1]
        ax.errorbar(np.log10(A_arr), resid, yerr=sd_arr, fmt="o", ms=5.5,
                    capsize=2.5, color=PAL.pos, lw=1.0)
        ax.axhline(0, color="gray", ls=":", lw=1)
        ax.set_xlabel(r"$\log_{10} a$")
        ax.set_ylabel(r"residual of $\widetilde{\mathcal{A}}_v^S(C_1)$")
        ax.set_title(r"Primary-fit residual vs $\log_{10} a$")

    fig.tight_layout()
    out = out_dir / f"{prefix}_backup_axes.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def plot_rho_S(scalars: dict, out_dir: Path, prefix: str):
    """Semantic endpoint correlation rho_S = Corr(Z_S, X_S) vs lsnr,
    per coupling. Tests the expert-derived mechanism for C1 > C4 on A_v_sem:
    C1 induces rho_S > 0 by using the semantic coordinate in the Euclidean
    cost as a tie-breaker; C4 ignores S so rho_S ~ 0. Closed-form value
    derived from transport_sem (zero-cost post-processing, no rerun).

    Left panel:  rho_S vs lsnr, one curve per coupling.
    Right panel: A_v_sem vs rho_S scatter, C1 vs C4 marker shape, testing
                 whether rho_S monotonically explains the A_v_sem gap.
    """
    apply_paper_style()
    plt.rcParams.update({"axes.labelsize": 15, "xtick.labelsize": 15,
                         "ytick.labelsize": 15, "axes.titlesize": 15})
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.5))

    couplings = ["C0", "C1", "C3@10", "C4"]
    bag = {c: {"lsnr": [], "rho_mean": [], "rho_sd": [],
               "av_mean": [], "av_sd": []} for c in couplings}
    for (D_G, A, sigma_s), entry in scalars.items():
        lsnr = entry["lsnr"]
        for c in couplings:
            cup = entry["couplings"].get(c)
            if cup is None:
                continue
            bag[c]["lsnr"].append(lsnr)
            bag[c]["rho_mean"].append(cup.get("rho_S_mean", float("nan")))
            bag[c]["rho_sd"].append(cup.get("rho_S_sd", 0.0))
            bag[c]["av_mean"].append(cup.get("A_v_sem_mean", float("nan")))
            bag[c]["av_sd"].append(cup.get("A_v_sem_sd", 0.0))

    # per-coupling marker shapes, shared with the right panel so one legend serves both
    markers = {"C0": "o", "C1": "s", "C3@10": "^", "C4": "D"}

    # ---- Left: rho_S vs lsnr (scatter; multiple (D_G, a) can share an lsnr) ----
    for c in couplings:
        x = np.array(bag[c]["lsnr"])
        y = np.array(bag[c]["rho_mean"])
        e = np.array(bag[c]["rho_sd"])
        axL.errorbar(x, y, yerr=e, fmt=markers[c], ms=5.5, capsize=2.5,
                     color=COUPLING_COLOR[c], label=COUPLING_LABEL[c],
                     lw=0, elinewidth=0.8, alpha=0.9)
    axL.axhline(0.0, color="gray", ls=":", lw=1)
    axL.set_xlabel(r"$\log_{10}(a^2 / D_G \sigma_s^2)$")
    axL.set_ylabel(r"$\rho_S = \mathrm{Corr}(Z_S, X_S)$ per coupling")
    axL.set_title(r"(a) Semantic endpoint correlation vs $\ell\mathrm{SNR}$")

    # ---- Right: A_v_sem vs rho_S (mechanism test) ----
    for c in couplings:
        x = np.array(bag[c]["rho_mean"])
        y = np.array(bag[c]["av_mean"])
        ex = np.array(bag[c]["rho_sd"])
        ey = np.array(bag[c]["av_sd"])
        axR.errorbar(x, y, yerr=ey, xerr=ex, fmt=markers[c], ms=6.0,
                     capsize=2.5, color=COUPLING_COLOR[c],
                     label=COUPLING_LABEL[c], lw=0, elinewidth=0.8,
                     alpha=0.9)
    # closed-form reference: at t=1/2, scalar Gaussian Var(U_S|Y_t=1/2) ~ 2(s-c)
    # where s = average of Var(Z_S), Var(X_S) and c = Cov.
    # Normalized: A_v_sem_norm ~ (2 sqrt(Var_Z Var_X)(1 - rho_S)) / E||U_S||^2
    # For visual reference, plot the qualitative line y = 1 - rho.
    rho_grid = np.linspace(0.0, 1.0, 50)
    axR.plot(rho_grid, 1.0 - rho_grid, ls=":", color="gray", lw=1.2,
             label=r"qualitative ref: $1 - \rho_S$")
    axR.set_xlabel(r"$\rho_S = \mathrm{Corr}(Z_S, X_S)$")
    axR.set_ylabel(r"$\widetilde{\mathcal{A}}_v^S(0.5\,|\,C)$")
    axR.set_title(r"(b) Mechanism: $\widetilde{\mathcal{A}}_v^S$ drops as $\rho_S$ rises")
    axR.legend(loc="best", fontsize=11)

    fig.tight_layout()
    out = out_dir / f"{prefix}_rho_S.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def plot_gaussian_prediction(scalars: dict, scalars_p4b: dict | None,
                                  out_dir: Path, prefix: str):
    """Gaussian-prediction check: measured raw A_v^S vs Gaussian-predicted A_v^S at t=1/2.

    The Gaussian formula (Remark X.Y):
       Var(U_S|Y_{1/2}) = 4q(1 - rho_S^2) / (1 + q + 2*rho_S*sqrt(q))
    where q = a^2 + sigma_s^2 and rho_S = Corr(Z_S, X_S).

    Plots scatter on log scale. Points along y = x = perfect prediction.
    P4b cells (sigma_s in {0.5, 2.0}) overlaid as different markers to test
    whether sigma_s collapse-break is explained by the Gaussian mechanism
    (in which case P4b points should also fall on y = x).
    """
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.5))

    couplings = ["C0", "C1", "C3@10", "C4"]
    # P1 points
    bag = {c: {"meas": [], "pred": []} for c in couplings}
    for (D_G, A, sigma_s), entry in scalars.items():
        for c in couplings:
            cup = entry["couplings"].get(c)
            if cup is None:
                continue
            bag[c]["meas"].append(cup["A_v_sem_raw_mean"])
            bag[c]["pred"].append(cup["A_v_gauss_pred_mean"])
    # P4b points (sigma_s != 1)
    p4b_bag = {c: {"meas": [], "pred": [], "sigma": []} for c in couplings}
    if scalars_p4b is not None:
        for (D_G, A, sigma_s), entry in scalars_p4b.items():
            for c in couplings:
                cup = entry["couplings"].get(c)
                if cup is None:
                    continue
                p4b_bag[c]["meas"].append(cup["A_v_sem_raw_mean"])
                p4b_bag[c]["pred"].append(cup["A_v_gauss_pred_mean"])
                p4b_bag[c]["sigma"].append(sigma_s)

    # ---- Left: P1 only (test the formula at fixed sigma_s = 1) ----
    all_x = []
    all_y = []
    for c in couplings:
        x = np.array(bag[c]["pred"])
        y = np.array(bag[c]["meas"])
        axL.scatter(x, y, s=44, color=COUPLING_COLOR[c],
                    label=COUPLING_LABEL[c], alpha=0.85, edgecolor="none")
        all_x.extend(x.tolist()); all_y.extend(y.tolist())
    # y = x reference
    if all_x:
        lo = min(min(all_x), min(all_y)) * 0.85
        hi = max(max(all_x), max(all_y)) * 1.10
        ref = np.linspace(lo, hi, 50)
        axL.plot(ref, ref, ls="--", color="gray", lw=1.0,
                 label=r"$y = x$ (Gaussian exact)")
    # residual stats
    arr_x = np.array(all_x); arr_y = np.array(all_y)
    mask = np.isfinite(arr_x) & np.isfinite(arr_y) & (arr_x > 0)
    if mask.any():
        rel_err = (arr_y[mask] - arr_x[mask]) / arr_x[mask]
        axL.text(0.04, 0.96,
                 fr"$\sigma_s{{=}}1$ heatmap: "
                 fr"mean rel err = {rel_err.mean():+.3f}, "
                 fr"|err| RMS = {np.sqrt((rel_err ** 2).mean()):.3f}",
                 transform=axL.transAxes, fontsize=8, va="top",
                 bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="white", edgecolor="gray", lw=0.5))
    axL.set_xlabel(r"Gaussian-predicted $\mathrm{Var}(U_S|Y_{1/2})$")
    axL.set_ylabel(r"Measured raw $A_v^S$ (kNN @ k=80, biased $1/k$)")
    axL.set_title(r"(a) Gaussian formula test on the $\sigma_s{=}1$ heatmap (25 cells)")
    axL.legend(loc="lower right", fontsize=7.5)
    axL.set_xscale("log"); axL.set_yscale("log")

    # ---- Right: P1 + P4b overlay (test sigma_s collapse break) ----
    sig_marker = {0.5: "s", 2.0: "D"}
    for c in couplings:
        x = np.array(bag[c]["pred"])
        y = np.array(bag[c]["meas"])
        axR.scatter(x, y, s=44, color=COUPLING_COLOR[c],
                    label=COUPLING_LABEL[c] + r" ($\sigma_s{=}1$)",
                    alpha=0.6, edgecolor="none")
        if scalars_p4b is not None:
            x4 = np.array(p4b_bag[c]["pred"])
            y4 = np.array(p4b_bag[c]["meas"])
            sigs = np.array(p4b_bag[c]["sigma"])
            for sig_v in [0.5, 2.0]:
                m = sigs == sig_v
                if m.any():
                    axR.scatter(x4[m], y4[m], s=90,
                                marker=sig_marker[sig_v],
                                color=COUPLING_COLOR[c],
                                facecolor="none",
                                linewidth=1.5,
                                label=(c + fr" ($\sigma_s$ sanity, $\sigma_s={sig_v:g}$)"
                                       if c == "C1" else None))
    # y = x ref using both P1 and P4b
    extra_x = []
    extra_y = []
    if scalars_p4b is not None:
        for c in couplings:
            extra_x.extend(p4b_bag[c]["pred"])
            extra_y.extend(p4b_bag[c]["meas"])
    full_x = all_x + extra_x
    full_y = all_y + extra_y
    if full_x:
        lo = min(min(full_x), min(full_y)) * 0.85
        hi = max(max(full_x), max(full_y)) * 1.10
        ref = np.linspace(lo, hi, 50)
        axR.plot(ref, ref, ls="--", color="gray", lw=1.0)
    # P4b residual stats
    if scalars_p4b is not None:
        ex = np.array(extra_x); ey = np.array(extra_y)
        m = np.isfinite(ex) & np.isfinite(ey) & (ex > 0)
        if m.any():
            rel_err4 = (ey[m] - ex[m]) / ex[m]
            axR.text(0.04, 0.96,
                     fr"$\sigma_s$ sanity ($\sigma_s \in \{{0.5, 2\}}$): "
                     fr"mean rel err = {rel_err4.mean():+.3f}, "
                     fr"|err| RMS = {np.sqrt((rel_err4 ** 2).mean()):.3f}",
                     transform=axR.transAxes, fontsize=8, va="top",
                     bbox=dict(boxstyle="round,pad=0.3",
                              facecolor="white", edgecolor="gray", lw=0.5))
    axR.set_xlabel(r"Gaussian-predicted $\mathrm{Var}(U_S|Y_{1/2})$")
    axR.set_ylabel(r"Measured raw $A_v^S$")
    axR.set_title(r"(b) Heatmap + $\sigma_s$ sanity cells: do off-axis $\sigma_s$ points lie on the same $y=x$?")
    axR.legend(loc="lower right", fontsize=6.5)
    axR.set_xscale("log"); axR.set_yscale("log")

    fig.tight_layout()
    out = out_dir / f"{prefix}_gaussian_prediction.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parents[2]
    parser.add_argument("--input", type=str,
                        default=str(here / "results" / "phase_diagram_heatmap.json"))
    parser.add_argument("--p4b", type=str,
                        default=str(here / "results" / "phase_diagram_sigma_sanity.json"),
                        help="P4b JSON for the Gaussian-prediction overlay; pass --p4b='' to skip")
    parser.add_argument("--out-prefix", type=str, default="phase_diagram")
    args = parser.parse_args()

    json_path = Path(args.input).resolve()
    out_dir = here / "figures"
    out_dir.mkdir(exist_ok=True)

    print(f"reading {json_path}")
    payload = load_cells(json_path)
    print(f"  stage={payload['stage']}  cells={len(payload['cells'])}  "
          f"wall_s={payload['wall_s']:.1f}")

    scalars = extract_scalars(payload, K=80, t_target=0.5)
    # Optional: load P4b for the Gaussian-prediction overlay
    scalars_p4b = None
    if args.p4b and Path(args.p4b).exists():
        print(f"reading P4b {args.p4b}")
        payload_p4b = load_cells(Path(args.p4b))
        scalars_p4b = extract_scalars(payload_p4b, K=80, t_target=0.5)
    # axis lists (sorted, unique)
    D_G_list = sorted({k[0] for k in scalars})
    A_list = sorted({k[1] for k in scalars})
    print(f"  D_G_grid={D_G_list}  A_grid={A_list}")

    # section 5.1
    warns = identity_sanity(scalars)
    if warns:
        print("\nIdentity sanity WARNINGS:")
        for w in warns:
            print(f"  {w}")
    else:
        print("\nIdentity sanity PASS (section 5.1)")

    # section 10.4
    c1c4 = c1_c4_collapse(scalars, lsnr_threshold=-3.0)
    if c1c4:
        print("\nC1<->C4 collapse FAILURES at low-SNR (section 10.4):")
        for m in c1c4:
            print(f"  {m}")
    else:
        print("\nC1<->C4 collapse: no failures at lsnr < -3 (section 10.4)")

    # Plots
    plot_mismatch_avsem(scalars, D_G_list, A_list, out_dir, args.out_prefix)
    fits = plot_collapse(scalars, out_dir, args.out_prefix)
    plot_paired_diagnostic(scalars, out_dir, args.out_prefix)
    plot_backup_axes(scalars, out_dir, args.out_prefix,
                          primary_fit=fits.get("A_v_sem_fit"))
    plot_rho_S(scalars, out_dir, args.out_prefix)
    plot_gaussian_prediction(scalars, scalars_p4b, out_dir,
                                  args.out_prefix)

    print("\n--- Summary ---")
    for name, fit in fits.items():
        if fit["ok"]:
            print(f"  C1 {name}: mu={fit['mu']:+.3f}  s={fit['s']:+.3f}  "
                  f"rms={fit['rms']:.4f}  lo={fit['lo']:.3f}  hi={fit['hi']:.3f}")
        else:
            print(f"  C1 {name}: FIT FAILED  {fit['error']}")


if __name__ == "__main__":
    main()
