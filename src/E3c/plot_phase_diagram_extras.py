"""Plot script for the phase-diagram supplementary stages.

Reads three JSONs and produces three figures:

  results/phase_diagram_lambda_sweep.json
            Lambda sweep at 5 corner+headline cells (C3@{1, 10, 100}).
            Tests whether the C3 phase boundary is intrinsic to the data
            or a property of the fixed lambda choice.

  results/phase_diagram_sigma_sanity.json + phase_diagram_heatmap.json
            sigma_s sanity at 4 cells; overlay P1 cells (sigma_s=1.0).
            Tests whether the lsnr collapse axis a^2/(D_G sigma_s^2)
            preserves A_v_sem(C1) when sigma_s varies.

  results/phase_diagram_t_sentinel.json
            Sentinel t-profile at 5 cells (5 t-points each).
            Tests whether t=0.5 is representative of the full t-trajectory.

Outputs PNG to figures/.

Usage:
    python plot_phase_diagram_extras.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.figure_style import apply_paper_style, Palette  # noqa: E402

apply_paper_style()
PAL = Palette()

COUPLING_COLOR = {
    "C0": PAL.C0,
    "C1": PAL.C1,
    "C3@1": PAL.C3_grad[0],
    "C3@10": PAL.C3_grad[2],
    "C3@100": PAL.C3_grad[4],
    "C4": PAL.C4,
}
COUPLING_LABEL = {
    "C0": r"$C_0$ indep",
    "C1": r"$C_1$ OT",
    "C3@1": r"$C_3@\lambda=1$",
    "C3@10": r"$C_3@\lambda=10$",
    "C3@100": r"$C_3@\lambda=100$",
    "C4": r"$C_4$ geom",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _scalars_one_cell(cell: dict, t_idx: int = 0) -> dict:
    """Per-coupling mean/SD over seeds at a given t-index for one cell.
    Returns dict[coupling_tag] -> {mismatch_mean, A_v_sem_mean, rho_S_mean, ...}"""
    D_G = cell["D_G"]
    A = cell["A"]
    sigma_s = cell["sigma_s"]
    var_xs = A * A + sigma_s * sigma_s
    by_c = {}
    for r in cell["runs"]:
        by_c.setdefault(r["coupling"], []).append(r)
    out = {}
    for c, runs in by_c.items():
        mis = np.array([r["pair_metrics"]["mismatch_rate"] for r in runs])
        ts = np.array([r["pair_metrics"]["transport_sem"] for r in runs])
        cov = 0.5 * (1.0 + var_xs - ts)
        rho_S = cov / np.sqrt(var_xs) if var_xs > 0 else np.zeros_like(cov)
        av = np.array([r["metrics_per_t"][t_idx]["cond_avg"]["80"]["sem"]["A_v_norm"]
                       for r in runs])
        rs = np.array([r["metrics_per_t"][t_idx]["cond_avg"]["80"]["sem"]["R_switch"]
                       for r in runs])
        out[c] = {
            "mismatch_mean": float(mis.mean()),
            "mismatch_sd":   float(mis.std(ddof=1) if len(mis) > 1 else 0.0),
            "A_v_sem_mean":  float(av.mean()),
            "A_v_sem_sd":    float(av.std(ddof=1) if len(av) > 1 else 0.0),
            "rho_S_mean":    float(rho_S.mean()),
            "rho_S_sd":      float(rho_S.std(ddof=1) if len(rho_S) > 1 else 0.0),
            "R_switch_mean": float(rs.mean()),
            "R_switch_sd":   float(rs.std(ddof=1) if len(rs) > 1 else 0.0),
        }
    return out


def _load(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# lambda sweep
# ---------------------------------------------------------------------------

def plot_lambda_sweep(p4a_path: Path, out_dir: Path, prefix: str):
    d = _load(p4a_path)
    lambdas = d["config"]["lambdas"]   # [1, 10, 100]

    cells = d["cells"]
    n = len(cells)
    apply_paper_style()
    plt.rcParams.update({"axes.labelsize": 14, "xtick.labelsize": 14,
                         "ytick.labelsize": 14, "axes.titlesize": 14})
    fig, axes = plt.subplots(2, n, figsize=(3.0 * n + 1.5, 6.5),
                             sharex=True, squeeze=False)

    for ci, cell in enumerate(cells):
        D_G, A, sigma_s = cell["D_G"], cell["A"], cell["sigma_s"]
        lsnr = cell["lsnr"]
        s = _scalars_one_cell(cell)
        title = fr"$D_G={D_G}, a={A:g}$  lsnr$={lsnr:+.2f}$"

        # Top: mismatch vs lambda
        ax = axes[0, ci]
        # C1 baseline (lambda-independent)
        c1_mis = s.get("C1", {}).get("mismatch_mean", float("nan"))
        ax.axhline(c1_mis, color=COUPLING_COLOR["C1"], ls=":", lw=1.0,
                   alpha=0.85, label=r"$C_1$")
        # C4 baseline
        c4_mis = s.get("C4", {}).get("mismatch_mean", float("nan"))
        ax.axhline(c4_mis, color=COUPLING_COLOR["C4"], ls="--", lw=1.0,
                   alpha=0.5, label=r"$C_4$")
        xs = lambdas
        ys = []
        ye = []
        for lam in lambdas:
            tag = f"C3@{int(lam)}" if lam == int(lam) else f"C3@{lam:g}"
            ys.append(s.get(tag, {}).get("mismatch_mean", float("nan")))
            ye.append(s.get(tag, {}).get("mismatch_sd", 0.0))
        ax.errorbar(xs, ys, yerr=ye, fmt="o-", ms=6, capsize=2.5,
                    color=PAL.ref, lw=1.4, label=r"$C_3@\lambda$")
        ax.axhline(0.5, color="gray", ls=":", lw=0.7, alpha=0.4)
        ax.set_xscale("log")
        ax.set_ylim(-0.03, 0.6)
        ax.set_title(title, fontsize=14)
        if ci == 0:
            ax.set_ylabel(r"mismatch")
            ax.legend(loc="upper right", fontsize=13)

        # Bottom: A_v_sem vs lambda
        ax = axes[1, ci]
        c1_av = s.get("C1", {}).get("A_v_sem_mean", float("nan"))
        ax.axhline(c1_av, color=COUPLING_COLOR["C1"], ls=":", lw=1.0,
                   alpha=0.85)
        c4_av = s.get("C4", {}).get("A_v_sem_mean", float("nan"))
        ax.axhline(c4_av, color=COUPLING_COLOR["C4"], ls="--", lw=1.0,
                   alpha=0.5)
        ys = []
        ye = []
        for lam in lambdas:
            tag = f"C3@{int(lam)}" if lam == int(lam) else f"C3@{lam:g}"
            ys.append(s.get(tag, {}).get("A_v_sem_mean", float("nan")))
            ye.append(s.get(tag, {}).get("A_v_sem_sd", 0.0))
        ax.errorbar(xs, ys, yerr=ye, fmt="s-", ms=6, capsize=2.5,
                    color=PAL.ref, lw=1.4)
        ax.set_xscale("log")
        ax.set_ylim(-0.03, 1.05)
        ax.set_xlabel(r"$\lambda$")
        if ci == 0:
            ax.set_ylabel(r"$\widetilde{\mathcal{A}}_v^S(0.5\,|\,C)$")

    fig.tight_layout()
    out = out_dir / f"{prefix}_lambda_sweep.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
# sigma_s sanity (overlay with P1 cells)
# ---------------------------------------------------------------------------

def plot_sigma_sanity(p4b_path: Path, p1_path: Path,
                           out_dir: Path, prefix: str):
    d_p4b = _load(p4b_path)
    d_p1 = _load(p1_path)

    # Collect P1 cells (sigma_s = 1.0) for overlay
    p1_points = []  # list of (lsnr, A_v_sem(C1)_mean, A_v_sem_sd, D_G, a, sigma_s)
    for cell in d_p1["cells"]:
        s = _scalars_one_cell(cell)
        p1_points.append((cell["lsnr"],
                          s["C1"]["A_v_sem_mean"],
                          s["C1"]["A_v_sem_sd"],
                          cell["D_G"], cell["A"], cell["sigma_s"]))
    p1_points.sort(key=lambda r: r[0])

    # Collect P4b points
    p4b_points = []
    for cell in d_p4b["cells"]:
        s = _scalars_one_cell(cell)
        p4b_points.append((cell["lsnr"],
                           s["C1"]["A_v_sem_mean"],
                           s["C1"]["A_v_sem_sd"],
                           cell["D_G"], cell["A"], cell["sigma_s"]))

    # Layout: two panels.  Left: A_v_sem(C1) vs lsnr, P1 (sigma=1) overlay
    # with P4b (sigma in {0.5, 2.0}).  Right: same but for rho_S(C1).
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.5))

    # Left
    x_p1 = np.array([r[0] for r in p1_points])
    y_p1 = np.array([r[1] for r in p1_points])
    e_p1 = np.array([r[2] for r in p1_points])
    axL.errorbar(x_p1, y_p1, yerr=e_p1, fmt="o", ms=5, capsize=2,
                 color=PAL.C1, alpha=0.45,
                 label=r"$\sigma_s{=}1$ heatmap (25 cells)")
    # P4b cells colored by sigma_s
    sig_colors = {0.5: "#e87a23", 2.0: "#3aa455"}
    for lsnr, av, sd, D_G, A, sigma_s in p4b_points:
        c = sig_colors.get(sigma_s, "black")
        axL.errorbar(lsnr, av, yerr=sd, fmt="s", ms=10, capsize=3,
                     color=c, lw=1.5,
                     label=(fr"$\sigma_s$ sanity, $\sigma_s={sigma_s:g}$"
                            if sigma_s not in {r[2] for r in p4b_points
                                                if id(r) < id((lsnr, av, sd, D_G, A, sigma_s))}
                            else None))
        axL.annotate(fr"$D_G={D_G},a={A:g},\sigma={sigma_s:g}$",
                     xy=(lsnr, av), xytext=(5, 7), textcoords="offset points",
                     fontsize=6.5, alpha=0.85)
    axL.set_xlabel(r"$\log_{10}(a^2 / D_G\sigma_s^2)$")
    axL.set_ylabel(r"$\widetilde{\mathcal{A}}_v^S(C_1)$ at $t=0.5$")
    axL.set_title(r"(a) Does the lsnr axis preserve $A_v^S(C_1)$ "
                  r"under $\sigma_s$ change?")
    # dedupe legend
    handles, labels = axL.get_legend_handles_labels()
    seen = set()
    h2, l2 = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l); h2.append(h); l2.append(l)
    axL.legend(h2, l2, loc="best", fontsize=7.5)

    # Right: same view but for rho_S(C1)
    p1_rho = []
    for cell in d_p1["cells"]:
        s = _scalars_one_cell(cell)
        p1_rho.append((cell["lsnr"], s["C1"]["rho_S_mean"], s["C1"]["rho_S_sd"]))
    p1_rho.sort(key=lambda r: r[0])
    p4b_rho = []
    for cell in d_p4b["cells"]:
        s = _scalars_one_cell(cell)
        p4b_rho.append((cell["lsnr"], s["C1"]["rho_S_mean"], s["C1"]["rho_S_sd"],
                        cell["D_G"], cell["A"], cell["sigma_s"]))
    x_p1 = np.array([r[0] for r in p1_rho])
    y_p1 = np.array([r[1] for r in p1_rho])
    e_p1 = np.array([r[2] for r in p1_rho])
    axR.errorbar(x_p1, y_p1, yerr=e_p1, fmt="o", ms=5, capsize=2,
                 color=PAL.C1, alpha=0.45,
                 label=r"$\sigma_s{=}1$ heatmap (25 cells)")
    for lsnr, av, sd, D_G, A, sigma_s in p4b_rho:
        c = sig_colors.get(sigma_s, "black")
        axR.errorbar(lsnr, av, yerr=sd, fmt="s", ms=10, capsize=3,
                     color=c, lw=1.5,
                     label=fr"$\sigma_s$ sanity, $\sigma_s={sigma_s:g}$"
                           if sigma_s not in {r[2] for r in p4b_rho
                                              if id(r) < id((lsnr, av, sd, D_G, A, sigma_s))}
                           else None)
    axR.set_xlabel(r"$\log_{10}(a^2 / D_G\sigma_s^2)$")
    axR.set_ylabel(r"$\rho_S(C_1)$")
    axR.set_title(r"(b) Does the lsnr axis preserve $\rho_S(C_1)$ "
                  r"under $\sigma_s$ change?")
    handles, labels = axR.get_legend_handles_labels()
    seen = set()
    h2, l2 = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l); h2.append(h); l2.append(l)
    axR.legend(h2, l2, loc="best", fontsize=7.5)

    fig.tight_layout()
    out = out_dir / f"{prefix}_sigma_sanity.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
# sentinel t-profile
# ---------------------------------------------------------------------------

def plot_t_profile(p2_path: Path, out_dir: Path, prefix: str):
    d = _load(p2_path)
    cells = d["cells"]
    t_grid = d["config"]["t_grid"]
    n = len(cells)
    fig, axes = plt.subplots(2, n, figsize=(3.0 * n + 1.5, 6.5),
                             sharex=True, squeeze=False)

    couplings = ["C0", "C1", "C3@10", "C4"]
    for ci, cell in enumerate(cells):
        D_G, A, sigma_s = cell["D_G"], cell["A"], cell["sigma_s"]
        lsnr = cell["lsnr"]

        # Build per-coupling per-t arrays (mean over seeds)
        per_c = {c: {"av_mean": [], "av_sd": [],
                     "rsw_mean": [], "rsw_sd": []} for c in couplings}
        for ti in range(len(t_grid)):
            by_c = {}
            for r in cell["runs"]:
                by_c.setdefault(r["coupling"], []).append(r)
            for c in couplings:
                avs = np.array([r["metrics_per_t"][ti]["cond_avg"]["80"]["sem"]["A_v_norm"]
                                for r in by_c[c]])
                rsws = np.array([r["metrics_per_t"][ti]["cond_avg"]["80"]["sem"]["R_switch"]
                                 for r in by_c[c]])
                per_c[c]["av_mean"].append(avs.mean())
                per_c[c]["av_sd"].append(avs.std(ddof=1) if len(avs) > 1 else 0.0)
                per_c[c]["rsw_mean"].append(rsws.mean())
                per_c[c]["rsw_sd"].append(rsws.std(ddof=1) if len(rsws) > 1 else 0.0)

        # Top: A_v_sem(t)
        ax = axes[0, ci]
        for c in couplings:
            ax.errorbar(t_grid, per_c[c]["av_mean"], yerr=per_c[c]["av_sd"],
                        fmt="o-", ms=4, capsize=2,
                        color=COUPLING_COLOR[c], label=COUPLING_LABEL[c],
                        lw=1.0)
        ax.axvline(0.5, color="gray", ls=":", lw=0.7, alpha=0.4)
        ax.set_title(fr"$D_G={D_G},a={A:g}$  lsnr$={lsnr:+.2f}$",
                     fontsize=9)
        if ci == 0:
            ax.set_ylabel(r"$\widetilde{\mathcal{A}}_v^S(t\,|\,C)$")
            ax.legend(loc="best", fontsize=7)

        # Bottom: R_switch(t)
        ax = axes[1, ci]
        for c in couplings:
            ax.errorbar(t_grid, per_c[c]["rsw_mean"], yerr=per_c[c]["rsw_sd"],
                        fmt="s-", ms=4, capsize=2,
                        color=COUPLING_COLOR[c], lw=1.0)
        ax.axvline(0.5, color="gray", ls=":", lw=0.7, alpha=0.4)
        ax.set_xlabel(r"$t$")
        if ci == 0:
            ax.set_ylabel(r"$R_{\rm switch}(t\,|\,C)$")

    fig.suptitle(r"Sentinel $t$-profile at 5 cells "
                 r"(5 seeds, $t\in\{0.1,0.3,0.5,0.7,0.9\}$): "
                 r"is $t=0.5$ representative?",
                 fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = out_dir / f"{prefix}_t_profile.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parents[2]
    parser.add_argument("--p4a", type=str,
                        default=str(here / "results" / "phase_diagram_lambda_sweep.json"))
    parser.add_argument("--p4b", type=str,
                        default=str(here / "results" / "phase_diagram_sigma_sanity.json"))
    parser.add_argument("--p2", type=str,
                        default=str(here / "results" / "phase_diagram_t_sentinel.json"))
    parser.add_argument("--p1", type=str,
                        default=str(here / "results" / "phase_diagram_heatmap.json"))
    parser.add_argument("--out-prefix", type=str, default="phase_diagram")
    args = parser.parse_args()

    out_dir = here / "figures"
    out_dir.mkdir(exist_ok=True)

    print(f"reading P4a {args.p4a}")
    plot_lambda_sweep(Path(args.p4a), out_dir, args.out_prefix)
    print(f"reading P4b {args.p4b} + P1 {args.p1}")
    plot_sigma_sanity(Path(args.p4b), Path(args.p1), out_dir,
                           args.out_prefix)
    print(f"reading P2 {args.p2}")
    plot_t_profile(Path(args.p2), out_dir, args.out_prefix)


if __name__ == "__main__":
    main()
