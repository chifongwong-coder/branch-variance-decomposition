"""Replot E1 figures from cached `results/e1_metrics.json`.

Splits the plotting role out of `e1_diffusion_binary_gaussian.py` so that
panel-letter tweaks, font/style updates, and caption-coupled re-renders no
longer require re-running the ~10-min diffusion simulation. Mirrors the
"data script -> JSON -> replot script" pattern used by E3
(`replot_e3_e2_from_cache.py`) and E5a (`plot_e5a_diagnostic.py`).

Inputs:  results/e1_metrics.json   (curves + argmax markers + sigma*)
Outputs: figures/e1_binary_oracle.{png,pdf}
         figures/e1_curves.{png,pdf}

Visual content (axes, ranges, colors, curves, markers) matches the
in-place plotting code at e1_diffusion_binary_gaussian.py:127-194 exactly;
only the data source differs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from core.figure_style import apply_paper_style  # noqa: E402

apply_paper_style()

JSON_PATH = HERE.parents[1] / "results" / "e1_metrics.json"
FIG_DIR = HERE.parents[1] / "figures"


def _save(fig, stem: str) -> None:
    FIG_DIR.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        out = FIG_DIR / f"{stem}.{ext}"
        fig.savefig(out, dpi=130, bbox_inches="tight")
        print(f"  wrote {out}")
    plt.close(fig)


def main() -> None:
    if not JSON_PATH.exists():
        raise FileNotFoundError(
            f"{JSON_PATH} not found; run e1_diffusion_binary_gaussian.py once "
            "to generate the cached metrics, then re-run this replot script."
        )

    d = json.load(open(JSON_PATH))
    cfg = d["config"]
    c = d["curves"]

    log_snr = np.array(c["log_snr"])
    H_bar = np.array(c["H_bar"])
    dH_dlogsnr = np.array(c["dH_dlogsnr"])
    switching_avg = np.array(c["switching_avg"])
    switching_x0 = np.array(c["switching_at_x0"])
    within = np.array(c["within"])
    abs_total_jac_avg = np.array(c["abs_total_jac_avg"])

    log_snr_star = float(d["predicted_rho_eq_1"]["log_snr_star"])
    m = cfg["m"]
    s = cfg["s"]
    n_samples = cfg["n_samples"]

    # ---------------- Figure 1: 4-panel summary ----------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    ax = axes[0, 0]
    ax.plot(log_snr, H_bar, lw=2)
    ax.axvline(log_snr_star, color="red", linestyle="--")
    ax.axhline(np.log(2.0), color="gray", linestyle=":", label=r"$\log 2$")
    ax.set_xlabel(r"$\log\mathrm{SNR}=-2\log\sigma$")
    ax.set_ylabel(r"$H_K(\sigma)$")
    ax.set_title("(a) Branch entropy")
    ax.legend(loc="best", fontsize=9)

    ax = axes[0, 1]
    ax.plot(log_snr, dH_dlogsnr, lw=2)
    ax.axvline(log_snr_star, color="red", linestyle="--")
    ax.set_xlabel(r"$\log\mathrm{SNR}$")
    ax.set_ylabel(r"$|\,dH_K/d\log\mathrm{SNR}\,|$")
    ax.set_title("(b) Entropy transition rate")

    ax = axes[1, 0]
    ax.plot(log_snr, switching_avg, lw=2, label=r"avg $r_+r_-(2m/\tau^2)^2$")
    ax.plot(log_snr, switching_x0, lw=1.2, ls="--",
            label=r"analytic peak at $x{=}0$: $m^2/\tau^4$")
    ax.axvline(log_snr_star, color="red", linestyle="--")
    ax.set_xlabel(r"$\log\mathrm{SNR}$")
    ax.set_ylabel("switching curvature")
    ax.set_title("(c) Responsibility-switching curvature")
    ax.set_yscale("symlog", linthresh=1e-3)
    ax.legend(loc="best", fontsize=9)

    ax = axes[1, 1]
    ax.plot(log_snr, switching_avg, lw=2, label="switching avg")
    ax.plot(log_snr, -within, lw=1.5, label=r"$1/\tau^2$ (within mag.)")
    ax.plot(log_snr, abs_total_jac_avg, lw=1.5,
            label=r"$|\,J_s\,|$ avg (total)")
    ax.axvline(log_snr_star, color="red", linestyle="--")
    ax.set_xlabel(r"$\log\mathrm{SNR}$")
    ax.set_ylabel("Jacobian magnitude")
    ax.set_title("(d) Jacobian decomposition")
    ax.set_yscale("log")
    ax.legend(loc="best", fontsize=9)

    fig.tight_layout()
    _save(fig, "e1_binary_oracle")

    # ---------------- Figure 2: combined normalized curves ----------------
    fig2, ax = plt.subplots(figsize=(7.5, 5))
    norm_dH = dH_dlogsnr / max(dH_dlogsnr.max(), 1e-12)
    norm_switch = switching_avg / max(switching_avg.max(), 1e-12)
    ax.plot(log_snr, H_bar / np.log(2.0), label=r"$H_K / \log 2$", lw=2)
    ax.plot(log_snr, norm_dH,
            label=r"$|dH_K/d\log\mathrm{SNR}|$ (normalized)", lw=2)
    ax.plot(log_snr, norm_switch,
            label="switching curvature (normalized)", lw=2)
    ax.axvline(log_snr_star, color="red", linestyle="--",
               label=fr"$\rho=1$ ($\log\mathrm{{SNR}}{{=}}{log_snr_star:.2f}$)")
    ax.set_xlabel(r"$\log\mathrm{SNR}=-2\log\sigma$")
    ax.set_ylabel("normalized magnitude (per curve)")
    ax.set_title("E1: critical window indicators")
    ax.legend(loc="best", fontsize=9)
    fig2.tight_layout()
    _save(fig2, "e1_curves")

    # ---------------- Combined figure (paper layout) ----------------
    # Combined layout: merge the 2x2 oracle grid + normalized indicators
    # into one figure so E1 reads as a single commitment-window result rather
    # than two standalone pages. Keeps e1_binary_oracle / e1_curves intact.
    _save_combined(log_snr, H_bar, dH_dlogsnr, switching_avg, switching_x0,
                   within, abs_total_jac_avg, log_snr_star, m, s, n_samples)


def _save_combined(log_snr, H_bar, dH_dlogsnr, switching_avg, switching_x0,
                   within, abs_total_jac_avg, log_snr_star, m, s, n_samples):
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(12, 11))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1.0, 1.0, 0.85], hspace=0.42, wspace=0.24)

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(log_snr, H_bar, lw=2, label=r"$H_K(\sigma)$")
    ax.axvline(log_snr_star, color="red", linestyle="--")
    ax.axhline(np.log(2.0), color="gray", linestyle=":", label=r"$\log 2$")
    ax.set_xlabel(r"$\log\mathrm{SNR}=-2\log\sigma$")
    ax.set_ylabel(r"$H_K(\sigma)$")
    ax.set_title("(a) Branch entropy")
    ax.legend(loc="best", fontsize=9)

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(log_snr, dH_dlogsnr, lw=2, label=r"$|\,dH_K/d\log\mathrm{SNR}\,|$")
    ax.axvline(log_snr_star, color="red", linestyle="--")
    ax.set_xlabel(r"$\log\mathrm{SNR}$")
    ax.set_ylabel(r"$|\,dH_K/d\log\mathrm{SNR}\,|$")
    ax.set_title("(b) Entropy transition rate")
    ax.legend(loc="best", fontsize=9)

    ax = fig.add_subplot(gs[1, 0])
    ax.plot(log_snr, switching_avg, lw=2, label=r"avg $r_+r_-(2m/\tau^2)^2$")
    ax.plot(log_snr, switching_x0, lw=1.2, ls="--",
            label=r"analytic peak at $x{=}0$: $m^2/\tau^4$")
    ax.axvline(log_snr_star, color="red", linestyle="--")
    ax.set_xlabel(r"$\log\mathrm{SNR}$")
    ax.set_ylabel("switching curvature")
    ax.set_title("(c) Responsibility-switching curvature")
    ax.set_yscale("symlog", linthresh=1e-3)
    ax.legend(loc="best", fontsize=9)

    ax = fig.add_subplot(gs[1, 1])
    ax.plot(log_snr, switching_avg, lw=2, label="switching avg")
    ax.plot(log_snr, -within, lw=1.5, label=r"$1/\tau^2$ (within mag.)")
    ax.plot(log_snr, abs_total_jac_avg, lw=1.5, label=r"$|\,J_s\,|$ avg (total)")
    ax.axvline(log_snr_star, color="red", linestyle="--")
    ax.set_xlabel(r"$\log\mathrm{SNR}$")
    ax.set_ylabel("Jacobian magnitude")
    ax.set_title("(d) Jacobian decomposition")
    ax.set_yscale("log")
    ax.legend(loc="best", fontsize=9)

    # bottom full-width row: normalized indicators
    ax = fig.add_subplot(gs[2, :])
    norm_dH = dH_dlogsnr / max(dH_dlogsnr.max(), 1e-12)
    norm_switch = switching_avg / max(switching_avg.max(), 1e-12)
    ax.plot(log_snr, H_bar / np.log(2.0), label=r"$H_K / \log 2$", lw=2)
    ax.plot(log_snr, norm_dH,
            label=r"$|dH_K/d\log\mathrm{SNR}|$ (normalized)", lw=2)
    ax.plot(log_snr, norm_switch, label="switching curvature (normalized)", lw=2)
    ax.axvline(log_snr_star, color="red", linestyle="--")
    ax.set_xlabel(r"$\log\mathrm{SNR}=-2\log\sigma$")
    ax.set_ylabel("normalized magnitude")
    ax.set_title("(e) Normalized critical-window indicators")
    ax.legend(loc="best", fontsize=9, ncol=1)

    _save(fig, "e1_combined")


if __name__ == "__main__":
    main()
