"""Per-mode 2D scatter of reflow endpoint distribution.

Visualizes the per-mode covariance result (endpoint-covariance figure) by plotting the actual
X_new endpoint coordinates from R1, colored by assigned mode, with 1σ
and 2σ Gaussian ellipses overlaid for both:
  - the THEORETICAL target N(mode_center, sigma^2 I)  (dashed)
  - the EMPIRICAL fitted N(mu_hat_m, Sigma_hat_m)     (solid)

This makes the "anisotropy 1.13, trace ratio 1.03" numerical summary
in the endpoint-covariance figure interpretable: if the empirical and theoretical ellipses are
nearly indistinguishable, the endpoint distribution is essentially
isotropic.

Reads X_new arrays from results/reflow_conjecture1.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.figure_style import apply_paper_style, Palette  # noqa: E402

apply_paper_style()
PAL = Palette()

R1_JSON = Path(__file__).resolve().parents[2] / "results" / "reflow_conjecture1.json"
OUT_DIR = Path(__file__).resolve().parents[2] / "figures"

MODE_COLORS = ["#3471a4", "#3aa455", "#e87a23", "#8c4ab5"]  # blue/green/orange/purple


def _cov_ellipse(mu, Sigma, n_std=1.0, **kwargs):
    """Return an Ellipse patch representing an n_std-sigma contour of
    the 2D Gaussian N(mu, Sigma)."""
    eigvals, eigvecs = np.linalg.eigh(Sigma)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    width = 2 * n_std * np.sqrt(max(eigvals[0], 0))
    height = 2 * n_std * np.sqrt(max(eigvals[1], 0))
    return Ellipse(xy=mu, width=width, height=height, angle=angle, **kwargs)


def _setup_panel(ax, MODES, sigma, max_radius=None):
    """Set up a 2D panel with mode centers and theoretical 1σ/2σ ellipses."""
    for m, mu in enumerate(MODES):
        ax.scatter(*mu, marker="+", color="black", s=120, lw=1.5, zorder=4)
        # Theoretical 1σ and 2σ ellipses (dashed black)
        for n_std in (1.0, 2.0):
            e = _cov_ellipse(mu, sigma**2 * np.eye(2), n_std=n_std,
                             fill=False, edgecolor="black", lw=0.8,
                             ls=":" if n_std == 1.0 else "--", alpha=0.7)
            ax.add_patch(e)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    if max_radius is None:
        max_radius = 3.0
    ax.set_xlim(-max_radius, max_radius)
    ax.set_ylim(-max_radius, max_radius)


def main():
    with open(R1_JSON) as f:
        d = json.load(f)
    seeds = d["seeds"]
    # Recover MODES + sigma from S_WIDTH (per-seed, all the same)
    S_WIDTH = float(seeds[0]["endpoint_stats"]["S_WIDTH"])
    # MODES is hardcoded in e3b_branch_refinement.py as the 4 corners at ±2
    MODES = np.array([[2.0, 2.0], [-2.0, -2.0], [-2.0, 2.0], [2.0, -2.0]])

    # Per-seed panel: 5 rows
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for si, s in enumerate(seeds):
        ax = axes[si]
        X_new = np.asarray(s["endpoint_stats"]["X_new"])
        mode_idx = np.asarray(s["endpoint_stats"]["mode_idx"])

        _setup_panel(ax, MODES, S_WIDTH)
        # subsample for plotting (20k points overwhelms scatter)
        n_plot = min(2000, len(X_new))
        rng = np.random.default_rng(0)
        sel = rng.choice(len(X_new), size=n_plot, replace=False)

        for m in range(4):
            mask = (mode_idx == m) & np.isin(np.arange(len(X_new)), sel)
            if mask.any():
                ax.scatter(X_new[mask, 0], X_new[mask, 1],
                           s=4, color=MODE_COLORS[m], alpha=0.45,
                           edgecolor="none",
                           label=f"mode {m}" if si == 0 else None)
            # Empirical fitted ellipse per mode (solid colored)
            sel_full = mode_idx == m
            if sel_full.sum() >= 3:
                Xm = X_new[sel_full]
                hat_mu = Xm.mean(axis=0)
                hat_Sigma = np.cov(Xm.T, ddof=1)
                for n_std, lw, alpha in ((1.0, 1.2, 0.85),
                                          (2.0, 1.0, 0.55)):
                    e = _cov_ellipse(hat_mu, hat_Sigma, n_std=n_std,
                                     fill=False,
                                     edgecolor=MODE_COLORS[m], lw=lw,
                                     alpha=alpha)
                    ax.add_patch(e)

        # Mode props + covariance summary in corner
        cov_sum = s["endpoint_stats"]["covariance_summary"]
        ax.text(0.02, 0.98,
                fr"seed {s['train_seed']}" + "\n"
                fr"max aniso = {cov_sum['max_anisotropy']:.2f}" + "\n"
                fr"max tr-dev = {cov_sum['max_trace_ratio_dev']:.3f}",
                transform=ax.transAxes, va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3",
                         facecolor="white", edgecolor="gray", lw=0.5))
        ax.set_title(fr"seed {s['train_seed']}: $X_{{\rm new}}$ subsample "
                     fr"(n={n_plot} of {len(X_new)})", fontsize=9)
        if si == 0:
            ax.legend(loc="lower right", fontsize=7, ncol=2)

    # Hide unused panel (we have 5 seeds, 6 axes)
    axes[5].axis("off")

    # Legend for the empirical vs theoretical ellipses in unused panel area
    legend_ax = axes[5]
    legend_lines = [
        plt.Line2D([], [], marker="+", color="black", ms=10, lw=0,
                   label="mode center $\\mu_m$"),
        plt.Line2D([], [], color="black", lw=0.8, ls=":",
                   label=r"theoretical $\sigma$ ellipse"),
        plt.Line2D([], [], color="black", lw=0.8, ls="--",
                   label=r"theoretical $2\sigma$ ellipse"),
        plt.Line2D([], [], color="gray", lw=1.2, ls="-",
                   label=r"empirical $\sigma$ ellipse"),
        plt.Line2D([], [], color="gray", lw=1.0, ls="-", alpha=0.55,
                   label=r"empirical $2\sigma$ ellipse"),
    ]
    legend_ax.legend(handles=legend_lines, loc="center", fontsize=9,
                     frameon=True)
    legend_ax.set_title("(legend)", fontsize=9)

    fig.suptitle(
        r"Reflow endpoint per-mode 2D distribution (5 seeds, sub-sampled); "
        r"empirical (solid) vs theoretical (dashed) ellipses",
        fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = OUT_DIR / "reflow_per_mode_scatter.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
