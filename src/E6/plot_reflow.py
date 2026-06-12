"""Primary plots for the reflow locality (Conjecture 1) result.

Reads `results/reflow_conjecture1.json` and produces:

  Per-t 4-panel profile (A_v, A_within, A_between, R_switch) for
         {C0, reflow, C1}; rows = {coarse K, fine K'}; 5-seed mean +/- SD.

  Locality scatter: Delta A_between(t) vs R_switch^{C0}(t) over
         the 19-point t-grid; 95 dots per panel (19 t x 5 seeds);
         panels = {coarse, fine}; Pearson + Spearman annotations.

  Per-t share curve  Delta A_b(t) / Delta A_v(t)  for {coarse, fine};
         highlights the top-25% R_switch^{C0} band (interior-filtered);
         5-seed mean +/- SD bands; reference line at 0.5.

All three visualize the decomposition profile, locality, and the
in-band share. None of these depend on the endpoint-purity discussion.

Outputs PNGs to figures/.
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
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.figure_style import apply_paper_style, Palette  # noqa: E402

apply_paper_style()
PAL = Palette()

LABEL_COLOR = {
    "C0":     PAL.C0,    # grey baseline
    "reflow": PAL.ref,   # red highlight
    "C1":     PAL.C1,    # blue baseline
}
LABEL_NICE = {
    "C0":     r"$C_0$ (held-out independent)",
    "reflow": r"reflow (ODE-pushed $Z_{\rm new}$)",
    "C1":     r"$C_1$ (Euclidean OT baseline)",
}
LABEL_TYPE_NICE = {
    "coarse": r"coarse $K$ (XOR superclass)",
    "fine":   r"fine $K'$ (mode index $\{0,1,2,3\}$)",
}


def _load(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _stack_per_t(seeds: list[dict], pipeline_key: str,
                 label_type: str, metric: str) -> np.ndarray:
    """Return (n_seeds, n_t) array of `metric` over the 19-point t-grid
    for the given pipeline ("dec_c0" / "dec_reflow" / "dec_c1") and label
    type ("coarse" / "fine")."""
    arr = np.array([
        s[pipeline_key][label_type][metric] for s in seeds
    ], dtype=np.float64)
    return arr


# ---------------------------------------------------------------------------
# per-t 4-panel profile (rows = label type, cols = metric)
# ---------------------------------------------------------------------------

def plot_profile(data: dict, out_dir: Path, prefix: str):
    """Per-t profile with symlog y-axis on the A_v/A_w/A_b panels
    (linthresh = 0.1) so the C0 magnitude (~0.5-10) AND the reflow/C1
    detail (~0.005-0.03) are both visible in one plot. R_switch panel
    stays linear since it's bounded in [0, 1]."""
    seeds = data["seeds"]
    t_grid = np.array(data["config"]["T_GRID"])

    metrics = [("A_v",       r"$A_v(t)$",       True),   # use symlog
               ("A_within",  r"$A_{\rm within}(t)$", True),
               ("A_between", r"$A_{\rm between}(t)$", True),
               ("R_switch",  r"$R_{\rm switch}(t)$",  False)]
    label_types = ["coarse", "fine"]
    pipelines = [("dec_c0", "C0"), ("dec_reflow", "reflow"), ("dec_c1", "C1")]

    apply_paper_style()
    plt.rcParams.update({"axes.labelsize": 18, "xtick.labelsize": 18,
                         "ytick.labelsize": 18, "axes.titlesize": 18})
    fig, axes = plt.subplots(len(label_types), len(metrics),
                             figsize=(15, 7.5), sharex=True, squeeze=False)

    for ri, lab_type in enumerate(label_types):
        for ci, (metric, metric_latex, use_symlog) in enumerate(metrics):
            ax = axes[ri, ci]
            for pipe_key, pipe_name in pipelines:
                arr = _stack_per_t(seeds, pipe_key, lab_type, metric)
                mean = arr.mean(axis=0)
                sd = arr.std(axis=0, ddof=1)
                color = LABEL_COLOR[pipe_name]
                ax.plot(t_grid, mean, lw=1.5, color=color,
                        label=LABEL_NICE[pipe_name] if (ri == 0 and ci == 0) else None,
                        marker="o", ms=3)
                ax.fill_between(t_grid, mean - sd, mean + sd,
                                color=color, alpha=0.18, edgecolor="none")
            ax.axvline(0.5, color="gray", ls=":", lw=0.6, alpha=0.5)
            if use_symlog:
                ax.set_yscale("symlog", linthresh=0.1, linscale=0.5)
                ax.set_ylim(-0.02, ax.get_ylim()[1])  # cut off slightly below 0
                # subtle gridline at the linthresh boundary
                ax.axhline(0.1, color="gray", ls=":", lw=0.5, alpha=0.4)
            if ri == 1:
                ax.set_xlabel(r"$t$")
            if ci == 0:
                ax.set_ylabel(LABEL_TYPE_NICE[lab_type] + "\n" + metric_latex,
                              fontsize=12)
            else:
                ax.set_ylabel(metric_latex)
            if ri == 0:
                title_fs = 12 if ci <= 1 else 18
                ax.set_title(metric_latex, fontsize=title_fs)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               bbox_to_anchor=(0.5, -0.06),
               ncol=3, fontsize=13, frameon=False)

    fig.tight_layout()
    out = out_dir / f"{prefix}_profile.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
# locality scatter (Delta A_b vs R_sw^{C0})
# ---------------------------------------------------------------------------

def plot_locality(data: dict, out_dir: Path, prefix: str):
    seeds = data["seeds"]
    t_grid = np.array(data["config"]["T_GRID"])
    n_t = len(t_grid)
    interior = (t_grid >= 0.1) & (t_grid <= 0.9)

    label_types = ["coarse", "fine"]
    apply_paper_style()
    plt.rcParams.update({"axes.labelsize": 15, "xtick.labelsize": 15,
                         "ytick.labelsize": 15, "axes.titlesize": 15})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.0), sharey=False)

    for ax, lab in zip(axes, label_types):
        rsw_all = []
        d_ab_all = []
        rsw_int = []
        d_ab_int = []
        for s in seeds:
            rsw_c0 = np.array(s["dec_c0"][lab]["R_switch"])
            ab_c0  = np.array(s["dec_c0"][lab]["A_between"])
            ab_rf  = np.array(s["dec_reflow"][lab]["A_between"])
            d_ab = ab_c0 - ab_rf
            rsw_all.extend(rsw_c0.tolist())
            d_ab_all.extend(d_ab.tolist())
            rsw_int.extend(rsw_c0[interior].tolist())
            d_ab_int.extend(d_ab[interior].tolist())
        rsw_all = np.array(rsw_all)
        d_ab_all = np.array(d_ab_all)
        rsw_int = np.array(rsw_int)
        d_ab_int = np.array(d_ab_int)

        # All points (light) and interior (full opacity)
        ax.scatter(rsw_all, d_ab_all, s=24, color="tab:blue",
                   alpha=0.55, edgecolor="none",
                   label="full t-grid")
        ax.scatter(rsw_int, d_ab_int, s=32, color=LABEL_COLOR["reflow"],
                   alpha=0.85, edgecolor="black", linewidth=0.4,
                   label=r"interior $t\in[0.1,0.9]$")

        # Linear fit on interior
        if len(rsw_int) >= 3:
            slope, intercept = np.polyfit(rsw_int, d_ab_int, 1)
            xs = np.linspace(rsw_int.min(), rsw_int.max(), 50)
            ax.plot(xs, slope * xs + intercept, ls="--", color="black",
                    lw=1.0, alpha=0.7,
                    label=fr"linear fit: slope $= {slope:+.2f}$")
            # Annotate the per-seed-mean correlations (each seed's own value from
            # the runner, averaged over seeds), matching the per-t table. The
            # pooled scatter and fit above are only the visualization; a Spearman
            # recomputed on the pooled points would differ from the reported mean.
            pear = float(np.mean([sd[f"b5_{lab}"]["correlations"]["pearson_Rsw_DeltaAb"]
                                  for sd in seeds]))
            sp = float(np.mean([sd[f"b5_{lab}"]["correlations"]["spearman_Rsw_DeltaAb"]
                                for sd in seeds]))
            stat_txt = (fr"Pearson $\rho = {pear:+.3f}$" + "\n"
                        fr"Spearman $= {sp:+.3f}$")
            ax.text(0.04, 0.96, stat_txt, transform=ax.transAxes,
                    fontsize=9, va="top",
                    bbox=dict(boxstyle="round,pad=0.3",
                             facecolor="white", edgecolor="gray", lw=0.5))
        ax.axhline(0, color="gray", ls=":", lw=0.6, alpha=0.5)
        ax.set_xlabel(r"$R_{\rm switch}^{C_0}(t)$")
        ax.set_ylabel(r"$\Delta A_{\rm between}(t)$")
        ax.set_title(f"({'a' if lab == 'coarse' else 'b'}) "
                     + LABEL_TYPE_NICE[lab])
        leg_loc = "center left" if lab == "fine" else "lower right"
        ax.legend(loc=leg_loc, fontsize=9)

    fig.tight_layout()
    out = out_dir / f"{prefix}_locality.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
# per-t share curve
# ---------------------------------------------------------------------------

def plot_endpoint_covariance(data: dict, out_dir: Path, prefix: str):
    """Per-(seed, mode) endpoint covariance diagnostics.

    4 panels: anisotropy, trace ratio, center offset, W2 to target.
    Each panel: 20 dots = 5 seeds * 4 modes, colored by seed and grouped
    by mode on the x-axis. Reference bands for expert's Good/Acceptable
    grading shaded.
    """
    seeds = data["seeds"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.0))

    metrics = [
        ("anisotropy",    r"anisotropy $\lambda_{\max}/\lambda_{\min}$",
         [(1.0, 1.2, "Good ($<1.2$)"),
          (1.2, 1.5, "Acceptable ($<1.5$)")],
         (0.95, 1.55)),
        ("trace_ratio",   r"trace ratio $\mathrm{tr}\widehat\Sigma_m / (2\sigma^2)$",
         [(0.9, 1.1, "Good ($[0.9, 1.1]$)"),
          (0.8, 1.2, "Acceptable ($[0.8, 1.2]$)")],
         (0.75, 1.25)),
        ("center_offset", r"center offset $\|\widehat\mu_m - \mu_m\|$",
         [(0.0, 0.05, "Good ($<0.05$)"),
          (0.0, 0.10, "Acceptable ($<0.10$)")],
         (-0.005, 0.12)),
        ("W2_to_target",  r"Gaussian $W_2$ to $N(\mu_m, \sigma^2 I)$",
         [],  # no formal band
         (-0.005, 0.10)),
    ]

    seed_cmap = plt.get_cmap("tab10")
    mode_x_jitter = {0: -0.18, 1: -0.06, 2: 0.06, 3: 0.18}

    for i, (ax, (metric_key, metric_label, bands, ylim)) in enumerate(zip(axes.flat, metrics)):
        # Shade Good/Acceptable bands behind the data
        for lo, hi, lbl in bands:
            if metric_key in ("anisotropy", "trace_ratio"):
                ax.axhspan(lo, hi, color=PAL.pos, alpha=0.06,
                           label=lbl if lo == bands[0][0] else None)
                if metric_key == "trace_ratio":
                    ax.axhline(1.0, color="gray", ls=":", lw=0.7, alpha=0.6)
                elif metric_key == "anisotropy":
                    ax.axhline(1.0, color="gray", ls=":", lw=0.7, alpha=0.6)
            else:
                ax.axhspan(lo, hi, color=PAL.pos, alpha=0.06,
                           label=lbl if lo == bands[0][0] else None)
            # Show the Good band edge explicitly
        # For trace_ratio + anisotropy: draw Good and Acceptable boundaries
        if bands:
            # The bands list is [(good_lo, good_hi, good_lbl), (acc_lo, acc_hi, acc_lbl)]
            # Draw boundary lines
            good_lo, good_hi, _ = bands[0]
            acc_lo, acc_hi, _ = bands[1]
            for v, lbl, ls in ((good_hi, "Good", "--"),
                               (acc_hi, "Acceptable", ":")):
                if v != good_lo and v != acc_lo:
                    ax.axhline(v, color=PAL.ref, ls=ls, lw=0.8, alpha=0.6,
                               label=fr"$\leq {v}$ ({lbl})")
            # For 2-sided bands (trace_ratio), also draw lower
            if metric_key == "trace_ratio":
                ax.axhline(good_lo, color=PAL.ref, ls="--", lw=0.8,
                           alpha=0.6)
                ax.axhline(acc_lo, color=PAL.ref, ls=":", lw=0.8,
                           alpha=0.6)

        # Scatter: x = mode_idx + per-seed jitter; y = metric value
        for si, s in enumerate(seeds):
            pm = s["endpoint_stats"]["per_mode_covariance"]
            for m in range(4):
                xs_jit = m + mode_x_jitter[si % 4]  # cycle if > 4 seeds
                # Actually use unique jitter per seed
            # Use per-seed offsets
            offsets = np.linspace(-0.18, 0.18, len(seeds))
            for m in range(4):
                v = pm[str(m)][metric_key]
                ax.scatter(m + offsets[si], v, s=55,
                           color=seed_cmap(si % 10),
                           edgecolor="black", linewidth=0.4,
                           alpha=0.92,
                           label=fr"seed {s['train_seed']}"
                                 if (m == 0 and ax is axes[0, 0]) else None)
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xticklabels([fr"mode 0", fr"mode 1", fr"mode 2", fr"mode 3"])
        ax.set_ylabel(metric_label)
        ax.set_title(f"({chr(97 + i)})", loc="left", fontweight="bold")
        ax.set_ylim(ylim)
        # Mean ± SD annotation
        all_vals = np.array([
            s["endpoint_stats"]["per_mode_covariance"][str(m)][metric_key]
            for s in seeds for m in range(4)
        ])
        ax.text(0.04, 0.95,
                fr"per-(seed, mode): mean = {all_vals.mean():.3f}, "
                fr"SD = {all_vals.std(ddof=1):.3f}",
                transform=ax.transAxes, fontsize=8, va="top",
                bbox=dict(boxstyle="round,pad=0.3",
                         facecolor="white", edgecolor="gray", lw=0.5))

    # Legend (consolidated): only show on top-left panel
    handles, labels = axes[0, 0].get_legend_handles_labels()
    seen = set()
    h2, l2 = [], []
    for h, l in zip(handles, labels):
        if l not in seen and l is not None:
            seen.add(l); h2.append(h); l2.append(l)
    axes[0, 0].legend(h2, l2, loc="upper right", fontsize=6.5, ncol=2)

    fig.suptitle(
        r"Per-(seed, mode) endpoint covariance vs target $N(\mu_m, \sigma^2 I)$"
        r"  (5 paired seeds, 4 modes each)",
        fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = out_dir / f"{prefix}_endpoint_covariance.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def plot_endpoint_calibration(data: dict, out_dir: Path, prefix: str):
    """Endpoint radial-CDF + mode-proportions calibration vs 2D
    Gaussian target. The headline
    endpoint diagnostic is "does the reflow endpoint match the target
    mode width" (radial CDF) and "is the endpoint distribution balanced
    across modes" (proportions). Both are calibration plots, not
    pass/fail gates.

    Left panel:  empirical CDF P(R<=r) from reflow endpoint distances,
                 per-seed light curves + 5-seed mean, vs theoretical
                 2D Gaussian CDF 1 - exp(-r^2 / 2 / sigma^2).
    Right panel: mode proportions bar chart (fine modes 0/1/2/3 + coarse
                 +/-) with uniform-target reference lines.
    """
    seeds = data["seeds"]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.0))

    # ---- Left: radial CDF ----
    S_WIDTH = float(seeds[0]["endpoint_stats"]["S_WIDTH"])
    all_dists_per_seed = [np.asarray(s["endpoint_stats"]["distances"])
                          for s in seeds]
    # Compose a fine r-grid for plotting
    r_max_sigma = 4.0
    r_grid_sigma = np.linspace(0.0, r_max_sigma, 200)
    r_grid_raw = r_grid_sigma * S_WIDTH

    # Theoretical 2D Gaussian CDF
    cdf_theory = 1.0 - np.exp(-(r_grid_sigma ** 2) / 2.0)
    axL.plot(r_grid_sigma, cdf_theory, ls="--", color="black", lw=1.4,
             label=r"theory: $1 - e^{-r^2/2}$ (2D Gaussian)",
             zorder=3)

    # Per-seed empirical CDFs (light)
    cdf_per_seed = np.zeros((len(seeds), len(r_grid_raw)))
    for si, dists in enumerate(all_dists_per_seed):
        sorted_d = np.sort(dists)
        cdf_per_seed[si] = np.searchsorted(sorted_d, r_grid_raw,
                                            side="right") / len(sorted_d)
        axL.plot(r_grid_sigma, cdf_per_seed[si], color=LABEL_COLOR["reflow"],
                 alpha=0.3, lw=0.7,
                 label="reflow per-seed" if si == 0 else None)
    # Mean across seeds
    cdf_mean = cdf_per_seed.mean(axis=0)
    cdf_sd = cdf_per_seed.std(axis=0, ddof=1)
    axL.plot(r_grid_sigma, cdf_mean, color=LABEL_COLOR["reflow"], lw=1.8,
             label="reflow 5-seed mean", zorder=2)
    axL.fill_between(r_grid_sigma, cdf_mean - cdf_sd, cdf_mean + cdf_sd,
                     color=LABEL_COLOR["reflow"], alpha=0.18,
                     edgecolor="none")

    # Annotate the 4 canonical radii
    for k in (1.0, 1.5, 2.0, 3.0):
        thy = 1.0 - np.exp(-(k ** 2) / 2.0)
        obs = float((all_dists_per_seed[0] <= k * S_WIDTH).mean())  # seed-0 just for label
        axL.axvline(k, color="gray", ls=":", lw=0.5, alpha=0.5)
        axL.text(k, -0.06, fr"${k:.1f}\sigma$", ha="center", va="top",
                 fontsize=8, color="gray", transform=axL.get_xaxis_transform())
    axL.set_xlabel(r"$r / \sigma_{\rm mode}$ (mode width $\sigma = " + f"{S_WIDTH:g}$)")
    axL.set_ylabel(r"$P(R \leq r)$  ($R = $ distance to nearest mode)")
    axL.set_xlim(0, r_max_sigma)
    axL.set_ylim(-0.02, 1.05)
    axL.set_title(r"(a) Reflow endpoint radial CDF vs 2D Gaussian target")

    # Inset: per-radius observed vs theoretical with seed SD bars
    ks = [1.0, 1.5, 2.0, 3.0]
    obs_means = []
    obs_sds = []
    thys = []
    for k in ks:
        per_seed = np.array([float((d <= k * S_WIDTH).mean())
                             for d in all_dists_per_seed])
        obs_means.append(per_seed.mean())
        obs_sds.append(per_seed.std(ddof=1))
        thys.append(1.0 - np.exp(-(k ** 2) / 2.0))
    # Place values as a small annotated table inside the axes
    txt_lines = [r"$r$ | obs (mean$\pm$SD) | theory | dev"]
    for k, om, os_, th in zip(ks, obs_means, obs_sds, thys):
        txt_lines.append(
            fr"${k:.1f}\sigma$ | ${om:.3f}\pm{os_:.3f}$ | ${th:.3f}$ | ${om-th:+.3f}$")
    axL.text(0.04, 0.95, "\n".join(txt_lines),
             transform=axL.transAxes, fontsize=7.5, va="top", ha="left",
             family="monospace",
             bbox=dict(boxstyle="round,pad=0.4",
                      facecolor="white", edgecolor="gray", lw=0.5))
    axL.legend(loc="lower right", fontsize=8)

    # ---- Right: mode proportions ----
    fine_props_per_seed = np.array([
        [s["endpoint_stats"]["mode_proportions_fine"][str(m)] for m in range(4)]
        for s in seeds
    ])  # (n_seeds, 4)
    coarse_props_per_seed = np.array([
        [s["endpoint_stats"]["mode_proportions_coarse"]["-1"],
         s["endpoint_stats"]["mode_proportions_coarse"]["1"]]
        for s in seeds
    ])  # (n_seeds, 2)

    # Bar chart for fine modes
    x_fine = np.arange(4)
    means_fine = fine_props_per_seed.mean(axis=0)
    sds_fine = fine_props_per_seed.std(axis=0, ddof=1)
    bars_fine = axR.bar(x_fine, means_fine, yerr=sds_fine, capsize=4,
                        color=LABEL_COLOR["reflow"], alpha=0.85,
                        edgecolor="black", linewidth=0.5,
                        label=r"fine mode (target $= 0.25$)")
    axR.axhline(0.25, color=LABEL_COLOR["reflow"], ls="--", lw=1.0,
                alpha=0.7)

    # Bar chart for coarse classes (offset to the right)
    x_coarse = np.array([5.5, 6.5])
    means_coarse = coarse_props_per_seed.mean(axis=0)
    sds_coarse = coarse_props_per_seed.std(axis=0, ddof=1)
    axR.bar(x_coarse, means_coarse, yerr=sds_coarse, capsize=4,
            color=PAL.C1, alpha=0.85, edgecolor="black", linewidth=0.5,
            label=r"coarse class (target $= 0.5$)")
    axR.axhline(0.5, color=PAL.C1, ls="--", lw=1.0, alpha=0.7,
                xmin=0.65, xmax=0.97)

    axR.set_xticks(list(x_fine) + list(x_coarse))
    axR.set_xticklabels([r"mode 0", r"mode 1", r"mode 2", r"mode 3",
                         r"$K=-1$", r"$K=+1$"], fontsize=9)
    axR.set_ylabel("endpoint proportion (5-seed mean $\\pm$ SD)")
    axR.set_ylim(0, 0.6)
    axR.set_title(r"(b) Reflow endpoint mode proportions")
    axR.legend(loc="upper right", fontsize=8)

    # Mode-balance deviation annotation
    max_dev_per_seed = np.array([
        s["endpoint_stats"]["mode_max_dev_from_uniform"] for s in seeds
    ])
    axR.text(0.02, 0.95,
             fr"max $|p_m - 0.25|$ (across modes): "
             fr"${max_dev_per_seed.mean():.3f} \pm {max_dev_per_seed.std(ddof=1):.3f}$",
             transform=axR.transAxes, fontsize=8,
             bbox=dict(boxstyle="round,pad=0.3",
                      facecolor="white", edgecolor="gray", lw=0.5))

    fig.suptitle(
        r"Reflow endpoint calibration vs 2D Gaussian target, 5 paired seeds",
        fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = out_dir / f"{prefix}_endpoint_calibration.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def plot_share_per_t(data: dict, out_dir: Path, prefix: str):
    seeds = data["seeds"]
    t_grid = np.array(data["config"]["T_GRID"])
    interior = (t_grid >= 0.1) & (t_grid <= 0.9)

    label_types = ["coarse", "fine"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.0), sharey=True)

    for ax, lab in zip(axes, label_types):
        # Stack per-t share across seeds
        share_per_seed = []
        for s in seeds:
            share_per_seed.append(np.array(s[f"b5_{lab}"]["share_per_t"]))
        share = np.stack(share_per_seed, axis=0)  # (n_seeds, n_t)
        # Mean + SD ignoring NaNs (where ΔA_v <= 0)
        mean = np.nanmean(share, axis=0)
        sd = np.nanstd(share, axis=0, ddof=1)

        # Highlight top-25% R_sw_c0 band (interior-filtered, averaged across seeds)
        rsw_c0_per_seed = np.array([s["dec_c0"][lab]["R_switch"] for s in seeds])
        rsw_mean = rsw_c0_per_seed.mean(axis=0)
        rsw_int_only = np.where(interior, rsw_mean, -np.inf)
        if interior.any():
            q25 = np.quantile(rsw_mean[interior], 0.75)
            band_mask = (rsw_mean >= q25) & interior
            # Find contiguous t ranges
            in_band = False
            band_ranges = []
            for ti in range(len(t_grid)):
                if band_mask[ti] and not in_band:
                    t_lo = t_grid[ti] - 0.025 if ti > 0 else t_grid[ti]
                    in_band = True
                elif not band_mask[ti] and in_band:
                    t_hi = t_grid[ti - 1] + 0.025
                    band_ranges.append((t_lo, t_hi))
                    in_band = False
            if in_band:
                band_ranges.append((t_lo, t_grid[-1] + 0.025))
            for t_lo, t_hi in band_ranges:
                ax.axvspan(t_lo, t_hi, color=PAL.pos, alpha=0.12,
                           label=r"top-25\% $R_{\rm switch}^{C_0}$ band"
                                 if (lab == "coarse" and t_lo == band_ranges[0][0])
                                 else None)

        # Plot share curve
        ax.plot(t_grid, mean, lw=1.5, color=LABEL_COLOR["reflow"],
                marker="o", ms=4,
                label="per-$t$ share (mean over 5 seeds)")
        ax.fill_between(t_grid, mean - sd, mean + sd,
                        color=LABEL_COLOR["reflow"], alpha=0.18,
                        edgecolor="none")
        ax.axhline(0.5, color="gray", ls="--", lw=0.8, alpha=0.7,
                   label=r"reference: share $= 0.5$")

        # Saturation reference (sat_ref_interior)
        sat_refs = [s[f"b5_{lab}"]["per_band"]["interior"]["saturation_share_ref"]
                    for s in seeds]
        sat_ref_mean = float(np.mean(sat_refs))
        ax.axhline(sat_ref_mean, color="black", ls=":", lw=0.6, alpha=0.7,
                   label=fr"$s^{{C_0}}_{{\rm interior}} = {sat_ref_mean:.2f}$")

        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"share $(t) = \Delta A_{\rm between}(t) / \Delta A_v(t)$")
        ax.set_title(f"({'a' if lab == 'coarse' else 'b'}) "
                     + LABEL_TYPE_NICE[lab])
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc="lower center", fontsize=8)

    fig.suptitle(
        r"Per-$t$ share curve, 5 paired seeds. Highlighted band = "
        r"top-25\% by $R_{\rm switch}^{C_0}$",
        fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = out_dir / f"{prefix}_share_per_t.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parents[2]
    parser.add_argument("--input", type=str,
                        default=str(here / "results"
                                   / "reflow_conjecture1.json"))
    parser.add_argument("--out-prefix", type=str, default="reflow")
    args = parser.parse_args()

    out_dir = here / "figures"
    out_dir.mkdir(exist_ok=True)

    print(f"reading {args.input}")
    data = _load(Path(args.input))
    print(f"  stage={data['stage']}  n_seeds={len(data['seeds'])}  "
          f"wall_s={data['wall_s']:.1f}")

    plot_profile(data, out_dir, args.out_prefix)
    plot_locality(data, out_dir, args.out_prefix)
    plot_share_per_t(data, out_dir, args.out_prefix)
    plot_endpoint_calibration(data, out_dir, args.out_prefix)
    plot_endpoint_covariance(data, out_dir, args.out_prefix)


if __name__ == "__main__":
    main()
