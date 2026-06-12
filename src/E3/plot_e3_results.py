"""Post-processing: generate paper-ready CI figures from E3 JSON.

Reads results/e3_metrics_<TAG>.json and produces:
  - e3_main_<TAG>_ci.png    : 4-panel summary (bar charts + Pareto) with
                               mean +/- std error bars across seeds
  - e3_curves_<TAG>_ci.png  : 4-panel t-curves with shaded CI bands

Multi-seed input is handled gracefully: with 1 seed all error bars are zero;
with >= 2 seeds the std is computed with ddof=1.

Run:
  E3C_TAG=phase1_pilot python3 plot_e3_results.py
  E3C_TAG=phase2 E3C_CI=stderr python3 plot_e3_results.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.figure_style import apply_paper_style, Palette, coupling_color
apply_paper_style()

# Paper-canonical plot config; change in-source if running a different tag.
# TAG points at the headline 10-seed run; OUTPUT_STEM is the paper-included
# filename (used directly so the paper does not need to be re-pointed every
# time the input JSON name changes).
TAG = "phase2_10seeds"
OUTPUT_STEM = "e3_main"       # produces e3_main_ci.png + .pdf
CURVES_STEM = "e3_curves"     # produces e3_curves_ci.png + .pdf
CI_TYPE = "std"       # error bars are sample SD over seeds
SUFFIX = "_ci"        # historical filename suffix; kept for back-compat
K_MAIN = 80           # headline kNN bandwidth

HERE = Path(__file__).resolve().parent
RES_DIR = HERE.parents[1] / "results"
FIG_DIR = HERE.parents[1] / "figures"
EPS = 1e-12

# ----------------------------------------------------------------------------
# Headline-only colorblind palette + math-symbol display labels
# (overrides figure_style.coupling_color for this script only; other figures
# in the paper continue to use the project-wide palette).
# ----------------------------------------------------------------------------

# Math-symbol display labels for tick / legend (replace 'C0_independent' etc.).
DISPLAY_LABEL = {
    "C0_independent":            r"$C_0$",
    "C1_euclidean_ot":           r"$C_1$",
    "C2_condition_aware_random": r"$C_2$",
    "C3_semOT_lam1":             r"$C_3^{1}$",
    "C3_semOT_lam3":             r"$C_3^{3}$",
    "C3_semOT_lam10":            r"$C_3^{10}$",
    "C3_semOT_lam30":            r"$C_3^{30}$",
    "C3_semOT_lam100":           r"$C_3^{100}$",
    "C3inf_blocked_ot":          r"$C_3^{\infty}$",
    "C4_geometry_only_ot":       r"$C_4$",
}

# Okabe-Ito colorblind-safe palette assignments. Non-C3 couplings get distinct
# categorical hues; the C3 family gets a sequential blue gradient that also
# encodes the lambda ordering. Combined with distinct markers (below) the
# encoding is double (color + shape) for accessibility.
_C3_BLUE_GRADIENT = [  # light -> dark, 6 steps for lam in {1,3,10,30,100,inf}
    "#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#08519c", "#08306b",
]

CB_COLOR = {
    "C0_independent":            "#000000",   # black
    "C1_euclidean_ot":           "#D55E00",   # vermillion
    "C2_condition_aware_random": "#E69F00",   # orange
    "C4_geometry_only_ot":       "#CC79A7",   # reddish purple
    # C3 entries are filled in via _build_color_map below, using the gradient
    # so that the lambda-ordering is also visually monotone.
}

# Distinct markers per coupling for the Pareto panel and any per-point plot;
# gives a second visual axis on top of color, useful for colorblind readers
# and for B/W print.
CB_MARKER = {
    "C0_independent":            "o",   # circle
    "C1_euclidean_ot":           "s",   # square
    "C2_condition_aware_random": "D",   # diamond
    "C3_semOT_lam1":             "v",   # down triangle
    "C3_semOT_lam3":             "^",   # up triangle
    "C3_semOT_lam10":            "<",   # left triangle
    "C3_semOT_lam30":            ">",   # right triangle
    "C3_semOT_lam100":           "p",   # pentagon
    "C3inf_blocked_ot":          "*",   # star
    "C4_geometry_only_ot":       "h",   # hexagon
}


def _display(name):
    """Return math-symbol display label, with raw name fallback."""
    return DISPLAY_LABEL.get(name, name)


def _marker(name):
    return CB_MARKER.get(name, "o")


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------

def load_results(tag):
    p = RES_DIR / f"e3_metrics_{tag}.json"
    with open(p) as f:
        return json.load(f)


def runs_by_name(runs_dict):
    by_name = {}
    for key, r in runs_dict.items():
        by_name.setdefault(r["name"], []).append(r)
    return by_name


def get_per_k(tm_block, k):
    """tm_block is a dict like tm['uncond'] or tm['cond_avg'].
    After JSON round-trip, integer keys become strings. Try both."""
    if str(k) in tm_block:
        return tm_block[str(k)]
    if k in tm_block:
        return tm_block[k]
    raise KeyError(f"k={k} not in {list(tm_block.keys())[:5]}")


def metric_over_t(run, kind, k, subspace, mk):
    """kind in {'uncond', 'cond_avg'}; subspace in {'full','geom','sem'}."""
    out = []
    for tm in run["metrics_per_t"]:
        if kind == "uncond":
            out.append(get_per_k(tm["uncond"], k)[subspace][mk])
        elif kind == "cond_avg":
            cd = tm.get("cond_avg", {})
            if not cd:
                out.append(np.nan)
            else:
                try:
                    out.append(get_per_k(cd, k)[subspace][mk])
                except KeyError:
                    out.append(np.nan)
        else:
            raise ValueError(f"unknown kind {kind}")
    return np.array(out)


def ci_from_stack(stack, ci_type=CI_TYPE):
    """stack shape (n_seeds, ...).  Returns (mean, half_width) for error bar.
    With n_seeds=1 returns zero half-width."""
    mean = np.nanmean(stack, axis=0)
    n = stack.shape[0]
    if n < 2:
        return mean, np.zeros_like(mean)
    if ci_type == "std":
        return mean, np.nanstd(stack, axis=0, ddof=1)
    if ci_type == "stderr":
        return mean, np.nanstd(stack, axis=0, ddof=1) / np.sqrt(n)
    raise ValueError(f"unknown ci_type {ci_type}")


# ----------------------------------------------------------------------------
# Aggregators: produce mean / err arrays across seeds
# ----------------------------------------------------------------------------

def agg_pair_metric(by_name, names, key):
    means, errs = [], []
    for n in names:
        vals = np.array([r["pair_metrics"][key] for r in by_name[n]])
        m, e = ci_from_stack(vals[:, None])  # shape (n_seeds, 1)
        means.append(float(m[0])); errs.append(float(e[0]))
    return np.array(means), np.array(errs)


def agg_max_t(by_name, names, kind, k, subspace, mk, scale=1.0):
    means, errs = [], []
    for n in names:
        per_seed_max = []
        for r in by_name[n]:
            arr = metric_over_t(r, kind, k, subspace, mk) * scale
            per_seed_max.append(np.nanmax(arr))
        per_seed_max = np.array(per_seed_max)
        m, e = ci_from_stack(per_seed_max[:, None])
        means.append(float(m[0])); errs.append(float(e[0]))
    return np.array(means), np.array(errs)


def agg_curve_t(by_name, names, kind, k, subspace, mk, scale=1.0):
    """Returns dict name -> (t_arr, mean_curve, err_curve)."""
    out = {}
    for n in names:
        stacks = []
        t_ref = None
        for r in by_name[n]:
            arr = metric_over_t(r, kind, k, subspace, mk) * scale
            stacks.append(arr)
            t_ref = np.array([tm["t"] for tm in r["metrics_per_t"]])
        stacks = np.stack(stacks, axis=0)  # (n_seeds, n_t)
        m, e = ci_from_stack(stacks)
        out[n] = (t_ref, m, e)
    return out


# ----------------------------------------------------------------------------
# Coupling ordering helper
# ----------------------------------------------------------------------------

def ordered_names(by_name):
    """Sort coupling names: C0, C1, C2, C3@λ ascending, C4, then sanity tail."""
    raw = list(by_name.keys())
    def key(n):
        if n.startswith("C0"): return (0, 0.0)
        if n.startswith("C1"): return (1, 0.0)
        if n.startswith("C2"): return (2, 0.0)
        if n.startswith("C3"):
            lam = by_name[n][0]["meta"].get("lambda", 0.0)
            return (3, lam)
        if n.startswith("C4"): return (4, 0.0)
        return (5, 0.0)
    return sorted(raw, key=key)


# ----------------------------------------------------------------------------
# Main: 4-panel summary with error bars
# ----------------------------------------------------------------------------

def _build_color_map(names, by_name):
    """Colorblind-safe palette: Okabe-Ito categorical hues for non-C3
    couplings + a sequential blue gradient over the C3 family (light->dark
    indexed by lambda; the hard-blocked C3^inf endpoint sits at the dark end).

    Headline-only override of figure_style.coupling_color; the rest of the
    paper continues to use the project-wide palette.
    """
    # Sort C3 entries by lambda, treating C3inf as +inf so the gradient end
    # naturally lands on the hard-blocked endpoint.
    def lam_key(n):
        if n == "C3inf_blocked_ot":
            return float("inf")
        return by_name[n][0]["meta"].get("lambda", 0.0)
    c3_names_sorted = sorted([n for n in names if n.startswith("C3")],
                             key=lam_key)
    c3_color = {}
    n_c3 = len(c3_names_sorted)
    for i, n in enumerate(c3_names_sorted):
        idx = min(i, len(_C3_BLUE_GRADIENT) - 1)
        # If sweep has fewer than the gradient length, stretch indexing so
        # the endpoints (lam=1 and C3inf) sit on the gradient extremes.
        if n_c3 > 1:
            idx = int(round(i * (len(_C3_BLUE_GRADIENT) - 1) / (n_c3 - 1)))
        c3_color[n] = _C3_BLUE_GRADIENT[idx]
    out = {}
    for n in names:
        if n.startswith("C3"):
            out[n] = c3_color[n]
        else:
            out[n] = CB_COLOR.get(n, "#666666")
    return out


def make_main_figure(results, tag, suffix=SUFFIX):
    runs = results["runs"]
    by_name = runs_by_name(runs)
    names = ordered_names(by_name)
    name_color = _build_color_map(names, by_name)
    n_seeds = max(len(by_name[n]) for n in names)

    fig, axes = plt.subplots(2, 2, figsize=(7.5, 6.5))

    display_labels = [_display(n) for n in names]

    # Panel 1: mismatch_rate
    ax = axes[0, 0]
    m, e = agg_pair_metric(by_name, names, "mismatch_rate")
    ax.bar(range(len(names)), m, yerr=e, capsize=4,
           color=[name_color[n] for n in names],
           edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(display_labels, rotation=0, ha="center", fontsize=10)
    ax.set_ylabel(r"$\Pr(K_X \neq C)$ after coupling")
    ax.set_title(rf"(a) mismatch rate "
                 rf"(mean $\pm$ {CI_TYPE} over {n_seeds} seeds)")
    ax.axhline(0.5, color="gray", ls=":", alpha=0.5)
    ax.text(5.5, 0.51, "random = 0.5", ha="center", va="bottom",
            fontsize=9, color="gray")
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 2: max_t semantic-only A_v_norm (conditional)
    ax = axes[0, 1]
    m, e = agg_max_t(by_name, names, "cond_avg", K_MAIN, "sem", "A_v_norm")
    ax.bar(range(len(names)), m, yerr=e, capsize=4,
           color=[name_color[n] for n in names],
           edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(display_labels, rotation=0, ha="center", fontsize=10)
    ax.set_ylabel(r"$\max_t\,\mathcal{A}_v^S(t|C) / E\|U_S\|^2$")
    ax.set_title(r"(b) max-t semantic-only conditional ambiguity")
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 3: max_t H(K_X|Y_t,C)/log 2 (semantic subspace)
    ax = axes[1, 0]
    m, e = agg_max_t(by_name, names, "cond_avg", K_MAIN, "sem", "H_KY",
                     scale=1.0 / np.log(2.0))
    ax.bar(range(len(names)), m, yerr=e, capsize=4,
           color=[name_color[n] for n in names],
           edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(display_labels, rotation=0, ha="center", fontsize=10)
    ax.set_ylabel(r"$\max_t\,H(K_X|Y_t, C)/\log 2$")
    ax.set_title("(c) max-t conditional branch entropy")
    ax.axhline(1.0, color="gray", ls=":", alpha=0.5)
    ax.text(5.5, 0.97, "log 2 (max)", ha="center", va="top",
            fontsize=9, color="gray")
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 4 (Pareto): transport_full vs max_t A_v_sem_norm(t|C)
    # Distinct marker per coupling on top of color, for accessibility / B&W print.
    ax = axes[1, 1]
    tc_m, tc_e = agg_pair_metric(by_name, names, "transport_full")
    av_m, av_e = agg_max_t(by_name, names, "cond_avg", K_MAIN, "sem",
                           "A_v_norm")
    for i, n in enumerate(names):
        ax.errorbar(tc_m[i], av_m[i], xerr=tc_e[i], yerr=av_e[i],
                    fmt=_marker(n), ms=6, capsize=4,
                    color=name_color[n],
                    markeredgecolor="black", markeredgewidth=0.6,
                    label=_display(n))
    ax.set_xlabel(r"transport cost $E\|X-Z\|^2$ (full)")
    ax.set_ylabel(r"$\max_t\,\mathcal{A}_v^S(t|C) / E\|U_S\|^2$")
    ax.set_title("(d) Pareto: transport vs semantic ambiguity")
    _dhandles = [Line2D([0], [0], marker=_marker(n), linestyle="none",
                        markerfacecolor=name_color[n], markeredgecolor="black",
                        markeredgewidth=0.6, markersize=6, label=_display(n))
                 for n in names]
    ax.legend(handles=_dhandles, loc="best", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = FIG_DIR / f"{OUTPUT_STEM}{suffix}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


# ----------------------------------------------------------------------------
# Curves figure with shaded CI bands
# ----------------------------------------------------------------------------

def make_curves_figure(results, tag, suffix=SUFFIX):
    runs = results["runs"]
    by_name = runs_by_name(runs)
    names = ordered_names(by_name)
    name_color = _build_color_map(names, by_name)
    n_seeds = max(len(by_name[n]) for n in names)

    plt.rcParams.update({"axes.labelsize": 16, "xtick.labelsize": 16,
                         "ytick.labelsize": 16, "axes.titlesize": 16})
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    def plot_panel(ax, kind, subspace, mk, ylabel, title, scale=1.0,
                   hline=None):
        curves = agg_curve_t(by_name, names, kind, K_MAIN, subspace, mk,
                             scale=scale)
        for n in names:
            t_arr, m, e = curves[n]
            ax.plot(t_arr, m, marker=_marker(n), ls="-", lw=1.5, ms=5,
                    color=name_color[n], alpha=0.95, label=_display(n),
                    markeredgecolor="black", markeredgewidth=0.4)
            if n_seeds > 1:
                ax.fill_between(t_arr, m - e, m + e,
                                color=name_color[n], alpha=0.15)
        ax.set_xlabel("t")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if hline is not None:
            ax.axhline(hline, color="gray", ls=":", alpha=0.5)
        ax.legend(loc="best", fontsize=10, ncol=2)
        ax.grid(True, alpha=0.3)

    plot_panel(axes[0, 0], "cond_avg", "full", "A_v_norm",
               r"$\mathcal{A}_v^{\rm full}(t|C) / E\|U\|^2$",
               r"(a) full-state conditional $\mathcal{A}_v(t|C)$")
    plot_panel(axes[0, 1], "cond_avg", "sem", "A_v_norm",
               r"$\mathcal{A}_v^S(t|C) / E\|U_S\|^2$",
               r"(b) semantic-only conditional $\mathcal{A}_v^S(t|C)$")
    plot_panel(axes[1, 0], "cond_avg", "sem", "H_KY",
               r"$H(K_X|Y_t,C)/\log 2$ (sem)",
               "(c) conditional branch entropy (semantic subspace)",
               scale=1.0 / np.log(2.0), hline=1.0)
    plot_panel(axes[1, 1], "cond_avg", "sem", "R_switch",
               r"$R_{\rm switch}^S(t|C)$",
               r"(d) semantic-only $R_{\rm switch}^S(t|C)$")

    fig.tight_layout()
    out = FIG_DIR / f"{CURVES_STEM}{suffix}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


# ----------------------------------------------------------------------------
# Headline numbers report
# ----------------------------------------------------------------------------

def print_summary(results, tag):
    runs = results["runs"]
    by_name = runs_by_name(runs)
    names = ordered_names(by_name)
    n_seeds = max(len(by_name[n]) for n in names)
    cfg = results["config"]
    print(f"E3 summary [{tag}]: N={cfg['N_PAIRS']}, "
          f"d_g={cfg['D_G']}, a={cfg['A']}, sigma_s={cfg['SIGMA_S']}, "
          f"{n_seeds} seeds")
    print("-" * 100)
    fmt = (f"{'coupling':30s} {'mismatch':>14s} "
           f"{'trans_full':>14s} {'A_v_sem(0.5|C)':>18s} {'H(K|Y,C)/log2':>18s}")
    print(fmt)
    print("-" * 100)
    for n in names:
        rs = by_name[n]
        mismatch = np.array([r["pair_metrics"]["mismatch_rate"] for r in rs])
        trans = np.array([r["pair_metrics"]["transport_full"] for r in rs])
        # take t=0.5 (index 9 of 19) for headline
        idx_mid = len(rs[0]["metrics_per_t"]) // 2
        av_sem_mid = []
        h_mid = []
        for r in rs:
            tm = r["metrics_per_t"][idx_mid]
            cd = tm.get("cond_avg", {})
            if cd:
                blk = get_per_k(cd, K_MAIN)
                av_sem_mid.append(blk["sem"]["A_v_norm"])
                h_mid.append(blk["sem"]["H_KY"] / np.log(2.0))
        av_sem_mid = np.array(av_sem_mid)
        h_mid = np.array(h_mid)
        def fmt_val(arr):
            if len(arr) >= 2:
                return f"{arr.mean():.3f} ± {arr.std(ddof=1):.3f}"
            return f"{arr.mean():.3f}"
        print(f"{n:30s} {fmt_val(mismatch):>14s} "
              f"{fmt_val(trans):>14s} {fmt_val(av_sem_mid):>18s} "
              f"{fmt_val(h_mid):>18s}")


def main():
    print(f"loading results/e3_metrics_{TAG}.json ...")
    results = load_results(TAG)
    print_summary(results, TAG)
    p1 = make_main_figure(results, TAG)
    p2 = make_curves_figure(results, TAG)
    print(f"wrote {p1}")
    print(f"wrote {p2}")


if __name__ == "__main__":
    main()
