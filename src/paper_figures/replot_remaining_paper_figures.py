"""Re-plot the three remaining paper figures from existing JSON metrics
using the unified paper style.  This avoids re-running the expensive
experiments (E3a binary ~10 min, E3b 4-mode ~9 min, E4 main ~11 min).

Inputs (read-only):
  results/e3a_metrics_phase1_pilot.json       (E3a binary)
  results/e3a_metrics_e3b_phase1_pilot.json   (E3b 4-mode XOR)
  results/e4_metrics_conservative.json      (E4 fixed-t MLP)

Outputs (overwrites):
  figures/e3a_curves_phase1_pilot.png
  figures/e3a_curves_e3b_phase1_pilot.png
  figures/e4_capacity_ladder_conservative.png
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.figure_style import apply_paper_style, Palette, coupling_color
apply_paper_style()

HERE = Path(__file__).resolve().parent
RES = HERE.parents[1] / "results"
FIG = HERE.parents[1] / "figures"
SIGMA_BOUND_BAND = 2.0


# ============================================================================
# E3a / E3b curves figure (4 panels)
# ============================================================================

def _paper_display_name(name, lam):
    """Map internal run names to paper notation ($C0$, $C3@\\lambda{=}10$, etc.)."""
    if name.startswith("C0"):
        return r"$C0$"
    if name.startswith("C1"):
        return r"$C1$"
    if name.startswith("C2"):
        return r"$C2$"
    if name.startswith("C3"):
        # format lambda nicely: 0.5, 1, 2, 5, 10 etc.
        lam_s = f"{lam:g}"
        return rf"$C3@\lambda{{=}}{lam_s}$"
    if name.startswith("C4"):
        return r"$C4$"
    return name


def _ordered_e3_couplings(by_name):
    """Sort coupling names in canonical order, then map to paper colours.
    Sanity-check variants (e.g. C3_semOT_lam0_sanity_eq_C1) are dropped to
    keep the figure consistent with paper notation; the paper §9 E3a/E3b text
    only refers to $C0,C1,C2,C3@\\lambda,C4$."""
    def is_sanity(n):
        return "_sanity_" in n
    visible = {n: rs for n, rs in by_name.items() if not is_sanity(n)}

    def key(n):
        if n.startswith("C0"): return (0, 0.0)
        if n.startswith("C1"): return (1, 0.0)
        if n.startswith("C2"): return (2, 0.0)
        if n.startswith("C3"):
            lam = visible[n][0]["meta"].get("lambda", 0.0)
            return (3, lam)
        if n.startswith("C4"): return (4, 0.0)
        return (5, 0.0)
    names = sorted(visible.keys(), key=key)
    # Coupling colour map; C3 entries use a warm gradient ordered by lambda
    c3_names = [n for n in names if n.startswith("C3")]
    c3_index = {n: i for i, n in enumerate(c3_names)}
    cmap = {}
    display = {}
    for n in names:
        if n.startswith("C3"):
            cmap[n] = coupling_color(n, lam_index=c3_index[n])
            lam = visible[n][0]["meta"].get("lambda", 0.0)
            display[n] = _paper_display_name(n, lam)
        else:
            cmap[n] = coupling_color(n)
            display[n] = _paper_display_name(n, 0.0)
    return names, cmap, display


def _stack_metric(seeds_data, metric_key, k=80):
    """Return (n_seeds, n_t) array of a per_k metric across seeds for one coupling."""
    return np.array([
        [mt["per_k"][str(k)][metric_key] for mt in run["metrics"]]
        for run in seeds_data
    ])


def replot_e3_curves(tag, c0_oracle_label=True, n_seeds_label=None):
    json_path = RES / f"e3a_metrics_{tag}.json"
    with open(json_path) as f:
        data = json.load(f)

    by_name = {}
    for r in data["runs"].values():
        by_name.setdefault(r["name"], []).append(r)
    names, name_color, display_name = _ordered_e3_couplings(by_name)
    n_seeds = max(len(by_name[n]) for n in names)

    fig, axes = plt.subplots(2, 2, figsize=(13, 7))
    titles = [r"(a) Normalized $\mathcal{A}_v(t) = \mathcal{A}_v/E\|U\|^2$",
              r"(b) $H(K_X | Y_t)/\log 2$",
              r"(c) $R_{\rm switch}(t) = \mathcal{A}_{\rm between}/\mathcal{A}_v$",
              r"(d) $1 - R_v^2(t)$"]
    metric_keys = ["A_v_norm", "H_KY", "R_switch", "one_minus_R_v2"]

    for name in names:
        seeds_data = by_name[name]
        t_arr = np.array([m["t"] for m in seeds_data[0]["metrics"]])
        for (i, j), mk, scale in [
            ((0, 0), "A_v_norm", 1.0),
            ((0, 1), "H_KY", 1.0 / np.log(2.0)),
            ((1, 0), "R_switch", 1.0),
            ((1, 1), "one_minus_R_v2", 1.0),
        ]:
            ys = _stack_metric(seeds_data, mk).mean(axis=0) * scale
            axes[i, j].plot(t_arr, ys, "o-", lw=1.5, ms=3.5,
                            color=name_color[name], alpha=0.9,
                            label=display_name[name])

    # Closed-form C0 oracle overlay (if available, binary only)
    if c0_oracle_label and "c0_oracle" in data and data["c0_oracle"]:
        cf_t = sorted(float(t) for t in data["c0_oracle"].keys())
        cf_r = [data["c0_oracle"][f"{t}"]["R_switch"] for t in cf_t]
        axes[1, 0].plot(cf_t, cf_r, "k:", lw=1.5, alpha=0.7,
                        label="C0 closed form")

    for ax, ttl in zip(axes.flat, titles):
        ax.set_xlabel("t")
        ax.set_title(ttl)

    # single shared legend on top (couplings are identical across panels)
    handles, labels, seen = [], [], set()
    for ax in axes.flat:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in seen:
                seen.add(l); handles.append(h); labels.append(l)
    fig.legend(handles, labels, loc="upper center", ncol=5, fontsize=8.5,
               frameon=True, framealpha=0.92, bbox_to_anchor=(0.5, 1.02))

    cfg = data["config"]
    target_type = cfg.get("TARGET_TYPE", "binary")
    label_extra = "" if not n_seeds_label else f", {n_seeds_label}"
    fig.tight_layout()
    out = FIG / f"e3a_curves_{tag}.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  wrote {out}")


# ============================================================================
# E4 main capacity ladder (5 panels, one per t)
# ============================================================================

def replot_e4_capacity_ladder(tag):
    json_path = RES / f"e4_metrics_{tag}.json"
    with open(json_path) as f:
        data = json.load(f)

    cfg = data["config"]
    t_grid = cfg["T_GRID"]
    hidden_ladder = cfg["HIDDEN_LADDER"]
    seeds = cfg["SEEDS"]
    av_cf = {float(k): v for k, v in data["av_closed_form"].items()}

    apply_paper_style()
    plt.rcParams.update({"axes.labelsize": 19, "xtick.labelsize": 14,
                         "ytick.labelsize": 14, "axes.titlesize": 19})
    fig, axes = plt.subplots(1, len(t_grid), figsize=(16, 4.2), squeeze=False)
    axes = axes[0]
    eval_color = Palette.neg          # blue
    approx_color = Palette.pos        # orange
    irred_color = Palette.C2          # green
    cf_color = Palette.avg            # black

    for ti, t in enumerate(t_grid):
        ax = axes[ti]
        Lm, Lsd, Am, Asd, Im, Isd = [], [], [], [], [], []
        for hidden in hidden_ladder:
            sub = [r for r in data["runs"]
                   if r["t"] == t and r["hidden"] == hidden]
            L = np.array([r["L_eval_final"] for r in sub])
            A = np.array([r["approx_eval_final"] for r in sub])
            I = np.array([r["irred_eval_final"] for r in sub])
            Lm.append(L.mean()); Lsd.append(L.std(ddof=1))
            Am.append(A.mean()); Asd.append(A.std(ddof=1))
            Im.append(I.mean()); Isd.append(I.std(ddof=1))
        x = np.array(hidden_ladder)
        ax.errorbar(x, Lm, yerr=Lsd, fmt="o-", capsize=3, lw=1.6,
                    color=eval_color, label="eval loss")
        ax.errorbar(x, Am, yerr=Asd, fmt="s--", capsize=3, lw=1.6,
                    color=approx_color,
                    label=r"$\|v_\theta - v^*\|^2$")
        ax.errorbar(x, Im, yerr=Isd, fmt="^:", capsize=3, lw=1.6,
                    color=irred_color, label="irred lower bound")
        ax.axhline(av_cf[t], color=cf_color, ls=":",
                   alpha=0.7, lw=1.2,
                   label=rf"$\mathcal{{A}}_v(t)$ CF: {av_cf[t]:.3f}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.xaxis.set_major_formatter(plt.matplotlib.ticker.FixedFormatter([str(int(v)) for v in x]))
        ax.xaxis.set_minor_locator(plt.matplotlib.ticker.NullLocator())
        ax.set_xlabel("hidden width")
        if ti == 0:
            ax.set_ylabel("loss")
            ax.legend(loc="center", fontsize=11)
        ax.set_title(f"t = {t}", fontsize=19)

    fig.tight_layout()
    out = FIG / f"e4_capacity_ladder_{tag}.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  wrote {out}")


def main():
    print("Replotting E3a binary curves...")
    replot_e3_curves("phase1_pilot")
    print("Replotting E3b 4-mode XOR curves...")
    replot_e3_curves("e3b_phase1_pilot")
    print("Replotting E4 main capacity ladder...")
    replot_e4_capacity_ladder("conservative")


if __name__ == "__main__":
    main()
