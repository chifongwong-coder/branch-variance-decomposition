"""Plot the forward-pass robustness figure for E5 and E7 from saved raw data.

Reads results/e5_checkpoint_robustness.json and the per-coupling E7 payoff JSONs,
and renders a 1x2 figure:
  (a) E5 model-grounded real-path de-biased between-branch signal across the same
      unconditional training trajectory at 10k/30k/50k steps (mean over 3 diagnostic
      seeds, error bars are seed SD). The ambiguity-to-commitment profile is stable;
      the zero crossing shifts earlier with more training.
  (b) E7 strict conditioning-gain identity across conditional models trained with
      C_0, C_1, C_3^inf, each evaluated in-distribution (on its own coupling's
      velocity targets): the strict ratio Delta_cross / S_post in the ambiguity
      window (t <= 0.5), with the profile cosine annotated. The identity (ratio
      near one, cosine near one) holds across all three couplings.

Pure plotting, no compute. Run:
  python3 plot_e5_e7_robustness.py
For the pdf sibling:  python3 _regen_pdf.py plot_e5_e7_robustness.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from core.figure_style import apply_paper_style                                # noqa: E402

RES = HERE.parents[1] / "results"
FIG = HERE.parents[1] / "figures"
E5_JSON = RES / "e5_checkpoint_robustness.json"
# Each coupling evaluated in-distribution: C0's training joint is the independent
# pairing (its canonical payoff JSON), C1/C3inf re-pair (z,x) with their own coupling.
PAYOFF_JSON = {
    "c0":    RES / "e7_conditioning_gain_payoff.json",
    "c1":    RES / "e7_conditioning_gain_payoff_c1_indist.json",
    "c3inf": RES / "e7_conditioning_gain_payoff_c3inf_indist.json",
}
SPLIT = 0.5   # ambiguity window boundary, matches the main E7 strict reading

STEP_LABELS = {"10000": "10k steps", "30000": "30k steps", "50000": "50k steps"}
COUPLING_LABELS = {"c0": r"$C_0$", "c1": r"$C_1$", "c3inf": r"$C_3^\infty$"}


def _cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def _strict_arrays(d, seed=None):
    """Delta_cross and S_post over the ambiguity window t <= SPLIT. seed=None uses
    the seed-averaged curves; an integer picks one seed."""
    T = [t for t in d["t"] if t <= SPLIT]
    if seed is None:
        dc = np.array([np.mean(d["mse_uncond"][str(t)]) - np.mean(d["mse_cond_true"][str(t)]) for t in T])
        sp = np.array([np.mean(d["s_post"][str(t)]) for t in T])
    else:
        dc = np.array([d["mse_uncond"][str(t)][seed] - d["mse_cond_true"][str(t)][seed] for t in T])
        sp = np.array([d["s_post"][str(t)][seed] for t in T])
    return dc, sp


def strict_ratio(d):
    dc, sp = _strict_arrays(d)
    return float(np.mean(dc / sp))


def strict_cos(d):
    dc, sp = _strict_arrays(d)
    return _cos(dc, sp)


def strict_ratio_per_seed(d):
    nseed = len(d["mse_uncond"][str(d["t"][0])])
    return np.array([float(np.mean(np.divide(*_strict_arrays(d, s)))) for s in range(nseed)])


def main():
    apply_paper_style()
    plt.rcParams.update({"axes.labelsize": 12, "xtick.labelsize": 12,
                         "ytick.labelsize": 12, "axes.titlesize": 12})
    e5 = json.load(open(E5_JSON))
    payoff = {k: json.load(open(p)) for k, p in PAYOFF_JSON.items()}

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.2, 3.6))

    # (a) E5 de-biased between-branch signal across training steps
    T5 = e5["t"]
    colors = {"10000": "#1f77b4", "30000": "#ff7f0e", "50000": "#2ca02c"}
    for S in e5["steps"]:
        arr = np.array([e5["by_step"][str(S)]["deb_oracle"][str(t)] for t in T5])  # (nt, nseed)
        m = arr.mean(1)
        sd = arr.std(1, ddof=1)
        axL.errorbar(T5, m, yerr=sd, marker="o", ms=4, capsize=2, lw=1.5,
                     color=colors[str(S)], label=STEP_LABELS[str(S)])
    axL.axhline(0.0, color="0.5", lw=0.8, ls="--")
    axL.set_xlabel(r"flow time $t$")
    axL.set_ylabel(r"de-biased $A_{\mathrm{between}}$")
    axL.set_title(r"(a) real-path stability across training")
    axL.legend(frameon=False, fontsize=7, loc="upper right")

    # (b) E7 strict payoff identity across coupling-trained models (in-distribution).
    # Marker = strict ratio Delta_cross/S_post over t<=0.5; error bar = per-seed SD;
    # profile cosine annotated above each point; dashed reference at ratio 1.
    order = ["c0", "c1", "c3inf"]
    xs = np.arange(len(order))
    bar_colors = ["#4c72b0", "#dd8452", "#55a868"]
    ratios = [strict_ratio(payoff[k]) for k in order]
    rsds = [strict_ratio_per_seed(payoff[k]).std(ddof=1) for k in order]
    coses = [strict_cos(payoff[k]) for k in order]
    axR.axhline(1.0, color="0.5", lw=0.8, ls="--")
    for x, r, e, c, col in zip(xs, ratios, rsds, coses, bar_colors):
        axR.errorbar(x, r, yerr=e, fmt="o", ms=8, capsize=4, lw=1.5, color=col)
        axR.text(x, r + e + 0.012, f"cos {c:.4f}", ha="center", va="bottom", fontsize=7.5)
    axR.set_xticks(xs)
    axR.set_xticklabels([COUPLING_LABELS[k] for k in order])
    axR.set_xlim(-0.5, len(order) - 0.5)
    axR.set_ylim(0.9, 1.25)
    axR.set_xlabel("conditional model coupling")
    axR.set_ylabel(r"strict $\Delta_{\mathrm{cross}}/S_{\mathrm{post}}$ ($t\leq0.5$)")
    axR.set_title(r"(b) strict payoff identity across couplings")

    fig.tight_layout()
    out = FIG / "e5_e7_robustness.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
