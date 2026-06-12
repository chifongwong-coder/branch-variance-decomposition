"""Plot the experiment E5 real-data validation figure from saved raw data.

Renders a 4-panel figure. Pure plotting, no compute, so it is cheap to re-run
while tuning the figure.

  (a) model-free feature audit: de-biased A_between under the oracle vs the
      unsupervised proxy, normalized to the oracle (K=2/3/10), at a matched 40
      neighbors per branch, read in the ambiguity window (t <= 1/2).
  (b) model-grounded real-path audit: de-biased A_between(t), oracle vs proxy (K=2).
  (c) label-free proxy recovery: recovery ratio vs t (K=2).
  (d) branch observability: neighborhood class purity vs t.

Panels (b)(c)(d) read results/e5_realdata_validation.json (written by
e5_realdata_validation.py). Panel (a) reads results/e5_recovery_vs_cardinality.json
(written by src/E5b/make_e5_recovery_aggregate.py, which aggregates the recovery
sweep src/E5b/expE5b_recovery_bandwidth.py); run both pipelines before plotting.

Panels (b) and (c) shade the ambiguity regime (where the de-biased oracle signal
is positive) and the committed regime (after the zero crossing); recovery is read
only in the ambiguity regime.

Run:  python3 src/E5/plot_e5_realdata_validation.py
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
from core.figure_style import apply_paper_style                                     # noqa: E402

RES = HERE.parents[1] / "results"
FIG = HERE.parents[1] / "figures"
JSON = RES / "e5_realdata_validation.json"
SPLIT = 0.45    # ambiguity / commitment boundary (the de-biased signal crosses zero near t=0.5)


def ms(lst):
    a = np.array(lst, dtype=float)
    sd = float(np.nanstd(a, ddof=1)) if np.sum(~np.isnan(a)) > 1 else 0.0
    return float(np.nanmean(a)), sd


def shade_regimes(ax, t, label=False):
    """Light shading: ambiguity regime up to SPLIT, committed regime after."""
    lo, hi = float(t.min()), float(t.max())
    ax.axvspan(lo, SPLIT, color="tab:green", alpha=0.06,
               label="ambiguity regime" if label else None)
    ax.axvspan(SPLIT, hi, color="gray", alpha=0.10,
               label="committed regime" if label else None)


def main():
    with open(JSON) as f:
        d = json.load(f)
    real = d["realpath"]
    t = np.array(real["t"], dtype=float)
    tk = [str(tt) for tt in real["t"]]

    apply_paper_style()
    plt.rcParams.update({"axes.labelsize": 13, "xtick.labelsize": 13,
                         "ytick.labelsize": 13, "axes.titlesize": 13})
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))

    # (a) model-free feature audit: de-biased A_between under the oracle labeling
    # vs the unsupervised proxy, normalized to the oracle (so oracle = 1 and the
    # proxy bar is the recovery ratio), read in the ambiguity window (t <= 1/2) at
    # a matched 40 neighbors per branch. K=10 carries no per-seed store, so no
    # error bars are drawn (the recovery means live in the aggregate JSON).
    axA = axes[0, 0]
    with open(RES / "e5_recovery_vs_cardinality.json") as f:
        rec = json.load(f)
    Ks = rec["K"]
    bidx = rec["neighbors_per_branch"].index(40)
    proxy = [rec["recovery_mean"][str(K)][bidx] for K in Ks]
    xs = np.arange(len(Ks)); w = 0.38
    axA.bar(xs - w / 2, [1.0] * len(Ks), w, color="tab:blue", label="oracle")
    axA.bar(xs + w / 2, proxy, w, color="tab:orange", label="proxy")
    axA.axhline(1.0, color="gray", lw=0.8, ls="--")
    axA.set_xticks(xs); axA.set_xticklabels([f"K={K}" for K in Ks])
    axA.set_ylim(0, 1.25)
    axA.set_ylabel(r"de-biased $\mathcal{A}_{\mathrm{between}}$ (norm. to oracle)")
    axA.set_title("(a) model-free proxy recovery")
    axA.legend(fontsize=8, loc="lower right")

    # (b) real-path de-biased A_between(t), oracle vs proxy (K=2)
    axB = axes[0, 1]
    shade_regimes(axB, t, label=True)
    for key, lab in (("deb_oracle", "oracle"), ("deb_proxy", "proxy")):
        mv = np.array([ms(real["2"][key][k])[0] for k in tk])
        sv = np.array([ms(real["2"][key][k])[1] for k in tk])
        axB.plot(t, mv, "o-", ms=3, label=lab)
        axB.fill_between(t, mv - sv, mv + sv, alpha=0.2)
    axB.axhline(0, color="gray", lw=0.6)
    axB.set_xlabel("t"); axB.set_ylabel(r"de-biased $\mathcal{A}_{\mathrm{between}}$ (K=2)")
    axB.set_title("(b) model-grounded real-path audit"); axB.legend(fontsize=8)

    # (c) proxy recovery ratio vs t (K=2)
    axC = axes[1, 0]
    shade_regimes(axC, t)
    mv = np.array([ms(real["2"]["recovery"][k])[0] for k in tk])
    sv = np.array([ms(real["2"]["recovery"][k])[1] for k in tk])
    axC.plot(t, mv, "o-", ms=3, color="tab:green")
    axC.fill_between(t, mv - sv, mv + sv, alpha=0.2, color="tab:green")
    axC.axhline(1, color="gray", lw=0.6, ls="--")
    axC.set_xlabel("t"); axC.set_ylabel("proxy recovery (K=2)")
    axC.set_title("(c) label-free proxy recovery"); axC.set_ylim(-0.5, 1.5)

    # (d) neighborhood class purity vs t
    axD = axes[1, 1]
    mv = np.array([ms(real["purity"][k])[0] for k in tk])
    sv = np.array([ms(real["purity"][k])[1] for k in tk])
    axD.plot(t, mv, "o-", ms=3, color="tab:purple")
    axD.fill_between(t, mv - sv, mv + sv, alpha=0.2, color="tab:purple")
    axD.axhline(0.1, color="gray", lw=0.6, ls=":")
    axD.set_xlabel("t"); axD.set_ylabel("neighborhood class purity")
    axD.set_title("(d) branch observability"); axD.set_ylim(0, 0.6)

    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "e5_realdata_validation.png", dpi=200)
    plt.close(fig)
    print(f"wrote {FIG / 'e5_realdata_validation.png'}")


if __name__ == "__main__":
    main()
