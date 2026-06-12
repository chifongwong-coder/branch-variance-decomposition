"""Plot the E7 branch-conditioning payoff figure from saved raw data.

Reads results/e7_conditioning_gain_payoff.json and renders a 2x2 figure:
  (a) MSE curves: unconditional, true-label conditional, label-averaged conditional.
  (b) strict BVD payoff: realized cross-model gain Delta_cross vs posterior-weighted
      spread S_post (the headline; ambiguity window t<=0.5 shaded).
  (c) within-checkpoint sanity: Delta_within vs uniform spread S_uni (no cross-model
      confound, CLIP-free).
  (d) alignment scatter: S_post vs Delta_cross over t<=0.5 with the 1:1 line.

Pure plotting, no compute. Run:
  python3 plot_e7_conditioning_gain_payoff.py
For the pdf sibling:  python3 _regen_pdf.py plot_e7_conditioning_gain_payoff.py
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
JSON = RES / "e7_conditioning_gain_payoff.json"
SPLIT = 0.5    # ambiguity window boundary


def main():
    with open(JSON) as f:
        d = json.load(f)
    T = np.array(d["t"], dtype=float)
    def per(key):
        return np.array([d[key][str(t)] for t in d["t"]], dtype=float)  # (nt, nseed)
    mu, mt, ma = per("mse_uncond"), per("mse_cond_true"), per("mse_cond_avg")
    su, sp = per("s_uni"), per("s_post")
    dc = mu - mt; dw = ma - mt
    def mss(a): return a.mean(1), a.std(1, ddof=1)
    amb = T <= SPLIT
    def cosrat(a, b, m=None):
        x, y = (a.mean(1), b.mean(1))
        if m is not None: x, y = x[m], y[m]
        return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y))), float(np.mean(x / y))

    apply_paper_style()
    plt.rcParams.update({"axes.labelsize": 13, "xtick.labelsize": 13,
                         "ytick.labelsize": 13, "axes.titlesize": 13})
    fig, ax = plt.subplots(2, 2, figsize=(11, 7.5))

    # (a) MSE curves
    for arr, lab in ((mu, r"$\mathrm{MSE}_{\rm uncond}$"),
                     (mt, r"$\mathrm{MSE}_{\rm cond,true}$"),
                     (ma, r"$\mathrm{MSE}_{\rm cond,avg}$")):
        m, s = mss(arr); ax[0, 0].plot(T, m, "o-", ms=3, label=lab)
        ax[0, 0].fill_between(T, m - s, m + s, alpha=0.15)
    ax[0, 0].set_xlabel("t"); ax[0, 0].set_ylabel("velocity MSE")
    ax[0, 0].set_title("(a) regression error"); ax[0, 0].legend(fontsize=8)

    # (b) strict payoff: Delta_cross vs S_post
    axB = ax[0, 1]
    axB.axvspan(float(T.min()), SPLIT, color="tab:green", alpha=0.06, label="ambiguity window")
    for arr, lab, c in ((dc, r"$\Delta_{\rm cross}$ (realized gain)", "tab:blue"),
                        (sp, r"$S_{\rm post}$ (strict $\mathcal{A}_{\rm between}$)", "tab:red")):
        m, s = mss(arr); axB.plot(T, m, "o-", ms=3, color=c, label=lab)
        axB.fill_between(T, m - s, m + s, alpha=0.2, color=c)
    axB.axhline(0, color="gray", lw=0.6)
    cb, rb = cosrat(dc, sp, amb)
    axB.set_xlabel("t"); axB.set_ylabel("velocity-MSE gain")
    axB.set_title(f"(b) strict payoff: cos={cb:.4f}, ratio={rb:.2f} (t<=0.5)")
    axB.legend(fontsize=8)

    # (c) within-checkpoint sanity: Delta_within vs S_uni
    axC = ax[1, 0]
    for arr, lab, c in ((dw, r"$\Delta_{\rm within}$ (same-model gain)", "tab:purple"),
                        (su, r"$S_{\rm uni}$ (uniform spread)", "tab:orange")):
        m, s = mss(arr); axC.plot(T, m, "o-", ms=3, color=c, label=lab)
        axC.fill_between(T, m - s, m + s, alpha=0.2, color=c)
    cc, rc = cosrat(dw, su)
    axC.set_xlabel("t"); axC.set_ylabel("velocity-MSE gain")
    axC.set_title(f"(c) within-checkpoint: cos={cc:.4f}, ratio={rc:.2f}")
    axC.legend(fontsize=8)

    # (d) alignment scatter S_post vs Delta_cross over t<=0.5
    axD = ax[1, 1]
    spm = sp.mean(1)[amb]; dcm = dc.mean(1)[amb]
    axD.scatter(spm, dcm, c=T[amb], cmap="viridis", s=30, zorder=3)
    lim = [0, max(spm.max(), dcm.max()) * 1.05]
    axD.plot(lim, lim, "--", color="gray", lw=0.8, label="$y=x$")
    axD.set_xlim(lim); axD.set_ylim(lim)
    axD.set_xlabel(r"$S_{\rm post}$ (predicted)"); axD.set_ylabel(r"$\Delta_{\rm cross}$ (realized)")
    axD.set_title("(d) predicted vs realized (t<=0.5)"); axD.legend(fontsize=8)

    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "e7_conditioning_gain_payoff.png", dpi=200)
    plt.close(fig)
    print(f"wrote {FIG / 'e7_conditioning_gain_payoff.png'}")


if __name__ == "__main__":
    main()
