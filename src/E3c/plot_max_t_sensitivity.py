"""Phase diagram P2 sentinel: max_t vs t=0.5 sensitivity figure.

The headline reading at t=0.5 is fine for most cells, but at high-a
cells where A_v^S(t) is monotone decreasing, t=0.5 underestimates the
maximum ambiguity. The P2 sentinel t-profile lets us quantify this.

For each P2 sentinel cell + coupling:
  read_t05    = A_v^S(C, t=0.5)
  read_max_t  = max_{t in {0.1, 0.3, 0.5, 0.7, 0.9}} A_v^S(C, t)
  ratio       = read_max_t / read_t05

Reports ratio table + a small figure showing t=0.5 vs max_t for each
cell. If ratio is uniformly close to 1, t=0.5 reading is fine; if it
deviates, the headline P1 collapse fit using t=0.5 should be checked.
"""

from __future__ import annotations

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

P2_JSON = Path(__file__).resolve().parents[2] / "results" / "phase_diagram_t_sentinel.json"
OUT_DIR = Path(__file__).resolve().parents[2] / "figures"

COUPLING_COLOR = {
    "C0":    PAL.C0,
    "C1":    PAL.C1,
    "C3@10": PAL.C3_grad[3],
    "C4":    PAL.C4,
}


def main():
    with open(P2_JSON) as f:
        d = json.load(f)
    cells = d["cells"]
    t_grid = d["config"]["t_grid"]
    K = d["config"]["K"]
    couplings = d["config"]["coupling_tags"]
    n_cells = len(cells)

    # Compute per-(cell, coupling): mean A_v_sem over seeds at each t,
    # then max_t and t=0.5 reading.
    ti_5 = int(np.argmin(np.abs(np.asarray(t_grid) - 0.5)))

    rows = []  # (cell_idx, D_G, a, sigma_s, lsnr, coupling, max_t, idx_max_t, t05)
    for ci, cell in enumerate(cells):
        D_G = cell["D_G"]
        A = cell["A"]
        sigma_s = cell["sigma_s"]
        lsnr = cell["lsnr"]
        by_c = {}
        for r in cell["runs"]:
            by_c.setdefault(r["coupling"], []).append(r)
        for c in couplings:
            avs = np.array([
                [r["metrics_per_t"][ti]["cond_avg"][str(K)]["sem"]["A_v_norm"]
                 for ti in range(len(t_grid))]
                for r in by_c[c]
            ])  # (n_seeds, n_t)
            mean_per_t = avs.mean(axis=0)
            sd_per_t = avs.std(axis=0, ddof=1) if avs.shape[0] > 1 else np.zeros_like(mean_per_t)
            t05 = float(mean_per_t[ti_5])
            sd_t05 = float(sd_per_t[ti_5])
            idx_max = int(np.argmax(mean_per_t))
            max_t = float(mean_per_t[idx_max])
            sd_max = float(sd_per_t[idx_max])
            t_at_max = float(t_grid[idx_max])
            rows.append({
                "D_G": D_G, "a": A, "sigma_s": sigma_s, "lsnr": lsnr,
                "coupling": c,
                "t05": t05, "sd_t05": sd_t05,
                "max_t": max_t, "sd_max": sd_max,
                "t_at_max": t_at_max,
                "ratio_max_over_t05": max_t / max(t05, 1e-12),
                "delta_max_minus_t05": max_t - t05,
            })

    # Print table
    print(f"{'D_G':>4} {'a':>5} {'lsnr':>7} {'coupling':>8} | "
          f"{'t=0.5':>7} {'max_t':>7} {'@t':>5} {'ratio':>6} {'delta':>7}")
    for r in rows:
        print(f"{r['D_G']:>4d} {r['a']:>5g} {r['lsnr']:>+7.2f} {r['coupling']:>8s} | "
              f"{r['t05']:>7.3f} {r['max_t']:>7.3f} {r['t_at_max']:>5.2f} "
              f"{r['ratio_max_over_t05']:>6.3f} {r['delta_max_minus_t05']:>+7.3f}")

    # Figure: 2 panels
    #   Left:  scatter (t=0.5, max_t) per (cell, coupling), with y=x line
    #   Right: bar chart of delta (max - t05) per cell, coloured by coupling
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.5))

    # Left: scatter
    for c in couplings:
        xs = np.array([r["t05"] for r in rows if r["coupling"] == c])
        ys = np.array([r["max_t"] for r in rows if r["coupling"] == c])
        ex = np.array([r["sd_t05"] for r in rows if r["coupling"] == c])
        ey = np.array([r["sd_max"] for r in rows if r["coupling"] == c])
        axL.errorbar(xs, ys, xerr=ex, yerr=ey, fmt="o", ms=7,
                     capsize=2.5, color=COUPLING_COLOR[c],
                     label=c, lw=1.0, elinewidth=0.7)
    # y = x
    lo, hi = 0.0, 1.05
    axL.plot([lo, hi], [lo, hi], ls="--", color="gray", lw=1.0,
             label=r"$y = x$ (t=0.5 == max_t)")
    axL.set_xlim(lo, hi)
    axL.set_ylim(lo, hi)
    axL.set_xlabel(r"$\widetilde{\mathcal{A}}_v^S(C, t=0.5)$")
    axL.set_ylabel(r"$\max_t\,\widetilde{\mathcal{A}}_v^S(C, t)$  over $t\in\{0.1,0.3,0.5,0.7,0.9\}$")
    axL.set_title(r"(a) Sentinel $\max_t$ sensitivity: $t=0.5$ vs $\max_t$ reading "
                  r"(5 sentinel cells $\times$ 4 couplings)")
    axL.legend(loc="upper left", fontsize=8)

    # Right: bar chart of delta per cell, colored by coupling
    cell_labels = []
    n_cells_unique = []
    seen = set()
    for r in rows:
        key = (r["D_G"], r["a"], r["sigma_s"])
        if key not in seen:
            seen.add(key)
            n_cells_unique.append(key)
            cell_labels.append(fr"$D_G={r['D_G']},a={r['a']:g}$" + "\n" + fr"lsnr$={r['lsnr']:+.2f}$")
    n = len(n_cells_unique)
    x_base = np.arange(n)
    width = 0.20
    for j, c in enumerate(couplings):
        deltas = []
        for key in n_cells_unique:
            for r in rows:
                if (r["D_G"], r["a"], r["sigma_s"]) == key and r["coupling"] == c:
                    deltas.append(r["delta_max_minus_t05"])
                    break
        offsets = (j - (len(couplings) - 1) / 2) * width
        axR.bar(x_base + offsets, deltas, width, color=COUPLING_COLOR[c],
                edgecolor="black", linewidth=0.4, label=c, alpha=0.85)
    axR.axhline(0, color="gray", lw=0.7)
    axR.set_xticks(x_base)
    axR.set_xticklabels(cell_labels, fontsize=8.5)
    axR.set_ylabel(r"$\max_t \widetilde{\mathcal{A}}_v^S - \widetilde{\mathcal{A}}_v^S|_{t=0.5}$")
    axR.set_title(r"(b) per-cell $\max_t - t{=}0.5$ deviation")
    axR.legend(loc="best", fontsize=8)

    fig.suptitle(
        r"Sentinel cells: $\max_t$ vs $t{=}0.5$ reading on $\widetilde{\mathcal{A}}_v^S$",
        fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = OUT_DIR / "phase_diagram_max_t_sensitivity.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
