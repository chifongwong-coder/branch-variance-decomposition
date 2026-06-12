"""E5a plot script 3 of 3: proxy quality summary.

Reads results/cifar10_proxy_quality.json and produces a 1x2 figure:
  (a) bar chart of Hungarian accuracy / NMI / ARI for CLIP and weakRP
  (b) confusion matrix heatmaps for CLIP (left half) and weakRP (right half)

Output: figures/e5_quality.{png,pdf}
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.figure_style import apply_paper_style, Palette
apply_paper_style()

HERE = Path(__file__).resolve().parent
JSON_PATH = HERE.parents[1] / "results" / "cifar10_proxy_quality.json"
OUT_PNG = HERE.parents[1] / "figures" / "e5_quality.png"

CIFAR_CLASSES = ["airplane", "auto", "bird", "cat", "deer",
                  "dog", "frog", "horse", "ship", "truck"]


def main():
    q = json.load(open(JSON_PATH))

    fig = plt.figure(figsize=(12.5, 4.5))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.2, 1.0, 1.0, 0.05])
    ax_bar = fig.add_subplot(gs[0, 0])
    ax_cm_clip = fig.add_subplot(gs[0, 1])
    ax_cm_rp = fig.add_subplot(gs[0, 2])
    ax_cb = fig.add_subplot(gs[0, 3])

    # ---- bar chart of summary metrics
    metrics = ["hungarian_acc", "nmi", "ari"]
    metric_display = ["Hungarian acc", "NMI", "ARI"]
    proxies = ["CLIP_kmeans", "weakRP_kmeans"]
    proxy_display = ["CLIP $k$-means", "weak-RP $k$-means"]
    proxy_color = [Palette.pos, Palette.neg]

    x = np.arange(len(metrics))
    width = 0.36
    for i, (p, disp, color) in enumerate(zip(proxies, proxy_display, proxy_color)):
        vals = [q[p][m] for m in metrics]
        ax_bar.bar(x + (i - 0.5) * width, vals, width,
                   color=color, edgecolor="#333", linewidth=0.8,
                   label=disp)
        # annotate
        for xi, v in zip(x + (i - 0.5) * width, vals):
            ax_bar.text(xi, v + 0.02, f"{v:.2f}", ha="center", va="bottom",
                        fontsize=8)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(metric_display)
    ax_bar.set_ylim(0, 1.05)
    ax_bar.set_title("(a) proxy quality vs CIFAR-10 oracle (50k train)",
                     fontsize=11)
    ax_bar.set_ylabel("score")
    ax_bar.grid(True, axis="y", alpha=0.22, lw=0.7)
    ax_bar.legend(loc="upper right", fontsize=9, framealpha=0.92)

    # ---- confusion matrices (row-normalised)
    vmin, vmax = 0.0, 1.0
    for ax, p, disp in [(ax_cm_clip, "CLIP_kmeans", "(b) CLIP $k$-means"),
                         (ax_cm_rp,   "weakRP_kmeans", "(c) weak-RP $k$-means")]:
        cm = np.array(q[p]["confusion_matrix"], dtype=np.float64)
        row_sum = cm.sum(axis=1, keepdims=True)
        cm_norm = cm / np.maximum(row_sum, 1)
        im = ax.imshow(cm_norm, vmin=vmin, vmax=vmax, cmap="YlOrRd",
                        aspect="equal", origin="upper")
        ax.set_title(f"{disp} confusion matrix (row-norm)", fontsize=10.5)
        ax.set_xlabel("predicted class")
        ax.set_ylabel("true class")
        ax.set_xticks(range(10))
        ax.set_xticklabels(CIFAR_CLASSES, rotation=60, ha="right", fontsize=8)
        ax.set_yticks(range(10))
        ax.set_yticklabels(CIFAR_CLASSES, fontsize=8)
        # annotate diagonals + large off-diagonals
        for i in range(10):
            for j in range(10):
                v = cm_norm[i, j]
                if v > 0.15 or i == j:
                    color = "white" if v > 0.5 else "#222"
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            color=color, fontsize=6.5)

    fig.colorbar(im, cax=ax_cb)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
