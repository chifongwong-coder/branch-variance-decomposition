"""Stitch the five reflow component figures into one vertical composite PNG.

This script reproduces the convenience composite that ships alongside the
individual reflow component files, so the expert can review one attachment
instead of five. Each input figure is rescaled to the widest input width
and stacked top-to-bottom with thin gray separators.

The composite is a derived artifact; the load-bearing figures are the
five individuals, and any of them can be regenerated from
plot_reflow.py + reflow_conjecture1.json. The composite itself
is just a presentation aid.

Usage:
    python stitch_reflow_composite.py

Inputs (must exist):
    figures/reflow_profile.png
    figures/reflow_locality.png
    figures/reflow_share_per_t.png
    figures/reflow_endpoint_calibration.png
    figures/reflow_endpoint_covariance.png

If the individual PNGs are missing (e.g. after a PDF-only cleanup), run

    python src/core/_regen_pdf.py src/E6/plot_reflow.py

first to regenerate them from the JSON.

Output:
    figures/reflow_combined.png
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw


FIGS = [
    "reflow_profile.png",
    "reflow_locality.png",
    "reflow_share_per_t.png",
    "reflow_endpoint_calibration.png",
    "reflow_endpoint_covariance.png",
]
OUTPUT = "reflow_combined.png"
SEP_HEIGHT_PX = 10
SEP_COLOR = "#dddddd"


def main():
    figures_dir = Path(__file__).resolve().parents[2] / "figures"
    missing = [f for f in FIGS if not (figures_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing inputs:\n  "
            + "\n  ".join(missing)
            + "\nRun `python src/core/_regen_pdf.py src/E6/plot_reflow.py` first."
        )

    imgs = [Image.open(figures_dir / f) for f in FIGS]
    target_w = max(img.width for img in imgs)

    # Normalize widths via Lanczos resampling
    scaled = []
    for img in imgs:
        if img.width != target_w:
            new_h = int(img.height * target_w / img.width)
            img = img.resize((target_w, new_h), Image.LANCZOS)
        scaled.append(img)

    total_h = (sum(s.height for s in scaled)
               + SEP_HEIGHT_PX * (len(scaled) - 1))
    combined = Image.new("RGB", (target_w, total_h), "white")

    y = 0
    for i, s in enumerate(scaled):
        combined.paste(s, (0, y))
        y += s.height
        if i < len(scaled) - 1:
            ImageDraw.Draw(combined).rectangle(
                [(0, y), (target_w, y + SEP_HEIGHT_PX)], fill=SEP_COLOR
            )
            y += SEP_HEIGHT_PX

    out_path = figures_dir / OUTPUT
    combined.save(out_path)
    print(f"wrote {out_path}: {combined.width} x {combined.height} px")


if __name__ == "__main__":
    main()
