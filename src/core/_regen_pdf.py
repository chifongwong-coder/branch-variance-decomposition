"""Monkey-patch matplotlib.savefig to also emit .pdf alongside the original
.png output, then exec the target plotting script.

Usage:
    python _regen_pdf.py <script.py>

Effect: every call to `fig.savefig(path.png, ...)` inside the target script
also writes `path.pdf` (vector). Original PNG output is unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.figure

_orig_savefig = matplotlib.figure.Figure.savefig


def _patched_savefig(self, fname, *args, **kwargs):
    _orig_savefig(self, fname, *args, **kwargs)
    fname_str = str(fname)
    if fname_str.lower().endswith(".png"):
        pdf_path = fname_str[:-4] + ".pdf"
        pdf_kwargs = {k: v for k, v in kwargs.items() if k != "dpi"}
        _orig_savefig(self, pdf_path, *args, **pdf_kwargs)


matplotlib.figure.Figure.savefig = _patched_savefig


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python _regen_pdf.py <script.py>", file=sys.stderr)
        sys.exit(1)
    script = Path(sys.argv[1]).resolve()
    sys.argv = [str(script)] + sys.argv[2:]
    with open(script) as f:
        code = compile(f.read(), str(script), "exec")
    exec(code, {"__name__": "__main__", "__file__": str(script)})


if __name__ == "__main__":
    main()
