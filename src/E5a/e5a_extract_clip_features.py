"""E5a Stage 1a: one-time CLIP feature extraction for CIFAR-10 train.

Loads CIFAR-10 train (50,000 images), runs them through CLIP ViT-B/32
(bicubic upscale 32->224 + CLIP normalization), and caches to a single
NPZ file.

Output: results/cifar10_clip_features.npz with keys:
    features      (50000, 512) float32   raw CLIP image embedding (visual_projection)
    features_l2   (50000, 512) float32   L2-normalised version (||x|| = 1)
    labels        (50000,)     int64     CIFAR-10 class index [0, 9]

Idempotent: if the NPZ already exists, the script prints a summary and exits.
Set FORCE=1 to overwrite.

Runtime: ~3-4 min on M3 Pro MPS, batch 64.
"""

from __future__ import annotations

import functools
import os
import subprocess
import sys
import time
from pathlib import Path

# 4-thread cap before numpy/torch import (mirrors E3b discipline)
for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(var, "4")

# flush every print
print = functools.partial(print, flush=True)

# concurrency guard
_self_pid = os.getpid()
_others = [int(p) for p in subprocess.run(
    ["pgrep", "-f", "e5a_extract_clip_features.py"],
    capture_output=True, text=True).stdout.split() if int(p) != _self_pid]
if _others:
    print(f"REFUSED TO START: another e5a_extract_clip_features.py is running "
          f"at PID(s) {_others}. kill them first.", file=sys.stderr)
    sys.exit(2)

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
OUT_PATH = HERE.parents[1] / "results" / "cifar10_clip_features.npz"
CIFAR_ROOT = HERE.parents[1] / "results" / "cifar10_cache"

BATCH_SIZE = 64
N_EXPECTED = 50_000        # full CIFAR-10 train
D_FEAT = 512               # CLIP ViT-B/32 projection dim
FORCE = bool(int(os.environ.get("FORCE", "0")))


def summarise_existing():
    if not OUT_PATH.exists():
        return False
    z = np.load(OUT_PATH)
    print(f"  cached file: {OUT_PATH}")
    print(f"  features:       shape={z['features'].shape}  "
          f"dtype={z['features'].dtype}")
    print(f"  features_l2:    shape={z['features_l2'].shape}  "
          f"||x||_2 mean={np.linalg.norm(z['features_l2'], axis=1).mean():.6f}")
    print(f"  labels:         shape={z['labels'].shape}  "
          f"per-class counts: {np.bincount(z['labels'], minlength=10).tolist()}")
    print(f"  size on disk:   {OUT_PATH.stat().st_size / 1e6:.1f} MB")
    return True


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if OUT_PATH.exists() and not FORCE:
        print(f"=== E5a Stage 1a: cached features detected ===")
        summarise_existing()
        print(f"\nSet FORCE=1 to recompute. Skipping.")
        return

    print(f"=== E5a Stage 1a: extract CLIP features for CIFAR-10 train ===")
    t0 = time.time()

    # ---- load CIFAR-10 train + CLIP-style transform
    print("Loading CIFAR-10 train (download if needed)...")
    import torchvision
    import torchvision.transforms as T
    transform = T.Compose([
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711]),
    ])
    ds = torchvision.datasets.CIFAR10(root=str(CIFAR_ROOT),
                                      train=True, download=True, transform=transform)
    if len(ds) != N_EXPECTED:
        raise RuntimeError(f"Expected {N_EXPECTED} CIFAR train images, got {len(ds)}")
    labels = np.array(ds.targets, dtype=np.int64)
    print(f"  {len(ds)} images loaded  ({time.time() - t0:.1f}s)")

    # ---- load CLIP
    print("Loading CLIP ViT-B/32 (use_safetensors=True)...")
    from transformers import CLIPModel
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32",
                                       use_safetensors=True)
    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cpu"))
    model = model.to(device).eval()
    print(f"  CLIP on {device.type}  ({time.time() - t0:.1f}s)")

    # ---- forward in batches
    feats = np.empty((len(ds), D_FEAT), dtype=np.float32)
    n_batches = (len(ds) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"\nForwarding {len(ds)} images in {n_batches} batches of {BATCH_SIZE}...")
    t_fwd_total = 0.0
    with torch.no_grad():
        for b_idx, b in enumerate(range(0, len(ds), BATCH_SIZE)):
            t_b0 = time.time()
            imgs = torch.stack([ds[i][0] for i in range(b, min(b + BATCH_SIZE, len(ds)))]).to(device)
            v_out = model.vision_model(pixel_values=imgs)
            pooled = v_out.pooler_output                # (B, 768)
            f = model.visual_projection(pooled)         # (B, 512)
            if device.type == "mps":
                torch.mps.synchronize()
            f_np = f.cpu().numpy().astype(np.float32)
            feats[b: b + f_np.shape[0]] = f_np
            t_fwd_total += time.time() - t_b0
            if (b_idx + 1) % 50 == 0 or b_idx == n_batches - 1:
                elapsed = time.time() - t0
                est_total = elapsed * n_batches / (b_idx + 1)
                eta = est_total - elapsed
                print(f"  batch {b_idx+1}/{n_batches}  "
                      f"avg/batch={t_fwd_total / (b_idx + 1):.3f}s  "
                      f"elapsed={elapsed:.1f}s  eta={eta:.0f}s")

    print(f"\nAll features computed.  total wall: {time.time() - t0:.1f}s")
    print(f"  features stats: min={feats.min():.4f}  max={feats.max():.4f}  "
          f"mean={feats.mean():.4f}  std={feats.std():.4f}")
    norms = np.linalg.norm(feats, axis=1)
    print(f"  ||x|| stats:    min={norms.min():.4f}  max={norms.max():.4f}  "
          f"mean={norms.mean():.4f}")

    # ---- L2-normalise and save
    print("L2-normalising...")
    feats_l2 = feats / norms[:, None]
    norms_l2 = np.linalg.norm(feats_l2, axis=1)
    print(f"  ||x||_2 after norm: mean={norms_l2.mean():.6f}  "
          f"min={norms_l2.min():.6f}  max={norms_l2.max():.6f}")

    print(f"Saving to {OUT_PATH} ...")
    np.savez_compressed(OUT_PATH,
                        features=feats,
                        features_l2=feats_l2.astype(np.float32),
                        labels=labels)
    print(f"  size: {OUT_PATH.stat().st_size / 1e6:.1f} MB")
    print(f"\nStage 1a DONE in {time.time() - t0:.1f}s total.")


if __name__ == "__main__":
    main()
