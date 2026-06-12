"""E7: data-side branch-variance diagnostics for a CIFAR-10 coupling.

For a given coupling and seed, this computes on held-out (z, x) pairs:
  - the branch-variance decomposition curves A_v(t), A_within(t), A_between(t),
    R_switch(t), with the CIFAR-10 class as the branch variable, estimated in
    CLIP feature space (reuses branch_decomp.decompose);
  - mean raw transport cost ||z - x||^2 under the run's coupling
    (recomputed on held-out pairs for the Pareto x-axis).

Writes results/e7_diag_<coupling>_seed<seed>.json.

Reuses: branch_decomp.py (estimator), the CLIP ViT-B/32 pipeline (as in E5),
faiss for the kNN graph. Device auto-selected (CUDA -> MPS -> CPU).

  python3 e7_cifar_fm_diagnostics.py --coupling c0 --seed 0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.branch_decomp import decompose
from E7.e7_cifar_fm_train import (load_cifar, select_device,
                               build_coupling)

HERE = Path(__file__).resolve().parent
RES = HERE.parents[1] / "results"

CIFAR_CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog",
                 "frog", "horse", "ship", "truck"]
T_GRID = [round(x, 3) for x in np.linspace(0.05, 0.95, 13)]
K_NN = 80


# ---------------------------------------------------------------------------
# CLIP (feature extraction for the BVD estimator)
# ---------------------------------------------------------------------------
def load_clip(dev):
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32",
                                      use_safetensors=True).to(dev).eval()
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, proc


@torch.no_grad()
def clip_image_features(model, dev, imgs):
    """imgs: (N,3,32,32) in [-1,1]. Returns (N,512) L2-normalised CLIP features.
    Bicubic-upscales to 224 and applies CLIP normalization (as in E5)."""
    import torch.nn.functional as F
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=dev)[None, :, None, None]
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=dev)[None, :, None, None]
    feats = []
    for b in range(0, imgs.shape[0], 256):
        x = imgs[b:b + 256].to(dev)
        x = (x + 1) / 2                          # [-1,1] -> [0,1]
        x = F.interpolate(x, size=224, mode="bicubic", align_corners=False)
        x = (x - mean) / std
        f = model.get_image_features(pixel_values=x)
        f = f if isinstance(f, torch.Tensor) else f.pooler_output
        f = f / f.norm(dim=-1, keepdim=True)
        feats.append(f.float().cpu().numpy())
    return np.concatenate(feats, 0).astype(np.float32)


# ---------------------------------------------------------------------------
# BVD curves in CLIP feature space on held-out pairs under the model coupling
# ---------------------------------------------------------------------------
def bvd_curves(clip_model, dev, args, rng):
    """Build held-out (z, x) pairs under the run's coupling, map x to CLIP
    feature space, and run the within/between decomposition with the CIFAR
    class as the branch label, on one global kNN graph per t."""
    from sklearn.neighbors import NearestNeighbors
    Xv, Yv = load_cifar(train=False)
    sub = rng.permutation(Xv.shape[0])[: args.n_pairs]
    x = Xv[sub]
    kx = Yv.numpy()[sub]
    feat = clip_image_features(clip_model, dev, x)        # (N,512) target feats
    N, d = feat.shape

    coupling = build_coupling(args.coupling, lambda_sem=args.lambda_sem)
    # Coupling-induced pairing in pixel space (the training coupling), computed
    # once and held fixed across t so that only the interpolant geometry varies.
    zp = rng.standard_normal((N, 3 * 32 * 32)).astype(np.float32)
    Xf = x.reshape(N, -1).numpy()
    c = kx[rng.permutation(N)]
    col, cond = coupling(zp, Xf, kx, c)
    fp = feat[col]                                          # (N,d) paired target feats
    # Scale-matched feature-space noise, following the E5 protocol
    # (e5a_diagnostic.run_one_t): Z ~ N(0, I/d), drawn once and shared across t.
    Zf = (rng.standard_normal(fp.shape) / np.sqrt(d)).astype(np.float32)
    U = (fp - Zf).astype(np.float64)                        # FM velocity x - z
    curves = {m: [] for m in ("A_v", "A_within", "A_between", "R_switch")}
    for t in T_GRID:
        # Feature-space FM interpolant Y = (1-t) Z + t X. Keeping the noise term
        # is what makes the kNN neighborhoods reorganize as t sweeps noise to
        # data, which a t-scaled target alone cannot do (scaling is kNN-invariant).
        Yt = ((1 - t) * Zf + t * fp).astype(np.float32)
        knn = NearestNeighbors(n_neighbors=K_NN, algorithm="brute",
                               metric="euclidean").fit(Yt)
        _, nn = knn.kneighbors(Yt)
        s = decompose(U[nn], cond[nn])
        agg = s["aggregated"]
        curves["A_v"].append(agg["A_v"])
        curves["A_within"].append(agg["A_within"])
        curves["A_between"].append(agg["A_between"])
        curves["R_switch"].append(agg["A_between"] / max(agg["A_v"], 1e-12))
    return {"t_grid": T_GRID, **curves}


# ---------------------------------------------------------------------------
# Transport cost
# ---------------------------------------------------------------------------
def transport_cost(args, rng, n=4096):
    Xv, Yv = load_cifar(train=False)
    sub = rng.permutation(Xv.shape[0])[:n]
    x = Xv[sub].reshape(n, -1).numpy()
    kx = Yv.numpy()[sub]
    z = rng.standard_normal((n, 3 * 32 * 32)).astype(np.float32)
    c = kx[rng.permutation(n)]
    col, _ = build_coupling(args.coupling, lambda_sem=args.lambda_sem)(z, x, kx, c)
    return float(((z - x[col]) ** 2).sum(1).mean())


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="E7 diagnostics")
    p.add_argument("--coupling", required=True,
                   choices=["c0", "c1", "c3", "c3inf"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lambda-sem", type=float, default=10.0, dest="lambda_sem")
    p.add_argument("--n-pairs", type=int, default=10000, dest="n_pairs")
    p.add_argument("--step", type=int, default=None,
                   help="tag the output json with a training-step suffix "
                        "(for saturation sweeps); default writes the unsuffixed json")
    args = p.parse_args()

    dev = select_device()
    rng = np.random.default_rng(args.seed + 777)
    print(f"E7 diagnostics  coupling={args.coupling} seed={args.seed}"
          f" device={dev.type} step={args.step}")

    clip_model, _ = load_clip(dev)

    print("  computing BVD curves (CLIP feature space)...")
    bvd = bvd_curves(clip_model, dev, args, rng)

    print("  transport cost...")
    tcost = transport_cost(args, rng)

    out = {
        "coupling": args.coupling, "seed": args.seed,
        "step": args.step,
        "bvd": bvd,
        "mean_transport_cost": tcost,
    }
    suffix = f"_step{args.step}" if args.step is not None else ""
    jpath = RES / f"e7_diag_{args.coupling}_seed{args.seed}{suffix}.json"
    with open(jpath, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  transport = {tcost:.2f}")
    print(f"  wrote {jpath}")


if __name__ == "__main__":
    main()
