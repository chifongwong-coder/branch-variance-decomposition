"""E5 robustness across uncond_c0 training checkpoints (robustness A).

Reruns the E5 model-grounded real-path leg (K=2) on the unconditional c0 model at
several training steps, to show the ambiguity-to-commitment profile is stable across
training amount rather than an artifact of the final checkpoint. Forward-pass only,
no training.

Run:  python3 e5_checkpoint_robustness.py
Writes results/e5_checkpoint_robustness.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from core.branch_decomp import decompose_fast                                       # noqa: E402
from E5.e5_realdata_validation import (coarsen, rand_label, velocity,            # noqa: E402
                                    un_sq_chunked, K_NN, N_NULL, T_REAL, N_REAL)
from E7.e7_cifar_fm_train_labelfree import UncondUNet                            # noqa: E402
from E7.e7_cifar_fm_train import select_device, load_cifar                       # noqa: E402
from E7.e7_cifar_fm_diagnostics import load_clip, clip_image_features            # noqa: E402

RES = HERE.parents[1] / "results"
STEPS = [10000, 30000, 50000]
SEEDS = [0, 1, 2]
K = 2


def realpath_k2(model, dev, clip_model, Xall, Yall):
    out = {key: {t: [] for t in T_REAL} for key in ("deb_oracle", "recovery", "purity")}
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(Xall.shape[0])[:N_REAL]
        x = Xall[idx]; kx = Yall.numpy()[idx].astype(np.int64)
        torch.manual_seed(seed + 100)
        z = torch.randn(N_REAL, 3, 32, 32)
        U = (x - z).reshape(N_REAL, -1).numpy().astype(np.float64)
        featx = clip_image_features(clip_model, dev, x)
        proxy = KMeans(n_clusters=K, n_init=10, random_state=0).fit_predict(featx).astype(np.int64)
        oracle = coarsen(kx, K)
        for t in T_REAL:
            Yt = (1 - t) * z + t * x
            x_hat = (Yt + (1 - t) * velocity(model, dev, Yt, t)).clamp(-1, 1)
            feat = clip_image_features(clip_model, dev, x_hat)
            nn = NearestNeighbors(n_neighbors=K_NN, algorithm="brute").fit(feat).kneighbors(feat)[1]
            Un = U[nn]; Un_sq = un_sq_chunked(Un)
            ab = lambda lab: decompose_fast(Un, lab, U_per_neighbor_sq=Un_sq)["aggregated"]["A_between"]
            nrng = np.random.default_rng(2000 + seed * 10 + K)
            null = float(np.mean([ab(rand_label(N_REAL, K, nrng)[nn]) for _ in range(N_NULL)]))
            d_or = ab(oracle[nn]) - null
            d_px = ab(proxy[nn]) - null
            out["deb_oracle"][t].append(d_or)
            out["recovery"][t].append(d_px / d_or if d_or > 1e-4 else np.nan)
            out["purity"][t].append(float((kx[nn] == kx[:, None]).mean()))
    return out


def main():
    dev = select_device()
    clip_model, proc = load_clip(dev)
    Xall, Yall = load_cifar(train=False)
    by_step = {}
    for S in STEPS:
        ck = torch.load(RES / f"e7_uncond_c0_seed0_step{S}.pt", map_location=dev)
        m = UncondUNet(base_ch=ck["config"]["base_ch"]).to(dev)
        m.load_state_dict(ck["ema"]); m.eval()
        by_step[S] = realpath_k2(m, dev, clip_model, Xall, Yall)
        print(f"  step {S} done", flush=True)

    def ms(l):
        a = np.array(l, float)
        return float(np.nanmean(a)), float(np.nanstd(a, ddof=1) if np.sum(~np.isnan(a)) > 1 else 0.0)
    print(f"\n  {'step':>7} {'t':>5} {'deb_or':>9} {'recov':>8} {'purity':>8}")
    for S in STEPS:
        for t in T_REAL:
            o = ms(by_step[S]["deb_oracle"][t]); r = ms(by_step[S]["recovery"][t]); p = ms(by_step[S]["purity"][t])
            print(f"  {S:>7} {t:>5.2f} {o[0]:>+9.2f} {r[0]:>8.2f} {p[0]:>8.3f}")

    ser = {"steps": STEPS, "t": T_REAL, "config": {"seeds": SEEDS, "K": K, "N": N_REAL},
           "by_step": {str(S): {k: {str(t): by_step[S][k][t] for t in T_REAL}
                                for k in ("deb_oracle", "recovery", "purity")} for S in STEPS}}
    with open(RES / "e5_checkpoint_robustness.json", "w") as f:
        json.dump(ser, f, indent=2)
    print(f"\nwrote {RES / 'e5_checkpoint_robustness.json'}")


if __name__ == "__main__":
    main()
