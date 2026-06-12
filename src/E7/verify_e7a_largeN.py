"""E7a follow-up: does the independent-A_between magnitude attenuation shrink as
the sample size N grows (so the kNN can resolve 10 classes locally)? Re-runs the
E7a comparison at N=4000 with k in {200,400} over the ambiguity window, to be read
against the overnight N=2000 result (k=200: cosine 0.946, ratio 0.565). If the
ratio climbs toward 1 at N=4000 / larger k, the attenuation is a finite-resolution
(sample-size) effect, not a fundamental one.

MPS. Light: ~10 min (no k=800). Writes results/verify_e7a_largeN.json.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from core import expE_common as C                                                  # noqa: E402
from E7 import expE7a_independent_abetween as E7a                                # noqa: E402
from E7.e7_cifar_fm_train import select_device, load_cifar, CondUNet        # noqa: E402
from E7.e7_cifar_fm_train_labelfree import UncondUNet                       # noqa: E402
from E7.e7_cifar_fm_diagnostics import load_clip, clip_image_features       # noqa: E402

RES = HERE.parents[1] / "results"
N = 4000
K_LIST = [200, 400]
T_GRID = [0.1, 0.2, 0.3, 0.4, 0.5]      # ambiguity window only
SEEDS = [0, 1]
N_NULL = 3


def main():
    t_start = time.time()
    dev = select_device()
    cu = torch.load(RES / "e7_uncond_c0_seed0.pt", map_location=dev)
    vu = UncondUNet(base_ch=cu["config"]["base_ch"]).to(dev)
    vu.load_state_dict(cu["ema"]); vu.eval()
    cc = torch.load(RES / "e7_c0_seed0.pt", map_location=dev)
    vc = CondUNet(base_ch=cc["config"]["base_ch"],
                  n_classes=cc["config"]["n_classes"]).to(dev)
    vc.load_state_dict(cc["ema"]); vc.eval()
    clip_model, _ = load_clip(dev)
    print(f"[verify] N={N} k={K_LIST} t={T_GRID} seeds={SEEDS}", flush=True)

    Xall, Yall = load_cifar(train=False)
    ab = {k: {t: [] for t in T_GRID} for k in K_LIST}
    dc = {t: [] for t in T_GRID}
    for seed in SEEDS:
        t0 = time.time()
        rng = np.random.default_rng(seed)
        idx = rng.permutation(Xall.shape[0])[:N]
        x = Xall[idx]; kt = Yall.numpy()[idx].astype(np.int64)
        ktt = torch.from_numpy(kt)
        torch.manual_seed(seed + 100)
        z = torch.randn(N, 3, 32, 32)
        U = (x - z).reshape(N, -1).numpy().astype(np.float64)
        for t in T_GRID:
            Yt = (1 - t) * z + t * x
            vu_t = E7a.v_uncond(vu, dev, Yt, t)
            xhat = (Yt + (1 - t) * vu_t).clamp(-1, 1)
            feat = clip_image_features(clip_model, dev, xhat)
            nn = C.knn_indices(feat, max(K_LIST))
            for k in K_LIST:
                ab[k][t].append(C.abetween_debiased(
                    U, nn[:, :k], kt, n_null=N_NULL, n_classes=10,
                    chunk=E7a.chunk_for(k), seed=seed * 99999 + int(t * 1000) + k
                )["A_between_deb"])
            vtrue = E7a.v_cond_class_vec(vc, dev, Yt, t, ktt)
            dc[t].append(float(E7a.sumsq((x - z) - vu_t).mean())
                         - float(E7a.sumsq((x - z) - vtrue).mean()))
        print(f"[verify] seed {seed} done in {time.time()-t0:.0f}s", flush=True)

    dcm = np.array([np.mean(dc[t]) for t in T_GRID])
    out = {"config": {"N": N, "k_list": K_LIST, "t_grid": T_GRID,
                      "seeds": SEEDS, "n_null": N_NULL},
           "baseline_N2000_k200": {"cosine": 0.946, "ratio": 0.565},
           "results": {}}
    print(f"\n[verify] N={N} independent A_between vs Delta_cross (ambiguity window):",
          flush=True)
    for k in K_LIST:
        abm = np.array([np.mean(ab[k][t]) for t in T_GRID])
        cw, rw = C.cosine_ratio(abm, dcm)
        out["results"][str(k)] = {"cosine": cw, "ratio": rw}
        print(f"[verify]   k={k}: cosine={cw:.4f}  ratio={rw:.3f}", flush=True)
    C.save_json("verify_e7a_largeN.json", out)
    print(f"[verify] DONE in {(time.time()-t_start)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
