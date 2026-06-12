"""Does an INDEPENDENT kNN estimate of A_between(t) predict the realized E7
conditioning gain Delta_cross(t)?

Published E7 compares Delta_cross to S_post, the conditional model's OWN
posterior-weighted between-class spread, which is partly self-referential. Here A_between(t) is estimated directly from the velocity
scatter with the verified kNN BVD estimator (cardinality-de-biased, over the
true 10-class partition, on a CLIP(x_hat) neighborhood, exactly as the E5
model-grounded leg), with no reference to the conditional model's spread. We
report it at several bandwidths k (10-class resolution improves with neighbors-
per-branch) and compare each to the realized cross-model gain
Delta_cross = MSE_uncond - MSE_cond_true. Agreement in the ambiguity window
turns E7 from a consistency check into an independent prediction.

Forward-pass only (2 forwards per t: uncond + true-class cond). Needs MPS.
Writes results/expE7a_independent_abetween.json.
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
from E7.e7_cifar_fm_train import select_device, load_cifar, CondUNet        # noqa: E402
from E7.e7_cifar_fm_train_labelfree import UncondUNet                       # noqa: E402
from E7.e7_cifar_fm_diagnostics import load_clip, clip_image_features       # noqa: E402

RES = HERE.parents[1] / "results"
SEEDS = [0, 1, 2]
N = 2000
T_GRID = [round(float(x), 3) for x in np.linspace(0.05, 0.95, 13)]
N_CLASSES = 10
K_NN_LIST = [80, 200]
N_NULL = 4
BATCH = 250
AMBIG = 0.5


@torch.no_grad()
def v_uncond(model, dev, xt, t):
    out = []
    for b in range(0, xt.shape[0], BATCH):
        xb = xt[b:b + BATCH].to(dev)
        tb = torch.full((xb.shape[0],), float(t), device=dev)
        out.append(model(xb, tb).cpu())
    return torch.cat(out, 0)


@torch.no_grad()
def v_cond_class_vec(model, dev, xt, t, kvec):
    out = []
    for b in range(0, xt.shape[0], BATCH):
        xb = xt[b:b + BATCH].to(dev)
        tb = torch.full((xb.shape[0],), float(t), device=dev)
        kb = kvec[b:b + BATCH].to(dev)
        out.append(model(xb, tb, kb).cpu())
    return torch.cat(out, 0)


def sumsq(a):
    return a.reshape(a.shape[0], -1).pow(2).sum(1)


def chunk_for(k, d=3072, budget=8e8):
    return max(40, int(budget / (k * d * 8)))


def main():
    dev = select_device()
    cu = torch.load(RES / "e7_uncond_c0_seed0.pt", map_location=dev)
    vu = UncondUNet(base_ch=cu["config"]["base_ch"]).to(dev)
    vu.load_state_dict(cu["ema"]); vu.eval()
    cc = torch.load(RES / "e7_c0_seed0.pt", map_location=dev)
    vc = CondUNet(base_ch=cc["config"]["base_ch"],
                  n_classes=cc["config"]["n_classes"]).to(dev)
    vc.load_state_dict(cc["ema"]); vc.eval()
    clip_model, _ = load_clip(dev)
    kmax = max(K_NN_LIST)
    print(f"[E7a] device={dev.type} N={N} seeds={SEEDS} t={len(T_GRID)}pts "
          f"k-list={K_NN_LIST}", flush=True)

    Xall, Yall = load_cifar(train=False)
    out = {"t": T_GRID, "config": {"seeds": SEEDS, "N": N, "k_nn_list": K_NN_LIST,
                                   "n_null": N_NULL, "n_classes": N_CLASSES,
                                   "ambig_t_max": AMBIG},
           "abetween_indep": {str(k): {str(t): [] for t in T_GRID} for k in K_NN_LIST},
           "delta_cross": {str(t): [] for t in T_GRID}}

    for seed in SEEDS:
        t0 = time.time()
        rng = np.random.default_rng(seed)
        idx = rng.permutation(Xall.shape[0])[:N]
        x = Xall[idx]
        ktrue = Yall.numpy()[idx].astype(np.int64)
        ktrue_t = torch.from_numpy(ktrue)
        torch.manual_seed(seed + 100)
        z = torch.randn(N, 3, 32, 32)
        U = (x - z).reshape(N, -1).numpy().astype(np.float64)   # (N,3072)
        for t in T_GRID:
            Yt = (1 - t) * z + t * x
            vu_t = v_uncond(vu, dev, Yt, t)
            x_hat = (Yt + (1 - t) * vu_t).clamp(-1, 1)
            feat = clip_image_features(clip_model, dev, x_hat)   # (N,512)
            nn_full = C.knn_indices(feat, kmax)
            for k in K_NN_LIST:
                # re-randomize the cardinality null per (seed, t) for clean stats
                null_seed = seed * 100000 + int(round(t * 1000))
                ab = C.abetween_debiased(U, nn_full[:, :k], ktrue, n_null=N_NULL,
                                         n_classes=N_CLASSES, chunk=chunk_for(k),
                                         seed=null_seed)["A_between_deb"]
                out["abetween_indep"][str(k)][str(t)].append(ab)
            vtrue = v_cond_class_vec(vc, dev, Yt, t, ktrue_t)
            mu = float(sumsq((x - z) - vu_t).mean())   # MSE_uncond
            mt = float(sumsq((x - z) - vtrue).mean())  # MSE_cond_true
            out["delta_cross"][str(t)].append(mu - mt)
        print(f"[E7a] seed {seed} done in {time.time()-t0:.0f}s", flush=True)
        C.save_json("expE7a_independent_abetween.json", out)   # incremental

    ts = np.array(T_GRID); msk = ts <= AMBIG
    dcm = np.array([np.mean(out["delta_cross"][str(t)]) for t in T_GRID])
    out["summary"] = {}
    for k in K_NN_LIST:
        abm = np.array([np.mean(out["abetween_indep"][str(k)][str(t)]) for t in T_GRID])
        cw, rw = C.cosine_ratio(abm[msk], dcm[msk])
        out["summary"][str(k)] = {"cosine_tle05": cw,
                                  "ratio_dcross_over_abetween_tle05": rw}
        print(f"[E7a] k={k:4d}: cosine(A_between_indep, Delta_cross | t<=0.5)={cw:.4f}  "
              f"ratio={rw:.3f}", flush=True)
    C.save_json("expE7a_independent_abetween.json", out)
    print("[E7a] done", flush=True)


if __name__ == "__main__":
    main()
