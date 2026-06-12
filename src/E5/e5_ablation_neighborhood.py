"""E5 ablation: model-grounded neighborhood CLIP(x_hat) vs true-endpoint CLIP(x).

Tests whether the real-path de-biased A_between(t) decay
could be an artifact of the one-step endpoint estimate x_hat sharpening with t,
rather than genuine branch commitment. We rerun the real-path leg with two
neighborhood constructions and compare:

  N1 (current, model-grounded): neighborhood on CLIP(x_hat_t),
     x_hat_t = Y_t + (1-t) v_theta(Y_t, t).  Depends on t through the model.
  N2 (true-endpoint, model-free): neighborhood on CLIP(x), the real clean image.
     Both x and U = x - z are t-independent, so N2's decomposition is the SAME
     at every t (a flat reference line). We still report it per t for a direct
     side-by-side, computing it once per seed and broadcasting.

Everything else is identical to e5_realdata_validation.py's real-path leg: real
velocity U = x - z, oracle = dataset class coarsened, proxy = k-means on CLIP(x),
random-K balanced null, k=80, 5 seeds, t-grid {0.1..0.6}, K=2 primary (+K=3).

Reading: if N1 decays through zero while N2 is flat, the t-profile is the model's
one-step belief evolution (the intended model-grounded signal), not a property of
the true-endpoint semantics. No training. Run:
  python3 e5_ablation_neighborhood.py
Writes results/e5_ablation_neighborhood.json.
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
                                    un_sq_chunked, SEEDS, K_REAL, T_REAL,
                                    N_REAL, K_NN, N_NULL)
from E7.e7_cifar_fm_train_labelfree import UncondUNet                            # noqa: E402
from E7.e7_cifar_fm_train import select_device, load_cifar                       # noqa: E402
from E7.e7_cifar_fm_diagnostics import load_clip, clip_image_features            # noqa: E402

RES = HERE.parents[1] / "results"


def decompose_set(U, nn, oracle, proxy, kx, K_list):
    """De-biased A_between for oracle/proxy + purity, over a fixed neighborhood nn."""
    Un = U[nn]; Un_sq = un_sq_chunked(Un)
    ab = lambda lab: decompose_fast(Un, lab, U_per_neighbor_sq=Un_sq)["aggregated"]["A_between"]
    out = {"purity": float((kx[nn] == kx[:, None]).mean())}
    for K in K_list:
        nrng = np.random.default_rng(3000 + K)
        null = float(np.mean([ab(rand_label(N_REAL, K, nrng)[nn]) for _ in range(N_NULL)]))
        d_or = ab(oracle[K][nn]) - null
        d_px = ab(proxy[K][nn]) - null
        out[K] = {"deb_oracle": d_or, "deb_proxy": d_px,
                  "recovery": (d_px / d_or if d_or > 1e-4 else np.nan)}
    return out


def main():
    dev = select_device()
    ckpt = torch.load(RES / "e7_uncond_c0_seed0.pt", map_location=dev)
    model = UncondUNet(base_ch=ckpt["config"]["base_ch"]).to(dev)
    model.load_state_dict(ckpt["ema"]); model.eval()
    clip_model, proc = load_clip(dev)
    Xall, Yall = load_cifar(train=False)

    out = {"t": T_REAL, "config": {"seeds": SEEDS, "K_real": K_REAL, "n_real": N_REAL,
                                   "k_nn": K_NN, "n_null": N_NULL},
           "N1": {"purity": {t: [] for t in T_REAL},
                  **{K: {k: {t: [] for t in T_REAL} for k in ("deb_oracle", "deb_proxy", "recovery")}
                     for K in K_REAL}},
           "N2": {"purity": [], **{K: {k: [] for k in ("deb_oracle", "deb_proxy", "recovery")}
                                   for K in K_REAL}}}

    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(Xall.shape[0])[:N_REAL]
        x = Xall[idx]; kx = Yall.numpy()[idx].astype(np.int64)
        torch.manual_seed(seed + 100)
        z = torch.randn(N_REAL, 3, 32, 32)
        U = (x - z).reshape(N_REAL, -1).numpy().astype(np.float64)
        featx = clip_image_features(clip_model, dev, x)                 # true-endpoint features
        proxy = {K: KMeans(n_clusters=K, n_init=10, random_state=0).fit_predict(featx).astype(np.int64)
                 for K in K_REAL}
        oracle = {K: coarsen(kx, K) for K in K_REAL}

        # N2: true-endpoint neighborhood, t-independent -> compute once per seed
        nn2 = NearestNeighbors(n_neighbors=K_NN, algorithm="brute").fit(featx).kneighbors(featx)[1]
        r2 = decompose_set(U, nn2, oracle, proxy, kx, K_REAL)
        out["N2"]["purity"].append(r2["purity"])
        for K in K_REAL:
            for k in ("deb_oracle", "deb_proxy", "recovery"):
                out["N2"][K][k].append(r2[K][k])

        # N1: model-grounded x_hat neighborhood, per t
        for t in T_REAL:
            Yt = (1 - t) * z + t * x
            x_hat = (Yt + (1 - t) * velocity(model, dev, Yt, t)).clamp(-1, 1)
            feat = clip_image_features(clip_model, dev, x_hat)
            nn1 = NearestNeighbors(n_neighbors=K_NN, algorithm="brute").fit(feat).kneighbors(feat)[1]
            r1 = decompose_set(U, nn1, oracle, proxy, kx, K_REAL)
            out["N1"]["purity"][t].append(r1["purity"])
            for K in K_REAL:
                for k in ("deb_oracle", "deb_proxy", "recovery"):
                    out["N1"][K][k][t].append(r1[K][k])
        print(f"  [ablation] seed {seed} done", flush=True)

    def ms(l):
        a = np.array(l, float); return float(np.nanmean(a)), float(np.nanstd(a, ddof=1) if np.sum(~np.isnan(a)) > 1 else 0.0)
    print("\n  N1 (CLIP(x_hat), model-grounded) vs N2 (CLIP(x), true endpoint), K=2:")
    print(f"  {'t':>5} {'N1 deb_or':>11} {'N1 recov':>10} | {'N2 deb_or':>11} {'N2 recov':>10}")
    n2o = ms(out["N2"][2]["deb_oracle"]); n2r = ms(out["N2"][2]["recovery"])
    for t in T_REAL:
        o1 = ms(out["N1"][2]["deb_oracle"][t]); r1 = ms(out["N1"][2]["recovery"][t])
        print(f"  {t:>5.2f} {o1[0]:>+11.2f} {r1[0]:>10.2f} | {n2o[0]:>+11.2f} {n2r[0]:>10.2f}  (N2 t-independent)")

    out_ser = {"t": T_REAL, "config": out["config"],
               "N1": {"purity": out["N1"]["purity"],
                      **{str(K): out["N1"][K] for K in K_REAL}},
               "N2": {"purity": out["N2"]["purity"],
                      **{str(K): out["N2"][K] for K in K_REAL}}}
    with open(RES / "e5_ablation_neighborhood.json", "w") as f:
        json.dump(out_ser, f, indent=2)
    print(f"\nwrote {RES / 'e5_ablation_neighborhood.json'}")


if __name__ == "__main__":
    main()
