"""Paper experiment E5: model-grounded real-path branch audit on CIFAR-10.

5 seeds, de-biased (random-K null) signals. (This leg reuses the shared CIFAR
Flow-Matching training code, whose files keep the e7_ prefix; the paper calls this
experiment E5.) The model-free recovery-vs-bandwidth leg lives separately in
src/E5b/expE5b_recovery_bandwidth.py, which feeds figure panel (a).

Model-grounded real-path audit: unconditional C0 model; real x (oracle =
  dataset class), z, U = x - z; Y_t=(1-t)z+t x; one-step endpoint
  x_hat=Y_t+(1-t)v_theta(Y_t,t); kNN on CLIP(x_hat); decompose the real U over
  oracle / proxy (k-means on CLIP(x)) / null; report per-t de-biased A_between,
  recovery, neighborhood purity, mean +/- SD over seeds.

Compute-only (no plotting). Saves the FULL raw per-seed / per-t data so the
figure can be regenerated without re-running. Run:
  python3 e5_realdata_validation.py
Writes results/e5_realdata_validation.json (raw); plot with
  python3 plot_e5_realdata_validation.py
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
from E7.e7_cifar_fm_train_labelfree import UncondUNet                            # noqa: E402
from E7.e7_cifar_fm_train import select_device, load_cifar                       # noqa: E402
from E7.e7_cifar_fm_diagnostics import load_clip, clip_image_features            # noqa: E402

RES = HERE.parents[1] / "results"
SEEDS = [0, 1, 2, 3, 4]
K_REAL = [2, 3]          # real-path leg: K=2/3 headline (K=10 is cardinality-dominated)
VEHICLES = {0, 1, 8, 9}
K3_MAP = {0: 0, 1: 0, 8: 0, 9: 0, 2: 1, 6: 1, 3: 2, 4: 2, 5: 2, 7: 2}

N_REAL = 2000
T_REAL = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
# Single fixed bandwidth, used for every K in K_REAL. At K=2 this is 40
# neighbors/branch (the value behind the figure panels). The paper's K=3
# real-path table column instead holds the branch count fixed at 40
# neighbors/branch, i.e. k = 40*K = 120 at K=3 (k=80 here gives only ~27/branch).
# To reproduce that column, set the bandwidth per K, e.g. k_for(K) = 40 * K,
# and build the kNN graph inside the `for K in K_REAL` loop.
K_NN = 80
N_NULL = 6


def coarsen(cls, K):
    if K == 2:
        return np.array([0 if c in VEHICLES else 1 for c in cls], dtype=np.int64)
    if K == 3:
        return np.array([K3_MAP[c] for c in cls], dtype=np.int64)
    return np.asarray(cls, dtype=np.int64)    # K=10: the full true-class set (identity)


def rand_label(N, K, rng):
    lab = np.tile(np.arange(K), N // K + 1)[:N]; rng.shuffle(lab)
    return lab.astype(np.int64)


def un_sq_chunked(Un, chunk=500):
    """Per-neighbor squared norm sum(-1), computed in row chunks to cap the
    transient memory (Un can be ~4 GB at N=2000, d=3072)."""
    out = np.empty(Un.shape[:2], dtype=np.float64)
    for c in range(0, Un.shape[0], chunk):
        out[c:c + chunk] = (Un[c:c + chunk] ** 2).sum(-1)
    return out


@torch.no_grad()
def velocity(model, dev, xt, t, batch=250):
    out = []
    for b in range(0, xt.shape[0], batch):
        xb = xt[b:b + batch].to(dev)
        tb = torch.full((xb.shape[0],), float(t), device=dev)
        out.append(model(xb, tb).cpu())
    return torch.cat(out, 0)


def run_realpath(model, dev, clip_model, proc):
    out = {"t": T_REAL}
    for K in K_REAL:
        out[K] = {key: {t: [] for t in T_REAL}
                  for key in ("deb_oracle", "deb_proxy", "recovery")}
    out["purity"] = {t: [] for t in T_REAL}
    Xall, Yall = load_cifar(train=False)
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(Xall.shape[0])[:N_REAL]
        x = Xall[idx]; kx = Yall.numpy()[idx].astype(np.int64)
        torch.manual_seed(seed + 100)
        z = torch.randn(N_REAL, 3, 32, 32)
        U = (x - z).reshape(N_REAL, -1).numpy().astype(np.float64)
        featx = clip_image_features(clip_model, dev, x)
        proxy = {K: KMeans(n_clusters=K, n_init=10, random_state=0).fit_predict(featx).astype(np.int64)
                 for K in K_REAL}
        oracle = {K: coarsen(kx, K) for K in K_REAL}
        for t in T_REAL:
            Yt = (1 - t) * z + t * x
            x_hat = (Yt + (1 - t) * velocity(model, dev, Yt, t)).clamp(-1, 1)
            feat = clip_image_features(clip_model, dev, x_hat)
            nn = NearestNeighbors(n_neighbors=K_NN, algorithm="brute").fit(feat).kneighbors(feat)[1]
            Un = U[nn]; Un_sq = un_sq_chunked(Un)
            ab = lambda lab: decompose_fast(Un, lab, U_per_neighbor_sq=Un_sq)["aggregated"]["A_between"]
            out["purity"][t].append(float((kx[nn] == kx[:, None]).mean()))
            for K in K_REAL:
                nrng = np.random.default_rng(2000 + seed * 10 + K)
                null = np.mean([ab(rand_label(N_REAL, K, nrng)[nn]) for _ in range(N_NULL)])
                d_or = ab(oracle[K][nn]) - null
                d_px = ab(proxy[K][nn]) - null
                out[K]["deb_oracle"][t].append(d_or)
                out[K]["deb_proxy"][t].append(d_px)
                out[K]["recovery"][t].append(d_px / d_or if d_or > 1e-4 else np.nan)
        print(f"  [realpath] seed {seed} done", flush=True)
    return out


def ms(lst):
    a = np.array(lst, dtype=float)
    return float(np.nanmean(a)), float(np.nanstd(a))


def main():
    print("=== E5 model-grounded real-path audit (uc0) ===")
    dev = select_device()
    ckpt = torch.load(RES / "e7_uncond_c0_seed0.pt", map_location=dev)
    model = UncondUNet(base_ch=ckpt["config"]["base_ch"]).to(dev)
    model.load_state_dict(ckpt["ema"]); model.eval()
    clip_model, proc = load_clip(dev)
    real_res = run_realpath(model, dev, clip_model, proc)
    print(f"  {'t':>5} {'purity':>8} {'deb_or K2':>11} {'recov K2':>10}")
    for t in T_REAL:
        print(f"  {t:>5.2f} {ms(real_res['purity'][t])[0]:>8.3f} "
              f"{ms(real_res[2]['deb_oracle'][t])[0]:>+11.3f} {ms(real_res[2]['recovery'][t])[0]:>10.2f}")

    # save the FULL raw per-seed / per-t data (no summarizing) so the figure and
    # any later analysis regenerate without re-running the compute.
    cfg = {"seeds": SEEDS, "K_real": K_REAL, "k_nn": K_NN, "n_null": N_NULL,
           "n_real": N_REAL, "t_real": T_REAL, "checkpoint": "e7_uncond_c0_seed0.pt"}
    out = {"config": cfg,
           "realpath": {"t": T_REAL,
                        "purity": {str(t): real_res["purity"][t] for t in T_REAL},
                        **{str(K): {k: {str(t): real_res[K][k][t] for t in T_REAL}
                                    for k in ("deb_oracle", "deb_proxy", "recovery")}
                           for K in K_REAL}}}
    with open(RES / "e5_realdata_validation.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {RES / 'e5_realdata_validation.json'} (raw per-seed data)")
    print("plot with: python3 plot_e5_realdata_validation.py")


if __name__ == "__main__":
    main()
