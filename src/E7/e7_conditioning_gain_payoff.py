"""Paper experiment E7: branch-conditioning gain as a real-model payoff for BVD.

Tests the population identity A_v(t) - A_within(t) = A_between(t): the velocity-MSE
recovered by conditioning a trained Flow-Matching model on the class. Forward-pass
only, no training. We report three curves plus a
within-conditional ablation that removes the cross-model excess-risk confound.

Models (EMA weights):
  v_uncond : e7_uncond_c0_seed0.pt (UncondUNet, no class input)
  v_cond   : e7_c0_seed0.pt        (CondUNet, class-conditional, 10 classes)
Both base_ch=96, 50k steps, C0 (independent) coupling, same architecture up to the
class embedding.

Per t (Y_t=(1-t)Z+tX, U=X-Z):
  MSE_uncond(t)   = E||U - v_uncond(Y_t,t)||^2
  MSE_cond_true(t)= E||U - v_cond(Y_t,t,k_true)||^2
  MSE_cond_avg(t) = E||U - (1/K) sum_k v_cond(Y_t,t,k)||^2     (label-averaged)
  S_uni(t)        = E (1/K) sum_k ||v_cond(Y_t,t,k) - v_bar||^2 (uniform class spread)
  S_post(t)       = E sum_k p(k|Y_t) ||v_cond(Y_t,t,k) - v_bar_post||^2 (BVD-strict)
                    with p(k|Y_t) a CLIP zero-shot posterior on the one-step endpoint
                    x_hat = Y_t + (1-t) v_uncond(Y_t,t) (label-free). S_post is the
                    posterior-weighted A_between robustness; CLIP enters only here.
Derived: Delta_cross = MSE_uncond - MSE_cond_true   (deployment gain, cross-model)
         Delta_within = MSE_cond_avg - MSE_cond_true (same-checkpoint gain, confound-free)

Reading: Delta_within and S_uni are both uniform-weighted, same conditional model;
their agreement is the clean payoff signal (it isolates the conditioning gain within
one network). S_uni is a class-steering / CFG-like surface, NOT the strict
posterior-weighted A_between (that distinction is stated in the paper). All MSEs are
summed over the 3072 velocity dimensions and averaged over samples, so every curve
is in the same units.

The checkpoint coupling is selectable with --coupling (default c0): it loads the
matching label-free / conditional pair e7_uncond_<coupling>_seed0.pt and
e7_<coupling>_seed0.pt. The C0 invocation is unchanged and writes the canonical
results/e7_conditioning_gain_payoff.json; other couplings write a suffixed file
(e.g. e7_conditioning_gain_payoff_c1.json).

Run:  python3 e7_conditioning_gain_payoff.py [--coupling c0|c1|c3|c3inf]
Writes results/e7_conditioning_gain_payoff[_<coupling>].json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from E7.e7_cifar_fm_train import select_device, load_cifar, CondUNet                # noqa: E402
from E7.e7_cifar_fm_train_labelfree import UncondUNet                              # noqa: E402
from E7.e7_cifar_fm_diagnostics import load_clip, clip_image_features, CIFAR_CLASSES  # noqa: E402
from E7.e7_cifar_fm_couplings import build_coupling                                # noqa: E402

CBATCH = 128   # minibatch size for the in-distribution coupling (matches training)


def apply_indist_coupling(z, x, ktrue, coup, rng):
    """Re-pair (z, x) with the training coupling, in minibatches of CBATCH, so the
    conditional model is queried on its own training joint. Returns paired targets
    x_p (source order), the conditioning label per source, and U = x_p - z. This
    reproduces e7_cifar_fm_train.py exactly (col reorders targets, cond is the label
    the model was trained to condition on). For c0 col is the identity, so this is a
    no-op that must reproduce the independent-pairing numbers."""
    fn = build_coupling(coup)
    N = z.shape[0]
    x_p = x.clone()
    cond = ktrue.copy()
    zf_all = z.reshape(N, -1).numpy()
    xf_all = x.reshape(N, -1).numpy()
    for s in range(0, N, CBATCH):
        e = min(s + CBATCH, N)
        kx = ktrue[s:e]
        c = kx[rng.permutation(e - s)]                       # desired class, same hist
        col, cnd = fn(zf_all[s:e], xf_all[s:e], kx, c)        # local indices
        x_p[s:e] = x[s:e][col]
        cond[s:e] = cnd
    return x_p, cond.astype(np.int64), x_p - z

RES = HERE.parents[1] / "results"
SEEDS = [0, 1, 2]
N = 2000
T_GRID = [round(x, 3) for x in np.linspace(0.05, 0.95, 13)]
N_CLASSES = 10
BATCH = 250


@torch.no_grad()
def v_uncond(model, dev, xt, t):
    out = []
    for b in range(0, xt.shape[0], BATCH):
        xb = xt[b:b + BATCH].to(dev)
        tb = torch.full((xb.shape[0],), float(t), device=dev)
        out.append(model(xb, tb).cpu())
    return torch.cat(out, 0)


@torch.no_grad()
def v_cond_class(model, dev, xt, t, k):
    """Conditional velocity with every sample set to class k."""
    out = []
    for b in range(0, xt.shape[0], BATCH):
        xb = xt[b:b + BATCH].to(dev)
        tb = torch.full((xb.shape[0],), float(t), device=dev)
        kb = torch.full((xb.shape[0],), int(k), device=dev, dtype=torch.long)
        out.append(model(xb, tb, kb).cpu())
    return torch.cat(out, 0)


def sumsq(a):
    """sum over the 3072 velocity dims, per sample -> (N,)."""
    return a.reshape(a.shape[0], -1).pow(2).sum(1)


@torch.no_grad()
def clip_text_features(clip_model, proc, dev):
    """L2-normalized CLIP text features for the 10 CIFAR class prompts (10, d)."""
    prompts = [f"a photo of a {c}" for c in CIFAR_CLASSES]
    txt = proc(text=prompts, return_tensors="pt", padding=True).to(dev)
    tf = clip_model.get_text_features(**txt)
    tf = tf if isinstance(tf, torch.Tensor) else tf.pooler_output
    return tf / tf.norm(dim=-1, keepdim=True)


@torch.no_grad()
def clip_posterior(clip_model, dev, tfeat, x_hat):
    """CLIP zero-shot class posterior p(k|x_hat) (N, 10) via softmax over cosine logits."""
    ifeat = torch.from_numpy(clip_image_features(clip_model, dev, x_hat)).to(dev)
    ifeat = ifeat / ifeat.norm(dim=-1, keepdim=True)
    logits = 100.0 * (ifeat @ tfeat.T)
    return torch.softmax(logits, dim=1).cpu()


def main():
    ap = argparse.ArgumentParser(description="E7 branch-conditioning-gain payoff (per coupling)")
    ap.add_argument("--coupling", default="c0", choices=["c0", "c1", "c3", "c3inf"],
                    help="checkpoint coupling to evaluate; loads the matching "
                         "label-free / conditional pair")
    ap.add_argument("--apply-coupling", action="store_true", dest="apply_coupling",
                    help="re-pair (z,x) with the training coupling so the model is "
                         "queried in-distribution (control for off-manifold eval)")
    args = ap.parse_args()
    coup = args.coupling
    dev = select_device()
    cu = torch.load(RES / f"e7_uncond_{coup}_seed0.pt", map_location=dev)
    vu = UncondUNet(base_ch=cu["config"]["base_ch"]).to(dev)
    vu.load_state_dict(cu["ema"]); vu.eval()
    cc = torch.load(RES / f"e7_{coup}_seed0.pt", map_location=dev)
    vc = CondUNet(base_ch=cc["config"]["base_ch"], n_classes=cc["config"]["n_classes"]).to(dev)
    vc.load_state_dict(cc["ema"]); vc.eval()
    clip_model, proc = load_clip(dev)
    tfeat = clip_text_features(clip_model, proc, dev)     # (10, d) for the CLIP posterior
    print(f"coupling={coup}  device={dev.type}  N={N}  seeds={SEEDS}  t-grid={len(T_GRID)} pts"
          f"  apply_coupling={args.apply_coupling}\n", flush=True)

    Xall, Yall = load_cifar(train=False)
    out = {"t": T_GRID, "config": {"seeds": SEEDS, "N": N, "n_classes": N_CLASSES,
                                   "coupling": coup, "apply_coupling": args.apply_coupling},
           **{key: {t: [] for t in T_GRID} for key in
              ("mse_uncond", "mse_cond_true", "mse_cond_avg", "s_uni", "s_post")}}

    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(Xall.shape[0])[:N]
        x = Xall[idx]; ktrue = Yall.numpy()[idx].astype(np.int64)
        torch.manual_seed(seed + 100)
        z = torch.randn(N, 3, 32, 32)
        if args.apply_coupling:
            x, ktrue, U = apply_indist_coupling(z, x, ktrue, coup, rng)
        else:
            U = x - z
        ktrue_t = torch.from_numpy(ktrue)
        for t in T_GRID:
            Yt = (1 - t) * z + t * x
            vu_t = v_uncond(vu, dev, Yt, t)                         # (N,3,32,32)
            vc_all = torch.stack([v_cond_class(vc, dev, Yt, t, k)   # (K,N,3,32,32)
                                  for k in range(N_CLASSES)], 0)
            vbar = vc_all.mean(0)                                   # label-averaged
            vtrue = vc_all[ktrue_t, torch.arange(N)]               # per-sample true class
            out["mse_uncond"][t].append(float(sumsq(U - vu_t).mean()))
            out["mse_cond_true"][t].append(float(sumsq(U - vtrue).mean()))
            out["mse_cond_avg"][t].append(float(sumsq(U - vbar).mean()))
            s = torch.stack([sumsq(vc_all[k] - vbar) for k in range(N_CLASSES)], 0).mean(0)
            out["s_uni"][t].append(float(s.mean()))
            # posterior-weighted spread (BVD-strict): p(k|Y_t) via CLIP zero-shot on x_hat
            x_hat = (Yt + (1 - t) * vu_t).clamp(-1, 1)
            p = clip_posterior(clip_model, dev, tfeat, x_hat)          # (N,10)
            vbar_post = torch.einsum('knchw,nk->nchw', vc_all, p)      # (N,3,32,32)
            sp = torch.stack([sumsq(vc_all[k] - vbar_post) * p[:, k]
                              for k in range(N_CLASSES)], 0).sum(0)
            out["s_post"][t].append(float(sp.mean()))
        print(f"  seed {seed} done", flush=True)

    def ms(l):
        a = np.array(l, float); return a.mean(), (a.std(ddof=1) if len(a) > 1 else 0.0)
    print(f"\n  {'t':>5} {'D_cross':>9} {'D_within':>9} {'S_uni':>9} {'S_post':>9}")
    for t in T_GRID:
        mu = ms(out["mse_uncond"][t])[0]; mt = ms(out["mse_cond_true"][t])[0]
        ma = ms(out["mse_cond_avg"][t])[0]; su = ms(out["s_uni"][t])[0]
        sp = ms(out["s_post"][t])[0]
        print(f"  {t:>5.2f} {mu-mt:>9.2f} {ma-mt:>9.2f} {su:>9.2f} {sp:>9.2f}")

    suffix = ("" if coup == "c0" else f"_{coup}") + ("_indist" if args.apply_coupling else "")
    out_path = RES / f"e7_conditioning_gain_payoff{suffix}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {out_path} (raw per-seed)")


if __name__ == "__main__":
    main()
