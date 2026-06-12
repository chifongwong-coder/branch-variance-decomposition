"""E7 label-free (unconditional) Flow-Matching training.

The discriminator experiment for D2: the model is NOT told the class, but the
COUPLING still uses the class labels to construct the pairing (c3inf pairs within
class). So class is supervision used only to build the transport plan, never an
input to the network. This is the clean E3 analogue: the coupling is the ONLY
semantic channel, so if a class-aware coupling still produces more class-coherent
samples / a BVD switching separation, the coupling demonstrably taught semantics
through path geometry alone, not through a handed label.

Difference from e7_cifar_fm_train.py:
  - model is UncondUNet: v_theta(x_t, t), no class embedding;
  - the coupling is built exactly as before (uses kx / desired class c), but the
    returned cond label is discarded instead of fed to the model.
Everything else (interpolant, FM loss, EMA, stepped checkpoints) is identical.

Outputs (under results/): e7_uncond_<coupling>_seed<seed>_step<step>.pt (stepped),
the canonical e7_uncond_<coupling>_seed<seed>.pt, and ..._seed<seed>.json.

Run:  python3 e7_cifar_fm_train_labelfree.py --coupling c3inf --seed 0 --steps 50000
"""
import argparse
import json
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from E7.e7_cifar_fm_train import (Block, TimeEmbed, load_cifar, EMA,        # noqa: E402
                               select_device, device_sync)
from E7.e7_cifar_fm_couplings import build_coupling                         # noqa: E402

RES = HERE.parents[1] / "results"


class UncondUNet(nn.Module):
    """v_theta(x_t, t) for 3x32x32, UNCONDITIONAL (no class input). Identical to
    CondUNet minus the class embedding, so it is the matched label-free twin."""

    def __init__(self, base_ch=96, t_dim=128):
        super().__init__()
        c = base_ch
        self.temb = TimeEmbed(t_dim, c)
        self.in_ = nn.Conv2d(3, c, 3, padding=1)
        self.d1 = Block(c, c)
        self.d2 = Block(c, c * 2)
        self.d3 = Block(c * 2, c * 2)
        self.mid = Block(c * 2, c * 2)
        self.attn = nn.MultiheadAttention(c * 2, 4, batch_first=True)
        self.u3 = Block(c * 2 + c * 2, c * 2)
        self.u2 = Block(c * 2 + c * 2, c)
        self.u1 = Block(c + c, c)
        self.out = nn.Conv2d(c, 3, 3, padding=1)
        self.pool = nn.AvgPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x, t):
        cond = self.temb(t)[:, :, None, None]
        h0 = self.in_(x) + cond
        h1 = self.d1(h0)
        h2 = self.d2(self.pool(h1))
        h3 = self.d3(self.pool(h2))
        m = self.mid(h3)
        B, C, H, W = m.shape
        a = m.flatten(2).transpose(1, 2)
        a, _ = self.attn(a, a, a)
        m = m + a.transpose(1, 2).reshape(B, C, H, W)
        u = self.u3(torch.cat([m, h3], 1))
        u = self.u2(torch.cat([self.up(u), h2], 1))
        u = self.u1(torch.cat([self.up(u), h1], 1))
        return self.out(u)


def _save(model, ema, args, n_params, step, res, canonical=False):
    name = (f"e7_uncond_{args.coupling}_seed{args.seed}.pt" if canonical
            else f"e7_uncond_{args.coupling}_seed{args.seed}_step{step}.pt")
    torch.save({"model": model.state_dict(),
                "ema": ema.shadow.state_dict(),
                "config": {"base_ch": args.base_ch}, "uncond": True,
                "coupling": args.coupling, "seed": args.seed, "step": step},
               res / name)


@torch.no_grad()
def eval_val_loss(model, dev, args, rng, n_batches=20):
    """Held-out FM loss under the run's OWN coupling pairing (P3-correct)."""
    Xv, Yv = load_cifar(train=False)
    Xv = Xv.to(dev)
    Yv = Yv.numpy()
    Nv = Xv.shape[0]
    coupling = build_coupling(args.coupling, lambda_sem=args.lambda_sem)
    model.eval()
    tot = 0.0
    for _ in range(n_batches):
        idx = rng.integers(0, Nv, size=args.batch)
        x = Xv[idx]
        kx = Yv[idx]
        z = torch.randn_like(x)
        c = kx[rng.permutation(args.batch)]
        col, _ = coupling(z.reshape(args.batch, -1).cpu().numpy(),
                          x.reshape(args.batch, -1).cpu().numpy(), kx, c)
        x = x[col]
        t = torch.rand(args.batch, device=dev)
        xt = (1 - t)[:, None, None, None] * z + t[:, None, None, None] * x
        u = x - z
        tot += float(((model(xt, t) - u) ** 2).mean().item())
    return tot / n_batches


def train(args):
    dev = select_device()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    RES.mkdir(parents=True, exist_ok=True)
    print(f"E7 label-free train  coupling={args.coupling}  seed={args.seed}  device={dev.type}")
    print(f"  base_ch={args.base_ch}  batch={args.batch}  steps={args.steps}  lambda={args.lambda_sem}")

    X, Y = load_cifar(train=True)
    X = X.to(dev)
    Y_np = Y.numpy()
    N = X.shape[0]
    model = UncondUNet(base_ch=args.base_ch).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  CIFAR-10 train: {N} images   model params: {n_params / 1e6:.2f} M (unconditional)")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    ema = EMA(model, decay=args.ema)
    coupling = build_coupling(args.coupling, lambda_sem=args.lambda_sem)

    log = []
    t_start = time.time()
    for step in range(1, args.steps + 1):
        idx = rng.integers(0, N, size=args.batch)
        x = X[idx]
        kx = Y_np[idx]
        z = torch.randn_like(x)
        c = kx[rng.permutation(args.batch)]
        # coupling STILL uses labels to build the pairing; cond is discarded.
        Zf = z.reshape(args.batch, -1).detach().cpu().numpy()
        Xf = x.reshape(args.batch, -1).detach().cpu().numpy()
        col, _ = coupling(Zf, Xf, kx, c)
        x = x[col]

        t = torch.rand(args.batch, device=dev)
        xt = (1 - t)[:, None, None, None] * z + t[:, None, None, None] * x
        u = x - z
        v = model(xt, t)                                  # NO class input
        loss = ((v - u) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        ema.update(model)

        if step % args.log_every == 0 or step == 1:
            log.append({"step": step, "loss": float(loss.item())})
            el = time.time() - t_start
            print(f"  step {step:6d}/{args.steps}  loss={loss.item():.4f}"
                  f"  {step / max(el, 1e-9):.1f} it/s", flush=True)
        if args.ckpt_every and step % args.ckpt_every == 0:
            _save(model, ema, args, n_params, step, RES)
        # Low-power duty-cycling: every throttle_every steps, drain the GPU queue
        # and idle for throttle_sec. Pure wall-clock knob (same seeds/ops/result);
        # off by default so default-run behavior is unchanged.
        if args.throttle_sec > 0 and args.throttle_every and step % args.throttle_every == 0:
            device_sync(dev)
            time.sleep(args.throttle_sec)

    val_loss = eval_val_loss(ema.shadow, dev, args, rng)
    out = {"coupling": args.coupling, "seed": args.seed, "uncond": True,
           "config": {"base_ch": args.base_ch, "batch": args.batch, "steps": args.steps,
                      "lr": args.lr, "ema": args.ema, "lambda_sem": args.lambda_sem,
                      "n_params": n_params},
           "train_log": log, "val_fm_loss": val_loss, "wall_s": time.time() - t_start}
    jpath = RES / f"e7_uncond_{args.coupling}_seed{args.seed}.json"
    with open(jpath, "w") as f:
        json.dump(out, f, indent=2)
    _save(model, ema, args, n_params, args.steps, RES)
    _save(model, ema, args, n_params, args.steps, RES, canonical=True)
    print(f"  wrote {jpath}  (val FM loss = {val_loss:.4f})")


def main():
    p = argparse.ArgumentParser(description="E7 label-free (unconditional) FM training")
    p.add_argument("--coupling", required=True, choices=["c0", "c1", "c3", "c3inf"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=50000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--base-ch", type=int, default=96, dest="base_ch")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--lambda-sem", type=float, default=10.0, dest="lambda_sem")
    p.add_argument("--log-every", type=int, default=500, dest="log_every")
    p.add_argument("--ckpt-every", type=int, default=5000, dest="ckpt_every")
    p.add_argument("--throttle-every", type=int, default=0, dest="throttle_every",
                   help="low-power duty-cycling: idle the GPU every N steps (0=off)")
    p.add_argument("--throttle-sec", type=float, default=0.0, dest="throttle_sec",
                   help="seconds to idle at each throttle point (cooldown)")
    train(p.parse_args())


if __name__ == "__main__":
    main()
