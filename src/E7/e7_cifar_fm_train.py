"""E7: train one class-conditional Flow-Matching model on CIFAR-10 under a
chosen minibatch coupling.

Pipeline:
  x ~ CIFAR-10 (pixels in [-1, 1]), z ~ N(0, I), linear interpolant
  x_t = (1 - t) z + t x, target velocity u = x - z, conditional model
  v_theta(x_t, t, k) trained with E||v - u||^2, t ~ Uniform[0, 1].

The coupling (C0 / C1 / C3@lambda / C3-inf) decides, per minibatch, which noise
source pairs with which real image and which class label conditions the pair
(see e7_cifar_fm_couplings.py). This is the only varied factor across runs.

Outputs (under results/):
  e7_<coupling>_seed<seed>.pt    model + EMA weights + config
  e7_<coupling>_seed<seed>.json  per-step train log + final val FM loss

Device is auto-selected (CUDA -> MPS -> CPU). Hard-coded defaults are
validation-scale; override with CLI flags. Run e.g.:
  python3 e7_cifar_fm_train.py --coupling c0   --seed 0
  python3 e7_cifar_fm_train.py --coupling c1   --seed 0
  python3 e7_cifar_fm_train.py --coupling c3inf --seed 0
"""
from __future__ import annotations

import argparse
import json
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from E7.e7_cifar_fm_couplings import build_coupling

HERE = Path(__file__).resolve().parent
RES = HERE.parents[1] / "results"
DATA = HERE.parents[1] / "results" / "cifar10_cache"  # reuse the E5 CIFAR cache location


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elif dev.type == "mps":
        torch.mps.synchronize()


# ---------------------------------------------------------------------------
# Model: compact class-conditional UNet (~4 M params at base_ch=96)
# ---------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ci, co, 3, padding=1), nn.GroupNorm(8, co), nn.SiLU(),
            nn.Conv2d(co, co, 3, padding=1), nn.GroupNorm(8, co), nn.SiLU())

    def forward(self, x):
        return self.net(x)


class TimeEmbed(nn.Module):
    def __init__(self, dim, out):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, out * 2), nn.SiLU(),
                                 nn.Linear(out * 2, out))

    def forward(self, t):  # t: (B,) in [0,1]
        half = self.dim // 2
        freqs = torch.exp(torch.linspace(0.0, np.log(1000.0), half,
                                         device=t.device))
        arg = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(arg), torch.cos(arg)], dim=-1)
        return self.mlp(emb)


class CondUNet(nn.Module):
    """v_theta(x_t, t, k) for 3x32x32, class-conditional (10 classes)."""

    def __init__(self, base_ch=96, n_classes=10, t_dim=128):
        super().__init__()
        c = base_ch
        self.temb = TimeEmbed(t_dim, c)
        self.cemb = nn.Embedding(n_classes, c)
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

    def forward(self, x, t, k):
        cond = (self.temb(t) + self.cemb(k))[:, :, None, None]
        h0 = self.in_(x) + cond
        h1 = self.d1(h0)            # 32
        h2 = self.d2(self.pool(h1))  # 16
        h3 = self.d3(self.pool(h2))  # 8
        m = self.mid(h3)
        B, C, H, W = m.shape
        a = m.flatten(2).transpose(1, 2)
        a, _ = self.attn(a, a, a)
        m = m + a.transpose(1, 2).reshape(B, C, H, W)
        u = self.u3(torch.cat([m, h3], 1))
        u = self.u2(torch.cat([self.up(u), h2], 1))
        u = self.u1(torch.cat([self.up(u), h1], 1))
        return self.out(u)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_cifar(train=True):
    """Return (images float32 (N,3,32,32) in [-1,1], labels int64 (N,)).
    Downloads CIFAR-10 into the E5 cache dir on first use."""
    import torchvision
    import torchvision.transforms as T
    tf = T.Compose([T.ToTensor(), T.Normalize((0.5,) * 3, (0.5,) * 3)])
    ds = torchvision.datasets.CIFAR10(root=str(DATA), train=train,
                                      download=True, transform=tf)
    xs = torch.stack([ds[i][0] for i in range(len(ds))])
    ys = torch.tensor([ds[i][1] for i in range(len(ds))], dtype=torch.long)
    return xs, ys


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.mul_(self.decay).add_(p, alpha=1 - self.decay)
        for s, p in zip(self.shadow.buffers(), model.buffers()):
            s.copy_(p)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(args):
    dev = select_device()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    RES.mkdir(parents=True, exist_ok=True)
    print(f"E7 train  coupling={args.coupling}  seed={args.seed}  device={dev.type}")
    print(f"  base_ch={args.base_ch}  batch={args.batch}  steps={args.steps}"
          f"  lambda={args.lambda_sem}")

    X, Y = load_cifar(train=True)
    X = X.to(dev)
    Y_np = Y.numpy()
    N = X.shape[0]
    print(f"  CIFAR-10 train: {N} images on {dev.type}")

    model = CondUNet(base_ch=args.base_ch).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model params: {n_params / 1e6:.2f} M")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    ema = EMA(model, decay=args.ema)
    coupling = build_coupling(args.coupling, lambda_sem=args.lambda_sem)

    log = []
    t_start = time.time()
    for step in range(1, args.steps + 1):
        idx = rng.integers(0, N, size=args.batch)
        x = X[idx]                                   # (B,3,32,32)
        kx = Y_np[idx]                               # target classes
        z = torch.randn_like(x)
        # desired class per source = permutation of target classes (same hist)
        c = kx[rng.permutation(args.batch)]

        # coupling on flattened pixels (CPU numpy via scipy Hungarian)
        Zf = z.reshape(args.batch, -1).detach().cpu().numpy()
        Xf = x.reshape(args.batch, -1).detach().cpu().numpy()
        col, cond = coupling(Zf, Xf, kx, c)
        x = x[col]                                   # paired targets
        cond_t = torch.from_numpy(np.ascontiguousarray(cond)).to(dev)

        t = torch.rand(args.batch, device=dev)
        xt = (1 - t)[:, None, None, None] * z + t[:, None, None, None] * x
        u = x - z
        v = model(xt, t, cond_t)
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

    # final held-out FM loss (val split, fixed coupling-free random pairing)
    val_loss = eval_val_loss(ema.shadow, dev, args, rng)
    out = {
        "coupling": args.coupling, "seed": args.seed,
        "config": {"base_ch": args.base_ch, "batch": args.batch,
                   "steps": args.steps, "lr": args.lr, "ema": args.ema,
                   "lambda_sem": args.lambda_sem, "n_params": n_params},
        "train_log": log,
        "val_fm_loss": val_loss,
        "wall_s": time.time() - t_start,
    }
    jpath = RES / f"e7_{args.coupling}_seed{args.seed}.json"
    with open(jpath, "w") as f:
        json.dump(out, f, indent=2)
    _save(model, ema, args, n_params, args.steps, RES)               # stepped final
    _save(model, ema, args, n_params, args.steps, RES, canonical=True)  # for diagnostics
    print(f"  wrote {jpath}  (val FM loss = {val_loss:.4f})")


@torch.no_grad()
def eval_val_loss(model, dev, args, rng, n_batches=20):
    """Held-out FM loss on the test split with random (coupling-free) pairing,
    so it measures the floor level, not the coupling."""
    Xv, Yv = load_cifar(train=False)
    Xv = Xv.to(dev)
    Yv = Yv.numpy()
    Nv = Xv.shape[0]
    model.eval()
    tot = 0.0
    for _ in range(n_batches):
        idx = rng.integers(0, Nv, size=args.batch)
        x = Xv[idx]
        k = torch.from_numpy(Yv[idx]).to(dev)
        z = torch.randn_like(x)
        t = torch.rand(args.batch, device=dev)
        xt = (1 - t)[:, None, None, None] * z + t[:, None, None, None] * x
        u = x - z
        v = model(xt, t, k)
        tot += float(((v - u) ** 2).mean().item())
    return tot / n_batches


def _save(model, ema, args, n_params, step, res, canonical=False):
    """Write a checkpoint. By default the filename carries the step
    (e7_<coupling>_seed<seed>_step<step>.pt) so intermediate checkpoints are
    preserved for a saturation sweep. With canonical=True it writes the
    step-free name (e7_<coupling>_seed<seed>.pt) that the diagnostics load."""
    name = (f"e7_{args.coupling}_seed{args.seed}.pt" if canonical
            else f"e7_{args.coupling}_seed{args.seed}_step{step}.pt")
    torch.save({"model": model.state_dict(),
                "ema": ema.shadow.state_dict(),
                "config": {"base_ch": args.base_ch, "n_classes": 10},
                "coupling": args.coupling, "seed": args.seed, "step": step},
               res / name)


def main():
    p = argparse.ArgumentParser(description="E7 CIFAR-10 FM training")
    p.add_argument("--coupling", required=True,
                   choices=["c0", "c1", "c3", "c3inf"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=50000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--base-ch", type=int, default=96, dest="base_ch")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--lambda-sem", type=float, default=10.0, dest="lambda_sem")
    p.add_argument("--log-every", type=int, default=500, dest="log_every")
    p.add_argument("--ckpt-every", type=int, default=10000, dest="ckpt_every")
    train(p.parse_args())


if __name__ == "__main__":
    main()
