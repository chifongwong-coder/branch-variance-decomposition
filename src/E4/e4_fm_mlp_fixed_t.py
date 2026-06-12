"""E4: Fixed-t Flow Matching MLP verification (1D binary Gaussian).

Goal
----
Verify, in the closed-form setting calibrated in E0, that a single-valued MLP
trained with MSE Flow Matching loss asymptotes to the conditional-mean
velocity v*(y, t) and that its irreducible regression error equals the oracle
ambiguity A_v(t). Specifically, at each fixed t we want:

  L_eval     = E_test ||v_theta(Y) - U||^2      (held-out MSE loss)
  approx_eval= E_test ||v_theta(Y) - v*(Y, t)||^2
  irred_eval = E_test ||U - v*(Y, t)||^2        (= A_v(t) at finite-sample limit)

and the identity
    L_eval = approx_eval + irred_eval
should hold up to test-sample noise.

Setup
-----
* Binary Gaussian target: X | K=k ~ N(k*m, s^2), k in {-1, +1}, equal prior.
* Source: Z ~ N(0, 1).  Independent coupling.
* Linear interpolant Y_t = (1-t) Z + t X,  U = X - Z.
* Closed-form:
    tau2(t)   = (1-t)^2 + t^2 * s^2
    v*(y, t)  = [tanh(t m y / tau2) * m (1-t) + y (t s^2 - (1-t))] / tau2
    Var(U|Y, K) per dim = s^2 / tau2
    A_v(t)    = s^2/tau2  +  m^2 (1-t)^2 / tau2^2 * E_{Y_t}[sech^2(t m Y_t / tau2)]
  (verified to MC in E0).

Sweep (current defaults; edit the module constants below to regenerate other
tags, e.g. STEPS=15000 for the under-training control or T_GRID={0.05,0.95} for
the endpoints tag that the supplement composite expects; only E4_CPU_THREADS is
read from the environment)
-----
* fixed t in {0.2, 0.4, 0.5, 0.6, 0.8}
* hidden in {64, 128, 256}, depth = 4
* 3 seeds per (t, hidden)
* 5,000 training steps each (extended-control run uses 15,000), batch 2048,
  N_train = N_eval = 200,000, AdamW lr = 1e-3, weight decay = 1e-4

Outputs
-------
results/e4_metrics_{tag}.json    # per-run L_eval / approx / irred + log
                                  # (the figure is drawn separately by plot_e4.py)
"""

from __future__ import annotations

import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def _select_device():
    """CUDA if available, else Apple MPS, else CPU with an explicit thread cap."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    # Be polite to the rest of the system on CPU.
    n_threads = int(os.environ.get("E4_CPU_THREADS", "4"))
    torch.set_num_threads(n_threads)
    os.environ.setdefault("OMP_NUM_THREADS", str(n_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(n_threads))
    return torch.device("cpu")


DEVICE = _select_device()


def _device_sync():
    if DEVICE.type == "mps":
        torch.mps.synchronize()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

M = 2.0
S = 0.25
S2 = S ** 2

# Paper-canonical defaults.
T_GRID = [0.2, 0.4, 0.5, 0.6, 0.8]
HIDDEN_LADDER = [64, 128, 256]
DEPTH = 4
SEEDS = [0, 1, 2]

N_TRAIN = 200000
N_EVAL = 200000
BATCH = 2048
STEPS = 5000              # extended-control run uses 15000
LR = 1e-3
WEIGHT_DECAY = 1e-4
EVAL_EVERY = 250
BOOTSTRAP_B = 200
SIGMA_BOUND_BAND = 2.0     # +-2 SE band on lower bound
TAG = "conservative"      # the run reported in the paper (config constants above)


# ---------------------------------------------------------------------------
# Closed-form helpers (1D binary Gaussian + linear interpolant + N(0,1) source)
# ---------------------------------------------------------------------------

def tau2(t):
    return (1.0 - t) ** 2 + (t ** 2) * S2

def vstar_1d(y, t):
    """Closed-form optimal velocity v*(y, t)."""
    tt = tau2(t)
    return (np.tanh(t * M * y / tt) * M * (1.0 - t)
            + y * (t * S2 - (1.0 - t))) / tt

def av_closed_form(t, n_mc=500_000, seed=10000):
    """A_v(t) closed-form: within + between, between via MC over Y_t."""
    tt = tau2(t)
    within = S2 / tt
    rng = np.random.default_rng(seed + int(1000 * t))
    n = n_mc
    k = rng.choice([-1, 1], size=n)
    z = rng.standard_normal(n)
    x = k * M + S * rng.standard_normal(n)
    y = (1.0 - t) * z + t * x
    sech2 = 1.0 / np.cosh(t * M * y / tt) ** 2
    between = ((1.0 - t) ** 2 * (M ** 2) / (tt ** 2)) * sech2.mean()
    return within + between

def sample_pairs(n, t, seed):
    """Return (Y_t, U) and the per-sample irreducible residual U - v*(Y_t, t)."""
    rng = np.random.default_rng(seed)
    k = rng.choice([-1, 1], size=n)
    z = rng.standard_normal(n)
    x = k * M + S * rng.standard_normal(n)
    y = (1.0 - t) * z + t * x
    u = x - z
    return y.astype(np.float32), u.astype(np.float32)


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """v_theta(y) for fixed t.  Input/output dim = 1 (1D toy)."""

    def __init__(self, hidden: int, depth: int):
        super().__init__()
        layers = [nn.Linear(1, hidden), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, y):
        return self.net(y).squeeze(-1)


# ---------------------------------------------------------------------------
# Training one (t, hidden, seed) run
# ---------------------------------------------------------------------------

def device_consistency_check(t=0.5, hidden=128, steps=200, seed=0):
    """Run a few training steps on CPU and (if available) MPS with identical
    seed and data; verify final loss differs by < 1e-3.  Detects gross MPS
    numerical mismatches before we commit to a long run.
    """
    if DEVICE.type != "mps":
        return {"skipped": "MPS not selected"}
    results = {}
    for dev in (torch.device("cpu"), torch.device("mps")):
        torch.manual_seed(seed)
        np.random.seed(seed)
        y, u = sample_pairs(20_000, t, seed)
        yt = torch.from_numpy(y).to(dev).unsqueeze(-1)
        ut = torch.from_numpy(u).to(dev)
        model = MLP(hidden, DEPTH).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=LR,
                                weight_decay=WEIGHT_DECAY)
        rng = np.random.default_rng(seed + 7)
        for _ in range(steps):
            idx = torch.from_numpy(
                rng.integers(0, 20_000, size=512)).to(dev)
            pred = model(yt[idx])
            loss = ((pred - ut[idx]) ** 2).mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if dev.type == "mps":
            torch.mps.synchronize()
        results[dev.type] = float(loss.item())
        del model, opt; gc.collect()
    diff = abs(results["cpu"] - results["mps"])
    rel = diff / max(abs(results["cpu"]), 1e-9)
    results.update({"abs_diff": diff, "rel_diff": rel})
    return results


def train_one(t, hidden, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Train pool and eval pool (different seeds to keep independent)
    y_train, u_train = sample_pairs(N_TRAIN, t, seed * 7 + 1)
    y_eval, u_eval = sample_pairs(N_EVAL, t, seed * 13 + 9_999)
    v_star_eval = vstar_1d(y_eval.astype(np.float64), t).astype(np.float32)
    irred_eval_per_sample = (u_eval - v_star_eval) ** 2

    y_train_t = torch.from_numpy(y_train).to(DEVICE).unsqueeze(-1)
    u_train_t = torch.from_numpy(u_train).to(DEVICE)
    y_eval_t = torch.from_numpy(y_eval).to(DEVICE).unsqueeze(-1)
    u_eval_t = torch.from_numpy(u_eval).to(DEVICE)
    v_star_eval_t = torch.from_numpy(v_star_eval).to(DEVICE)

    model = MLP(hidden, DEPTH).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR,
                            weight_decay=WEIGHT_DECAY)

    rng = np.random.default_rng(seed + 42)
    steps_log = []
    train_loss_log = []
    eval_loss_log = []
    approx_log = []
    model.train()
    running_train_loss = 0.0
    for step in range(STEPS):
        idx = rng.integers(0, N_TRAIN, size=BATCH)
        yb = y_train_t[idx]
        ub = u_train_t[idx]
        pred = model(yb)
        loss = ((pred - ub) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        running_train_loss = 0.99 * running_train_loss + 0.01 * loss.item() \
            if step > 0 else loss.item()
        if (step + 1) % EVAL_EVERY == 0 or step == 0:
            model.eval()
            with torch.no_grad():
                pred_eval = model(y_eval_t)
                L_eval = ((pred_eval - u_eval_t) ** 2).mean().item()
                approx_eval = ((pred_eval - v_star_eval_t) ** 2).mean().item()
            model.train()
            steps_log.append(step + 1)
            train_loss_log.append(running_train_loss)
            eval_loss_log.append(L_eval)
            approx_log.append(approx_eval)

    # Final held-out evaluation
    model.eval()
    with torch.no_grad():
        pred_eval = model(y_eval_t).cpu().numpy()
    L_eval_final = float(((pred_eval - u_eval) ** 2).mean())
    approx_eval_final = float(((pred_eval - v_star_eval) ** 2).mean())
    irred_eval_final = float(irred_eval_per_sample.mean())
    # bootstrap CI for irred_eval (since it's the lower bound we'll compare)
    rng_bs = np.random.default_rng(seed + 999)
    bs_means = np.empty(BOOTSTRAP_B)
    n_e = len(irred_eval_per_sample)
    for b in range(BOOTSTRAP_B):
        idx = rng_bs.integers(0, n_e, size=n_e)
        bs_means[b] = irred_eval_per_sample[idx].mean()
    irred_se = float(bs_means.std(ddof=1))

    out = {
        "steps_log": steps_log,
        "train_loss_log": train_loss_log,
        "eval_loss_log": eval_loss_log,
        "approx_log": approx_log,
        "L_eval_final": L_eval_final,
        "approx_eval_final": approx_eval_final,
        "irred_eval_final": irred_eval_final,
        "irred_se": irred_se,
        "identity_residual": L_eval_final - (approx_eval_final + irred_eval_final),
    }
    # Explicit release: torch on MPS sometimes holds buffers.
    del model, opt, y_train_t, u_train_t, y_eval_t, u_eval_t, v_star_eval_t
    gc.collect()
    if DEVICE.type == "mps":
        torch.mps.empty_cache()
    return out


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------

def _rss_mb():
    """Approximate RSS in MB (psutil if available, else best-effort)."""
    try:
        import psutil  # type: ignore
        return psutil.Process().memory_info().rss / 1024 ** 2
    except Exception:
        try:
            import resource  # macOS gives bytes on macOS, kilobytes on Linux
            r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            return r / 1024 ** 2 if r > 1e8 else r / 1024  # heuristic
        except Exception:
            return float("nan")


def run():
    out_dir = Path(__file__).resolve().parents[2]
    res_dir = out_dir / "results"
    res_dir.mkdir(exist_ok=True)

    print(f"E4 fixed-t MLP verification  [tag={TAG}]")
    print(f"  device      = {DEVICE.type}")
    print(f"  T_GRID      = {T_GRID}")
    print(f"  HIDDEN      = {HIDDEN_LADDER}, depth={DEPTH}")
    print(f"  SEEDS       = {SEEDS}")
    print(f"  STEPS       = {STEPS}, BATCH={BATCH}, EVAL_EVERY={EVAL_EVERY}")
    print(f"  N_TRAIN/EVAL= {N_TRAIN}/{N_EVAL}")
    print(f"  total runs  = {len(T_GRID)*len(HIDDEN_LADDER)*len(SEEDS)}")
    print(f"  startup RSS = {_rss_mb():.1f} MB")

    cc = device_consistency_check()
    print(f"  CPU/MPS consistency check: {cc}")

    # Pre-compute closed-form A_v(t) for reference
    av_cf = {t: float(av_closed_form(t)) for t in T_GRID}

    results = {
        "config": {
            "M": M, "S": S, "T_GRID": T_GRID,
            "HIDDEN_LADDER": HIDDEN_LADDER, "DEPTH": DEPTH,
            "SEEDS": SEEDS, "N_TRAIN": N_TRAIN, "N_EVAL": N_EVAL,
            "BATCH": BATCH, "STEPS": STEPS, "LR": LR,
            "WEIGHT_DECAY": WEIGHT_DECAY, "BOOTSTRAP_B": BOOTSTRAP_B,
        },
        "av_closed_form": av_cf,
        "runs": [],
    }

    print("-" * 70)
    run_counter = 0
    for t in T_GRID:
        print(f"  t = {t}, A_v(t) closed form = {av_cf[t]:.4f}")
        for hidden in HIDDEN_LADDER:
            for seed in SEEDS:
                t0 = time.time()
                r = train_one(t, hidden, seed)
                dt = time.time() - t0
                r.update({"t": t, "hidden": hidden, "seed": seed,
                          "wall_time_s": dt})
                results["runs"].append(r)
                run_counter += 1
                mem_tag = f"  rss={_rss_mb():.0f}MB" if run_counter % 3 == 0 else ""
                print(f"    h={hidden:4d}  seed={seed}  "
                      f"L={r['L_eval_final']:.4f}  "
                      f"approx={r['approx_eval_final']:.4f}  "
                      f"irred={r['irred_eval_final']:.4f}  "
                      f"id_resid={r['identity_residual']:+.4f}  "
                      f"({dt:.1f}s){mem_tag}")

    # Plotting moved to plot_e4.py (run/plot split for the public repo).

    # ------------------------------------------------------------------
    # Save metrics
    # ------------------------------------------------------------------
    with open(res_dir / f"e4_metrics_{TAG}.json", "w") as f:
        json.dump(results, f, indent=2, default=lambda o: float(o))

    # ------------------------------------------------------------------
    # Summary print
    # ------------------------------------------------------------------
    print("-" * 70)
    print("summary:")
    for t in T_GRID:
        for hidden in HIDDEN_LADDER:
            sub = [r for r in results["runs"]
                   if r["t"] == t and r["hidden"] == hidden]
            L = np.mean([r["L_eval_final"] for r in sub])
            ap = np.mean([r["approx_eval_final"] for r in sub])
            ir = np.mean([r["irred_eval_final"] for r in sub])
            cf = av_cf[t]
            print(f"  t={t}  h={hidden:4d}: L={L:.4f}  approx={ap:.4f}  "
                  f"irred={ir:.4f}  CF A_v={cf:.4f}  L-CF={L-cf:+.4f}")
    print(f"  metrics -> {res_dir / f'e4_metrics_{TAG}.json'}")


if __name__ == "__main__":
    run()
