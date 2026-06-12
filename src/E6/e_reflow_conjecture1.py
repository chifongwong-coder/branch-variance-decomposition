"""Reflow Conjecture 1 driver (E3b 4-mode XOR, t-conditioned FM MLP + RK4).

Tests Conjecture 1:

  A single reflow iteration starting from C_0 should reduce A_v(t), with
  its between-branch reduction Delta A_between(t) concentrated in regions
  where the initial coupling has high R_switch^{C_0}(t).

There is no share-threshold gate; locality is tested by per-t curves,
rho_P(R_sw^{C0}, Delta A_b) > 0, and explicit per-band metrics.
Decomposition is computed for BOTH the coarse XOR label K and the fine
mode_idx label K': under coarse K the toy assigns 70% of A_v to A_within,
but fine K' reassigns some of this to A_between.

Baselines reported side by side for each seed:
  C0       independent coupling on a held-out (Z, X) pool
  reflow   ODE-pushed pairs (Z_new, X_new = ODE(Z_new))
  C1       Euclidean OT on the same held-out (Z, X) pool

Stages
  R0 smoke      1 seed, N=5_000, 1k steps (end-to-end sanity check)
  R1 full       5 paired seeds, N=20_000, 30k steps, 19 t-points
  R2 ode_sanity 1 seed, RK4 step counts {50, 100, 200} (solver convergence)

Quality gate (eval-MSE-based, 3-level Green/Yellow/Red):
  Green:  integrated eval MSE <= 1.10 * oracle floor
          AND high-R_switch-band eval MSE <= 1.20 * floor in band
          --> proceed to report reflow diagnostics
  Yellow: integrated eval MSE <= 1.20 * floor, but band > 1.20
          --> extend to 50k steps OR run one width-512 control seed
  Red:    integrated > 1.20 * floor or no plateau
          --> do not run / do not report

Per-band metric block (separately for coarse and fine labels):
  per band B in {interior, top40_R_sw, top25_R_sw, peak}:
      saturation_share_ref = sum_{t in B} A_b^{C0}(t) / sum_{t in B} A_v^{C0}(t)
      share                = sum Delta A_b / sum Delta A_v  (diagnostic)
      removal_v            = sum Delta A_v / sum A_v^{C0}
      removal_between      = sum Delta A_b / sum A_b^{C0}
      removal_within       = sum Delta A_w / sum A_w^{C0}
  correlations on interior:
      Pearson + Spearman of (R_sw^{C0}, Delta A_b) and (R_sw^{C0}, Delta A_v)
  contrast:
      top25-vs-bottom25 mean Delta A_b (by R_sw^{C0})

Reads from e3b_branch_refinement.py:
  sample_data_4mode, MODES, K_X_PER_MODE, S_WIDTH

Reads from branch_decomp.py:
  decompose (biased 1/k convention)
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

# CPU thread caps must precede numpy / torch import (mirrors e4).
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "4")

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from E3b.e3b_branch_refinement import (
    sample_data_4mode, MODES, K_X_PER_MODE, S_WIDTH,
    couple_c0_independent, couple_c1_hungarian,
)
from core.branch_decomp import decompose


# ---------------------------------------------------------------------------
# Paper-canonical settings (hard-coded; --stage selects the run)
# ---------------------------------------------------------------------------

# MLP architecture (t-conditioned, single net for all t)
WIDTH = 256
DEPTH = 4
T_EMBED_DIM = 32       # Fourier features for t embedding

# Training (R1 full)
N_PAIRS_FULL = 20_000
N_PAIRS_SMOKE = 5_000
STEPS_FULL = 30_000
STEPS_SMOKE = 1_000
STEPS_YELLOW_EXTEND = 50_000
BATCH = 2048
LR = 1e-3
WEIGHT_DECAY = 1e-4
EVAL_EVERY = 1_000

# ODE sampler
RK4_STEPS_DEFAULT = 100
RK4_STEPS_SANITY = [50, 100, 200]
ODE_T_START = 0.0
ODE_T_END = 1.0

# Decomposition (paper-canonical kNN bandwidth, biased 1/k)
K_KNN = 80

# Quality gate thresholds
GREEN_INTEG_MULT = 1.10
GREEN_BAND_MULT = 1.20
YELLOW_INTEG_MULT = 1.20
HIGH_RSW_BAND_FRACTION = 0.4   # top-40% of t-grid by R_switch^{C0}

# Endpoint purity threshold (fraction of X_new within 1.5 sigma of nearest mode)
PURITY_RADIUS_SIGMA = 1.5
PURITY_TARGET = 0.95

# 19-point t-grid (matches E3b/E3 paper convention)
T_GRID = np.linspace(0.05, 0.95, 19).tolist()

# Paired seed plan: 5 train seeds, 5 sample seeds
TRAIN_SEEDS_FULL = [0, 1, 2, 3, 4]
SAMPLE_SEEDS_FULL = [100, 101, 102, 103, 104]
C0_BASELINE_SEED_OFFSET = 1000   # train_seed + 1000 -> held-out C0 pool seed

# Share scalar reporting: integrated over t in [0.1, 0.9]
SHARE_T_LO = 0.1
SHARE_T_HI = 0.9


# ---------------------------------------------------------------------------
# Device selection (MPS if available, CPU otherwise)
# ---------------------------------------------------------------------------

def _select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    n_threads = int(os.environ.get("OMP_NUM_THREADS", "4"))
    torch.set_num_threads(n_threads)
    return torch.device("cpu")


DEVICE = _select_device()


def _device_sync():
    if DEVICE.type == "mps":
        torch.mps.synchronize()


# ---------------------------------------------------------------------------
# t-conditioned MLP velocity model
# ---------------------------------------------------------------------------

class FourierTimeEmbedding(nn.Module):
    """Fixed Fourier features for t in [0, 1]. T_EMBED_DIM/2 frequencies."""

    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0, "T_EMBED_DIM must be even"
        # log-spaced frequencies in [1, ~64]
        freqs = torch.exp(torch.linspace(0.0, np.log(64.0), dim // 2))
        self.register_buffer("freqs", freqs)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t shape: (B,); returns (B, dim)
        arg = 2.0 * np.pi * t[:, None] * self.freqs[None, :]
        return torch.cat([torch.sin(arg), torch.cos(arg)], dim=-1)


class TimeConditionedMLP(nn.Module):
    """v_theta(y, t) for 2-D state. Concatenates y with Fourier(t)."""

    def __init__(self, state_dim: int, hidden: int, depth: int,
                 t_embed_dim: int):
        super().__init__()
        self.t_embed = FourierTimeEmbedding(t_embed_dim)
        in_dim = state_dim + t_embed_dim
        layers = [nn.Linear(in_dim, hidden), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers += [nn.Linear(hidden, state_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # y: (B, state_dim), t: (B,)
        te = self.t_embed(t)
        x = torch.cat([y, te], dim=-1)
        return self.net(x)


# ---------------------------------------------------------------------------
# Training: paper-canonical FM MSE loss with uniform t sampling
# ---------------------------------------------------------------------------

def train_model(Z: np.ndarray, X: np.ndarray, seed: int,
                steps: int, log_every: int = 1_000) -> dict:
    """Train a t-conditioned MLP on (Z, X) FM pairs. Returns the model and
    a per-step training log."""
    torch.manual_seed(seed)

    N, state_dim = Z.shape
    Zt = torch.from_numpy(Z.astype(np.float32)).to(DEVICE)
    Xt = torch.from_numpy(X.astype(np.float32)).to(DEVICE)
    Ut = Xt - Zt   # target velocity = X - Z (linear interpolant)

    model = TimeConditionedMLP(state_dim, WIDTH, DEPTH, T_EMBED_DIM).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR,
                            weight_decay=WEIGHT_DECAY)

    rng = np.random.default_rng(seed + 42)
    train_log = []

    model.train()
    running_loss = 0.0
    t0 = time.time()
    for step in range(steps):
        # sample minibatch
        idx = rng.integers(0, N, size=BATCH)
        idx_t = torch.from_numpy(idx).to(DEVICE)
        t_batch = torch.rand(BATCH, device=DEVICE)
        zb = Zt[idx_t]
        xb = Xt[idx_t]
        ub = Ut[idx_t]
        yb = (1.0 - t_batch[:, None]) * zb + t_batch[:, None] * xb

        pred = model(yb, t_batch)
        loss = ((pred - ub) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        running_loss = (0.99 * running_loss + 0.01 * loss.item()
                        if step > 0 else loss.item())

        if (step + 1) % log_every == 0 or step == 0:
            train_log.append({
                "step": step + 1,
                "running_loss": running_loss,
            })

    _device_sync()
    elapsed = time.time() - t0
    model.eval()
    return {
        "model": model,
        "train_log": train_log,
        "wall_s": elapsed,
    }


# ---------------------------------------------------------------------------
# Eval: per-t MSE on a held-out (Z, X) pool
# ---------------------------------------------------------------------------

def eval_per_t_mse(model: nn.Module, Z: np.ndarray, X: np.ndarray,
                   t_grid: list[float]) -> dict:
    """For each t in t_grid, compute E_x ||v_theta(Y_t) - U||^2 on the full
    pool, summed over state dimensions (matches branch_decomp's A_v scale)."""
    Zt = torch.from_numpy(Z.astype(np.float32)).to(DEVICE)
    Xt = torch.from_numpy(X.astype(np.float32)).to(DEVICE)
    Ut = Xt - Zt
    per_t = []
    with torch.no_grad():
        for t in t_grid:
            t_tensor = torch.full((len(Z),), float(t), device=DEVICE)
            Y = (1.0 - t) * Zt + t * Xt
            pred = model(Y, t_tensor)
            # sum over state dimensions, then average over samples --
            # matches branch_decomp.decompose's A_v convention.
            mse = ((pred - Ut) ** 2).sum(dim=-1).mean().item()
            per_t.append(float(mse))
    return {"t_grid": list(t_grid), "mse": per_t}


# ---------------------------------------------------------------------------
# ODE sampler: RK4 from t=0 (Z source) to t=1 (X endpoint)
# ---------------------------------------------------------------------------

def ode_sample_rk4(model: nn.Module, Z_new: np.ndarray,
                   n_steps: int = RK4_STEPS_DEFAULT) -> np.ndarray:
    """Push Z_new through the learned velocity field via RK4. Returns X_new
    of the same shape as Z_new."""
    Zt = torch.from_numpy(Z_new.astype(np.float32)).to(DEVICE)
    h = (ODE_T_END - ODE_T_START) / n_steps
    with torch.no_grad():
        y = Zt.clone()
        for i in range(n_steps):
            t_curr = ODE_T_START + i * h
            t1 = torch.full((len(y),), t_curr, device=DEVICE)
            t2 = torch.full((len(y),), t_curr + 0.5 * h, device=DEVICE)
            t3 = torch.full((len(y),), t_curr + 0.5 * h, device=DEVICE)
            t4 = torch.full((len(y),), t_curr + h, device=DEVICE)
            k1 = model(y, t1)
            k2 = model(y + 0.5 * h * k1, t2)
            k3 = model(y + 0.5 * h * k2, t3)
            k4 = model(y + h * k3, t4)
            y = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    _device_sync()
    return y.cpu().numpy().astype(np.float64)


# ---------------------------------------------------------------------------
# Endpoint labeling: nearest mode + 1.5 sigma purity
# ---------------------------------------------------------------------------

def endpoint_label(X_new: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """For each X_new[i], find the nearest mode in MODES. Returns:
        mode_idx_new:  (N,) int  fine label (0..3)
        K_X_new:       (N,) int  XOR coarse label (sign of K_X_PER_MODE[mode])
        endpoint_stats: dict with:
            radial_mass_1_5_sigma: fraction within 1.5 * S_WIDTH
                                  (renamed from "purity_1_5_sigma"; the
                                  observed value should be compared to the
                                  theoretical 2D Gaussian mass at the same
                                  radius, NOT to a fixed threshold)
            mean_dist, q90_dist, median_dist: distance stats (raw units)
            frac_within_{1,1_5,2,3}sigma: cumulative fractions for radial
                                          CDF calibration
            theoretical_2d_gaussian_frac_*: corresponding theoretical mass
                                            1 - exp(-k^2/2) for a 2D
                                            Gaussian mode with width S_WIDTH
            max_radial_CDF_deviation: max |observed - theoretical| over
                                      k in {1, 1.5, 2, 3} sigma. Headline
                                      "endpoint calibration" scalar.
            mode_proportions_fine: 4-tuple of fractions per mode index
            mode_proportions_coarse: dict {-1: frac, +1: frac}
            mode_max_dev_from_uniform: max |frac - 0.25| over fine modes
                                       (uniform target = 0.25 each)
            distances: (N,) array of distances to nearest mode (raw units)
                       -- enables full empirical CDF plotting
    """
    # (N, 4) distances to each mode
    d2 = ((X_new[:, None, :] - MODES[None, :, :]) ** 2).sum(-1)
    mode_idx_new = np.argmin(d2, axis=1).astype(np.int64)
    dist = np.sqrt(d2[np.arange(len(X_new)), mode_idx_new])
    K_X_new = K_X_PER_MODE[mode_idx_new].astype(np.int64)
    N = len(X_new)

    # Radial mass fractions at canonical radii
    radii_sigma = [1.0, 1.5, 2.0, 3.0]
    frac_obs = {k: float((dist <= k * S_WIDTH).mean()) for k in radii_sigma}
    # Theoretical 2D Gaussian: P(R <= k sigma) = 1 - exp(-k^2 / 2)
    frac_thy = {k: float(1.0 - np.exp(-(k ** 2) / 2.0)) for k in radii_sigma}
    max_dev = max(abs(frac_obs[k] - frac_thy[k]) for k in radii_sigma)

    # Mode proportions
    fine_props = {int(m): float((mode_idx_new == m).mean()) for m in range(4)}
    coarse_props = {-1: float((K_X_new == -1).mean()),
                    1:  float((K_X_new == 1).mean())}
    mode_max_dev_uniform = max(abs(fine_props[m] - 0.25) for m in range(4))

    # Per-mode covariance diagnostics.
    # For each fine mode m:
    #   center offset: ||hat_mu_m - mu_m||
    #   trace ratio:   tr(hat_Sigma_m) / (2 * sigma^2)        -- should ~~ 1
    #   eig_min/max:   eigenvalues of hat_Sigma_m / sigma^2   -- should ~~ 1
    #   anisotropy:    eig_max / eig_min                      -- should ~~ 1
    #   W2_to_target:  Gaussian Wasserstein-2 to N(mu_m, sigma^2 I)
    sigma2 = float(S_WIDTH ** 2)
    per_mode_cov = {}
    cov_summary = {
        "max_center_offset":   0.0,
        "max_anisotropy":      0.0,
        "max_trace_ratio_dev": 0.0,   # |trace_ratio - 1|
        "max_W2_to_target":    0.0,
    }
    for m in range(4):
        sel = mode_idx_new == m
        n_m = int(sel.sum())
        if n_m < 2:
            per_mode_cov[m] = {"n": n_m,
                                "center_offset": float("nan"),
                                "trace_ratio":   float("nan"),
                                "eig_min_ratio": float("nan"),
                                "eig_max_ratio": float("nan"),
                                "anisotropy":    float("nan"),
                                "W2_to_target":  float("nan"),
                                "hat_mu":        [float("nan"), float("nan")]}
            continue
        Xm = X_new[sel]                              # (n_m, 2)
        hat_mu = Xm.mean(axis=0)                      # (2,)
        ctr_offset = float(np.linalg.norm(hat_mu - MODES[m]))
        # Sample covariance (unbiased N-1)
        hat_Sigma = np.cov(Xm.T, ddof=1)              # (2, 2)
        eigvals = np.linalg.eigvalsh(hat_Sigma)       # ascending
        eig_min = float(max(eigvals[0], 0.0))         # clamp tiny negative
        eig_max = float(eigvals[1])
        trace_ratio = float(np.trace(hat_Sigma) / (2.0 * sigma2))
        eig_min_ratio = eig_min / sigma2
        eig_max_ratio = eig_max / sigma2
        anisotropy = float(eig_max / max(eig_min, 1e-12))
        # Gaussian W^2 to target N(mu_m, sigma^2 I):
        #   W2^2 = ||hat_mu - mu||^2 + tr(hat_Sigma + sigma^2 I
        #                                  - 2 (sigma I hat_Sigma sigma I)^{1/2})
        # For Sigma_target = sigma^2 I, the cross term simplifies:
        #   2 (sigma I hat_Sigma sigma I)^{1/2} = 2 sigma * hat_Sigma^{1/2}
        #   tr(2 sigma hat_Sigma^{1/2}) = 2 sigma * tr(hat_Sigma^{1/2})
        #                             = 2 sigma * (sqrt(eig_max) + sqrt(eig_min))
        sigma = float(S_WIDTH)
        tr_cross = 2.0 * sigma * (np.sqrt(max(eig_max, 0.0))
                                  + np.sqrt(max(eig_min, 0.0)))
        W2_sq = (ctr_offset ** 2
                 + float(np.trace(hat_Sigma)) + 2.0 * sigma2 - tr_cross)
        W2 = float(np.sqrt(max(W2_sq, 0.0)))

        per_mode_cov[m] = {
            "n":              n_m,
            "hat_mu":         [float(hat_mu[0]), float(hat_mu[1])],
            "center_offset":  ctr_offset,
            "trace_ratio":    trace_ratio,
            "eig_min_ratio":  eig_min_ratio,
            "eig_max_ratio":  eig_max_ratio,
            "anisotropy":     anisotropy,
            "W2_to_target":   W2,
        }
        cov_summary["max_center_offset"] = max(
            cov_summary["max_center_offset"], ctr_offset)
        cov_summary["max_anisotropy"] = max(
            cov_summary["max_anisotropy"], anisotropy)
        cov_summary["max_trace_ratio_dev"] = max(
            cov_summary["max_trace_ratio_dev"], abs(trace_ratio - 1.0))
        cov_summary["max_W2_to_target"] = max(
            cov_summary["max_W2_to_target"], W2)
    # Convert int keys to string for JSON serialization
    per_mode_cov = {str(k): v for k, v in per_mode_cov.items()}

    endpoint_stats = {
        # canonical headline scalar -- renamed from "purity_1_5_sigma"
        "radial_mass_1_5_sigma":  frac_obs[1.5],
        # radial calibration
        "frac_within_1sigma":     frac_obs[1.0],
        "frac_within_1_5sigma":   frac_obs[1.5],
        "frac_within_2sigma":     frac_obs[2.0],
        "frac_within_3sigma":     frac_obs[3.0],
        "theoretical_2d_gaussian_frac_1sigma":   frac_thy[1.0],
        "theoretical_2d_gaussian_frac_1_5sigma": frac_thy[1.5],
        "theoretical_2d_gaussian_frac_2sigma":   frac_thy[2.0],
        "theoretical_2d_gaussian_frac_3sigma":   frac_thy[3.0],
        "max_radial_CDF_deviation": float(max_dev),
        # distance stats
        "mean_dist":              float(dist.mean()),
        "median_dist":            float(np.median(dist)),
        "q90_dist":               float(np.quantile(dist, 0.90)),
        # mode proportions (vs uniform 0.25 / 0.5 target)
        "mode_proportions_fine":  fine_props,
        "mode_proportions_coarse": coarse_props,
        "mode_max_dev_from_uniform": float(mode_max_dev_uniform),
        # per-mode covariance diagnostics
        "per_mode_covariance":     per_mode_cov,
        "covariance_summary":      cov_summary,
        # raw arrays for plotting
        "distances":              dist.tolist(),
        "mode_idx":               mode_idx_new.tolist(),
        "X_new":                  X_new.tolist(),
        # bookkeeping
        "n":                      int(N),
        "S_WIDTH":                float(S_WIDTH),
    }
    return mode_idx_new, K_X_new, endpoint_stats


# ---------------------------------------------------------------------------
# Decomposition wrapper: A_v / A_within / A_between / R_switch for one t
# ---------------------------------------------------------------------------

def decompose_one_t(Z: np.ndarray, X: np.ndarray, K_X: np.ndarray,
                    mode_idx: np.ndarray, t: float, k: int = K_KNN) -> dict:
    """Single-t kNN decomposition with biased 1/k. Computes BOTH coarse
    (K_X, XOR superclass) and fine (mode_idx in {0,1,2,3}) decompositions
    using the SAME neighborhood (fine K' reveals
    sub-mode path switching hidden inside coarse K).

    Returns {"coarse": {...}, "fine": {...}} with the same scalar keys
    in each block.
    """
    Y = (1.0 - t) * Z + t * X
    U = X - Z
    tree = cKDTree(Y)
    _, idx = tree.query(Y, k=k, workers=4)
    U_n = U[idx]            # (N, k, d)
    K_n = K_X[idx]          # (N, k)   coarse XOR label
    M_n = mode_idx[idx]     # (N, k)   fine mode index

    def _pack(s):
        agg = s["aggregated"]
        a_w = agg["A_within"]
        a_b = agg["A_between"]
        return {
            "A_v": float(agg["A_v"]),
            "A_within": float(a_w),
            "A_between": float(a_b),
            "R_switch": float(a_b / max(a_w + a_b, 1e-12)),
            "identity_resid": float(agg["identity_residual_max_abs"]),
        }

    return {
        "coarse": _pack(decompose(U_n, K_n)),
        "fine":   _pack(decompose(U_n, M_n)),
    }


def decompose_full_t_grid(Z: np.ndarray, X: np.ndarray, K_X: np.ndarray,
                          mode_idx: np.ndarray, t_grid: list[float]) -> dict:
    """Run decompose_one_t over the t-grid. Returns
    {"coarse": {"A_v": [...], ...}, "fine": {"A_v": [...], ...}}.
    """
    keys = ("A_v", "A_within", "A_between", "R_switch", "identity_resid")
    per_t = {label: {k: [] for k in keys} for label in ("coarse", "fine")}
    for t in t_grid:
        m = decompose_one_t(Z, X, K_X, mode_idx, float(t))
        for label in ("coarse", "fine"):
            for k in keys:
                per_t[label][k].append(m[label][k])
    return per_t


# ---------------------------------------------------------------------------
# per-band scalars (coarse and fine labels)
# ---------------------------------------------------------------------------
#
# For each band B in {interior, top40_R_sw, top25_R_sw, peak}:
#   saturation_share_ref = sum_{t in B} A_b^{C0}(t) / sum_{t in B} A_v^{C0}(t)
#       Reference value: what share would equal in the reflow-saturates
#       (A_v^{reflow} -> 0) limit. Cannot exceed this; threshold rule
#       "share > 0.5" is meaningless when this ref is < 0.5.
#   share        = sum ΔA_b / sum ΔA_v   (the original quantity, reported
#                                          as DIAGNOSTIC not gate)
#   removal_v    = sum ΔA_v / sum A_v^{C0}       (fraction of C0 floor removed)
#   removal_between = sum ΔA_b / sum A_b^{C0}    (fraction of between removed)
#   removal_within  = sum ΔA_w / sum A_w^{C0}    (fraction of within removed)
#
# Plus Pearson + Spearman correlations of (R_sw^{C0}, ΔA_b) and
# (R_sw^{C0}, ΔA_v) on interior, and a high-vs-low contrast scalar.

def _band_masks(t_grid: list[float], rsw_c0: np.ndarray) -> dict:
    """Compute the per-band masks.
    Primary high-R_sw bands are INTERIOR-filtered (top X% within
    t in [0.1, 0.9]) to avoid endpoint-noise t-points being selected when
    R_sw spikes near edges.
    """
    t = np.asarray(t_grid)
    interior = (t >= SHARE_T_LO) & (t <= SHARE_T_HI)
    if not interior.any():
        empty = np.zeros_like(rsw_c0, dtype=bool)
        return {"interior": interior, "top40_R_sw": empty,
                "top25_R_sw": empty, "peak": empty}
    # Quantiles taken over rsw_c0 restricted to interior
    rsw_int = rsw_c0[interior]
    q40 = np.quantile(rsw_int, 1.0 - 0.40)
    q25 = np.quantile(rsw_int, 1.0 - 0.25)
    top40 = (rsw_c0 >= q40) & interior
    top25 = (rsw_c0 >= q25) & interior
    # Peak is the single interior t with max R_sw
    peak = np.zeros_like(rsw_c0, dtype=bool)
    rsw_masked = np.where(interior, rsw_c0, -np.inf)
    peak[int(np.argmax(rsw_masked))] = True
    return {
        "interior":   interior,
        "top40_R_sw": top40,
        "top25_R_sw": top25,
        "peak":       peak,
    }


def compute_b5_scalars(t_grid: list[float], dec_c0: dict,
                       dec_reflow: dict, dec_c1: dict | None = None) -> dict:
    """Per-band metric block. `dec_c0`, `dec_reflow`, `dec_c1` are single-label-type
    structures with per-t arrays for A_v, A_within, A_between, R_switch.
    Returns per-band scalars + correlations + contrast + share curve +
    negative-mass + (if dec_c1) distance-to-C1 gaps.

    Conventions:
      - top bands interior-filtered (see _band_masks)
      - contrast: keep difference, drop unstable ratio, add bounded contrast_rel
      - negative: frac + neg_mass + min_rel_delta
      - distance_to_c1: gap_Av and gap_Ab to C1 baseline per band
    """
    t = np.asarray(t_grid)
    av_c0 = np.asarray(dec_c0["A_v"])
    aw_c0 = np.asarray(dec_c0["A_within"])
    ab_c0 = np.asarray(dec_c0["A_between"])
    rsw_c0 = np.asarray(dec_c0["R_switch"])
    av_rf = np.asarray(dec_reflow["A_v"])
    aw_rf = np.asarray(dec_reflow["A_within"])
    ab_rf = np.asarray(dec_reflow["A_between"])

    delta_av = av_c0 - av_rf
    delta_aw = aw_c0 - aw_rf
    delta_ab = ab_c0 - ab_rf

    eps = 1e-12
    bands = _band_masks(t_grid, rsw_c0)
    interior = bands["interior"]

    per_band = {}
    for name, mask in bands.items():
        if not mask.any():
            per_band[name] = {k: float("nan") for k in
                              ("saturation_share_ref", "share",
                               "removal_v", "removal_between", "removal_within")}
            per_band[name]["n_t"] = 0
            continue
        d_v_s = delta_av[mask].sum()
        d_b_s = delta_ab[mask].sum()
        d_w_s = delta_aw[mask].sum()
        av_c0_s = av_c0[mask].sum()
        ab_c0_s = ab_c0[mask].sum()
        aw_c0_s = aw_c0[mask].sum()
        per_band[name] = {
            "n_t":                int(mask.sum()),
            "saturation_share_ref": float(ab_c0_s / max(av_c0_s, eps)),
            "share":              (float(d_b_s / d_v_s)
                                   if abs(d_v_s) > eps else float("nan")),
            "removal_v":          float(d_v_s / max(av_c0_s, eps)),
            "removal_between":    float(d_b_s / max(ab_c0_s, eps)),
            "removal_within":     float(d_w_s / max(aw_c0_s, eps)),
        }

    # Correlations on the interior (the locality test)
    correlations = {}
    if interior.sum() >= 3:
        rsw_i = rsw_c0[interior]
        dab_i = delta_ab[interior]
        dav_i = delta_av[interior]
        correlations["pearson_Rsw_DeltaAb"] = float(
            np.corrcoef(rsw_i, dab_i)[0, 1])
        correlations["pearson_Rsw_DeltaAv"] = float(
            np.corrcoef(rsw_i, dav_i)[0, 1])
        sp_ab = spearmanr(rsw_i, dab_i)
        sp_av = spearmanr(rsw_i, dav_i)
        correlations["spearman_Rsw_DeltaAb"] = float(sp_ab.statistic)
        correlations["spearman_Rsw_DeltaAv"] = float(sp_av.statistic)
    else:
        for k in ("pearson_Rsw_DeltaAb", "pearson_Rsw_DeltaAv",
                  "spearman_Rsw_DeltaAb", "spearman_Rsw_DeltaAv"):
            correlations[k] = float("nan")

    # Contrast: top25_R_sw (interior) vs bottom25 (interior); drop ratio
    if interior.any():
        rsw_int = rsw_c0[interior]
        q25_low = np.quantile(rsw_int, 0.25)
        bottom25 = (rsw_c0 <= q25_low) & interior
    else:
        bottom25 = np.zeros_like(rsw_c0, dtype=bool)
    top25 = bands["top25_R_sw"]
    if bottom25.any() and top25.any():
        top25_mean = float(delta_ab[top25].mean())
        bottom25_mean = float(delta_ab[bottom25].mean())
        # bounded relative contrast (in [-1, 1], well-defined at zero)
        denom_rel = abs(top25_mean) + abs(bottom25_mean) + eps
        contrast = {
            "top25_mean_DeltaAb":    top25_mean,
            "bottom25_mean_DeltaAb": bottom25_mean,
            "difference":            top25_mean - bottom25_mean,
            "contrast_rel": float((top25_mean - bottom25_mean) / denom_rel),
        }
    else:
        contrast = {k: float("nan") for k in
                    ("top25_mean_DeltaAb", "bottom25_mean_DeltaAb",
                     "difference", "contrast_rel")}

    # Per-t share curve (NaN where delta_av <= 0)
    share_curve = np.where(delta_av > eps, delta_ab / delta_av, np.nan)

    # Negative-mass diagnostics on interior
    if interior.any():
        d_av_int = delta_av[interior]
        av_c0_int = av_c0[interior]
        # frac of t-points where reflow LOCALLY increases A_v
        frac_neg = float((d_av_int < 0).mean())
        # negative mass: sum(max(-dA_v, 0)) / sum(A_v^C0)  (relative)
        neg_mass = float(np.maximum(-d_av_int, 0).sum() / max(av_c0_int.sum(), eps))
        # worst-case per-t relative drop
        rel_delta = d_av_int / (av_c0_int + eps)
        min_rel = float(rel_delta.min())
    else:
        frac_neg = float("nan")
        neg_mass = float("nan")
        min_rel = float("nan")

    negative = {
        "frac_neg_delta_av_interior":  frac_neg,
        "neg_mass_delta_av_interior":  neg_mass,
        "min_rel_delta_av_interior":   min_rel,
    }

    # Distance-to-C1 gaps (optional; only if dec_c1 is provided).
    # gap_Av(B) = sum_B (A_v^reflow - A_v^C1) / sum_B A_v^C0
    # gap_Ab(B) = sum_B (A_b^reflow - A_b^C1) / sum_B A_v^C0
    # Positive gap => reflow worse than C1; ~0 => reflow ~ C1; negative => reflow better
    distance_to_c1 = None
    if dec_c1 is not None:
        av_c1 = np.asarray(dec_c1["A_v"])
        ab_c1 = np.asarray(dec_c1["A_between"])
        distance_to_c1 = {}
        for band_name in ("interior", "top25_R_sw"):
            mask = bands[band_name]
            if not mask.any():
                distance_to_c1[band_name] = {"gap_Av": float("nan"),
                                              "gap_Ab": float("nan")}
                continue
            denom = max(av_c0[mask].sum(), eps)
            gap_av = float((av_rf[mask] - av_c1[mask]).sum() / denom)
            gap_ab = float((ab_rf[mask] - ab_c1[mask]).sum() / denom)
            distance_to_c1[band_name] = {
                "gap_Av": gap_av,
                "gap_Ab": gap_ab,
            }

    return {
        "per_band":     per_band,
        "correlations": correlations,
        "contrast":     contrast,
        "share_per_t":  share_curve.tolist(),
        "negative":     negative,
        "distance_to_c1": distance_to_c1,
    }


# ---------------------------------------------------------------------------
# Quality gate: 3 sub-gates
# ---------------------------------------------------------------------------
#
#   regression_gate  -- eval-MSE vs A_v^{C0}(t) oracle floor (was the old gate)
#   endpoint_gate    -- endpoint_purity threshold + distance stats
#   solver_gate      -- RK4 step convergence (only set if solver sanity ran)

def regression_gate(eval_mse: list[float], oracle_floor: list[float],
                    R_switch_c0: list[float], t_grid: list[float]) -> dict:
    """eval-MSE-based gate. The high-R_sw
    band used here should match the (interior-filtered) primary band used
    elsewhere."""
    mse = np.asarray(eval_mse)
    floor = np.asarray(oracle_floor)
    rsw = np.asarray(R_switch_c0)
    bands = _band_masks(t_grid, rsw)
    band = bands["top40_R_sw"]  # interior-filtered top 40% by R_sw_c0

    integ_mse = mse.sum()
    integ_floor = floor.sum()
    integ_ratio = float(integ_mse / max(integ_floor, 1e-12))

    if band.any():
        band_ratio = float(mse[band].sum() / max(floor[band].sum(), 1e-12))
    else:
        band_ratio = float("nan")

    if (integ_ratio <= GREEN_INTEG_MULT
            and band_ratio <= GREEN_BAND_MULT):
        level = "Green"
    elif integ_ratio <= YELLOW_INTEG_MULT:
        level = "Yellow"
    else:
        level = "Red"

    return {
        "level": level,
        "integ_ratio": integ_ratio,
        "band_ratio": band_ratio,
        "band_indices": [int(i) for i in np.where(band)[0]],
    }


def endpoint_gate(endpoint_stats: dict) -> dict:
    """Endpoint calibration gate.

    The old "purity >= 0.95 at 1.5*sigma" threshold was incorrect: for a
    true 2D Gaussian mode (which is the target distribution), the radial
    mass within 1.5*sigma is only 1 - exp(-1.5^2 / 2) ~ 0.675, so the
    0.95 threshold would reject even a perfectly-calibrated endpoint.

    New gate: max |observed - theoretical 2D Gaussian| over the canonical
    radii {1, 1.5, 2, 3}*sigma. Threshold tuned to seed-SD scale (~0.01).
      Green:  max_dev <  0.02   (within ~2 seed-SDs of theoretical)
      Yellow: max_dev <  0.05
      Red:    max_dev >= 0.05
    Mode proportion deviation from uniform 0.25 is also reported but does
    not gate; informational only."""
    max_dev = endpoint_stats["max_radial_CDF_deviation"]
    if max_dev < 0.02:
        level = "Green"
    elif max_dev < 0.05:
        level = "Yellow"
    else:
        level = "Red"
    return {
        "level": level,
        "max_radial_CDF_deviation": max_dev,
        "radial_mass_1_5_sigma":   endpoint_stats["radial_mass_1_5_sigma"],
        "mean_dist":               endpoint_stats["mean_dist"],
        "q90_dist":                endpoint_stats["q90_dist"],
        "frac_within_1sigma":      endpoint_stats["frac_within_1sigma"],
        "frac_within_1_5sigma":    endpoint_stats["frac_within_1_5sigma"],
        "frac_within_2sigma":      endpoint_stats["frac_within_2sigma"],
        "frac_within_3sigma":      endpoint_stats["frac_within_3sigma"],
        "theoretical_2d_gaussian_frac_1sigma":
            endpoint_stats["theoretical_2d_gaussian_frac_1sigma"],
        "theoretical_2d_gaussian_frac_1_5sigma":
            endpoint_stats["theoretical_2d_gaussian_frac_1_5sigma"],
        "theoretical_2d_gaussian_frac_2sigma":
            endpoint_stats["theoretical_2d_gaussian_frac_2sigma"],
        "theoretical_2d_gaussian_frac_3sigma":
            endpoint_stats["theoretical_2d_gaussian_frac_3sigma"],
        "mode_proportions_fine":   endpoint_stats["mode_proportions_fine"],
        "mode_proportions_coarse": endpoint_stats["mode_proportions_coarse"],
        "mode_max_dev_from_uniform":
            endpoint_stats["mode_max_dev_from_uniform"],
    }


def solver_gate(rk4_sweep: dict | None,
                near_zero_floor: float = 1e-3) -> dict:
    """RK4 solver-step convergence gate.
    rk4_sweep: dict mapping step-count (int) -> {A_v_t05, A_b_t05, purity}
    Green if |X_100 - X_200| / max(|X_100|, |X_200|) < 0.02 on all "meaningful"
    scalars. A metric is "meaningful" if max(|X_100|, |X_200|) >= near_zero_floor
    (default 1e-3 in normalized A_v / probability units); below that the
    relative diff is numerical noise rather than solver disagreement, so
    we report it but skip in the gate decision."""
    if rk4_sweep is None or len(rk4_sweep) < 2:
        return {"level": "NotRun", "note": "solver sanity not run for this seed"}
    if 100 not in rk4_sweep or 200 not in rk4_sweep:
        return {"level": "NotRun", "note": "need both step=100 and step=200"}
    x100 = rk4_sweep[100]
    x200 = rk4_sweep[200]
    rels = {}
    eps = 1e-9
    skipped_for_near_zero = []
    rels_meaningful = []
    for key in ("A_v_t05", "A_b_t05", "purity"):
        v100 = x100[key]
        v200 = x200[key]
        max_abs = max(abs(v100), abs(v200))
        rels[key] = float(abs(v200 - v100) / max(max_abs, eps))
        if max_abs < near_zero_floor:
            skipped_for_near_zero.append(key)
        else:
            rels_meaningful.append(rels[key])
    worst = max(rels_meaningful) if rels_meaningful else 0.0
    if worst < 0.02:
        level = "Green"
    elif worst < 0.05:
        level = "Yellow"
    else:
        level = "Red"
    return {
        "level": level,
        "worst_rel_diff_100_vs_200": worst,
        "near_zero_skipped":         skipped_for_near_zero,
        "per_metric_rel_diff_100_vs_200": rels,
        "values_per_step": {str(k): v for k, v in rk4_sweep.items()},
        "near_zero_floor": near_zero_floor,
    }


def combine_gates(reg: dict, ep: dict, solver: dict) -> str:
    """Worst-of-three (NotRun is treated as 'don't know' = no constraint)."""
    levels = []
    for g in (reg, ep, solver):
        L = g.get("level")
        if L is None or L == "NotRun":
            continue
        levels.append(L)
    if "Red" in levels:
        return "Red"
    if "Yellow" in levels:
        return "Yellow"
    if levels:
        return "Green"
    return "NotRun"


# ---------------------------------------------------------------------------
# Single-seed driver: train, ODE sample, decompose C0/reflow/C1
# ---------------------------------------------------------------------------

def _solver_sweep_quick(model: nn.Module, Z_new: np.ndarray,
                        K_X_c0_for_floor: np.ndarray | None = None,
                        ) -> dict:
    """Lightweight RK4-step sanity sweep: 50/100/200 steps, ONE seed.
    Returns dict {step_count: {A_v_t05, A_b_t05, purity}} for solver_gate.
    Uses the same model and Z_new; only ODE step count varies."""
    out = {}
    for steps in RK4_STEPS_SANITY:
        X_new_s = ode_sample_rk4(model, Z_new, n_steps=steps)
        mode_idx_s, K_X_s, ep_s = endpoint_label(X_new_s)
        # Cheap A_v / A_b at t = 0.5 using fresh kNN (no need for full t-grid)
        Z_dummy = torch.from_numpy(Z_new.astype(np.float32))  # placeholder for math
        # Re-use decompose_one_t at coarse K
        # Need a fake mode_idx (not used in coarse decomposition output we want)
        d05 = decompose_one_t(Z_new, X_new_s, K_X_s, mode_idx_s, t=0.5)
        out[int(steps)] = {
            "A_v_t05":  d05["coarse"]["A_v"],
            "A_b_t05":  d05["coarse"]["A_between"],
            "purity":   ep_s["radial_mass_1_5_sigma"],
        }
    return out


def run_one_seed(train_seed: int, sample_seed: int, N: int, steps: int,
                 t_grid: list[float],
                 rk4_steps: int = RK4_STEPS_DEFAULT,
                 run_solver_sweep: bool = False) -> dict:
    """End-to-end pipeline for one paired (train_seed, sample_seed)."""
    print(f"\n--- seed pair (train={train_seed}, sample={sample_seed}) "
          f"N={N} steps={steps} rk4={rk4_steps}", flush=True)
    t_start = time.time()

    # ---- training pool ----
    Z_tr, X_tr, K_Z_tr, K_X_tr, mode_idx_tr = sample_data_4mode(N, train_seed)

    # ---- train t-conditioned MLP ----
    print(f"   training {steps} steps on {N} pairs ...", flush=True)
    out = train_model(Z_tr, X_tr, train_seed, steps=steps, log_every=1_000)
    model = out["model"]
    train_log = out["train_log"]
    train_wall = out["wall_s"]
    print(f"   train wall: {train_wall:.1f}s   final_loss "
          f"{train_log[-1]['running_loss']:.4f}", flush=True)

    # ---- held-out C0 pool (train_seed + 1000) ----
    c0_seed = train_seed + C0_BASELINE_SEED_OFFSET
    Z_c0, X_c0, K_Z_c0, K_X_c0, mode_idx_c0 = sample_data_4mode(N, c0_seed)

    # ---- per-t eval MSE on the held-out pool ----
    eval_out = eval_per_t_mse(model, Z_c0, X_c0, t_grid)
    eval_mse = eval_out["mse"]

    # ---- fresh Z_new for ODE sampling (sample_seed) ----
    rng_s = np.random.default_rng(sample_seed)
    Z_new = rng_s.standard_normal((N, 2)).astype(np.float64)

    # ---- ODE push ----
    print(f"   ODE sampling RK4 steps={rk4_steps} ...", flush=True)
    t_ode = time.time()
    X_new = ode_sample_rk4(model, Z_new, n_steps=rk4_steps)
    ode_wall = time.time() - t_ode
    print(f"   ODE wall: {ode_wall:.1f}s", flush=True)

    # ---- endpoint labels + full stats (mean / q90 /
    #      frac_within_{1,2,3}sigma) ----
    mode_idx_new, K_X_new, endpoint_stats = endpoint_label(X_new)
    radial_mass = endpoint_stats["radial_mass_1_5_sigma"]
    max_dev = endpoint_stats["max_radial_CDF_deviation"]
    print(f"   endpoint radial_mass(1.5sigma)={radial_mass:.3f}  "
          f"mean_dist={endpoint_stats['mean_dist']:.3f}  "
          f"q90={endpoint_stats['q90_dist']:.3f}  "
          f"max_CDF_dev_vs_2dGauss={max_dev:.3f}", flush=True)

    # ---- C1 OT baseline on the held-out C0 pool (paper-canonical) ----
    rng_c1 = np.random.default_rng(train_seed * 1000 + 1)
    Z_c1, X_c1, K_Z_c1, K_X_c1, mode_idx_c1 = couple_c1_hungarian(
        Z_c0.copy(), X_c0.copy(), K_Z_c0.copy(), K_X_c0.copy(),
        mode_idx_c0.copy(), rng_c1)

    # ---- decomposition on 19-point t-grid for {C0, reflow, C1},
    #      computing BOTH coarse K (XOR) and fine K' (mode_idx) labels
    #      (under coarse K 70% counted as within; fine K' reassigns some).
    #      ~2x compute vs single-label.
    print(f"   decomposing {len(t_grid)} t-points x 3 pipelines "
          f"(coarse + fine) ...", flush=True)
    t_dec = time.time()
    dec_c0 = decompose_full_t_grid(Z_c0, X_c0, K_X_c0, mode_idx_c0, t_grid)
    dec_rf = decompose_full_t_grid(Z_new, X_new, K_X_new, mode_idx_new, t_grid)
    dec_c1 = decompose_full_t_grid(Z_c1, X_c1, K_X_c1, mode_idx_c1, t_grid)
    dec_wall = time.time() - t_dec
    print(f"   decompose wall: {dec_wall:.1f}s", flush=True)

    # ---- B5 scalars on (C0 vs reflow) for BOTH coarse and fine labels.
    #      C1 is passed in to compute the distance_to_c1 block.
    b5_coarse = compute_b5_scalars(t_grid, dec_c0["coarse"], dec_rf["coarse"],
                                   dec_c1["coarse"])
    b5_fine   = compute_b5_scalars(t_grid, dec_c0["fine"],   dec_rf["fine"],
                                   dec_c1["fine"])

    # ---- optional solver sanity sweep on this seed only ----
    rk4_sweep = None
    if run_solver_sweep:
        print(f"   solver sanity sweep RK4 in {RK4_STEPS_SANITY} ...", flush=True)
        t_solver = time.time()
        rk4_sweep = _solver_sweep_quick(model, Z_new)
        print(f"   solver sweep wall: {time.time() - t_solver:.1f}s", flush=True)

    # ---- 3 sub-gates ----
    reg_gate = regression_gate(eval_mse,
                               dec_c0["coarse"]["A_v"],
                               dec_c0["coarse"]["R_switch"],
                               t_grid)
    ep_gate  = endpoint_gate(endpoint_stats)
    sol_gate = solver_gate(rk4_sweep)
    overall  = combine_gates(reg_gate, ep_gate, sol_gate)
    print(f"   quality_gate combined: {overall}  "
          f"(regression={reg_gate['level']}, endpoint={ep_gate['level']}, "
          f"solver={sol_gate['level']})", flush=True)

    # ---- short readout summary ----
    pb_c = b5_coarse["per_band"]
    cor_c = b5_coarse["correlations"]
    print(f"   B5 coarse  interior:   share={pb_c['interior']['share']:+.3f}  "
          f"sat_ref={pb_c['interior']['saturation_share_ref']:.3f}  "
          f"removal_v={pb_c['interior']['removal_v']:+.3f}", flush=True)
    print(f"   B5 coarse  top25:      share={pb_c['top25_R_sw']['share']:+.3f}  "
          f"sat_ref={pb_c['top25_R_sw']['saturation_share_ref']:.3f}", flush=True)
    print(f"   B5 coarse  rho_P={cor_c['pearson_Rsw_DeltaAb']:+.3f}  "
          f"sp={cor_c['spearman_Rsw_DeltaAb']:+.3f}", flush=True)
    pb_f = b5_fine["per_band"]
    cor_f = b5_fine["correlations"]
    print(f"   B5 fine    interior:   share={pb_f['interior']['share']:+.3f}  "
          f"sat_ref={pb_f['interior']['saturation_share_ref']:.3f}  "
          f"removal_v={pb_f['interior']['removal_v']:+.3f}", flush=True)
    print(f"   B5 fine    rho_P={cor_f['pearson_Rsw_DeltaAb']:+.3f}",
          flush=True)
    if b5_coarse["distance_to_c1"] is not None:
        gap_int = b5_coarse["distance_to_c1"]["interior"]
        print(f"   coarse gap-to-C1  interior: gap_Av={gap_int['gap_Av']:+.3f}  "
              f"gap_Ab={gap_int['gap_Ab']:+.3f}", flush=True)
    neg = b5_coarse["negative"]
    print(f"   coarse neg     frac={neg['frac_neg_delta_av_interior']:.3f}  "
          f"mass={neg['neg_mass_delta_av_interior']:+.4f}  "
          f"min_rel={neg['min_rel_delta_av_interior']:+.3f}", flush=True)

    wall = time.time() - t_start
    return {
        "train_seed": int(train_seed),
        "sample_seed": int(sample_seed),
        "N": int(N),
        "steps": int(steps),
        "rk4_steps": int(rk4_steps),
        "train_wall_s": train_wall,
        "ode_wall_s": ode_wall,
        "decompose_wall_s": dec_wall,
        "total_wall_s": wall,
        "train_log": train_log,
        "eval_per_t": eval_out,
        "endpoint_stats": endpoint_stats,
        # decomposition: each baseline has {coarse, fine}
        "dec_c0": dec_c0,
        "dec_reflow": dec_rf,
        "dec_c1": dec_c1,
        # per-band metric block, separate for coarse vs fine
        "b5_coarse": b5_coarse,
        "b5_fine":   b5_fine,
        # 3 sub-gates
        "regression_gate": reg_gate,
        "endpoint_gate":   ep_gate,
        "solver_gate":     sol_gate,
        "overall_gate":    overall,
    }


# ---------------------------------------------------------------------------
# Stage drivers
# ---------------------------------------------------------------------------

def stage_smoke() -> dict:
    """R0 smoke: 1 paired seed, N=5_000, steps=1_000, t=0.5 + 19-grid eval.
    Includes solver-sanity sweep (cheap on small N)."""
    print(f"=== Stage R0 SMOKE  (1 seed, N={N_PAIRS_SMOKE}, "
          f"steps={STEPS_SMOKE}) ===", flush=True)
    t_start = time.time()
    seed_run = run_one_seed(train_seed=0, sample_seed=100,
                            N=N_PAIRS_SMOKE, steps=STEPS_SMOKE,
                            t_grid=T_GRID, rk4_steps=RK4_STEPS_DEFAULT,
                            run_solver_sweep=True)
    elapsed = time.time() - t_start
    print(f"\nR0 SMOKE complete in {elapsed:.1f}s ({elapsed/60:.1f} min)",
          flush=True)
    return {
        "stage": "R0_smoke",
        "config": {
            "N": N_PAIRS_SMOKE, "steps": STEPS_SMOKE,
            "WIDTH": WIDTH, "DEPTH": DEPTH, "T_EMBED_DIM": T_EMBED_DIM,
            "K_KNN": K_KNN, "RK4_STEPS": RK4_STEPS_DEFAULT,
            "T_GRID": T_GRID,
        },
        "wall_s": elapsed,
        "seeds": [seed_run],
    }


def stage_full(train_seeds: list[int], sample_seeds: list[int]) -> dict:
    """R1 full: 5 paired seeds, N=20_000, steps=30k, 19 t-points.
    Solver sanity sweep runs on the FIRST seed only (one-shot
    verification), then subsequent seeds reuse the same RK4 step count
    and reference the first-seed solver_gate output."""
    assert len(train_seeds) == len(sample_seeds), \
        "train_seeds and sample_seeds must be paired (equal length)"
    print(f"=== Stage R1 FULL  ({len(train_seeds)} paired seeds, "
          f"N={N_PAIRS_FULL}, steps={STEPS_FULL}) ===", flush=True)
    t_start = time.time()
    seeds_out = []
    for i, (ts, ss) in enumerate(zip(train_seeds, sample_seeds)):
        seed_run = run_one_seed(train_seed=ts, sample_seed=ss,
                                N=N_PAIRS_FULL, steps=STEPS_FULL,
                                t_grid=T_GRID, rk4_steps=RK4_STEPS_DEFAULT,
                                run_solver_sweep=(i == 0))
        seeds_out.append(seed_run)
        gc.collect()
    elapsed = time.time() - t_start
    print(f"\nR1 FULL complete in {elapsed:.1f}s ({elapsed/60:.1f} min)",
          flush=True)
    return {
        "stage": "R1_full",
        "config": {
            "N": N_PAIRS_FULL, "steps": STEPS_FULL,
            "WIDTH": WIDTH, "DEPTH": DEPTH, "T_EMBED_DIM": T_EMBED_DIM,
            "K_KNN": K_KNN, "RK4_STEPS": RK4_STEPS_DEFAULT,
            "T_GRID": T_GRID,
            "train_seeds": train_seeds, "sample_seeds": sample_seeds,
            "C0_BASELINE_SEED_OFFSET": C0_BASELINE_SEED_OFFSET,
        },
        "wall_s": elapsed,
        "seeds": seeds_out,
    }


def stage_ode_sanity() -> dict:
    """R2 ODE solver sanity: one seed, RK4 step counts {50, 100, 200}.
    Trains once at the smoke setting, then samples three times at different
    step counts; confirms the diagnostic is solver-converged at 100."""
    print(f"=== Stage R2 ODE SANITY  (1 seed, N={N_PAIRS_SMOKE}, "
          f"steps={STEPS_SMOKE}, RK4 in {RK4_STEPS_SANITY}) ===",
          flush=True)
    t_start = time.time()
    train_seed = 0
    sample_seed = 100
    runs = []
    for rk4 in RK4_STEPS_SANITY:
        seed_run = run_one_seed(train_seed=train_seed, sample_seed=sample_seed,
                                N=N_PAIRS_SMOKE, steps=STEPS_SMOKE,
                                t_grid=T_GRID, rk4_steps=rk4)
        runs.append(seed_run)
        gc.collect()
    elapsed = time.time() - t_start
    print(f"\nR2 ODE SANITY complete in {elapsed:.1f}s ({elapsed/60:.1f} min)",
          flush=True)
    return {
        "stage": "R2_ode_sanity",
        "config": {
            "N": N_PAIRS_SMOKE, "steps": STEPS_SMOKE,
            "WIDTH": WIDTH, "DEPTH": DEPTH, "T_EMBED_DIM": T_EMBED_DIM,
            "K_KNN": K_KNN, "RK4_STEPS_SWEEP": RK4_STEPS_SANITY,
            "T_GRID": T_GRID,
        },
        "wall_s": elapsed,
        "rk4_sweep": runs,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True,
                        choices=["R0", "R1", "R2"],
                        help="experiment stage to run")
    parser.add_argument("--train-seeds", type=str, default=None,
                        help="comma-separated train seeds (default: stage default)")
    parser.add_argument("--sample-seeds", type=str, default=None,
                        help="comma-separated sample seeds (default: stage default)")
    parser.add_argument("--out-tag", type=str, default=None,
                        help="suffix for output JSON filename")
    args = parser.parse_args()

    out_dir = Path(__file__).resolve().parents[2] / "results"
    out_dir.mkdir(exist_ok=True)

    print(f"e_reflow_conjecture1 driver  stage={args.stage}  device={DEVICE.type}")
    print(f"  WIDTH={WIDTH} DEPTH={DEPTH} T_EMBED_DIM={T_EMBED_DIM}")
    print(f"  K_KNN={K_KNN} RK4_STEPS={RK4_STEPS_DEFAULT}")
    print(f"  OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}")
    print("-" * 78)

    if args.stage == "R0":
        result = stage_smoke()
    elif args.stage == "R1":
        train_seeds = (TRAIN_SEEDS_FULL if args.train_seeds is None
                       else [int(s) for s in args.train_seeds.split(",")])
        sample_seeds = (SAMPLE_SEEDS_FULL if args.sample_seeds is None
                        else [int(s) for s in args.sample_seeds.split(",")])
        result = stage_full(train_seeds, sample_seeds)
    elif args.stage == "R2":
        result = stage_ode_sanity()
    else:
        raise ValueError(args.stage)

    # Output names are descriptive (no internal run code in the filename);
    # --stage keeps its short code as an internal CLI selector only.
    stage_out = {
        "R0": "reflow_conjecture1_smoke.json",
        "R1": "reflow_conjecture1.json",
        "R2": "reflow_conjecture1_ode_sanity.json",
    }
    if args.out_tag:
        out_path = out_dir / f"reflow_conjecture1_{args.out_tag}.json"
    else:
        out_path = out_dir / stage_out[args.stage]
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=float)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
