"""E1: Binary Gaussian diffusion oracle.

Verify that the predicted critical SNR window for a 1D binary Gaussian
mixture appears as joint peaks in:
  - |d H_bar / d log SNR|  (entropy transition)
  - average responsibility-switching curvature
  - average |total score Jacobian|

Setting:
  X | K=k ~ N(k m, s^2),  k in {-1, +1}, equal prior.
  X_sigma = X + sigma * eps,  eps ~ N(0, 1).
  Component variance after noising: tau^2 = s^2 + sigma^2.
  log SNR := -2 log sigma  (VE-style; up to a constant if signal var != 1).

For 1D binary mixture:
  posterior  r_+(x) = sigmoid(2 m x / tau^2),  r_- = 1 - r_+.
  branch entropy  H(K|x) = -r_+ log r_+ - r_- log r_-.
  mixture score  s(x) = (m tanh(m x / tau^2) - x) / tau^2.
  score derivative
    d/dx s(x)
      = -1/tau^2  +  (m^2 / tau^4) sech^2(m x / tau^2)
      = within  +  switching,
  with within = -1/tau^2 (constant, both components share J_k = -1/tau^2)
  and switching = Var_K|x(s_K(x)) = r_+ r_- (2 m / tau^2)^2.

Outputs:
  figures/e1_binary_oracle.png   (4-panel summary)
  figures/e1_curves.png          (single combined log-y plot)
  results/e1_metrics.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.figure_style import apply_paper_style, Palette
apply_paper_style()

# Config
M = 2.0
S = 0.25
N_SAMPLES = 200_000
SIGMA_LOG_RANGE = (-2.0, 2.0)  # log10 sigma
SIGMA_N = 121
SEED = 0
EPS = 1e-12


def metrics_at_sigma(x_clean, sigma, m=M, s=S, rng=None):
    rng = rng if rng is not None else np.random.default_rng()
    tau2 = s ** 2 + sigma ** 2
    noise = rng.standard_normal(x_clean.shape)
    x_noisy = x_clean + sigma * noise

    # logit P(K=+1|x) = 2 m x / tau^2
    a = 2.0 * m * x_noisy / tau2
    r_plus = 1.0 / (1.0 + np.exp(-a))
    r_minus = 1.0 - r_plus

    H = -(r_plus * np.log(np.clip(r_plus, EPS, 1.0))
          + r_minus * np.log(np.clip(r_minus, EPS, 1.0)))

    # switching curvature per sample
    sech2 = 1.0 / np.cosh(m * x_noisy / tau2) ** 2
    switching = (m ** 2) * sech2 / (tau2 ** 2)

    within = -1.0 / tau2  # scalar
    total_jac = within + switching

    return {
        "sigma": float(sigma),
        "tau2": float(tau2),
        "log_snr": float(-2.0 * np.log(sigma)),
        "rho": float(m / np.sqrt(tau2)),
        "H_bar": float(H.mean()),
        "H_at_boundary": float(np.log(2.0)),
        "switching_avg": float(switching.mean()),
        "switching_max_per_sample": float(switching.max()),
        "switching_at_x0": float((m ** 2) / (tau2 ** 2)),  # x=0 analytic peak
        "within": float(within),
        "total_jac_avg": float(total_jac.mean()),
        "abs_total_jac_avg": float(np.abs(total_jac).mean()),
    }


def find_rho_eq_one_sigma(m=M, s=S):
    # rho = m / sqrt(s^2 + sigma^2) = 1  =>  sigma^2 = m^2 - s^2
    val = m ** 2 - s ** 2
    return float(np.sqrt(val)) if val > 0 else float("nan")


def run():
    rng = np.random.default_rng(SEED)

    # Clean samples
    k_clean = rng.choice([-1, 1], size=N_SAMPLES)
    x_clean = k_clean * M + S * rng.standard_normal(N_SAMPLES)

    sigmas = np.logspace(SIGMA_LOG_RANGE[0], SIGMA_LOG_RANGE[1], SIGMA_N)
    rows = [metrics_at_sigma(x_clean, sig, rng=rng) for sig in sigmas]

    log_snr = np.array([r["log_snr"] for r in rows])
    log_sigma = np.array([np.log(r["sigma"]) for r in rows])
    rho = np.array([r["rho"] for r in rows])
    H_bar = np.array([r["H_bar"] for r in rows])
    switching_avg = np.array([r["switching_avg"] for r in rows])
    switching_x0 = np.array([r["switching_at_x0"] for r in rows])
    within = np.array([r["within"] for r in rows])
    total_jac_avg = np.array([r["total_jac_avg"] for r in rows])
    abs_total_jac_avg = np.array([r["abs_total_jac_avg"] for r in rows])

    dH_dlogsnr = np.abs(np.gradient(H_bar, log_snr))

    sigma_star = find_rho_eq_one_sigma()
    log_snr_star = float(-2.0 * np.log(sigma_star))

    out_dir = Path(__file__).resolve().parents[2]
    fig_dir = out_dir / "figures"
    res_dir = out_dir / "results"
    fig_dir.mkdir(exist_ok=True)
    res_dir.mkdir(exist_ok=True)

    # 4-panel summary
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    ax = axes[0, 0]
    ax.plot(log_snr, H_bar, lw=2)
    ax.axvline(log_snr_star, color="red", linestyle="--",
               label=fr"$\rho{{=}}1$ ($\log\mathrm{{SNR}}{{=}}{log_snr_star:.2f}$)")
    ax.axhline(np.log(2.0), color="gray", linestyle=":", label=r"$\log 2$")
    ax.set_xlabel(r"$\log\mathrm{SNR}=-2\log\sigma$")
    ax.set_ylabel(r"$\bar H_K(\sigma)$")
    ax.set_title("Branch entropy")
    ax.legend(loc="best", fontsize=9)

    ax = axes[0, 1]
    ax.plot(log_snr, dH_dlogsnr, lw=2)
    ax.axvline(log_snr_star, color="red", linestyle="--")
    ax.set_xlabel(r"$\log\mathrm{SNR}$")
    ax.set_ylabel(r"$|\,d\bar H_K/d\log\mathrm{SNR}\,|$")
    ax.set_title("Entropy transition rate")

    ax = axes[1, 0]
    ax.plot(log_snr, switching_avg, lw=2, label=r"avg $r_+r_-(2m/\tau^2)^2$")
    ax.plot(log_snr, switching_x0, lw=1.2, ls="--",
            label=r"analytic peak at $x{=}0$: $m^2/\tau^4$")
    ax.axvline(log_snr_star, color="red", linestyle="--")
    ax.set_xlabel(r"$\log\mathrm{SNR}$")
    ax.set_ylabel("switching curvature")
    ax.set_title("Responsibility-switching curvature")
    ax.set_yscale("symlog", linthresh=1e-3)
    ax.legend(loc="best", fontsize=9)

    ax = axes[1, 1]
    ax.plot(log_snr, switching_avg, lw=2, label="switching avg")
    ax.plot(log_snr, -within, lw=1.5, label=r"$1/\tau^2$ (within mag.)")
    ax.plot(log_snr, abs_total_jac_avg, lw=1.5,
            label=r"$|\,J_s\,|$ avg (total)")
    ax.axvline(log_snr_star, color="red", linestyle="--")
    ax.set_xlabel(r"$\log\mathrm{SNR}$")
    ax.set_ylabel("Jacobian magnitude")
    ax.set_title("Jacobian decomposition")
    ax.set_yscale("log")
    ax.legend(loc="best", fontsize=9)

    fig.suptitle(
        f"E1 Binary Gaussian diffusion oracle  (m={M}, s={S}, N={N_SAMPLES})",
        fontsize=12)
    fig.tight_layout()
    fig.savefig(fig_dir / "e1_binary_oracle.png", dpi=130)
    plt.close(fig)

    # Combined log-y curve in a single panel for paper-style readability
    fig2, ax = plt.subplots(figsize=(7.5, 5))
    # Rescale so all curves fit; normalize each to its max within its window
    norm_dH = dH_dlogsnr / max(dH_dlogsnr.max(), 1e-12)
    norm_switch = switching_avg / max(switching_avg.max(), 1e-12)
    ax.plot(log_snr, H_bar / np.log(2.0), label=r"$\bar H_K / \log 2$", lw=2)
    ax.plot(log_snr, norm_dH,
            label=r"$|d\bar H_K/d\log\mathrm{SNR}|$ (normalized)", lw=2)
    ax.plot(log_snr, norm_switch,
            label="switching curvature (normalized)", lw=2)
    ax.axvline(log_snr_star, color="red", linestyle="--",
               label=fr"$\rho=1$ ($\log\mathrm{{SNR}}{{=}}{log_snr_star:.2f}$)")
    ax.set_xlabel(r"$\log\mathrm{SNR}=-2\log\sigma$")
    ax.set_ylabel("normalized magnitude (per curve)")
    ax.set_title("E1: critical window indicators")
    ax.legend(loc="best", fontsize=9)
    fig2.tight_layout()
    fig2.savefig(fig_dir / "e1_curves.png", dpi=130)
    plt.close(fig2)

    summary = {
        "config": {
            "m": M, "s": S, "n_samples": N_SAMPLES,
            "seed": SEED,
            "sigma_log10_range": SIGMA_LOG_RANGE, "sigma_n": SIGMA_N,
        },
        "predicted_rho_eq_1": {
            "sigma_star": sigma_star,
            "log_snr_star": log_snr_star,
        },
        "argmax_dH": {
            "log_snr": float(log_snr[int(np.argmax(dH_dlogsnr))]),
            "sigma": float(sigmas[int(np.argmax(dH_dlogsnr))]),
            "value": float(dH_dlogsnr.max()),
        },
        "argmax_switching": {
            "log_snr": float(log_snr[int(np.argmax(switching_avg))]),
            "sigma": float(sigmas[int(np.argmax(switching_avg))]),
            "value": float(switching_avg.max()),
        },
        "H_bar_endpoints": [float(H_bar[0]), float(H_bar[-1])],
        "curves": {
            "log_snr": log_snr.tolist(),
            "log_sigma": log_sigma.tolist(),
            "rho": rho.tolist(),
            "H_bar": H_bar.tolist(),
            "dH_dlogsnr": dH_dlogsnr.tolist(),
            "switching_avg": switching_avg.tolist(),
            "switching_at_x0": switching_x0.tolist(),
            "within": within.tolist(),
            "total_jac_avg": total_jac_avg.tolist(),
            "abs_total_jac_avg": abs_total_jac_avg.tolist(),
        },
    }
    with open(res_dir / "e1_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Console report
    print("E1 binary Gaussian diffusion oracle")
    print("-" * 60)
    print(f"  config: m={M}, s={S}, N={N_SAMPLES}")
    print(f"  rho=1 predicted at sigma*={sigma_star:.3f}, "
          f"log SNR*={log_snr_star:.3f}")
    # sigmas[0] is smallest sigma (high SNR), sigmas[-1] is largest (low SNR).
    print(f"  H_bar: {H_bar[0]:.3f} at sigma={sigmas[0]:.3f} (high SNR) "
          f"-> {H_bar[-1]:.3f} at sigma={sigmas[-1]:.3f} (low SNR)")
    print(f"  argmax |dH/dlog SNR| at log SNR = "
          f"{summary['argmax_dH']['log_snr']:.3f} "
          f"(sigma={summary['argmax_dH']['sigma']:.3f})")
    print(f"  argmax switching curvature at log SNR = "
          f"{summary['argmax_switching']['log_snr']:.3f} "
          f"(sigma={summary['argmax_switching']['sigma']:.3f})")
    print(f"  figures -> {fig_dir}")
    print(f"  metrics -> {res_dir / 'e1_metrics.json'}")


if __name__ == "__main__":
    run()
