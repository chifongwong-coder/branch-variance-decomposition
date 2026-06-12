# Branch-variance decomposition (BVD)

*Diagnostics for Flow Matching and diffusion generative models.*

**Branch-variance decomposition (BVD)** splits the branch-conditional second
moment of the generative regression target (the velocity-variance floor in
Flow Matching, the score covariance in diffusion) into **within-branch** and
**between-branch** components. The between-branch share
`R_switch(t) = A_between / A_v` turns the choice of source-target coupling (and
of the representation in which branches are defined) into a single *measurable*
quantity.

This repository ships **code**, plus the paper's figures as vector **PDF** under
`figures/`. No experiment outputs, caches, or datasets are committed: every
script regenerates its own metrics under `results/` and its raster figures under
`figures/` when you run it (both directories are created on first run; runtime
outputs there are git-ignored, only the shipped `figures/*.pdf` are tracked). The
two exceptions are the schematic (`fig1_schematic`) and the image teaser
(`image_branch_ambiguity_teaser`), which are hand-built illustrative figures
shipped as vector PDF without a data-driven regenerator.

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.10+. The toy experiments (E0-E3c) need only
NumPy/SciPy/POT; the neural experiments (E4, E6) add PyTorch; the real-data and
real-model experiments (E5, E5a, E7) additionally need `torchvision` (CIFAR-10),
`transformers` (CLIP ViT-B/32), and `faiss-cpu` (exact kNN at d=512).

## Data and models

- **Dataset: CIFAR-10** (public). Downloaded automatically by `torchvision` on
  the first run of any E5/E5a/E7 script (~170 MB, cached under `results/`). No
  manual download or licensing step.
- **Pretrained encoder: CLIP ViT-B/32** (public). Downloaded
  automatically from the Hugging Face Hub on first run (~600 MB). Used only as a
  frozen feature extractor and zero-shot posterior; it is never fine-tuned.
- **Trained Flow-Matching checkpoints** (the unconditional and class-conditional
  CIFAR FM models that E5's real-path audit and all of E7 consume). These are
  **not** shipped (~29 MB each, git-ignored). You have two options:
  1. **Train them yourself** with the E7 scripts (see [Neural pipelines](#neural-pipelines-cifar-10--clip--trained-fm-models)):
     `src/E7/e7_cifar_fm_train_labelfree.py` and `src/E7/e7_cifar_fm_train.py`
     (about 1.5-3 h each on an Apple M3 or a GPU), or
  2. **Request the trained weights from the authors.**

The toy experiments (E0-E4, E6) use **no external data**: their inputs are
synthetic and generated in-code from fixed seeds, so they reproduce with nothing
to download.

## Repository layout

Code lives under `src/`, grouped by experiment; shared modules are in `src/core/`:

```
src/core/                 branch_decomp.py (within/between estimator), figure_style.py
src/E0/ ... src/E7/       one directory per experiment (runner + plotter + build scripts)
src/paper_figures/        cross-experiment replotters
```

Scripts read and write the shared `results/` and `figures/` directories at the
**repository root** (so experiments that reuse another's caches or checkpoints,
e.g. E5 reusing E7's, find them).

## How to run

Every experiment is a standalone script, run from the repo root by path:

```bash
python3 src/E2/e2_crossing_velocity.py      # runs E2 -> results/e2_metrics.json
python3 src/E1/plot_e1_from_cache.py        # a plotter  -> figures/
```

Two runners select sub-stages with `--stage`:

- `python3 src/E3c/e_phase_diagram.py --stage {P0,P1,P2,P4a,P4b}` (geometry-semantics sweep; one stage runs the headline 5x5 grid, the others are satellite checks).
- `python3 src/E6/e_reflow_conjecture1.py --stage {R0,R1,R2}` (reflow-locality check; one stage is the main paired-seed run, the others are smoke and solver-sanity checks).

The result-file tag is set at the top of each runner (a `TAG` constant, or the
`E3C_TAG` environment variable for E3); the plot scripts expect the tag the
paper used, so match them when regenerating (see each runner's header).

## Experiment map

Couplings are `C0` (independent), `C1` (Euclidean OT), `C2` (label-correct
random within-condition), `C3@lambda` (semantic-cost OT), `C3^inf` (the
hard-blocked `lambda -> inf` limit), `C4` (geometry-only OT). Each runner /
plotter below lives in `src/<Name>/` (shared modules in `src/core/`).

| Name | What it measures | Runner(s) | Plotter(s) |
|------|------------------|-----------|------------|
| **E0** | Math / implementation sanity (no figure) | `e0_math_sanity.py` | - |
| **E1** | Binary-Gaussian diffusion oracle: closed-form vs empirical responsibility-switching curvature and entropy transition over a finite commitment window | `e1_diffusion_binary_gaussian.py` | `plot_e1_from_cache.py` |
| **E2** | Crossing-path FM oracle: closed-form floor within 0.025% (+ seed stability) | `e2_crossing_velocity.py`, `e2_seed_stability_check.py` | `plot_e2.py`, `paper_figures/replot_e3_e2_from_cache.py` |
| **E3a** | Coupling comparison, binary target (Euclidean OT removes ambiguity when geometry encodes branches) | `e3a_coupling_comparison.py` | `plot_e3a.py`, `paper_figures/replot_remaining_paper_figures.py` |
| **E3b** | Branch-refinement sanity on the 4-mode XOR target | `e3b_branch_refinement.py` | `plot_e3b_main_v7.py`, `plot_e3b_refinement.py` |
| **E3** | 33-D conditional-mismatch toy; C0-C4 + Sinkhorn couplings; the 0.50 -> 0.04 mechanism result | `e3_coupling_comparison.py`, `analyse_c3_margin.py` | `plot_e3_results.py`, `paper_figures/replot_e3_e2_from_cache.py`, `build_sinkhorn_comparison_figure.py`, `build_sinkhorn_appendix_table.py`, `build_bandwidth_table.py` |
| **E3c** | Geometry-semantics scaling sweep (phase diagram over `(D_G, a, sigma_s)`) | `e_phase_diagram.py` | `plot_phase_diagram.py`, `plot_phase_diagram_extras.py`, `plot_max_t_sensitivity.py` |
| **E4** | Fixed-t FM MLP: capacity ladder saturating the floor; within/between decomposition | `e4_fm_mlp_fixed_t.py`, `e4_s3_within_between_decomp.py` | `plot_e4.py`, `paper_figures/replot_remaining_paper_figures.py`, `build_e4_supplement_composite.py` |
| **E5** | Real-data branch validation: CIFAR-10 CLIP features + a trained unconditional FM model; cardinality-debiased proxy recovery (~1.0 across K=2/3/10 once the kNN bandwidth tracks K, see E5b) and the model-grounded real-path audit | `e5_realdata_validation.py`, `e5_ablation_neighborhood.py`, `e5_checkpoint_robustness.py` | `plot_e5_realdata_validation.py` |
| **E5a** | Real-feature proxy-quality + L^1-stability companion on CIFAR-10 CLIP features | `e5a_diagnostic.py` (+ `e5a_extract_clip_features.py`, `e5a_prepare_branches.py`, `e5a_fast.py`) | `plot_e5a_diagnostic.py`, `plot_e5a_bound_test.py`, `plot_e5a_quality.py` |
| **E5b** | Bandwidth scales with branch cardinality: the de-biasing null grows as K/k, so k must grow with K to hold the neighbors-per-branch ratio; at 20-40 neighbors per branch the proxy recovery stays ~1.0 across K=2/3/10, while a single fixed k depresses it at fine K. Feeds the recovery bars in E5 figure panel (a) | `expE5b_recovery_bandwidth.py`, `make_e5_recovery_aggregate.py` | (aggregate -> `e5_recovery_vs_cardinality.json`) |
| **E6** | Reflow-locality check: where one reflow iteration reduces between-branch switching | `e_reflow_conjecture1.py` | `plot_reflow.py`, `plot_reflow_per_mode_scatter.py`, `stitch_reflow_composite.py` |
| **E7** | Real-model branch-conditioning payoff on trained CIFAR FM checkpoints: the gain identity `A_v - A_within = A_between`, posterior-weighted spread predicts the conditioning gain across three couplings (C0/C1/C3inf, cosine >= 0.9998), each evaluated in-distribution on its own coupling's targets | `e7_cifar_fm_train.py`, `e7_cifar_fm_train_labelfree.py`, `e7_cifar_fm_couplings.py`, `e7_cifar_fm_diagnostics.py`, `e7_conditioning_gain_payoff.py`, `e7_bvd_granularity_proxy.py`, `e7_bvd_sensitivity_check.py` | `plot_e7_conditioning_gain_payoff.py`, `plot_e7.py`, `plot_e5_e7_robustness.py` |
| **E7a** | Independent kNN estimate of A_between(t) vs the realized conditioning gain (uses none of the conditional model's velocities), with a large-N resolution follow-up | `expE7a_independent_abetween.py`, `verify_e7a_largeN.py` | - |

### Shared modules

- `branch_decomp.py` - the core label-agnostic local within/between decomposition
  of the conditional-velocity-variance floor (biased `1/k` convention, so the
  split is exactly additive). Imported by most experiments.
- `figure_style.py` - unified matplotlib style and coupling colour palette.

### Neural pipelines (CIFAR-10 + CLIP / trained FM models)

E5, E5a, and E7 use CIFAR-10 and a CLIP image encoder, and E5/E7 use trained
Flow-Matching checkpoints. Build the prerequisites first:

```bash
# 1) CLIP feature cache (shared by E5a AND E5 real-data) + the E5a diagnostic
python3 src/E5a/e5a_extract_clip_features.py    # downloads CIFAR-10 + CLIP ViT-B/32 -> results/cifar10_clip_features.npz
python3 src/E5a/e5a_prepare_branches.py         # branch labels from the features above
python3 src/E5a/e5a_diagnostic.py               # E5a proxy diagnostic + L^1 bound test

# 2) Train the CIFAR FM checkpoints (each trainer REQUIRES --coupling; writes results/*.pt)
python3 src/E7/e7_cifar_fm_train_labelfree.py --coupling c0    --seed 0  # -> e7_uncond_c0_seed0.pt (+ step ckpts every 10k; needed by E5)
python3 src/E7/e7_cifar_fm_train_labelfree.py --coupling c1    --seed 0  # -> e7_uncond_c1_seed0.pt    (robustness leg)
python3 src/E7/e7_cifar_fm_train_labelfree.py --coupling c3inf --seed 0  # -> e7_uncond_c3inf_seed0.pt (robustness leg)
python3 src/E7/e7_cifar_fm_train.py --coupling c0    --seed 0            # -> e7_c0_seed0.pt
python3 src/E7/e7_cifar_fm_train.py --coupling c1    --seed 0            # -> e7_c1_seed0.pt    (robustness leg)
python3 src/E7/e7_cifar_fm_train.py --coupling c3inf --seed 0            # -> e7_c3inf_seed0.pt (robustness leg)

# 3) E7 payoff + robustness (consume the checkpoints from step 2)
python3 src/E7/e7_conditioning_gain_payoff.py                                    # C0: uses e7_uncond_c0_seed0.pt + e7_c0_seed0.pt
python3 src/E7/e7_conditioning_gain_payoff.py --coupling c1    --apply-coupling  # C1 in-distribution: e7_uncond_c1_seed0.pt + e7_c1_seed0.pt
python3 src/E7/e7_conditioning_gain_payoff.py --coupling c3inf --apply-coupling  # C3inf in-distribution: e7_uncond_c3inf_seed0.pt + e7_c3inf_seed0.pt

# 4) E5 real-data validation (reuses the step-1 CLIP cache + the step-2 label-free checkpoint)
python3 src/E5/e5_realdata_validation.py           # -> results/e5_realdata_validation.json (figure panels b,c,d)
python3 src/E5/e5_checkpoint_robustness.py         # uses the 10k/30k/50k step checkpoints from step 2

# 5) E5b model-free recovery vs bandwidth across cardinality (figure panel a; CLIP cache only, no model)
python3 src/E5b/expE5b_recovery_bandwidth.py --K 2  --ks 40,80     # resumable per-seed -> results/expE5b_recovery_bandwidth.json
python3 src/E5b/expE5b_recovery_bandwidth.py --K 3  --ks 60,120
python3 src/E5b/expE5b_recovery_bandwidth.py --K 10 --ks 200,400
python3 src/E5b/make_e5_recovery_aggregate.py      # -> results/e5_recovery_vs_cardinality.json
python3 src/E5/plot_e5_realdata_validation.py      # renders the 4-panel E5 figure
```

The trained `*.pt` checkpoints are not committed (git-ignored, ~29 MB each); the
training scripts above regenerate them (and the label-free trainer also writes a
checkpoint every 10k steps, which `e5_checkpoint_robustness.py` consumes). The
published checkpoints are available from the authors on request.

**Checkpoint configuration.** The commands above use the trainer defaults, which
are the configuration behind the paper's checkpoints: `base_ch=96`
(~3.85 M parameters), batch 128, 50k steps, AdamW with `lr=2e-4` and weight decay
`1e-4`, EMA decay 0.999, and `--lambda-sem 10` for the semantic-cost couplings
(`c3` / `c3inf`). The conditional `CondUNet` is the label-free `UncondUNet` plus a
10-class embedding, so the two are matched twins up to that embedding. These values
are the trainer defaults in `e7_cifar_fm_train.py` and
`e7_cifar_fm_train_labelfree.py` (`weight_decay` is hard-coded; the rest are CLI
flags); override them with `--base-ch`, `--batch`,
`--steps`, `--lr`, `--ema`, and `--lambda-sem` to vary the configuration. The E7
payoff itself (`e7_conditioning_gain_payoff.py`) is forward-pass only and uses
`N=2000` evaluation samples over 3 seeds.

### Hardware and first run

The PyTorch experiments (E4, E6, E5/E5a CLIP forward, E7 training) auto-select a
device: CUDA, then Apple MPS, else CPU. The toy experiments (E0-E3c) are CPU-only
NumPy/SciPy. First run of the CIFAR/CLIP pipelines downloads CIFAR-10 (~170 MB)
and CLIP ViT-B/32 (~600 MB) from the Hugging Face Hub, so they need outbound
internet and ~1 GB of free disk for caches under `results/`. Runtimes quoted in
the scripts are measured on an Apple M3 Pro with MPS.

### Sanity check

`python3 src/E2/e2_crossing_velocity.py` is the quickest way to confirm a working setup: CPU-only,
single-seed, deterministic, a couple of minutes. At `t = 0.5` it reports a peak
`A_v = 2.0150` (closed form `2.0400`, matching to within `0.025%` after the
`k/(k-1)` bandwidth correction), `R_switch = 0.98`, and a within/between split of
`0.0391 / 1.9758`. The pure-NumPy/SciPy toys (E0-E3c) reproduce bitwise;
the PyTorch parts (E4, E6, E5/E5a CLIP forward, E7) are seeded but reproduce
within seed-level variation across CUDA / MPS / CPU and BLAS builds.

## License

Released under the MIT License (see `LICENSE`).
