# TFT Autoresearch — Experiment Report (2026-05-25)

**Prepared by:** Karthik Mattu
**Date:** May 25, 2026
**Cluster:** RIT Research Computing (sporcsubmit), A100 GPUs

---

## Executive Summary

Ran 10 experiments (exp_029 through exp_038) exploring the low-alpha regime under
the revised constraint caps. **SmoothAsymmetricHuberLoss with delta=0.1, alpha=1.5
(exp_031) is the new best**, achieving composite_score = 777,320 — the first
experiment to pass both revised constraints. Follow-up fine-tuning (batch 3)
confirmed this config sits on a narrow sweet spot — every neighboring perturbation
fails the FP cap.

---

## Constraint Caps (Revised)

| Metric | Baseline | Cap | Multiplier |
|--------|----------|-----|------------|
| best_test_mse | 0.002252 | 0.003378 | 1.5x |
| focus_fp | 806,408 | 967,690 | 1.2x |

The FP cap was tightened from 2.0x to 1.2x by Farhan, invalidating all prior
experiments from exp_002 onward (all had focus_fp > 967,690).

---

## Today's Experiments — Full Results

### Batch 1 (exp_029 to exp_032)

| ID | Loss Function | alpha | delta | wd | MSE | focus_fp | focus_fn | composite | Pearson | Best Epoch | MSE ok? | FP ok? | Status |
|----|--------------|-------|-------|----|-----|----------|----------|-----------|---------|------------|---------|--------|--------|
| exp_029 | AsymmetricMSE | 1.5 | — | 0 | 0.002620 | 962,397 | 901,827 | 901,827 | 0.9809 | 82 | PASS | PASS | ok |
| exp_030 | AsymmetricMSE | 1.75 | — | 0 | 0.002682 | 1,136,198 | 832,984 | — | 0.9805 | 45 | PASS | FAIL | constraint_fail |
| **exp_031** | **SmoothHuber** | **1.5** | **0.1** | **0** | **0.002314** | **853,856** | **777,320** | **777,320** | **0.9831** | **94** | **PASS** | **PASS** | **ok+improved** |
| exp_032 | AsymmetricMSE | 1.5 | — | 1e-6 | 0.003171 | 1,198,014 | 986,389 | — | 0.9767 | 44 | PASS | FAIL | constraint_fail |

### Batch 2 (exp_033 to exp_035)

| ID | Loss Function | alpha | delta | wd | MSE | focus_fp | focus_fn | composite | Pearson | Best Epoch | MSE ok? | FP ok? | Status |
|----|--------------|-------|-------|----|-----|----------|----------|-----------|---------|------------|---------|--------|--------|
| exp_033 | AsymmetricMSE | 1.625 | — | 0 | 0.002319 | 995,962 | 794,722 | — | 0.9831 | 63 | PASS | FAIL | constraint_fail |
| exp_034 | SmoothHuber | 1.5 | 0.2 | 0 | 0.003651 | 1,250,682 | 1,073,037 | — | 0.9732 | 53 | FAIL | FAIL | constraint_fail |
| exp_035 | SmoothHuber | 1.25 | 0.1 | 0 | 0.002578 | 923,180 | 872,477 | 872,477 | 0.9811 | 65 | PASS | PASS | ok |

All experiments used: lr=1e-4, batch_size=32, dropout=0.0, grad_clip=0.5, warmup+cosine schedule, patience=40.

---

## Experiments That Pass Both Constraints

Only 3 of 10 experiments pass both revised constraints:

| Rank | ID | Loss | alpha | delta | composite | MSE | focus_fp | Pearson |
|------|----|------|-------|-------|-----------|-----|----------|---------|
| 1 | **exp_031** | SmoothHuber | 1.5 | 0.1 | **777,320** | 0.002314 | 853,856 | 0.9831 |
| 2 | exp_035 | SmoothHuber | 1.25 | 0.1 | 872,477 | 0.002578 | 923,180 | 0.9811 |
| 3 | exp_029 | AsymmetricMSE | 1.5 | — | 901,827 | 0.002620 | 962,397 | 0.9809 |

---

## Key Findings

### 1. SmoothHuber outperforms AsymmetricMSE at matching alpha

At alpha=1.5, SmoothHuber (exp_031, composite=777,320) beats AsymmetricMSE
(exp_029, composite=901,827) by 124k FN — a 14% improvement. The Huber loss's
linear tail on large residuals provides better gradient behavior.

### 2. delta=0.1 is the right Huber width; delta=0.2 fails

| delta | MSE | focus_fp | Status |
|-------|-----|----------|--------|
| 0.1 | 0.002314 | 853,856 | PASS |
| 0.2 | 0.003651 | 1,250,682 | FAIL (both) |

delta=0.2 makes the quadratic zone too wide, degrading both MSE and FP.
delta=0.1 remains the sweet spot for targets in [0,1].

### 3. AsymmetricMSE alpha boundary is between 1.5 and 1.625

| alpha | focus_fp | FP cap (967,690) | Status |
|-------|----------|-------------------|--------|
| 1.5 | 962,397 | PASS (by 5k) | ok |
| 1.625 | 995,962 | FAIL (by 28k) | constraint_fail |
| 1.75 | 1,136,198 | FAIL (by 169k) | constraint_fail |

AsymmetricMSE hits the FP wall very quickly above alpha=1.5.

### 4. Lowering alpha trades FP headroom for worse FN (composite)

| alpha | Loss | focus_fp | focus_fn (composite) |
|-------|------|----------|---------------------|
| 1.25 | SmoothHuber | 923,180 | 872,477 |
| 1.5 | SmoothHuber | 853,856 | 777,320 |

alpha=1.5 dominates alpha=1.25 on both metrics: lower FP (854k vs 923k,
more headroom under the 968k cap) and lower composite (777k vs 872k).
Reducing alpha below 1.5 hurts both FN and FP.

### 5. Weight decay hurts at this configuration

exp_032 (wd=1e-6) increased FP from 962k to 1,198k vs the identical
config without wd (exp_029). L2 regularization actively harms performance here.

---

## Comparison to Prior Best (exp_018)

| Metric | exp_018 (old best) | exp_031 (new best) | Delta |
|--------|-------------------|-------------------|-------|
| Loss function | AsymMSE (alpha=2.9375) | SmoothHuber (delta=0.1, alpha=1.5) | — |
| composite_score | 695,629 | 777,320 | +81,691 (+12%) |
| best_test_mse | 0.002817 | 0.002314 | -0.000503 (-18%) |
| focus_fp | 1,431,727 | 853,856 | -577,871 (-40%) |
| Pearson | 0.9800 | 0.9831 | +0.0031 |
| Revised FP cap? | FAIL (1.43M > 968k) | PASS (854k < 968k) | — |

exp_018 had a lower composite (695,629) but **fails the revised FP cap**.
exp_031 is the first experiment to pass both revised constraints while maintaining
strong composite performance. The MSE and Pearson correlation are also substantially
better.

---

## Batch 3 (exp_036 to exp_038) — Fine-tuning around exp_031

These experiments tested perturbations of exp_031's winning config.

| ID | Loss Function | alpha | delta | lr | MSE | focus_fp | focus_fn | composite | Pearson | Best Epoch | MSE ok? | FP ok? | Status |
|----|--------------|-------|-------|----|-----|----------|----------|-----------|---------|------------|---------|--------|--------|
| exp_036 | SmoothHuber | 1.5 | 0.1 | **2e-4** | 0.002527 | 1,044,450 | 793,766 | — | 0.9816 | 36 | PASS | FAIL | constraint_fail |
| exp_037 | SmoothHuber | **1.375** | 0.1 | 1e-4 | 0.002773 | 1,049,643 | 932,162 | — | 0.9797 | 48 | PASS | FAIL | constraint_fail |
| exp_038 | SmoothHuber | **1.625** | 0.1 | 1e-4 | 0.002529 | 1,014,598 | 815,806 | — | 0.9816 | 61 | PASS | FAIL | constraint_fail |

**All 3 fail the revised FP cap.** This reveals that exp_031's config sits on a
narrow sweet spot:

### 6. exp_031's config is uniquely optimal — neighbors all fail FP

| Variation | focus_fp | vs exp_031 (854k) | FP cap? |
|-----------|----------|-------------------|---------|
| exp_031: α=1.5, lr=1e-4 | 853,856 | baseline | PASS |
| exp_036: α=1.5, lr=2e-4 | 1,044,450 | +191k | FAIL |
| exp_037: α=1.375, lr=1e-4 | 1,049,643 | +196k | FAIL |
| exp_038: α=1.625, lr=1e-4 | 1,014,598 | +161k | FAIL |

Every perturbation — higher lr, lower alpha, higher alpha — pushes FP 160-196k
above exp_031. The model's FP behavior is extremely sensitive to these parameters.
This could indicate exp_031 found a narrow basin, or that training variance plays
a significant role.

---

## Updated Leaderboard — All Experiments Passing Both Constraints

| Rank | ID | Loss | alpha | delta | lr | composite | MSE | focus_fp | Pearson |
|------|----|------|-------|-------|----|-----------|-----|----------|---------|
| 1 | **exp_031** | SmoothHuber | 1.5 | 0.1 | 1e-4 | **777,320** | 0.002314 | 853,856 | 0.9831 |
| 2 | exp_035 | SmoothHuber | 1.25 | 0.1 | 1e-4 | 872,477 | 0.002578 | 923,180 | 0.9811 |
| 3 | exp_029 | AsymmetricMSE | 1.5 | — | 1e-4 | 901,827 | 0.002620 | 962,397 | 0.9809 |

Only 3 out of 10 experiments pass the revised constraints (30% pass rate).

---

## Recommended Next Steps

1. **Reproduce exp_031** — re-run the exact same config to confirm the result
   is stable and not a lucky seed. If it reproduces within ~5%, the config is
   solid. If not, we need to consider seed-averaging.

2. **Try lr=1.5e-4 with SmoothHuber α=1.5** — halfway between the passing
   lr=1e-4 (exp_031) and the failing lr=2e-4 (exp_036). If FP stays under cap,
   the lower composite from higher lr could be a win.

3. **SmoothHuber α=1.5 with --no_warmup** — exp_028 showed flat LR was
   equivalent to warmup+cosine for AsymMSE; SmoothHuber may behave differently,
   and removing warmup simplifies the schedule.

---

## Summary Statistics — Today's Session

| Metric | Value |
|--------|-------|
| Total experiments run | 10 (exp_029 to exp_038) |
| Pass both constraints | 3 (30%) |
| Best composite_score | 777,320 (exp_031) |
| Best MSE | 0.002314 (exp_031) |
| Best Pearson | 0.9831 (exp_031) |
| GPU hours consumed | ~23h (across all jobs + resumes) |

---

## Infrastructure Notes

- SLURM time limit was initially 2h30m, which caused exp_029 and exp_031 to be
  killed prematurely. Resume-from-checkpoint support was added and time limit
  increased to 5h for subsequent experiments.
- Training takes ~1.5-2.5h per experiment on A100 GPUs with early stopping
  (patience=40). Full 200 epochs would take ~4.5h.
