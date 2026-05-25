# Claude Code Instructions — TFT Autoresearch (air)

## What this project is
Minimize `composite_score` (sum of false negatives over τ ∈ [0.10, 0.60]) for a TFT
binding-line classifier on the RTS-96 dataset. Full rules in `program.md`.

## Current state (as of 2026-05-25)

### Constraints (UPDATED — tighter than original)
- `best_test_mse` ≤ **0.003378** (1.5× baseline 0.002252)
- `focus_fp`      ≤ **967,690**  (1.2× baseline 806,408) ← was 2.0×, tightened by Farhan

### Best result so far
**exp_018**: composite_score = 695,629 | MSE = 0.002817 | focus_fp = 1,431,727
- Config: `AsymmetricMSE(under_penalty=2.9375)` + dropout=0.0 + lr=2e-4
- **Fails the revised FP cap** (1,431,727 > 967,690) — no longer the accepted answer

### Why the new FP cap invalidates everything above α≈1.0
Every prior experiment from exp_002 onward had focus_fp > 967,690.
The entire α ∈ (1.0, 2.0) range is unexplored territory.

---

## Pending experiments — run these first

Four commits are already on `main`. Each is a ready-to-run config.
**Pull first, then submit them in parallel:**

```bash
git pull
```

Submit in order, waiting for each job to reach R (running) state before moving to the next:

```bash
# 1. exp_029 — AsymMSE α=1.5, lr=1e-4 (bisection midpoint)
git checkout 4ef957f
sbatch run_experiment.sh exp_029
# squeue -u $USER  →  wait until STATUS = R, then continue

# 2. exp_030 — AsymMSE α=1.75, lr=1e-4 (upper bound probe)
git checkout 9e1aab8
sbatch run_experiment.sh exp_030
# wait until R

# 3. exp_031 — SmoothHuber δ=0.1 α=1.5, lr=1e-4 (Huber retry)
git checkout 3b7ab7b
sbatch run_experiment.sh exp_031
# wait until R

# 4. exp_032 — AsymMSE α=1.5, lr=1e-4, wd=1e-6 (L2 regularization)
git checkout b3e612f   # HEAD
sbatch run_experiment.sh exp_032
```

All four will then be running simultaneously on SLURM (~95 min each).

---

## Polling for results

Poll all four at once:

```bash
for exp in exp_029 exp_030 exp_031 exp_032; do
  SCORE="output/$exp/score.json"
  if [ -f "$SCORE" ]; then
    echo "=== $exp DONE ==="
    cat "$SCORE"
  else
    echo "--- $exp still running ---"
  fi
done
```

Or watch squeue:
```bash
watch -n 30 "squeue -u $USER"
```

---

## Recording results

For each completed experiment, append a row to `results.tsv`:

```
exp_NNN  <commit_hash>  <timestamp>  <status>  <composite_score>  <best_test_mse>  <focus_fn>  <focus_fp>  <pearson>  <best_epoch>  <wall_minutes>  <notes>
```

- `status`: `ok+improved` / `ok` / `constraint_fail` / `crashed` / `time_limit`
- `composite_score = focus_fn` (they are the same metric)
- Mark `constraint_fail` and set composite=1e9 if either cap is breached:
  - MSE > 0.003378, OR
  - focus_fp > 967,690  ← new stricter cap

---

## Decision logic after batch results

### For exp_029 (α=1.5) and exp_030 (α=1.75):
These probe whether AsymMSE can stay under the new FP cap.

| exp_029 passes FP? | exp_030 passes FP? | Next step |
|---|---|---|
| ✓ | ✓ | Both in range — bisect upward: try α=1.875, compare composites |
| ✓ | ✗ | α=1.5 is viable, 1.75 is not — bisect [1.5, 1.75]: try α=1.625 |
| ✗ | ✗ | Both too high — bisect downward: try α=1.25 |

Take the passing experiment with the **lowest composite_score** as the new best.

### For exp_031 (SmoothHuber):
- If MSE cap passes: compare composite to best AsymMSE — keep if better
- If MSE cap fails: record constraint_fail; try δ=0.2 with same α=1.5 in a follow-up

### For exp_032 (L2 reg wd=1e-6):
- Compare to exp_029 (same config, no wd) — keep if composite improves AND both constraints pass
- If SLURM time-limit: wd is preventing convergence; try wd=1e-7 next

---

## Git workflow per experiment

If an experiment **improves** on best-so-far AND passes both constraints:
```bash
git checkout main
git cherry-pick <exp_NNN_commit_hash>
# already committed — just keep it
```

If an experiment **fails or is worse**:
```bash
# Do NOT revert the commit — it's already been superseded by later commits.
# Just record it in results.tsv with the correct status and move on.
```

Commit results.tsv after each batch:
```bash
git add results.tsv
git commit -m "log exp_029-032 results — <one line summary>"
git push
```

---

## What you may NOT change
- `TemporalFusionTransformer` class (architecture)
- `score.py`, `eval.py`
- SLURM resource directives in `run_experiment.sh`
- Anything outside the `# AGENT-EDIT START/END` fences in `train.py` and `run_experiment.sh`

## Key file locations (on this cluster)
- Working dir: `/shared/rc/career/km5503/tc_uc/autoresearch`
- Data: `../data/rts96`
- Outputs: `output/exp_NNN/score.json` (completion signal), `best_model.pt`, `loss_curve.png`
- Full experiment log: `results.tsv`
- Search rules and lever menu: `program.md`
