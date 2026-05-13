# Autoresearch: TFT Asymmetric-Loss Exploration — RTS-96

## Goal
Reduce under-prediction (false negatives) in the TFT model for binding-line
classification, per Farhan's directive. The composite metric to MINIMIZE is
the sum of false negatives across decision thresholds τ ∈ [0.10, 0.60].

## Baseline to beat (run3_repro_baseline)
- composite_score (focus_fn over τ∈[0.10,0.60])   : 1,026,392
- best_test_mse                                    : 0.002252
- focus_fp (FP over τ∈[0.10,0.60])                 : 806,408

## Hard constraints (composite_score = 1e9 if violated)
- best_test_mse  ≤ 1.5 × 0.002252  = 0.003378
- focus_fp       ≤ 2.0 × 806,408   = 1,612,816

## What you can edit
1. `train.py` — ONLY between `# AGENT-EDIT START/END` fences:
   - `loss`        : criterion = ...
   - `optimizer`   : torch.optim.Adam/AdamW(..., weight_decay=...)
   - `module_norm` : torch.nn.utils.weight_norm(...) wrapping
2. `run_experiment.sh` — ONLY between `# AGENT-EDIT START/END train_args` fence:
   - --lr, --batch_size, --epochs, --patience, --dropout, --grad_clip
   - --no_warmup, --no_cosine flags

## What you may NOT touch
- TemporalFusionTransformer class (architecture)
- Data loader, training loop structure, scheduler logic
- score.py, eval.py
- SLURM resource directives in run_experiment.sh

## Lever menu (with bounds)
L1: Loss function (from losses.py)
    - nn.MSELoss()                                          (baseline)
    - AsymmetricMSE(under_penalty=α)               α ∈ [1.5, 5.0]
    - SmoothAsymmetricHuberLoss(delta=δ, under_penalty=α)
                                                   δ ∈ [0.05, 0.20]
                                                   α ∈ [1.5, 5.0]
L2: Weight decay (in optimizer)
    - 0                                                     (baseline)
    - Adam   weight_decay ∈ [1e-6, 1e-3]
    - AdamW  weight_decay ∈ [1e-6, 1e-3]
L3: Weight norm (module_norm fence)
    - None                                                  (baseline)
    - Wrap output_layer Linears
    - Wrap pos_wise_ff Linears
    - Both
L4: Training hyperparameters (run_experiment.sh)
    - --lr            [1e-5, 1e-3]
    - --batch_size    [16, 256]
    - --epochs        [50, 600]
    - --patience      any (incl. 9999 = disabled)
    - --dropout       [0.0, 0.3]
    - --grad_clip     [0.1, 5.0]
    - --no_warmup / --no_cosine: free

## Exploration discipline
- Exp 1–3 : isolate L1 only. Try AsymmetricMSE at α ∈ {2.0, 3.0, 5.0}.
- Exp 4–6 : isolate L1 only. Try SmoothAsymmetricHuberLoss at
            (δ=0.1, α=3.0), (δ=0.1, α=5.0), (δ=0.05, α=3.0).
- Exp 7–10: take the best L1 from above; add L2 weight_decay variations.
- Exp 11+ : free exploration, may combine up to 2 levers per experiment.
- NEVER combine all 3 levers without explicit human approval (post in chat).
- Each experiment changes ONE thing from prior best, unless the prior 3
  experiments all failed (in which case revert to baseline and try a
  different lever).

## How to run one experiment
1. Edit train.py and/or run_experiment.sh within the fences.
2. git add -A && git commit -m "exp_NNN: <one-line description>"
3. sbatch run_experiment.sh exp_NNN
4. Poll squeue every 60s until done.
5. Read output/exp_NNN/score.json
6. Append a row to results.tsv (see schema below).
7. If composite_score < best_so_far: keep commit.
   Else: git reset --hard HEAD~1 (revert files, keep results.tsv row).

## results.tsv schema (tab-separated)
exp_id  commit_hash  timestamp  status  composite_score  best_test_mse
focus_fn  focus_fp  pearson  best_epoch  wall_minutes  notes

(Detailed configuration lives in the git commit + diff. Don't duplicate it
in results.tsv — read it from git when needed.)

## Stop conditions
- After 20 experiments per session (context budget).
- If composite_score has not improved over baseline after 5 experiments
  AND we're past exp 6, post in chat for guidance.
- If 3 consecutive experiments crash (not just constraint-fail), STOP and
  post in chat.
- If wall-clock per experiment exceeds 2.5 hours, the SLURM time limit
  will kill it and the run is marked failed. Don't go beyond patience=200.

## NEVER STOP unless one of the above triggers.
