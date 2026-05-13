#!/bin/bash
#SBATCH --job-name=tft_exp
#SBATCH --partition=tier3
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --exclude=spr-a-02

# ── Environment ──────────────────────────────────────────────
export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH
spack env activate gearnerf-x86_64-25022801
export PYTHONUNBUFFERED=1

cd /shared/rc/career/km5503/tc_uc/autoresearch
mkdir -p logs output

RUN_NAME=${1:?Usage: sbatch run_experiment.sh <run_name>}

echo "=== Experiment: $RUN_NAME ==="
echo "Job ID : $SLURM_JOB_ID  |  Node: $(hostname)  |  Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
echo ""

# ─── [1/3] Train ─────────────────────────────────────────────
echo "─── [1/3] Train ───"
# ============ AGENT-EDIT START: train_args ============
# Default: baseline run3_repro args. Bounds in program.md.
# Agent may add/remove --no_warmup, --no_cosine flags.
python -u train.py \
    --run_name    $RUN_NAME \
    --data_dir    ../data/rts96 \
    --output_base output \
    --lr          1e-4 \
    --batch_size  32 \
    --epochs      200 \
    --patience    40 \
    --dropout     0.05 \
    --grad_clip   0.5
# ============ AGENT-EDIT END: train_args ============
echo ""

# ─── [2/3] Inference ─────────────────────────────────────────
echo "─── [2/3] Eval (inference) ───"
python -u eval.py \
    --model_path  output/$RUN_NAME/best_model.pt \
    --output_dir  output/$RUN_NAME \
    --data_dir    ../data/rts96
echo ""

# ─── [3/3] Score ─────────────────────────────────────────────
echo "─── [3/3] Score ───"
python -u score.py \
    --predictions output/$RUN_NAME/predictions.npy \
    --targets     output/$RUN_NAME/targets.npy \
    --output      output/$RUN_NAME/score.json
echo ""
echo "Finished: $(date)"
