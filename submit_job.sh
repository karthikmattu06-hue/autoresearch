#!/bin/bash
# submit_job.sh <exp_name> <train_args>
# Copies latest train.py+losses.py to Drive, drops a job JSON, waits for score.json.

set -e
EXP_NAME="${1:?Usage: submit_job.sh <exp_name> <train_args>}"
TRAIN_ARGS="${2:-}"

DRIVE="$HOME/Library/CloudStorage/GoogleDrive-km5503@g.rit.edu/My Drive/tc_uc_autoresearch"
CODE_SRC="$HOME/tc_uc/autoresearch"

# Push latest code to Drive
cp "$CODE_SRC/train.py"  "$DRIVE/autoresearch/train.py"
cp "$CODE_SRC/losses.py" "$DRIVE/autoresearch/losses.py"

# Drop job JSON
JOB_FILE="$DRIVE/jobs/pending/${EXP_NAME}.json"
cat > "$JOB_FILE" <<EOF
{
  "exp_name": "$EXP_NAME",
  "train_args": "$TRAIN_ARGS",
  "submitted_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF
echo "[submit] $EXP_NAME queued at $(date -u +%H:%M:%SZ)"

# Poll Drive for score.json (max 3h)
SCORE_FILE="$DRIVE/autoresearch/output/${EXP_NAME}/score.json"
DEADLINE=$(( $(date +%s) + 10800 ))
while [ ! -f "$SCORE_FILE" ]; do
    if [ "$(date +%s)" -gt "$DEADLINE" ]; then
        echo "[timeout] 3h elapsed — no score.json"
        exit 1
    fi
    sleep 30
done
sleep 5
echo "[done] score.json ready"
cat "$SCORE_FILE"
