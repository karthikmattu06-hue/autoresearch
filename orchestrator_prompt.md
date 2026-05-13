# Orchestrator Prompt for the Autoresearch Loop

Paste the section below into a **fresh Claude Code session** running on your
M4. Before pasting, complete the one-time setup at the bottom of this file.

────────────────────────────────────────────────────────────────────────────
ONE-TIME SETUP (do this once per workday, before launching CC session)
────────────────────────────────────────────────────────────────────────────

1. Add to your `~/.ssh/config` (one-time, persists across sessions):

    Host rit-rc
        HostName sporcsubmit.rc.rit.edu
        User km5503
        ControlMaster auto
        ControlPath ~/.ssh/cm-%r@%h:%p
        ControlPersist 12h

2. Authenticate once via Duo (this opens the persistent SSH socket):

    ssh rit-rc "echo connected; hostname"
    # → approve Duo push on phone → "connected ... sporcsubmit"
    # All subsequent ssh/rsync/scp to 'rit-rc' reuse this socket for 12 hours.

3. Confirm local repo is in place at:

    ~/tc_uc/autoresearch/
        ├── train.py           (with AGENT-EDIT fences)
        ├── losses.py
        ├── eval.py
        ├── score.py
        ├── run_experiment.sh  (v2, chained train+eval+score)
        ├── program.md
        └── output/run3_repro_baseline/  (baseline already populated)

4. Confirm cluster mirror exists at:

    /shared/rc/career/km5503/tc_uc/autoresearch/   (rit-rc)

5. git init the local repo if not already:

    cd ~/tc_uc/autoresearch && git init && git add -A
    git commit -m "baseline: run3_repro_baseline locked"

────────────────────────────────────────────────────────────────────────────
CC SESSION PROMPT — paste everything below into a fresh CC session
────────────────────────────────────────────────────────────────────────────

You are running an autoresearch loop for a TFT power-grid model. Your job is
to propose, run, score, and ratchet experiments to minimize a composite_score
metric defined in program.md.

## Role / mode
- You are NOT a human assistant in this session. You are an autonomous agent
  running a research loop. Do not summarize or chat. Take actions.
- Read program.md FIRST. It is authoritative. If anything below contradicts
  it, program.md wins.
- Work in ~/tc_uc/autoresearch on this M4.
- The cluster is reachable via `ssh rit-rc` (ControlMaster pre-authenticated).
- Cluster path mirror: /shared/rc/career/km5503/tc_uc/autoresearch

## The loop (repeat until a stop condition in program.md fires)

For each iteration:

1. READ STATE
   - `cat program.md` (refresh on rules)
   - `cat results.tsv` if exists, else create with header:
       exp_id\tcommit_hash\ttimestamp\tstatus\tcomposite_score\tbest_test_mse\tfocus_fn\tfocus_fp\tpearson\tbest_epoch\twall_minutes\tnotes
   - `git log --oneline | head -20`
   - Determine `best_so_far` = min composite_score from results.tsv where
     status=='ok'. If results.tsv is empty (or has only header), use the
     baseline number from program.md (1,026,392).
   - Determine next exp number: count existing exp_* rows in results.tsv,
     add 1, zero-pad to 3 digits. E.g., exp_001, exp_002, ...

2. PROPOSE
   - Decide what ONE change to make based on program.md's exploration
     discipline and results so far. Print your reasoning in 2-3 lines.
   - If you're at exp 11+ and want to combine 2 levers, that's allowed.
   - NEVER combine all 3 levers. If you think you need to, STOP and post
     in chat for human approval.

3. EDIT
   - Edit train.py and/or run_experiment.sh ONLY within AGENT-EDIT fences.
   - Run `git diff` to confirm changes are inside the fences. If diff shows
     edits outside fences, abort and `git checkout -- <file>`.

4. COMMIT
   - `git add -A && git commit -m "exp_NNN: <one-line description>"`

5. SYNC UP
   - `rsync -av --exclude='output/' --exclude='logs/' --exclude='.git/' \
       ~/tc_uc/autoresearch/ rit-rc:/shared/rc/career/km5503/tc_uc/autoresearch/`

6. SUBMIT
   - `ssh rit-rc "cd /shared/rc/career/km5503/tc_uc/autoresearch && sbatch run_experiment.sh exp_NNN"`
   - Capture the job ID from sbatch output (e.g., "Submitted batch job 12345").

7. POLL
   - Sleep 60s, then `ssh rit-rc "squeue -j <jobid> -h"`.
   - If empty output → job done. If still queued/running → sleep 60s and
     check again.
   - Hard timeout: if a job is still in queue (state PD) after 30 minutes,
     STOP and post in chat — the queue is too contended to continue.
   - Hard timeout: if a job has been running for > 2h45m, something's wrong;
     check the log via ssh and stop.

8. SYNC DOWN
   - `rsync -av rit-rc:/shared/rc/career/km5503/tc_uc/autoresearch/output/exp_NNN/ \
       ~/tc_uc/autoresearch/output/exp_NNN/`
   - Also pull the SLURM log: 
       `rsync -av rit-rc:/shared/rc/career/km5503/tc_uc/autoresearch/logs/tft_exp_<jobid>.out \
        ~/tc_uc/autoresearch/logs/`

9. INTERPRET
   - If output/exp_NNN/score.json exists → parse it.
     - status = "ok" if constraint_ok else "constraint_failed"
   - If score.json missing → status = "crashed" (training or eval failed).
     Inspect the SLURM log briefly to understand why, but DO NOT spend more
     than 60 seconds debugging — just log the failure and move on.

10. RECORD
    - Append one row to results.tsv:
        exp_NNN  <commit_hash>  <iso_timestamp>  <status>  <composite_score>
        <best_test_mse>  <focus_fn>  <focus_fp>  <pearson>  <best_epoch>
        <wall_minutes>  <one_line_note>
    - Fields are tab-separated. Use NaN or "" for missing values on crashes.

11. RATCHET
    - If status=='ok' AND composite_score < best_so_far:
        Keep commit. Print "★ new best: <composite_score>"
    - Else:
        `git reset --hard HEAD~1`  (revert train.py + run_experiment.sh)
        BUT keep results.tsv changes:
        `git add results.tsv && git commit -m "exp_NNN: log only (reverted)"`
        Print "✗ no improvement, reverted"

12. LOOP
    - Check stop conditions in program.md. If any fire → STOP and report.
    - Else → goto 1.

## Stop & report
When a stop condition fires (20 experiments hit / no improvement after 5 /
3 crashes in a row / explicit need-human signal), output:

- Summary table of all experiments run this session (read from results.tsv)
- Best composite_score achieved this session vs baseline 1,026,392
- The current best git commit hash (so the human can `git checkout` it)
- Your reasoning about what to try next session
- Then STOP. Do not start more experiments.

## Hard rules (cannot be relaxed)
- Never edit score.py, eval.py, or the TemporalFusionTransformer class.
- Never edit outside AGENT-EDIT fences.
- Never run more than ONE experiment in parallel.
- Never bypass the git commit/revert ratchet.
- Never make more than 5 minutes of edits without committing.

Begin by reading program.md, then results.tsv (or noting it doesn't exist),
then proposing exp_001.
