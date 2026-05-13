"""
score.py — Compute metrics on (predictions, targets) → score.json.

Identical metric definition for ANY model (Farhan's, ours, autoresearch agent's).
Pure NumPy — no sklearn (hangs on large arrays for Polish-scale).

Outputs:
  - Continuous: MSE, MAE, Pearson
  - Per-threshold (τ ∈ {0.05, 0.10, ..., 0.95}): accuracy, F1, TP, FP, FN, TN, errors=FP+FN
  - Aggregate sums: total_FN, total_FP, total_errors (across all 19 thresholds)
  - Farhan's focus range (τ ∈ {0.10, ..., 0.60}, 11 thresholds): ΣFN_focus, ΣFP_focus
  - composite_score:
        ΣFN_focus     if MSE ≤ 1.5 × baseline_mse
        1e9           otherwise (hard fail — autoresearch ratchet rejects)
"""
import argparse
import json
import numpy as np
from pathlib import Path


def confusion(pred_bin: np.ndarray, target_bin: np.ndarray):
    """Direct NumPy TP/FP/FN/TN — sklearn-free for Polish-scale safety."""
    tp = int(np.logical_and(pred_bin,  target_bin).sum())
    fp = int(np.logical_and(pred_bin, ~target_bin).sum())
    fn = int(np.logical_and(~pred_bin,  target_bin).sum())
    tn = int(np.logical_and(~pred_bin, ~target_bin).sum())
    return tp, fp, fn, tn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--predictions', required=True,
                    help='Path to predictions.npy (sigmoid output, [0,1])')
    ap.add_argument('--targets', required=True,
                    help='Path to targets.npy (or actual.npy)')
    ap.add_argument('--output', required=True,
                    help='Path to write score.json')
    ap.add_argument('--baseline_mse', type=float, default=0.002252,
                    help='Baseline MSE for composite_score constraint')
    ap.add_argument('--mse_slack', type=float, default=1.5,
                    help='Hard fail if MSE > slack × baseline_mse')
    ap.add_argument('--baseline_focus_fp', type=int, default=806408,
                    help='Baseline focus_fp (Σ FP over τ∈[0.10,0.60]) — our run3_repro')
    ap.add_argument('--fp_slack', type=float, default=2.0,
                    help='Hard fail if focus_fp > slack × baseline_focus_fp')
    args = ap.parse_args()

    pred = np.load(args.predictions, allow_pickle=True).astype(np.float32)
    targ = np.load(args.targets,     allow_pickle=True).astype(np.float32)
    assert pred.shape == targ.shape, f"shape mismatch: {pred.shape} vs {targ.shape}"

    pred_flat = pred.flatten()
    targ_flat = targ.flatten()

    # ── Continuous metrics ─────────────────────────────────────
    mse = float(((pred_flat - targ_flat) ** 2).mean())
    mae = float(np.abs(pred_flat - targ_flat).mean())
    pearson = float(np.corrcoef(pred_flat, targ_flat)[0, 1])

    # ── Threshold sweep ────────────────────────────────────────
    thresholds = [round(0.05 * i, 2) for i in range(1, 20)]   # 0.05..0.95
    focus_range = [round(0.05 * i, 2) for i in range(2, 13)]  # 0.10..0.60

    per_threshold = {}
    for t in thresholds:
        pb = pred >= t
        tb = targ >= t
        tp, fp, fn, tn = confusion(pb, tb)
        total = tp + fp + fn + tn
        acc = (tp + tn) / total if total else 0.0
        f1  = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        per_threshold[t] = {
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'accuracy': acc, 'f1': f1, 'errors': fp + fn,
        }

    total_fn      = sum(v['fn']     for v in per_threshold.values())
    total_fp      = sum(v['fp']     for v in per_threshold.values())
    total_errors  = sum(v['errors'] for v in per_threshold.values())
    focus_fn      = sum(per_threshold[t]['fn']     for t in focus_range)
    focus_fp      = sum(per_threshold[t]['fp']     for t in focus_range)
    focus_errors  = sum(per_threshold[t]['errors'] for t in focus_range)

    # ── Composite ──────────────────────────────────────────────
    mse_cap = args.mse_slack * args.baseline_mse
    fp_cap  = args.fp_slack  * args.baseline_focus_fp
    mse_ok  = mse        <= mse_cap
    fp_ok   = focus_fp   <= fp_cap
    constraint_ok = mse_ok and fp_ok
    composite_score = focus_fn if constraint_ok else 1e9

    out = {
        'mse': mse,
        'mae': mae,
        'pearson': pearson,
        'thresholds': thresholds,
        'focus_range': focus_range,
        'per_threshold': per_threshold,
        'total_fn': total_fn,
        'total_fp': total_fp,
        'total_errors': total_errors,
        'focus_fn': focus_fn,
        'focus_fp': focus_fp,
        'focus_errors': focus_errors,
        'baseline_mse': args.baseline_mse,
        'mse_cap': mse_cap,
        'baseline_focus_fp': args.baseline_focus_fp,
        'fp_cap': fp_cap,
        'mse_ok': mse_ok,
        'fp_ok': fp_ok,
        'constraint_ok': constraint_ok,
        'composite_score': composite_score,
        'predictions_path': str(Path(args.predictions).resolve()),
        'targets_path':     str(Path(args.targets).resolve()),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(out, f, indent=2)

    # ── Print summary ──────────────────────────────────────────
    print(f"MSE: {mse:.6f}  (cap: {mse_cap:.6f}, ok: {mse_ok})")
    print(f"MAE: {mae:.6f}  Pearson: {pearson:.4f}")
    print(f"ΣFN (all τ):       {total_fn:,}")
    print(f"ΣFN (τ ∈ {focus_range[0]:.2f}-{focus_range[-1]:.2f}): {focus_fn:,}  ← composite_score")
    print(f"ΣFP (focus):       {focus_fp:,}  (cap: {fp_cap:,.0f}, ok: {fp_ok})")
    print(f"Constraints OK:    {constraint_ok}")
    if not constraint_ok:
        print(f"  → composite_score forced to 1e9 (hard fail)")
    print(f"Saved: {args.output}")


if __name__ == '__main__':
    main()
