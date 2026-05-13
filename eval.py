"""
eval.py — Load trained TFT, run inference on RTS-96 test set, save predictions.

Output: predictions.npy + targets.npy in OUTPUT_DIR (shape (1826, 24, 120) float32).
Skips inference if predictions.npy already exists (per Karthik's caching convention).
"""
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import sys

# Reuse the model definition from train.py — same directory
from train import TemporalFusionTransformer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_path', required=True,
                    help='Path to best_model.pt (saved by train.py)')
    ap.add_argument('--data_dir', default='../data/rts96',
                    help='Directory containing RTS96_T_s_*.npy files')
    ap.add_argument('--output_dir', required=True,
                    help='Where to save predictions.npy and targets.npy')
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--force', action='store_true',
                    help='Re-run inference even if predictions.npy exists')
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_path = output_dir / 'predictions.npy'
    targ_path = output_dir / 'targets.npy'

    # ── Skip if cached ─────────────────────────────────────────
    if pred_path.exists() and targ_path.exists() and not args.force:
        print(f"Predictions already exist at {pred_path}. Skipping inference.")
        print("Use --force to re-run.")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── Load checkpoint ────────────────────────────────────────
    print(f"Loading checkpoint: {args.model_path}")
    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    if not isinstance(ckpt, dict) or 'model_state_dict' not in ckpt:
        sys.exit("ERROR: checkpoint format not recognized (expected dict with 'model_state_dict')")

    train_args = ckpt.get('args', {})
    print(f"  Best epoch saved: {ckpt.get('epoch', '?')}")
    print(f"  Train args: hidden={train_args.get('hidden_size', 256)}, "
          f"heads={train_args.get('num_heads', 8)}, "
          f"dropout={train_args.get('dropout', 0.05)}")

    # ── Load test data ─────────────────────────────────────────
    data_dir = Path(args.data_dir)
    encoder_test = np.load(data_dir / 'RTS96_T_s_X_test.npy',
                           allow_pickle=True).astype(np.float32)
    decoder_test = np.load(data_dir / 'RTS96_T_s_Y_test_unflattened.npy',
                           allow_pickle=True).astype(np.float32)
    print(f"  encoder_test: {encoder_test.shape}")
    print(f"  decoder_test: {decoder_test.shape}")

    num_features = encoder_test.shape[2]

    # ── Build model + load weights ─────────────────────────────
    model = TemporalFusionTransformer(
        num_features=num_features,
        hidden_size=train_args.get('hidden_size', 256),
        num_heads=train_args.get('num_heads', 8),
        dropout=train_args.get('dropout', 0.05),
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params:,}")

    # ── Inference ──────────────────────────────────────────────
    test_dataset = TensorDataset(
        torch.from_numpy(encoder_test).float(),
        torch.from_numpy(decoder_test).float(),
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=0)

    all_preds = []
    with torch.no_grad():
        for enc, dec in test_loader:
            enc = enc.to(device)
            dec = dec.to(device)
            pred = model(enc, dec)        # already sigmoid'd in forward()
            all_preds.append(pred.cpu().numpy())

    predictions = np.concatenate(all_preds, axis=0).astype(np.float32)
    targets     = decoder_test.astype(np.float32)

    # ── Save ───────────────────────────────────────────────────
    np.save(pred_path, predictions)
    np.save(targ_path, targets)
    print(f"\nSaved:")
    print(f"  {pred_path}  shape={predictions.shape}  range=[{predictions.min():.4f}, {predictions.max():.4f}]")
    print(f"  {targ_path}  shape={targets.shape}      range=[{targets.min():.4f}, {targets.max():.4f}]")

    # Sanity
    mse = ((predictions - targets) ** 2).mean()
    print(f"\nQuick MSE check: {mse:.6f}")


if __name__ == '__main__':
    main()
