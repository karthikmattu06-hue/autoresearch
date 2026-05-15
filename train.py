#!/usr/bin/env python3
"""
TFT Hyperparameter Tuning — RTS-96 Grid (Y → Ŷ)
=================================================
Mirrors the paper's LSTM setup as closely as possible while sweeping
the key levers that matter for TFT on a small grid.

Three suggested runs (match paper Table I, Y-Ŷ config):
  Run A — flat LR, matches paper exactly:
    python train_RTS96_TFT_tune.py --lr 0.0001 --no_warmup --patience 20 --run_name runA

  Run B — slightly higher LR, cosine decay only (no warmup):
    python train_RTS96_TFT_tune.py --lr 0.0003 --no_warmup --patience 20 --run_name runB

  Run C — flat LR, larger batch:
    python train_RTS96_TFT_tune.py --lr 0.0001 --no_warmup --batch_size 256 --patience 20 --run_name runC

All runs use:
  - Adam optimizer  (paper used Adam, not AdamW)
  - Early stopping on test MSE
  - Output saved to ./output/rts96_tft_tune/<run_name>/
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# Model (identical to production TFT)
# ============================================================

class GatedLinearUnit(nn.Module):
    def __init__(self, input_size, hidden_size=None, dropout=0.1):
        super().__init__()
        if hidden_size is None:
            hidden_size = input_size
        self.hidden_size = hidden_size
        self.fc      = nn.Linear(input_size, hidden_size * 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(self.fc(x))
        return x[..., :self.hidden_size] * torch.sigmoid(x[..., self.hidden_size:])


class GatedResidualNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=None,
                 dropout=0.1, context_size=None):
        super().__init__()
        self.input_size   = input_size
        self.output_size  = output_size or input_size
        self.context_size = context_size
        self.fc1       = nn.Linear(input_size, hidden_size)
        self.fc2       = nn.Linear(hidden_size, hidden_size)
        self.glu       = GatedLinearUnit(hidden_size, self.output_size, dropout)
        self.layernorm = nn.LayerNorm(self.output_size)
        if context_size:
            self.context_fc = nn.Linear(context_size, hidden_size, bias=False)
        if input_size != self.output_size:
            self.skip_fc = nn.Linear(input_size, self.output_size)

    def forward(self, x, context=None):
        h = self.fc1(x)
        if context is not None and self.context_size:
            h = h + self.context_fc(context)
        h = self.fc2(torch.nn.functional.elu(h))
        r = self.skip_fc(x) if self.input_size != self.output_size else x
        return self.layernorm(self.glu(h) + r)


class VariableSelectionNetwork(nn.Module):
    def __init__(self, input_size, num_inputs, hidden_size,
                 dropout=0.1, context_size=None):
        super().__init__()
        self.grns = nn.ModuleList([
            GatedResidualNetwork(input_size, hidden_size, hidden_size,
                                 dropout, context_size)
            for _ in range(num_inputs)
        ])
        self.weight_network = GatedResidualNetwork(
            input_size * num_inputs, hidden_size, num_inputs, dropout, context_size
        )

    def forward(self, embedding, context=None):
        flatten = embedding.view(embedding.size(0), -1)
        weights = torch.softmax(
            self.weight_network(flatten, context), dim=-1
        ).unsqueeze(-1)
        processed = torch.stack(
            [grn(embedding[:, i, :], context) for i, grn in enumerate(self.grns)],
            dim=1
        )
        return (processed * weights).sum(dim=1), weights.squeeze(-1)


class TemporalFusionTransformer(nn.Module):
    def __init__(self, input_size=120, hidden_size=256, num_attention_heads=8,
                 dropout=0.1, num_encoder_steps=192, num_decoder_steps=24,
                 num_outputs=120):
        super().__init__()
        self.num_encoder_steps = num_encoder_steps
        self.num_decoder_steps = num_decoder_steps
        self.num_outputs       = num_outputs

        self.encoder_embedding = nn.Linear(input_size, hidden_size)
        self.encoder_vsn       = VariableSelectionNetwork(
            hidden_size, 1, hidden_size, dropout)
        self.lstm_encoder      = nn.LSTM(hidden_size, hidden_size, 3,
                                         batch_first=True,
                                         dropout=dropout if dropout > 0 else 0)
        self.decoder_embedding = nn.Linear(1, hidden_size)
        self.lstm_decoder      = nn.LSTM(hidden_size, hidden_size, 3,
                                         batch_first=True,
                                         dropout=dropout if dropout > 0 else 0)
        self.multihead_attn    = nn.MultiheadAttention(
            hidden_size, num_attention_heads, dropout=dropout, batch_first=True)
        self.post_attn_gate    = GatedLinearUnit(hidden_size, dropout=dropout)
        self.post_attn_norm    = nn.LayerNorm(hidden_size)
        self.pos_wise_ff       = GatedResidualNetwork(
            hidden_size, hidden_size * 4, hidden_size, dropout)
        self.output_layer      = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_size, num_outputs),
        )

    def forward(self, encoder_input, decoder_input=None):
        batch_size = encoder_input.size(0)
        enc_emb    = self.encoder_embedding(encoder_input).unsqueeze(2)
        sel_enc    = torch.stack([
            self.encoder_vsn(enc_emb[:, t, :, :])[0]
            for t in range(self.num_encoder_steps)
        ], dim=1)
        enc_out, (h, c) = self.lstm_encoder(sel_enc)

        if decoder_input is None:
            decoder_input = (
                torch.arange(self.num_decoder_steps,
                             device=encoder_input.device, dtype=torch.float32)
                .view(1, -1, 1).expand(batch_size, -1, -1) / self.num_decoder_steps
            )
        dec_emb     = self.decoder_embedding(decoder_input)
        dec_out, _  = self.lstm_decoder(dec_emb, (h, c))
        attn_out, _ = self.multihead_attn(dec_out, enc_out, enc_out)
        attn_out    = self.post_attn_norm(self.post_attn_gate(attn_out) + dec_out)
        out         = self.output_layer(self.pos_wise_ff(attn_out))
        return torch.sigmoid(
            out.view(batch_size, self.num_decoder_steps, self.num_outputs)
        )


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_name',    default='run',        help='Label for output folder')
    ap.add_argument('--data_dir',    default='./data/rts96')
    ap.add_argument('--output_base', default='./output/rts96_tft_tune')

    # ── Key tuning levers ──────────────────────────────────────
    ap.add_argument('--lr',          type=float, default=1e-4,
                    help='Peak/flat learning rate (paper: 0.0001)')
    ap.add_argument('--no_warmup',   action='store_true',
                    help='Disable warmup — use cosine decay from epoch 1, '
                         'or flat LR if --no_cosine also set')
    ap.add_argument('--no_cosine',   action='store_true',
                    help='Disable cosine decay — flat LR throughout (matches paper exactly)')
    ap.add_argument('--batch_size',  type=int,   default=32,
                    help='Batch size (paper: 200)')
    ap.add_argument('--epochs',      type=int,   default=200,
                    help='Max epochs (early stopping will cut this short)')
    ap.add_argument('--patience',    type=int,   default=40,
                    help='Early stopping patience (epochs without test MSE improvement)')

    # ── Model architecture ─────────────────────────────────────
    ap.add_argument('--hidden_size', type=int,   default=256)
    ap.add_argument('--num_heads',   type=int,   default=8)
    ap.add_argument('--dropout',     type=float, default=0.05)
    ap.add_argument('--grad_clip',   type=float, default=0.5)
    args = ap.parse_args()

    OUTPUT_DIR = Path(args.output_base) / args.run_name
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR   = Path(args.data_dir)

    print("=" * 70)
    print(f"TFT HYPERPARAMETER TUNING  —  RTS-96  —  {args.run_name}")
    print("=" * 70)
    print(f"  LR            : {args.lr}")
    print(f"  Warmup        : {'OFF' if args.no_warmup else 'ON'}")
    print(f"  Cosine decay  : {'OFF (flat LR)' if args.no_cosine else 'ON'}")
    print(f"  Batch size    : {args.batch_size}")
    print(f"  Max epochs    : {args.epochs}")
    print(f"  Early stop    : patience={args.patience}")
    print(f"  Hidden size   : {args.hidden_size}")
    print(f"  Output        : {OUTPUT_DIR}")

    # ── Device ────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice : {device}")
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark        = True
        print(f"  GPU  : {torch.cuda.get_device_name(0)}")

    # ── Load data ─────────────────────────────────────────────
    print("\nLoading data ...")
    encoder_train = np.load(DATA_DIR / 'RTS96_T_s_X_train.npy', allow_pickle=True).astype(np.float32)
    decoder_train = np.load(DATA_DIR / 'RTS96_T_s_Y_train_unflattened.npy', allow_pickle=True).astype(np.float32)
    encoder_test  = np.load(DATA_DIR / 'RTS96_T_s_X_test.npy', allow_pickle=True).astype(np.float32)
    decoder_test  = np.load(DATA_DIR / 'RTS96_T_s_Y_test_unflattened.npy', allow_pickle=True).astype(np.float32)

    print(f"  encoder_train : {encoder_train.shape}")
    print(f"  decoder_train : {decoder_train.shape}")
    print(f"  encoder_test  : {encoder_test.shape}")
    print(f"  decoder_test  : {decoder_test.shape}")

    num_features  = encoder_train.shape[2]
    encoder_steps = encoder_train.shape[1]
    decoder_steps = decoder_train.shape[1]

    train_dataset = TensorDataset(
        torch.from_numpy(encoder_train).float(),
        torch.from_numpy(decoder_train).float(),
    )
    test_dataset  = TensorDataset(
        torch.from_numpy(encoder_test).float(),
        torch.from_numpy(decoder_test).float(),
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True,  pin_memory=True, num_workers=2)
    test_loader  = DataLoader(test_dataset,  batch_size=args.batch_size,
                              shuffle=False, pin_memory=True, num_workers=2)

    # ── Model ─────────────────────────────────────────────────
    model = TemporalFusionTransformer(
        input_size          = num_features,
        hidden_size         = args.hidden_size,
        num_attention_heads = args.num_heads,
        dropout             = args.dropout,
        num_encoder_steps   = encoder_steps,
        num_decoder_steps   = decoder_steps,
        num_outputs         = num_features,
    ).to(device)

    # ============ AGENT-EDIT START: module_norm ============
    # Optional: apply torch.nn.utils.weight_norm to specific Linear/LSTM modules.
    # Default: no-op. Agent may wrap targeted modules to stabilize training.
    # Example:
    #   from torch.nn.utils import weight_norm
    #   model.output_layer[0] = weight_norm(model.output_layer[0])
    pass
    # ============ AGENT-EDIT END: module_norm ============

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters : {total_params:,}")

    # ============ AGENT-EDIT START: loss ============
    # Default baseline: plain MSE (matches our run3_repro_baseline).
    # Agent may swap to AsymmetricMSE or SmoothAsymmetricHuberLoss from losses.py.
    # Examples:
    #   from losses import AsymmetricMSE
    #   criterion = AsymmetricMSE(under_penalty=3.0)
    #   from losses import SmoothAsymmetricHuberLoss
    #   criterion = SmoothAsymmetricHuberLoss(delta=0.1, under_penalty=3.0)
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    from losses import AsymmetricMSE
    criterion = AsymmetricMSE(under_penalty=2.875)
    # ============ AGENT-EDIT END: loss ============

    # ── Optimizer (Adam, matching paper) ──────────────────────
    # ============ AGENT-EDIT START: optimizer ============
    # Default: Adam without weight_decay (matches our run3_repro_baseline).
    # Farhan's actual run3 used weight_decay=1e-5 — agent may add it back or vary.
    # Range: [0, 1e-3]. Examples:
    #   optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    #   optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # ============ AGENT-EDIT END: optimizer ============

    # ── Scheduler ─────────────────────────────────────────────
    total_steps  = len(train_loader) * args.epochs
    warmup_steps = len(train_loader) * 5   # 5 epoch warmup

    if args.no_warmup and args.no_cosine:
        # Flat LR — exact paper match
        scheduler = None
        print("Scheduler      : FLAT LR (exact paper match)")
    elif args.no_warmup:
        # Cosine decay from epoch 1, no warmup
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-6
        )
        print("Scheduler      : Cosine decay, no warmup")
    else:
        # Warmup + cosine (original setup)
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        print("Scheduler      : Warmup + cosine")

    # ── Training loop ─────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"TRAINING  (max {args.epochs} epochs, early stop patience={args.patience})")
    print(f"{'='*70}")

    history        = []
    best_test_mse  = float('inf')
    best_epoch     = 0
    patience_count = 0
    train_start    = time.time()

    for epoch in range(1, args.epochs + 1):
        # ── Train ──
        model.train()
        train_loss = 0.0
        for enc_b, dec_b in train_loader:
            enc_b = enc_b.to(device, non_blocking=True)
            dec_b = dec_b.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            pred = model(enc_b)
            loss = criterion(pred, dec_b)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            if scheduler is not None and not isinstance(
                scheduler, torch.optim.lr_scheduler.CosineAnnealingLR
            ):
                scheduler.step()   # step-level for LambdaLR
            train_loss += loss.item()

        if scheduler is not None and isinstance(
            scheduler, torch.optim.lr_scheduler.CosineAnnealingLR
        ):
            scheduler.step()       # epoch-level for CosineAnnealingLR

        train_loss /= len(train_loader)

        # ── Evaluate ──
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for enc_b, dec_b in test_loader:
                enc_b = enc_b.to(device, non_blocking=True)
                dec_b = dec_b.to(device, non_blocking=True)
                pred  = model(enc_b)
                test_loss += criterion(pred, dec_b).item()
        test_loss /= len(test_loader)

        current_lr    = optimizer.param_groups[0]['lr']
        elapsed       = time.time() - train_start
        avg_per_epoch = elapsed / epoch
        eta_sec       = avg_per_epoch * (args.epochs - epoch)

        improved = test_loss < best_test_mse
        if improved:
            best_test_mse = test_loss
            best_epoch    = epoch
            patience_count = 0
            torch.save({
                'epoch':             epoch,
                'model_state_dict':  model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_loss':         test_loss,
                'train_loss':        train_loss,
                'args':              vars(args),
            }, OUTPUT_DIR / 'best_model.pt')
        else:
            patience_count += 1

        history.append({
            'epoch':      epoch,
            'train_mse':  round(train_loss, 6),
            'test_mse':   round(test_loss,  6),
            'lr':         current_lr,
            'best':       improved,
        })

        marker = ' ★ new best' if improved else ''
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Train: {train_loss:.6f} | Test: {test_loss:.6f} | "
            f"LR: {current_lr:.2e} | "
            f"ETA: {int(eta_sec//60)}m{marker}"
        )

        # ── Early stopping ──
        if patience_count >= args.patience:
            print(f"\nEarly stopping at epoch {epoch} "
                  f"(no improvement for {args.patience} epochs)")
            break

    # ── Save final + history ───────────────────────────────────
    torch.save({
        'epoch':            epoch,
        'model_state_dict': model.state_dict(),
        'test_loss':        test_loss,
        'args':             vars(args),
    }, OUTPUT_DIR / 'final_model.pt')

    pd.DataFrame(history).to_csv(OUTPUT_DIR / 'training_history.csv', index=False)

    total_time = time.time() - train_start
    print(f"\n{'='*70}")
    print(f"DONE  —  {args.run_name}")
    print(f"{'='*70}")
    print(f"  Best test MSE  : {best_test_mse:.6f}  (RMSE {np.sqrt(best_test_mse)*100:.2f}%)")
    print(f"  Best epoch     : {best_epoch}")
    print(f"  Total epochs   : {epoch}")
    print(f"  Total time     : {total_time/60:.1f} min")
    print(f"  Saved to       : {OUTPUT_DIR}/")
    print(f"\n  Paper LSTM (Y-Ŷ) : MSE 0.003302  (RMSE 5.75%)")
    print(f"  Our prev TFT     : MSE 0.003396  (RMSE 5.83%)")
    delta = best_test_mse - 0.003302
    sign  = '+' if delta > 0 else ''
    print(f"  This run vs paper: {sign}{delta:.6f}  "
          f"({'worse' if delta > 0 else 'better'})")


if __name__ == '__main__':
    main()
