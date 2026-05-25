"""
losses.py — Asymmetric loss functions for the autoresearch agent.

Both follow the same API:
    forward(y_pred, y_true) → scalar
Penalize under-prediction (y_true > y_pred) more than over-prediction.
"""
import torch
import torch.nn as nn


class AsymmetricMSE(nn.Module):
    """
    L = under_penalty * r²  if r > 0  (under-prediction)
    L = over_penalty  * r²  if r ≤ 0  (over-prediction)
    where r = y_true - y_pred.
    α weights the already-squared error (α·e²), NOT the error before squaring.
    """
    def __init__(self, under_penalty: float = 3.0,
                 over_penalty:  float = 1.0,
                 reduction:     str   = 'mean'):
        super().__init__()
        self.under_penalty = under_penalty
        self.over_penalty  = over_penalty
        self.reduction     = reduction

    def forward(self, y_pred, y_true):
        r = y_true - y_pred
        w = torch.where(r > 0, self.under_penalty, self.over_penalty)
        loss = w * r.pow(2)
        if self.reduction == 'mean': return loss.mean()
        if self.reduction == 'sum':  return loss.sum()
        return loss


class SmoothAsymmetricHuberLoss(nn.Module):
    """
    Per Farhan's spec (Huber + asymmetry):
       |r| ≤ δ:  weight * 0.5 * r²            (quadratic, gentle)
       |r| > δ:  weight * (δ|r| - 0.5*δ²)     (linear, robust)
    weight = under_penalty if r > 0 else over_penalty.
    For normalized targets in [0,1], δ=0.1 is the recommended start (Farhan's doc).
    """
    def __init__(self, delta:         float = 0.1,
                 under_penalty: float = 3.0,
                 over_penalty:  float = 1.0,
                 reduction:     str   = 'mean'):
        super().__init__()
        self.delta         = delta
        self.under_penalty = under_penalty
        self.over_penalty  = over_penalty
        self.reduction     = reduction

    def forward(self, y_pred, y_true):
        r = y_true - y_pred
        w = torch.where(r > 0, self.under_penalty, self.over_penalty)
        abs_r     = r.abs()
        quadratic = torch.clamp(abs_r, max=self.delta)
        linear    = abs_r - quadratic
        huber     = 0.5 * quadratic.pow(2) + self.delta * linear
        loss = w * huber
        if self.reduction == 'mean': return loss.mean()
        if self.reduction == 'sum':  return loss.sum()
        return loss
