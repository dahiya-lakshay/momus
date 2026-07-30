"""
Combined Dice + BCE loss for binary tamper-mask segmentation.

BCE alone struggles with the heavy class imbalance here (tampered
regions are typically 3-15% of image area); Dice loss directly
optimizes for mask overlap and is far less sensitive to that
imbalance. We use a weighted sum of both, weights configurable via
config.yaml -> train.dice_weight / bce_weight.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    probs = probs.flatten(1)
    target = target.flatten(1)
    intersection = (probs * target).sum(dim=1)
    union = probs.sum(dim=1) + target.sum(dim=1)
    dice = (2 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()


class CombinedDiceBCELoss(nn.Module):
    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce_val = self.bce(logits, target)
        dice_val = dice_loss(logits, target)
        return self.bce_weight * bce_val + self.dice_weight * dice_val, {
            "bce": bce_val.item(),
            "dice": dice_val.item(),
        }


def build_loss(cfg: dict) -> CombinedDiceBCELoss:
    t_cfg = cfg["train"]
    return CombinedDiceBCELoss(dice_weight=t_cfg["dice_weight"], bce_weight=t_cfg["bce_weight"])
