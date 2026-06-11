"""Losses for the multi-task detector.

The central problem is extreme class imbalance: in a 1024x1024 tile a
vessel covers ~10 of 1,048,576 pixels. A plain BCE would be dominated by
the easy background and the model would learn to output zero everywhere.

Heatmap loss — penalty-reduced focal loss (CornerNet/CenterNet):

    pixel is a peak (y == 1):    (1 - p)^alpha * log(p)
    pixel is background:         (1 - y)^beta * p^alpha * log(1 - p)

  * the focal term (1-p)^alpha / p^alpha down-weights pixels the model
    already gets right, focusing gradient on the hard ones;
  * the (1-y)^beta term softens the penalty *near* a true peak, because
    pixels one step from a vessel centre are not really negatives.

Attribute losses are computed only at ground-truth points and masked where
the attribute is unknown (xView3 has NaNs in is_fishing / length).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import gather_at_points


def heatmap_focal_loss(pred_logits: torch.Tensor, target: torch.Tensor,
                       alpha: float = 2.0, beta: float = 4.0) -> torch.Tensor:
    """CenterNet penalty-reduced pixelwise focal loss.

    pred_logits, target: (B, 1, H, W); target is the rendered Gaussian map
    with exact peaks equal to 1.
    """
    pred = torch.sigmoid(pred_logits).clamp(1e-6, 1 - 1e-6)
    pos = target.eq(1.0).float()
    neg = 1.0 - pos

    pos_loss = -((1 - pred) ** alpha) * torch.log(pred) * pos
    neg_loss = -((1 - target) ** beta) * (pred ** alpha) * torch.log(1 - pred) * neg

    n_pos = pos.sum().clamp(min=1.0)
    return (pos_loss.sum() + neg_loss.sum()) / n_pos


def masked_point_bce(logit_map: torch.Tensor, rows: torch.Tensor,
                     cols: torch.Tensor, target: torch.Tensor,
                     mask: torch.Tensor) -> torch.Tensor:
    """BCE at ground-truth points, ignoring masked (unknown) entries."""
    logits = gather_at_points(logit_map, rows, cols)
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (loss * mask).sum() / mask.sum().clamp(min=1.0)


def masked_point_l1(pred_map: torch.Tensor, rows: torch.Tensor,
                    cols: torch.Tensor, target: torch.Tensor,
                    mask: torch.Tensor) -> torch.Tensor:
    """Smooth-L1 (Huber) regression at ground-truth points, masked."""
    preds = gather_at_points(pred_map, rows, cols)
    loss = F.smooth_l1_loss(preds, target, reduction="none")
    return (loss * mask).sum() / mask.sum().clamp(min=1.0)


class MultiTaskLoss(nn.Module):
    """Weighted sum of heatmap + vessel + fishing + length losses."""

    def __init__(self, cfg):
        super().__init__()
        t = cfg.train
        self.w_heatmap = t.w_heatmap
        self.w_vessel = t.w_vessel
        self.w_fishing = t.w_fishing
        self.w_length = t.w_length

    def forward(self, outputs: dict, batch: dict) -> dict:
        rows, cols = batch["pt_rows"], batch["pt_cols"]
        valid = batch["valid"]

        l_heat = heatmap_focal_loss(outputs["heatmap"], batch["heatmap"])
        l_vessel = masked_point_bce(outputs["vessel"], rows, cols,
                                    batch["vessel"], batch["vessel_mask"] * valid)
        l_fishing = masked_point_bce(outputs["fishing"], rows, cols,
                                     batch["fishing"], batch["fishing_mask"] * valid)
        l_length = masked_point_l1(outputs["length"], rows, cols,
                                   batch["length"], batch["length_mask"] * valid)

        total = (self.w_heatmap * l_heat + self.w_vessel * l_vessel
                 + self.w_fishing * l_fishing + self.w_length * l_length)
        return {
            "total": total,
            "heatmap": l_heat.detach(),
            "vessel": l_vessel.detach(),
            "fishing": l_fishing.detach(),
            "length": l_length.detach(),
        }
