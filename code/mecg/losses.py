"""
Matryoshka multi-granularity loss.

L = (1/|M|) * sum_{m in M} w_m * BCE( g_m(z[:m]), y )

Notes on changes vs. the original submission
--------------------------------------------
* Label smoothing on a *multi-label* BCE objective is unusual and was applied
  unconditionally before. It is now off by default and exposed as an ablation
  (`label_smoothing: 0.0`), because it shifts targets to [eps/K, 1-eps+eps/K]
  and can only affect calibration, not AUC ranking -- so it should be justified
  empirically rather than assumed.
* Optional positive-class weighting (`pos_weight`) for the imbalanced HYP class.
* The loss no longer owns the classifiers; it consumes a dict of logits.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn


def compute_nesting_weights(nesting_dims: List[int], strategy: str = "equal"
                            ) -> Dict[int, float]:
    """Per-dimension loss weights, normalised so that mean(w) == 1."""
    dims = sorted(int(d) for d in nesting_dims)
    n = len(dims)
    if strategy == "equal":
        raw = {d: 1.0 for d in dims}
    elif strategy == "linear":
        raw = {d: float(i + 1) for i, d in enumerate(dims)}
    elif strategy == "exponential":
        raw = {d: float(2 ** i) for i, d in enumerate(dims)}
    elif strategy == "inverse":
        # up-weight the small dimensions (tests the "small dims are what matter"
        # hypothesis directly)
        raw = {d: float(n - i) for i, d in enumerate(dims)}
    else:
        raise ValueError(f"Unknown weight strategy '{strategy}'.")
    total = sum(raw.values())
    scale = n / total
    return {d: w * scale for d, w in raw.items()}


class MatryoshkaObjective(nn.Module):
    def __init__(self,
                 nesting_dims: List[int],
                 num_classes: int = 5,
                 weight_strategy: str = "equal",
                 label_smoothing: float = 0.0,
                 pos_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.nesting_dims = sorted(int(d) for d in nesting_dims)
        self.num_classes = num_classes
        self.label_smoothing = float(label_smoothing)
        self.weights = compute_nesting_weights(self.nesting_dims, weight_strategy)
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight)
        else:
            self.pos_weight = None

    def _smooth(self, targets: torch.Tensor) -> torch.Tensor:
        if self.label_smoothing <= 0:
            return targets
        eps = self.label_smoothing
        return targets * (1.0 - eps) + eps / self.num_classes

    def forward(self, logits_by_dim: Dict[int, torch.Tensor],
                targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        tgt = self._smooth(targets)
        total = None
        per_dim = {}
        for dim, logits in logits_by_dim.items():
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits, tgt.to(logits.dtype),
                pos_weight=(self.pos_weight.to(logits.dtype)
                            if self.pos_weight is not None else None),
                reduction="mean",
            )
            w = self.weights.get(dim, 1.0)
            total = (w * loss) if total is None else total + w * loss
            per_dim[dim] = loss.detach()
        total = total / max(len(logits_by_dim), 1)
        return {"total_loss": total, "per_dim_loss": per_dim}
