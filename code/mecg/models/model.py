"""
Unified ECG model = backbone + head.

One class now covers all conditions the reviewers asked to compare:

    head=mrl    , backbone=inception1d   -> Inc1D-MRL   (proposed)
    head=mrl    , backbone=xresnet1d101  -> XRes101-MRL
    head=mrl-e  , backbone=*             -> MRL-E variant
    head=linear , backbone=*, dim=m      -> fixed-dimension baseline

Critically, this makes the *same-backbone* fixed-dimension baseline trivial to
run, which Reviewer 2 identified as the key missing comparison.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .backbones import create_backbone
from .heads import build_head


class ECGModel(nn.Module):
    def __init__(self, config: dict, verbose: bool = True):
        super().__init__()
        mcfg = config.get("model", {})
        ncfg = config.get("nesting", {})

        self.backbone_name = mcfg.get("backbone", "inception1d")
        self.head_kind = mcfg.get("head", "mrl")
        self.num_classes = int(mcfg.get("num_classes", 5))
        self.input_channels = int(mcfg.get("input_channels", 12))
        self.embedding_dim = int(mcfg.get("embedding_dim", 512))
        self.dropout = float(mcfg.get("dropout", 0.3))

        if self.head_kind in ("linear", "fixed"):
            self.nesting_dims = [self.embedding_dim]
        else:
            self.nesting_dims = sorted(
                int(d) for d in ncfg.get("dims", [16, 32, 64, 128, 256, 512])
            )
            self.embedding_dim = self.nesting_dims[-1]

        self.backbone = create_backbone(
            name=self.backbone_name,
            input_channels=self.input_channels,
            embedding_dim=self.embedding_dim,
            dropout=self.dropout,
            verbose=verbose,
            **mcfg.get("backbone_kwargs", {}),
        )
        self.head = build_head(self.head_kind, self.nesting_dims, self.num_classes)

        if verbose:
            bb = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
            hd = sum(p.numel() for p in self.head.parameters() if p.requires_grad)
            print(f"  head={self.head_kind} dims={self.nesting_dims}")
            print(f"  params: total={bb + hd:,} backbone={bb:,} head={hd:,}")

    # ------------------------------------------------------------------
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        return self.head(self.backbone(x))

    @torch.no_grad()
    def logits_from_embedding(self, z: torch.Tensor, dim: Optional[int] = None
                              ) -> torch.Tensor:
        dim = dim if dim is not None else self.head.max_dim
        return self.head.logits_at(z, dim)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor, dim: Optional[int] = None
                      ) -> torch.Tensor:
        return torch.sigmoid(self.logits_from_embedding(self.backbone(x), dim))


@torch.no_grad()
def probs_from_embeddings(model: ECGModel, embeddings: torch.Tensor, dim: int,
                          device: torch.device, chunk: int = 8192):
    """
    Score a stored embedding matrix in chunks.

    The original code moved the entire test embedding matrix to the accelerator
    in one call; chunking keeps memory flat and lets the same routine be used on
    the much larger external datasets (MIMIC-IV-ECG, CODE-15%).
    """
    outs = []
    for i in range(0, embeddings.shape[0], chunk):
        z = embeddings[i:i + chunk].to(device, non_blocking=True)
        outs.append(torch.sigmoid(model.head.logits_at(z, dim)).float().cpu())
    return torch.cat(outs, dim=0).numpy()
