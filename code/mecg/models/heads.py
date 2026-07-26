"""
Classification heads.

The original submission put the nesting classifiers *inside* a class called
`MatryoshkaLoss`. That conflated trainable inference parameters with the loss
function, which is confusing and made the MRL-E variant awkward to express.
Here the head owns the parameters and exposes logits; the loss is a separate
function in `mecg.losses`.

Heads
-----
MatryoshkaHead   : one independent nn.Linear per nesting dimension  (MRL)
MatryoshkaEHead  : a single K x d weight matrix, column-sliced       (MRL-E)
LinearHead       : a single classifier at one fixed dimension        (baseline)
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn


class MatryoshkaHead(nn.Module):
    """Independent linear classifier per nesting dimension (Kusupati et al.)."""

    def __init__(self, nesting_dims: List[int], num_classes: int = 5):
        super().__init__()
        self.nesting_dims = sorted(int(d) for d in nesting_dims)
        self.num_classes = num_classes
        self.classifiers = nn.ModuleDict({
            str(d): nn.Linear(d, num_classes) for d in self.nesting_dims
        })
        for clf in self.classifiers.values():
            nn.init.xavier_uniform_(clf.weight)
            nn.init.zeros_(clf.bias)

    @property
    def max_dim(self) -> int:
        return self.nesting_dims[-1]

    def logits_at(self, z: torch.Tensor, dim: int) -> torch.Tensor:
        # nn.ModuleDict keys are strings, so the membership test must be on
        # str(dim); testing the int directly always fails.
        key = str(int(dim))
        if key not in self.classifiers:
            raise KeyError(f"dim={dim} not in nesting_dims={self.nesting_dims}")
        return self.classifiers[key](z[:, :int(dim)])

    def forward(self, z: torch.Tensor) -> Dict[int, torch.Tensor]:
        return {d: self.logits_at(z, d) for d in self.nesting_dims}

    def classifier_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class MatryoshkaEHead(nn.Module):
    """
    MRL-E: a single weight matrix W of shape (K, d_max) shared across all
    granularities; the classifier at dimension m uses W[:, :m].

    Parameter count is K*d_max + K instead of sum_m (K*m + K).
    """

    def __init__(self, nesting_dims: List[int], num_classes: int = 5):
        super().__init__()
        self.nesting_dims = sorted(int(d) for d in nesting_dims)
        self.num_classes = num_classes
        d_max = self.nesting_dims[-1]
        self.weight = nn.Parameter(torch.empty(num_classes, d_max))
        self.bias = nn.Parameter(torch.zeros(num_classes, len(self.nesting_dims)))
        nn.init.xavier_uniform_(self.weight)
        self._dim_index = {d: i for i, d in enumerate(self.nesting_dims)}

    @property
    def max_dim(self) -> int:
        return self.nesting_dims[-1]

    def logits_at(self, z: torch.Tensor, dim: int) -> torch.Tensor:
        if dim not in self._dim_index:
            raise KeyError(f"dim={dim} not in nesting_dims={self.nesting_dims}")
        w = self.weight[:, :dim]
        b = self.bias[:, self._dim_index[dim]]
        return torch.nn.functional.linear(z[:, :dim], w, b)

    def forward(self, z: torch.Tensor) -> Dict[int, torch.Tensor]:
        return {d: self.logits_at(z, d) for d in self.nesting_dims}

    def classifier_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class LinearHead(nn.Module):
    """Single-dimension classifier for fixed-dimension baselines."""

    def __init__(self, embedding_dim: int, num_classes: int = 5):
        super().__init__()
        self.nesting_dims = [int(embedding_dim)]
        self.num_classes = num_classes
        self.classifier = nn.Linear(embedding_dim, num_classes)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    @property
    def max_dim(self) -> int:
        return self.nesting_dims[0]

    def logits_at(self, z: torch.Tensor, dim: Optional[int] = None) -> torch.Tensor:
        return self.classifier(z)

    def forward(self, z: torch.Tensor) -> Dict[int, torch.Tensor]:
        return {self.nesting_dims[0]: self.classifier(z)}

    def classifier_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_head(kind: str, nesting_dims: List[int], num_classes: int = 5) -> nn.Module:
    kind = kind.lower()
    if kind == "mrl":
        return MatryoshkaHead(nesting_dims, num_classes)
    if kind in ("mrl-e", "mrl_e", "mrle"):
        return MatryoshkaEHead(nesting_dims, num_classes)
    if kind in ("linear", "fixed"):
        return LinearHead(int(nesting_dims[-1]), num_classes)
    raise ValueError(f"Unknown head kind '{kind}'.")
