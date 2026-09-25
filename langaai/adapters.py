"""Per-side entry adapters.

The two towers share an architecture but not a distribution — one sees antibody
variable domains, the other arbitrary antigens — so each side gets its own
LayerNorm affine and its own projection. The projection is identity-initialised,
so training starts from normalised raw features and any rotation away from them
has to be earned.

Contents:
    SideAdapter: LayerNorm + identity-init Linear, one instance per side.
        forward: Normalise then project.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["SideAdapter"]


class SideAdapter(nn.Module):
    """LayerNorm followed by an identity-initialised linear map."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, dim, bias=False)
        nn.init.eye_(self.proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.norm(x))
