"""PCPNet: Pattern-Cluster Projection Network.

Idea
----
1. Soft-cluster each trajectory step into shared mobility *patterns*.
2. Project the shared pattern vector into the current *business-district*
   (spatio / 商圈) subspace.
3. Keep a user-specific residual; fuse shared⊙region and specific via a
   learned additive vs projective gate.
4. Score with preference · POI + Markov transition (baseline-competitive head).
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .blocks import (
    AddProjGate,
    CausalLSTMEncoder,
    CheckinFeatureEncoder,
    NextPoiScoreHead,
    RegionProjection,
    SoftPatternCluster,
)


class PCPNet(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_pois: int,
        num_categories: int,
        poi_region: torch.Tensor,
        embedding_dim: int = 64,
        context_dim: int = 32,
        hidden_dim: int = 128,
        n_pattern: int = 32,
        proj_rank: int = 32,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.register_buffer("poi_region", poi_region.long())
        num_regions = int(self.poi_region.max().item()) + 1

        self.feats = CheckinFeatureEncoder(
            num_users, num_pois, num_categories, num_regions, embedding_dim, context_dim, dropout
        )
        self.encoder = CausalLSTMEncoder(self.feats.out_dim + context_dim, hidden_dim, dropout=dropout)
        self.patterns = SoftPatternCluster(hidden_dim, n_pattern)
        self.region_proj = RegionProjection(num_regions, hidden_dim, rank=proj_rank)
        self.user_specific = nn.Sequential(
            nn.Linear(context_dim + hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.fuse = AddProjGate(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.score = NextPoiScoreHead(num_pois, hidden_dim, embedding_dim)

        for m in self.user_specific:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        x, user_e, region_e = self.feats(batch, self.poi_region)
        user_rep = user_e[:, None, :].expand(-1, x.size(1), -1)
        h = self.encoder(torch.cat([x, user_rep], dim=-1), batch["lengths"])

        _, pattern = self.patterns(h)  # shared pattern mixture
        safe_poi = batch["poi"].clamp(max=self.poi_region.numel() - 1)
        region_ids = self.poi_region[safe_poi]
        region_ids = torch.where(
            batch["poi"].eq(self.feats.poi_pad), torch.zeros_like(region_ids), region_ids
        )
        shared = self.region_proj(pattern, region_ids)  # 商圈投影后的 shared
        specific = self.user_specific(torch.cat([user_rep, h], dim=-1))
        state = self.dropout(self.fuse(shared, specific))
        return self.score(state, batch["poi"])
