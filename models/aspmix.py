"""ASPMix: Additive / Projection Mixture of user-shared and user-specific prefs.

Idea
----
- *Shared* branch: population LSTM over check-in features (common dynamics).
- *Specific* branch: user-conditioned LSTM (personal residual dynamics).
- Two composition operators always computed:
    z_add  = shared + specific
    z_proj = Proj_u(shared)   (user low-rank projection of shared space)
- A context gate mixes add vs proj — matching the hypothesis that the
  shared/specific relation may be additive *or* projective.
- Spatio region features condition both branches (商圈 context).
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .blocks import (
    CausalLSTMEncoder,
    CheckinFeatureEncoder,
    NextPoiScoreHead,
    RegionProjection,
)


class ASPMix(nn.Module):
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
        del n_pattern  # reserved for API symmetry with siblings
        self.register_buffer("poi_region", poi_region.long())
        num_regions = int(self.poi_region.max().item()) + 1

        self.feats = CheckinFeatureEncoder(
            num_users, num_pois, num_categories, num_regions, embedding_dim, context_dim, dropout
        )
        self.shared_enc = CausalLSTMEncoder(self.feats.out_dim, hidden_dim, dropout=dropout)
        self.specific_enc = CausalLSTMEncoder(
            self.feats.out_dim + context_dim, hidden_dim, dropout=dropout
        )
        self.user_proj = RegionProjection(num_users, hidden_dim, rank=proj_rank)
        self.region_bias = nn.Linear(context_dim, hidden_dim)
        self.gate = nn.Sequential(
            nn.Linear(2 * hidden_dim + context_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.dropout = nn.Dropout(dropout)
        self.score = NextPoiScoreHead(num_pois, hidden_dim, embedding_dim)

        nn.init.xavier_uniform_(self.region_bias.weight)
        nn.init.zeros_(self.region_bias.bias)
        for m in self.gate:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        x, user_e, region_e = self.feats(batch, self.poi_region)
        lengths = batch["lengths"]
        user_rep = user_e[:, None, :].expand(-1, x.size(1), -1)

        shared = self.shared_enc(x, lengths) + self.region_bias(region_e)
        specific = self.specific_enc(torch.cat([x, user_rep], dim=-1), lengths)

        # reuse RegionProjection module with user ids as "region" keys
        user_ids = batch["user"][:, None].expand(-1, x.size(1))
        z_proj = self.user_proj(shared, user_ids) + specific
        z_add = shared + specific

        g = torch.sigmoid(self.gate(torch.cat([shared, specific, user_rep], dim=-1)))
        state = self.dropout(g * z_add + (1.0 - g) * z_proj)
        return self.score(state, batch["poi"])
