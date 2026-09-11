"""DualClusterNet: joint pattern-cluster × spatio-cluster factorization.

Idea
----
- Pattern soft-clustering captures *what* mobility routine is active.
- Spatio soft-clustering captures *where* (商圈) it happens.
- Shared interaction = elementwise product of projected pattern/spatio
  (population-level grammar).
- User-specific FiLM modulates the shared interaction.
- Additive / projective gate fuses shared and specific before scoring.
"""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn

from .blocks import (
    AddProjGate,
    CausalLSTMEncoder,
    CheckinFeatureEncoder,
    NextPoiScoreHead,
)


class DualClusterNet(nn.Module):
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
        n_spatio: int = 32,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.register_buffer("poi_region", poi_region.long())
        num_regions = int(self.poi_region.max().item()) + 1

        self.feats = CheckinFeatureEncoder(
            num_users, num_pois, num_categories, num_regions, embedding_dim, context_dim, dropout
        )
        self.encoder = CausalLSTMEncoder(self.feats.out_dim + context_dim, hidden_dim, dropout=dropout)

        self.n_pattern = n_pattern
        self.n_spatio = n_spatio
        self.pattern_proto = nn.Parameter(torch.randn(n_pattern, hidden_dim) * 0.02)
        self.spatio_proto = nn.Parameter(torch.randn(n_spatio, hidden_dim) * 0.02)
        self.pattern_assign = nn.Linear(hidden_dim, n_pattern)
        self.spatio_assign = nn.Linear(hidden_dim + context_dim, n_spatio)

        self.pattern_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.spatio_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.shared_mix = nn.Linear(hidden_dim, hidden_dim)

        self.user_gamma = nn.Linear(context_dim, hidden_dim)
        self.user_beta = nn.Linear(context_dim, hidden_dim)
        self.specific = nn.Linear(hidden_dim + context_dim, hidden_dim)
        self.fuse = AddProjGate(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.score = NextPoiScoreHead(num_pois, hidden_dim, embedding_dim)

        for m in (
            self.pattern_assign,
            self.spatio_assign,
            self.pattern_proj,
            self.spatio_proj,
            self.shared_mix,
            self.user_gamma,
            self.user_beta,
            self.specific,
        ):
            nn.init.xavier_uniform_(m.weight)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        x, user_e, region_e = self.feats(batch, self.poi_region)
        user_rep = user_e[:, None, :].expand(-1, x.size(1), -1)
        h = self.encoder(torch.cat([x, user_rep], dim=-1), batch["lengths"])

        alpha_p = torch.softmax(self.pattern_assign(h) / math.sqrt(h.size(-1)), dim=-1)
        alpha_s = torch.softmax(
            self.spatio_assign(torch.cat([h, region_e], dim=-1))
            / math.sqrt(h.size(-1) + region_e.size(-1)),
            dim=-1,
        )
        pattern = alpha_p @ self.pattern_proto
        spatio = alpha_s @ self.spatio_proto

        shared = torch.tanh(
            self.shared_mix(self.pattern_proj(pattern) * self.spatio_proj(spatio))
        )
        gamma = torch.sigmoid(self.user_gamma(user_rep))
        beta = self.user_beta(user_rep)
        shared_u = gamma * shared + beta
        specific = torch.tanh(self.specific(torch.cat([h, user_rep], dim=-1)))
        state = self.dropout(self.fuse(shared_u, specific))
        return self.score(state, batch["poi"])
