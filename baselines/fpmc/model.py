"""PyTorch FPMC adapted to the unified full-candidate Next-POI protocol."""

from __future__ import annotations

from typing import Dict, List

import torch
from torch import nn


class FPMC(nn.Module):
    """Factorizing Personalized Markov Chains.

    The score for candidate ``j`` after POI ``i`` for user ``u`` is

        <U_u, V_j> + <M_i, N_j> + b_j,

    combining a user-specific long-term preference with a first-order Markov
    transition.  Basket size is one, which is the standard sequential form for
    Next-POI recommendation.
    """

    def __init__(
        self,
        num_users: int,
        num_pois: int,
        num_categories: int,
        mf_dim: int = 64,
        mc_dim: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_pois = int(num_pois)
        self.poi_padding_idx = self.num_pois
        self.category_padding_idx = int(num_categories)
        self.user_preference = nn.Embedding(num_users, mf_dim)
        self.poi_preference = nn.Embedding(num_pois, mf_dim)
        self.previous_poi = nn.Embedding(num_pois + 1, mc_dim, padding_idx=self.poi_padding_idx)
        self.next_poi = nn.Embedding(num_pois, mc_dim)
        self.poi_bias = nn.Parameter(torch.zeros(num_pois))
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in (
            self.user_preference,
            self.poi_preference,
            self.previous_poi,
            self.next_poi,
        ):
            nn.init.normal_(module.weight, std=0.01)
        with torch.no_grad():
            self.previous_poi.weight[self.poi_padding_idx].zero_()
        nn.init.zeros_(self.poi_bias)

    def forward(self, batch: Dict[str, torch.Tensor | List[str]]) -> torch.Tensor:
        user = batch["user"]
        poi = batch["poi"]
        assert isinstance(user, torch.Tensor) and isinstance(poi, torch.Tensor)
        user_vec = self.dropout(self.user_preference(user))
        previous_vec = self.dropout(self.previous_poi(poi))

        preference_scores = user_vec @ self.poi_preference.weight.t()  # (B,V)
        transition_scores = torch.einsum("bsd,vd->bsv", previous_vec, self.next_poi.weight)
        return transition_scores + preference_scores[:, None, :] + self.poi_bias
