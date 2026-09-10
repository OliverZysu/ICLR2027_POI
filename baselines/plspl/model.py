"""Personalized Long- and Short-term Preference Learning (PLSPL)."""

from __future__ import annotations

import math
from typing import Dict, List

import torch
from torch import nn


class PLSPL(nn.Module):
    """Causal PyTorch adaptation of PLSPL for the shared benchmark protocol.

    Two recurrent branches model short-term POI and category transitions.  A
    causal attention branch aggregates the entire observed prefix as long-term
    context.  User-specific mixture weights combine the three branches before
    full-candidate scoring.  No future category, time or POI is exposed.
    """

    def __init__(
        self,
        num_users: int,
        num_pois: int,
        num_categories: int,
        embedding_dim: int = 64,
        context_dim: int = 32,
        hidden_dim: int = 96,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_pois = int(num_pois)
        self.poi_padding_idx = self.num_pois
        self.category_padding_idx = int(num_categories)
        self.hidden_dim = int(hidden_dim)

        self.poi_embedding = nn.Embedding(
            num_pois + 1, embedding_dim, padding_idx=self.poi_padding_idx
        )
        self.category_embedding = nn.Embedding(
            num_categories + 1,
            context_dim,
            padding_idx=self.category_padding_idx,
        )
        self.user_embedding = nn.Embedding(num_users, context_dim)
        self.hour_embedding = nn.Embedding(24, context_dim)
        self.weekday_embedding = nn.Embedding(7, context_dim)

        self.poi_lstm = nn.LSTM(
            embedding_dim + 2 * context_dim,
            hidden_dim,
            batch_first=True,
        )
        self.category_lstm = nn.LSTM(
            3 * context_dim,
            hidden_dim,
            batch_first=True,
        )

        self.long_context = nn.Linear(embedding_dim + 3 * context_dim, hidden_dim)
        self.long_query = nn.Linear(context_dim, hidden_dim, bias=False)
        self.poi_short_projection = nn.Linear(hidden_dim, hidden_dim)
        self.category_short_projection = nn.Linear(hidden_dim, hidden_dim)
        self.long_projection = nn.Linear(hidden_dim, hidden_dim)
        self.user_mixture = nn.Embedding(num_users, 3)
        self.output_embedding = nn.Embedding(num_pois, hidden_dim)
        self.poi_bias = nn.Parameter(torch.zeros(num_pois))
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for embedding in (
            self.poi_embedding,
            self.category_embedding,
            self.user_embedding,
            self.hour_embedding,
            self.weekday_embedding,
            self.output_embedding,
        ):
            nn.init.xavier_uniform_(embedding.weight)
        nn.init.zeros_(self.user_mixture.weight)
        nn.init.zeros_(self.poi_bias)
        for lstm in (self.poi_lstm, self.category_lstm):
            for name, parameter in lstm.named_parameters():
                if "weight" in name:
                    nn.init.xavier_uniform_(parameter)
                else:
                    nn.init.zeros_(parameter)
        for layer in (
            self.long_context,
            self.long_query,
            self.poi_short_projection,
            self.category_short_projection,
            self.long_projection,
        ):
            if layer.weight.ndim >= 2:
                nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        with torch.no_grad():
            self.poi_embedding.weight[self.poi_padding_idx].zero_()
            self.category_embedding.weight[self.category_padding_idx].zero_()

    def _causal_long_context(
        self,
        context: torch.Tensor,
        user_query: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = context.shape
        key_score = (context * user_query[:, None, :]).sum(-1) / math.sqrt(hidden_dim)
        positions = torch.arange(seq_len, device=context.device)
        valid_keys = positions[None, :] < lengths[:, None]
        causal = positions[None, :] <= positions[:, None]  # query row t attends to key <= t
        allowed = causal[None, :, :] & valid_keys[:, None, :]
        logits = key_score[:, None, :].expand(batch_size, seq_len, seq_len)
        logits = logits.masked_fill(~allowed, torch.finfo(logits.dtype).min)
        attention = torch.softmax(logits, dim=-1)
        attention = torch.where(allowed, attention, torch.zeros_like(attention))
        return torch.bmm(attention, context)

    def forward(self, batch: Dict[str, torch.Tensor | List[str]]) -> torch.Tensor:
        poi = batch["poi"]
        category = batch["category"]
        user = batch["user"]
        hour = batch["hour"]
        weekday = batch["weekday"]
        lengths = batch["lengths"]
        assert all(
            isinstance(x, torch.Tensor)
            for x in (poi, category, user, hour, weekday, lengths)
        )

        poi_e = self.poi_embedding(poi)
        cat_e = self.category_embedding(category)
        user_e = self.user_embedding(user)
        hour_e = self.hour_embedding(hour)
        weekday_e = self.weekday_embedding(weekday)
        repeated_user = user_e[:, None, :].expand(-1, poi.size(1), -1)

        poi_short, _ = self.poi_lstm(torch.cat([poi_e, repeated_user, hour_e], dim=-1))
        category_short, _ = self.category_lstm(
            torch.cat([cat_e, repeated_user, hour_e], dim=-1)
        )

        context = torch.tanh(
            self.long_context(torch.cat([poi_e, cat_e, hour_e, weekday_e], dim=-1))
        )
        query = self.long_query(user_e)
        long_state = self._causal_long_context(context, query, lengths)

        branches = torch.stack(
            [
                torch.tanh(self.poi_short_projection(poi_short)),
                torch.tanh(self.category_short_projection(category_short)),
                torch.tanh(self.long_projection(long_state)),
            ],
            dim=2,
        )  # (B,S,3,H)
        mixture = torch.softmax(self.user_mixture(user), dim=-1)
        fused = (branches * mixture[:, None, :, None]).sum(dim=2)
        fused = self.dropout(fused)
        return torch.einsum("bsd,vd->bsv", fused, self.output_embedding.weight) + self.poi_bias
