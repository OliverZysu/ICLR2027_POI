"""Spatio-Temporal RNN for immediate Next-POI prediction."""

from __future__ import annotations

from typing import Dict, List

import torch
from torch import nn


class STRNN(nn.Module):
    """RNN with continuous interpolation over time and distance matrices.

    Time and distance features are gaps between *observed prefix events*.  The
    future target timestamp/coordinate is never supplied to the model.
    """

    def __init__(
        self,
        num_users: int,
        num_pois: int,
        num_categories: int,
        hidden_dim: int = 64,
        time_bins: int = 12,
        distance_bins: int = 12,
        max_time_hours: float = 720.0,
        max_distance_km: float = 500.0,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if time_bins < 2 or distance_bins < 2:
            raise ValueError("time_bins and distance_bins must be >= 2")
        self.num_pois = int(num_pois)
        self.poi_padding_idx = self.num_pois
        self.category_padding_idx = int(num_categories)
        self.hidden_dim = int(hidden_dim)
        self.time_bins = int(time_bins)
        self.distance_bins = int(distance_bins)
        self.max_time_hours = float(max_time_hours)
        self.max_distance_km = float(max_distance_km)

        self.poi_embedding = nn.Embedding(num_pois + 1, hidden_dim, padding_idx=self.poi_padding_idx)
        self.user_embedding = nn.Embedding(num_users, hidden_dim)
        self.output_embedding = nn.Embedding(num_pois, hidden_dim)
        self.time_matrices = nn.Parameter(torch.empty(time_bins, hidden_dim, hidden_dim))
        self.distance_matrices = nn.Parameter(
            torch.empty(distance_bins, hidden_dim, hidden_dim)
        )
        self.recurrent = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(hidden_dim))
        self.poi_bias = nn.Parameter(torch.zeros(num_pois))
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.poi_embedding.weight)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.output_embedding.weight)
        nn.init.xavier_uniform_(self.time_matrices)
        nn.init.xavier_uniform_(self.distance_matrices)
        nn.init.orthogonal_(self.recurrent.weight)
        nn.init.zeros_(self.bias)
        nn.init.zeros_(self.poi_bias)
        with torch.no_grad():
            self.poi_embedding.weight[self.poi_padding_idx].zero_()

    @staticmethod
    def _interpolate(
        values: torch.Tensor,
        table: torch.Tensor,
        max_value: float,
    ) -> torch.Tensor:
        # Log spacing allocates useful resolution to the many short gaps while
        # still covering long gaps.  Adjacent matrices are linearly interpolated.
        bins = table.size(0)
        denominator = torch.log1p(values.new_tensor(max(max_value, 1e-6)))
        position = torch.log1p(values.clamp(min=0.0, max=max_value)) / denominator
        position = position * float(bins - 1)
        lower = position.floor().long().clamp(0, bins - 1)
        upper = (lower + 1).clamp(max=bins - 1)
        alpha = (position - lower.float()).view(-1, 1, 1)
        return table[lower] * (1.0 - alpha) + table[upper] * alpha

    def forward(self, batch: Dict[str, torch.Tensor | List[str]]) -> torch.Tensor:
        poi = batch["poi"]
        user = batch["user"]
        lengths = batch["lengths"]
        delta_t = batch["delta_time_h"]
        delta_d = batch["delta_distance_km"]
        assert all(
            isinstance(x, torch.Tensor) for x in (poi, user, lengths, delta_t, delta_d)
        )

        x = self.poi_embedding(poi)
        batch_size, seq_len, _ = x.shape
        h = x.new_zeros(batch_size, self.hidden_dim)
        user_vec = self.user_embedding(user)
        outputs = []
        for step in range(seq_len):
            t_matrix = self._interpolate(
                delta_t[:, step], self.time_matrices, self.max_time_hours
            )
            d_matrix = self._interpolate(
                delta_d[:, step], self.distance_matrices, self.max_distance_km
            )
            transformed = torch.bmm(t_matrix, x[:, step].unsqueeze(-1)).squeeze(-1)
            transformed = torch.bmm(d_matrix, transformed.unsqueeze(-1)).squeeze(-1)
            proposal = torch.sigmoid(transformed + self.recurrent(h) + self.bias)
            valid = step < lengths
            h = torch.where(valid[:, None], proposal, h)
            outputs.append(h)
        hidden = self.dropout(torch.stack(outputs, dim=1) + user_vec[:, None, :])
        return torch.einsum("bsd,vd->bsv", hidden, self.output_embedding.weight) + self.poi_bias
