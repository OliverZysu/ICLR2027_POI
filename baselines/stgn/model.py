"""Spatio-Temporal Gated Network (STGN) for Next-POI recommendation."""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn.functional as F
from torch import nn


class SpatioTemporalGatedCell(nn.Module):
    """STGN/STGCN recurrent cell from the AAAI 2019 formulation.

    ``coupled=False`` implements STGN.  ``coupled=True`` implements the paper's
    parameter-reduced STGCN variant, where input/forget behaviour is coupled.
    This STGCN is *not* the unrelated graph-convolution model with the same
    acronym.
    """

    def __init__(self, input_dim: int, hidden_dim: int, coupled: bool = False) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.coupled = coupled
        gate_count = 2 if coupled else 3  # i,g or i,f,g
        self.base_gates = nn.Linear(input_dim + hidden_dim, gate_count * hidden_dim)

        self.x_t1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.x_t2 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.x_d1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.x_d2 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.raw_t1 = nn.Parameter(torch.zeros(hidden_dim))
        self.t2 = nn.Parameter(torch.zeros(hidden_dim))
        self.raw_d1 = nn.Parameter(torch.zeros(hidden_dim))
        self.d2 = nn.Parameter(torch.zeros(hidden_dim))
        self.b_t1 = nn.Parameter(torch.zeros(hidden_dim))
        self.b_t2 = nn.Parameter(torch.zeros(hidden_dim))
        self.b_d1 = nn.Parameter(torch.zeros(hidden_dim))
        self.b_d2 = nn.Parameter(torch.zeros(hidden_dim))

        self.output_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_time = nn.Parameter(torch.zeros(hidden_dim))
        self.output_distance = nn.Parameter(torch.zeros(hidden_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in (
            self.base_gates,
            self.x_t1,
            self.x_t2,
            self.x_d1,
            self.x_d2,
            self.output_gate,
        ):
            for name, parameter in module.named_parameters():
                if parameter.ndim >= 2:
                    nn.init.xavier_uniform_(parameter)
                else:
                    nn.init.zeros_(parameter)
        for parameter in (
            self.raw_t1,
            self.t2,
            self.raw_d1,
            self.d2,
            self.b_t1,
            self.b_t2,
            self.b_d1,
            self.b_d2,
            self.output_time,
            self.output_distance,
        ):
            nn.init.zeros_(parameter)

    @staticmethod
    def _nested_gate(
        x_term: torch.Tensor,
        delta: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        return torch.sigmoid(x_term + torch.sigmoid(delta[:, None] * weight) + bias)

    def forward(
        self,
        x: torch.Tensor,
        h_prev: torch.Tensor,
        c_prev: torch.Tensor,
        delta_time_h: torch.Tensor,
        delta_distance_km: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        log_t = torch.log1p(delta_time_h.clamp_min(0.0))
        log_d = torch.log1p(delta_distance_km.clamp_min(0.0))
        base = self.base_gates(torch.cat([h_prev, x], dim=-1))
        if self.coupled:
            input_gate, candidate = base.chunk(2, dim=-1)
            input_gate = torch.sigmoid(input_gate)
            candidate = torch.tanh(candidate)
            forget_gate = None
        else:
            input_gate, forget_gate, candidate = base.chunk(3, dim=-1)
            input_gate = torch.sigmoid(input_gate)
            forget_gate = torch.sigmoid(forget_gate)
            candidate = torch.tanh(candidate)

        # The paper constrains the short-term T1/D1 interval weights to be
        # non-positive; -softplus provides that constraint during optimization.
        t1 = self._nested_gate(self.x_t1(x), log_t, -F.softplus(self.raw_t1), self.b_t1)
        t2 = self._nested_gate(self.x_t2(x), log_t, self.t2, self.b_t2)
        d1 = self._nested_gate(self.x_d1(x), log_d, -F.softplus(self.raw_d1), self.b_d1)
        d2 = self._nested_gate(self.x_d2(x), log_d, self.d2, self.b_d2)

        if self.coupled:
            short_strength = input_gate * t1 * d1
            c_short = (1.0 - short_strength) * c_prev + short_strength * candidate
            c_new = (1.0 - input_gate) * c_prev + input_gate * t2 * d2 * candidate
        else:
            assert forget_gate is not None
            c_short = forget_gate * c_prev + input_gate * t1 * d1 * candidate
            c_new = forget_gate * c_prev + input_gate * t2 * d2 * candidate

        output = torch.sigmoid(
            self.output_gate(torch.cat([h_prev, x], dim=-1))
            + log_t[:, None] * self.output_time
            + log_d[:, None] * self.output_distance
        )
        h_new = output * torch.tanh(c_short)
        return h_new, c_new


class STGN(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_pois: int,
        num_categories: int,
        embedding_dim: int = 64,
        hidden_dim: int = 64,
        dropout: float = 0.2,
        coupled: bool = False,
    ) -> None:
        super().__init__()
        self.num_pois = int(num_pois)
        self.poi_padding_idx = self.num_pois
        self.category_padding_idx = int(num_categories)
        self.hidden_dim = int(hidden_dim)
        self.coupled = bool(coupled)

        self.poi_embedding = nn.Embedding(
            num_pois + 1, embedding_dim, padding_idx=self.poi_padding_idx
        )
        self.user_embedding = nn.Embedding(num_users, hidden_dim)
        self.cell = SpatioTemporalGatedCell(embedding_dim, hidden_dim, coupled=coupled)
        self.output_embedding = nn.Embedding(num_pois, hidden_dim)
        self.poi_bias = nn.Parameter(torch.zeros(num_pois))
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.poi_embedding.weight)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.output_embedding.weight)
        nn.init.zeros_(self.poi_bias)
        with torch.no_grad():
            self.poi_embedding.weight[self.poi_padding_idx].zero_()

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
        c = x.new_zeros(batch_size, self.hidden_dim)
        outputs = []
        for step in range(seq_len):
            h_new, c_new = self.cell(x[:, step], h, c, delta_t[:, step], delta_d[:, step])
            valid = step < lengths
            h = torch.where(valid[:, None], h_new, h)
            c = torch.where(valid[:, None], c_new, c)
            outputs.append(h)
        hidden = self.dropout(torch.stack(outputs, dim=1) + self.user_embedding(user)[:, None, :])
        return torch.einsum("bsd,vd->bsv", hidden, self.output_embedding.weight) + self.poi_bias
