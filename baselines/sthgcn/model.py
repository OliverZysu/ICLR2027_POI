"""STHGCN for the shared closed-set Next-POI recommendation protocol."""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn


def _causal_mask(length: int, device: torch.device) -> torch.Tensor:
    return torch.triu(
        torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1
    )


def _padding_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    return torch.arange(max_len, device=lengths.device)[None, :] >= lengths[:, None]


class SinusoidalPosition(nn.Module):
    def __init__(self, dim: int, max_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32)[:, None]
        divisor = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32)
            * (-math.log(10000.0) / dim)
        )
        encoding = torch.zeros(max_len, dim)
        encoding[:, 0::2] = torch.sin(position * divisor)
        encoding[:, 1::2] = torch.cos(
            position * divisor[: encoding[:, 1::2].shape[1]]
        )
        self.register_buffer("encoding", encoding, persistent=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.encoding[: x.size(1)][None, :, :])


class CausalTransformer(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_layers: int,
        ff_dim: int,
        dropout: float,
        max_len: int = 512,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
        )
        self.position = SinusoidalPosition(dim, max_len=max_len, dropout=dropout)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        x = self.position(x)
        seq_len = x.size(1)
        output = self.encoder(
            x.transpose(0, 1),
            mask=_causal_mask(seq_len, x.device),
            src_key_padding_mask=_padding_mask(lengths, seq_len),
        )
        return self.norm(output.transpose(0, 1))


class SparseResidualPropagation(nn.Module):
    def __init__(self, dim: int, layers: int = 1, dropout: float = 0.1) -> None:
        super().__init__()
        self.linears = nn.ModuleList(
            [nn.Linear(dim, dim, bias=False) for _ in range(layers)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        states = [x]
        hidden = x
        for linear, norm in zip(self.linears, self.norms):
            message = torch.sparse.mm(adjacency, linear(hidden))
            hidden = norm(hidden + self.dropout(F.gelu(message)))
            states.append(hidden)
        return torch.stack(states, dim=0).mean(dim=0)


class HypergraphPropagation(nn.Module):
    def __init__(self, dim: int, layers: int = 1, dropout: float = 0.1) -> None:
        super().__init__()
        self.edge_linears = nn.ModuleList(
            [nn.Linear(dim, dim, bias=False) for _ in range(layers)]
        )
        self.node_linears = nn.ModuleList(
            [nn.Linear(dim, dim, bias=False) for _ in range(layers)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_from_node: torch.Tensor,
        node_from_edge: torch.Tensor,
    ) -> torch.Tensor:
        states = [x]
        hidden = x
        for edge_linear, node_linear, norm in zip(
            self.edge_linears, self.node_linears, self.norms
        ):
            edge_hidden = torch.sparse.mm(edge_from_node, edge_linear(hidden))
            node_hidden = torch.sparse.mm(node_from_edge, node_linear(edge_hidden))
            hidden = norm(hidden + self.dropout(F.gelu(node_hidden)))
            states.append(hidden)
        return torch.stack(states, dim=0).mean(dim=0)


class STHGCN(nn.Module):
    """Spatio-temporal hierarchical hypergraph convolutional network.

    Trajectory, collaborative, region, region-time, and transition structures
    are built only from the training split. Their high-order POI views are
    fused before GRU and causal Transformer trajectory modelling.
    """

    def __init__(
        self,
        num_users: int,
        num_pois: int,
        num_categories: int,
        context: Dict[str, torch.Tensor],
        embedding_dim: int = 64,
        context_dim: int = 24,
        hidden_dim: int = 128,
        hyper_layers: int = 2,
        num_heads: int = 4,
        transformer_layers: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_pois = int(num_pois)
        self.poi_padding_idx = self.num_pois
        self.category_padding_idx = int(num_categories)

        for name in (
            "session_sp",
            "session_ps",
            "collab_up",
            "collab_pu",
            "st_ep",
            "st_pe",
            "region_rp",
            "region_pr",
            "transition",
        ):
            self.register_buffer(name, context[name])

        self.poi_embedding = nn.Embedding(
            num_pois + 1, embedding_dim, padding_idx=self.poi_padding_idx
        )
        self.user_embedding = nn.Embedding(num_users, context_dim)
        self.category_embedding = nn.Embedding(
            num_categories + 1, context_dim, padding_idx=self.category_padding_idx
        )
        self.hour_embedding = nn.Embedding(24, context_dim)
        self.weekday_embedding = nn.Embedding(7, context_dim)

        self.session_hypergraph = HypergraphPropagation(
            embedding_dim, hyper_layers, dropout
        )
        self.collaborative_hypergraph = HypergraphPropagation(
            embedding_dim, hyper_layers, dropout
        )
        self.spatiotemporal_hypergraph = HypergraphPropagation(
            embedding_dim, hyper_layers, dropout
        )
        self.region_hypergraph = HypergraphPropagation(
            embedding_dim, 1, dropout
        )
        self.transition_graph = SparseResidualPropagation(
            embedding_dim, 1, dropout
        )

        self.view_query = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.view_key = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.fusion_norm = nn.LayerNorm(embedding_dim)

        input_dim = embedding_dim + 4 * context_dim
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.temporal_encoder = CausalTransformer(
            hidden_dim,
            num_heads,
            transformer_layers,
            4 * hidden_dim,
            dropout,
        )
        self.output_projection = nn.Linear(hidden_dim, embedding_dim, bias=False)
        self.poi_bias = nn.Parameter(torch.zeros(num_pois))
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        embeddings = (
            self.poi_embedding,
            self.user_embedding,
            self.category_embedding,
            self.hour_embedding,
            self.weekday_embedding,
        )
        for embedding in embeddings:
            nn.init.xavier_uniform_(embedding.weight)
        nn.init.zeros_(self.poi_bias)
        with torch.no_grad():
            self.poi_embedding.weight[self.poi_padding_idx].zero_()
            self.category_embedding.weight[self.category_padding_idx].zero_()

    def _encode_pois(self) -> torch.Tensor:
        base = self.poi_embedding.weight[: self.num_pois]
        session = self.session_hypergraph(base, self.session_sp, self.session_ps)
        collaborative = self.collaborative_hypergraph(
            session, self.collab_up, self.collab_pu
        )
        spatiotemporal = self.spatiotemporal_hypergraph(
            base, self.st_ep, self.st_pe
        )
        region = self.region_hypergraph(base, self.region_rp, self.region_pr)
        transition = self.transition_graph(base, self.transition)

        views = torch.stack(
            [base, session, collaborative, spatiotemporal, region, transition],
            dim=1,
        )
        query = self.view_query(base)[:, None, :]
        score = (query * self.view_key(views)).sum(dim=-1)
        score = score / math.sqrt(float(base.size(-1)))
        weights = torch.softmax(score, dim=1)
        fused = base + (views * weights[:, :, None]).sum(dim=1)
        return self.fusion_norm(fused)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        poi = batch["poi"]
        category = batch["category"]
        user = batch["user"]
        hour = batch["hour"]
        weekday = batch["weekday"]
        lengths = batch["lengths"]

        candidate = self._encode_pois()
        candidate_with_padding = torch.cat(
            [candidate, candidate.new_zeros(1, candidate.size(-1))], dim=0
        )
        user_e = self.user_embedding(user)[:, None, :].expand(-1, poi.size(1), -1)
        inputs = self.input_projection(
            torch.cat(
                [
                    candidate_with_padding[poi],
                    self.category_embedding(category),
                    user_e,
                    self.hour_embedding(hour),
                    self.weekday_embedding(weekday),
                ],
                dim=-1,
            )
        )

        packed = nn.utils.rnn.pack_padded_sequence(
            inputs,
            lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, _ = self.gru(packed)
        recurrent, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=poi.size(1),
        )
        hidden = self.temporal_encoder(recurrent, lengths) + recurrent
        hidden = self.dropout(hidden)
        query = self.output_projection(hidden)
        return torch.einsum("bsd,vd->bsv", query, candidate) + self.poi_bias

    def compute_auxiliary_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        del batch
        return self.poi_bias.sum() * 0.0
