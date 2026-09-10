"""DCHL for the shared closed-set Next-POI recommendation protocol."""

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
            raise ValueError("embedding_dim must be divisible by num_heads")
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
        encoded = self.encoder(
            x.transpose(0, 1),
            mask=_causal_mask(seq_len, x.device),
            src_key_padding_mask=_padding_mask(lengths, seq_len),
        )
        return self.norm(encoded.transpose(0, 1))


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


def _sampled_symmetric_infonce(
    first: torch.Tensor,
    second: torch.Tensor,
    temperature: float,
    max_samples: int,
) -> torch.Tensor:
    sample_count = min(first.size(0), second.size(0), int(max_samples))
    if sample_count <= 1:
        return (first.sum() + second.sum()) * 0.0
    if first.size(0) > sample_count:
        indices = torch.randperm(first.size(0), device=first.device)[:sample_count]
        first = first[indices]
        second = second[indices]
    else:
        first = first[:sample_count]
        second = second[:sample_count]
    first = F.normalize(first, dim=-1)
    second = F.normalize(second, dim=-1)
    logits = first @ second.transpose(0, 1)
    logits = logits / max(float(temperature), 1e-4)
    target = torch.arange(sample_count, device=first.device)
    return 0.5 * (
        F.cross_entropy(logits, target)
        + F.cross_entropy(logits.transpose(0, 1), target)
    )


class FactorisedHypergraphEncoder(nn.Module):
    def __init__(
        self, dim: int, num_factors: int, layers: int, dropout: float
    ) -> None:
        super().__init__()
        if dim % num_factors != 0:
            raise ValueError("embedding_dim must be divisible by num_factors")
        self.num_factors = int(num_factors)
        factor_dim = dim // num_factors
        self.encoders = nn.ModuleList(
            [
                HypergraphPropagation(factor_dim, layers=layers, dropout=dropout)
                for _ in range(num_factors)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_from_node: torch.Tensor,
        node_from_edge: torch.Tensor,
    ) -> torch.Tensor:
        chunks = x.chunk(self.num_factors, dim=-1)
        outputs = [
            encoder(chunk, edge_from_node, node_from_edge)
            for encoder, chunk in zip(self.encoders, chunks)
        ]
        return torch.cat(outputs, dim=-1)


class FactorisedGraphEncoder(nn.Module):
    def __init__(
        self, dim: int, num_factors: int, layers: int, dropout: float
    ) -> None:
        super().__init__()
        if dim % num_factors != 0:
            raise ValueError("embedding_dim must be divisible by num_factors")
        self.num_factors = int(num_factors)
        factor_dim = dim // num_factors
        self.encoders = nn.ModuleList(
            [
                SparseResidualPropagation(factor_dim, layers=layers, dropout=dropout)
                for _ in range(num_factors)
            ]
        )

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        outputs = [
            encoder(chunk, adjacency)
            for encoder, chunk in zip(
                self.encoders, x.chunk(self.num_factors, dim=-1)
            )
        ]
        return torch.cat(outputs, dim=-1)


class DCHL(nn.Module):
    """Disentangled contrastive hypergraph learning model.

    Collaborative, geographical, and directed-transition views are constructed
    from the training split. Each view is factorised, adaptively fused, and
    jointly trained with contrastive and factor-decorrelation objectives.
    """

    def __init__(
        self,
        num_users: int,
        num_pois: int,
        num_categories: int,
        context: Dict[str, torch.Tensor],
        embedding_dim: int = 64,
        context_dim: int = 24,
        num_factors: int = 4,
        hyper_layers: int = 1,
        transformer_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.2,
        contrastive_weight: float = 0.05,
        disentangle_weight: float = 0.01,
        temperature: float = 0.2,
        max_contrastive_samples: int = 384,
    ) -> None:
        super().__init__()
        if embedding_dim % num_factors != 0:
            raise ValueError("embedding_dim must be divisible by num_factors")

        self.num_pois = int(num_pois)
        self.poi_padding_idx = self.num_pois
        self.category_padding_idx = int(num_categories)
        self.num_factors = int(num_factors)
        self.contrastive_weight = float(contrastive_weight)
        self.disentangle_weight = float(disentangle_weight)
        self.temperature = float(temperature)
        self.max_contrastive_samples = int(max_contrastive_samples)

        for name in (
            "collab_up",
            "collab_pu",
            "geo_graph",
            "transition",
            "transition_rev",
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

        self.collaborative_encoder = FactorisedHypergraphEncoder(
            embedding_dim, num_factors, hyper_layers, dropout
        )
        self.geographical_encoder = FactorisedGraphEncoder(
            embedding_dim, num_factors, hyper_layers, dropout
        )
        self.transition_encoder = FactorisedGraphEncoder(
            embedding_dim, num_factors, hyper_layers, dropout
        )
        self.factor_view_logits = nn.Parameter(torch.zeros(num_factors, 3))
        self.poi_norm = nn.LayerNorm(embedding_dim)

        input_dim = embedding_dim + 4 * context_dim
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.GELU(),
            nn.LayerNorm(embedding_dim),
        )
        self.sequence_encoder = CausalTransformer(
            embedding_dim,
            num_heads,
            transformer_layers,
            4 * embedding_dim,
            dropout,
        )
        self.output_projection = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.poi_bias = nn.Parameter(torch.zeros(num_pois))
        self.dropout = nn.Dropout(dropout)

        self._cached_views = None
        self._cached_fused = None
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
        collaborative = self.collaborative_encoder(
            base, self.collab_up, self.collab_pu
        )
        geographical = self.geographical_encoder(base, self.geo_graph)
        transition = 0.5 * (
            self.transition_encoder(base, self.transition)
            + self.transition_encoder(base, self.transition_rev)
        )

        views = [collaborative, geographical, transition]
        factor_dim = base.size(-1) // self.num_factors
        weights = torch.softmax(self.factor_view_logits, dim=-1)
        fused_parts = []
        for factor in range(self.num_factors):
            start = factor * factor_dim
            end = (factor + 1) * factor_dim
            stack = torch.stack(
                [view[:, start:end] for view in views], dim=1
            )
            fused_parts.append(
                (stack * weights[factor][None, :, None]).sum(dim=1)
            )
        fused = self.poi_norm(base + torch.cat(fused_parts, dim=-1))
        self._cached_views = (collaborative, geographical, transition)
        self._cached_fused = fused
        return fused

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
        inputs = torch.cat(
            [
                candidate_with_padding[poi],
                self.category_embedding(category),
                user_e,
                self.hour_embedding(hour),
                self.weekday_embedding(weekday),
            ],
            dim=-1,
        )
        hidden = self.sequence_encoder(
            self.dropout(self.input_projection(inputs)), lengths
        )
        query = self.output_projection(hidden)
        return torch.einsum("bsd,vd->bsv", query, candidate) + self.poi_bias

    def compute_auxiliary_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        del batch
        if self._cached_views is None or self._cached_fused is None:
            return self.poi_bias.sum() * 0.0

        collaborative, geographical, transition = self._cached_views
        contrastive = (
            _sampled_symmetric_infonce(
                collaborative,
                geographical,
                self.temperature,
                self.max_contrastive_samples,
            )
            + _sampled_symmetric_infonce(
                collaborative,
                transition,
                self.temperature,
                self.max_contrastive_samples,
            )
            + _sampled_symmetric_infonce(
                geographical,
                transition,
                self.temperature,
                self.max_contrastive_samples,
            )
        ) / 3.0

        sample_count = min(
            self._cached_fused.size(0), self.max_contrastive_samples
        )
        factors = self._cached_fused[:sample_count].reshape(
            sample_count, self.num_factors, -1
        )
        factors = F.normalize(factors, dim=-1)
        gram = torch.einsum("nfd,ngd->fg", factors, factors)
        gram = gram / max(sample_count, 1)
        identity = torch.eye(
            self.num_factors, device=gram.device, dtype=gram.dtype
        )
        disentangle = ((gram - identity) ** 2).mean()
        return (
            self.contrastive_weight * contrastive
            + self.disentangle_weight * disentangle
        )
