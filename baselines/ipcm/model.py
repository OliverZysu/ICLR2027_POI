"""iPCM for the shared closed-set Next-POI recommendation protocol."""

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


def _safe_cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    valid = target.ge(0)
    if not bool(valid.any()):
        return logits.sum() * 0.0
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1), ignore_index=-1)


class IPCM(nn.Module):
    """Individualised preference-cluster model.

    User, region, and temporal preference prototypes are constructed from the
    training split. A user-conditioned gate combines these prototypes with the
    current POI, followed by causal Transformer trajectory modelling and
    training-only probability adjustment priors.
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
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.2,
        region_loss_weight: float = 0.2,
        prior_weight: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_pois = int(num_pois)
        self.poi_padding_idx = self.num_pois
        self.category_padding_idx = int(num_categories)
        self.region_loss_weight = float(region_loss_weight)

        self.register_buffer("poi_region", context["poi_region"].long())
        self.register_buffer("collab_up", context["collab_up"])
        self.register_buffer("region_rp", context["region_rp"])
        self.register_buffer("user_poi_prior", context["user_poi_prior"])
        self.register_buffer("time_poi_prior", context["time_poi_prior"])
        self.register_buffer("region_poi_prior", context["region_poi_prior"])
        self.num_regions = int(self.poi_region.max().item()) + 1

        self.poi_embedding = nn.Embedding(
            num_pois + 1, embedding_dim, padding_idx=self.poi_padding_idx
        )
        self.user_embedding = nn.Embedding(num_users, context_dim)
        self.category_embedding = nn.Embedding(
            num_categories + 1, context_dim, padding_idx=self.category_padding_idx
        )
        self.hour_embedding = nn.Embedding(24, context_dim)
        self.weekday_embedding = nn.Embedding(7, context_dim)
        self.region_embedding = nn.Embedding(self.num_regions, context_dim)

        self.user_proto_projection = nn.Linear(embedding_dim, hidden_dim)
        self.region_proto_projection = nn.Linear(embedding_dim, hidden_dim)
        self.poi_projection = nn.Linear(embedding_dim, hidden_dim)
        self.time_projection = nn.Linear(2 * context_dim, hidden_dim)
        self.gate = nn.Linear(hidden_dim + 2 * context_dim, 4)
        self.input_projection = nn.Sequential(
            nn.Linear(hidden_dim + 3 * context_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        self.transformer = CausalTransformer(
            hidden_dim, num_heads, num_layers, 4 * hidden_dim, dropout
        )
        self.output_embedding = nn.Embedding(num_pois, hidden_dim)
        self.candidate_cluster_projection = nn.Linear(
            embedding_dim, hidden_dim, bias=False
        )
        self.region_head = nn.Linear(hidden_dim, self.num_regions)
        self.poi_bias = nn.Parameter(torch.zeros(num_pois))

        initial_prior = math.log(max(float(prior_weight), 1e-4))
        self.raw_prior_weights = nn.Parameter(
            torch.full((3,), float(initial_prior))
        )
        self.dropout = nn.Dropout(dropout)
        self._cached_region_logits = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        embeddings = (
            self.poi_embedding,
            self.user_embedding,
            self.category_embedding,
            self.hour_embedding,
            self.weekday_embedding,
            self.region_embedding,
            self.output_embedding,
        )
        for embedding in embeddings:
            nn.init.xavier_uniform_(embedding.weight)
        nn.init.zeros_(self.poi_bias)
        with torch.no_grad():
            self.poi_embedding.weight[self.poi_padding_idx].zero_()
            self.category_embedding.weight[self.category_padding_idx].zero_()

    def _prototypes(self):
        base = self.poi_embedding.weight[: self.num_pois]
        user_prototype = torch.sparse.mm(self.collab_up, base)
        region_prototype = torch.sparse.mm(self.region_rp, base)
        return base, user_prototype, region_prototype

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        poi = batch["poi"]
        category = batch["category"]
        user = batch["user"]
        hour = batch["hour"]
        weekday = batch["weekday"]
        lengths = batch["lengths"]

        base, user_prototype, region_prototype = self._prototypes()
        base_with_padding = torch.cat(
            [base, base.new_zeros(1, base.size(-1))], dim=0
        )
        safe_poi = poi.clamp_max(self.num_pois - 1)
        region = self.poi_region[safe_poi]

        current = self.poi_projection(base_with_padding[poi])
        user_proto = self.user_proto_projection(user_prototype[user])
        user_proto = user_proto[:, None, :].expand(-1, poi.size(1), -1)
        region_proto = self.region_proto_projection(region_prototype[region])
        temporal_proto = self.time_projection(
            torch.cat(
                [self.hour_embedding(hour), self.weekday_embedding(weekday)],
                dim=-1,
            )
        )
        user_e = self.user_embedding(user)[:, None, :].expand(-1, poi.size(1), -1)
        region_e = self.region_embedding(region)

        gate = torch.softmax(
            self.gate(torch.cat([current, user_e, region_e], dim=-1)), dim=-1
        )
        stacked = torch.stack(
            [current, user_proto, region_proto, temporal_proto], dim=2
        )
        cluster = (stacked * gate[:, :, :, None]).sum(dim=2)

        inputs = self.input_projection(
            torch.cat(
                [cluster, self.category_embedding(category), user_e, region_e],
                dim=-1,
            )
        )
        hidden = self.dropout(self.transformer(inputs, lengths))
        self._cached_region_logits = self.region_head(hidden)

        candidate = self.output_embedding.weight + self.candidate_cluster_projection(
            region_prototype[self.poi_region]
        )
        logits = torch.einsum("bsd,vd->bsv", hidden, candidate) + self.poi_bias

        prior_weights = F.softplus(self.raw_prior_weights)
        user_prior = self.user_poi_prior[user].to(dtype=logits.dtype)
        user_prior = user_prior[:, None, :]
        time_index = (weekday * 24 + hour).clamp(0, 167)
        time_prior = self.time_poi_prior[time_index].to(dtype=logits.dtype)
        region_prior = self.region_poi_prior[region].to(dtype=logits.dtype)
        return (
            logits
            + prior_weights[0] * user_prior
            + prior_weights[1] * time_prior
            + prior_weights[2] * region_prior
        )

    def compute_auxiliary_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self._cached_region_logits is None:
            return self.poi_bias.sum() * 0.0
        target = batch["target"]
        safe_target = target.clamp_min(0)
        target_region = self.poi_region[safe_target].masked_fill(target.lt(0), -1)
        loss = _safe_cross_entropy(self._cached_region_logits, target_region)
        return self.region_loss_weight * loss
