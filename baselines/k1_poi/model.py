"""K1-POI for the shared closed-set Next-POI recommendation protocol."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn


def _safe_cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    valid = target.ge(0)
    if not bool(valid.any()):
        return logits.sum() * 0.0
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1), ignore_index=-1)


class K1POI(nn.Module):
    """Last-check-in candidate recall followed by neural reranking.

    The transition, geographic, personal, temporal, and popularity recall
    channels are estimated from the training split. Scores are still returned
    for every candidate POI so evaluation remains identical to GETNext/STAN.
    """

    def __init__(
        self,
        num_users: int,
        num_pois: int,
        num_categories: int,
        context: Dict[str, torch.Tensor],
        embedding_dim: int = 96,
        context_dim: int = 24,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        recall_loss_weight: float = 0.25,
    ) -> None:
        super().__init__()
        self.num_pois = int(num_pois)
        self.poi_padding_idx = self.num_pois
        self.category_padding_idx = int(num_categories)
        self.recall_loss_weight = float(recall_loss_weight)

        for name in (
            "transition_top_idx",
            "transition_top_score",
            "geo_top_idx",
            "geo_top_score",
            "user_poi_prior",
            "time_poi_prior",
            "global_poi_prior",
            "poi_region",
        ):
            self.register_buffer(name, context[name])
        self.num_regions = int(self.poi_region.max().item()) + 1

        self.poi_embedding = nn.Embedding(
            num_pois + 1, embedding_dim, padding_idx=self.poi_padding_idx
        )
        self.candidate_embedding = nn.Embedding(num_pois, hidden_dim)
        self.user_embedding = nn.Embedding(num_users, context_dim)
        self.category_embedding = nn.Embedding(
            num_categories + 1, context_dim, padding_idx=self.category_padding_idx
        )
        self.hour_embedding = nn.Embedding(24, context_dim)
        self.weekday_embedding = nn.Embedding(7, context_dim)
        self.region_embedding = nn.Embedding(self.num_regions, context_dim)

        self.query = nn.Sequential(
            nn.Linear(embedding_dim + 5 * context_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.candidate_region_projection = nn.Linear(
            context_dim, hidden_dim, bias=False
        )
        self.candidate_bias = nn.Parameter(torch.zeros(num_pois))
        self.raw_recall_weights = nn.Parameter(torch.zeros(5))
        self.recall_temperature = nn.Parameter(torch.tensor(1.0))
        self._cached_recall = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        embeddings = (
            self.poi_embedding,
            self.candidate_embedding,
            self.user_embedding,
            self.category_embedding,
            self.hour_embedding,
            self.weekday_embedding,
            self.region_embedding,
        )
        for embedding in embeddings:
            nn.init.xavier_uniform_(embedding.weight)
        nn.init.zeros_(self.candidate_bias)
        with torch.no_grad():
            self.poi_embedding.weight[self.poi_padding_idx].zero_()
            self.category_embedding.weight[self.category_padding_idx].zero_()

    def _scatter_channel(
        self,
        output: torch.Tensor,
        current_poi: torch.Tensor,
        indices: torch.Tensor,
        scores: torch.Tensor,
        weight: torch.Tensor,
    ) -> None:
        safe_poi = current_poi.clamp_max(self.num_pois - 1)
        selected_indices = indices[safe_poi]
        selected_scores = scores[safe_poi].to(dtype=output.dtype)
        output.scatter_add_(
            dim=-1,
            index=selected_indices,
            src=weight * selected_scores,
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        poi = batch["poi"]
        category = batch["category"]
        user = batch["user"]
        hour = batch["hour"]
        weekday = batch["weekday"]

        safe_poi = poi.clamp_max(self.num_pois - 1)
        region = self.poi_region[safe_poi]
        batch_size, seq_len = poi.shape
        user_e = self.user_embedding(user)[:, None, :].expand(-1, seq_len, -1)
        query = self.query(
            torch.cat(
                [
                    self.poi_embedding(poi),
                    self.category_embedding(category),
                    user_e,
                    self.hour_embedding(hour),
                    self.weekday_embedding(weekday),
                    self.region_embedding(region),
                ],
                dim=-1,
            )
        )

        candidate = self.candidate_embedding.weight + self.candidate_region_projection(
            self.region_embedding(self.poi_region)
        )
        rerank = (
            torch.einsum("bsd,vd->bsv", query, candidate)
            + self.candidate_bias
        )

        weights = F.softplus(self.raw_recall_weights)
        recall = torch.zeros_like(rerank)
        self._scatter_channel(
            recall,
            poi,
            self.transition_top_idx,
            self.transition_top_score,
            weights[0],
        )
        self._scatter_channel(
            recall,
            poi,
            self.geo_top_idx,
            self.geo_top_score,
            weights[1],
        )
        recall = recall + weights[2] * self.user_poi_prior[user].to(
            dtype=rerank.dtype
        )[:, None, :]
        time_index = (weekday * 24 + hour).clamp(0, 167)
        recall = recall + weights[3] * self.time_poi_prior[time_index].to(
            dtype=rerank.dtype
        )
        recall = recall + weights[4] * self.global_poi_prior.to(
            dtype=rerank.dtype
        )[None, None, :]
        recall = recall / self.recall_temperature.abs().clamp_min(0.1)
        self._cached_recall = recall
        return rerank + recall

    def compute_auxiliary_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self._cached_recall is None:
            return self.candidate_bias.sum() * 0.0
        return self.recall_loss_weight * _safe_cross_entropy(
            self._cached_recall, batch["target"]
        )
