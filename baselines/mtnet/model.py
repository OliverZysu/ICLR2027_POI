"""MTNet for the shared closed-set Next-POI recommendation protocol.

This implementation keeps the method-specific mobility-tree interactions and
multi-task supervision while conforming to the batch dictionary produced by
``utils.baseline_train`` in the host project.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn


class TreeInteractionCell(nn.Module):
    """Gated child-to-parent interaction used at one mobility-tree level."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.iou = nn.Linear(2 * dim, 3 * dim)
        self.forget = nn.Linear(2 * dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, child: torch.Tensor, parent_context: torch.Tensor) -> torch.Tensor:
        joined = torch.cat([child, parent_context], dim=-1)
        input_gate, output_gate, update = self.iou(joined).chunk(3, dim=-1)
        state = (
            torch.sigmoid(self.forget(joined)) * parent_context
            + torch.sigmoid(input_gate) * torch.tanh(update)
        )
        return self.norm(torch.sigmoid(output_gate) * torch.tanh(state) + child)


def _masked_prefix_mean(x: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
    weights = allowed.to(dtype=x.dtype)
    return torch.bmm(weights, x) / weights.sum(dim=-1, keepdim=True).clamp_min(1.0)


def _safe_cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    valid = target.ge(0)
    if not bool(valid.any()):
        return logits.sum() * 0.0
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1), ignore_index=-1)


class MTNet(nn.Module):
    """Multi-granularity mobility-tree network.

    The hierarchy is check-in -> hour -> 4-hour period -> relative day -> root.
    Relative-day indices are reconstructed from observed inter-check-in gaps;
    no target-side time or coordinate is used.
    """

    def __init__(
        self,
        num_users: int,
        num_pois: int,
        num_categories: int,
        poi_region: torch.Tensor,
        poi_category: torch.Tensor,
        embedding_dim: int = 64,
        context_dim: int = 24,
        hidden_dim: int = 128,
        dropout: float = 0.3,
        category_loss_weight: float = 0.3,
        region_loss_weight: float = 0.3,
    ) -> None:
        super().__init__()
        self.num_pois = int(num_pois)
        self.poi_padding_idx = self.num_pois
        self.category_padding_idx = int(num_categories)
        self.category_loss_weight = float(category_loss_weight)
        self.region_loss_weight = float(region_loss_weight)

        self.register_buffer("poi_region", poi_region.long())
        self.register_buffer("poi_category", poi_category.long())
        self.num_regions = int(self.poi_region.max().item()) + 1

        self.poi_embedding = nn.Embedding(
            num_pois + 1, embedding_dim, padding_idx=self.poi_padding_idx
        )
        self.user_embedding = nn.Embedding(num_users, context_dim)
        self.category_embedding = nn.Embedding(
            num_categories + 1, context_dim, padding_idx=self.category_padding_idx
        )
        self.region_embedding = nn.Embedding(self.num_regions, context_dim)
        self.hour_embedding = nn.Embedding(24, context_dim)
        self.weekday_embedding = nn.Embedding(7, context_dim)

        self.leaf_projection = nn.Sequential(
            nn.Linear(embedding_dim + 5 * context_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        self.hour_cell = TreeInteractionCell(hidden_dim)
        self.period_cell = TreeInteractionCell(hidden_dim)
        self.day_cell = TreeInteractionCell(hidden_dim)
        self.root_cell = TreeInteractionCell(hidden_dim)

        self.output_embedding = nn.Embedding(num_pois, hidden_dim)
        self.poi_bias = nn.Parameter(torch.zeros(num_pois))
        self.category_head = nn.Linear(hidden_dim, num_categories)
        self.region_head = nn.Linear(hidden_dim, self.num_regions)
        self.task_log_vars = nn.Parameter(torch.zeros(2))
        self.dropout = nn.Dropout(dropout)

        self._cached_category_logits = None
        self._cached_region_logits = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        embeddings = (
            self.poi_embedding,
            self.user_embedding,
            self.category_embedding,
            self.region_embedding,
            self.hour_embedding,
            self.weekday_embedding,
            self.output_embedding,
        )
        for embedding in embeddings:
            nn.init.xavier_uniform_(embedding.weight)
        nn.init.zeros_(self.poi_bias)
        with torch.no_grad():
            self.poi_embedding.weight[self.poi_padding_idx].zero_()
            self.category_embedding.weight[self.category_padding_idx].zero_()

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        poi = batch["poi"]
        category = batch["category"]
        user = batch["user"]
        hour = batch["hour"]
        weekday = batch["weekday"]
        delta_time_h = batch["delta_time_h"]
        lengths = batch["lengths"]

        safe_poi = poi.clamp_max(self.num_pois - 1)
        region = self.poi_region[safe_poi]
        user_e = self.user_embedding(user)[:, None, :].expand(-1, poi.size(1), -1)

        leaf = self.leaf_projection(
            torch.cat(
                [
                    self.poi_embedding(poi),
                    self.category_embedding(category),
                    user_e,
                    self.region_embedding(region),
                    self.hour_embedding(hour),
                    self.weekday_embedding(weekday),
                ],
                dim=-1,
            )
        )

        # Only observed input-side gaps are used to reconstruct relative days.
        relative_day = torch.floor(torch.cumsum(delta_time_h, dim=1) / 24.0).long()
        seq_len = leaf.size(1)
        position = torch.arange(seq_len, device=leaf.device)
        causal = position[None, :, None].ge(position[None, None, :])
        valid_keys = position[None, None, :].lt(lengths[:, None, None])
        valid_queries = position[None, :, None].lt(lengths[:, None, None])
        base_mask = causal & valid_keys & valid_queries

        same_day = relative_day[:, :, None].eq(relative_day[:, None, :])
        same_hour = same_day & hour[:, :, None].eq(hour[:, None, :])
        same_period = same_day & (hour[:, :, None] // 4).eq(hour[:, None, :] // 4)

        hidden = self.hour_cell(leaf, _masked_prefix_mean(leaf, base_mask & same_hour))
        hidden = self.period_cell(
            hidden, _masked_prefix_mean(hidden, base_mask & same_period)
        )
        hidden = self.day_cell(hidden, _masked_prefix_mean(hidden, base_mask & same_day))
        hidden = self.root_cell(hidden, _masked_prefix_mean(hidden, base_mask))
        hidden = self.dropout(hidden)

        self._cached_category_logits = self.category_head(hidden)
        self._cached_region_logits = self.region_head(hidden)
        return torch.einsum("bsd,vd->bsv", hidden, self.output_embedding.weight) + self.poi_bias

    def compute_auxiliary_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self._cached_category_logits is None or self._cached_region_logits is None:
            return self.poi_bias.sum() * 0.0

        target = batch["target"]
        safe_target = target.clamp_min(0)
        target_category = self.poi_category[safe_target].masked_fill(target.lt(0), -1)
        target_region = self.poi_region[safe_target].masked_fill(target.lt(0), -1)

        category_loss = _safe_cross_entropy(self._cached_category_logits, target_category)
        region_loss = _safe_cross_entropy(self._cached_region_logits, target_region)

        weighted_category = 0.5 * torch.exp(-self.task_log_vars[0]) * category_loss
        weighted_category = weighted_category + 0.5 * self.task_log_vars[0]
        weighted_region = 0.5 * torch.exp(-self.task_log_vars[1]) * region_loss
        weighted_region = weighted_region + 0.5 * self.task_log_vars[1]
        return (
            self.category_loss_weight * weighted_category
            + self.region_loss_weight * weighted_region
        )
