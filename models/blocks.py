"""Shared building blocks for cluster / shared-specific Next-POI models."""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def init_embeddings(*embs: nn.Embedding, pad_idxs: Optional[Tuple[Tuple[nn.Embedding, int], ...]] = None) -> None:
    for emb in embs:
        nn.init.xavier_uniform_(emb.weight)
    if pad_idxs:
        with torch.no_grad():
            for emb, idx in pad_idxs:
                emb.weight[idx].zero_()


class CheckinFeatureEncoder(nn.Module):
    """Fuse POI / category / time / region / user into per-step inputs."""

    def __init__(
        self,
        num_users: int,
        num_pois: int,
        num_categories: int,
        num_regions: int,
        emb_dim: int = 64,
        ctx_dim: int = 32,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_pois = num_pois
        self.poi_pad = num_pois
        self.cat_pad = num_categories

        self.poi = nn.Embedding(num_pois + 1, emb_dim, padding_idx=self.poi_pad)
        self.cat = nn.Embedding(num_categories + 1, ctx_dim, padding_idx=self.cat_pad)
        self.user = nn.Embedding(num_users, ctx_dim)
        self.hour = nn.Embedding(24, ctx_dim)
        self.weekday = nn.Embedding(7, ctx_dim)
        self.region = nn.Embedding(num_regions, ctx_dim)
        self.delta_proj = nn.Linear(2, ctx_dim)
        self.dropout = nn.Dropout(dropout)

        self.out_dim = emb_dim + 5 * ctx_dim
        init_embeddings(
            self.poi,
            self.cat,
            self.user,
            self.hour,
            self.weekday,
            self.region,
            pad_idxs=((self.poi, self.poi_pad), (self.cat, self.cat_pad)),
        )
        nn.init.xavier_uniform_(self.delta_proj.weight)
        nn.init.zeros_(self.delta_proj.bias)

    def forward(self, batch: Dict[str, torch.Tensor], poi_region: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        poi = batch["poi"]
        safe = poi.clamp(max=self.num_pois - 1)
        region_ids = poi_region[safe]
        region_ids = torch.where(poi.eq(self.poi_pad), torch.zeros_like(region_ids), region_ids)

        poi_e = self.poi(poi)
        cat_e = self.cat(batch["category"])
        user_e = self.user(batch["user"])
        hour_e = self.hour(batch["hour"].clamp(0, 23))
        weekday_e = self.weekday(batch["weekday"].clamp(0, 6))
        region_e = self.region(region_ids)
        delta = torch.stack(
            [batch["delta_time_h"].clamp(min=0.0, max=168.0), batch["delta_distance_km"].clamp(min=0.0, max=50.0)],
            dim=-1,
        )
        delta_e = torch.tanh(self.delta_proj(delta))
        user_rep = user_e[:, None, :].expand(-1, poi.size(1), -1)
        x = torch.cat([poi_e, cat_e, hour_e, weekday_e, region_e, delta_e], dim=-1)
        x = self.dropout(x)
        return x, user_e, region_e


class CausalLSTMEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int = 1, dropout: float = 0.2) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            in_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu().clamp(min=1), batch_first=True, enforce_sorted=False
        )
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=x.size(1))
        return self.norm(out)


class SoftPatternCluster(nn.Module):
    """Assign each step to K shared trajectory-pattern prototypes."""

    def __init__(self, hidden_dim: int, n_pattern: int = 32) -> None:
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(n_pattern, hidden_dim) * 0.02)
        self.assign = nn.Linear(hidden_dim, n_pattern)
        nn.init.xavier_uniform_(self.assign.weight)
        nn.init.zeros_(self.assign.bias)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # h: (B,S,H) -> soft assignment (B,S,K), mixed pattern (B,S,H)
        logits = self.assign(h) / math.sqrt(h.size(-1))
        alpha = torch.softmax(logits, dim=-1)
        mixed = alpha @ self.prototypes
        return alpha, mixed


class RegionProjection(nn.Module):
    """Project a shared vector into a business-district (region) subspace."""

    def __init__(self, num_regions: int, hidden_dim: int, rank: int = 32) -> None:
        super().__init__()
        self.rank = rank
        self.U = nn.Embedding(num_regions, hidden_dim * rank)
        self.V = nn.Embedding(num_regions, rank * hidden_dim)
        nn.init.normal_(self.U.weight, std=0.02)
        nn.init.normal_(self.V.weight, std=0.02)

    def forward(self, shared: torch.Tensor, region_ids: torch.Tensor) -> torch.Tensor:
        # shared (B,S,H), region_ids (B,S)
        b, s, h = shared.shape
        r = self.rank
        u = self.U(region_ids).view(b, s, h, r)
        v = self.V(region_ids).view(b, s, r, h)
        # low-rank projection: shared @ U @ V
        mid = torch.einsum("bsh,bshr->bsr", shared, u)
        return torch.einsum("bsr,bsrh->bsh", mid, v)


class AddProjGate(nn.Module):
    """Mix additive and projective compositions of shared / specific."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        for m in self.gate:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, shared: torch.Tensor, specific: torch.Tensor) -> torch.Tensor:
        add = shared + specific
        proj = self.proj(shared * specific)
        g = torch.sigmoid(self.gate(torch.cat([shared, specific], dim=-1)))
        return g * add + (1.0 - g) * proj


class NextPoiScoreHead(nn.Module):
    """Full-candidate scores = preference + first-order transition + bias."""

    def __init__(self, num_pois: int, hidden_dim: int, emb_dim: int) -> None:
        super().__init__()
        self.num_pois = num_pois
        self.poi_pad = num_pois
        self.out = nn.Embedding(num_pois, hidden_dim)
        self.prev = nn.Embedding(num_pois + 1, emb_dim, padding_idx=self.poi_pad)
        self.nxt = nn.Embedding(num_pois, emb_dim)
        self.bias = nn.Parameter(torch.zeros(num_pois))
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.xavier_uniform_(self.prev.weight)
        nn.init.xavier_uniform_(self.nxt.weight)
        with torch.no_grad():
            self.prev.weight[self.poi_pad].zero_()
        nn.init.zeros_(self.bias)

    def forward(self, state: torch.Tensor, poi: torch.Tensor) -> torch.Tensor:
        pref = torch.einsum("bsd,vd->bsv", state, self.out.weight)
        trans = torch.einsum("bsd,vd->bsv", self.prev(poi), self.nxt.weight)
        return pref + trans + self.bias
