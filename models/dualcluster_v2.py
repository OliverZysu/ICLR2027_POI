"""Grounded shared-pattern / personal-residual Next-POI model family.

This is new research code, NOT a claimed SOTA result. Each forward call is
causal, uses observed check-ins only, and returns full-vocabulary log scores
(B,S,V). Ground-truth next POI/category/region are used ONLY in auxiliary_loss.
The existing DualClusterNet remains unmodified as the `original` control.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict

import torch
from torch import nn
import torch.nn.functional as F

from .blocks import CheckinFeatureEncoder, CausalLSTMEncoder, NextPoiScoreHead


@dataclass(frozen=True)
class Variant:
    pattern: bool = True
    projection: bool = True
    hierarchy: bool = True
    auxiliary: bool = True
    repeat: bool = True
    balance: bool = True
    shuffled_space: bool = False


VARIANTS = {
    "backbone": Variant(False, False, False, False, False, False),
    "grounded": Variant(True, False, False, False, False, True),
    "projected": Variant(True, True, False, False, False, True),
    "hierarchical": Variant(True, True, True, True, False, True),
    "repeat_control": Variant(False, False, False, False, True, False),
    "full": Variant(),
    "no_pattern": Variant(pattern=False, projection=False, balance=False),
    "no_projection": Variant(projection=False),
    "no_hierarchy": Variant(hierarchy=False),
    "no_auxiliary": Variant(auxiliary=False),
    "no_repeat": Variant(repeat=False),
    "no_balance": Variant(balance=False),
    "shuffled_space": Variant(shuffled_space=True),
}


def get_variant(name: str) -> Variant:
    if name not in VARIANTS:
        raise ValueError(f"Unknown variant {name!r}; choose {list(VARIANTS)}")
    return VARIANTS[name]


def valid_steps(batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    # Do not infer the input mask from future targets.
    return torch.arange(batch["poi"].size(1), device=batch["poi"].device)[None] < batch["lengths"][:, None]


class DualClusterV2(nn.Module):
    def __init__(
        self, num_users: int, num_pois: int, num_categories: int,
        poi_region: torch.Tensor, region_membership: torch.Tensor,
        poi_category: torch.Tensor, variant: str = "full",
        embedding_dim: int = 64, context_dim: int = 32,
        hidden_dim: int = 128, n_pattern: int = 32, proj_rank: int = 16,
        dropout: float = 0.2, temperature: float = 0.2,
        category_weight: float = 0.1, region_weight: float = 0.1,
        balance_weight: float = 0.01, confidence_weight: float = 0.001,
        decorrelation_weight: float = 0.001, repeat_weight: float = 0.05,
    ) -> None:
        super().__init__()
        if min(num_users, num_pois, num_categories, n_pattern, proj_rank) < 1:
            raise ValueError("All model cardinalities must be positive")
        if temperature <= 0 or not 0 <= dropout < 1:
            raise ValueError("temperature > 0 and 0 <= dropout < 1 are required")
        self.variant_name = variant
        self.variant = get_variant(variant)
        self.num_pois = num_pois
        self.num_categories = num_categories
        self.n_pattern = n_pattern
        self.hidden_dim = hidden_dim
        self.temperature = float(temperature)
        self.loss_weights = dict(category=category_weight, region=region_weight,
                                 balance=balance_weight, confidence=confidence_weight,
                                 decorrelation=decorrelation_weight, repeat=repeat_weight)
        membership = region_membership.float()
        if membership.ndim != 2 or membership.shape[0] != num_pois:
            raise ValueError("region_membership must be (num_pois, num_regions)")
        if not torch.isfinite(membership).all() or (membership < 0).any():
            raise ValueError("Invalid region probabilities")
        membership = membership / membership.sum(-1, keepdim=True).clamp_min(1e-12)
        if not torch.allclose(membership.sum(-1), torch.ones(num_pois), atol=1e-5):
            raise ValueError("Every candidate needs nonzero geographic membership")
        if self.variant.shuffled_space:
            # Negative control: deliberately break POI-geography correspondence.
            # Fixed independently of training seed; never used by the main model.
            perm = torch.randperm(num_pois, generator=torch.Generator().manual_seed(991))
            membership = membership[perm]
            poi_region = poi_region[perm]
        self.register_buffer("poi_region", poi_region.long())
        self.register_buffer("membership", membership)
        self.register_buffer("poi_category", poi_category.long())
        self.num_regions = membership.shape[1]
        self.feats = CheckinFeatureEncoder(num_users, num_pois, num_categories,
                                          self.num_regions, embedding_dim, context_dim, dropout)
        self.encoder = CausalLSTMEncoder(self.feats.out_dim + context_dim,
                                        hidden_dim, dropout=dropout)
        self.score = NextPoiScoreHead(num_pois, hidden_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        if self.variant.pattern:
            # No POI ID, user ID, absolute coordinates, or region IDs enter this
            # semantic branch. Shared embedding tables are not shared with user inputs.
            self.semantic_category = nn.Embedding(num_categories + 1, context_dim,
                                                  padding_idx=num_categories)
            self.semantic_hour = nn.Embedding(24, context_dim)
            self.semantic_weekday = nn.Embedding(7, context_dim)
            self.semantic_delta = nn.Linear(2, context_dim)
            self.semantic_encoder = CausalLSTMEncoder(4 * context_dim, hidden_dim, dropout=dropout)
            self.prototypes = nn.Parameter(torch.randn(n_pattern, hidden_dim) / math.sqrt(hidden_dim))
            self.semantic_query = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.region_codes = nn.Embedding(self.num_regions, hidden_dim)
            self.spatial_to_shared = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.shared_norm = nn.LayerNorm(hidden_dim)
            self.personal_residual = nn.Linear(hidden_dim + context_dim, hidden_dim)
            self.state_norm = nn.LayerNorm(hidden_dim)
            # Start as a small residual upgrade to a competitive backbone.
            self.shared_scale = nn.Parameter(torch.tensor(0.1))
            self.personal_scale = nn.Parameter(torch.tensor(0.1))
            if self.variant.projection:
                self.proj_down = nn.Linear(hidden_dim, proj_rank, bias=False)
                self.proj_up = nn.Linear(proj_rank, hidden_dim, bias=False)
                nn.init.normal_(self.proj_up.weight, std=0.01)
                self.proj_condition = nn.Linear(hidden_dim + context_dim, proj_rank)
                self.composition_gate = nn.Linear(2 * hidden_dim + context_dim, 1)
        if self.variant.hierarchy or self.variant.auxiliary:
            self.region_head = nn.Linear(hidden_dim + context_dim + self.num_regions, self.num_regions)
            if self.variant.pattern:
                self.pattern_region = nn.Parameter(torch.zeros(n_pattern, self.num_regions))
            if self.variant.hierarchy:
                self.hierarchy_gate = nn.Linear(hidden_dim, 1)
                nn.init.zeros_(self.hierarchy_gate.weight)
                nn.init.constant_(self.hierarchy_gate.bias, -2.0)
        if self.variant.auxiliary:
            self.category_head = nn.Linear(hidden_dim, num_categories)
        if self.variant.repeat:
            self.copy_query = nn.Linear(hidden_dim, embedding_dim, bias=False)
            self.copy_key = nn.Linear(embedding_dim, embedding_dim, bias=False)
            self.copy_decay = nn.Parameter(torch.tensor(-1.0))
            self.repeat_gate = nn.Linear(hidden_dim + context_dim, 1)
            nn.init.zeros_(self.repeat_gate.weight)
            nn.init.constant_(self.repeat_gate.bias, -2.0)
        self._cache: Dict[str, torch.Tensor] = {}
        self.auxiliary_scale = 1.0

    def _semantic(self, batch: Dict[str, torch.Tensor]):
        delta = torch.stack([torch.log1p(batch["delta_time_h"].clamp(0, 168)),
                             torch.log1p(batch["delta_distance_km"].clamp(0, 50))], -1)
        sx = torch.cat([self.semantic_category(batch["category"]),
                        self.semantic_hour(batch["hour"].clamp(0, 23)),
                        self.semantic_weekday(batch["weekday"].clamp(0, 6)),
                        torch.tanh(self.semantic_delta(delta))], -1)
        semantic = self.semantic_encoder(self.dropout(sx), batch["lengths"])
        query = F.normalize(self.semantic_query(semantic), dim=-1)
        prototypes = F.normalize(self.prototypes, dim=-1)
        assignment = (query @ prototypes.T / self.temperature).softmax(-1)
        # Restore representation scale after cosine-normalized assignment.
        mixed = (assignment @ prototypes) * math.sqrt(self.hidden_dim)
        return assignment, mixed

    @staticmethod
    def hierarchical_distribution(logp: torch.Tensor, region_logits: torch.Tensor,
                                  membership: torch.Tensor) -> torch.Tensor:
        """Exact soft-region mixture, without a hard region candidate filter.

        P(v|r,H) ∝ P_base(v|H) * membership(v,r).
        Sum_r P(r|H) P(v|r,H) is normalized over ALL candidates.
        Uses O(B*S*(V+R)+V*R) memory, not O(B*S*V*R).
        """
        p = logp.float().exp()
        pi = region_logits.float().softmax(-1)
        z = (p @ membership.float()).clamp_min(1e-20)
        ph = p * ((pi / z) @ membership.float().T)
        ph = ph.clamp_min(1e-30)
        return (ph / ph.sum(-1, keepdim=True).clamp_min(1e-30)).log()

    def _copy_distribution(self, state, batch):
        b, s, _ = state.shape
        q = self.copy_query(state)
        k = self.copy_key(self.feats.poi(batch["poi"]))
        logits = q @ k.transpose(-1, -2) / math.sqrt(q.size(-1))
        idx = torch.arange(s, device=state.device)
        lag = (idx[:, None] - idx[None, :]).clamp_min(0).float()
        logits = logits - F.softplus(self.copy_decay) * torch.log1p(lag)[None]
        visible = (idx[None, :] <= idx[:, None])[None] & valid_steps(batch)[:, None, :]
        attention = logits.float().masked_fill(~visible, float("-inf")).softmax(-1)
        # Padded query positions still have a valid preceding key, but are never
        # supervised/evaluated. All sequences are checked nonempty by the loader.
        copy = state.new_zeros(b, s, self.num_pois, dtype=torch.float32)
        ids = batch["poi"].clamp_max(self.num_pois - 1)[:, None, :].expand(-1, s, -1)
        copy = copy.scatter_add(-1, ids, attention)
        return copy, attention

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        self._cache = {}
        mask = valid_steps(batch)
        features = dict(batch)
        # Saturation-resistant observed interval transform, shared by all v2 controls.
        features["delta_time_h"] = torch.log1p(batch["delta_time_h"].clamp(0, 168))
        features["delta_distance_km"] = torch.log1p(batch["delta_distance_km"].clamp(0, 50))
        x, user, _ = self.feats(features, self.poi_region)
        u = user[:, None].expand(-1, x.size(1), -1)
        h = self.encoder(torch.cat([x, u], -1), batch["lengths"])
        observed_regions = self.membership[batch["poi"].clamp_max(self.num_pois - 1)]
        observed_regions = observed_regions * mask[..., None]
        counts = mask.cumsum(1).clamp_min(1)[..., None]
        activity = observed_regions.cumsum(1) / counts
        state, pattern = h, h
        if self.variant.pattern:
            alpha, pattern = self._semantic(batch)
            source = activity @ self.region_codes.weight
            shared = self.shared_norm(pattern + self.spatial_to_shared(source))
            personal = torch.tanh(self.personal_residual(torch.cat([h, u], -1)))
            skip = h + torch.tanh(self.personal_scale) * personal
            additive = skip + torch.tanh(self.shared_scale) * shared
            if self.variant.projection:
                coeff = torch.tanh(self.proj_condition(torch.cat([source, u], -1)))
                projected = shared + self.proj_up(self.proj_down(shared) * coeff)
                projective = skip + torch.tanh(self.shared_scale) * projected
                gate = torch.sigmoid(self.composition_gate(torch.cat([shared, personal, u], -1)))
                state = self.state_norm(gate * additive + (1 - gate) * projective)
                self._cache["composition_gate"] = gate
            else:
                state = self.state_norm(additive)
            self._cache.update(alpha=alpha, shared=shared, personal=personal)
        base_scores = self.score(self.dropout(state), batch["poi"])
        logp = F.log_softmax(base_scores.float(), -1)
        if self.variant.hierarchy or self.variant.auxiliary:
            region_logits = self.region_head(torch.cat([state, u, activity], -1))
            if self.variant.pattern:
                region_logits = region_logits + self._cache["alpha"] @ self.pattern_region
            self._cache["region_logits"] = region_logits
            if self.variant.hierarchy:
                hierarchical = self.hierarchical_distribution(logp, region_logits, self.membership)
                gate_logit = self.hierarchy_gate(state).float()
                logp = torch.logaddexp(F.logsigmoid(-gate_logit) + logp,
                                      F.logsigmoid(gate_logit) + hierarchical)
                self._cache["hierarchy_gate"] = gate_logit.sigmoid()
        if self.variant.auxiliary:
            # Force the abstract pattern to carry category semantics.
            self._cache["category_logits"] = self.category_head(pattern)
        if self.variant.repeat:
            copy, attention = self._copy_distribution(state, batch)
            gate_logit = self.repeat_gate(torch.cat([state, u], -1)).float()
            # -inf for unseen candidates is valid inside this expert: the other
            # expert retains strictly positive support over the full vocabulary.
            copy_logp = copy.clamp_min(1e-30).log().masked_fill(copy <= 0, float("-inf"))
            logp = torch.logaddexp(F.logsigmoid(-gate_logit) + logp,
                                  F.logsigmoid(gate_logit) + copy_logp)
            self._cache.update(repeat_logit=gate_logit.squeeze(-1), copy_attention=attention)
        self._cache["mask"] = mask
        return logp

    def auxiliary_loss(self, batch: Dict[str, torch.Tensor]):
        """Targets are deliberately accessed only here, never in forward()."""
        c = self._cache
        mask = c["mask"] & batch["target"].ge(0)
        target = batch["target"].clamp_min(0)
        zero = self.score.bias.sum() * 0.0
        terms: Dict[str, torch.Tensor] = {}
        if self.variant.auxiliary:
            terms["category"] = F.cross_entropy(c["category_logits"][mask], self.poi_category[target[mask]])
            # Soft geographic labels; do not pretend KMeans discovers named business districts.
            labels = self.membership[target[mask]]
            terms["region"] = -(labels * c["region_logits"][mask].log_softmax(-1)).sum(-1).mean()
        if self.variant.pattern and self.variant.balance:
            # One assignment per trajectory limits domination by long sequences.
            rows = torch.arange(mask.size(0), device=mask.device)
            q = c["alpha"][rows, batch["lengths"] - 1].clamp_min(1e-8)
            marginal = q.mean(0)
            terms["balance"] = (marginal * (marginal.log() + math.log(self.n_pattern))).sum()
            terms["confidence"] = -(q * q.log()).sum(-1).mean()
        if self.variant.pattern:
            # Distinct from assignment balance; no_balance removes only the
            # balance/confidence clustering objectives, not branch decorrelation.
            sh = F.normalize(c["shared"][mask], dim=-1)
            pe = F.normalize(c["personal"][mask], dim=-1)
            terms["decorrelation"] = (sh * pe).sum(-1).square().mean()
        if self.variant.repeat:
            s = target.size(1)
            visible = torch.arange(s, device=target.device)[None, :] <= torch.arange(s, device=target.device)[:, None]
            previous_match = (target[..., None] == batch["poi"][:, None, :]) & visible[None]
            previous_match = previous_match & c["mask"][:, None, :]
            is_repeat = previous_match.any(-1).float()
            terms["repeat"] = F.binary_cross_entropy_with_logits(c["repeat_logit"][mask], is_repeat[mask])
        weighted = sum((self.loss_weights[k] * v for k, v in terms.items()), zero)
        return weighted * self.auxiliary_scale, {k: float(v.detach()) for k, v in terms.items()}

    @torch.no_grad()
    def diagnostics(self) -> Dict[str, float]:
        c, out = self._cache, {}
        if not c:
            return out
        m = c["mask"]
        if "alpha" in c:
            q = c["alpha"][m].float().clamp_min(1e-8)
            marginal = q.mean(0)
            out["pattern_effective_clusters"] = float(torch.exp(-(marginal * marginal.log()).sum()))
            out["pattern_assignment_entropy"] = float(-(q * q.log()).sum(-1).mean())
            out["pattern_max_mass"] = float(marginal.max())
            out["shared_personal_cos2"] = float(F.cosine_similarity(c["shared"][m], c["personal"][m]).square().mean())
            out["shared_scale"] = float(torch.tanh(self.shared_scale))
            out["personal_scale"] = float(torch.tanh(self.personal_scale))
        for name in ("composition_gate", "hierarchy_gate"):
            if name in c:
                out[name] = float(c[name][m].mean())
        if "repeat_logit" in c:
            out["repeat_gate"] = float(c["repeat_logit"][m].sigmoid().mean())
        return out
