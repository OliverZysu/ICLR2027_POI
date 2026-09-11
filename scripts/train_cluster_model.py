#!/usr/bin/env python3
"""Train cluster / shared-specific designed models under the baseline protocol.

Models: PCPNet, DualClusterNet, ASPMix
Protocol: same as FPMC/PLSPL — full-candidate (B,S,V), stepwise CE/BPR.

Usage:
  CUDA_VISIBLE_DEVICES=0 python3 -u scripts/train_cluster_model.py --model pcpnet --city NYC
  CUDA_VISIBLE_DEVICES=1 python3 -u scripts/train_cluster_model.py --model dualcluster --city TKY
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.aspmix import ASPMix
from models.dualcluster import DualClusterNet
from models.pcpnet import PCPNet
from utils.baseline_train import add_common_args, build_mappings, train_baseline
from utils.data import load_processed_splits

MODEL_REGISTRY = {
    "pcpnet": ("PCPNet", PCPNet),
    "dualcluster": ("DualClusterNet", DualClusterNet),
    "aspmix": ("ASPMix", ASPMix),
}


def _mapping_signature(mappings) -> str:
    return "%d-%d-%d" % (mappings.num_users, mappings.num_pois, mappings.num_categories)


def build_regions(train_df: pd.DataFrame, mappings, num_regions: int, seed: int) -> torch.Tensor:
    coords = mappings.poi_coords.astype(np.float64)
    train_pois = sorted(
        {mappings.poi_id2idx[str(p)] for p in train_df["POI_id"].astype(str) if str(p) in mappings.poi_id2idx}
    )
    train_indices = np.asarray(train_pois, dtype=np.int64)
    if len(train_indices) == 0:
        return torch.zeros(mappings.num_pois, dtype=torch.long)
    train_coordinates = coords[train_indices]
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(train_coordinates)
    scaled = scaler.transform(coords)
    unique_count = max(1, len(np.unique(train_coordinates, axis=0)))
    cluster_count = max(1, min(int(num_regions), len(train_indices), unique_count))
    if cluster_count == 1:
        labels = np.zeros(mappings.num_pois, dtype=np.int64)
    else:
        kmeans = KMeans(
            n_clusters=cluster_count,
            random_state=int(seed),
            n_init=10,
            max_iter=300,
        )
        kmeans.fit(scaled_train)
        labels = kmeans.predict(scaled).astype(np.int64)
    return torch.tensor(labels, dtype=torch.long)


def load_or_build_regions(city: str, mappings, args) -> torch.Tensor:
    train_df, _, _, processed_dir = load_processed_splits(city)
    fingerprint = hashlib.sha1(
        (
            "cluster|%s|%d|%d|%d|%d"
            % (
                _mapping_signature(mappings),
                len(train_df),
                mappings.num_pois,
                int(args.num_regions),
                int(args.seed),
            )
        ).encode("utf-8")
    ).hexdigest()[:16]
    cache_path = processed_dir / (".cluster_regions_%s.pt" % fingerprint)
    if cache_path.exists() and not args.rebuild_context:
        try:
            payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(cache_path, map_location="cpu")
        return payload["poi_region"]
    poi_region = build_regions(train_df, mappings, args.num_regions, args.seed)
    torch.save({"poi_region": poi_region}, cache_path)
    return poi_region


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train PCPNet / DualClusterNet / ASPMix")
    add_common_args(p)
    p.add_argument("--model", type=str, required=True, choices=sorted(MODEL_REGISTRY.keys()))
    p.add_argument("--embedding-dim", type=int, default=64)
    p.add_argument("--context-dim", type=int, default=32)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--n-pattern", type=int, default=32)
    p.add_argument("--n-spatio", type=int, default=32)
    p.add_argument("--proj-rank", type=int, default=32)
    p.add_argument("--num-regions", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--rebuild-context", action="store_true")
    p.set_defaults(loss="ce", lr=1e-3, batch=32, epochs=30, patience=8)
    return p


def make_model(mappings, args):
    name, cls = MODEL_REGISTRY[args.model]
    poi_region = load_or_build_regions(args.city, mappings, args)
    common = dict(
        num_users=mappings.num_users,
        num_pois=mappings.num_pois,
        num_categories=mappings.num_categories,
        poi_region=poi_region,
        embedding_dim=args.embedding_dim,
        context_dim=args.context_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    )
    if args.model == "pcpnet":
        model = cls(**common, n_pattern=args.n_pattern, proj_rank=args.proj_rank)
    elif args.model == "dualcluster":
        model = cls(**common, n_pattern=args.n_pattern, n_spatio=args.n_spatio)
    else:
        model = cls(**common, n_pattern=args.n_pattern, proj_rank=args.proj_rank)
    model.display_name = name  # type: ignore[attr-defined]
    return model


def main() -> None:
    args = build_parser().parse_args()
    display_name, _ = MODEL_REGISTRY[args.model]
    # train_baseline needs make_model(mappings, args) only
    train_baseline(display_name, make_model, args)


if __name__ == "__main__":
    main()
