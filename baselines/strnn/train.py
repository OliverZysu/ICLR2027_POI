"""Train/evaluate ST-RNN under the project's unified Next-POI setting."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from baselines.strnn.model import STRNN
from utils.baseline_train import add_common_args, train_baseline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ST-RNN Next-POI baseline")
    add_common_args(parser)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--time-bins", type=int, default=12)
    parser.add_argument("--distance-bins", type=int, default=12)
    parser.add_argument("--max-time-hours", type=float, default=720.0)
    parser.add_argument("--max-distance-km", type=float, default=500.0)
    parser.add_argument("--dropout", type=float, default=0.2)
    return parser


def make_model(mappings, args):
    return STRNN(
        num_users=mappings.num_users,
        num_pois=mappings.num_pois,
        num_categories=mappings.num_categories,
        hidden_dim=args.hidden_dim,
        time_bins=args.time_bins,
        distance_bins=args.distance_bins,
        max_time_hours=args.max_time_hours,
        max_distance_km=args.max_distance_km,
        dropout=args.dropout,
    )


def main() -> None:
    args = build_parser().parse_args()
    train_baseline("ST-RNN", make_model, args)


if __name__ == "__main__":
    main()
