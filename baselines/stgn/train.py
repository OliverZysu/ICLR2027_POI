"""Train/evaluate STGN under the project's unified Next-POI setting."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from baselines.stgn.model import STGN
from utils.baseline_train import add_common_args, train_baseline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="STGN Next-POI baseline")
    add_common_args(parser)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)
    return parser


def make_model(mappings, args):
    return STGN(
        num_users=mappings.num_users,
        num_pois=mappings.num_pois,
        num_categories=mappings.num_categories,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        coupled=False,
    )


def main() -> None:
    args = build_parser().parse_args()
    train_baseline("STGN", make_model, args)


if __name__ == "__main__":
    main()
