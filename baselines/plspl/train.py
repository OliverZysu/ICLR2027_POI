"""Train/evaluate PLSPL under the project's unified Next-POI setting."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from baselines.plspl.model import PLSPL
from utils.baseline_train import add_common_args, train_baseline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PLSPL Next-POI baseline")
    add_common_args(parser)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--context-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.2)
    return parser


def make_model(mappings, args):
    return PLSPL(
        num_users=mappings.num_users,
        num_pois=mappings.num_pois,
        num_categories=mappings.num_categories,
        embedding_dim=args.embedding_dim,
        context_dim=args.context_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    )


def main() -> None:
    args = build_parser().parse_args()
    train_baseline("PLSPL", make_model, args)


if __name__ == "__main__":
    main()
