"""Train/evaluate FPMC under the project's unified Next-POI setting."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from baselines.fpmc.model import FPMC
from utils.baseline_train import add_common_args, train_baseline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FPMC Next-POI baseline")
    add_common_args(parser)
    parser.set_defaults(loss="bpr", lr=1e-3)
    parser.add_argument("--mf-dim", type=int, default=64)
    parser.add_argument("--mc-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.0)
    return parser


def make_model(mappings, args):
    return FPMC(
        num_users=mappings.num_users,
        num_pois=mappings.num_pois,
        num_categories=mappings.num_categories,
        mf_dim=args.mf_dim,
        mc_dim=args.mc_dim,
        dropout=args.dropout,
    )


def main() -> None:
    args = build_parser().parse_args()
    train_baseline("FPMC", make_model, args)


if __name__ == "__main__":
    main()
