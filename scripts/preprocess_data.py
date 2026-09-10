#!/usr/bin/env python3
"""Preprocess raw Foursquare CSVs into shared Next-POI splits."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.data import preprocess_foursquare


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--city", type=str, default="all", choices=["NYC", "TKY", "all"])
    p.add_argument("--min-checkins", type=int, default=10)
    args = p.parse_args()
    cities = ["NYC", "TKY"] if args.city == "all" else [args.city]
    for c in cities:
        preprocess_foursquare(c, min_checkins=args.min_checkins)


if __name__ == "__main__":
    main()
