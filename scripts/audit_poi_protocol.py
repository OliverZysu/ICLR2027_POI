#!/usr/bin/env python3
"""Server preflight: compare strict vs existing legacy query/vocabulary contracts."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_dualcluster_v2 import parser as training_parser, prepare_data, atomic_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cities", default="NYC,TKY")
    p.add_argument("--data-dir", default="", help="Only use for one-city preflight")
    args = p.parse_args()
    cities = [x.strip().upper() for x in args.cities.split(",") if x.strip()]
    if not cities or not set(cities) <= {"NYC", "TKY"}:
        p.error("Use NYC and/or TKY")
    if args.data_dir and len(cities) != 1:
        p.error("--data-dir is only supported with one --cities value")
    report = {}
    for city in cities:
        report[city] = {}
        for protocol in ("train_only", "legacy_union"):
            a = training_parser().parse_args(["--city", city, "--protocol", protocol, "--data-dir", args.data_dir])
            *_, info = prepare_data(a)
            report[city][protocol] = info
            print(city, protocol, "candidates=", info["candidate_size"], "queries=", info["num_queries"],
                  "signature=", info["protocol_sha256"][:12])
    path = ROOT / "logs/DualClusterV2/protocol_preflight.json"
    atomic_json(report, path)
    print("Full report:", path)
    print("Legacy GETNext/STAN use separate dataset/forward code and are NOT repaired by this new evaluator.")
    print("Run comparative local baselines through main.py --baseline NAME to use matched strict queries.")

if __name__ == "__main__":
    main()
