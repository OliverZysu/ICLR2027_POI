#!/usr/bin/env python3
"""Summarize matched protocols, freeze validation selections, paired user bootstrap."""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_dualcluster_v2 import atomic_json, digest


def collect(root):
    rows = []
    for path in sorted(Path(root).glob("*/*/seed_*/metrics.json")):
        d = json.loads(path.read_text())
        if d.get("smoke") or not d.get("fit_complete"):
            continue
        d["_path"] = str(path)
        rows.append(d)
    return rows


def freeze(rows, path, cities, seeds, top):
    if top < 1:
        raise ValueError("top must be positive")
    rows = [r for r in rows if r["city"] in cities and r["seed"] in seeds and r["protocol"]["protocol"] == "train_only"]
    if any(r.get("test") is not None for r in rows):
        raise ValueError("Selection requires validation-only runs. Test-exposed runs cannot be a fresh tuning pool")
    if Path(path).exists():
        raise FileExistsError("Refusing to overwrite a frozen selection")
    protocols = {city: {r["protocol"]["protocol_sha256"] for r in rows if r["city"] == city} for city in cities}
    if any(len(v) != 1 for v in protocols.values()):
        raise ValueError(f"Need exactly one matched protocol per city; found {protocols}")
    groups = defaultdict(list)
    for r in rows:
        groups[r["experiment"]].append(r)
    required = {(city, seed) for city in cities for seed in seeds}
    complete = {k: v for k, v in groups.items() if {(r["city"], r["seed"]) for r in v} == required}
    if not complete:
        raise ValueError("No experiment covers every requested city/seed")
    def value(records):
        return float(np.mean([4*r["best_val"]["acc@1"]+r["best_val"]["acc@10"] for r in records]))
    candidates = [k for k, v in complete.items() if not v[0]["args"]["baseline"] and v[0]["args"]["variant"] not in ("original", "backbone", "repeat_control")]
    if not candidates:
        raise ValueError("No improved-model candidates completed")
    selected = sorted(candidates, key=lambda k: (-value(complete[k]), k))[:top]
    control_families = defaultdict(list)
    for k, records in complete.items():
        a = records[0]["args"]
        if a["baseline"]:
            control_families["baseline_" + a["baseline"]].append(k)
        elif a["variant"] in ("original", "backbone", "repeat_control"):
            control_families[a["variant"]].append(k)
    # With a budgeted hyperparameter grid, choose each baseline/control's best
    # validation configuration instead of handicapping it with a default run.
    controls = [max(keys, key=lambda k: (value(complete[k]), k)) for keys in control_families.values()]
    signatures = {digest(r["code_sha256"]) for r in rows}
    if len(signatures) != 1:
        raise ValueError("Pilot model code differs across runs; rerun or partition the selection pool")
    records = []
    omit = {"city", "seed", "mode", "device", "data_dir", "output_root", "resume", "overwrite", "smoke", "smoke_samples"}
    for key in selected + sorted(controls):
        a = {k: v for k, v in complete[key][0]["args"].items() if k not in omit}
        # Absent baseline is represented by empty string in argparse, omit it.
        if not a.get("baseline"):
            a.pop("baseline", None)
        for other in complete[key][1:]:
            b = {k: v for k, v in other["args"].items() if k not in omit}
            if not b.get("baseline"):
                b.pop("baseline", None)
            if digest(a) != digest(b):
                raise ValueError(f"Model hyperparameters differ across cities/seeds for {key}")
        a["experiment"] = key
        a["_selection_metadata"] = {"role": "selected" if key in selected else "control", "mean_validation_score": value(complete[key]),
                                    "pilot_sources": [r["_path"] for r in complete[key]]}
        records.append(a)
    locked = {"schema_version": 1, "cities": cities, "pilot_seeds": seeds, "primary_metric": "4*val.acc@1+val.acc@10",
              "test_used_for_selection": False, "code_sha256": rows[0]["code_sha256"], "protocol_sha256_by_city": {c: next(iter(p)) for c, p in protocols.items()},
              "experiments": records}
    locked["selection_sha256"] = digest(locked)
    atomic_json(locked, Path(path))
    print("FROZEN winners:", selected, "controls:", sorted(controls), "->", path)


def compare(a_path, b_path, bootstrap=2000, seed=42):
    if bootstrap < 2:
        raise ValueError("At least two bootstrap resamples are required")
    a_path, b_path = Path(a_path), Path(b_path)
    pa = json.loads((a_path.parent / "protocol.json").read_text())
    pb = json.loads((b_path.parent / "protocol.json").read_text())
    if pa["protocol_sha256"] != pb["protocol_sha256"]:
        raise ValueError("Cannot compare predictions from different data/query protocols")
    a, b = np.load(a_path, allow_pickle=False), np.load(b_path, allow_pickle=False)
    ia, ib = np.argsort(a["query_id"]), np.argsort(b["query_id"])
    if not np.array_equal(a["query_id"][ia], b["query_id"][ib]) or len(np.unique(a["query_id"])) != len(ia):
        raise ValueError("Query IDs are missing, duplicated, or mismatched")
    if not np.array_equal(a["target"][ia], b["target"][ib]) or not np.array_equal(a["user_id"][ia], b["user_id"][ib]):
        raise ValueError("Targets/user identities differ")
    ra, rb = a["rank"][ia], b["rank"][ib]
    user, inv = np.unique(a["user_id"][ia], return_inverse=True)
    count = np.bincount(inv)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(user), (bootstrap, len(user)))
    results = {"sign": "A minus B", "unit": "user-cluster paired bootstrap, conditional on trained checkpoints",
               "queries": len(ra), "users": len(user), "resamples": bootstrap}
    for key, va, vb in (("acc@1", ra<=1, rb<=1), ("acc@10", ra<=10, rb<=10), ("mrr", 1/ra, 1/rb)):
        diff = va.astype(float) - vb.astype(float)
        sums = np.bincount(inv, weights=diff)
        distribution = sums[draws].sum(-1) / count[draws].sum(-1)
        results[key] = {"difference": float(diff.mean()), "ci95": np.quantile(distribution, [0.025,0.975]).tolist()}
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default=str(ROOT / "results/DualClusterV2"))
    p.add_argument("--csv", default="")
    p.add_argument("--freeze", default="")
    p.add_argument("--cities", default="NYC,TKY")
    p.add_argument("--seeds", default="42")
    p.add_argument("--top", type=int, default=2)
    p.add_argument("--compare", nargs=2, metavar=("A_NPZ", "B_NPZ"))
    p.add_argument("--bootstrap", type=int, default=2000)
    args = p.parse_args()
    if args.compare:
        print(json.dumps(compare(*args.compare, bootstrap=args.bootstrap), indent=2)); return
    rows = collect(args.root)
    groups = defaultdict(list)
    for r in rows:
        groups[(r["city"], r["protocol"]["protocol_sha256"], r["experiment"])].append(r)
    export = []
    for (city, protocol, experiment), records in sorted(groups.items()):
        rec = dict(city=city, protocol=protocol, experiment=experiment, seeds=",".join(str(r["seed"]) for r in records),
                   n=len(records), validation_score=float(np.mean([4*r["best_val"]["acc@1"]+r["best_val"]["acc@10"] for r in records])))
        for metric in ("acc@1", "acc@5", "acc@10", "mrr"):
            values = [r["test"][metric] for r in records if r.get("test") is not None]
            rec[f"test_{metric}_mean"] = float(np.mean(values)) if values else ""
            rec[f"test_{metric}_std"] = float(np.std(values, ddof=1)) if len(values)>1 else ""
            rec["tested_seeds"] = len(values)
        print(f"{city} protocol={protocol[:10]} {experiment:22} seeds={rec['seeds']:10} val_score={rec['validation_score']:.5f} test_acc1={rec['test_acc@1_mean']}")
        export.append(rec)
    if not rows:
        print("No completed non-smoke experiments found")
    if args.csv and export:
        path = Path(args.csv); path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(export[0])); writer.writeheader(); writer.writerows(export)
    if args.freeze:
        freeze(rows, args.freeze, [x.strip().upper() for x in args.cities.split(",")], [int(x) for x in args.seeds.split(",")], args.top)

if __name__ == "__main__":
    main()
