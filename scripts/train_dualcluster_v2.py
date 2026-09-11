#!/usr/bin/env python3
"""Audited full-candidate Next-POI training; no changes to host utils.

Default `fit` never evaluates test. `test` loads a frozen best checkpoint.
`train-test` is provided for locked runs/smoke tests, not hyperparameter search.
All experiment files are isolated from the pre-existing benchmark results.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from functools import partial
import hashlib
import importlib
import json
import logging
import math
import os
from pathlib import Path
import random
import re
import sys
import time
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from utils.baseline_train import (IdMappings, TrajectoryNextPoiDataset, build_mappings,
                                 collate_fn, bpr_loss, _haversine_km)
from utils.metrics import evaluate_ranking, format_metrics
from utils.common import set_seed
from models.dualcluster import DualClusterNet
from models.dualcluster_v2 import DualClusterV2, VARIANTS

VERSION = "dc-v2-audit-1"
BASELINES = ("fpmc", "strnn", "stgn", "stgcn", "plspl", "mtnet", "dchl", "ipcm", "k1_poi", "sthgcn")


def digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False, default=str), encoding="utf-8")
    os.replace(tmp, path)


def atomic_torch(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: Path):
    # Only load checkpoints you created/trust. weights_only=False is required
    # for optimizer and Python/NumPy RNG state, not for untrusted downloads.
    return torch.load(path, map_location="cpu", weights_only=False)


def mapping_payload(maps: IdMappings):
    return dict(user_id2idx=maps.user_id2idx, poi_id2idx=maps.poi_id2idx,
                cat_id2idx={str(k): v for k, v in maps.cat_id2idx.items()},
                poi_category=[maps.poi_idx2cat_idx[i] for i in range(maps.num_pois)],
                poi_coords=maps.poi_coords.tolist())


def strict_mappings(train: pd.DataFrame, processed: Path) -> IdMappings:
    """Vocabulary and static POI metadata from train ONLY; graph order if valid."""
    empty = train.iloc[:0]
    maps = build_mappings(train, empty, empty)
    graph_path = processed / "graph_X.csv"
    if graph_path.exists():
        ids = pd.read_csv(graph_path, dtype={"node_name/poi_id": str})["node_name/poi_id"].tolist()
        if len(ids) != len(set(ids)) or set(ids) != set(maps.poi_id2idx):
            raise ValueError("graph_X POIs differ from training vocabulary or contain duplicates. "
                             "Audit/rebuild that graph; do not reuse graph/checkpoint indices silently.")
        order = [maps.poi_id2idx[p] for p in ids]
        maps = IdMappings(maps.user_id2idx, {p: i for i, p in enumerate(ids)}, maps.cat_id2idx,
                          {i: maps.poi_idx2cat_idx[j] for i, j in enumerate(order)}, maps.poi_coords[order])
    return maps


class StrictTrajectoryDataset(Dataset):
    """One last-target query per original held-out trajectory, no gap bridging.

    Training: each known contiguous segment, all shifted targets, last max_len+1.
    Held-out: keep only the original trajectory's final known contiguous suffix;
    do NOT replace an unknown final target by a different earlier target.
    """
    def __init__(self, frame, maps, max_len=50, min_history=1, training=False):
        self.samples = []
        self.exclusions = Counter()
        self.maps = maps
        for tid, g in frame.groupby("trajectory_id", sort=True):
            g = g.sort_values("_utc", kind="mergesort")
            users = g["user_id"].astype(str).unique()
            if len(users) != 1:
                raise ValueError(f"Trajectory {tid} contains multiple users")
            uid = str(users[0])
            if uid not in maps.user_id2idx:
                self.exclusions["unknown_user_trajectories"] += 1
                continue
            segments, current = [], []
            for row in g.to_dict("records"):
                if str(row["POI_id"]) not in maps.poi_id2idx:
                    self.exclusions["unknown_poi_checkins"] += 1
                    if training and current:
                        segments.append(current)
                    current = []
                else:
                    current.append(row)
            if current:
                segments.append(current)
            # For eval, only `current` is the original final contiguous suffix.
            if not training:
                segments = [current] if current else []
                if not current:
                    self.exclusions["unknown_final_target_trajectories"] += 1
            made = 0
            for seg_num, rows in enumerate(segments):
                rows = rows[-(max_len + 1):]
                if len(rows) - 1 < min_history:
                    self.exclusions["short_segments"] += 1
                    continue
                poi = [maps.poi_id2idx[str(r["POI_id"])] for r in rows]
                times = [r["_utc"] for r in rows]
                dt, dd = [0.0], [0.0]
                for i in range(1, len(rows) - 1):
                    dt.append(max(0., (times[i] - times[i-1]).total_seconds() / 3600.))
                    dd.append(_haversine_km(float(rows[i-1]["latitude"]), float(rows[i-1]["longitude"]),
                                            float(rows[i]["latitude"]), float(rows[i]["longitude"])))
                identity = [str(tid), uid, str(times[-1]), str(rows[-1]["POI_id"])]
                self.samples.append(dict(
                    traj_id=str(tid), qid=digest(identity), user_raw=uid,
                    user=maps.user_id2idx[uid], poi=poi[:-1], target=poi[1:],
                    category=[maps.poi_idx2cat_idx[p] for p in poi[:-1]],
                    hour=[int(float(r["norm_in_day_time"]) * 24) % 24 for r in rows[:-1]],
                    weekday=[int(r["day_of_week"]) % 7 for r in rows[:-1]],
                    delta_time_h=dt, delta_distance_km=dd,
                    input_end_timestamp=str(times[-2]), target_timestamp=str(times[-1]),
                ))
                made += 1
            if not made:
                self.exclusions["no_query_trajectories"] += 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def collate_audited(rows, poi_pad, cat_pad):
    batch = collate_fn(rows, poi_pad, cat_pad)
    batch["query_ids"] = [x["qid"] for x in rows]
    batch["user_ids"] = [x["user_raw"] for x in rows]
    return batch


def validate_frames(frames, strict=True):
    bounds, duplicate_keys = [], []
    cols = ["user_id", "POI_id", "POI_catid_code", "latitude", "longitude", "UTC_time",
            "trajectory_id", "norm_in_day_time", "day_of_week"]
    for split, df in zip(("train", "val", "test"), frames):
        missing = set(cols) - set(df.columns)
        if missing:
            raise ValueError(f"{split} missing columns {missing}")
        if df.empty:
            raise ValueError(f"{split} is empty")
        df["_utc"] = pd.to_datetime(df["UTC_time"], utc=True, errors="coerce")
        if df[cols].isna().any().any() or df["_utc"].isna().any():
            raise ValueError(f"{split} has missing/invalid required values")
        numeric = df[["latitude", "longitude", "POI_catid_code", "norm_in_day_time", "day_of_week"]].to_numpy(float)
        if not np.isfinite(numeric).all():
            raise ValueError(f"{split} has nonfinite feature values")
        if not df["latitude"].between(-90, 90).all() or not df["longitude"].between(-180, 180).all():
            raise ValueError(f"{split} invalid geographic coordinates")
        bounds.append((df["_utc"].min(), df["_utc"].max()))
        duplicate_keys.append(set(zip(df["user_id"].astype(str), df["POI_id"].astype(str), df["_utc"].astype(str))))
    ordered = bounds[0][1] <= bounds[1][0] and bounds[1][1] <= bounds[2][0]
    overlap = {f"{a}-{b}": len(duplicate_keys[a] & duplicate_keys[b]) for a, b in ((0,1),(0,2),(1,2))}
    if strict and (not ordered or any(overlap.values())):
        raise ValueError(f"Nonchronological or overlapping splits: ordered={ordered}, overlaps={overlap}")
    return dict(chronological=bool(ordered), overlaps=overlap,
                time_bounds=[[str(x), str(y)] for x, y in bounds])


def prepare_data(args):
    processed = Path(args.data_dir).resolve() if args.data_dir else ROOT / "datasets" / "processed" / args.city
    paths = [processed / f"{args.city}_{name}.csv" for name in ("train", "val", "test")]
    for p in paths:
        if not p.is_file():
            raise FileNotFoundError(f"Missing processed split: {p}; no synthetic data is substituted.")
    frames = [pd.read_csv(p, dtype={"user_id": str, "POI_id": str, "trajectory_id": str}) for p in paths]
    data_checks = validate_frames(frames, strict=args.protocol == "train_only")
    if args.protocol == "train_only":
        maps = strict_mappings(frames[0], processed)
        datasets = [StrictTrajectoryDataset(df, maps, args.max_len, args.min_history, training=(i == 0))
                    for i, df in enumerate(frames)]
    else:
        if args.min_history != 1:
            raise ValueError("legacy_union preserves the existing min_history=1 rule")
        maps = build_mappings(*frames)
        datasets = [TrajectoryNextPoiDataset(df, maps, args.max_len) for df in frames]
        for ds in datasets:
            ds.exclusions = {"legacy_warning": "union vocabulary includes held-out POIs and metadata"}
            for row in ds.samples:
                row["qid"] = digest([row["traj_id"], row["poi"], row["target"]])
                row["user_raw"] = str(row["user"])
    if args.smoke:
        for ds in datasets:
            ds.samples = ds.samples[:max(args.smoke_samples, 1)]
    for split, ds in zip(("train", "val", "test"), datasets):
        if len(ds) == 0:
            raise ValueError(f"No valid {split} queries after protocol filtering")
    for split, ds in zip(("train", "val", "test"), datasets):
        if split != "train" and len({r["qid"] for r in ds.samples}) != len(ds):
            raise ValueError(f"Duplicate {split} query identifiers; inspect duplicated check-ins")
    query_hashes = {split: digest([(r["qid"], r["poi"], r["target"][-1]) for r in ds.samples])
                    for split, ds in zip(("train", "val", "test"), datasets)}
    provenance = {
        "version": VERSION, "protocol": args.protocol, "candidate_scope": "train-only" if args.protocol == "train_only" else "legacy-union",
        "candidate_size": maps.num_pois, "num_users": maps.num_users, "num_categories": maps.num_categories,
        "max_len": args.max_len, "min_history": args.min_history,
        "train_supervision": "all shifted valid steps", "eval_supervision": "one original final target per trajectory",
        "unknown_rule": "cut_at_unknown_no_target_replacement" if args.protocol == "train_only" else "legacy",
        "selection": "4*val.acc@1+val.acc@10", "future_target_features": False,
        "files": {p.name: file_hash(p) for p in paths}, "mappings_sha256": digest(mapping_payload(maps)),
        "query_sha256": query_hashes, "num_queries": {k: len(d) for k, d in zip(("train", "val", "test"), datasets)},
        "exclusions": {k: dict(d.exclusions) for k, d in zip(("train", "val", "test"), datasets)},
        "data_checks": data_checks, "smoke": bool(args.smoke),
    }
    provenance["protocol_sha256"] = digest(provenance)
    return frames, maps, datasets, processed, provenance


def build_spatial_context(train, maps, regions=64, cluster_seed=42):
    """KMeans in approximate local kilometres, fit only training POI locations."""
    fit_ids = np.array(sorted({maps.poi_id2idx[str(p)] for p in train["POI_id"]}), dtype=np.int64)
    coords = maps.poi_coords.astype(np.float64)
    origin = coords[fit_ids].mean(0)
    xy = np.column_stack([(coords[:, 1] - origin[1]) * (math.pi / 180) * 6371 * math.cos(math.radians(origin[0])),
                          (coords[:, 0] - origin[0]) * (math.pi / 180) * 6371])
    r = min(int(regions), len(np.unique(xy[fit_ids], axis=0)))
    r = max(1, r)
    km = KMeans(n_clusters=r, random_state=int(cluster_seed), n_init=10).fit(xy[fit_ids])
    d2 = ((xy[:, None] - km.cluster_centers_[None]) ** 2).sum(-1)
    nearest = d2[fit_ids].min(-1)
    positive = nearest[nearest > 1e-10]
    scale2 = max(float(np.median(positive)) if positive.size else 1.0, 0.01)
    logits = -(d2 - d2.min(-1, keepdims=True)) / scale2
    membership = np.exp(np.maximum(logits, -60))
    membership = membership / membership.sum(-1, keepdims=True)
    membership = (membership + 1e-6 / r) / (1 + 1e-6)
    return dict(poi_region=torch.tensor(d2.argmin(-1), dtype=torch.long),
                region_membership=torch.tensor(membership, dtype=torch.float32),
                poi_category=torch.tensor([maps.poi_idx2cat_idx[i] for i in range(maps.num_pois)]),
                metadata=dict(num_regions=r, origin_latlon=origin.tolist(), centers_local_km=km.cluster_centers_.tolist(),
                              scale2_km=scale2, fitted_on="training POI coordinates only", cluster_seed=int(cluster_seed)))


def model_factory(args, maps, train, context_dir):
    common = dict(num_users=maps.num_users, num_pois=maps.num_pois, num_categories=maps.num_categories)
    if args.baseline:
        mod = importlib.import_module(f"baselines.{args.baseline}.train")
        ba = mod.build_parser().parse_args([])
        ba.city, ba.seed, ba.rebuild_context = args.city, args.cluster_seed, True
        extra = json.loads(args.baseline_args)
        forbidden = {"city", "seed", "rebuild_context", "loss", "lr", "weight_decay", "batch", "epochs", "max_len", "num_neg", "smoke", "no_cuda", "data_dir", "output_root", "patience"}
        if not isinstance(extra, dict) or set(extra) & forbidden:
            raise ValueError("baseline-args only overrides architecture, not protocol/training/runtime options")
        unknown = set(extra) - set(vars(ba))
        if unknown:
            raise ValueError(f"Unknown baseline args: {unknown}")
        for key, value in extra.items():
            setattr(ba, key, value)
        # Statistical caches live in this run's isolated, fingerprinted directory.
        # Existing server cache files are neither read nor overwritten.
        context_dir.mkdir(parents=True, exist_ok=True)
        if hasattr(mod, "build_context"):
            ctx = mod.build_context(train, maps, context_dir, ba)
            model = mod.make_model(maps, ba, ctx)
        else:
            model = mod.make_model(maps, ba)
        return model, {"baseline": args.baseline, "implementation": "existing local port, not recertified official reproduction",
                       "baseline_args": vars(ba)}, (ba.loss if args.loss == "auto" else args.loss)
    if args.variant == "original":
        from scripts.train_cluster_model import build_regions
        region = build_regions(train, maps, args.num_regions, args.cluster_seed)
        model = DualClusterNet(**common, poi_region=region, embedding_dim=args.embedding_dim,
                               context_dim=args.context_dim, hidden_dim=args.hidden_dim,
                               n_pattern=args.n_pattern, n_spatio=args.n_spatio, dropout=args.dropout)
        return model, {"variant": "original", "implementation": "unchanged uploaded DualClusterNet"}, "ce"
    spatial = build_spatial_context(train, maps, args.num_regions, args.cluster_seed)
    kwargs = {key: getattr(args, key) for key in
              ("embedding_dim", "context_dim", "hidden_dim", "n_pattern", "proj_rank", "dropout", "temperature",
               "category_weight", "region_weight", "balance_weight", "confidence_weight", "decorrelation_weight", "repeat_weight")}
    model = DualClusterV2(**common, **{k: spatial[k] for k in ("poi_region", "region_membership", "poi_category")},
                          variant=args.variant, **kwargs)
    return model, {"variant": args.variant, "flags": asdict(model.variant), "kwargs": kwargs,
                   "spatial": spatial["metadata"]}, "ce"


def move_batch(batch, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def make_loaders(datasets, maps, args, generator):
    fn = partial(collate_audited, poi_pad=maps.num_pois, cat_pad=maps.num_categories)
    return [DataLoader(ds, batch_size=args.batch if i == 0 else args.eval_batch,
                       shuffle=i == 0, collate_fn=fn, num_workers=args.num_workers,
                       generator=generator if i == 0 else None, pin_memory=args.device.startswith("cuda"))
            for i, ds in enumerate(datasets)]


@torch.no_grad()
def evaluate(model, loader, device, prediction_path=None, analysis_regions=None):
    model.eval()
    weighted = Counter()
    rows, targets_out, users, qids, repeat_flags, region_flags = [], [], [], [], [], []
    total_tokens, loss_total, n = 0, 0.0, 0
    diagnostic_totals, diagnostic_n = Counter(), 0
    pattern_mass, pattern_count = None, 0
    for raw in loader:
        b = move_batch(raw, device)
        scores = model(b)
        if not torch.isfinite(scores).all():
            raise FloatingPointError("Nonfinite full-vocabulary evaluation scores")
        loss = F.cross_entropy(scores.reshape(-1, scores.size(-1)), b["target"].reshape(-1), ignore_index=-1, reduction="sum")
        nt = int(b["target"].ge(0).sum())
        loss_total += float(loss); total_tokens += nt
        index = b["lengths"] - 1
        rr = torch.arange(len(index), device=device)
        final_scores = scores[rr, index].cpu().numpy()
        labels = b["target"][rr, index].cpu().numpy()
        m = evaluate_ranking(final_scores, labels)  # reuse unchanged host metric implementation
        nb = len(labels)
        for key, value in m.items():
            if key != "num_samples":
                weighted[key] += value * nb
        n += nb
        if hasattr(model, "diagnostics"):
            for key, value in model.diagnostics().items():
                diagnostic_totals[key] += value * nb
            diagnostic_n += nb
            cache = getattr(model, "_cache", {})
            if "alpha" in cache:
                q = cache["alpha"][cache["mask"]].detach().float().cpu()
                pattern_mass = q.sum(0) if pattern_mass is None else pattern_mass + q.sum(0)
                pattern_count += len(q)
        # Preserve exact per-query ranks, not only rounded aggregate metrics.
        for i, (row, label) in enumerate(zip(final_scores, labels)):
            rank = int(np.where(np.argsort(-row) == label)[0][0]) + 1
            rows.append(rank); targets_out.append(int(label))
            qids.append(raw["query_ids"][i]); users.append(raw["user_ids"][i])
            length = int(raw["lengths"][i])
            hist = raw["poi"][i, :length]
            repeat_flags.append(bool((hist == int(label)).any()))
            if analysis_regions is not None:
                region_flags.append(int(analysis_regions[int(hist[-1])] == analysis_regions[int(label)]))
            else:
                region_flags.append(-1)  # unavailable, NOT "cross-region"
    if n == 0 or total_tokens == 0:
        raise ValueError("Evaluation contains no valid targets")
    metrics = {k: float(v / n) for k, v in weighted.items()}
    metrics.update(num_samples=n, loss=loss_total / total_tokens)
    rank_array = np.asarray(rows, dtype=np.int64)
    groups = [("repeat_prefix", np.asarray(repeat_flags)), ("explore_prefix", ~np.asarray(repeat_flags))]
    if analysis_regions is not None:
        groups += [("same_region", np.asarray(region_flags) == 1), ("cross_region", np.asarray(region_flags) == 0)]
    for group, membership in groups:
        selected = rank_array[membership]
        metrics[group] = dict(n=int(len(selected)), **({"acc@1": float((selected <= 1).mean()),
                             "acc@10": float((selected <= 10).mean()), "mrr": float((1 / selected).mean())} if len(selected) else {}))
    if prediction_path:
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(prediction_path, query_id=np.asarray(qids, dtype=str), user_id=np.asarray(users, dtype=str),
                            target=np.asarray(targets_out), rank=rank_array, repeat_prefix=np.asarray(repeat_flags),
                            same_hard_region=np.asarray(region_flags, dtype=np.int8))
    diag = {k: v / diagnostic_n for k, v in diagnostic_totals.items()} if diagnostic_n else {}
    if pattern_mass is not None:
        mass = (pattern_mass / pattern_count).clamp_min(1e-12)
        diag["pattern_mass_global"] = mass.tolist()
        diag["pattern_effective_clusters_global"] = float((-(mass * mass.log()).sum()).exp())
    return metrics, diag


def code_fingerprint():
    paths = [ROOT / "models/dualcluster_v2.py", Path(__file__), ROOT / "utils/baseline_train.py",
             ROOT / "utils/metrics.py", ROOT / "models/blocks.py", ROOT / "models/dualcluster.py",
             ROOT / "scripts/train_cluster_model.py"]
    paths.extend(sorted((ROOT / "baselines").glob("*/model.py")))
    paths.extend(sorted((ROOT / "baselines").glob("*/train.py")))
    return {str(p.relative_to(ROOT)): file_hash(p) for p in paths}


def config_signature(args):
    ignored = {"mode", "device", "resume", "overwrite", "output_root", "data_dir", "epochs", "num_workers", "eval_batch"}
    return digest({k: v for k, v in vars(args).items() if k not in ignored})


def rng_state(generator):
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [], "loader": generator.get_state()}


def restore_rng(state, generator):
    random.setstate(state["python"]); np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"]); generator.set_state(state["loader"])
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def paths_for(args):
    exp = args.experiment or ("local_" + args.baseline if args.baseline else args.variant)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", exp) or exp in (".", ".."):
        raise ValueError("experiment must be a plain filename-safe name, not a path")
    if args.smoke and not exp.endswith("_smoke"):
        exp += "_smoke"
    base = Path(args.output_root).resolve() if args.output_root else ROOT
    key = Path("DualClusterV2") / exp / args.city / f"seed_{args.seed}"
    return {"experiment": exp, "checkpoint": base / "checkpoints" / key,
            "result": base / "results" / key, "log": base / "logs" / key}


def run(args):
    if args.epochs < 1 or args.batch < 1 or args.max_len < 1 or args.min_history < 1 or args.min_history > args.max_len:
        raise ValueError("Invalid epoch/batch/history bounds")
    if args.protocol == "legacy_union" and args.baseline:
        # Still allowed for diagnosis, but never confused with the strict table.
        print("WARNING: legacy-union is diagnostic, not the strict GETNext closed-set protocol.")
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")
    if args.eval_batch < 1 or args.num_regions < 1 or args.n_pattern < 1 or args.proj_rank < 1 or args.temperature <= 0:
        raise ValueError("Batch, clustering, rank and temperature must be positive")
    if args.lr <= 0 or args.grad_clip <= 0 or not 0 <= args.dropout < 1 or args.weight_decay < 0:
        raise ValueError("Invalid optimization/dropout bounds")
    if not args.baseline and args.loss not in ("auto", "ce"):
        raise ValueError("New/original DualCluster models use cross-entropy; BPR is only supported for local baseline controls")
    initial_paths = paths_for(args)
    if args.mode == "test":
        best_path = initial_paths["checkpoint"] / "best.pt"
        frozen = load_checkpoint(best_path)
        # Evaluation reconstructs the saved architecture, not today's defaults.
        runtime = {k: getattr(args, k) for k in ("device", "data_dir", "output_root", "eval_batch", "num_workers")}
        args = argparse.Namespace(**frozen["args"])
        for k, value in runtime.items():
            if value != "":
                setattr(args, k, value)
        args.mode, args.resume, args.overwrite = "test", False, False
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; check server environment or use --device cpu for smoke tests")
    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
    set_seed(args.seed)
    if args.cpu_threads:
        torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)
    paths = paths_for(args)
    if args.overwrite and args.mode != "test" and not args.resume:
        stamp = str(time.time_ns())
        for key in ("checkpoint", "result", "log"):
            old = paths[key]
            if old.exists():
                old.rename(old.with_name("archive_" + old.name + "_" + stamp))
    for key in ("checkpoint", "result", "log"):
        paths[key].mkdir(parents=True, exist_ok=True)
    best_path = paths["checkpoint"] / "best.pt"
    last_path = paths["checkpoint"] / "last.pt"
    result_path = paths["result"] / "metrics.json"
    if args.resume and result_path.exists() and json.loads(result_path.read_text()).get("test") is not None:
        raise ValueError("This run has already exposed test metrics; continuing it would erase the frozen test record. Use a new experiment ID")
    if args.mode != "test" and (best_path.exists() or last_path.exists() or result_path.exists()) and not (args.resume or args.overwrite):
        raise FileExistsError(f"Run exists: {paths['checkpoint']}; use a new --experiment or --resume; no silent overwriting")
    logger = logging.getLogger(f"dc-v2-{os.getpid()}")
    for old_handler in list(logger.handlers):
        old_handler.flush(); old_handler.close(); logger.removeHandler(old_handler)
    logger.setLevel(logging.INFO); logger.propagate = False
    formatter = logging.Formatter("%(asctime)s %(message)s")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(paths["log"] / "train.log", encoding="utf-8")):
        handler.setFormatter(formatter); logger.addHandler(handler)
    started = time.time()
    frames, maps, datasets, processed, provenance = prepare_data(args)
    # Validate before replacing provenance: failed evaluation/resume must not alter it.
    protocol_path = paths["result"] / "protocol.json"
    if protocol_path.exists() and (args.mode == "test" or args.resume):
        previous = json.loads(protocol_path.read_text())
        if previous["protocol_sha256"] != provenance["protocol_sha256"]:
            raise ValueError("Checkpoint/data protocol fingerprint mismatch; original provenance was preserved")
    if args.mode != "test" and not args.resume:
        atomic_json(provenance, protocol_path)
    logger.info("device=%s protocol=%s V=%d query counts=%s", device, args.protocol, maps.num_pois, provenance["num_queries"])
    generator = torch.Generator().manual_seed(args.seed)
    loaders = make_loaders(datasets, maps, args, generator)
    model, model_config, loss_kind = model_factory(args, maps, frames[0], paths["checkpoint"] / "context")
    model = model.to(device)
    # Evaluation subgroups use one fixed TRAIN-fitted geography for all models,
    # independent of their own region count, clustering, or shuffled-space control.
    analysis_regions = build_spatial_context(frames[0], maps, 64, 42)["poi_region"].to(device)
    fingerprint = code_fingerprint()
    params = sum(p.numel() for p in model.parameters())
    logger.info("experiment=%s trainable parameters=%d loss=%s", paths["experiment"], params, loss_kind)
    common_result = {"schema_version": VERSION, "experiment": paths["experiment"], "city": args.city, "seed": args.seed,
                     "protocol": provenance, "args": vars(args), "model_config": model_config,
                     "trainable_parameters": params, "code_sha256": fingerprint,
                     "diagnostic_regions": {"requested_clusters": 64, "seed": 42, "fit": "train POIs only",
                                            "mapping_sha256": digest(analysis_regions.cpu().tolist())},
                     "environment": {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
                                     "device": str(device), "cuda": torch.version.cuda}, "smoke": bool(args.smoke)}
    if args.mode == "test":
        if frozen["protocol_sha256"] != provenance["protocol_sha256"]:
            raise ValueError("Checkpoint/data protocol fingerprint mismatch; do not evaluate against changed mappings or queries")
        if frozen["code_sha256"] != fingerprint:
            raise ValueError("Model/training code changed since checkpoint creation; audit changes before evaluation")
        if result_path.exists() and json.loads(result_path.read_text()).get("test") is not None:
            raise FileExistsError("Test already evaluated for this run. Existing test files are immutable; use saved predictions for analysis")
        model.load_state_dict(frozen["model"])
        test, diagnostic = evaluate(model, loaders[2], device, paths["result"] / "test_predictions.npz", analysis_regions)
        out = json.loads(result_path.read_text()) if result_path.exists() else common_result
        out.update(test=test, test_diagnostics=diagnostic, tested_checkpoint_sha256=file_hash(best_path), test_epoch=frozen["epoch"])
        atomic_json(out, result_path)
        logger.info("TEST %s", format_metrics(test)); return out
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    best_score, best_epoch, best_val = -float("inf"), 0, None
    patience_left, history, start_epoch = args.patience, [], 1
    if args.resume:
        saved = load_checkpoint(last_path)
        if saved["protocol_sha256"] != provenance["protocol_sha256"] or saved["config_signature"] != config_signature(args):
            raise ValueError("Resume configuration/protocol changed")
        if saved["code_sha256"] != fingerprint:
            raise ValueError("Resume code fingerprint changed")
        model.load_state_dict(saved["model"]); optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        best_score, best_epoch, best_val = saved["best_score"], saved["best_epoch"], saved["best_val"]
        patience_left, history = saved["patience_left"], saved["history"]
        start_epoch = saved["epoch"] + 1
        restore_rng(saved["rng"], generator)
        if patience_left <= 0:
            raise ValueError("Run already early-stopped; a continuation changes the experiment and needs a new experiment ID")
    for epoch in range(start_epoch, args.epochs + 1):
        begin = time.time(); model.train()
        if hasattr(model, "auxiliary_scale"):
            model.auxiliary_scale = min(1., epoch / max(1, args.aux_warmup))
        loss_sum, tokens, aux_sum, steps = 0., 0, Counter(), 0
        for raw in loaders[0]:
            b = move_batch(raw, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(b)
            if logits.shape[:2] != b["target"].shape or logits.size(-1) != maps.num_pois:
                raise ValueError("Model must return (B,S,V) with the common candidate vocabulary")
            if loss_kind == "bpr":
                main_loss = bpr_loss(logits, b["target"], args.num_neg)
            else:
                main_loss = F.cross_entropy(logits.reshape(-1, maps.num_pois), b["target"].reshape(-1), ignore_index=-1)
            aux, details = logits.new_zeros(()), {}
            if hasattr(model, "auxiliary_loss"):
                aux, details = model.auxiliary_loss(b)
            elif hasattr(model, "compute_auxiliary_loss"):
                aux = model.compute_auxiliary_loss(b)
                details = {"local_baseline_auxiliary": float(aux.detach())}
            loss = main_loss + aux
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite training loss at epoch {epoch}; no batches silently skipped")
            loss.backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip, error_if_nonfinite=True)
            optimizer.step()
            n = int(b["target"].ge(0).sum())
            loss_sum += float(main_loss.detach()) * n; tokens += n; steps += 1
            for key, value in details.items():
                aux_sum[key] += value
        if not steps:
            raise ValueError("No optimizer updates were performed")
        val, diag = evaluate(model, loaders[1], device, analysis_regions=analysis_regions)
        if not math.isfinite(val["loss"]):
            raise FloatingPointError("Nonfinite validation loss")
        scheduler.step(val["loss"])
        score = 4 * val["acc@1"] + val["acc@10"]
        improved = score > best_score
        if improved:
            best_score, best_epoch, best_val, patience_left = score, epoch, val, args.patience
        else:
            patience_left -= 1
        record = dict(epoch=epoch, train_loss=loss_sum / max(tokens, 1), updates=steps,
                      auxiliary={k: v / steps for k, v in aux_sum.items()}, val=val, diagnostics=diag,
                      seconds=time.time() - begin, lr=optimizer.param_groups[0]["lr"])
        history.append(record)
        logger.info("epoch=%d train_loss=%.5f val[%s] lr=%.3g seconds=%.1f diag=%s", epoch, record["train_loss"],
                    format_metrics(val), record["lr"], record["seconds"], json.dumps(diag))
        checkpoint = dict(schema_version=VERSION, model=model.state_dict(), optimizer=optimizer.state_dict(),
                          scheduler=scheduler.state_dict(), args=vars(args), model_config=model_config,
                          config_signature=config_signature(args), code_sha256=fingerprint,
                          protocol_sha256=provenance["protocol_sha256"], protocol=provenance,
                          mappings=mapping_payload(maps), epoch=epoch, best_epoch=best_epoch, best_score=best_score,
                          best_val=best_val, patience_left=patience_left, history=history, rng=rng_state(generator))
        if improved:
            atomic_torch(checkpoint, best_path)
        atomic_torch(checkpoint, last_path)
        out = dict(common_result, best_val=best_val, best_epoch=best_epoch, history=history,
                   fit_complete=False, test=None, elapsed_seconds=time.time() - started)
        atomic_json(out, result_path)
        if patience_left <= 0:
            logger.info("early stopping at epoch %d; best epoch %d", epoch, best_epoch)
            break
    if not best_path.exists():
        raise RuntimeError("No best checkpoint was saved")
    best = load_checkpoint(best_path)
    model.load_state_dict(best["model"])
    final_val, diag = evaluate(model, loaders[1], device, paths["result"] / "val_predictions.npz", analysis_regions)
    out = dict(common_result, best_val=final_val, best_epoch=best["epoch"], val_diagnostics=diag,
               history=history, fit_complete=True, test=None, elapsed_seconds=time.time() - started,
               checkpoint_sha256=file_hash(best_path))
    if args.mode == "train-test":
        out["test"], out["test_diagnostics"] = evaluate(model, loaders[2], device, paths["result"] / "test_predictions.npz", analysis_regions)
        out["test_epoch"] = best["epoch"]
    atomic_json(out, result_path)
    logger.info("saved %s; test_evaluated=%s", result_path, out["test"] is not None)
    return out


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--city", choices=["NYC", "TKY"], default="NYC")
    p.add_argument("--variant", choices=["original"] + list(VARIANTS), default="full")
    p.add_argument("--baseline", choices=BASELINES, default="")
    p.add_argument("--baseline-args", default="{}", help="JSON overrides for local baseline architecture only")
    p.add_argument("--experiment", default="", help="Immutable run name; use a new name for every hyperparameter config")
    p.add_argument("--protocol", choices=["train_only", "legacy_union"], default="train_only")
    p.add_argument("--mode", choices=["fit", "test", "train-test"], default="fit")
    p.add_argument("--data-dir", default="", help="Directory containing CITY_train/val/test.csv; no automatic downloads")
    p.add_argument("--output-root", default="", help="Default project root")
    p.add_argument("--device", default="auto")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--eval-batch", type=int, default=32)
    p.add_argument("--max-len", type=int, default=50)
    p.add_argument("--min-history", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cluster-seed", type=int, default=42, help="Fixed geographic clustering across optimization seeds")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--grad-clip", type=float, default=5.)
    p.add_argument("--loss", choices=["auto", "ce", "bpr"], default="auto")
    p.add_argument("--num-neg", type=int, default=10)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--cpu-threads", type=int, default=4)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--embedding-dim", type=int, default=64)
    p.add_argument("--context-dim", type=int, default=32)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--n-pattern", type=int, default=32)
    p.add_argument("--n-spatio", type=int, default=32, help="Used only by original DualClusterNet")
    p.add_argument("--num-regions", type=int, default=64)
    p.add_argument("--proj-rank", type=int, default=16)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--category-weight", type=float, default=0.1)
    p.add_argument("--region-weight", type=float, default=0.1)
    p.add_argument("--balance-weight", type=float, default=0.01)
    p.add_argument("--confidence-weight", type=float, default=0.001)
    p.add_argument("--decorrelation-weight", type=float, default=0.001)
    p.add_argument("--repeat-weight", type=float, default=0.05)
    p.add_argument("--aux-warmup", type=int, default=5)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--overwrite", action="store_true", help="Explicitly replace this isolated run; never a legacy result")
    p.add_argument("--smoke", action="store_true", help="Synthetic/subset engineering test, excluded from research summaries")
    p.add_argument("--smoke-samples", type=int, default=8)
    return p


def main():
    run(parser().parse_args())


if __name__ == "__main__":
    main()
