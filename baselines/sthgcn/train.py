"""Train/evaluate STHGCN using only the host project's existing ``utils``."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from functools import partial
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.baseline_train import (
    TrajectoryNextPoiDataset,
    add_common_args,
    bpr_loss,
    build_mappings,
    ce_loss,
    collate_fn,
    evaluate_model,
)
from utils.common import PROJECT_ROOT, ensure_dir, get_device, set_seed
from utils.data import load_processed_splits
from utils.metrics import format_metrics


def _canonical(value) -> str:
    return "" if pd.isna(value) else str(value)


def _mapping_signature(mappings) -> str:
    ordered = sorted(mappings.poi_id2idx.items(), key=lambda item: item[1])
    text = "\0".join(item[0] for item in ordered)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _mapped_rows(train_df: pd.DataFrame, mappings) -> pd.DataFrame:
    columns = [
        column
        for column in (
            "user_id",
            "POI_id",
            "trajectory_id",
            "UTC_time",
            "day_of_week",
            "norm_in_day_time",
        )
        if column in train_df.columns
    ]
    work = train_df[columns].copy()
    work["user_idx"] = work["user_id"].map(
        lambda value: mappings.user_id2idx.get(_canonical(value), -1)
    )
    work["poi_idx"] = work["POI_id"].map(
        lambda value: mappings.poi_id2idx.get(_canonical(value), -1)
    )
    work = work[(work["user_idx"] >= 0) & (work["poi_idx"] >= 0)].copy()
    work["timestamp"] = pd.to_datetime(work["UTC_time"], utc=True, errors="coerce")
    fallback = pd.Series(np.arange(len(work)), index=work.index, dtype=np.float64)
    timestamp_ns = work["timestamp"].astype("int64", errors="ignore")
    if not np.issubdtype(timestamp_ns.dtype, np.number):
        work["timestamp_order"] = fallback
    else:
        numeric = pd.to_numeric(timestamp_ns, errors="coerce")
        work["timestamp_order"] = numeric.where(numeric.notna(), fallback)
    normalized = pd.to_numeric(
        work.get("norm_in_day_time", pd.Series(0.0, index=work.index)),
        errors="coerce",
    ).fillna(0.0)
    work["hour"] = np.floor(normalized * 24.0).clip(0, 23).astype(int)
    weekday = pd.to_numeric(
        work.get("day_of_week", pd.Series(0, index=work.index)), errors="coerce"
    ).fillna(0)
    work["weekday"] = weekday.clip(0, 6).astype(int)
    return work.sort_values(
        ["trajectory_id", "timestamp_order"], kind="mergesort"
    ).reset_index(drop=True)


def _normalised_sparse(
    rows: Sequence[int],
    columns: Sequence[int],
    values: Sequence[float],
    shape: Tuple[int, int],
) -> torch.Tensor:
    if len(rows) == 0:
        indices = torch.empty((2, 0), dtype=torch.long)
        data = torch.empty(0, dtype=torch.float32)
        return torch.sparse_coo_tensor(indices, data, shape).coalesce()
    counts = Counter()
    for row, column, value in zip(rows, columns, values):
        counts[(int(row), int(column))] += float(value)
    row_array = np.asarray([key[0] for key in counts], dtype=np.int64)
    column_array = np.asarray([key[1] for key in counts], dtype=np.int64)
    value_array = np.asarray([counts[key] for key in counts], dtype=np.float64)
    degree = np.bincount(row_array, weights=value_array, minlength=shape[0])
    value_array = value_array / np.maximum(degree[row_array], 1e-12)
    indices = torch.tensor(np.vstack([row_array, column_array]), dtype=torch.long)
    data = torch.tensor(value_array, dtype=torch.float32)
    return torch.sparse_coo_tensor(indices, data, shape).coalesce()


def _bipartite(pairs, num_left: int, num_right: int):
    counts = Counter((int(left), int(right)) for left, right in pairs)
    rows = [key[0] for key in counts]
    columns = [key[1] for key in counts]
    values = [float(counts[key]) for key in counts]
    return (
        _normalised_sparse(rows, columns, values, (num_left, num_right)),
        _normalised_sparse(columns, rows, values, (num_right, num_left)),
    )


def _training_poi_indices(work: pd.DataFrame, num_pois: int) -> np.ndarray:
    if len(work) == 0:
        return np.arange(num_pois, dtype=np.int64)
    return np.unique(work["poi_idx"].astype(int).to_numpy())


def _build_regions(work: pd.DataFrame, mappings, num_regions: int, seed: int) -> torch.Tensor:
    coordinates = np.asarray(mappings.poi_coords, dtype=np.float64).copy()
    coordinates = np.nan_to_num(coordinates, nan=0.0, posinf=0.0, neginf=0.0)
    if len(coordinates) == 1:
        return torch.zeros(1, dtype=torch.long)
    scaled = coordinates.copy()
    scaled[:, 1] *= np.cos(np.radians(float(np.mean(scaled[:, 0]))))
    train_indices = _training_poi_indices(work, mappings.num_pois)
    train_coordinates = scaled[train_indices]
    unique_count = max(1, len(np.unique(train_coordinates, axis=0)))
    cluster_count = max(1, min(int(num_regions), len(train_indices), unique_count))
    if cluster_count == 1:
        labels = np.zeros(mappings.num_pois, dtype=np.int64)
    else:
        model = KMeans(
            n_clusters=cluster_count,
            random_state=int(seed),
            n_init=10,
            max_iter=300,
        )
        model.fit(train_coordinates)
        labels = model.predict(scaled).astype(np.int64)
    return torch.tensor(labels, dtype=torch.long)


def _dense_prior(rows: np.ndarray, columns: np.ndarray, shape: Tuple[int, int]) -> torch.Tensor:
    matrix = np.zeros(shape, dtype=np.float32)
    if len(rows):
        np.add.at(matrix, (rows.astype(np.int64), columns.astype(np.int64)), 1.0)
    matrix = np.log1p(matrix)
    matrix = matrix / np.maximum(matrix.max(axis=1, keepdims=True), 1e-6)
    return torch.tensor(matrix, dtype=torch.float32)


def _topk_rows(counters, num_rows: int, k: int):
    k = max(1, int(k))
    indices = np.zeros((num_rows, k), dtype=np.int64)
    scores = np.zeros((num_rows, k), dtype=np.float32)
    for row in range(num_rows):
        items = counters.get(row, Counter()).most_common(k)
        if not items:
            indices[row, :] = row
            continue
        normalizer = max(np.log1p(float(items[0][1])), 1e-6)
        for position, (column, count) in enumerate(items):
            indices[row, position] = int(column)
            scores[row, position] = np.log1p(float(count)) / normalizer
        if len(items) < k:
            indices[row, len(items) :] = int(items[0][0])
    return torch.tensor(indices, dtype=torch.long), torch.tensor(scores, dtype=torch.float32)


def _geo_neighbors(mappings, k: int):
    coordinates = np.asarray(mappings.poi_coords, dtype=np.float64).copy()
    coordinates = np.nan_to_num(coordinates, nan=0.0, posinf=0.0, neginf=0.0)
    num_pois = len(coordinates)
    if num_pois == 1:
        return np.zeros((1, 1), dtype=np.int64), np.ones((1, 1), dtype=np.float32)
    scaled = coordinates.copy()
    scaled[:, 1] *= np.cos(np.radians(float(np.mean(scaled[:, 0]))))
    neighbor_count = min(max(2, int(k) + 1), num_pois)
    distances, neighbors = cKDTree(scaled).query(scaled, k=neighbor_count)
    if neighbors.ndim == 1:
        neighbors = neighbors[:, None]
        distances = distances[:, None]
    neighbors = neighbors[:, 1:]
    distances = distances[:, 1:]
    if neighbors.shape[1] == 0:
        neighbors = np.arange(num_pois, dtype=np.int64)[:, None]
        distances = np.zeros((num_pois, 1), dtype=np.float64)
    positive = distances[distances > 0]
    scale = max(float(np.median(positive)) if len(positive) else 1.0, 1e-8)
    scores = np.exp(-distances / scale).astype(np.float32)
    return neighbors.astype(np.int64), scores

from baselines.sthgcn.model import STHGCN


def build_context(train_df: pd.DataFrame, mappings, processed_dir: Path, args) -> Dict:
    fingerprint = hashlib.sha1(
        (
            "sthgcn|%s|%d|%d|%d|%d"
            % (
                _mapping_signature(mappings),
                len(train_df),
                mappings.num_pois,
                int(args.num_regions),
                int(args.seed),
            )
        ).encode("utf-8")
    ).hexdigest()[:16]
    cache_path = processed_dir / (".sthgcn_context_%s.pt" % fingerprint)
    if cache_path.exists() and not args.rebuild_context:
        try:
            return torch.load(cache_path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(cache_path, map_location="cpu")

    work = _mapped_rows(train_df, mappings)
    num_users = mappings.num_users
    num_pois = mappings.num_pois
    poi_region = _build_regions(work, mappings, args.num_regions, args.seed)
    num_regions = int(poi_region.max().item()) + 1

    collab_up, collab_pu = _bipartite(
        zip(work["user_idx"], work["poi_idx"]), num_users, num_pois
    )
    session_ids = {
        trajectory_id: index
        for index, trajectory_id in enumerate(work["trajectory_id"].drop_duplicates())
    }
    session_sp, session_ps = _bipartite(
        (
            (session_ids[trajectory_id], int(poi_idx))
            for trajectory_id, poi_idx in zip(
                work["trajectory_id"], work["poi_idx"]
            )
        ),
        len(session_ids),
        num_pois,
    )
    region_rp, region_pr = _bipartite(
        ((int(poi_region[poi]), poi) for poi in range(num_pois)),
        num_regions,
        num_pois,
    )

    row_regions = poi_region[
        torch.tensor(work["poi_idx"].astype(int).to_numpy(), dtype=torch.long)
    ].tolist()
    bucket_keys = list(
        zip(
            row_regions,
            work["weekday"].astype(int).tolist(),
            (work["hour"].astype(int) // 4).tolist(),
        )
    )
    bucket_ids = {key: index for index, key in enumerate(dict.fromkeys(bucket_keys))}
    st_ep, st_pe = _bipartite(
        (
            (bucket_ids[key], int(poi_idx))
            for key, poi_idx in zip(bucket_keys, work["poi_idx"])
        ),
        len(bucket_ids),
        num_pois,
    )

    transitions = Counter()
    for _, trajectory in work.groupby("trajectory_id", sort=False):
        sequence = trajectory["poi_idx"].astype(int).tolist()
        for source, destination in zip(sequence[:-1], sequence[1:]):
            transitions[(source, destination)] += 1
    transition_rows = [key[0] for key in transitions]
    transition_columns = [key[1] for key in transitions]
    transition_values = [float(transitions[key]) for key in transitions]
    transition = _normalised_sparse(
        transition_rows,
        transition_columns,
        transition_values,
        (num_pois, num_pois),
    )

    context = {
        "session_sp": session_sp,
        "session_ps": session_ps,
        "collab_up": collab_up,
        "collab_pu": collab_pu,
        "st_ep": st_ep,
        "st_pe": st_pe,
        "region_rp": region_rp,
        "region_pr": region_pr,
        "transition": transition,
        "metadata": {
            "fingerprint": fingerprint,
            "source": "training split only",
            "num_users": int(num_users),
            "num_pois": int(num_pois),
            "num_regions": int(num_regions),
            "num_sessions": int(len(session_ids)),
            "num_spatiotemporal_hyperedges": int(len(bucket_ids)),
            "transition_edges": int(len(transitions)),
        },
    }
    torch.save(context, cache_path)
    return context


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="STHGCN Next-POI baseline")
    add_common_args(parser)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--context-dim", type=int, default=24)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--hyper-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--transformer-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--num-regions", type=int, default=64)
    parser.add_argument("--eval-batch", type=int, default=0)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-eval-examples", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--rebuild-context", action="store_true")
    return parser


def make_model(mappings, args, context: Dict) -> STHGCN:
    return STHGCN(
        num_users=mappings.num_users,
        num_pois=mappings.num_pois,
        num_categories=mappings.num_categories,
        context=context,
        embedding_dim=args.embedding_dim,
        context_dim=args.context_dim,
        hidden_dim=args.hidden_dim,
        hyper_layers=args.hyper_layers,
        num_heads=args.num_heads,
        transformer_layers=args.transformer_layers,
        dropout=args.dropout,
    )

def _move_batch(batch: Dict, device: torch.device) -> Dict:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _load_checkpoint(path: Path, device: torch.device) -> Dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def run_experiment(args: argparse.Namespace) -> Dict:
    if args.smoke:
        args.epochs = 1
        args.batch = min(args.batch, 8)
        args.max_train_batches = 1
        args.max_eval_examples = args.max_eval_examples or 16
        args.patience = 1

    set_seed(args.seed)
    device = get_device(prefer_cuda=not args.no_cuda)
    city = args.city.upper()
    run_tag = city + ("_smoke" if args.smoke else "")
    train_df, val_df, test_df, processed_dir = load_processed_splits(city)
    mappings = build_mappings(train_df, val_df, test_df)
    context = build_context(train_df, mappings, processed_dir, args)

    train_dataset = TrajectoryNextPoiDataset(train_df, mappings, max_len=args.max_len)
    val_dataset = TrajectoryNextPoiDataset(val_df, mappings, max_len=args.max_len)
    test_dataset = TrajectoryNextPoiDataset(test_df, mappings, max_len=args.max_len)
    if args.max_eval_examples > 0:
        val_dataset.samples = val_dataset.samples[: args.max_eval_examples]
        test_dataset.samples = test_dataset.samples[: args.max_eval_examples]
    if not train_dataset or not val_dataset or not test_dataset:
        raise RuntimeError(
            "empty split after filtering: train=%d val=%d test=%d"
            % (len(train_dataset), len(val_dataset), len(test_dataset))
        )

    collate = partial(
        collate_fn,
        poi_pad=mappings.num_pois,
        cat_pad=mappings.num_categories,
    )
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    eval_batch = args.eval_batch if args.eval_batch > 0 else args.batch
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch,
        shuffle=True,
        collate_fn=collate,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=eval_batch,
        shuffle=False,
        collate_fn=collate,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch,
        shuffle=False,
        collate_fn=collate,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = make_model(mappings, args, context).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    checkpoint_dir = ensure_dir(PROJECT_ROOT / "checkpoints" / "STHGCN" / run_tag)
    result_dir = ensure_dir(PROJECT_ROOT / "results" / "STHGCN")
    checkpoint_path = Path(checkpoint_dir) / "best.pt"
    result_path = Path(result_dir) / (run_tag + ".json")

    best_score = -float("inf")
    best_validation: Optional[Dict] = None
    best_epoch = 0
    patience_left = args.patience
    history = []
    started = time.time()

    print(
        "model=STHGCN city=%s device=%s users=%d pois=%d train/val/test=%d/%d/%d"
        % (
            city,
            device,
            mappings.num_users,
            mappings.num_pois,
            len(train_dataset),
            len(val_dataset),
            len(test_dataset),
        )
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_main = 0.0
        total_aux = 0.0
        batches = 0
        progress = tqdm(train_loader, desc="STHGCN-%s ep%d" % (city, epoch), leave=False)
        for batch_index, batch in enumerate(progress):
            if args.max_train_batches > 0 and batch_index >= args.max_train_batches:
                break
            batch_on_device = _move_batch(batch, device)
            logits = model(batch_on_device)
            if args.loss == "bpr":
                main_loss = bpr_loss(logits, batch_on_device["target"], args.num_neg)
            else:
                main_loss = ce_loss(logits, batch_on_device["target"])
            auxiliary_loss = model.compute_auxiliary_loss(batch_on_device)
            loss = main_loss + auxiliary_loss
            if not bool(torch.isfinite(loss)):
                optimizer.zero_grad(set_to_none=True)
                continue
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            total_loss += float(loss.item())
            total_main += float(main_loss.item())
            total_aux += float(auxiliary_loss.item())
            batches += 1

        validation = evaluate_model(model, val_loader, device)
        scheduler.step(validation["loss"])
        score = 4.0 * validation["acc@1"] + validation["acc@10"]
        record = {
            "epoch": epoch,
            "train_loss": total_loss / max(batches, 1),
            "main_loss": total_main / max(batches, 1),
            "auxiliary_loss": total_aux / max(batches, 1),
            "validation": validation,
            "selection_score": score,
        }
        history.append(record)
        print(
            "[STHGCN-%s] epoch=%d loss=%.6f main=%.6f aux=%.6f val=[%s loss=%.6f]"
            % (
                city,
                epoch,
                record["train_loss"],
                record["main_loss"],
                record["auxiliary_loss"],
                format_metrics(validation),
                validation["loss"],
            )
        )

        if score > best_score:
            best_score = score
            best_validation = dict(validation)
            best_epoch = epoch
            patience_left = args.patience
            torch.save(
                {
                    "model_name": "STHGCN",
                    "city": city,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "args": vars(args),
                    "val_metrics": validation,
                    "selection_score": score,
                    "mappings": {
                        "user_id2idx": mappings.user_id2idx,
                        "poi_id2idx": mappings.poi_id2idx,
                        "cat_id2idx": mappings.cat_id2idx,
                        "poi_idx2cat_idx": mappings.poi_idx2cat_idx,
                    },
                    "context_metadata": context.get("metadata", {}),
                },
                checkpoint_path,
            )
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("early stopping")
                break

    if best_validation is None or not checkpoint_path.exists():
        raise RuntimeError("STHGCN did not produce a valid checkpoint")
    checkpoint = _load_checkpoint(checkpoint_path, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate_model(model, test_loader, device)

    result = {
        "model": "STHGCN",
        "city": city,
        "setting": "closed-set immediate next-POI recommendation",
        "best_epoch": best_epoch,
        "selection_metric": "4 * val_acc@1 + val_acc@10",
        "best_val": best_validation,
        "test": test_metrics,
        "elapsed_seconds": time.time() - started,
        "checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)),
        "context_metadata": context.get("metadata", {}),
        "args": vars(args),
        "history": history,
    }
    with open(result_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, default=float)
    print("[STHGCN-%s] TEST %s" % (city, format_metrics(test_metrics)))
    print("saved checkpoint -> %s" % checkpoint_path)
    print("saved result -> %s" % result_path)
    return result


def main() -> None:
    run_experiment(build_parser().parse_args())


if __name__ == "__main__":
    main()
