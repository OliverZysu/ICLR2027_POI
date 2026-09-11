#!/usr/bin/env python3
"""Train one designed model (Model1–Model5) on NYC or TKY.

These models take a POI history ``(B, L)`` and predict next-POI logits ``(B, V)``,
unlike baselines that score every timestep as ``(B, S, V)``.

Usage:
  CUDA_VISIBLE_DEVICES=0 python3 -u scripts/train_designed_model.py --model model1 --city NYC
  CUDA_VISIBLE_DEVICES=1 python3 -u scripts/train_designed_model.py --model model4 --city TKY
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models import Model1, Model2, Model3, Model4, Model5
from utils.baseline_train import IdMappings, build_mappings
from utils.common import PROJECT_ROOT, ensure_dir, get_device, set_seed
from utils.data import load_processed_splits
from utils.metrics import evaluate_ranking, format_metrics

MODEL_REGISTRY = {
    "model1": ("Model1", Model1),
    "model2": ("Model2", Model2),
    "model3": ("Model3", Model3),
    "model4": ("Model4", Model4),
    "model5": ("Model5", Model5),
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train designed Model1–Model5")
    p.add_argument("--model", type=str, required=True, choices=sorted(MODEL_REGISTRY.keys()))
    p.add_argument("--city", type=str, default="NYC", choices=["NYC", "TKY"])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--max-len", type=int, default=50)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--n-pattern", type=int, default=32, help="Model1 pattern count")
    p.add_argument("--n-region", type=int, default=64, help="Model4 region count")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--no-cuda", action="store_true")
    return p


def _region_bounds(coords: np.ndarray) -> Tuple[float, float, float, float]:
    lat_min = float(coords[:, 0].min())
    lat_max = float(coords[:, 0].max())
    lon_min = float(coords[:, 1].min())
    lon_max = float(coords[:, 1].max())
    if lat_max <= lat_min:
        lat_max = lat_min + 1e-3
    if lon_max <= lon_min:
        lon_max = lon_min + 1e-3
    return lat_min, lat_max, lon_min, lon_max


def poi_to_region(
    poi_idx: int,
    coords: np.ndarray,
    n_region: int,
    bounds: Tuple[float, float, float, float],
) -> int:
    """Map POI lat/lon into a square grid of size ~sqrt(n_region)^2."""
    grid = max(1, int(math.sqrt(n_region)))
    lat_min, lat_max, lon_min, lon_max = bounds
    lat, lon = float(coords[poi_idx, 0]), float(coords[poi_idx, 1])
    ri = int((lat - lat_min) / (lat_max - lat_min) * grid)
    cj = int((lon - lon_min) / (lon_max - lon_min) * grid)
    ri = min(max(ri, 0), grid - 1)
    cj = min(max(cj, 0), grid - 1)
    return min(ri * grid + cj, n_region - 1)


class NextPoiHistoryDataset(Dataset):
    """One sample = history POIs -> next POI (last check-in of a trajectory)."""

    def __init__(
        self,
        df: pd.DataFrame,
        maps: IdMappings,
        max_len: int,
        n_region: int,
        require_user_in_train: bool = True,
    ) -> None:
        self.samples: List[dict] = []
        bounds = _region_bounds(maps.poi_coords)
        for traj_id, tdf in df.groupby("trajectory_id"):
            uid = str(traj_id).split("_")[0]
            if require_user_in_train and uid not in maps.user_id2idx:
                continue
            tdf = tdf.sort_values("UTC_time") if "UTC_time" in tdf.columns else tdf
            pois: List[int] = []
            for _, row in tdf.iterrows():
                pid = str(row["POI_id"])
                if pid not in maps.poi_id2idx:
                    continue
                pois.append(maps.poi_id2idx[pid])
            if len(pois) < 2:
                continue
            history = pois[:-1]
            target = pois[-1]
            if len(history) > max_len:
                history = history[-max_len:]
            last_hist = history[-1]
            self.samples.append(
                {
                    "history": history,
                    "target": target,
                    "region": poi_to_region(last_hist, maps.poi_coords, n_region, bounds),
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]


def collate_histories(batch: List[dict], pad_idx: int) -> Dict[str, torch.Tensor]:
    lengths = torch.tensor([len(x["history"]) for x in batch], dtype=torch.long)
    max_l = int(lengths.max().item())
    B = len(batch)
    hist = torch.full((B, max_l), pad_idx, dtype=torch.long)
    # left-pad so the last column is always a real POI (matches encoder design)
    for i, x in enumerate(batch):
        L = len(x["history"])
        hist[i, max_l - L :] = torch.tensor(x["history"], dtype=torch.long)
    return {
        "history": hist,
        "lengths": lengths,
        "target": torch.tensor([x["target"] for x in batch], dtype=torch.long),
        "region": torch.tensor([x["region"] for x in batch], dtype=torch.long),
    }


def make_model(short: str, n_poi: int, args: argparse.Namespace) -> nn.Module:
    name, cls = MODEL_REGISTRY[short]
    kwargs = {"n_poi": n_poi, "dim": args.dim}
    if short == "model1":
        kwargs["n_pattern"] = args.n_pattern
    if short == "model4":
        kwargs["n_region"] = args.n_region
    model = cls(**kwargs)
    model.display_name = name  # type: ignore[attr-defined]
    return model


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> Dict[str, float]:
    model.eval()
    all_scores, all_targets = [], []
    total_loss, n_batches = 0.0, 0
    for batch in loader:
        hist = batch["history"].to(device)
        lengths = batch["lengths"].to(device)
        target = batch["target"].to(device)
        region = batch["region"].to(device)
        logits = model(hist, lengths=lengths, region=region)
        loss = F.cross_entropy(logits, target)
        total_loss += float(loss.item())
        n_batches += 1
        all_scores.append(logits.detach().cpu().numpy())
        all_targets.append(target.detach().cpu().numpy())
    scores = np.concatenate(all_scores, axis=0) if all_scores else np.zeros((0, 1))
    targets = np.concatenate(all_targets, axis=0) if all_targets else np.zeros((0,), dtype=np.int64)
    metrics = evaluate_ranking(list(scores), list(targets))
    metrics["loss"] = total_loss / max(n_batches, 1)
    return metrics


def train_one(args: argparse.Namespace) -> Dict:
    set_seed(args.seed)
    device = get_device(prefer_cuda=not args.no_cuda)
    display_name, _ = MODEL_REGISTRY[args.model]
    print(f"device={device}, model={display_name}, city={args.city}")

    train_df, val_df, test_df, _ = load_processed_splits(args.city)
    maps = build_mappings(train_df, val_df, test_df)
    print(f"users={maps.num_users}, pois={maps.num_pois}, max_len={args.max_len}")

    train_ds = NextPoiHistoryDataset(
        train_df, maps, args.max_len, args.n_region, require_user_in_train=True
    )
    val_ds = NextPoiHistoryDataset(
        val_df, maps, args.max_len, args.n_region, require_user_in_train=True
    )
    test_ds = NextPoiHistoryDataset(
        test_df, maps, args.max_len, args.n_region, require_user_in_train=True
    )
    print(f"samples train/val/test: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")
    if len(train_ds) == 0 or len(val_ds) == 0 or len(test_ds) == 0:
        raise RuntimeError("Empty split; check processed data.")

    pad_idx = maps.num_pois
    collate = lambda batch: collate_histories(batch, pad_idx)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        collate_fn=collate,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,
        collate_fn=collate,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch,
        shuffle=False,
        collate_fn=collate,
        num_workers=args.num_workers,
    )

    model = make_model(args.model, maps.num_pois, args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    ckpt_dir = ensure_dir(PROJECT_ROOT / "checkpoints" / display_name / args.city)
    result_dir = ensure_dir(PROJECT_ROOT / "results" / display_name)

    best_score = -1.0
    best_val: Optional[Dict[str, float]] = None
    patience_left = args.patience
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total_loss, n_batches = 0.0, 0
        for batch in tqdm(
            train_loader, desc=f"{display_name}-{args.city} ep{epoch}", leave=False
        ):
            hist = batch["history"].to(device)
            lengths = batch["lengths"].to(device)
            target = batch["target"].to(device)
            region = batch["region"].to(device)
            logits = model(hist, lengths=lengths, region=region)
            loss = F.cross_entropy(logits, target)
            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                continue
            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            total_loss += float(loss.item())
            n_batches += 1

        train_loss = total_loss / max(n_batches, 1)
        val_m = evaluate(model, val_loader, device)
        scheduler.step(val_m["loss"])
        history.append(
            {
                "epoch": epoch,
                "train": {"loss": train_loss},
                "val": val_m,
                "sec": time.time() - t0,
            }
        )
        print(
            f"[{display_name}-{args.city}] epoch {epoch}/{args.epochs} "
            f"train_loss={train_loss:.4f} val[{format_metrics(val_m)} "
            f"loss={val_m['loss']:.4f}] time={time.time() - t0:.1f}s"
        )

        if not np.isfinite(val_m["loss"]):
            print(f"[{display_name}-{args.city}] non-finite val loss; stop.")
            break

        score = val_m["acc@1"] * 4 + val_m["acc@10"]
        if score > best_score:
            best_score = score
            best_val = val_m
            patience_left = args.patience
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "model_name": display_name,
                    "val_metrics": val_m,
                    "num_pois": maps.num_pois,
                },
                ckpt_dir / "best.pt",
            )
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("early stopping")
                break

    if best_val is None or not (ckpt_dir / "best.pt").exists():
        raise RuntimeError(f"{display_name}-{args.city}: no finite checkpoint saved.")

    ckpt = torch.load(ckpt_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    test_m = evaluate(model, test_loader, device)
    print(f"[{display_name}-{args.city}] TEST {format_metrics(test_m)}")

    out = {
        "model": display_name,
        "city": args.city,
        "device": str(device),
        "best_val": best_val,
        "test": test_m,
        "history": history,
        "args": vars(args),
    }
    out_path = result_dir / f"{args.city}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"saved results -> {out_path}")
    print(f"saved checkpoint -> {ckpt_dir / 'best.pt'}")
    return out


def main() -> None:
    args = build_parser().parse_args()
    train_one(args)


if __name__ == "__main__":
    main()
