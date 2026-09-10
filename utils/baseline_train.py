"""Unified train/eval harness for FPMC / PLSPL / STGN / STGCN / ST-RNN.

All models consume a shared padded batch dict and return full-candidate scores
of shape ``(B, S, V)``.  Evaluation uses the last valid timestep (Next-POI).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from utils.common import PROJECT_ROOT, ensure_dir, get_device, set_seed
from utils.data import load_processed_splits
from utils.metrics import evaluate_ranking, format_metrics


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--city", type=str, default="NYC", choices=["NYC", "TKY"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--max-len", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--num-neg", type=int, default=10)
    parser.add_argument("--loss", type=str, default="ce", choices=["ce", "bpr"])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    return parser


@dataclass
class IdMappings:
    user_id2idx: Dict[str, int]
    poi_id2idx: Dict[str, int]
    cat_id2idx: Dict[int, int]
    poi_idx2cat_idx: Dict[int, int]
    poi_coords: np.ndarray  # (V, 2) lat, lon

    @property
    def num_users(self) -> int:
        return len(self.user_id2idx)

    @property
    def num_pois(self) -> int:
        return len(self.poi_id2idx)

    @property
    def num_categories(self) -> int:
        return len(self.cat_id2idx)


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(min(1.0, max(0.0, a))))


def build_mappings(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> IdMappings:
    users = sorted({str(u) for u in train_df["user_id"].tolist()})
    user_id2idx = {u: i for i, u in enumerate(users)}

    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    # Prefer train-first POI order for stability, then add unseen from val/test.
    poi_ids: List[str] = []
    seen = set()
    for df in (train_df, val_df, test_df):
        for p in df["POI_id"].astype(str).tolist():
            if p not in seen:
                seen.add(p)
                poi_ids.append(p)
    poi_id2idx = {p: i for i, p in enumerate(poi_ids)}

    cat_codes = sorted({int(c) for c in all_df["POI_catid_code"].tolist()})
    cat_id2idx = {c: i for i, c in enumerate(cat_codes)}

    # representative coordinates / category per POI (last observation wins)
    coords = np.zeros((len(poi_ids), 2), dtype=np.float64)
    poi_idx2cat_idx: Dict[int, int] = {}
    for _, row in all_df.iterrows():
        pid = str(row["POI_id"])
        idx = poi_id2idx[pid]
        coords[idx, 0] = float(row["latitude"])
        coords[idx, 1] = float(row["longitude"])
        poi_idx2cat_idx[idx] = cat_id2idx[int(row["POI_catid_code"])]

    return IdMappings(user_id2idx, poi_id2idx, cat_id2idx, poi_idx2cat_idx, coords)


class TrajectoryNextPoiDataset(Dataset):
    """One sample = one trajectory prefix sequence used to predict next POIs."""

    def __init__(
        self,
        df: pd.DataFrame,
        maps: IdMappings,
        max_len: int = 50,
        require_user_in_train: bool = True,
    ) -> None:
        self.samples: List[dict] = []
        for traj_id, tdf in df.groupby("trajectory_id"):
            uid = str(traj_id).split("_")[0]
            if require_user_in_train and uid not in maps.user_id2idx:
                continue
            tdf = tdf.sort_values("UTC_time") if "UTC_time" in tdf.columns else tdf
            pois, cats, hours, weekdays, lats, lons, times = [], [], [], [], [], [], []
            for _, row in tdf.iterrows():
                pid = str(row["POI_id"])
                if pid not in maps.poi_id2idx:
                    continue
                pois.append(maps.poi_id2idx[pid])
                cats.append(maps.poi_idx2cat_idx[maps.poi_id2idx[pid]])
                # hour in local day; weekday already present
                hours.append(int(float(row["norm_in_day_time"]) * 24) % 24)
                weekdays.append(int(row["day_of_week"]) % 7)
                lats.append(float(row["latitude"]))
                lons.append(float(row["longitude"]))
                times.append(pd.Timestamp(row["UTC_time"]))

            if len(pois) < 2:
                continue

            # keep last max_len+1 check-ins so input length <= max_len
            if len(pois) > max_len + 1:
                pois = pois[-(max_len + 1) :]
                cats = cats[-(max_len + 1) :]
                hours = hours[-(max_len + 1) :]
                weekdays = weekdays[-(max_len + 1) :]
                lats = lats[-(max_len + 1) :]
                lons = lons[-(max_len + 1) :]
                times = times[-(max_len + 1) :]

            # inputs predict next; length S = L-1
            inp_poi = pois[:-1]
            tgt_poi = pois[1:]
            inp_cat = cats[:-1]
            inp_hour = hours[:-1]
            inp_weekday = weekdays[:-1]

            delta_t = [0.0]
            delta_d = [0.0]
            for i in range(1, len(inp_poi)):
                dt_h = max(0.0, (times[i] - times[i - 1]).total_seconds() / 3600.0)
                dd = _haversine_km(lats[i - 1], lons[i - 1], lats[i], lons[i])
                delta_t.append(float(dt_h))
                delta_d.append(float(dd))

            self.samples.append(
                {
                    "traj_id": str(traj_id),
                    "user": maps.user_id2idx[uid],
                    "poi": inp_poi,
                    "category": inp_cat,
                    "hour": inp_hour,
                    "weekday": inp_weekday,
                    "delta_time_h": delta_t,
                    "delta_distance_km": delta_d,
                    "target": tgt_poi,
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]


def _pad_long(seqs: Sequence[Sequence[int]], pad: int) -> torch.Tensor:
    tensors = [torch.tensor(s, dtype=torch.long) for s in seqs]
    return pad_sequence(tensors, batch_first=True, padding_value=pad)


def _pad_float(seqs: Sequence[Sequence[float]], pad: float = 0.0) -> torch.Tensor:
    tensors = [torch.tensor(s, dtype=torch.float) for s in seqs]
    return pad_sequence(tensors, batch_first=True, padding_value=pad)


def collate_fn(batch: List[dict], poi_pad: int, cat_pad: int) -> Dict[str, torch.Tensor]:
    lengths = torch.tensor([len(x["poi"]) for x in batch], dtype=torch.long)
    return {
        "user": torch.tensor([x["user"] for x in batch], dtype=torch.long),
        "poi": _pad_long([x["poi"] for x in batch], poi_pad),
        "category": _pad_long([x["category"] for x in batch], cat_pad),
        "hour": _pad_long([x["hour"] for x in batch], 0),
        "weekday": _pad_long([x["weekday"] for x in batch], 0),
        "delta_time_h": _pad_float([x["delta_time_h"] for x in batch], 0.0),
        "delta_distance_km": _pad_float([x["delta_distance_km"] for x in batch], 0.0),
        "target": _pad_long([x["target"] for x in batch], -1),
        "lengths": lengths,
    }


def _valid_mask(target: torch.Tensor) -> torch.Tensor:
    return target >= 0


def ce_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # logits (B,S,V), target (B,S)
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        target.reshape(-1),
        ignore_index=-1,
    )


def bpr_loss(logits: torch.Tensor, target: torch.Tensor, num_neg: int) -> torch.Tensor:
    mask = _valid_mask(target)
    if mask.sum() == 0:
        return logits.new_tensor(0.0)
    b, s, v = logits.shape
    pos = logits.gather(-1, target.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    neg_idx = torch.randint(0, v, (b, s, num_neg), device=logits.device)
    # avoid accidental positives
    neg_idx = torch.where(neg_idx == target.clamp(min=0).unsqueeze(-1), (neg_idx + 1) % v, neg_idx)
    neg = logits.gather(-1, neg_idx)
    loss = -F.logsigmoid(pos.unsqueeze(-1) - neg).mean(dim=-1)
    return loss[mask].mean()


@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    all_scores, all_targets = [], []
    total_loss, n_batches = 0.0, 0
    for batch in loader:
        batch_t = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()
        }
        logits = model(batch_t)
        target = batch_t["target"]
        lengths = batch_t["lengths"]
        loss = ce_loss(logits, target)
        total_loss += float(loss.item())
        n_batches += 1
        for i, seq_len in enumerate(lengths.tolist()):
            if seq_len <= 0:
                continue
            tgt = int(target[i, seq_len - 1].item())
            if tgt < 0:
                continue
            all_scores.append(logits[i, seq_len - 1].detach().cpu().numpy())
            all_targets.append(tgt)
    metrics = evaluate_ranking(all_scores, all_targets)
    metrics["loss"] = total_loss / max(n_batches, 1)
    return metrics


def train_baseline(
    model_name: str,
    make_model: Callable,
    args: argparse.Namespace,
) -> Dict:
    set_seed(args.seed)
    device = get_device()
    print(f"device={device}, model={model_name}, city={args.city}")

    train_df, val_df, test_df, _ = load_processed_splits(args.city)
    maps = build_mappings(train_df, val_df, test_df)
    print(
        f"users={maps.num_users}, pois={maps.num_pois}, cats={maps.num_categories}, "
        f"max_len={args.max_len}, loss={args.loss}"
    )

    train_ds = TrajectoryNextPoiDataset(train_df, maps, max_len=args.max_len, require_user_in_train=True)
    val_ds = TrajectoryNextPoiDataset(val_df, maps, max_len=args.max_len, require_user_in_train=True)
    test_ds = TrajectoryNextPoiDataset(test_df, maps, max_len=args.max_len, require_user_in_train=True)
    print(f"traj train/val/test: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")

    poi_pad = maps.num_pois
    cat_pad = maps.num_categories
    collate = lambda batch: collate_fn(batch, poi_pad, cat_pad)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        collate_fn=collate,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False, collate_fn=collate, num_workers=args.num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch, shuffle=False, collate_fn=collate, num_workers=args.num_workers
    )

    model = make_model(maps, args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    safe_name = model_name.replace(" ", "_")
    ckpt_dir = ensure_dir(PROJECT_ROOT / "checkpoints" / safe_name / args.city)
    result_dir = ensure_dir(PROJECT_ROOT / "results" / safe_name)

    best_score = -1.0
    best_val: Optional[Dict[str, float]] = None
    patience_left = args.patience
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total_loss, n_batches = 0.0, 0
        for batch in tqdm(train_loader, desc=f"{safe_name}-{args.city} ep{epoch}", leave=False):
            batch_t = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()
            }
            logits = model(batch_t)
            target = batch_t["target"]
            if args.loss == "bpr":
                loss = bpr_loss(logits, target, args.num_neg)
            else:
                loss = ce_loss(logits, target)
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
        val_m = evaluate_model(model, val_loader, device)
        scheduler.step(val_m["loss"])
        # light train metric: reuse val-style last-step ranking on a train subset is expensive;
        # report train loss + val metrics.
        train_m = {"loss": train_loss}
        history.append({"epoch": epoch, "train": train_m, "val": val_m, "sec": time.time() - t0})
        print(
            f"[{safe_name}-{args.city}] epoch {epoch}/{args.epochs} "
            f"train_loss={train_loss:.4f} val[{format_metrics(val_m)} loss={val_m['loss']:.4f}] "
            f"time={time.time()-t0:.1f}s"
        )

        if not np.isfinite(val_m["loss"]):
            print(f"[{safe_name}-{args.city}] non-finite val loss; stop and keep best.")
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
                    "model_name": model_name,
                    "val_metrics": val_m,
                    "num_users": maps.num_users,
                    "num_pois": maps.num_pois,
                    "num_categories": maps.num_categories,
                },
                ckpt_dir / "best.pt",
            )
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("early stopping")
                break

    if best_val is None or not (ckpt_dir / "best.pt").exists():
        raise RuntimeError(f"{safe_name}-{args.city}: no finite checkpoint saved.")

    ckpt = torch.load(ckpt_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    test_m = evaluate_model(model, test_loader, device)
    print(f"[{safe_name}-{args.city}] TEST {format_metrics(test_m)}")

    out = {
        "model": model_name,
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
