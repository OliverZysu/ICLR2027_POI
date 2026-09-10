"""Train / evaluate STAN on Next POI recommendation (shared metrics).

Data follows the same processed NYC/TKY splits as GETNext.
Model architecture follows the official STAN implementation.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from baselines.stan.model import HOURS, STAN
from utils.common import ensure_dir, get_device, set_seed
from utils.data import load_graph, load_processed_splits
from utils.metrics import evaluate_ranking, format_metrics


def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * math.asin(math.sqrt(a)) * 6371.0


def build_poi_distance(X_df, poi_id2idx):
    n = len(poi_id2idx)
    coords = np.zeros((n, 2), dtype=np.float64)
    for _, row in X_df.iterrows():
        idx = poi_id2idx[str(row["node_name/poi_id"])]
        coords[idx, 0] = float(row["latitude"])
        coords[idx, 1] = float(row["longitude"])
    # vectorized approx via haversine for all pairs is heavy; use euclidean deg * scale for speed,
    # then refine — for STAN paper they use haversine. Precompute once with broadcasting.
    mat = np.zeros((n, n), dtype=np.float32)
    # chunked haversine
    for i in tqdm(range(n), desc="poi-dist"):
        lat1 = np.radians(coords[i, 0])
        lon1 = np.radians(coords[i, 1])
        lat2 = np.radians(coords[:, 0])
        lon2 = np.radians(coords[:, 1])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        mat[i] = (2 * np.arcsin(np.sqrt(np.clip(a, 0, 1))) * 6371.0).astype(np.float32)
    return mat


class StanTrajDataset(Dataset):
    """Each sample: one trajectory for next-POI (predict last POI given prefix)."""

    def __init__(self, df, poi_id2idx, user_id2idx, max_len=50):
        self.max_len = max_len
        self.samples = []
        for traj_id, tdf in df.groupby("trajectory_id"):
            uid = str(traj_id).split("_")[0]
            if uid not in user_id2idx:
                continue
            tdf = tdf.sort_values("UTC_time") if "UTC_time" in tdf.columns else tdf
            pois, times, lats, lons = [], [], [], []
            for _, row in tdf.iterrows():
                pid = str(row["POI_id"])
                if pid not in poi_id2idx:
                    continue
                pois.append(poi_id2idx[pid] + 1)  # 1-based for STAN padding
                # hour-in-week feature
                if "local_time" in row and pd.notna(row["local_time"]):
                    lt = pd.to_datetime(row["local_time"], utc=True)
                else:
                    lt = pd.to_datetime(row["UTC_time"], utc=True)
                hour = int(lt.dayofweek) * 24 + int(lt.hour) + 1  # 1..168
                times.append(hour)
                lats.append(float(row["latitude"]))
                lons.append(float(row["longitude"]))
            if len(pois) < 2:
                continue
            # truncate keeping the end (most recent)
            if len(pois) > max_len + 1:
                pois = pois[-(max_len + 1) :]
                times = times[-(max_len + 1) :]
                lats = lats[-(max_len + 1) :]
                lons = lons[-(max_len + 1) :]
            user = user_id2idx[uid] + 1
            self.samples.append(
                {
                    "user": user,
                    "pois": pois,
                    "times": times,
                    "lats": lats,
                    "lons": lons,
                }
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def make_interval_mats(pois, times, lats, lons, max_len):
    """Build mat1 (M,M,2) spatial/temporal intervals for trajectory prefix."""
    m = len(pois)
    mat = np.zeros((max_len, max_len, 2), dtype=np.float32)
    for i in range(m):
        for j in range(m):
            mat[i, j, 0] = haversine(lons[i], lats[i], lons[j], lats[j])
            mat[i, j, 1] = abs(times[i] - times[j])
    return mat


def collate_train(batch, max_len, device):
    # returns lists; trainer handles variable mask lengths
    return batch


def sampling_prob(prob, label, num_neg, loc_max):
    """Balanced sampler from official STAN."""
    num_label = prob.shape[0]
    label = label.view(-1)
    init_label = torch.arange(num_label, dtype=torch.long)
    init_prob = torch.zeros(num_label, num_neg + num_label)

    random_ig = random.sample(range(1, loc_max + 1), num_neg)
    # remap: labels are 0-based (poi_idx), while embeddings use 1-based.
    # Our model outputs scores over L = loc_max locations with index 0..L-1 corresponding to poi 1..L
    label_set = set(int(x) for x in label.tolist())
    while any((x - 1) in label_set for x in random_ig):
        random_ig = random.sample(range(1, loc_max + 1), num_neg)

    for k in range(num_label):
        for i in range(num_neg + num_label):
            if i < num_label:
                init_prob[k, i] = prob[k, label[i]]
            else:
                init_prob[k, i] = prob[k, random_ig[i - num_label] - 1]
    return init_prob, init_label


def forward_prefix(model, sample, mat2s, max_len, device, prefix_len):
    """Run STAN on first `prefix_len` check-ins, predict next (0-based poi index)."""
    pois = sample["pois"][:prefix_len]
    times = sample["times"][:prefix_len]
    lats = sample["lats"][:prefix_len]
    lons = sample["lons"][:prefix_len]
    user = sample["user"]

    traj = torch.zeros(1, max_len, 3, dtype=torch.long, device=device)
    for i in range(prefix_len):
        traj[0, i, 0] = user
        traj[0, i, 1] = pois[i]
        traj[0, i, 2] = times[i]

    mat1 = torch.zeros(1, max_len, max_len, 2, dtype=torch.float32, device=device)
    m = make_interval_mats(pois, times, lats, lons, max_len)
    mat1[0] = torch.from_numpy(m)

    # time intervals from each history step to "next" time (target time)
    # For next-POI at step prefix_len, use times of next check-in if available else last+1
    if prefix_len < len(sample["times"]):
        tgt_time = sample["times"][prefix_len]
    else:
        tgt_time = times[-1]
    vec = torch.zeros(1, max_len, dtype=torch.float32, device=device)
    for i in range(prefix_len):
        vec[0, i] = abs(tgt_time - times[i])

    traj_len = torch.tensor([prefix_len], dtype=torch.long, device=device)
    logits = model(traj, mat1, mat2s, vec, traj_len)  # (1, L)
    return logits


def run_eval(model, dataset, mat2s, max_len, device):
    model.eval()
    scores_list, targets = [], []
    with torch.no_grad():
        for sample in tqdm(dataset, desc="eval", leave=False):
            # predict last POI given all but last
            prefix = len(sample["pois"]) - 1
            if prefix < 1:
                continue
            logits = forward_prefix(model, sample, mat2s, max_len, device, prefix)
            scores_list.append(logits[0].detach().cpu().numpy())
            targets.append(sample["pois"][prefix] - 1)  # 0-based
    return evaluate_ranking(scores_list, targets)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", type=str, default="NYC", choices=["NYC", "TKY"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--embed-dim", type=int, default=50)
    parser.add_argument("--max-len", type=int, default=50)
    parser.add_argument("--num-neg", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--max-train-steps-per-traj", type=int, default=8)
    parser.add_argument("--dist-cache", type=str, default="")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"device={device}, city={args.city}")

    train_df, val_df, test_df, _ = load_processed_splits(args.city)
    _, X_df, data_dir = load_graph(args.city)

    poi_ids = X_df["node_name/poi_id"].astype(str).tolist()
    poi_id2idx = {p: i for i, p in enumerate(poi_ids)}
    user_ids = [str(u) for u in sorted(set(train_df["user_id"].tolist()))]
    user_id2idx = {u: i for i, u in enumerate(user_ids)}

    dist_path = Path(args.dist_cache) if args.dist_cache else data_dir / "poi_dist.npy"
    if dist_path.exists():
        mat2s = np.load(dist_path)
        print(f"loaded distance matrix {mat2s.shape} from {dist_path}")
    else:
        mat2s = build_poi_distance(X_df, poi_id2idx)
        np.save(dist_path, mat2s)
        print(f"saved distance matrix -> {dist_path}")
    mat2s_t = torch.from_numpy(mat2s).to(device)

    train_ds = StanTrajDataset(train_df, poi_id2idx, user_id2idx, max_len=args.max_len)
    val_ds = StanTrajDataset(val_df, poi_id2idx, user_id2idx, max_len=args.max_len)
    test_ds = StanTrajDataset(test_df, poi_id2idx, user_id2idx, max_len=args.max_len)
    print(f"traj train/val/test: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")

    # estimate interval ranges from a sample of train
    su = sl = tu = tl = None
    for i, sample in enumerate(train_ds):
        m = len(sample["pois"]) - 1
        if m < 1:
            continue
        mat = make_interval_mats(sample["pois"][:m], sample["times"][:m], sample["lats"][:m], sample["lons"][:m], args.max_len)
        smax, smin = float(mat[:m, :m, 0].max()), float(mat[:m, :m, 0].min())
        tmax, tmin = float(mat[:m, :m, 1].max()), float(mat[:m, :m, 1].min())
        su = smax if su is None else max(su, smax)
        sl = smin if sl is None else min(sl, smin)
        tu = tmax if tu is None else max(tu, tmax)
        tl = tmin if tl is None else min(tl, tmin)
        if i > 500:
            break
    if su is None or su <= sl:
        su, sl = 1.0, 0.0
    if tu is None or tu <= tl:
        tu, tl = 1.0, 0.0
    ex = (su, sl, tu, tl)
    print(f"ex(su,sl,tu,tl)={ex}")

    l_max = len(poi_id2idx)
    u_max = len(user_id2idx)
    model = STAN(
        t_dim=HOURS + 1,
        l_dim=l_max + 1,
        u_dim=u_max + 1,
        embed_dim=args.embed_dim,
        ex=ex,
        max_len=args.max_len,
        device=device,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    ckpt_dir = ensure_dir(ROOT / "checkpoints" / "STAN" / args.city)
    result_dir = ensure_dir(ROOT / "results" / "STAN")
    best_score = -1.0
    best_val = None
    patience_left = args.patience
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        losses = []
        # shuffle indices
        indices = list(range(len(train_ds)))
        random.shuffle(indices)
        for idx in tqdm(indices, desc=f"train-ep{epoch}", leave=False):
            sample = train_ds[idx]
            full = len(sample["pois"])
            if full < 2:
                continue
            # sample a few prefix lengths for speed
            candidates = list(range(1, full))
            if len(candidates) > args.max_train_steps_per_traj:
                candidates = sorted(random.sample(candidates, args.max_train_steps_per_traj))
            for prefix in candidates:
                label = sample["pois"][prefix] - 1  # 0-based
                logits = forward_prefix(model, sample, mat2s_t, args.max_len, device, prefix)
                # balanced sampler
                prob_s, label_s = sampling_prob(logits, torch.tensor([label], device=device), args.num_neg, l_max)
                loss = F.cross_entropy(prob_s.to(device), label_s.to(device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(float(loss.item()))

        val_m = run_eval(model, val_ds, mat2s_t, args.max_len, device)
        train_loss = float(np.mean(losses)) if losses else 0.0
        score = val_m["acc@1"] * 4 + val_m["acc@10"]
        history.append({"epoch": epoch, "train_loss": train_loss, "val": val_m, "sec": time.time() - t0})
        print(
            f"[STAN-{args.city}] epoch {epoch}/{args.epochs} "
            f"loss={train_loss:.3f} val[{format_metrics(val_m)}] time={time.time()-t0:.1f}s"
        )
        if score > best_score:
            best_score = score
            best_val = val_m
            patience_left = args.patience
            torch.save(
                {"model": model.state_dict(), "args": vars(args), "ex": ex, "val_metrics": val_m},
                ckpt_dir / "best.pt",
            )
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("early stopping")
                break

    ckpt = torch.load(ckpt_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    test_m = run_eval(model, test_ds, mat2s_t, args.max_len, device)
    print(f"[STAN-{args.city}] TEST {format_metrics(test_m)}")

    out = {
        "model": "STAN",
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


if __name__ == "__main__":
    main()
