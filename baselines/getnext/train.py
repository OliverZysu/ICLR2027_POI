"""Train / evaluate GETNext on Next POI recommendation.

Forward pass aligned with official GETNext (songyangco/GETNext):
- node features use raw checkin_cnt / lat / lon (no extra normalization)
- pad_sequence(..., batch_first=True) then feed Transformer as (B, S, E)
- src_mask size = batch size (official quirk), not trajectory length
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import OneHotEncoder
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from baselines.getnext.model import GETNext
from utils.common import ensure_dir, get_device, set_seed
from utils.data import calculate_laplacian_matrix, load_graph, load_processed_splits
from utils.metrics import evaluate_ranking, format_metrics


def masked_mse_loss(input, target, mask_value=-1):
    mask = target == mask_value
    out = (input[~mask] - target[~mask]) ** 2
    return out.mean()


class TrajectoryDataset(Dataset):
    def __init__(
        self,
        df,
        poi_id2idx,
        user_id2idx,
        time_feature="norm_in_day_time",
        short_thres=2,
        require_user_in_dict=True,
    ):
        self.samples = []
        for traj_id, traj_df in df.groupby("trajectory_id"):
            user_id = str(traj_id).split("_")[0]
            if require_user_in_dict and user_id not in user_id2idx:
                continue
            traj_df = traj_df.sort_values("UTC_time") if "UTC_time" in traj_df.columns else traj_df
            poi_ids = traj_df["POI_id"].tolist()
            times = traj_df[time_feature].tolist()
            poi_idxs = []
            time_vals = []
            for p, t in zip(poi_ids, times):
                if p in poi_id2idx:
                    poi_idxs.append(poi_id2idx[p])
                    time_vals.append(float(t))
            if len(poi_idxs) < short_thres:
                continue
            input_seq = [(poi_idxs[i], time_vals[i]) for i in range(len(poi_idxs) - 1)]
            label_seq = [(poi_idxs[i + 1], time_vals[i + 1]) for i in range(len(poi_idxs) - 1)]
            if len(input_seq) < short_thres:
                continue
            self.samples.append((str(traj_id), input_seq, label_seq))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def build_node_features(X_df, feature_cat="poi_catid"):
    """Official GETNext node features: raw checkin + one-hot cat + lat/lon."""
    raw_X = X_df[["checkin_cnt", feature_cat, "latitude", "longitude"]].to_numpy()
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    one_hot = encoder.fit_transform(raw_X[:, 1].reshape(-1, 1))
    if hasattr(one_hot, "toarray"):
        one_hot = one_hot.toarray()
    num_cats = one_hot.shape[1]
    X = np.zeros((raw_X.shape[0], raw_X.shape[1] - 1 + num_cats), dtype=np.float32)
    X[:, 0] = raw_X[:, 0].astype(np.float32)
    X[:, 1 : num_cats + 1] = one_hot.astype(np.float32)
    X[:, num_cats + 1 :] = raw_X[:, 2:].astype(np.float32)
    return X, encoder, num_cats


def traj_to_embeddings(model, traj_id, input_seq, maps, poi_embeddings, device):
    user_id = traj_id.split("_")[0]
    if user_id not in maps["user_id2idx"]:
        return None
    user_idx = maps["user_id2idx"][user_id]
    user_emb = model.user_embed_model(torch.LongTensor([user_idx]).to(device)).squeeze(0)
    embeds = []
    for poi_idx, t in input_seq:
        poi_emb = poi_embeddings[poi_idx]
        time_emb = model.time_embed_model(torch.tensor([t], dtype=torch.float, device=device)).squeeze(0)
        cat_idx = maps["poi_idx2cat_idx"][poi_idx]
        cat_emb = model.cat_embed_model(torch.LongTensor([cat_idx]).to(device)).squeeze(0)
        fused1 = model.embed_fuse_model1(user_emb, poi_emb)
        fused2 = model.embed_fuse_model2(time_emb, cat_emb)
        embeds.append(torch.cat((fused1, fused2), dim=-1))
    return torch.stack(embeds)


def run_epoch(model, loader, X, A, maps, device, optimizer=None, time_loss_weight=10.0, grad_clip=0.0):
    """One epoch. Matches official GETNext batching / mask convention."""
    train = optimizer is not None
    model.train(train)
    criterion_poi = nn.CrossEntropyLoss(ignore_index=-1)
    criterion_cat = nn.CrossEntropyLoss(ignore_index=-1)

    total_loss = 0.0
    n_batches = 0
    all_scores, all_targets = [], []

    for batch in tqdm(loader, desc="train" if train else "eval", leave=False):
        # Official: mask length = current batch size, applied on (B, S, E) src.
        src_mask = model.seq_model.generate_square_subsequent_mask(len(batch)).to(device)

        poi_embeddings = model.poi_embed_model(X, A)
        attn_map = model.node_attn_model(X, A)

        batch_embeds, batch_lens, batch_in = [], [], []
        batch_y_poi, batch_y_time, batch_y_cat = [], [], []
        for traj_id, input_seq, label_seq in batch:
            emb = traj_to_embeddings(model, traj_id, input_seq, maps, poi_embeddings, device)
            if emb is None:
                continue
            batch_embeds.append(emb)
            batch_lens.append(len(input_seq))
            batch_in.append([x[0] for x in input_seq])
            batch_y_poi.append(torch.LongTensor([x[0] for x in label_seq]))
            batch_y_time.append(torch.FloatTensor([x[1] for x in label_seq]))
            batch_y_cat.append(torch.LongTensor([maps["poi_idx2cat_idx"][x[0]] for x in label_seq]))

        if not batch_embeds:
            continue
        if len(batch_embeds) != len(batch):
            src_mask = model.seq_model.generate_square_subsequent_mask(len(batch_embeds)).to(device)

        # Official: (B, S, E) into TransformerEncoder (no transpose to S-first)
        x = pad_sequence(batch_embeds, batch_first=True, padding_value=-1).to(device)
        y_poi = pad_sequence(batch_y_poi, batch_first=True, padding_value=-1).to(device)
        y_time = pad_sequence(batch_y_time, batch_first=True, padding_value=-1).to(device)
        y_cat = pad_sequence(batch_y_cat, batch_first=True, padding_value=-1).to(device)

        y_pred_poi, y_pred_time, y_pred_cat = model.seq_model(x, src_mask)

        adjusted = y_pred_poi.clone()
        for i, in_seq in enumerate(batch_in):
            for j, p in enumerate(in_seq):
                adjusted[i, j] = y_pred_poi[i, j] + attn_map[p]

        loss_poi = criterion_poi(adjusted.transpose(1, 2), y_poi)
        loss_time = masked_mse_loss(y_pred_time.squeeze(-1), y_time)
        loss_cat = criterion_cat(y_pred_cat.transpose(1, 2), y_cat)
        loss = loss_poi + time_loss_weight * loss_time + loss_cat

        if train:
            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                continue
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

        if not torch.isfinite(loss):
            continue
        total_loss += float(loss.item())
        n_batches += 1

        # Shared protocol: last-timestep next-POI ranking metrics
        for i, seq_len in enumerate(batch_lens):
            scores = adjusted[i, seq_len - 1].detach().cpu().numpy()
            tgt = int(y_poi[i, seq_len - 1].item())
            if tgt < 0:
                continue
            all_scores.append(scores)
            all_targets.append(tgt)

    metrics = evaluate_ranking(all_scores, all_targets)
    metrics["loss"] = total_loss / max(n_batches, 1)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", type=str, default="NYC", choices=["NYC", "TKY"])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--grad-clip", type=float, default=0.0, help="0 disables clipping")
    parser.add_argument("--poi-embed-dim", type=int, default=128)
    parser.add_argument("--user-embed-dim", type=int, default=128)
    parser.add_argument("--time-embed-dim", type=int, default=32)
    parser.add_argument("--cat-embed-dim", type=int, default=32)
    parser.add_argument("--node-attn-nhid", type=int, default=128)
    parser.add_argument("--gcn-nhid", type=int, nargs="+", default=[32, 64])
    parser.add_argument("--gcn-dropout", type=float, default=0.3)
    parser.add_argument("--transformer-nhid", type=int, default=1024)
    parser.add_argument("--transformer-nlayers", type=int, default=2)
    parser.add_argument("--transformer-nhead", type=int, default=2)
    parser.add_argument("--transformer-dropout", type=float, default=0.3)
    parser.add_argument("--time-loss-weight", type=float, default=10.0)
    parser.add_argument("--short-traj-thres", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"device={device}, city={args.city}, align=official-forward")

    train_df, val_df, test_df, data_dir = load_processed_splits(args.city)
    raw_A, X_df, _ = load_graph(args.city)

    feat_col = "poi_catid" if "poi_catid" in X_df.columns else "poi_catid_code"
    X, encoder, num_cats_oh = build_node_features(X_df, feature_cat=feat_col)
    A = calculate_laplacian_matrix(raw_A)

    poi_ids = X_df["node_name/poi_id"].astype(str).tolist()
    poi_id2idx = {p: i for i, p in enumerate(poi_ids)}
    cat_ids = sorted(set(X_df["poi_catid_code"].tolist()))
    cat_id2idx = {c: i for i, c in enumerate(cat_ids)}
    poi_idx2cat_idx = {}
    for _, row in X_df.iterrows():
        poi_idx2cat_idx[poi_id2idx[str(row["node_name/poi_id"])]] = cat_id2idx[row["poi_catid_code"]]

    user_ids = [str(u) for u in sorted(set(train_df["user_id"].tolist()))]
    user_id2idx = {u: i for i, u in enumerate(user_ids)}
    maps = {
        "user_id2idx": user_id2idx,
        "poi_id2idx": poi_id2idx,
        "cat_id2idx": cat_id2idx,
        "poi_idx2cat_idx": poi_idx2cat_idx,
    }

    train_ds = TrajectoryDataset(train_df, poi_id2idx, user_id2idx, short_thres=args.short_traj_thres)
    val_ds = TrajectoryDataset(val_df, poi_id2idx, user_id2idx, short_thres=args.short_traj_thres)
    test_ds = TrajectoryDataset(test_df, poi_id2idx, user_id2idx, short_thres=args.short_traj_thres)
    print(
        f"traj train/val/test: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}, "
        f"pois={len(poi_id2idx)}, users={len(user_id2idx)}, Xabsmax={float(np.abs(X).max()):.1f}"
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True, collate_fn=lambda x: x, num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False, collate_fn=lambda x: x, num_workers=args.num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch, shuffle=False, collate_fn=lambda x: x, num_workers=args.num_workers
    )

    num_cats = len(cat_id2idx)
    model = GETNext(
        gcn_nfeat=X.shape[1],
        gcn_nhid=args.gcn_nhid,
        poi_embed_dim=args.poi_embed_dim,
        gcn_dropout=args.gcn_dropout,
        node_attn_nhid=args.node_attn_nhid,
        num_users=len(user_id2idx),
        user_embed_dim=args.user_embed_dim,
        time_embed_dim=args.time_embed_dim,
        num_cats=num_cats,
        cat_embed_dim=args.cat_embed_dim,
        num_pois=len(poi_id2idx),
        transformer_nhead=args.transformer_nhead,
        transformer_nhid=args.transformer_nhid,
        transformer_nlayers=args.transformer_nlayers,
        transformer_dropout=args.transformer_dropout,
    ).to(device)

    X_t = torch.from_numpy(X).to(device)
    A_t = torch.from_numpy(A).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=5)

    ckpt_dir = ensure_dir(ROOT / "checkpoints" / "GETNext" / args.city)
    result_dir = ensure_dir(ROOT / "results" / "GETNext")
    best_score = -1.0
    best_val = None
    patience_left = args.patience
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_m = run_epoch(
            model, train_loader, X_t, A_t, maps, device, optimizer, args.time_loss_weight, args.grad_clip
        )
        with torch.no_grad():
            val_m = run_epoch(
                model, val_loader, X_t, A_t, maps, device, None, args.time_loss_weight, args.grad_clip
            )

        if not np.isfinite(train_m["loss"]) or not np.isfinite(val_m["loss"]):
            print(
                f"[GETNext-{args.city}] epoch {epoch}: non-finite loss "
                f"(train={train_m['loss']}, val={val_m['loss']}). stop and keep best."
            )
            break

        scheduler.step(val_m["loss"])
        score = val_m["acc@1"] * 4 + val_m["acc@10"]
        history.append({"epoch": epoch, "train": train_m, "val": val_m, "sec": time.time() - t0})
        print(
            f"[GETNext-{args.city}] epoch {epoch}/{args.epochs} "
            f"train[{format_metrics(train_m)} loss={train_m['loss']:.3f}] "
            f"val[{format_metrics(val_m)} loss={val_m['loss']:.3f}] "
            f"time={time.time()-t0:.1f}s"
        )

        if score > best_score:
            best_score = score
            best_val = val_m
            patience_left = args.patience
            torch.save(
                {
                    "model": model.state_dict(),
                    "maps": maps,
                    "args": vars(args),
                    "val_metrics": val_m,
                    "X_shape": list(X.shape),
                    "num_cats": num_cats,
                    "align": "official-forward",
                },
                ckpt_dir / "best.pt",
            )
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("early stopping")
                break

    if best_val is None or not (ckpt_dir / "best.pt").exists():
        raise RuntimeError(f"GETNext-{args.city}: no finite checkpoint saved; training failed.")

    ckpt = torch.load(ckpt_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    with torch.no_grad():
        test_m = run_epoch(
            model, test_loader, X_t, A_t, maps, device, None, args.time_loss_weight, args.grad_clip
        )
    print(f"[GETNext-{args.city}] TEST {format_metrics(test_m)}")

    out = {
        "model": "GETNext",
        "city": args.city,
        "device": str(device),
        "align": "official-forward",
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


if __name__ == "__main__":
    main()
