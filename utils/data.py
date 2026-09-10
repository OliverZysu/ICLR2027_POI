"""Shared data utilities for Next POI recommendation."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

from .common import PROJECT_ROOT, ensure_dir


DATASET_RAW = {
    "NYC": PROJECT_ROOT / "datasets" / "Foursquare-NYC" / "dataset_TSMC2014_NYC.csv",
    "TKY": PROJECT_ROOT / "datasets" / "Foursquare-Tokyo" / "dataset_TSMC2014_TKY.csv",
}

PROCESSED_DIR = PROJECT_ROOT / "datasets" / "processed"


def calculate_laplacian_matrix(adj_mat: np.ndarray) -> np.ndarray:
    """Random-walk normalized adjacency with self-loops (hat_rw_normd_lap_mat)."""
    n_vertex = adj_mat.shape[0]
    id_mat = np.identity(n_vertex, dtype=np.float32)
    wid_adj = adj_mat.astype(np.float32) + id_mat
    deg = wid_adj.sum(axis=1, keepdims=True)
    deg = np.maximum(deg, 1e-8)
    return (wid_adj / deg).astype(np.float32)


def _parse_utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce")


def preprocess_foursquare(
    city: str,
    min_checkins: int = 10,
    traj_time_gap_hours: float = 24.0,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    out_dir: Optional[Path] = None,
) -> Path:
    """Preprocess raw Foursquare CSV following GETNext settings.

    - filter users/POIs with < min_checkins
    - split user sequences into trajectories by 24h gap
    - chronological 80/10/10 split by check-in time
    - build trajectory flow graph from train set
    """
    city = city.upper()
    assert city in ("NYC", "TKY")
    raw_path = DATASET_RAW[city]
    out_dir = Path(out_dir) if out_dir else PROCESSED_DIR / city
    ensure_dir(out_dir)

    print(f"[preprocess] loading {raw_path}")
    df = pd.read_csv(raw_path)
    # unify column names
    df = df.rename(
        columns={
            "userId": "user_id",
            "venueId": "POI_id",
            "venueCategoryId": "POI_catid",
            "venueCategory": "POI_catname",
            "timezoneOffset": "timezone",
            "utcTimestamp": "UTC_time",
        }
    )
    df["UTC_time"] = _parse_utc(df["UTC_time"])
    df = df.dropna(subset=["UTC_time", "user_id", "POI_id"])
    df["user_id"] = df["user_id"].astype(int)
    df["POI_id"] = df["POI_id"].astype(str)
    df["POI_catid"] = df["POI_catid"].astype(str)

    # iterative filtering until stable
    while True:
        n0 = len(df)
        poi_cnt = df["POI_id"].value_counts()
        user_cnt = df["user_id"].value_counts()
        keep_poi = poi_cnt[poi_cnt >= min_checkins].index
        keep_user = user_cnt[user_cnt >= min_checkins].index
        df = df[df["POI_id"].isin(keep_poi) & df["user_id"].isin(keep_user)]
        if len(df) == n0:
            break

    # category codes
    cat_ids = sorted(df["POI_catid"].unique())
    cat2code = {c: i for i, c in enumerate(cat_ids)}
    df["POI_catid_code"] = df["POI_catid"].map(cat2code)

    # local time / day features
    df = df.sort_values(["user_id", "UTC_time"]).reset_index(drop=True)
    # timezone is minutes offset from UTC
    df["local_time"] = df["UTC_time"] + pd.to_timedelta(df["timezone"], unit="m")
    df["day_of_week"] = df["local_time"].dt.dayofweek
    minutes = df["local_time"].dt.hour * 60 + df["local_time"].dt.minute
    df["norm_in_day_time"] = minutes / (24 * 60.0)

    # split trajectories by 24h gap per user
    gap = pd.Timedelta(hours=traj_time_gap_hours)
    traj_ids = []
    for uid, udf in df.groupby("user_id", sort=False):
        times = udf["UTC_time"].tolist()
        tid = 1
        prev = None
        for t in times:
            if prev is not None and (t - prev) > gap:
                tid += 1
            traj_ids.append(f"{uid}_{tid}")
            prev = t
    df["trajectory_id"] = traj_ids

    # remove ultra-short trajectories (<2 check-ins => cannot form next-POI pair)
    traj_len = df.groupby("trajectory_id").size()
    df = df[df["trajectory_id"].isin(traj_len[traj_len >= 2].index)].copy()

    # chronological split on global time
    df = df.sort_values("UTC_time").reset_index(drop=True)
    n = len(df)
    t1 = int(n * train_ratio)
    t2 = int(n * (train_ratio + val_ratio))
    train_df = df.iloc[:t1].copy()
    val_df = df.iloc[t1:t2].copy()
    test_df = df.iloc[t2:].copy()

    # drop trajectories that become <2 after split
    for split_df, name in [(train_df, "train"), (val_df, "val"), (test_df, "test")]:
        lens = split_df.groupby("trajectory_id").size()
        keep = lens[lens >= 2].index
        if name == "train":
            train_df = split_df[split_df["trajectory_id"].isin(keep)].copy()
        elif name == "val":
            val_df = split_df[split_df["trajectory_id"].isin(keep)].copy()
        else:
            test_df = split_df[split_df["trajectory_id"].isin(keep)].copy()

    # relative time features (GETNext style)
    def add_relative(split_df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for tid, tdf in split_df.groupby("trajectory_id", sort=False):
            tdf = tdf.sort_values("UTC_time")
            base = tdf["UTC_time"].iloc[0]
            day_shift = (tdf["UTC_time"] - base).dt.total_seconds() / 86400.0
            tdf = tdf.copy()
            tdf["norm_day_shift"] = day_shift
            tdf["norm_relative_time"] = day_shift + tdf["norm_in_day_time"]
            rows.append(tdf)
        return pd.concat(rows, ignore_index=True) if rows else split_df

    train_df = add_relative(train_df)
    val_df = add_relative(val_df)
    test_df = add_relative(test_df)

    cols = [
        "user_id",
        "POI_id",
        "POI_catid",
        "POI_catid_code",
        "POI_catname",
        "latitude",
        "longitude",
        "timezone",
        "UTC_time",
        "local_time",
        "day_of_week",
        "norm_in_day_time",
        "trajectory_id",
        "norm_day_shift",
        "norm_relative_time",
    ]
    train_df[cols].to_csv(out_dir / f"{city}_train.csv", index=False)
    val_df[cols].to_csv(out_dir / f"{city}_val.csv", index=False)
    test_df[cols].to_csv(out_dir / f"{city}_test.csv", index=False)

    print("[preprocess] building trajectory flow graph from train")
    G = build_global_poi_graph(train_df)
    save_graph_csv(G, out_dir)

    meta = {
        "city": city,
        "num_users_train": int(train_df["user_id"].nunique()),
        "num_pois_graph": int(G.number_of_nodes()),
        "num_train": int(len(train_df)),
        "num_val": int(len(val_df)),
        "num_test": int(len(test_df)),
        "min_checkins": min_checkins,
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("[preprocess] done:", meta)
    return out_dir


def build_global_poi_graph(df: pd.DataFrame) -> nx.DiGraph:
    G = nx.DiGraph()
    for _, row in tqdm(df.iterrows(), total=len(df), desc="nodes"):
        node = row["POI_id"]
        if node not in G.nodes:
            G.add_node(
                node,
                checkin_cnt=1,
                poi_catid=row["POI_catid"],
                poi_catid_code=int(row["POI_catid_code"]),
                poi_catname=row["POI_catname"],
                latitude=float(row["latitude"]),
                longitude=float(row["longitude"]),
            )
        else:
            G.nodes[node]["checkin_cnt"] += 1

    previous_poi_id = None
    previous_traj_id = None
    for _, row in tqdm(df.iterrows(), total=len(df), desc="edges"):
        poi_id = row["POI_id"]
        traj_id = row["trajectory_id"]
        if previous_poi_id is None or previous_traj_id != traj_id:
            previous_poi_id = poi_id
            previous_traj_id = traj_id
            continue
        if G.has_edge(previous_poi_id, poi_id):
            G.edges[previous_poi_id, poi_id]["weight"] += 1
        else:
            G.add_edge(previous_poi_id, poi_id, weight=1)
        previous_poi_id = poi_id
        previous_traj_id = traj_id
    return G


def save_graph_csv(G: nx.DiGraph, dst_dir: Path):
    nodelist = list(G.nodes())
    A = nx.adjacency_matrix(G, nodelist=nodelist)
    np.savetxt(dst_dir / "graph_A.csv", A.todense(), delimiter=",")
    with open(dst_dir / "graph_X.csv", "w") as f:
        print(
            "node_name/poi_id,checkin_cnt,poi_catid,poi_catid_code,poi_catname,latitude,longitude",
            file=f,
        )
        for node in nodelist:
            attr = G.nodes[node]
            print(
                f"{node},{attr['checkin_cnt']},{attr['poi_catid']},{attr['poi_catid_code']},"
                f"{attr['poi_catname']},{attr['latitude']},{attr['longitude']}",
                file=f,
            )


def load_processed_splits(city: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Path]:
    city = city.upper()
    d = PROCESSED_DIR / city
    train = pd.read_csv(d / f"{city}_train.csv")
    val = pd.read_csv(d / f"{city}_val.csv")
    test = pd.read_csv(d / f"{city}_test.csv")
    return train, val, test, d


def load_graph(city: str):
    d = PROCESSED_DIR / city.upper()
    npy = d / "graph_A.npy"
    if npy.exists():
        A = np.load(npy).astype(np.float32)
    else:
        A = np.loadtxt(d / "graph_A.csv", delimiter=",").astype(np.float32)
    X_df = pd.read_csv(d / "graph_X.csv")
    return A, X_df, d
