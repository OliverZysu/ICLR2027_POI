"""Next-POI ranking metrics: Acc@k, Recall@k, MRR.

For single ground-truth next POI, Acc@k == Recall@k == HitRate@k.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Union

import numpy as np
import torch


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def ranking_metrics_from_scores(
    scores: Union[np.ndarray, torch.Tensor],
    target: Union[int, np.ndarray, torch.Tensor],
    ks: Sequence[int] = (1, 5, 10),
) -> Dict[str, float]:
    """Compute metrics for one or a batch of score vectors.

    Args:
        scores: (V,) or (B, V) ranking scores (higher is better)
        target: scalar or (B,) ground-truth POI index
    """
    scores = _to_numpy(scores)
    target = _to_numpy(target)

    if scores.ndim == 1:
        scores = scores[None, :]
        target = np.array([int(target)])
    else:
        target = target.reshape(-1).astype(np.int64)

    batch = scores.shape[0]
    # ranks: 0-based position of target among descending scores
    # argsort descending
    order = np.argsort(-scores, axis=1)
    # position of target
    matches = order == target[:, None]
    # first True along axis=1
    ranks = matches.argmax(axis=1)
    # if somehow missing (should not), rank = V
    missing = ~matches.any(axis=1)
    ranks[missing] = scores.shape[1]

    metrics = {}
    for k in ks:
        hit = (ranks < k).astype(np.float64).mean()
        metrics[f"acc@{k}"] = float(hit)
        if k in (5, 10):
            metrics[f"recall@{k}"] = float(hit)
    metrics["mrr"] = float((1.0 / (ranks + 1)).mean())
    return metrics


def evaluate_ranking(
    all_scores: Iterable,
    all_targets: Iterable,
    ks: Sequence[int] = (1, 5, 10),
) -> Dict[str, float]:
    """Aggregate metrics over many samples.

    each scores: (V,), each target: int
    """
    ranks: List[int] = []
    for scores, target in zip(all_scores, all_targets):
        scores = _to_numpy(scores).reshape(-1)
        target = int(_to_numpy(target).reshape(-1)[0])
        order = np.argsort(-scores)
        # rank of target
        pos = int(np.where(order == target)[0][0])
        ranks.append(pos)

    ranks_arr = np.asarray(ranks, dtype=np.int64)
    out: Dict[str, float] = {}
    for k in ks:
        hit = float((ranks_arr < k).mean()) if len(ranks_arr) else 0.0
        out[f"acc@{k}"] = hit
        if k in (5, 10):
            out[f"recall@{k}"] = hit
    out["mrr"] = float((1.0 / (ranks_arr + 1)).mean()) if len(ranks_arr) else 0.0
    out["num_samples"] = float(len(ranks_arr))
    return out


def format_metrics(metrics: Dict[str, float]) -> str:
    keys = ["acc@1", "acc@5", "acc@10", "recall@5", "recall@10", "mrr"]
    parts = []
    for k in keys:
        if k in metrics:
            parts.append(f"{k}={metrics[k]:.4f}")
    if "num_samples" in metrics:
        parts.append(f"n={int(metrics['num_samples'])}")
    return ", ".join(parts)
