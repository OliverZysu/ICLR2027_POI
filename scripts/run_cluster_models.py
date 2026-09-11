#!/usr/bin/env python3
"""Dual-GPU orchestrator for PCPNet / DualClusterNet / ASPMix × NYC/TKY."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import shlex
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs" / "cluster_models"
ENTRY = "scripts/train_cluster_model.py"

MODELS: List[Tuple[str, str]] = [
    ("pcpnet", "PCPNet"),
    ("dualcluster", "DualClusterNet"),
    ("aspmix", "ASPMix"),
]
CITIES = ("NYC", "TKY")


@dataclass
class JobResult:
    model: str
    city: str
    gpu: int
    returncode: int
    seconds: float
    log_path: str
    error: str = ""


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _append(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(text if text.endswith("\n") else text + "\n")


def build_jobs(models: Optional[List[str]], cities: Optional[List[str]]) -> List[Tuple[str, str, str]]:
    wanted_m = set(models) if models else None
    wanted_c = set(cities) if cities else None
    jobs = []
    for short, result_name in MODELS:
        if wanted_m is not None and short not in wanted_m:
            continue
        for city in CITIES:
            if wanted_c is not None and city not in wanted_c:
                continue
            jobs.append((short, result_name, city))
    return jobs


def run_one(short: str, city: str, gpu: int, extra_args: List[str], summary_path: Path) -> JobResult:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{short}_{city}_gpu{gpu}.log"
    cmd = [sys.executable, "-u", str(ROOT / ENTRY), "--model", short, "--city", city, *extra_args]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONUNBUFFERED"] = "1"
    start = time.time()
    _append(summary_path, f"[{_now()}] START  gpu={gpu} {short} {city} cmd={' '.join(cmd)}")
    print(f"[{_now()}] START  gpu={gpu} {short} {city}", flush=True)
    err = ""
    try:
        with log_path.open("w", encoding="utf-8") as logf:
            logf.write(f"# cmd: CUDA_VISIBLE_DEVICES={gpu} {' '.join(cmd)}\n")
            logf.flush()
            proc = subprocess.run(
                cmd, cwd=str(ROOT), env=env, stdout=logf, stderr=subprocess.STDOUT, check=False
            )
            rc = int(proc.returncode)
    except Exception:
        rc = -1
        err = traceback.format_exc()
        _append(log_path, err)
    elapsed = time.time() - start
    status = "OK" if rc == 0 else "FAIL"
    msg = f"[{_now()}] {status}   gpu={gpu} {short} {city} rc={rc} sec={elapsed:.1f} log={log_path}"
    _append(summary_path, msg)
    print(msg, flush=True)
    return JobResult(short, city, gpu, rc, elapsed, str(log_path), err)


class GpuScheduler:
    def __init__(self, gpus: List[int]):
        self._lock = threading.Lock()
        self._free = list(gpus)

    def acquire(self) -> int:
        while True:
            with self._lock:
                if self._free:
                    return self._free.pop(0)
            time.sleep(0.2)

    def release(self, gpu: int) -> None:
        with self._lock:
            self._free.append(gpu)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=str, default="0,1")
    parser.add_argument("--models", type=str, default="")
    parser.add_argument("--cities", type=str, default="NYC,TKY")
    parser.add_argument("--extra-args", type=str, default="")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    gpus = [int(x) for x in args.gpus.split(",") if x.strip() != ""]
    models = [x.strip() for x in args.models.split(",") if x.strip()] or None
    cities = [x.strip().upper() for x in args.cities.split(",") if x.strip()]
    extra_args = shlex.split(args.extra_args)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = LOG_DIR / "summary.txt"
    results_json = LOG_DIR / "job_results.json"
    _append(summary_path, f"\n===== orchestrator start {_now()} gpus={gpus} =====")

    jobs = build_jobs(models, cities)
    if args.skip_existing:
        kept = []
        for short, result_name, city in jobs:
            out = ROOT / "results" / result_name / f"{city}.json"
            if out.exists():
                _append(summary_path, f"[{_now()}] SKIP   {short} {city}")
                print(f"[{_now()}] SKIP   {short} {city}", flush=True)
            else:
                kept.append((short, result_name, city))
        jobs = kept

    print(f"[{_now()}] queued {len(jobs)} jobs on GPUs {gpus}", flush=True)
    if not jobs:
        print("Nothing to run.", flush=True)
        return

    scheduler = GpuScheduler(gpus)
    results: List[JobResult] = []
    lock = threading.Lock()

    def _task(job: Tuple[str, str, str]) -> JobResult:
        short, _, city = job
        gpu = scheduler.acquire()
        try:
            return run_one(short, city, gpu, extra_args, summary_path)
        finally:
            scheduler.release(gpu)

    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futs = [pool.submit(_task, j) for j in jobs]
        for fut in as_completed(futs):
            try:
                res = fut.result()
            except Exception:
                tb = traceback.format_exc()
                _append(summary_path, f"[{_now()}] WORKER_EXC {tb}")
                continue
            with lock:
                results.append(res)

    results_json.write_text(
        json.dumps([asdict(r) for r in sorted(results, key=lambda r: (r.model, r.city))], indent=2),
        encoding="utf-8",
    )
    n_ok = sum(1 for r in results if r.returncode == 0)
    n_fail = sum(1 for r in results if r.returncode != 0)
    _append(summary_path, f"[{_now()}] DONE   ok={n_ok} fail={n_fail} total={len(results)}")
    print(f"[{_now()}] DONE ok={n_ok} fail={n_fail}", flush=True)
    sys.exit(1 if n_fail or len(results) != len(jobs) else 0)


if __name__ == "__main__":
    main()
