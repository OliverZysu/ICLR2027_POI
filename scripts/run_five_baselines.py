#!/usr/bin/env python3
"""Reproduce Next-POI baseline suites on NYC + TKY using both GPUs.

Default suite (new): MTNet, DCHL, iPCM, K1-POI, STHGCN
Classic suite:       FPMC, PLSPL, STGCN, STGN, ST-RNN

- Continues after individual job failures
- Dual-GPU worker queue
- Suitable for nohup / background execution

Usage:
  nohup python3 -u scripts/run_five_baselines.py --gpus 0,1 \\
      > logs/five_baselines/orchestrator.log 2>&1 &

  # or
  bash scripts/run_five_baselines.sh
  bash scripts/run_five_baselines.sh --suite classic
  bash scripts/run_five_baselines.sh --skip-existing
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs" / "five_baselines"

# (short_name, entry_script, result_dirname)
SUITE_BASELINES: Dict[str, List[Tuple[str, str, str]]] = {
    "new": [
        ("mtnet", "run_mtnet.py", "MTNet"),
        ("dchl", "run_dchl.py", "DCHL"),
        ("ipcm", "run_ipcm.py", "iPCM"),
        ("k1_poi", "run_k1_poi.py", "K1-POI"),
        ("sthgcn", "run_sthgcn.py", "STHGCN"),
    ],
    "classic": [
        ("fpmc", "run_fpmc.py", "FPMC"),
        ("plspl", "run_plspl.py", "PLSPL"),
        ("stgcn", "run_stgcn.py", "STGCN"),
        ("stgn", "run_stgn.py", "STGN"),
        ("strnn", "run_strnn.py", "ST-RNN"),
    ],
}
CITIES = ("NYC", "TKY")


@dataclass
class JobResult:
    baseline: str
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
        f.write(text)
        if not text.endswith("\n"):
            f.write("\n")


def build_jobs(
    baseline_defs: List[Tuple[str, str, str]],
    baselines: Optional[List[str]],
    cities: Optional[List[str]],
) -> List[Tuple[str, str, str, str]]:
    wanted_b = set(baselines) if baselines else None
    wanted_c = set(cities) if cities else None
    jobs = []
    for short, script, result_name in baseline_defs:
        if wanted_b is not None and short not in wanted_b:
            continue
        for city in CITIES:
            if wanted_c is not None and city not in wanted_c:
                continue
            jobs.append((short, script, result_name, city))
    return jobs


def run_one(
    short: str,
    script: str,
    city: str,
    gpu: int,
    extra_args: List[str],
    summary_path: Path,
) -> JobResult:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{short}_{city}_gpu{gpu}.log"
    cmd = [sys.executable, "-u", str(ROOT / script), "--city", city, *extra_args]
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
                cmd,
                cwd=str(ROOT),
                env=env,
                stdout=logf,
                stderr=subprocess.STDOUT,
                check=False,
            )
            rc = int(proc.returncode)
    except Exception:
        rc = -1
        err = traceback.format_exc()
        _append(log_path, err)
    elapsed = time.time() - start
    status = "OK" if rc == 0 else "FAIL"
    msg = f"[{_now()}] {status}   gpu={gpu} {short} {city} rc={rc} sec={elapsed:.1f} log={log_path}"
    if err:
        msg += f" err={err.splitlines()[-1] if err else ''}"
    _append(summary_path, msg)
    print(msg, flush=True)
    return JobResult(short, city, gpu, rc, elapsed, str(log_path), err)


class GpuScheduler:
    """Assign jobs to free GPUs; failures never stop the queue."""

    def __init__(self, gpus: List[int]):
        self.gpus = list(gpus)
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
    all_short = sorted({b[0] for suite in SUITE_BASELINES.values() for b in suite})
    parser = argparse.ArgumentParser(
        description="Dual-GPU runner for 5 baselines × 2 cities"
    )
    parser.add_argument("--gpus", type=str, default="0,1", help="Comma-separated GPU ids")
    parser.add_argument(
        "--suite",
        type=str,
        default="new",
        choices=sorted(SUITE_BASELINES.keys()),
        help="Baseline suite: 'new' (MTNet/DCHL/iPCM/K1-POI/STHGCN) or 'classic'",
    )
    parser.add_argument(
        "--baselines",
        type=str,
        default="",
        help=(
            "Optional comma-separated subset of short names within the suite, e.g. "
            "'mtnet,dchl'. Empty = all in suite. Known: " + ",".join(all_short)
        ),
    )
    parser.add_argument("--cities", type=str, default="NYC,TKY")
    parser.add_argument(
        "--extra-args",
        type=str,
        default="",
        help="Extra CLI args appended to every job, e.g. '--epochs 20 --batch 32'",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a job if results/<MODEL>/<CITY>.json already exists",
    )
    args = parser.parse_args()

    gpus = [int(x) for x in args.gpus.split(",") if x.strip() != ""]
    if not gpus:
        raise SystemExit("No GPUs specified")
    baseline_defs = SUITE_BASELINES[args.suite]
    baselines = [x.strip() for x in args.baselines.split(",") if x.strip()] or None
    cities = [x.strip().upper() for x in args.cities.split(",") if x.strip()]
    extra_args = [tok for tok in args.extra_args.split(" ") if tok]

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = LOG_DIR / "summary.txt"
    results_json = LOG_DIR / f"job_results_{args.suite}.json"
    _append(
        summary_path,
        f"\n===== orchestrator start {_now()} suite={args.suite} gpus={gpus} =====",
    )

    jobs = build_jobs(baseline_defs, baselines, cities)
    if args.skip_existing:
        kept = []
        for short, script, result_name, city in jobs:
            out = ROOT / "results" / result_name / f"{city}.json"
            if out.exists():
                _append(summary_path, f"[{_now()}] SKIP   {short} {city} (exists {out})")
                print(f"[{_now()}] SKIP   {short} {city}", flush=True)
            else:
                kept.append((short, script, result_name, city))
        jobs = kept

    print(
        f"[{_now()}] suite={args.suite} queued {len(jobs)} jobs on GPUs {gpus}",
        flush=True,
    )
    if not jobs:
        print("Nothing to run.", flush=True)
        return

    scheduler = GpuScheduler(gpus)
    results: List[JobResult] = []
    results_lock = threading.Lock()

    def _task(job: Tuple[str, str, str, str]) -> JobResult:
        short, script, _result_name, city = job
        gpu = scheduler.acquire()
        try:
            return run_one(short, script, city, gpu, extra_args, summary_path)
        finally:
            scheduler.release(gpu)

    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [pool.submit(_task, job) for job in jobs]
        for fut in as_completed(futures):
            try:
                res = fut.result()
            except Exception:
                tb = traceback.format_exc()
                _append(summary_path, f"[{_now()}] WORKER_EXC {tb}")
                print(tb, flush=True)
                continue
            with results_lock:
                results.append(res)

    payload = [asdict(r) for r in sorted(results, key=lambda r: (r.baseline, r.city))]
    results_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    n_ok = sum(1 for r in results if r.returncode == 0)
    n_fail = sum(1 for r in results if r.returncode != 0)
    _append(
        summary_path,
        f"[{_now()}] DONE   suite={args.suite} ok={n_ok} fail={n_fail} "
        f"total={len(results)} -> {results_json}",
    )
    print(
        f"[{_now()}] DONE suite={args.suite} ok={n_ok} fail={n_fail}. See {summary_path}",
        flush=True,
    )
    # Always exit 0 so one baseline failure does not kill the outer nohup job.
    sys.exit(0)


if __name__ == "__main__":
    main()
