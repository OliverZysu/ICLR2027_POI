#!/usr/bin/env python3
"""One worker per GPU, validation-only search, immutable run IDs, honest exit codes."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import itertools
import json
import os
from pathlib import Path
import queue
import shlex
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_dualcluster_v2 import atomic_json, digest, paths_for, parser as train_parser


def arguments(config):
    out = []
    for key, value in config.items():
        opt = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                out.append(opt)
        elif value is not None:
            out += [opt, json.dumps(value) if isinstance(value, (dict, list)) else str(value)]
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpus", default="0,1", help="Physical GPU IDs, or cpu for engineering tests")
    p.add_argument("--stage", choices=["pilot", "ablation", "tune", "fair_lr", "diagnostic", "confirm", "test"], default="pilot")
    p.add_argument("--config", default=str(ROOT / "configs/dualcluster_experiments.json"))
    p.add_argument("--selection", default="", help="Validation-only frozen selection JSON for confirm/test")
    p.add_argument("--cities", default="NYC,TKY")
    p.add_argument("--seeds", default="42")
    p.add_argument("--experiments", default="", help="Optional subset of experiment names")
    p.add_argument("--extra-args", default="", help="Training options; cannot override selection/protocol/experiment")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    gpus = [x.strip() for x in args.gpus.split(",") if x.strip()]
    if not gpus or len(gpus) != len(set(gpus)) or any(x != "cpu" and not x.isdigit() for x in gpus):
        p.error("Provide distinct nonnegative GPU IDs, or cpu")
    cities = [x.strip().upper() for x in args.cities.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    if not cities or not seeds or not set(cities) <= {"NYC", "TKY"}:
        p.error("Nonempty valid cities/seeds are required")
    cfg = json.loads(Path(args.config).read_text())
    if args.stage in ("confirm", "test"):
        if not args.selection:
            p.error("confirm/test require a frozen --selection; test is never a tuning stage")
        locked = json.loads(Path(args.selection).read_text())
        signature = locked.pop("selection_sha256", "")
        if digest(locked) != signature:
            p.error("Frozen selection was modified")
        if not set(cities) <= set(locked["cities"]):
            p.error("Cities differ from the frozen selection")
        configs = locked["experiments"]
    else:
        configs = [{**cfg["defaults"], **exp} for exp in cfg["stages"][args.stage]]
    wanted = {x for x in args.experiments.split(",") if x}
    if wanted:
        missing = wanted - {x["experiment"] for x in configs}
        if missing:
            p.error(f"Unknown experiments: {missing}")
        configs = [c for c in configs if c["experiment"] in wanted]
    extra = shlex.split(args.extra_args)
    reserved = {"--mode", "--variant", "--baseline", "--baseline-args", "--experiment", "--protocol", "--city", "--seed"}
    if any(x.split("=")[0] in reserved for x in extra):
        p.error("extra-args cannot override frozen model/protocol/identity; define a new config instead")
    if args.stage in ("confirm", "test"):
        allowed = {"--data-dir", "--output-root", "--num-workers", "--eval-batch", "--cpu-threads", "--resume"}
        if any(x.startswith("--") and x.split("=")[0] not in allowed for x in extra):
            p.error("Frozen runs accept runtime options only, not hyperparameter changes")
    jobs = []
    checked_protocols = set()
    for c, city, seed in itertools.product(configs, cities, seeds):
        call_config = {k: v for k, v in c.items() if k != "_selection_metadata"}
        call_config.update(city=city, seed=seed, mode="test" if args.stage == "test" else "fit")
        argv = arguments(call_config) + extra
        parsed = train_parser().parse_args(argv)
        if args.stage in ("confirm", "test"):
            from scripts.train_dualcluster_v2 import prepare_data, code_fingerprint
            if code_fingerprint() != locked["code_sha256"]:
                raise ValueError("Model code changed after validation selection; do not silently unfreeze the experiment")
            protocol_key = (city, parsed.data_dir, parsed.protocol, parsed.max_len, parsed.min_history, parsed.smoke)
            if protocol_key not in checked_protocols:
                current = prepare_data(parsed)[-1]
                if current["protocol_sha256"] != locked["protocol_sha256_by_city"][city]:
                    raise ValueError(f"{city}: data/query protocol changed after validation selection")
                checked_protocols.add(protocol_key)
        job_paths = paths_for(parsed)
        result = job_paths["result"] / "metrics.json"
        if args.skip_existing and result.exists():
            stored = json.loads(result.read_text())
            complete = stored.get("test") is not None if args.stage == "test" else stored.get("fit_complete") is True
            # Do not skip a different hyperparameter run just because a path exists.
            from scripts.train_dualcluster_v2 import config_signature
            oldargs = argparse.Namespace(**stored["args"])
            if args.stage != "test" and (config_signature(oldargs) != config_signature(parsed) or oldargs.epochs != parsed.epochs):
                raise ValueError(f"Existing run config differs: {result}")
            if complete:
                from scripts.train_dualcluster_v2 import prepare_data, code_fingerprint
                current_protocol = prepare_data(parsed)[-1]
                if current_protocol["protocol_sha256"] != stored["protocol"]["protocol_sha256"]:
                    raise ValueError(f"Existing completed run has different data/protocol: {result}")
                if code_fingerprint() != stored["code_sha256"]:
                    raise ValueError(f"Existing completed run has different model code: {result}")
                print("SKIP completed", c["experiment"], city, seed, flush=True)
                continue
        jobs.append(dict(experiment=c["experiment"], city=city, seed=seed, argv=argv,
                         launcher_log=str(job_paths["log"] / "launcher.log")))
    if args.dry_run:
        for i, job in enumerate(jobs):
            gpu = gpus[i % len(gpus)]
            device = "cpu" if gpu == "cpu" else "cuda"
            print(f"CUDA_VISIBLE_DEVICES={gpu} " + shlex.join([sys.executable, "-u", "main.py", *job["argv"], "--device", device]))
        print(f"DRY RUN: {len(jobs)} jobs; no processes launched")
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"_{os.getpid()}"
    manifest = ROOT / "logs" / "DualClusterV2" / f"queue_{args.stage}_{stamp}.json"
    records, lock = [], threading.Lock()
    work = queue.Queue()
    for job in jobs:
        work.put(job)
    def worker(gpu):
        while True:
            try:
                job = work.get_nowait()
            except queue.Empty:
                break
            path = Path(job["launcher_log"]); path.parent.mkdir(parents=True, exist_ok=True)
            env = os.environ.copy()
            if gpu != "cpu":
                env["CUDA_VISIBLE_DEVICES"] = gpu
            env["PYTHONUNBUFFERED"] = "1"
            device = "cpu" if gpu == "cpu" else "cuda"
            cmd = [sys.executable, "-u", str(ROOT / "main.py"), *job["argv"], "--device", device]
            print("START", gpu, job["experiment"], job["city"], job["seed"], flush=True)
            start = time.time()
            try:
                with path.open("a", encoding="utf-8") as log:
                    log.write("\nCOMMAND: " + shlex.join(cmd) + "\n"); log.flush()
                    proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
                    rc, error = proc.returncode, ""
            except Exception as exc:
                rc, error = -1, repr(exc)
            record = dict(job, gpu=gpu, returncode=rc, error=error, seconds=time.time() - start)
            with lock:
                records.append(record)
                atomic_json(dict(stage=args.stage, jobs=records, queued=len(jobs)), manifest)
            print("OK" if rc == 0 else "FAIL", gpu, job["experiment"], job["city"], f"rc={rc}", flush=True)
            work.task_done()
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        list(pool.map(worker, gpus))
    fail = sum(r["returncode"] != 0 for r in records)
    print(f"COMPLETE success={len(records)-fail} failure={fail} manifest={manifest}", flush=True)
    if fail or len(records) != len(jobs):
        raise SystemExit(1)

if __name__ == "__main__":
    main()
