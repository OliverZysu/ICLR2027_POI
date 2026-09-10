#!/usr/bin/env bash
# Background dual-GPU runner for 5 baselines × NYC/TKY.
#
# Default suite (new):
#   MTNet / DCHL / iPCM / K1-POI / STHGCN
# Classic suite:
#   FPMC / PLSPL / STGCN / STGN / ST-RNN
#
# Usage:
#   bash scripts/run_five_baselines.sh
#   bash scripts/run_five_baselines.sh --suite new
#   bash scripts/run_five_baselines.sh --suite classic
#   bash scripts/run_five_baselines.sh --gpus 1
#   bash scripts/run_five_baselines.sh --skip-existing
#   bash scripts/run_five_baselines.sh --extra-args "--epochs 20 --batch 32"
#   bash scripts/run_five_baselines.sh --baselines mtnet,dchl
#
# Continues after individual failures. Logs:
#   logs/five_baselines/orchestrator_*.log
#   logs/five_baselines/summary.txt
#   logs/five_baselines/<baseline>_<city>_gpu#.log

set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p logs/five_baselines

EXTRA=("$@")
# If user did not pass --suite, default to the new five baselines.
has_suite=0
for arg in "${EXTRA[@]+"${EXTRA[@]}"}"; do
  if [[ "$arg" == "--suite" || "$arg" == --suite=* ]]; then
    has_suite=1
    break
  fi
done
if [[ $has_suite -eq 0 ]]; then
  EXTRA=(--suite new "${EXTRA[@]+"${EXTRA[@]}"}")
fi

TS="$(date +%Y%m%d_%H%M%S)"
ORCH_LOG="logs/five_baselines/orchestrator_${TS}.log"
PID_FILE="logs/five_baselines/orchestrator.pid"

nohup python3 -u scripts/run_five_baselines.py "${EXTRA[@]}" \
  >"$ORCH_LOG" 2>&1 &
echo $! >"$PID_FILE"

echo "started orchestrator pid=$(cat "$PID_FILE")"
echo "orchestrator log: $ROOT/$ORCH_LOG"
echo "per-job logs:     $ROOT/logs/five_baselines/"
echo "summary:          $ROOT/logs/five_baselines/summary.txt"
echo
echo "monitor:"
echo "  tail -f $ORCH_LOG"
echo "  tail -f logs/five_baselines/summary.txt"
echo "  nvidia-smi"
