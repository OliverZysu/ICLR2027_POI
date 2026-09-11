#!/usr/bin/env bash
# Background dual-GPU runner for designed Model1–Model5 × NYC/TKY.
#
# Usage:
#   bash scripts/run_five_designed_models.sh
#   bash scripts/run_five_designed_models.sh --gpus 0,1
#   bash scripts/run_five_designed_models.sh --skip-existing
#   bash scripts/run_five_designed_models.sh --models model1,model2
#   bash scripts/run_five_designed_models.sh --extra-args "--epochs 20 --batch 64"
#
# Logs:
#   logs/five_designed/orchestrator_*.log
#   logs/five_designed/summary.txt
#   logs/five_designed/<model>_<city>_gpu#.log

set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p logs/five_designed

EXTRA=("$@")

TS="$(date +%Y%m%d_%H%M%S)"
ORCH_LOG="logs/five_designed/orchestrator_${TS}.log"
PID_FILE="logs/five_designed/orchestrator.pid"

nohup python3 -u scripts/run_five_designed_models.py "${EXTRA[@]+"${EXTRA[@]}"}" \
  >"$ORCH_LOG" 2>&1 &
echo $! >"$PID_FILE"

echo "started orchestrator pid=$(cat "$PID_FILE")"
echo "orchestrator log: $ROOT/$ORCH_LOG"
echo "per-job logs:     $ROOT/logs/five_designed/"
echo "summary:          $ROOT/logs/five_designed/summary.txt"
echo
echo "monitor:"
echo "  tail -f $ORCH_LOG"
echo "  tail -f logs/five_designed/summary.txt"
echo "  nvidia-smi"
