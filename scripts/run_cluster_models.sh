#!/usr/bin/env bash
# Background dual-GPU runner for PCPNet / DualClusterNet / ASPMix × NYC/TKY.
#
# Usage:
#   bash scripts/run_cluster_models.sh
#   bash scripts/run_cluster_models.sh --skip-existing
#   bash scripts/run_cluster_models.sh --models pcpnet,aspmix
#   bash scripts/run_cluster_models.sh --extra-args "--epochs 30 --batch 32"

set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p logs/cluster_models

EXTRA=("$@")
TS="$(date +%Y%m%d_%H%M%S)"
ORCH_LOG="logs/cluster_models/orchestrator_${TS}.log"
PID_FILE="logs/cluster_models/orchestrator.pid"

nohup python3 -u scripts/run_cluster_models.py "${EXTRA[@]+"${EXTRA[@]}"}" \
  >"$ORCH_LOG" 2>&1 &
echo $! >"$PID_FILE"

echo "started orchestrator pid=$(cat "$PID_FILE")"
echo "orchestrator log: $ROOT/$ORCH_LOG"
echo "per-job logs:     $ROOT/logs/cluster_models/"
echo "summary:          $ROOT/logs/cluster_models/summary.txt"
echo
echo "monitor:"
echo "  tail -f $ORCH_LOG"
echo "  tail -f logs/cluster_models/summary.txt"
echo "  nvidia-smi"
