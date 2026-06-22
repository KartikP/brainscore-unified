#!/bin/bash
# Start the openpi server for π0-FAST DROID.
# Hosts the policy on localhost:8080. Both stock and wrapped evals talk to
# this server. The two evals therefore exercise the SAME π0 inference; only
# the obs/action data path differs.

set -euo pipefail

CHECKPOINT="$HOME/molmo_validation/checkpoints/pi0_fast_droid_jointpos"
LOG="$HOME/molmo_validation/pi_server.log"

echo "=== Starting π0-FAST openpi server ==="
echo "checkpoint:  $CHECKPOINT"
echo "log:         $LOG"
echo "port:        8080 (default)"
echo ""

source "$HOME/miniconda/etc/profile.d/conda.sh"
conda activate openpi

cd "$HOME/molmo_validation/openpi"

# serve_policy.py loads the checkpoint and serves it on port 8080
nohup python scripts/serve_policy.py policy:checkpoint \
    --policy.config="pi0_fast_droid_jointpos" \
    --policy.dir="$CHECKPOINT" \
    > "$LOG" 2>&1 &

PID=$!
echo "$PID" > "$HOME/molmo_validation/pi_server.pid"
echo "started, pid=$PID"
echo ""
echo "tail $LOG to monitor; expect 'Loaded checkpoint' + 'Listening on port 8080'"
echo "stop with: kill \$(cat $HOME/molmo_validation/pi_server.pid)"
