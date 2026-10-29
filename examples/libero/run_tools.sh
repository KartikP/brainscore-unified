#!/usr/bin/env bash
# Tool route: official evaluator -> serve_tool_bridge.py -> serve_tools.py, one fresh
# policy server and bridge per condition, then compare.py against the baseline.
# Baseline means the UMI tool route without tools; run_pair.sh supplies the native reference.
# Tools target action_out_proj unit 0 at denoising iterations 0 and 1, not every iteration.
# Uses only pre-staged inputs (checkout, local checkpoint, prepared interpreters);
# downloads nothing and starts no cloud resources. Run on the host that has them.
#
# Interpreters (required environment variables):
#   UMI_POLICY_PYTHON     pinned OpenPI runtime with the UMI packages importable
#   UMI_BRIDGE_PYTHON     UMI environment with websockets and the pinned openpi-client
#   UMI_EVALUATOR_PYTHON  LIBERO runtime (Python 3.8) for evaluate.py
# Optional: UMI_READY_SECONDS (server readiness bound, default 1200),
#   UMI_POLICY_PORT / UMI_BRIDGE_PORT (default 8000 / 8001). Simulator settings such as
#   MUJOCO_GL are taken from the caller's environment.
set -euo pipefail
usage="Usage: bash run_tools.sh OPENPI_DIR CHECKPOINT_DIR EVIDENCE_DIR SUITE TRIALS_PER_TASK CONDITION_SECONDS [MODE ...]
  MODE: recording and/or ablation (default: both); baseline always runs first.
  CONDITION_SECONDS: wall-clock limit for each evaluator run; it is stopped when exceeded."
fail() { echo "$1" >&2; exit 2; }
[[ $# -ge 6 ]] || fail "$usage"
for name in UMI_POLICY_PYTHON UMI_BRIDGE_PYTHON UMI_EVALUATOR_PYTHON; do
  [[ -n ${!name:-} ]] || fail "Set $name (see the header of this script)"
done
[[ -d $1 && -d $2 ]] || fail "OPENPI_DIR and CHECKPOINT_DIR must be existing local directories"
openpi_dir=$(cd "$1" && pwd)
checkpoint=$(cd "$2" && pwd)
evidence_dir=$3
suite=$4
trials=$5
condition_seconds=$6
shift 6
modes=("$@")
[[ ${#modes[@]} -gt 0 ]] || modes=(recording ablation)
[[ $suite =~ ^libero_(spatial|object|goal|10)$ ]] || fail "Unknown suite: $suite"
[[ $trials =~ ^[0-9]+$ ]] && ((trials >= 1 && trials <= 50)) || fail "TRIALS_PER_TASK must be 1 to 50"
[[ $condition_seconds =~ ^[0-9]+$ ]] && ((condition_seconds >= 1)) || fail "CONDITION_SECONDS must be a positive integer"
for mode in "${modes[@]}"; do
  [[ $mode == recording || $mode == ablation ]] || fail "Unknown mode: $mode"
done
[[ ${#modes[@]} -le 2 ]] || fail "Use each mode only once"
if [[ ${#modes[@]} -eq 2 ]]; then
  [[ ${modes[0]} != "${modes[1]}" ]] || fail "Use each mode only once"
fi
ready_seconds=${UMI_READY_SECONDS:-1200}
[[ $ready_seconds =~ ^[0-9]+$ ]] && ((ready_seconds >= 1)) || fail "UMI_READY_SECONDS must be a positive integer"
policy_port=${UMI_POLICY_PORT:-8000}
bridge_port=${UMI_BRIDGE_PORT:-8001}
script_dir=$(cd "$(dirname "$0")" && pwd)
unified_dir=$(cd "$script_dir/../.." && pwd)
"$UMI_BRIDGE_PYTHON" - "$policy_port" "$bridge_port" <<'PY'
import socket, sys
try:
    ports = [int(value) for value in sys.argv[1:]]
    if len(set(ports)) != 2 or any(not 1 <= port <= 65535 for port in ports):
        raise ValueError
except ValueError:
    print('Use distinct ports between 1 and 65535', file=sys.stderr)
    raise SystemExit(2)
for port in ports:
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', port))
PY
mkdir "$evidence_dir"  # never reuse an evidence directory
evidence_dir=$(cd "$evidence_dir" && pwd)
client_path="$openpi_dir/packages/openpi-client/src"
policy_path="$openpi_dir/src:$client_path:$unified_dir${PYTHONPATH:+:$PYTHONPATH}"
bridge_path="$client_path:$unified_dir${PYTHONPATH:+:$PYTHONPATH}"

pids=()
stop() {  # SIGTERM, then SIGKILL after 30 s; only processes this script started
  local pid=$1 waited=0
  kill -0 "$pid" 2>/dev/null || { wait "$pid" 2>/dev/null || true; return 0; }
  kill -TERM "$pid" 2>/dev/null || true
  while kill -0 "$pid" 2>/dev/null && ((waited < 30)); do sleep 1; waited=$((waited + 1)); done
  kill -KILL "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
}
cleanup() {
  local pid
  for pid in ${pids[@]+"${pids[@]}"}; do stop "$pid"; done
  pids=()
}
trap cleanup EXIT
trap 'exit 130' INT TERM

wait_ready() {  # PORT PID LOG: a plain TCP connection, so no websocket session starts
  "$UMI_BRIDGE_PYTHON" - "$1" "$2" "$ready_seconds" <<'PY' || { echo "See $3" >&2; return 1; }
import os, socket, sys, time
port, pid, limit = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
deadline = time.monotonic() + limit
while time.monotonic() < deadline:
    try:
        os.kill(pid, 0)
    except OSError:
        raise SystemExit('Server exited before it was ready')
    try:
        socket.create_connection(('127.0.0.1', port), timeout=5).close()
        raise SystemExit(0)
    except OSError:
        time.sleep(2)
raise SystemExit('Server was not ready within %g s' % limit)
PY
}

{
  git -C "$openpi_dir" rev-parse HEAD
  git -C "$unified_dir" rev-parse HEAD
  git -C "$unified_dir" status --porcelain
} >"$evidence_dir/revisions.txt" 2>&1 || true
for role in policy:POLICY bridge:BRIDGE evaluator:EVALUATOR; do
  python_var=UMI_${role#*:}_PYTHON
  { "${!python_var}" -c 'import sys; print(sys.executable); print(sys.version)'
    "${!python_var}" -m pip freeze; } >"$evidence_dir/${role%%:*}-environment.txt" 2>&1 || true
done

run_condition() {
  local mode=$1 route=$2 policy_pid bridge_pid eval_pid
  # A fresh policy process per condition restores its initial random state.
  PYTHONPATH="$policy_path" "$UMI_POLICY_PYTHON" "$script_dir/serve_tools.py" \
    --checkpoint "$checkpoint" --port "$policy_port" --steps 0 1 \
    >"$evidence_dir/$mode-policy.log" 2>&1 &
  policy_pid=$!
  pids+=("$policy_pid")
  wait_ready "$policy_port" "$policy_pid" "$evidence_dir/$mode-policy.log"
  PYTHONPATH="$bridge_path" "$UMI_BRIDGE_PYTHON" "$script_dir/serve_tool_bridge.py" \
    --upstream "ws://127.0.0.1:$policy_port" --port "$bridge_port" --mode "$mode" \
    --suite "$suite" --trials-per-task "$trials" --execution-horizon 5 \
    --out "$evidence_dir/$mode-records" >"$evidence_dir/$mode-bridge.log" 2>&1 &
  bridge_pid=$!
  pids+=("$bridge_pid")
  wait_ready "$bridge_port" "$bridge_pid" "$evidence_dir/$mode-bridge.log"
  PYTHONPATH="$client_path" "$UMI_EVALUATOR_PYTHON" "$script_dir/evaluate.py" \
    --openpi "$openpi_dir" --out "$evidence_dir/$mode-eval" --route "$route" \
    --suite "$suite" --trials "$trials" --port "$bridge_port" \
    >"$evidence_dir/$mode-evaluator.log" 2>&1 &
  eval_pid=$!
  pids+=("$eval_pid")
  local start=$SECONDS
  while kill -0 "$eval_pid" 2>/dev/null; do
    if ((SECONDS - start >= condition_seconds)); then
      echo "$mode evaluation exceeded $condition_seconds s; stopping it" >&2
      stop "$eval_pid"
      return 1
    fi
    sleep 2
  done
  wait "$eval_pid" || { echo "$mode evaluation failed; see $evidence_dir/$mode-evaluator.log" >&2; return 1; }
  cleanup
}

run_condition baseline reference
for mode in "${modes[@]}"; do
  run_condition "$mode" umi
  PYTHONPATH="$bridge_path" "$UMI_BRIDGE_PYTHON" "$script_dir/compare.py" \
    --reference "$evidence_dir/baseline-eval" --umi "$evidence_dir/$mode-eval" \
    --reference-experiment "$evidence_dir/baseline-records/connection-1" \
    --umi-experiment "$evidence_dir/$mode-records/connection-1" \
    --out "$evidence_dir/comparison-$mode.json"
done
echo "Finished. Closed-loop outcomes are descriptive; see comparison-*.json in $evidence_dir."
