#!/usr/bin/env bash
# Run on a dedicated Linux/NVIDIA host after building the three images.
set -euo pipefail
if [[ $# != 4 ]]; then
  echo "Usage: bash run_pair.sh OPENPI_DIR EVIDENCE_DIR SUITE TRIALS_PER_TASK" >&2
  exit 2
fi
openpi_dir=$(cd "$1" && pwd)
evidence_dir=$2
suite=$3
trials=$4
[[ $suite =~ ^libero_(spatial|object|goal|10)$ ]] || exit 2
[[ $trials =~ ^[0-9]+$ ]] && ((trials >= 1 && trials <= 50)) || exit 2
mkdir "$evidence_dir"
evidence_dir=$(cd "$evidence_dir" && pwd)
script_dir=$(cd "$(dirname "$0")" && pwd)
run_id="umi-libero-$(date +%s)-$$"
policy_name="$run_id-policy"
bridge_name="$run_id-bridge"
cache_dir=${OPENPI_DATA_HOME:-"$HOME/.cache/openpi"}
mkdir -p "$cache_dir"
if [[ -n ${UMI_XLA_AUTOTUNE_SOURCE:-} ]]; then
  [[ ${UMI_XLA_AUTOTUNE_CACHE:-0} == 1 ]] || exit 2
  mkdir -p "$evidence_dir/autotune"
  cp "$UMI_XLA_AUTOTUNE_SOURCE" "$evidence_dir/autotune/results.textproto"
fi
python3 - <<'PY'
import socket
for port in (8000, 8001):
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', port))
PY

cleanup() {
  docker stop -t 30 "$bridge_name" >/dev/null 2>&1 || true
  docker logs "$bridge_name" >"$evidence_dir/${phase:-setup}-bridge.log" 2>&1 || true
  docker rm "$bridge_name" >/dev/null 2>&1 || true
  docker stop -t 30 "$policy_name" >/dev/null 2>&1 || true
  docker logs "$policy_name" >"$evidence_dir/${phase:-setup}-policy.log" 2>&1 || true
  docker rm "$policy_name" >/dev/null 2>&1 || true
}
trap cleanup EXIT
trap 'exit 130' INT TERM

wait_port() {
  python3 - "$1" "$2" <<'PY'
import subprocess, sys, time, urllib.request
deadline = time.monotonic() + 1200
while time.monotonic() < deadline:
    running = subprocess.check_output(
        ['docker', 'inspect', '--format', '{{.State.Running}}', sys.argv[2]], text=True).strip()
    if running != 'true':
        raise SystemExit('Server exited before readiness; inspect its saved log')
    try:
        urllib.request.urlopen('http://127.0.0.1:' + sys.argv[1] + '/healthz', timeout=5).close()
        break
    except OSError:
        time.sleep(2)
else:
    raise SystemExit('Server did not become ready within 20 minutes')
PY
}

start_servers() {
  local route=$1
  local xla_flags=""
  if [[ ${UMI_XLA_AUTOTUNE_CACHE:-0} == 1 ]]; then
    mkdir -p "$evidence_dir/autotune"
    if [[ $phase == reference && -z ${UMI_XLA_AUTOTUNE_SOURCE:-} ]]; then
      xla_flags="--xla_gpu_dump_autotune_results_to=/autotune/results.textproto"
    else
      xla_flags="--xla_gpu_load_autotune_results_from=/autotune/results.textproto --xla_gpu_require_complete_aot_autotune_results=true"
    fi
  fi
  printf '%s\n' "$xla_flags" >"$evidence_dir/$phase-xla-flags.txt"
  mkdir -p "$evidence_dir/autotune"
  docker run -d --name "$policy_name" --network host --gpus all \
    -v "$openpi_dir:/app:ro" -v "$cache_dir:/openpi_assets" \
    -v "$evidence_dir/autotune:/autotune" \
    -e OPENPI_DATA_HOME=/openpi_assets -e XLA_FLAGS="$xla_flags" \
    umi-libero-policy bash -c \
    'cd /app && PYTHONPATH=/app/src:/app/packages/openpi-client/src exec /.venv/bin/python scripts/serve_policy.py --env LIBERO --port 8000' >/dev/null
  wait_port 8000 "$policy_name"
  # Content hashes accompany the checkpoint URL, which can change upstream.
  python3 - "$cache_dir/openpi-assets/checkpoints/pi05_libero" "$evidence_dir/$phase-checkpoint.json" <<'PY'
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
files = {}
for path in sorted(root.rglob('*')):
    if path.is_file():
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
                digest.update(block)
        files[str(path.relative_to(root))] = {'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}
if not files:
    raise SystemExit('Checkpoint files missing')
pathlib.Path(sys.argv[2]).write_text(json.dumps(files, indent=2) + '\n')
PY
  docker run -d --name "$bridge_name" --network host \
    -v "$openpi_dir:/openpi:ro" -v "$evidence_dir:/evidence" \
    umi-libero-bridge python examples/libero/serve_bridge.py \
    --route "$route" --out "/evidence/$phase-record" >/dev/null
  wait_port 8001 "$bridge_name"
}

evaluate() {
  docker run --rm --network host --gpus all \
    -e MUJOCO_GL=egl -e MUJOCO_EGL_DEVICE_ID=0 \
    -e NVIDIA_DRIVER_CAPABILITIES=all -e PYOPENGL_PLATFORM=egl \
    -v "$openpi_dir:/app:ro" -v "$script_dir:/umi:ro" -v "$evidence_dir:/evidence" \
    umi-libero-runtime /.venv/bin/python /umi/evaluate.py \
    --openpi /app --out "/evidence/$phase" --route "$phase" \
    --suite "$suite" --trials "$trials" --port 8001
}

nvidia-smi >"$evidence_dir/nvidia-smi.txt"
docker image inspect umi-libero-policy umi-libero-runtime umi-libero-bridge >"$evidence_dir/images.json"
docker run --rm umi-libero-bridge python -m pip freeze >"$evidence_dir/bridge-freeze.txt"
docker run --rm umi-libero-policy uv pip freeze --python /.venv/bin/python >"$evidence_dir/policy-freeze.txt"
docker run --rm umi-libero-runtime uv pip freeze --python /.venv/bin/python >"$evidence_dir/runtime-freeze.txt"

phase=reference
start_servers reference
evaluate 2>&1 | tee "$evidence_dir/reference.log"
cleanup

# A fresh policy process restores the initial RNG stream before corpus replay.
phase=replay
start_servers umi
replay_source=${UMI_REPLAY_RECORD:-"$evidence_dir/reference-record"}
printf '%s\n' "$replay_source" >"$evidence_dir/replay-source.txt"
docker run --rm --network host -v "$openpi_dir:/openpi:ro" -v "$evidence_dir:/evidence" \
  -v "$replay_source:/replay-source:ro" \
  -v "$script_dir/replay.py:/replay.py:ro" \
  umi-libero-bridge python /replay.py --record /replay-source \
  --out /evidence/replay.json --atol 0
cleanup

phase=umi
start_servers umi
evaluate 2>&1 | tee "$evidence_dir/umi.log"
cleanup
phase=finished
cmp "$evidence_dir/reference-checkpoint.json" "$evidence_dir/replay-checkpoint.json"
cmp "$evidence_dir/reference-checkpoint.json" "$evidence_dir/umi-checkpoint.json"
docker run --rm -v "$evidence_dir:/evidence" umi-libero-bridge \
  python examples/libero/compare.py --reference /evidence/reference --umi /evidence/umi \
  --out /evidence/comparison.json
echo "Finished. Inspect per-trial reports and replay.json in $evidence_dir."
