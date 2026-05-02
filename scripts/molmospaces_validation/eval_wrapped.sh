#!/bin/bash
# Wrapped eval — π0-FAST DROID via UnifiedInterfaceEvalConfig.
# Same checkpoint, same server — but obs/action is routed through our
# EnvironmentStep schema (BrainScoreModel.process(env_step) + action_fn).
# Tests that the unified-interface schema is end-to-end lossless.

set -euo pipefail

CHECKPOINT="$HOME/molmo_validation/checkpoints/pi0_fast_droid_jointpos"
BENCHMARK_DIR="$HOME/.cache/molmo-spaces-resources/benchmarks/molmospaces-bench-v1/procthor-10k/FrankaPickDroidMiniBench/FrankaPickDroidMiniBench_json_benchmark_20251231"
OUTPUT_DIR="$HOME/molmo_validation/results/wrapped"
WRAPPER_DIR="$HOME/molmo_validation/wrapper"
MAX_EPISODES="${1:-100}"

mkdir -p "$OUTPUT_DIR"

echo "=== Wrapped eval — UnifiedInterfaceEvalConfig ==="
echo "checkpoint:  $CHECKPOINT"
echo "benchmark:   $BENCHMARK_DIR"
echo "output:      $OUTPUT_DIR"
echo "wrapper dir: $WRAPPER_DIR"
echo "episodes:    $MAX_EPISODES"
echo ""

source "$HOME/miniconda/etc/profile.d/conda.sh"
conda activate mlspaces

cd "$HOME/molmo_validation/molmospaces"

# PYTHONPATH points to wrapper_adapter.py + eval_configs.py
export PYTHONPATH="$WRAPPER_DIR:${PYTHONPATH:-}"

python molmo_spaces/evaluation/eval_main.py \
    "eval_configs:UnifiedInterfaceEvalConfig" \
    --benchmark_dir "$BENCHMARK_DIR" \
    --checkpoint_path "$CHECKPOINT" \
    --task_horizon_steps 500 \
    --output_dir "$OUTPUT_DIR" \
    --max_episodes "$MAX_EPISODES" \
    2>&1 | tee "$OUTPUT_DIR/run.log"

echo ""
echo "=== Wrapped eval done ==="
grep -E "Success rate|Output directory" "$OUTPUT_DIR/run.log" | tail -5
