#!/bin/bash
# Native eval — π0-FAST DROID via stock PiPolicyEvalConfig.
# Anchors against published leaderboard number (22.0% ± 2.6% on MS-Pick).

set -euo pipefail

CHECKPOINT="$HOME/molmo_validation/checkpoints/pi0_fast_droid_jointpos"
BENCHMARK_DIR="$HOME/.cache/molmo-spaces-resources/benchmarks/molmospaces-bench-v1/procthor-10k/FrankaPickDroidMiniBench/FrankaPickDroidMiniBench_json_benchmark_20251231"
OUTPUT_DIR="$HOME/molmo_validation/results/native"
MAX_EPISODES="${1:-100}"  # default 100, override via CLI arg

mkdir -p "$OUTPUT_DIR"

echo "=== Native eval — stock PiPolicyEvalConfig ==="
echo "checkpoint:  $CHECKPOINT"
echo "benchmark:   $BENCHMARK_DIR"
echo "output:      $OUTPUT_DIR"
echo "episodes:    $MAX_EPISODES"
echo ""

source "$HOME/miniconda/etc/profile.d/conda.sh"
conda activate mlspaces

cd "$HOME/molmo_validation/molmospaces"

python molmo_spaces/evaluation/eval_main.py \
    "molmo_spaces.evaluation.configs.evaluation_configs:PiPolicyEvalConfig" \
    --benchmark_dir "$BENCHMARK_DIR" \
    --checkpoint_path "$CHECKPOINT" \
    --task_horizon_steps 500 \
    --output_dir "$OUTPUT_DIR" \
    --max_episodes "$MAX_EPISODES" \
    2>&1 | tee "$OUTPUT_DIR/run.log"

echo ""
echo "=== Native eval done ==="
grep -E "Success rate|Output directory" "$OUTPUT_DIR/run.log" | tail -5
