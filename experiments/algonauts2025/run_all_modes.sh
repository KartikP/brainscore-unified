#!/usr/bin/env bash
# Run modes in order of importance: video, audio, text, concat first.
# Banded only if concat doesn't reach paper baseline.
set -e
source ~/miniconda/etc/profile.d/conda.sh
conda activate bsu

cd /home/ubuntu

echo "=== AUDIO_ONLY ==="
python /tmp/score_algonauts_multimodal.py --mode audio_only \
    --out_json /tmp/algonauts_audio_only.json \
    --alpha_grid 1,10,100,1000,10000,100000,1000000

echo "=== CONCAT ==="
python /tmp/score_algonauts_multimodal.py --mode concat \
    --out_json /tmp/algonauts_concat.json \
    --alpha_grid 1,10,100,1000,10000,100000,1000000

echo "=== ALL DONE ==="
ls -la /tmp/algonauts_*.json
