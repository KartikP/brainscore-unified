#!/usr/bin/env bash
# Reproducible setup for the isolated `gemma4` env that runs Gemma-4-12B
# (model_type=gemma4_unified, released 2026-06; needs transformers >=5.x, which is
# incompatible with the brainscore main env's pinned transformers 4.57). Discovered
# over ~8 iterations; pinned here so a clean g5 instance reproduces it deterministically.
#
# Key gotchas baked in below:
#  - transformers 5.10 imports torch.float8_e8m0fnu -> needs torch >= 2.7 (NOT 2.6).
#  - `pip install -U torch` grabbed torch 2.12+cu130 -> bitsandbytes has no CUDA-13
#    build (libnvJitLink.so.13 missing). Pin torch 2.8 + cu128 (driver 580 supports it).
#  - torchvision MUST match torch (0.23 for 2.8) or `torchvision::nms does not exist`.
#  - gemma-4-12B (base) is apache-2.0/ungated but has NO chat template; use -it.
#  - -it ships chat_template.jinja (not .json) -> run with HF_HUB_OFFLINE=1 so the
#    processor uses the local file instead of 401-ing on an online .json fetch.
#  - 4-bit (bitsandbytes) quantizes the encoder-free vision embedder's patch_dense,
#    whose weight dtype becomes Byte; the model casts pixels to it -> layernorm crash.
#    FIX: BitsAndBytesConfig(..., llm_int8_skip_modules=['patch_dense','embedding_projection','lm_head']).
set -euo pipefail
CONDA=${CONDA:-/home/ubuntu/miniconda}
REPO=${REPO:-/home/ubuntu/brain-score-unified}

"$CONDA/bin/conda" create -y -n gemma4 python=3.11
GP="$CONDA/envs/gemma4/bin/pip"
"$GP" install "torch==2.8.*" "torchvision==0.23.*" --index-url https://download.pytorch.org/whl/cu128
"$GP" install transformers==5.10.1 accelerate bitsandbytes pillow "numpy<2" safetensors sentencepiece
"$GP" install -e "$REPO/core" -e "$REPO/unified"     # brainscore_core + unified harness/benchmarks
"$GP" install gymnasium minigrid                       # complex embodied games

GPY="$CONDA/envs/gemma4/bin/python"
"$GPY" - <<'PY'
import torch, transformers
assert hasattr(torch, "float8_e8m0fnu"), "torch too old for transformers 5.x"
import bitsandbytes, torchvision.ops, gymnasium, minigrid
from transformers import AutoConfig, AutoModelForImageTextToText
print("torch", torch.__version__, "transformers", transformers.__version__, "cuda", torch.cuda.is_available())
print("gemma4 env OK")
PY

# Download the instruction-tuned weights (apache-2.0, ungated):
#   "$CONDA/envs/gemma4/bin/hf" download google/gemma-4-12B-it
# Run inference with HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 once cached.
echo "gemma4 env ready. Remember: HF_HUB_OFFLINE=1 for cached -it; 4-bit skip-modules per header."
