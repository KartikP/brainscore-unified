#!/usr/bin/env bash
# Run only on the new, dedicated Ubuntu 24.04 GPU worker.
set -euo pipefail

MIRAGE_RUN_ROOT="${MIRAGE_RUN_ROOT:-/opt/mirage-run}"
export HF_HOME="${MIRAGE_RUN_ROOT}/cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_XET_CACHE="${HF_HOME}/xet"
export TORCH_HOME="${MIRAGE_RUN_ROOT}/cache/torch"
export XDG_CACHE_HOME="${MIRAGE_RUN_ROOT}/cache/xdg"
export PIP_CACHE_DIR="${MIRAGE_RUN_ROOT}/cache/pip"
export UV_CACHE_DIR="${MIRAGE_RUN_ROOT}/cache/uv"
export UV_PYTHON_INSTALL_DIR="${MIRAGE_RUN_ROOT}/python"
export SCRATCHPATH="${MIRAGE_RUN_ROOT}/scratch"
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
mkdir -p "${MIRAGE_RUN_ROOT}" "${SCRATCHPATH}" "${MIRAGE_RUN_ROOT}/logs"

sudo apt-get update
sudo apt-get install -y python3.12-venv ffmpeg git
python3.12 -m venv "${MIRAGE_RUN_ROOT}/bootstrap"
"${MIRAGE_RUN_ROOT}/bootstrap/bin/pip" install 'uv==0.9.7'
UV="${MIRAGE_RUN_ROOT}/bootstrap/bin/uv"
"${UV}" venv --python 3.12 "${MIRAGE_RUN_ROOT}/native"
"${UV}" pip install --python "${MIRAGE_RUN_ROOT}/native/bin/python" \
  'torch==2.7.1' 'torchvision==0.22.1' 'torchaudio==2.7.1' \
  --index-url https://download.pytorch.org/whl/cu128
"${UV}" pip install --python "${MIRAGE_RUN_ROOT}/native/bin/python" \
  'numpy==1.26.4' 'pandas==2.3.3' 'pydantic==2.12.3' 'PyYAML==6.0.3' \
  'safetensors==0.6.2' 'x-transformers==2.11.15' 'h5py==3.15.1' \
  'moviepy==2.2.1' 'tqdm==4.67.1' 'transformers==4.57.6' \
  'huggingface-hub==0.36.0' 'accelerate==1.11.0' 'soundfile==0.13.1' \
  'julius==0.2.7' 'spacy==3.8.7' 'Levenshtein==0.27.3' \
  'opencv-python-headless==4.11.0.86' 'exca==0.4.5'

if [[ ! -d "${MIRAGE_RUN_ROOT}/mirage/.git" ]]; then
  git clone https://github.com/epflneuroailab/mirage.git "${MIRAGE_RUN_ROOT}/mirage"
fi
git -C "${MIRAGE_RUN_ROOT}/mirage" checkout --detach c594e3124e2a9cb7b8a26a750b373e583a7fb297

"${UV}" python install 3.11.14
"${UV}" venv --python 3.11.14 "${MIRAGE_RUN_ROOT}/host"
"${UV}" pip install --python "${MIRAGE_RUN_ROOT}/host/bin/python" \
  'numpy==1.26.4' 'pandas==2.3.3' 'xarray==2022.3.0' 'scipy==1.15.3' \
  'scikit-learn==1.7.2' 'importlib-metadata==4.13.0' 'result-caching==0.1.0' \
  netCDF4 boto3 Pillow psycopg2-binary fire peewee pybtex requests psutil \
  tqdm pytest pytest-timeout

"${MIRAGE_RUN_ROOT}/native/bin/python" - <<'PY'
import torch
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name())
print('CUDA architectures:', torch.cuda.get_arch_list())
value = torch.ones((32, 32), device='cuda', dtype=torch.bfloat16)
assert (value @ value).sum().isfinite()
PY
"${UV}" pip freeze --python "${MIRAGE_RUN_ROOT}/native/bin/python" > "${MIRAGE_RUN_ROOT}/logs/native-freeze.txt"
"${UV}" pip freeze --python "${MIRAGE_RUN_ROOT}/host/bin/python" > "${MIRAGE_RUN_ROOT}/logs/host-freeze.txt"
