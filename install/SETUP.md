# Brain-Score UMI local setup

This guide installs the four-repository UMI workspace on a macOS or Linux
laptop.

## Requirements

- Git
- Conda, Miniconda, or Miniforge
- macOS or Linux
- approximately 10 GB of free disk for the environment and small model assets

The supported shared environment is:

| Component | Supported version |
| --- | --- |
| Python | 3.11 |
| NumPy | 1.26.x, always below 2 |
| xarray | 2022.3.0 |
| scikit-learn | 1.5.x |
| Transformers | 4.57.x |
| importlib-metadata | below 5 |

Language features that require scikit-learn 1.6 or newer are not supported in
the shared UMI environment.

## One-command setup

Run from the directory containing setup.sh:

~~~bash
bash setup.sh
conda activate brainscore-unified
~~~

The script:

1. Clones missing core/, vision/, language/, and unified/ repositories from
   their unified-model-interface-v2 branches.
2. Refuses an existing sibling repository on a different branch.
3. Creates brainscore-unified from environment-unified.yml.
4. Installs all four repositories editably, including notebook dependencies.
5. Runs pip check, imports all four packages, and verifies the Transformers
   cache API needed by language scoring.

The script refuses to overwrite an existing conda environment. Choose a
different name when needed:

~~~bash
UMI_ENV_NAME=brainscore-unified-test bash setup.sh
~~~

## Manual setup

Run this **from the workspace root** — the directory holding `core/`, `vision/`,
`language/` and `unified/`. The environment file's editable installs are relative
paths, so from anywhere else (including `unified/install/`, where the file lives) pip
fails with four "path does not exist" errors.

~~~bash
cd <workspace-root>
conda env create -n brainscore-unified -f unified/install/environment-unified.yml
conda activate brainscore-unified
python -m pip check
python -c "import brainscore"
python -c "from transformers import DynamicCache; assert hasattr(DynamicCache(), 'to_legacy_cache')"
~~~

If you created a smaller environment without notebook extras, install them
from the workspace root:

~~~bash
python -m pip install -e "unified[notebooks]"
~~~

## Verify the first local result

Notebook 01 uses deterministic, in-memory data and does not download a model:

~~~bash
cd unified/notebooks
jupyter nbconvert --to notebook --execute --stdout 01_quickstart_layer_mapping.ipynb >/tmp/umi-quickstart.ipynb
~~~

Then follow the [notebook manifest](../notebooks/README.md) or the
[getting-started guide](../docs/getting_started.md).

## Hardware

### CPU

All laptop-safe synthetic notebooks run on CPU. The installed PyTorch wheel
follows the platform default.

### Apple Silicon and MPS

The standard macOS arm64 PyTorch wheel includes MPS support. Confirm it with:

~~~bash
python -c "import torch; print(torch.backends.mps.is_available())"
~~~

Small local demonstrations work on MPS. Large Qwen, BLIP-2, V-JEPA, video, and
multimodal scoring remain outside the laptop-safe path.

### NVIDIA CUDA

Confirm the installed wheel and driver before a GPU run:

~~~bash
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
~~~

If CUDA is unavailable, install the PyTorch wheel matching the machine's CUDA
runtime using the official PyTorch package index, then rerun
python -m pip check.

## Repository layout

~~~text
Brain-Score Unified/
|-- core/
|-- vision/
|-- language/
|-- unified/
|-- environment-unified.yml
|-- setup.sh
|-- README.md
|-- SETUP.md
~~~

All four repositories must remain on unified-model-interface-v2 for this
distribution.

## Troubleshooting

### The conda environment already exists

The setup script will not mutate it. Remove it explicitly or choose another
name with UMI_ENV_NAME.

### A package imports but pip check fails

Treat this as an invalid installation. Recreate the environment from
environment-unified.yml; successful imports alone do not prove compatible
dependency metadata.

### DynamicCache has no to_legacy_cache

Transformers drifted to version 5 or another unsupported version. Recreate the
environment and confirm transformers.__version__ is 4.57.x.

### ModuleNotFoundError: brainscore

Confirm all four editable installs:

~~~bash
python -m pip show brainscore brainscore-core brainscore-vision brainscore-language
~~~

From the workspace root, repair the editable installs with:

~~~bash
python -m pip install -e core -e vision -e language -e "unified[notebooks]"
~~~

### Cached results do not reflect a code change

Brain-Score caches scores and activations. Use an isolated
RESULTCACHING_HOME for experiments whose model state or hooks change, and
consult the perturbation caveats in the UMI API cookbook.
