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
| scikit-learn | 1.7.x |
| Transformers | 4.57.x |
| importlib-metadata | below 5 |

Use the shared constraints in `install/v2-constraints.txt` for the reviewed CPU
integration profile. Full platform and scientific qualification is tracked in
[the production policy](../docs/archive/production_release.md).

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

Both ways use conda-forge only (the environment file lists `nodefaults`), so a fresh
miniconda does not stop at the terms of service of Anaconda's default channels, and the
script accepts nothing on your behalf.

## Reproducible setup (the scoring platform)

On Linux x86_64 with Python 3.11 and CUDA 12.4, one command installs every package at the
version of the reference scoring machine:

~~~bash
bash setup.sh --lock
conda activate brainscore-unified
python -m brainscore.doctor     # lists the Python and every package that differs from the lock
~~~

It reads the Python version from the line `# python==3.11.15` of the lock and creates the
environment with `conda create --override-channels -c conda-forge python==3.11.15 pip`.
It installs `unified/install/lock-linux-py311-cu124.txt` (it names the PyTorch CUDA 12.4 index
itself, and also `pip`, `setuptools` and `wheel`, so the environment ends with their locked
versions; the environment fingerprint counts them), then installs the four repositories from their checkouts with `--no-deps`.
On another platform the option stops with a message; the pinned setup above and the bounds of the doctor
apply there. Do not combine the lock with `v2-constraints.txt`: that file is the CPU test
profile. It names the lock's versions (torch and torchvision without the `+cu124` tag, so
the CPU wheel installs).

The doctor compares the Python, `pip`, `setuptools`, `wheel` and every package with the lock, and
prints the environment fingerprint and the math threads; every score records both
(see [Score a model](../docs/guides/score_a_model.md) for when two scores compare bit for bit).

The script refuses to overwrite an existing conda environment. Choose a
different name when needed:

~~~bash
UMI_ENV_NAME=brainscore-unified-test bash setup.sh
~~~

## Manual setup

With the four repositories already checked out side by side, the environment file is
unnecessary; the shared constraints file holds the shared dependencies that move scores
or break imports at their tested versions (not every package; see the reproducible setup):

~~~bash
cd <workspace-root>
conda create -y -n brainscore-unified --override-channels -c conda-forge python=3.11
conda activate brainscore-unified
pip install -c unified/install/v2-constraints.txt -e ./core -e ./vision -e ./language -e "./unified[notebooks,test]"
python -m pip check
python -c "import brainscore"
~~~

Using `environment-unified.yml` directly instead? It must be copied beside the
repositories first — conda resolves its relative editable paths from the file's own
directory:

~~~bash
cd <workspace-root>
cp unified/install/environment-unified.yml .
conda env create -n brainscore-unified -f environment-unified.yml
conda activate brainscore-unified
python -m pip check
python -c "import brainscore"
python -c "from transformers import DynamicCache; assert hasattr(DynamicCache(), 'to_legacy_cache')"
~~~

For an environment created without the notebook extras, install them
from the workspace root:

~~~bash
python -m pip install -c unified/install/v2-constraints.txt -e "unified[notebooks]"
~~~

## Verify the first local result

Lesson 1 uses images and weights committed with the lessons and downloads nothing:

~~~bash
cd unified/notebooks/lessons
jupyter nbconvert --to notebook --execute --stdout 01_your_first_subject.ipynb >/tmp/umi-lesson01.ipynb
~~~

Then follow the [notebook manifest](../notebooks/README.md) or the
[getting-started guide](../docs/README.md).

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
python -m pip install -c unified/install/v2-constraints.txt -e core -e vision -e language -e "unified[notebooks]"
~~~

### Cached results do not reflect a code change

Brain-Score caches scores and activations. Use an isolated
RESULTCACHING_HOME for experiments whose model state or hooks change, and
consult the perturbation caveats in the UMI API cookbook.
