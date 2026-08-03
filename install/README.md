# Installing Brain-Score UMI (cold machine)

These three files are the bootstrap. `setup.sh` clones all four repositories at the
`unified-model-interface-v2` branch, builds the pinned conda environment, and runs a
consistency check.

## One-liner (no prior checkout)

~~~bash
mkdir brainscore-umi && cd brainscore-umi
base=https://raw.githubusercontent.com/KartikP/brainscore-unified/unified-model-interface-v2/install
curl -fsSLO "$base/setup.sh"
curl -fsSLO "$base/environment-unified.yml"
bash setup.sh          # clones the four repos here + builds the env
conda activate brainscore-unified
~~~

Override the env name with `UMI_ENV_NAME=my-env bash setup.sh` if `brainscore-unified`
already exists.

## If you already have the four repositories

`setup.sh` is for a machine with no checkout. If `core/`, `vision/`, `language/` and
`unified/` are already sitting side by side, create the environment directly — **from the
workspace root**, not from inside any of them:

~~~bash
cd <workspace-root>          # the directory containing all four
conda env create -n brainscore-unified -f unified/install/environment-unified.yml
conda activate brainscore-unified
~~~

**Why the working directory matters.** The environment file ends with relative editable
installs:

~~~yaml
- -e ./core
- -e ./vision
- -e ./language
- -e ./unified[notebooks]
~~~

Those resolve against wherever you invoke conda. Run it from `unified/` — the natural
thing to try, since that is where the file lives — and `./core` points at
`unified/core`, which does not exist. pip then fails with four "path does not exist"
errors that do not mention the real problem.

`setup.sh` avoids this by copying the file to the workspace root before creating the
environment, which is why the bootstrap path has no such caveat.

## Common mistakes

| Symptom | Cause |
| --- | --- |
| `PackagesNotFoundError: install/environment-unified.yml` | `conda create -f` means *force*. Use `conda env create -f`. |
| four `path does not exist` errors on the pip step | run from the workspace root, not from `unified/` |
| `CondaValueError: prefix already exists` | that env name is taken; use `-n` with another name |

## Requirements

- `conda` (miniconda or miniforge). On a fresh miniconda, `setup.sh` accepts the
  Anaconda-channel Terms of Service; miniforge does not require this step.
- `git`, outbound HTTPS to GitHub, ~10 GB free disk.

## After installing

`setup.sh` writes `install-manifest.txt` with the exact commit of each repo. Keep it.
First run:
`unified/notebooks/01_quickstart_layer_mapping.ipynb`.

See [SETUP.md](SETUP.md) for supported versions, hardware notes (CPU / Apple MPS /
NVIDIA CUDA), and troubleshooting.
