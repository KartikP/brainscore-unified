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
cp unified/install/environment-unified.yml .
conda env create -n brainscore-unified -f environment-unified.yml
conda activate brainscore-unified
~~~

**Why the copy is required.** The environment file ends with relative editable
installs:

~~~yaml
- -e ./core
- -e ./vision
- -e ./language
- -e ./unified[notebooks]
~~~

Those resolve against **the directory holding the yml**, because conda runs its pip step
with the working directory set there — not against wherever you invoke conda. Point at
the file in place and `./core` means `unified/install/core`, which does not exist:

~~~
ERROR: ./core is not a valid editable requirement. It should either be a path to a
local project or a VCS URL ...
CondaEnvException: Pip failed
~~~

Invoking from the workspace root does **not** fix this; the file has to be moved there.

`setup.sh` avoids this by copying the file to the workspace root before creating the
environment, which is why the bootstrap path has no such caveat.

## Common mistakes

| Symptom | Cause |
| --- | --- |
| `PackagesNotFoundError: install/environment-unified.yml` | `conda create -f` means *force*. Use `conda env create -f`. |
| `./core is not a valid editable requirement` | copy the yml to the workspace root first; conda resolves its relative paths from the file's own directory |
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
