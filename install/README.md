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

`setup.sh` and the environment file are for a machine with no checkout. If `core/`,
`vision/`, `language/` and `unified/` already sit side by side, skip both — the pins live
in the packages, so pip alone is enough:

~~~bash
cd <workspace-root>
conda create -y -n brainscore-unified python=3.11
conda activate brainscore-unified
pip install -e ./core -e ./vision -e ./language -e "./unified[notebooks,test]"
~~~

**Do not point conda at `install/environment-unified.yml` in place.** Its `-e ./core`
entries are relative, and conda runs its pip step from *the file's own directory*, so
`./core` becomes `unified/install/core`:

~~~
ERROR: ./core is not a valid editable requirement.
CondaEnvException: Pip failed
~~~

Invoking conda from the workspace root does not help — the file itself has to be there.
`setup.sh` downloads it to the workspace root, which is why the bootstrap never hits
this. If you want to use the file anyway, `cp` it beside the repositories first.

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
