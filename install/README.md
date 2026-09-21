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

## With the four repositories already present

`setup.sh` and the environment file are for a machine with no checkout. If `core/`,
`vision/`, `language/` and `unified/` already sit side by side, skip both — the pins live
in the packages, so pip alone is enough:

~~~bash
cd <workspace-root>
conda create -y -n brainscore-unified python=3.11
conda activate brainscore-unified
python -m pip install -c unified/install/v2-constraints.txt -e ./core -e ./vision -e ./language -e "./unified[notebooks,test]"
python -m pip check
python -m brainscore.doctor
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
this. To use the file directly, `cp` it beside the repositories first.

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

## Reproduce the reviewed peer revisions

The candidate peer versions are not published on PyPI. Supply all four local
repositories in one pip operation, as above, or build all four wheels before
installing them together. An isolated install of unified alone cannot resolve
these candidate dependencies from PyPI.

`peer-revisions.json` records the exact core, vision, and language commits used
by integration CI for this unified revision. For a fresh workspace, first clone
unified at the revision you want to evaluate, then run from the workspace root:

```bash
git clone --branch unified-model-interface-v2 https://github.com/KartikP/brainscore-unified.git unified
python3.11 - <<'PYTHON'
import json, subprocess
from pathlib import Path
peers = json.loads(Path('unified/install/peer-revisions.json').read_text())
for name, revision in peers.items():
    subprocess.run(['git', 'clone', '--no-checkout',
                    f'https://github.com/brain-score/{name}.git', name], check=True)
    subprocess.run(['git', '-C', name, 'checkout', '--detach', revision], check=True)
PYTHON
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -c unified/install/v2-constraints.txt -e ./core -e ./vision -e ./language -e "./unified[test]"
python -m pip check
python -m brainscore.doctor
python unified/examples/partner_integration.py --out /tmp/umi-first-experiment
```

Record the unified commit alongside the peer manifest. Use new output directories
for each experiment. This is a v2 integration profile, not approval for a general
production release or a replacement for CUDA scientific qualification.
