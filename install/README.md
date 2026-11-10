# Installing Brain-Score UMI (cold machine)

These three files are the bootstrap. `setup.sh` clones unified at the
`unified-model-interface-v2` branch and core, vision and language at the commits it
pins in `peer-revisions.json`, builds the pinned conda environment, and runs a
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
`unified/notebooks/lessons/01_your_first_subject.ipynb`.

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

## Check the installed packages

After building and installing the four candidate wheels, run from outside the
checkouts. Use absolute paths for the workspace, wheels and report:

```bash
python -I -m brainscore.validation.workspace \
  --root /path/to/workspace \
  --mode installed \
  --wheels /path/to/wheels \
  --out /path/to/reports/installed.json
```

This runs the same required offline tests as the source profile. It copies tests,
fixtures and examples into temporary directories, verifies installed package
files against the wheels, and rejects imports from source checkouts. Reports
include wheel hashes, package versions, import locations and skip reasons.
Each suite gets a temporary home and cache directories; result caching and model
downloads are disabled. Cache-integrity tests enable their own temporary caches.

This checks package contents and behavior. The separate new-user trial covers
installation from the guide, first-time data access and the first benchmark run.

## Run the offline CI on several candidates

`run_candidate.sh` holds every command of the GitHub Actions offline workflow.
The workflow calls one phase per step; a Linux machine with Python 3.11 runs all
of them. Give a local checkout path or `[owner/repo@]ref` for each repository
that changes; the other peers use the pins in `peer-revisions.json`:

```bash
unified/install/run_candidate.sh --work ~/umi-candidate --venv ~/umi-candidate/venvs \
  --core ~/src/core --vision brain-score/vision@my-branch \
  --language ~/src/language --unified ~/src/unified all
```

Each job gets a fresh virtual environment and its own folder under
`~/umi-candidate/qualification`. As on GitHub, a failed step skips the rest of
its job, the jobs run independently, and any failure fails the run. Uncommitted
changes in a local checkout are built and tested: the script copies each local
checkout into a snapshot under `--work` and builds only the snapshot, so local
paths are only read. They must lie outside `--work` and `--venv`, and must not
contain symlinks that lead outside them; `--venv` must be a folder inside
`--work`. `--work` must not exist or must be empty, so give each run a new
folder. Use the script on copies (for example an rsync of a worktree to EC2),
never on the only copy of uncommitted work. Git hooks never run. Fetching the private unified repository needs Git credentials.
To run the same set on GitHub, start the offline workflow manually with the
`core_revision`, `vision_revision` and `language_revision` inputs (full SHAs).

## Check that the pins agree

`check_pins.py` reads the core pin and the unified pin at every site and fails
when they disagree:

```bash
python unified/install/check_pins.py --root /path/to/siblings
python unified/install/check_pins.py --core <sha> --vision my-branch --language ../language
```
