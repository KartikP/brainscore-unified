#!/usr/bin/env bash
set -euo pipefail

# Brain-Score Unified Model Interface local bootstrap.
# Usage: bash setup.sh
# Optional: UMI_ENV_NAME=my-env bash setup.sh

WORKDIR="$(cd "$(dirname "$0")" && pwd)"
BRANCH="unified-model-interface-v2"
ENV_NAME="${UMI_ENV_NAME:-brainscore-unified}"
ENV_FILE="$WORKDIR/environment-unified.yml"

clone_or_verify() {
    local name="$1"
    local url="$2"
    local destination="$WORKDIR/$name"

    if [[ ! -e "$destination" ]]; then
        echo "Cloning $name at $BRANCH"
        git clone --branch "$BRANCH" --single-branch --depth 1 "$url" "$destination"
        return
    fi

    if [[ ! -d "$destination/.git" ]]; then
        echo "Error: $destination exists but is not a Git checkout." >&2
        exit 1
    fi

    local current_branch
    current_branch="$(git -C "$destination" branch --show-current)"
    if [[ "$current_branch" != "$BRANCH" ]]; then
        echo "Error: $name is on '$current_branch'; expected '$BRANCH'." >&2
        echo "Use a clean checkout or switch that repository explicitly." >&2
        exit 1
    fi
    echo "$name already exists on $BRANCH"
}

echo "Brain-Score UMI local setup"
echo "Root: $WORKDIR"
echo "Environment: $ENV_NAME"

clone_or_verify core "https://github.com/brain-score/core.git"
clone_or_verify vision "https://github.com/brain-score/vision.git"
clone_or_verify language "https://github.com/brain-score/language.git"
clone_or_verify unified "https://github.com/KartikP/brainscore-unified.git"

# Record the exact commit of each repo so a run is reproducible / citable
# (branches move; this pins what was actually installed).
{
    echo "# Brain-Score UMI install manifest"
    for r in core vision language unified; do
        printf '%-9s %s %s\n' "$r" "$BRANCH" "$(git -C "$WORKDIR/$r" rev-parse HEAD)"
    done
} | tee "$WORKDIR/install-manifest.txt"

# Orientation for the workspace root. The four repositories are separate checkouts, so
# nothing owns this directory and a fresh clone would otherwise have no landing page.
cat > "$WORKDIR/README.md" <<'ROOT_README'
# Brain-Score UMI workspace

Four repositories, one interface. `unified/` is where you start.

| Path | What it is |
| --- | --- |
| `unified/` | the cross-domain `brainscore` package: models, benchmarks, metrics, templates |
| `core/` | shared contracts, assemblies, scoring utilities |
| `vision/`, `language/` | domain models, benchmarks, and extraction helpers |

## Start here

```bash
conda activate brainscore-unified
python -m brainscore.doctor
jupyter notebook unified/notebooks/01_quickstart_layer_mapping.ipynb
```

`python -m brainscore.doctor` prints which interpreter answered, whether every
dependency sits inside the range scoring is verified against, and which
user-supplied data assets are present. Run it whenever a result surprises you.
Drift is worth catching early because it moves scores rather than raising them
as errors. If the command cannot import `brainscore_core` at all, the active
environment is the wrong one — easy to do on a machine with several.

Then, in order:

1. [Concepts](unified/docs/concepts.md) — the vocabulary. Ten minutes, and it makes
   everything else legible.
2. [Notebook path](unified/notebooks/README.md) — sixteen notebooks in four arcs; take
   the arc that matches your goal rather than all of them.
3. [Extending UMI](unified/EXTENDING.md) and [templates](unified/templates/) — to add a
   model, benchmark, metric, dataset, or capability. Every template runs as-is.

Coming from classic Brain-Score? [Start here instead](unified/docs/from_brain_score.md).

`install-manifest.txt` records the exact commit of each repository from this install.
ROOT_README

if [[ ! -f "$ENV_FILE" ]]; then
    echo "Error: missing $ENV_FILE" >&2
    exit 1
fi

if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
    echo "Error: conda environment '$ENV_NAME' already exists." >&2
    echo "Choose another name with UMI_ENV_NAME or remove it explicitly." >&2
    exit 1
fi

# Recent conda (>=24.x) gates environment creation behind the Anaconda channels'
# Terms of Service, which a fresh install has not accepted -- `conda env create`
# then fails with CondaToSNonInteractiveError. Accept it non-interactively when
# the subcommand exists (older conda has no `tos`); harmless if already accepted.
if conda tos --help >/dev/null 2>&1; then
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >/dev/null 2>&1 || true
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r    >/dev/null 2>&1 || true
fi

cd "$WORKDIR"
conda env create --name "$ENV_NAME" --file "$ENV_FILE"

echo "Refreshing editable installs from this checkout"
conda run --name "$ENV_NAME" python -m pip install \
    -e "$WORKDIR/core" \
    -e "$WORKDIR/vision" \
    -e "$WORKDIR/language" \
    -e "$WORKDIR/unified[notebooks]"

echo "Verifying package consistency"
conda run --name "$ENV_NAME" python -m pip check
conda run --name "$ENV_NAME" python -c "import brainscore, brainscore_core, brainscore_vision, brainscore_language; print('Brain-Score imports: OK')"
conda run --name "$ENV_NAME" python -c "from transformers import DynamicCache; assert hasattr(DynamicCache(), 'to_legacy_cache'); print('Transformers cache API: OK')"
conda run --name "$ENV_NAME" python -c "import numpy, sklearn, transformers, xarray; print('numpy', numpy.__version__); print('xarray', xarray.__version__); print('scikit-learn', sklearn.__version__); print('transformers', transformers.__version__)"
# Same check the user is told to run, so a fresh install is verified by the tool
# they will actually use rather than by a one-off snippet that can drift from it.
conda run --name "$ENV_NAME" python -m brainscore.doctor

echo
echo "Setup complete."
echo "Activate with: conda activate $ENV_NAME"
echo "Start with: unified/notebooks/01_quickstart_layer_mapping.ipynb"
