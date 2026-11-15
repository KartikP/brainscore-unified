#!/usr/bin/env bash
set -euo pipefail

# Brain-Score Unified Model Interface local bootstrap.
# Usage: bash setup.sh            the pinned environment (environment-unified.yml)
#        bash setup.sh --lock     the exact scoring platform (Linux x86_64, Python 3.11,
#                                 CUDA 12.4): every package at the version of
#                                 unified/install/lock-linux-py311-cu124.txt
# Optional: UMI_ENV_NAME=my-env bash setup.sh
# Both create the environment from conda-forge only, so a fresh miniconda does not stop
# at the terms of service of Anaconda's default channels.

LOCK=0
case "${1:-}" in
    "") ;;
    --lock) LOCK=1 ;;
    *) echo "Usage: bash setup.sh [--lock]" >&2; exit 2 ;;
esac
if [[ "$LOCK" == 1 && ( "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ) ]]; then
    echo "Error: --lock installs the Linux x86_64 scoring platform; this is $(uname -s) $(uname -m)." >&2
    echo "Run bash setup.sh without --lock here." >&2
    exit 2
fi

WORKDIR="$(cd "$(dirname "$0")" && pwd)"
BRANCH="unified-model-interface-v2"
ENV_NAME="${UMI_ENV_NAME:-brainscore-unified}"
ENV_FILE="$WORKDIR/environment-unified.yml"

clone_or_verify() {
    # Unified is cloned at the branch tip; core, vision and language at the commit
    # unified was tested with (install/peer-revisions.json), passed as $3.
    local name="$1"
    local url="$2"
    local pinned="${3:-}"
    local destination="$WORKDIR/$name"

    if [[ ! -e "$destination" ]]; then
        echo "Cloning $name at $BRANCH${pinned:+ (pinned to ${pinned:0:7})}"
        git clone --branch "$BRANCH" --single-branch --depth 1 "$url" "$destination"
        if [[ -n "$pinned" ]]; then
            git -C "$destination" fetch --quiet --depth 1 origin "$pinned"
            git -C "$destination" checkout --quiet "$pinned"
        fi
        return
    fi

    if [[ ! -d "$destination/.git" ]]; then
        echo "Error: $destination exists but is not a Git checkout." >&2
        exit 1
    fi

    if [[ -n "$pinned" && "$(git -C "$destination" rev-parse HEAD)" == "$pinned" ]]; then
        echo "$name already exists at the pinned ${pinned:0:7}"
        return
    fi
    local current_branch
    current_branch="$(git -C "$destination" branch --show-current)"
    if [[ "$current_branch" != "$BRANCH" ]]; then
        echo "Error: $name is on '${current_branch:-a detached commit}'; expected '$BRANCH'${pinned:+ or ${pinned:0:7}}." >&2
        echo "Use a clean checkout or switch that repository explicitly." >&2
        exit 1
    fi
    echo "$name already exists on $BRANCH"
}

echo "Brain-Score UMI local setup"
echo "Root: $WORKDIR"
echo "Environment: $ENV_NAME"

clone_or_verify unified "https://github.com/KartikP/brainscore-unified.git"
peer() {    # the commit unified pins for a peer repository
    python3 -c "import json, sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])" \
        "$WORKDIR/unified/install/peer-revisions.json" "$1"
}
clone_or_verify core "https://github.com/brain-score/core.git" "$(peer core)"
clone_or_verify vision "https://github.com/brain-score/vision.git" "$(peer vision)"
clone_or_verify language "https://github.com/brain-score/language.git" "$(peer language)"

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
jupyter notebook unified/notebooks/lessons/01_your_first_subject.ipynb
```

`python -m brainscore.doctor` prints which interpreter answered, whether every
dependency sits inside the range scoring is verified against, and which
user-supplied data assets are present. Run it whenever a result surprises you.
Drift is worth catching early because it moves scores rather than raising them
as errors. If the command cannot import `brainscore_core` at all, the active
environment is the wrong one — easy to do on a machine with several.

Then, in order:

1. [What UMI is](unified/docs/01_what_is_umi.md) and [core concepts](unified/docs/02_core_concepts.md):
   the vocabulary. Twenty minutes, and it makes everything else legible.
2. [Lessons](unified/notebooks/README.md#lessons): eleven short notebooks, each building on
   the last; then the [experiments](unified/notebooks/README.md#experiments), which use them
   on scientific questions.
3. [Manual](unified/docs/README.md): guides for scoring a model, running experiments,
   adding data, and extending UMI.

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

# Only conda-forge is used: the environment file lists it with `nodefaults`, and the lock
# path passes --override-channels. Anaconda's default channels, whose terms of service a
# fresh miniconda asks to accept, are never contacted; nothing is accepted on your behalf.
cd "$WORKDIR"
if [[ "$LOCK" == 1 ]]; then
    LOCK_FILE="$WORKDIR/unified/install/lock-linux-py311-cu124.txt"
    conda create --yes --name "$ENV_NAME" --override-channels --channel conda-forge python=3.11 pip
    echo "Installing every package at the version of $LOCK_FILE"
    conda run --name "$ENV_NAME" python -m pip install -r "$LOCK_FILE"
    echo "Installing the four repositories from this checkout (their dependencies are in the lock)"
    conda run --name "$ENV_NAME" python -m pip install --no-deps \
        -e "$WORKDIR/core" \
        -e "$WORKDIR/vision" \
        -e "$WORKDIR/language" \
        -e "$WORKDIR/unified"
else
    conda env create --name "$ENV_NAME" --file "$ENV_FILE"

    echo "Refreshing editable installs from this checkout"
    conda run --name "$ENV_NAME" python -m pip install \
        -c "$WORKDIR/unified/install/v2-constraints.txt" \
        -e "$WORKDIR/core" \
        -e "$WORKDIR/vision" \
        -e "$WORKDIR/language" \
        -e "$WORKDIR/unified[notebooks]"
fi

echo "Verifying package consistency"
conda run --name "$ENV_NAME" python -m pip check
conda run --name "$ENV_NAME" python -c "import brainscore, brainscore_core, brainscore_vision, brainscore_language; print('Brain-Score imports: OK')"
# Check the cache operation the code uses (slice_kv_cache), not one transformers
# version's method: transformers 5 removed to_legacy_cache, and both are supported.
conda run --name "$ENV_NAME" python -c "import torch; from transformers import DynamicCache; from brainscore_language.model_helpers.huggingface import slice_kv_cache; cache = DynamicCache(); kv = torch.zeros(1, 1, 3, 2); cache.update(kv, kv, 0); assert slice_kv_cache(cache, 2).get_seq_length() == 2; print('Transformers cache API: OK')"
conda run --name "$ENV_NAME" python -c "import numpy, sklearn, transformers, xarray; print('numpy', numpy.__version__); print('xarray', xarray.__version__); print('scikit-learn', sklearn.__version__); print('transformers', transformers.__version__)"
# Same check the user is told to run, so a fresh install is verified by the tool
# they will actually use rather than by a one-off snippet that can drift from it.
conda run --name "$ENV_NAME" python -m brainscore.doctor

echo
echo "Setup complete."
echo "Activate with: conda activate $ENV_NAME"
echo "Start with: unified/docs/01_what_is_umi.md, then unified/notebooks/lessons/01_your_first_subject.ipynb"
