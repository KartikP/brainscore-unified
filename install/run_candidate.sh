#!/usr/bin/env bash
# Run the UMI offline CI checks on one set of four checkouts.
#
# GitHub Actions (.github/workflows/offline-tests.yml) calls one phase per
# step. A development machine calls "all", which also fetches the checkouts and
# creates fresh virtual environments, so both run the same commands.
#
# Usage:
#   install/run_candidate.sh [options] PHASE...
#
# Options:
#   --work DIR       Workspace that holds core/ vision/ language/ unified/ and
#                    the outputs (default: $UMI_WORK or ./umi-candidate).
#   --core VALUE     A local checkout path, or [owner/repo@]ref (full SHA,
#   --vision VALUE   branch or tag). A peer without a value uses the pin in
#   --language VALUE unified/install/peer-revisions.json. Unified without a
#   --unified VALUE  value uses the checkout that holds this script.
#   --venv DIR       Create a fresh virtual environment per job under DIR
#                    (used by "all"; GitHub Actions uses its own Python).
#
# Phases (in the order "all" runs them):
#   prepare          Fetch or link the four checkouts and record revisions.
#   pins             Report pin agreement (install/check_pins.py).
#   offline          verify build contents example torch install
#                    installed-checks packages notebooks
#   openpi-client    verify-core openpi-client-install openpi-record
#                    openpi-client-test
#   openpi-instrumentation
#                    verify-core openpi-checkout openpi-torch
#                    openpi-instrumentation-install openpi-record
#                    openpi-instrumentation-test
#   all              Every phase above.
#   resolve          GitHub Actions only: resolve the candidate and its peers.
#
# Example on a development machine (test four branch tips together):
#   install/run_candidate.sh --work ~/umi-candidate --venv ~/umi-venvs \
#     --core brain-score/core@my-branch --vision ../vision \
#     --language brain-score/language@<sha> --unified ../unified all

set -eo pipefail

TORCH_VERSION=2.13.0
TORCHVISION_VERSION=0.28.0
TORCH_INDEX=https://download.pytorch.org/whl/cpu
OPENPI_REPOSITORY=Physical-Intelligence/openpi
OPENPI_REVISION=215abfb217dbac7d5f1273282331b9b1866c0479
SCRIPT_UNIFIED=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

export RESULTCACHING_DISABLE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export BS_INSTALL_DEPENDENCIES=no

WORK=${UMI_WORK:-$PWD/umi-candidate}
VENV=
CORE=
VISION=
LANGUAGE=
UNIFIED=
PHASES=()
while [ $# -gt 0 ]; do
    case "$1" in
        --work) WORK=$2; shift 2 ;;
        --venv) VENV=$2; shift 2 ;;
        --core) CORE=$2; shift 2 ;;
        --vision) VISION=$2; shift 2 ;;
        --language) LANGUAGE=$2; shift 2 ;;
        --unified) UNIFIED=$2; shift 2 ;;
        -h|--help) sed -n '2,40p' "${BASH_SOURCE[0]}"; exit 0 ;;
        -*) echo "Unknown option: $1" >&2; exit 2 ;;
        *) PHASES+=("$1"); shift ;;
    esac
done
[ ${#PHASES[@]} -gt 0 ] || { echo 'Name at least one phase (see --help)' >&2; exit 2; }

mkdir -p "$WORK"
WORK=$(cd "$WORK" && pwd)
RECORD=$WORK/qualification/revisions.json
QUAL=${UMI_QUALIFICATION:-$WORK/qualification}
TMP_DIR=${RUNNER_TEMP:-$WORK/tmp}
PYTHON=${UMI_PYTHON:-$(command -v python || command -v python3)}
mkdir -p "$QUAL" "$TMP_DIR"

repository_of() {
    case "$1" in
        unified) echo KartikP/brainscore-unified ;;
        *) echo "brain-score/$1" ;;
    esac
}

# Link a local path, or fetch one exact ref into a workspace-owned clone.
place() {
    local name=$1 value=$2 dest=$WORK/$1 repository ref
    if [ -d "$value" ]; then
        value=$(cd "$value" && pwd)
        if [ -L "$dest" ] || [ ! -e "$dest" ]; then
            ln -sfn "$value" "$dest"
        else
            rm -rf "$dest" && ln -s "$value" "$dest"
        fi
        return
    fi
    repository=$(repository_of "$name")
    ref=$value
    case "$value" in
        */*@*) repository=${value%@*}; ref=${value##*@} ;;
    esac
    [ -L "$dest" ] && rm "$dest"
    [ -d "$dest/.git" ] || { rm -rf "$dest"; git init -q "$dest"; }
    git -C "$dest" fetch -q --depth 1 "https://github.com/$repository.git" "$ref"
    git -C "$dest" checkout -q --detach FETCH_HEAD
    echo "$repository" > "$WORK/.repository-$name"
}

pin_of() {
    "$PYTHON" -c 'import json, sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' \
        "$WORK/unified/install/peer-revisions.json" "$1"
}

phase_prepare() {
    # Refuse to move HEAD in checkouts this script did not create.
    if [ -n "$(ls -A "$WORK" 2>/dev/null | grep -v -e '^qualification$' -e '^tmp$')" ] \
            && [ ! -f "$WORK/.umi-candidate" ]; then
        echo "$WORK is not empty and was not created by this script" >&2
        exit 1
    fi
    touch "$WORK/.umi-candidate"
    rm -f "$WORK"/.repository-*
    place unified "${UNIFIED:-$SCRIPT_UNIFIED}"
    local name value overrides=() repository revision
    for name in core vision language; do
        case "$name" in
            core) value=$CORE ;; vision) value=$VISION ;; language) value=$LANGUAGE ;;
        esac
        if [ -n "$value" ]; then
            place "$name" "$value"
            repository=$(cat "$WORK/.repository-$name" 2>/dev/null || repository_of "$name")
            revision=$(git -C "$WORK/$name" rev-parse HEAD)
            overrides+=(--override "$name=$repository@$revision")
        else
            place "$name" "$(pin_of "$name")"
        fi
    done
    repository=$(cat "$WORK/.repository-unified" 2>/dev/null || repository_of unified)
    revision=$(git -C "$WORK/unified" rev-parse HEAD)
    CANDIDATE=unified CANDIDATE_REPOSITORY=$repository CANDIDATE_REVISION=$revision \
    UNIFIED_REPOSITORY=$repository UNIFIED_REVISION=$revision \
        "$PYTHON" "$WORK/unified/brainscore/validation/revisions.py" \
        --manifest "$WORK/unified/install/peer-revisions.json" --record "$RECORD" \
        ${overrides[@]+"${overrides[@]}"}
    for name in core vision language unified; do
        if ! git -C "$WORK/$name" diff --quiet HEAD || \
                [ -n "$(git -C "$WORK/$name" ls-files --others --exclude-standard)" ]; then
            echo "note: $name has uncommitted changes; they are built and tested" >&2
        fi
    done
}

phase_pins() {
    local args=() name
    for name in core vision language unified; do
        [ -e "$WORK/$name" ] && args+=("--$name" "$WORK/$name")
    done
    "$PYTHON" "$WORK/unified/install/check_pins.py" ${args[@]+"${args[@]}"}
}

# GitHub Actions only: resolve the candidate (CANDIDATE* and UNIFIED_REPOSITORY
# from the workflow) and optional OVERRIDE_CORE/VISION/LANGUAGE peers
# ([owner/repo@]full-sha). Writes the record and the checkout step outputs.
phase_resolve() {
    local overrides=() name value
    for name in core vision language; do
        case "$name" in
            core) value=$OVERRIDE_CORE ;; vision) value=$OVERRIDE_VISION ;; language) value=$OVERRIDE_LANGUAGE ;;
        esac
        [ -z "$value" ] || overrides+=(--override "$name=$value")
    done
    UNIFIED_REVISION=$(git -C "$WORK/unified" rev-parse HEAD) \
        "$PYTHON" "$WORK/unified/brainscore/validation/revisions.py" \
        --manifest "$WORK/unified/install/peer-revisions.json" --record "$RECORD" \
        ${overrides[@]+"${overrides[@]}"}
}

verify_checkouts() {
    "$PYTHON" "$WORK/unified/brainscore/validation/revisions.py" --root "$WORK" \
        --record "$RECORD" --verify "$@"
}

phase_verify() { verify_checkouts; }
phase_verify_core() { verify_checkouts core unified; }

phase_build() {
    cd "$WORK"
    rm -rf wheels example-wheels
    "$PYTHON" -m pip install build setuptools==68.2.2 wheel
    local repo
    for repo in core vision language unified; do
        "$PYTHON" -m build --wheel --no-isolation --outdir wheels "$repo"
    done
}

phase_contents() {
    cd "$WORK"
    "$PYTHON" unified/brainscore/validation/package_contents.py --wheels wheels \
        --out "$QUAL/package-contents.json"
}

phase_example() {
    cd "$WORK"
    "$PYTHON" -m build --wheel --no-isolation --outdir example-wheels unified/examples/partner_tool
}

phase_torch() {
    [ "$(uname -s)" = Linux ] || { echo 'CPU Torch is installed on Linux only'; return; }
    "$PYTHON" -m pip install "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" \
        --index-url "$TORCH_INDEX"
}

phase_install() {
    cd "$WORK"
    local unified_wheels=(wheels/brainscore-*.whl)
    "$PYTHON" -m pip install -c unified/install/v2-constraints.txt wheels/*.whl "${unified_wheels[0]}[test]"
    "$PYTHON" -m pip install --no-deps example-wheels/*.whl
    "$PYTHON" -m pip check
    "$PYTHON" -m pip freeze > resolved-dependencies.txt
}

phase_installed_checks() {
    export UMI_TEST_CPU_ONLY=1
    cd "$TMP_DIR"
    "$PYTHON" -m brainscore.doctor
    "$PYTHON" "$WORK/unified/examples/partner_integration.py" --out "$TMP_DIR/partner-demo"
    "$PYTHON" -I -m umi_partner_tool --out "$TMP_DIR/external-tool-demo"
    "$PYTHON" -I -m umi_partner_tool.toolbox --out "$TMP_DIR/installed-examples"
    "$PYTHON" -I -m pytest "$WORK/unified/tests/test_production_tools.py" \
        "$WORK/unified/tests/test_droid_policy.py" "$WORK/unified/examples/partner_tool/tests" \
        -q --import-mode=importlib -p no:cacheprovider --junitxml="$QUAL/external-examples.xml"
}

phase_packages() {
    export UMI_TEST_CPU_ONLY=1
    cd "$TMP_DIR"
    "$PYTHON" -I -m brainscore.validation.workspace --root "$WORK" \
        --mode installed --wheels "$WORK/wheels" --out "$QUAL/installed.json"
}

phase_notebooks() {
    export UMI_TEST_CPU_ONLY=1
    cd "$TMP_DIR"
    "$PYTHON" -I -m brainscore.validation.notebooks --source "$WORK/unified/notebooks" \
        --out "$QUAL/notebooks" --timeout 180
}

phase_openpi_client_install() {
    cd "$WORK"
    "$PYTHON" -m pip install -c unified/install/v2-constraints.txt ./core \
        -r unified/install/openpi-client-tests.txt
    "$PYTHON" -m pip check
}

phase_openpi_checkout() {
    local dest=$WORK/openpi
    [ -d "$dest/.git" ] || { rm -rf "$dest"; git init -q "$dest"; }
    git -C "$dest" fetch -q --depth 1 "https://github.com/$OPENPI_REPOSITORY.git" "$OPENPI_REVISION"
    git -C "$dest" checkout -q --detach FETCH_HEAD
}

phase_openpi_torch() {
    "$PYTHON" -m pip install "torch==$TORCH_VERSION" --index-url "$TORCH_INDEX"
}

phase_openpi_instrumentation_install() {
    cd "$WORK"
    "$PYTHON" -m pip install -c unified/install/v2-constraints.txt ./core \
        -r unified/install/openpi-instrumentation-tests.txt
    "$PYTHON" -m pip check
}

phase_openpi_record() {
    "$PYTHON" -m pip freeze > "$QUAL/dependencies.txt"
    git -C "$WORK/unified" rev-parse HEAD > "$QUAL/unified-revision.txt"
    git -C "$WORK/core" rev-parse HEAD > "$QUAL/core-revision.txt"
    if [ -d "$WORK/openpi" ]; then
        git -C "$WORK/openpi" rev-parse HEAD > "$QUAL/openpi-revision.txt"
    fi
}

phase_openpi_client_test() {
    cd "$WORK/unified"
    PYTHONPATH="$WORK/unified:$WORK/core" "$PYTHON" -m pytest tests/test_openpi_remote.py -q -ra \
        -p no:cacheprovider -p brainscore.validation.required_checks \
        --umi-repository openpi-client \
        --umi-check-report "$QUAL/openpi-client.json" \
        --junitxml="$QUAL/openpi-client.xml" --timeout=30 \
        2>&1 | tee "$QUAL/openpi-client.log"
}

phase_openpi_instrumentation_test() {
    export UMI_TEST_CPU_ONLY=1 JAX_PLATFORMS=cpu
    cd "$WORK/unified"
    OPENPI_SOURCE="$WORK/openpi" PYTHONPATH="$WORK/unified:$WORK/core" \
        "$PYTHON" -m pytest tests/test_openpi_instrumentation.py tests/test_libero_transport.py -q -ra \
        -p no:cacheprovider -p brainscore.validation.required_checks \
        --umi-repository openpi-instrumentation \
        --umi-check-report "$QUAL/openpi-instrumentation.json" \
        --junitxml="$QUAL/openpi-instrumentation.xml" --timeout=120 \
        2>&1 | tee "$QUAL/openpi-instrumentation.log"
}

OFFLINE_PHASES="verify build contents example torch install installed-checks packages notebooks"
CLIENT_PHASES="verify-core openpi-client-install openpi-record openpi-client-test"
INSTRUMENTATION_PHASES="verify-core openpi-checkout openpi-torch openpi-instrumentation-install openpi-record openpi-instrumentation-test"

# Phases change directory with absolute paths only. Do not call run_phase in an
# if, || or && context: Bash then ignores errexit inside the phase.
run_phase() {
    echo "==> $1"
    "phase_$(echo "$1" | tr '-' '_')"
}

# Run one job's phases in a fresh virtual environment (when --venv is set)
# and with its own output folder, as GitHub Actions runs each job on a new runner.
run_job() {
    local job=$1 phases=$2 phase
    (
        QUAL=$WORK/qualification/$job
        mkdir -p "$QUAL"
        if [ -n "$VENV" ]; then
            "${UMI_BASE_PYTHON:-python3.11}" -m venv --clear "$VENV/$job"
            PATH=$VENV/$job/bin:$PATH
            PYTHON=python
        fi
        for phase in $phases; do run_phase "$phase"; done
    )
}

for phase in "${PHASES[@]}"; do
    case "$phase" in
        offline) run_job offline "$OFFLINE_PHASES" ;;
        openpi-client) run_job openpi-client "$CLIENT_PHASES" ;;
        openpi-instrumentation) run_job openpi-instrumentation "$INSTRUMENTATION_PHASES" ;;
        all)
            run_phase prepare
            # Branch tips under test usually move pins, so report without failing.
            phase_pins || echo 'warning: pin sites disagree (see the table above)' >&2
            run_job offline "$OFFLINE_PHASES"
            run_job openpi-client "$CLIENT_PHASES"
            run_job openpi-instrumentation "$INSTRUMENTATION_PHASES"
            echo "PASS: all jobs passed. Outputs are in $WORK/qualification"
            ;;
        *)
            if ! declare -F "phase_$(echo "$phase" | tr '-' '_')" > /dev/null; then
                echo "Unknown phase: $phase (see --help)" >&2
                exit 2
            fi
            run_phase "$phase"
            ;;
    esac
done
