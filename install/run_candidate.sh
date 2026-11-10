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
#                    the outputs (default: $UMI_WORK or ~/umi-candidate).
#   --core VALUE     A local checkout path, or [owner/repo@]ref (full SHA,
#   --vision VALUE   branch or tag). A peer without a value uses the pin in
#   --language VALUE unified/install/peer-revisions.json. Unified without a
#   --unified VALUE  value uses the checkout that holds this script.
#   --venv DIR       Create a fresh virtual environment per job under DIR, a
#                    folder inside --work (for example --work/venvs). Used by
#                    "all"; GitHub Actions uses its own Python.
#   --job NAME       Export the environment of this workflow job (offline,
#                    openpi-client, openpi-instrumentation). GitHub Actions
#                    sets the same values in the workflow; "all" uses this.
#
# Local paths are only read. prepare copies each one (HEAD commit and working
# tree, including uncommitted files) into a snapshot under --work; builds and
# Git commands run on the snapshots. Local paths must lie outside --work and
# --venv, and --work and --venv must lie outside them. A local path with a
# symlink that leads outside it is refused. --work must not exist or must be
# empty: each run starts clean and nothing is reused. Git hooks never run.
#
# Phases (in the order "all" runs them):
#   prepare          Fetch or snapshot the four checkouts; record revisions.
#   pins             Report pin agreement (install/check_pins.py).
#   offline          verify build contents example torch install
#                    installed-checks packages notebooks
#   openpi-client    verify-core openpi-client-install openpi-client-record
#                    openpi-client-test
#   openpi-instrumentation
#                    verify-core openpi-checkout openpi-torch
#                    openpi-instrumentation-install
#                    openpi-instrumentation-record openpi-instrumentation-test
#   all              Every phase above. As in GitHub Actions, the three jobs
#                    run independently, a failed phase skips the later phases
#                    of its job except the record phases, and any failure
#                    fails the run.
#   resolve          GitHub Actions only: resolve the candidate and its peers.
#   print-env        Print the environment of --job (used by the tests).
#
# Example on a development machine (test four branch tips together):
#   install/run_candidate.sh --work ~/umi-candidate --venv ~/umi-candidate/venvs \
#     --core brain-score/core@my-branch --vision ~/src/vision \
#     --language brain-score/language@<sha> --unified ~/src/unified all

set -eo pipefail

TORCH_VERSION=2.13.0
TORCHVISION_VERSION=0.28.0
TORCH_INDEX=https://download.pytorch.org/whl/cpu
OPENPI_REPOSITORY=Physical-Intelligence/openpi
OPENPI_REVISION=215abfb217dbac7d5f1273282331b9b1866c0479
SELF=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/$(basename "${BASH_SOURCE[0]}")
SCRIPT_UNIFIED=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
# Entries under --work that this script may replace. Anything else is left alone.
MANAGED_NAMES=' core vision language unified openpi '
ALWAYS_PHASES=' openpi-client-record openpi-instrumentation-record '

WORK=${UMI_WORK:-$HOME/umi-candidate}
VENV=
JOB=
CORE=
VISION=
LANGUAGE=
UNIFIED=
PHASES=()
while [ $# -gt 0 ]; do
    case "$1" in
        --work) WORK=$2; shift 2 ;;
        --venv) VENV=$2; shift 2 ;;
        --job) JOB=$2; shift 2 ;;
        --core) CORE=$2; shift 2 ;;
        --vision) VISION=$2; shift 2 ;;
        --language) LANGUAGE=$2; shift 2 ;;
        --unified) UNIFIED=$2; shift 2 ;;
        -h|--help) sed -n '2,53p' "$SELF"; exit 0 ;;
        -*) echo "Unknown option: $1" >&2; exit 2 ;;
        *) PHASES+=("$1"); shift ;;
    esac
done
[ ${#PHASES[@]} -gt 0 ] || { echo 'Name at least one phase (see --help)' >&2; exit 2; }

die() { echo "run_candidate: $*" >&2; exit 1; }

# The environment of each workflow job. offline-tests.yml sets the same values
# in its job env blocks; tests/test_run_candidate.py checks that they agree.
job_env() {
    export RESULTCACHING_DISABLE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 BS_INSTALL_DEPENDENCIES=no
    case "$1" in
        offline) export UMI_TEST_CPU_ONLY=1 ;;
        openpi-client) ;;
        openpi-instrumentation) export JAX_PLATFORMS=cpu UMI_TEST_CPU_ONLY=1 ;;
        *) die "unknown job: $1" ;;
    esac
}
[ -z "$JOB" ] || job_env "$JOB"

canonical() { (cd "$1" && pwd -P); }

# Canonical form of a path that may not exist yet.
planned_path() {
    if [ -d "$1" ]; then
        canonical "$1"
    else
        echo "$(planned_path "$(dirname "$1")")/$(basename "$1")"
    fi
}

# True when canonical path $1 equals canonical path $2 or lies inside it.
inside() {
    case "$1/" in
        "$2"/*) return 0 ;;
        *) return 1 ;;
    esac
}

# Reject a local path that overlaps --work, before anything is changed.
check_user_path() {
    local path
    path=$(canonical "$1")
    if inside "$path" "$WORK"; then
        die "local path $path is inside --work $WORK; give a path outside it"
    fi
    if inside "$WORK" "$path"; then
        die "--work $WORK is inside local path $path; choose a work folder outside it"
    fi
}

# A '.' or '..' component could make the planned path differ from the folder
# that is finally used, so such paths are refused. The remaining components
# are plain names, and planned_path resolves every existing part physically.
physical_path() {
    case "/$1/" in
        */./*|*/../*) die "$2 must not contain '.' or '..' components: $1" ;;
    esac
    planned_path "$1"
}

WORK=$(physical_path "$WORK" --work)
[ -z "$VENV" ] || VENV=$(physical_path "$VENV" --venv)
# Local checkouts are only read: prepare copies each one into a snapshot under
# --work, and every build and Git command runs on the snapshot. Check every
# local path before anything is created. GitHub Actions never runs prepare:
# its checkouts are already in --work and belong to the runner.
case " ${PHASES[*]} " in
    *" prepare "*|*" all "*)
        for value in "${UNIFIED:-$SCRIPT_UNIFIED}" "$CORE" "$VISION" "$LANGUAGE"; do
            if [ -n "$value" ] && [ -d "$value" ]; then
                check_user_path "$value"
                if [ -n "$VENV" ] && { inside "$VENV" "$(canonical "$value")" || inside "$(canonical "$value")" "$VENV"; }; then
                    die "--venv $VENV overlaps local path $value"
                fi
            fi
        done
        # Every run starts from an empty --work: nothing from an earlier run (a
        # clone, a link, a virtual environment or an output folder) is reused.
        if [ -e "$WORK" ] && [ -n "$(ls -A "$WORK")" ]; then
            die "--work $WORK must not exist or must be empty; choose a new folder"
        fi
        ;;
esac
# Virtual environments live in their own folder under --work, apart from the
# snapshots and outputs.
if [ -n "$VENV" ]; then
    case "$VENV" in
        "$WORK"/*) ;;
        *) die "--venv must be a folder inside --work ($WORK)" ;;
    esac
    venv_top=${VENV#"$WORK"/}
    venv_top=${venv_top%%/*}
    case " core vision language unified openpi wheels example-wheels qualification tmp " in
        *" $venv_top "*) die "--venv must not be inside $WORK/$venv_top" ;;
    esac
fi
mkdir -p "$WORK"
RECORD=$WORK/qualification/revisions.json
QUAL=${UMI_QUALIFICATION:-$WORK/qualification}
TMP_DIR=${RUNNER_TEMP:-$WORK/tmp}
PYTHON=${UMI_PYTHON:-$(command -v python || command -v python3)}
mkdir -p "$QUAL" "$TMP_DIR"

managed_path() {
    case "$MANAGED_NAMES" in
        *" $1 "*) echo "$WORK/$1" ;;
        *) die "not a managed workspace entry: $1" ;;
    esac
}

# Every Git command runs without hooks: no global core.hooksPath and no
# template hooks can run during a clone or checkout.
git_quiet() { git -c core.hooksPath=/dev/null "$@"; }

# A new entry under --work. --work starts empty, so an existing entry means a
# second placement in the same run, which is a bug.
new_entry() {
    local dest
    dest=$(managed_path "$1")
    [ ! -e "$dest" ] && [ ! -L "$dest" ] || die "$dest already exists"
    echo "$dest"
}

# Put one exact ref in a new clone under --work.
fetch_clone() {
    local name=$1 repository=$2 ref=$3 dest
    dest=$(new_entry "$name")
    git_quiet init -q --template= "$dest"
    git_quiet -C "$dest" fetch -q --depth 1 "https://github.com/$repository.git" "$ref"
    git_quiet -C "$dest" checkout -q --detach FETCH_HEAD
}

repository_of() {
    case "$1" in
        unified) echo KartikP/brainscore-unified ;;
        *) echo "brain-score/$1" ;;
    esac
}

# Print each symlink below $1 (outside .git) whose target resolves outside $1.
outside_links() {
    "$PYTHON" -I -c '
import os, sys
root = os.path.realpath(sys.argv[1])
for folder, folders, files in os.walk(root):
    if folder == root and ".git" in folders:
        folders.remove(".git")
    for name in folders + files:
        path = os.path.join(folder, name)
        if os.path.islink(path):
            target = os.path.realpath(path)
            if os.path.commonpath([root, target]) != root:
                print(f"{os.path.relpath(path, root)} -> {os.readlink(path)}")
' "$1"
}

# Copy a local checkout into a snapshot under --work. The source is only read:
# "git rev-parse" and a local "git clone" read its repository, and tar reads
# its files. The snapshot has the same HEAD commit and the same working tree,
# including uncommitted and untracked files, so all later builds and Git
# commands run on the snapshot and never on the source. A symlink that leads
# out of the source would let a build write through the snapshot, so such a
# source is refused; relative links inside the source are copied as links.
snapshot() {
    local name=$1 source dest revision links
    source=$(canonical "$2")
    revision=$(GIT_OPTIONAL_LOCKS=0 git_quiet -C "$source" rev-parse --verify HEAD) \
        || die "$source is not a Git checkout with a commit"
    links=$(outside_links "$source") || die "cannot check the symlinks in $source"
    if [ -n "$links" ]; then
        die "$source has symlinks that lead outside it; remove them or give a ref:"$'\n'"$links"
    fi
    dest=$(new_entry "$name")
    GIT_OPTIONAL_LOCKS=0 git_quiet clone -q --template= --no-hardlinks --no-checkout -- "$source" "$dest"
    git_quiet -C "$dest" checkout -q --detach "$revision"
    find "$dest" -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +
    (cd "$source" && tar -cf - --exclude=./.git --exclude=.git .) | (cd "$dest" && tar -xf -)
    [ "$(git_quiet -C "$dest" rev-parse HEAD)" = "$revision" ] || die "snapshot of $source has the wrong HEAD"
}

# Snapshot a local path (checked by check_user_path first), or fetch a ref.
place() {
    local name=$1 value=$2 repository ref
    if [ -d "$value" ]; then
        snapshot "$name" "$value"
        return
    fi
    repository=$(repository_of "$name")
    ref=$value
    case "$value" in
        */*@*) repository=${value%@*}; ref=${value##*@} ;;
    esac
    fetch_clone "$name" "$repository" "$ref"
    echo "$repository" > "$WORK/.repository-$name"
}

pin_of() {
    "$PYTHON" -c 'import json, sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' \
        "$WORK/unified/install/peer-revisions.json" "$1"
}

phase_prepare() {
    local name value overrides=() repository revision
    # The local paths and the empty --work were checked at startup.
    place unified "${UNIFIED:-$SCRIPT_UNIFIED}"
    for name in core vision language; do
        case "$name" in
            core) value=$CORE ;; vision) value=$VISION ;; language) value=$LANGUAGE ;;
        esac
        if [ -n "$value" ]; then
            place "$name" "$value"
            repository=$(cat "$WORK/.repository-$name" 2>/dev/null || repository_of "$name")
            revision=$(git_quiet -C "$WORK/$name" rev-parse HEAD)
            overrides+=(--override "$name=$repository@$revision")
        else
            value=$(pin_of "$name")
            place "$name" "$value"
        fi
    done
    repository=$(cat "$WORK/.repository-unified" 2>/dev/null || repository_of unified)
    revision=$(git_quiet -C "$WORK/unified" rev-parse HEAD)
    CANDIDATE=unified CANDIDATE_REPOSITORY=$repository CANDIDATE_REVISION=$revision \
    UNIFIED_REPOSITORY=$repository UNIFIED_REVISION=$revision \
        "$PYTHON" "$WORK/unified/brainscore/validation/revisions.py" \
        --manifest "$WORK/unified/install/peer-revisions.json" --record "$RECORD" \
        ${overrides[@]+"${overrides[@]}"}
    for name in core vision language unified; do
        if ! git_quiet -C "$WORK/$name" diff --quiet HEAD || \
                [ -n "$(git_quiet -C "$WORK/$name" ls-files --others --exclude-standard)" ]; then
            echo "note: $name has uncommitted changes; they are built and tested" >&2
        fi
    done
}

phase_pins() {
    local args=() name
    for name in core vision language unified; do
        if [ -e "$WORK/$name" ]; then
            args+=("--$name" "$WORK/$name")
        fi
    done
    "$PYTHON" "$WORK/unified/install/check_pins.py" ${args[@]+"${args[@]}"}
}

phase_print_env() {
    [ -n "$JOB" ] || die 'print-env needs --job'
    env | grep -E '^(RESULTCACHING_DISABLE|HF_HUB_OFFLINE|TRANSFORMERS_OFFLINE|BS_INSTALL_DEPENDENCIES|UMI_TEST_CPU_ONLY|JAX_PLATFORMS)=' | sort
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
        if [ -n "$value" ]; then
            overrides+=(--override "$name=$value")
        fi
    done
    UNIFIED_REVISION=$(git_quiet -C "$WORK/unified" rev-parse HEAD) \
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
    if [ "$(uname -s)" != Linux ]; then
        echo 'CPU Torch is installed on Linux only'
        return
    fi
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
    cd "$TMP_DIR"
    "$PYTHON" -I -m brainscore.validation.workspace --root "$WORK" \
        --mode installed --wheels "$WORK/wheels" --out "$QUAL/installed.json"
}

phase_notebooks() {
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
    fetch_clone openpi "$OPENPI_REPOSITORY" "$OPENPI_REVISION"
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

record_environment() {
    "$PYTHON" -m pip freeze > "$QUAL/dependencies.txt"
    git_quiet -C "$WORK/unified" rev-parse HEAD > "$QUAL/unified-revision.txt"
    git_quiet -C "$WORK/core" rev-parse HEAD > "$QUAL/core-revision.txt"
}

phase_openpi_client_record() { record_environment; }

phase_openpi_instrumentation_record() {
    record_environment
    git_quiet -C "$WORK/openpi" rev-parse HEAD > "$QUAL/openpi-revision.txt"
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
CLIENT_PHASES="verify-core openpi-client-install openpi-client-record openpi-client-test"
INSTRUMENTATION_PHASES="verify-core openpi-checkout openpi-torch openpi-instrumentation-install openpi-instrumentation-record openpi-instrumentation-test"

# Phases change directory with absolute paths only. Do not call run_phase in an
# if, || or && context: Bash then ignores errexit inside the phase.
run_phase() {
    echo "==> $1"
    "phase_$(echo "$1" | tr '-' '_')"
}

# Run one job like a GitHub Actions job: a fresh virtual environment (with
# --venv), its own output folder and environment, and each phase in a new
# process, as each workflow step is. After a failed phase, only the record
# phases (if: always() in the workflow) still run. Returns 1 on any failure.
run_job() {
    local job=$1 phases=$2
    (
        failed=0
        export UMI_QUALIFICATION=$WORK/qualification/$job
        mkdir -p "$UMI_QUALIFICATION"
        if [ -n "$VENV" ]; then
            if [ -e "$VENV/$job" ] || [ -L "$VENV/$job" ]; then
                echo "run_candidate: $VENV/$job already exists" >&2
                exit 1
            fi
            "${UMI_BASE_PYTHON:-python3.11}" -m venv "$VENV/$job" || exit 1
            export PATH="$VENV/$job/bin:$PATH" UMI_PYTHON="$VENV/$job/bin/python"
        fi
        for phase in $phases; do
            if [ "$failed" = 1 ]; then
                case "$ALWAYS_PHASES" in
                    *" $phase "*) ;;
                    *) echo "==> $phase skipped after a failure"; continue ;;
                esac
            fi
            if ! "$BASH" "$SELF" --work "$WORK" --job "$job" "$phase"; then
                echo "FAIL: $job/$phase" >&2
                failed=1
            fi
        done
        exit "$failed"
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
            summary=
            status=0
            for job in offline openpi-client openpi-instrumentation; do
                case "$job" in
                    offline) phases=$OFFLINE_PHASES ;;
                    openpi-client) phases=$CLIENT_PHASES ;;
                    openpi-instrumentation) phases=$INSTRUMENTATION_PHASES ;;
                esac
                if run_job "$job" "$phases"; then
                    summary="$summary$job: PASS"$'\n'
                else
                    summary="$summary$job: FAIL"$'\n'
                    status=1
                fi
            done
            printf '%s' "$summary"
            echo "Outputs are in $WORK/qualification"
            exit "$status"
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
