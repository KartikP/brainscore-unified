# Running tests

## Layout

A test file sits in `tests/test_<part>/`, where `<part>` is the part of the `brainscore` package that it tests (`models`, `model_helpers`, `benchmarks`, `data`, `metrics`, `harnesses`, `experiments`, `tools`, `validation`, `visualization`; `model` for the private `_model` part). A test of one benchmark plugin is in that plugin's folder: `brainscore/benchmarks/<name>/test.py`, and more files as `test_<topic>.py` beside it; `tests/test_benchmarks/` keeps the tests of shared benchmark code and of several benchmarks. A file that tests the package as a whole (discovery, install, documents, templates, architecture rules, top-level modules) is in `tests/test_package/`. `tests/brainscore_model/` holds the `BrainScoreModel` tests. Shared helpers (`*_stand_in.py`, `pseudo_benchmarks.py`, `conftest.py`, `markers.py`, `fixtures/`) stay in `tests/`; import them as `from tests.<helper> import ...`.

Use the coordinated environment from [install/README.md](../install/README.md).

## Four-repository offline checks

From the directory containing `core/`, `vision/`, `language/` and `unified/`:

```bash
UMI_TEST_CPU_ONLY=1 python -m brainscore.validation.workspace --root . --out qualification/source.json
```

The runner tests each repository in a separate process. It records revisions, local changes, selected tests, failures, skip reasons and unselected files. Unexpected skips, empty suites, missing reports and timeouts fail. The [selection](../brainscore/validation/workspace.py) includes shared UMI code and legacy helpers; it does not run every plugin. `UMI_TEST_CPU_ONLY=1` disables automatic CUDA/MPS selection in all four test suites; device-selection tests can override it. Runtime model selection is unchanged.

GitHub Actions builds the four pinned wheels, runs installed examples, and executes these source checks on Ubuntu/macOS CPU. Core, vision and language use explicit selections. Unified uses these markers:

| Marker | Purpose |
| --- | --- |
| `unit` | Offline checks; default for tests without a tier |
| `integration` | Offline component combinations, usually with fixture models |
| `slow` | Real models, datasets or expensive computation |
| `private_access` | Restricted resources, usually combined with `slow` |

For a quick unified-only run, from `unified/` (the same selection as the per-PR runner):

```bash
RESULTCACHING_DISABLE=1 pytest tests brainscore/benchmarks -m "(unit or integration) and not qualification"
```

With no path, pytest collects the same two folders (`testpaths` in `pyproject.toml`). It never collects a model plugin's `test.py`, because that file loads real weights; name the model folder to run it. A benchmark plugin folder can run alone from the checkout (`pytest brainscore/benchmarks/<name>`) with the same markers. A plugin folder collected from an installed wheel without the checkout is not supported, because its tests import helpers from `tests/`. This plain command permits skips. Use the coordinated runner for required-check accounting. Fixture tests and scientific qualification are separate evidence.

## OpenPI client checks

The separate Linux job needs only core and the client package. From the workspace root, in a fresh Python 3.11 environment:

```bash
python -m pip install -c unified/install/v2-constraints.txt \
    ./core -r unified/install/openpi-client-tests.txt
python -m pip check
cd unified
PYTHONPATH="$PWD:$PWD/../core" python -m pytest tests/test_experiments/test_openpi_remote.py \
    -q -ra -p brainscore.validation.required_checks \
    --umi-repository openpi-client --umi-check-report qualification/openpi.json \
    --timeout=30
```

These tests use real local WebSockets with fixture policies. Every selected case must run: no skips are allowed. No JAX, Torch, GPU, trained checkpoint or simulator is needed. The CPU flag has no effect when Torch is absent.

The main profile's 27 missing-client exemptions do not apply to this job. Local JAX instrumentation and trained-policy comparisons have separate requirements in the matrix.

## Real models and data

Run the matrix's integration/qualification selections only after staging their data and checkpoints. `pytest -m slow` is not a complete release qualification command. Scientific compatibility uses all six cases (four MajajHong2015 PLS, two Pereira2018 ridge) in `RESULTCACHING_DISABLE=1 python -m brainscore.validation.run_parity`; the matrix records the other required domain comparisons.

Keep submission/database tests in isolated services without production credentials. Save failed runs alongside successful results.
