# Running tests

Use the coordinated environment from [install/README.md](../install/README.md). The [release matrix](../docs/release_test_matrix.md) maps capabilities to tests, real-model comparisons and runner requirements.

## Four-repository offline checks

From the directory containing `core/`, `vision/`, `language/` and `unified/`:

```bash
python -m brainscore.validation.workspace --root . --out qualification/source.json
```

The runner tests each repository in a separate process. It records revisions, local changes, selected tests, failures, skip reasons and unselected files. Unexpected skips, empty suites, missing reports and timeouts fail. The [selection](../brainscore/validation/workspace.py) includes shared UMI code and legacy helpers; it does not run every plugin.

GitHub Actions builds the four pinned wheels, runs installed examples, and executes these source checks on Ubuntu/macOS CPU. Core, vision and language use explicit selections. Unified uses these markers:

| Marker | Purpose |
| --- | --- |
| `unit` | Offline checks; default for tests without a tier |
| `integration` | Offline component combinations, usually with fixture models |
| `slow` | Real models, datasets or expensive computation |
| `private_access` | Restricted resources, usually combined with `slow` |

For a quick unified-only run:

```bash
pytest -m "unit or integration"
```

This plain command permits skips. Use the coordinated runner for required-check accounting. Fixture tests and scientific qualification are separate evidence.

## OpenPI client checks

The separate Linux job needs only core and the client package. From the workspace root, in a fresh Python 3.11 environment:

```bash
python -m pip install -c unified/install/v2-constraints.txt \
    ./core -r unified/install/openpi-client-tests.txt
python -m pip check
cd unified
PYTHONPATH="$PWD:$PWD/../core" python -m pytest tests/test_openpi_remote.py \
    -q -ra -p brainscore.validation.required_checks \
    --umi-repository openpi-client --umi-check-report qualification/openpi.json \
    --timeout=30
```

These tests use real local WebSockets with fixture policies. Every selected case must run: no skips are allowed. No JAX, Torch, GPU, trained checkpoint or simulator is needed. Leave `UMI_TEST_CPU_ONLY` unset in this minimal environment; that flag configures Torch tests in the main profile.

The main profile's 27 missing-client exemptions do not apply to this job. Local JAX instrumentation and trained-policy comparisons have separate requirements in the matrix.

## Real models and data

Run the matrix's integration/qualification selections only after staging their data and checkpoints. `pytest -m slow` is not a complete release qualification command. Scientific compatibility uses all eight cases in `python -m brainscore.validation.run_parity`; the matrix records the other required domain comparisons.

Keep submission/database tests in isolated services without production credentials. Save failed runs alongside successful results.
