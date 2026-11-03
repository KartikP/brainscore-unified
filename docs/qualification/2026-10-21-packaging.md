# Packaging verification — October 21, 2026

Built all four packages from clean staging directories with Python 3.11 and
setuptools 68.2.2. Installed the wheels in a fresh Python 3.11 virtual environment,
using `install/v2-constraints.txt`. No existing environment was modified.

## Results

| Installed package | Required offline tests passed |
| --- | ---: |
| Core | 791 |
| Vision | 239 |
| Language | 68 |
| Unified | 1,309 |
| Total | 2,407 |

There were 43 allowed skips, zero unexpected skips, and 62 deselected scientific
cases. This is the required offline profile, not every plugin integration test.
The runner copied tests outside the checkouts, verified installed files against
wheel hashes, isolated home/cache directories, and disabled AWS credentials.

All four wheels pass the contents check. Previously packaged license notices
are retained. Vision no longer installs global `tests` or `scripts` packages;
core omits internal BrainIO tests. The vision wheel changed from approximately
37.4 to 30.5 MiB; language changed from 0.75 to 0.57 MiB.

All 15 selected offline UMI tutorials executed successfully against the installed
packages in fresh kernels. Four download/data-dependent tutorials were explicitly
excluded. The doctor and both partner examples pass with a temporary home and
explicit cache directories. All 17 retained domain notebooks pass syntax checks. Their
full dataset-dependent scientific workflows were not executed.

## Earlier unsuccessful checks

An initial sandboxed run encountered process-inspection and Jupyter-socket
permission errors and was stopped. The final run permitted those local
operations and passed. An initial doctor invocation used the host's unwritable
cache; its replacement used isolated directories and passed. Neither initial
attempt is counted as successful qualification.

## Scope

The archive, portability decisions, and licensing boundary are described in
[package contents](../package_contents.md). Licenses are preserved, not newly
approved by these checks. Historical source review and scientific CPU/GPU runs
remain separate tasks. No packages were published and no scores were changed.

Wheel hashes and check counts: [machine-readable results](2026-10-21-packaging.json).
Retained notebook inventory: [file list](2026-10-21-packaging-inventory.json).

Reproduce from the coordinated peer revisions:

```sh
python -m build --wheel --no-isolation --outdir wheels core
python -m build --wheel --no-isolation --outdir wheels vision
python -m build --wheel --no-isolation --outdir wheels language
python -m build --wheel --no-isolation --outdir wheels unified
python unified/brainscore/validation/package_contents.py \
    --wheels wheels --out qualification/package-contents.json
```

Install into a fresh Python 3.11 environment using the commands in
`.github/workflows/offline-tests.yml`, then run `brainscore.validation.workspace`
with `--mode installed` from outside the source checkouts. Build from clean
checkouts or staging directories so old build directories cannot leak files.
