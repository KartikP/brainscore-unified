# Coordinated CI

Every v2 PR should test the four packages together. GitHub Actions runs the
offline checks; Jenkins runs selected plugin tests and data-dependent checks.

## Which revisions run?

The reusable [offline workflow](../.github/workflows/offline-tests.yml) takes the
repository under test, its source repository and its full commit SHA. It replaces
only that peer's entry in `install/peer-revisions.json`. Fork PRs use the fork's
head commit. Unified PRs use their own manifest and head commit.

Before building, the workflow verifies every checkout against the requested
commit. `qualification/revisions.json` records the resolved set. Tests run
against built wheels outside the checkouts, with isolated application caches,
no private credentials and explicit skip accounting. The workflow does not
publish packages or scores.

Core, Vision and Language callers pin both the reusable workflow and its Unified
checkout to the same immutable commit. Update these together after a coordinated
baseline changes. A peer workflow uses only a read-only GitHub token; it does not
inherit secrets or dispatch a privileged workflow in another repository.

## Required checks

These checks are required on `unified-model-interface-v2` in all four
repositories, verified on October 15, 2026. Protection applies to administrators
and requires branches to be up to date before merging.

| Repository | Coordinated checks | Existing domain checks to retain |
| --- | --- | --- |
| Unified | `offline (ubuntu-24.04)`, `offline (macos-26)`, `OpenPI client protocol`, `OpenPI local instrumentation` | None |
| Core | `UMI v2 / offline (ubuntu-24.04)`, `UMI v2 / offline (macos-26)`, `UMI v2 / OpenPI client protocol`, `UMI v2 / OpenPI local instrumentation` | `Core Unit tests (AWS Jenkins, AWS Execution)` |
| Vision | Same four `UMI v2 / ...` checks | `Vision Unittests, Non-Plugins`, `Vision Unittests, Plugins` |
| Language | Same four `UMI v2 / ...` checks | `Language Unittests, Non-Plugins`, `Language Unittests, Plugins`, `Language Integration Tests` |

When changing this list, confirm the reported names and read back the protection
settings afterward. An infrastructure PR alone does not qualify the live Vision
v2 environment. See [merge automation](ci_automation.md) for the other actions
that PRs and merges can trigger.

## Longer integration tests

Reuse the existing Jenkins selectors: a model change selects its model tests,
a benchmark change selects its benchmark tests, and shared-helper changes select
the broader domain suite. Each plugin should include a small representative
model/benchmark pair, expected output shape and score behavior, and its data and
resource requirements. These checks complement the offline four-package suite.
The [targeted integration selections](integration_tests.md) define the proposed
first profiles and the checks needed before enabling them in Jenkins.

Full vision qualification, trained-policy qualification and performance budgets
remain separate profiles in the [release test matrix](release_test_matrix.md).
Missing data, an empty test selection or a resource failure is not scientific
qualification. The first-use trial remains under roadmap #2.

## Notebook and local policy checks

The offline jobs also execute the [15 offline tutorials](../notebooks/README.md#automated-offline-check)
against installed packages. Their reports and executed copies are retained with
the other qualification artifacts. No weights or private datasets are required.

`OpenPI local instrumentation` uses a separate CPU environment with
[explicit dependency pins](../install/openpi-instrumentation-tests.txt) and OpenPI
revision `215abfb217dbac7d5f1273282331b9b1866c0479`. It runs
`test_openpi_instrumentation.py` and `test_libero_transport.py` with the
`openpi-instrumentation` reporting profile, which permits no skips. These small
JIT/loopback tests fit hosted CI; trained-policy and simulator runs stay separate.
The source checkout is used for this profile; it is not installed-wheel or
trained-DROID qualification.

Hosted qualification passed. All three peer callers now pin the workflow and
runner to Unified `78074cbf7a9bbfacde09bb21abcc95981a51d060`. Both OpenPI jobs
are required merge checks in all four repositories. The notebooks run inside
the already-required offline jobs.

To repeat the local policy profile, create a Python 3.11 environment, check out
OpenPI at the revision above, and run from the Unified checkout:

```bash
python -m pip install -c install/v2-constraints.txt ../core \
  -r install/openpi-instrumentation-tests.txt
python -m pip check
OPENPI_SOURCE=/path/to/pinned/openpi PYTHONPATH=. JAX_PLATFORMS=cpu \
  RESULTCACHING_DISABLE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python -m pytest tests/test_openpi_instrumentation.py tests/test_libero_transport.py \
  -q -ra -p brainscore.validation.required_checks \
  --umi-repository openpi-instrumentation \
  --umi-check-report /tmp/openpi-instrumentation.json --timeout=120
```
