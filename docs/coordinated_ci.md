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
repositories, verified on October 14, 2026. Protection applies to administrators
and requires branches to be up to date before merging.

| Repository | Coordinated checks | Existing domain checks to retain |
| --- | --- | --- |
| Unified | `offline (ubuntu-24.04)`, `offline (macos-14)`, `OpenPI client protocol` | None |
| Core | `UMI v2 / offline (ubuntu-24.04)`, `UMI v2 / offline (macos-14)`, `UMI v2 / OpenPI client protocol` | `Core Unit tests (AWS Jenkins, AWS Execution)` |
| Vision | Same three `UMI v2 / ...` checks | `Vision Unittests, Non-Plugins`, `Vision Unittests, Plugins` |
| Language | Same three `UMI v2 / ...` checks | `Language Unittests, Non-Plugins`, `Language Unittests, Plugins`, `Language Integration Tests` |

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

Full vision qualification, trained-policy qualification and performance budgets
remain separate profiles in the [release test matrix](release_test_matrix.md).
Missing data, an empty test selection or a resource failure is not scientific
qualification. The first-use trial remains under roadmap #2.
