# CI and merge automation

Reviewed October 14, 2026. UMI release remains manual and owned by the release
maintainer. A passing test run does not authorize publishing packages or scores.

## What runs on v2?

| Action | Current trigger | Effect on v2 |
| --- | --- | --- |
| Coordinated tests | PRs and pushes to v2 in all four repositories | Build and test packages; no publication. |
| Jenkins domain tests | Domain PRs selected by changed files | Run selected tests with pinned peers. A successful skip does not qualify the runtime. |
| Vision production scoring and metadata updates | Merged PRs targeting `master` | No automatic v2 merge trigger. |
| Language production scoring | Merged PRs targeting `main`, with scoring changes | No automatic v2 merge trigger. |
| PyPI releases | Production `main`/`master` pushes or explicit manual dispatch | No automatic v2 push trigger. The three `pypi` environments permit only their production branch. |
| Unified candidate qualification | Manual dispatch with peer commit IDs | Builds and tests packages; does not publish them. |
| Infrastructure image publication | `main` changes under `docker/`, `scoring/`, or `resource_tracking/`; manual dispatch | The Jenkins script changes under `ci/` do not rebuild or publish images. |

## PR-time controls still need attention

These are existing automation paths, separate from merge-triggered publication:

- Vision mapping accepts a mapping label or a web-submission branch without a
  target-branch restriction. Its auto-merge workflows also lack that restriction.
- Language's submission orchestrator can generate metadata, perform mapping,
  commit changes, and auto-merge qualifying web submissions without restricting
  the PR target to `main`.
- Vision's status-triggered observer lives on the default branch. Editing only
  the v2 copy does not control every status-event action; see GitHub's
  [status-event rules](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#status).

Until explicit branch guards are implemented, keep v2 work on ordinary feature
branches and do not apply mapping or auto-merge submission labels. Required
checks prevent merging failing code; they do not require a human to initiate a
merge after the checks pass. Manual metadata and publication workflows must not
be dispatched as part of qualification.

To close the automation portion of roadmap #16, enforce the agreed target-branch
rules and test both PR and status-event routes without triggering production
scoring, metadata writes, emails, or publication. The release maintainer owns
these controls; CI maintainers implement and verify them.

## Live Jenkins acceptance

For a Vision v2 PR that selects tests, retain evidence of:

1. The infrastructure commit, container image, Python version, exact PR commit,
   and pinned Core revision used by the worker.
2. A fresh environment, successful dependency/runtime checks, executed tests,
   and the final test status. Setup failures and skipped selections are separate
   outcomes.
3. Worker termination and cleanup after completion or failure.

[Vision #2601](https://github.com/brain-score/vision/pull/2601) adds a real CPU
wrapper/adapter parity test and selects the non-plugin job. Its synthetic images
and fixed weights test runtime compatibility, not scientific score parity.
The full scientific Vision matrix remains a separate qualification.

## Sources

The review inspected both v2 and default-branch workflows, plus live GitHub
branch protection and `pypi` environment policies. No publishing workflow was run.
Manual workflow availability and its end-to-end execution were not qualified.

- [Core production workflows, `2e1cb4e`](https://github.com/brain-score/core/tree/2e1cb4ea2708cb2a7df77b2a77bb1c10e445cb17/.github/workflows).
- [Vision v2 workflows, `5c2e793`](https://github.com/brain-score/vision/tree/5c2e7938bc66390b37dd87e8fb8a33187723b7a2/.github/workflows) and [production workflows, `d60041d`](https://github.com/brain-score/vision/tree/d60041d28dbf52759a733e3b3b57a0b41df83be6/.github/workflows).
- [Language v2 workflows, `35fe1a9`](https://github.com/brain-score/language/tree/35fe1a96ca219565a01143d7e10d9a9e27720166/.github/workflows) and [production workflows, `34a8800`](https://github.com/brain-score/language/tree/34a880087d819ebebb9d02b3aff9078b5f5d688e/.github/workflows).
- [Unified workflows, `f411ac1`](https://github.com/KartikP/brainscore-unified/tree/f411ac1aedeea3dc24f99dfc2a6668bfb166a2be/.github/workflows).
- [Infrastructure, `edfe9fe`](https://github.com/brain-score/infrastructure/tree/edfe9fe3dffbed857ae02ccef1aef05b725b0063).
