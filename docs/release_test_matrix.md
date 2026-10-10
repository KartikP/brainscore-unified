# Release test matrix

Use GitHub Actions for short, isolated checks. Use Jenkins workers for changed plugins, staged datasets, services, and longer CPU/GPU evaluations. Both should consume the same four-repository revision manifest and report the exact tests run.

This is the test-routing proposal for [#18](https://github.com/brain-score/unified-model-interface/issues/18), reviewed October 7, 2026. Rows marked **proposed** are not configured jobs. Final required capabilities and environments depend on [#6](https://github.com/brain-score/unified-model-interface/issues/6). Role names below identify responsibility, not new assignments to people.

## Verified baseline

[GitHub run 37627147300](https://github.com/KartikP/brainscore-unified/actions/runs/37627147300) passed on Ubuntu 24.04 and macOS 14, Python 3.11, CPU:

| Check | Result on each platform |
| --- | --- |
| Coordinated wheel build, installation and dependency check | Passed |
| Installed-package tests outside source | 28 passed; partner examples also passed |
| Core source | 355 passed |
| Vision source | 75 passed |
| Language source | 63 passed |
| Unified source | 955 passed, 43 allowed skips, 64 deselected |
| Source total | 1,448 passed, zero unexpected skips |

Tested revisions: core `0508b51`, vision `afe286b4`, language `47e57dc`, unified `01135cb`. Vision's subsequent README-only commit `e7de68af` is not in this pinned set. These counts are test outcomes, not coverage percentages or scientific qualification.

The source profile reports 36 core, 31 vision, 9 language and 6 unified test files with no selected test IDs. Some are collection-skipped modules. The inventory covers `tests/test_*.py`, not the full plugin catalog or embedded BrainIO test tree. Do not interpret the omitted files as uniformly failing or unnecessary.

## One execution matrix

**Implemented** means a current runner executes the stated selection. **Proposed** means installation, selection or scheduling still needs implementation and verification. Limits in parentheses are existing limits, not measured runtimes. New compute budgets must be set before scheduling.

| ID / check | Selection and purpose | Runner / trigger | Prerequisites and completion criterion | Status / responsible role |
| --- | --- | --- | --- | --- |
| F1 — Coordinated offline baseline | [workspace selection](../brainscore/validation/workspace.py): contracts, adapters, cache integrity, preflight, tools, records, reasoning, synthetic robotics and Gymnasium; plus installed tests/examples | GitHub Actions, unified PRs and v2 pushes; Ubuntu/macOS CPU (40-minute job; 30 minutes per source suite) | Four pinned revisions, constraints, built wheels and test extra. Every required case runs; no unexpected skips, empty suites or missing reports | **Implemented**; CI maintainer |
| F2 — Additional shared and legacy tests | Add eligible omitted assembly, readout, unit selection, memory, streaming, metric, preprocessing and plugin-selection checks. Split mixed files at test-case level | GitHub Actions, proposed extension to F1 | No network, private data or model download. Record selected IDs and timings before promotion; retain needed cases in integration profiles if they cannot run offline | **Proposed**; core/domain maintainers |
| F3 — Remote policy protocol | `tests/test_openpi_remote.py`: loopback requests, timeouts, disconnects, invalid responses, cleanup, concurrent-client handling | GitHub Actions, dedicated Linux CPU job on relevant PRs and v2 pushes | Pin/install `openpi_client` and test dependencies in an isolated compatible environment. All selected cases run, including the 27 currently skipped cases; local sockets only, no trained weights | **Proposed**; robotics/tool maintainer |
| I1 — Changed-plugin integration | Existing vision/language plugin selectors, plus shared plugin installation/import/environment tests. Model PR: declared smoke benchmark; benchmark PR: representative compatible models. Shared-code changes: representative pairs in both domains | Jenkins, relevant PR head; scheduled sweep for shared changes | Coordinated revisions, plugin dependencies, data/checkpoint access and CPU/GPU sizing. Assert nonempty expected tests for changed plugins; setup, OOM, timeout and failed tests cannot qualify a change | Existing domain runners; **UMI coordination proposed**; domain + CI maintainers |
| I2 — Real-model and data integration | Legacy vision transformation/integration tests; language embedding/Hugging Face/container/localization tests; unified `test_text_wrapper.py`, `test_vlm_vision_wrapper.py`, `test_roar_benchmark.py`; representative audio/video adapters | Jenkins, relevant PRs plus scheduled CPU/GPU jobs | Pre-staged, identified weights/data; licensed access where required. Verify loading, output shape/semantics, repeated calls and recorded expected results. Use GPU for cases that require it, not the whole tier | **Proposed coordinated selection**; domain maintainers |
| I3 — Submission and storage integration | Core/vision/language submission tests, metadata-only flows, compatible/incompatible/failed score persistence | Jenkins, relevant PRs and candidate runs | Disposable database/service fixtures and test credentials. Verify score provenance and failure/N/A behavior; no production writes. Move pure mocked cases to F2 | Existing tests; **isolated coordinated profile proposed**; scoring/database maintainers; #22 |
| I4 — Local policy instrumentation and transport | `test_openpi_instrumentation.py`, `test_libero_transport.py`; OpenPI/JAX internal recording, interventions and reference/UMI transport | Dedicated pinned OpenPI environment on Jenkins; relevant changes and candidate runs | Pinned OpenPI source/JAX dependencies. Remove the two collection/case skip exceptions in this profile. Small-model tests must run and clean up correctly; this is not trained-policy success | **Proposed automation**; robotics/tool maintainer |
| I5 — Atlas-dependent checks | Five skipped LanA tests in `test_lebel2023_language_mask.py` | Jenkins data job, relevant changes and candidate if mask support is selected | Licensed/available atlas, checksum, declared mask convention. All five tests run; missing atlas fails this profile | **Proposed**; language/data maintainer |
| Q1 — Scientific compatibility | All eight `RELEASE_CASES` through [run_parity](../brainscore/validation/run_parity.py); legacy/adapter/native comparisons, four repaired Pereira cases; anchored regression pairs and current-upstream HMAX control | Jenkins scheduled qualification and exact final candidate; supported CPU and NVIDIA environments | Fixed weights/data, FP32 device policy, cold extraction, sufficient RAM. Adapter exactness and native numerical budgets from [numerical policy](numerical_policy.md); retain raw/normalized scores and failed observations. Historical HMAX mismatch stays separate under #21 | Existing manual harness; **scheduled/final-artifact qualification proposed**; scientific/domain maintainers; #19/#20 |
| Q2 — Trained robotics | Reference versus UMI OpenPI/LIBERO recordings, actions, tool cleanup and task success; separate trained-DROID evaluation on fixed episodes | Jenkins approved GPU/simulator runs; relevant changes and candidate | Pinned checkpoint, evaluator, seeds, task/episode manifest and action semantics. Compare reference and UMI on identical cases. LIBERO smoke success does not qualify trained DROID or full-suite reproduction | Recorded LIBERO smoke evidence; **automation and DROID qualification proposed**; robotics/scientific maintainers |
| Q3 — Cache and resource cost | Real CLIP/Pereira cold/warm scoring; changed input/weight invalidation; 3B-class hashing and trained inference; large stimulus-file hashing | Jenkins CPU/GPU performance profile after extraction changes and on candidate | Same model/data/settings across paired runs; isolated caches. Warm unchanged-weight validation: median ≤50 ms, zero weight-byte rereads. Report first hash, full fingerprint, file reads, runtime, memory and score separately; set other budgets before accepting them | Local CPU evidence; **broader qualification proposed**; performance/domain maintainers |
| D1 — Tutorials and figures | Execute offline notebooks/examples; redraw archived figures; run selected measured-data notebooks with supplied assets | GitHub Actions for lightweight offline examples; Jenkins for data/model reruns, on relevant changes and candidate | No execution errors, correct links/assets, explicit measured versus illustrative outputs. Image rendering is not a new scientific measurement; verify numerical captions against exported metadata | Partner examples in F1; **notebook automation proposed**; docs/domain maintainers |
| O1 — Operational/publication checks | One root-hygiene module and eight Algonauts submission-bundler skips; package assets, licenses, repository contents | Maintainer/operations validation; package or submission changes | Run external tools at pinned paths in their owning workspace, or replace obsolete tests with tests of a maintained implementation. Require schema/archive checks for advertised submission paths. Do not classify these as scientific passes | **Proposed routing**; publication/submission maintainers; #7/#13 |

A new model needs targeted integration with existing benchmarks. A new benchmark needs representative models. Neither requires every catalog plugin on every PR. The release profile still needs an explicit representative set spanning legacy, adapter and native paths.

## Close the skip and selection gaps

The 43 allowed skips in F1 consist of **27 remote OpenPI cases (F3), one local OpenPI module and one LIBERO transport case (I4), five LanA cases (I5), and nine operational entries (O1)**. A module-level skip can hide multiple tests; one reported skip is not one qualified behavior.

The 64 deselected cases span `test_benchmark_parity_slow.py`, `test_regression_baselines.py`, `test_roar_benchmark.py`, `test_text_wrapper.py` and `test_vlm_vision_wrapper.py`. Route parity/baselines to Q1 and real-wrapper/ROAR cases to I2. Move cheap registry-only assertions into F2 rather than keeping them behind a file-wide slow marker.

For each promoted selection, store exact test IDs, dependencies, environment, expected collection and an owner role. A supported case may be excluded from the fast profile only when its other required profile is identified. In that required profile, missing assets and skips fail; an explicit unsupported scope decision is different from a passing test.

## Shared execution rules

- **Revision set:** start from `install/peer-revisions.json`; for a peer PR, override that peer with its exact head SHA and record all four resulting revisions. A unified-only trigger cannot protect core/vision/language PRs. Add cross-repository triggering and check reporting before calling this a per-repository merge gate.
- **One test implementation:** reuse the repository runners and domain selectors. GitHub/Jenkins should supply environments and schedules, not separate assertions.
- **Report actual work:** save selected/deselected/skipped IDs and reasons, JUnit or equivalent outcomes, all revisions, wheel hashes, dependency lock, data/checkpoint identifiers, precision/device, cache settings, durations and resource use. A zero exit with absent evidence is a failure. An OOM/UNSTABLE Jenkins outcome cannot satisfy a required release check.
- **Bound expensive runs:** check dependencies, credentials/data availability, disk and expected memory before downloading/loading models. Set a hard runtime/compute limit and verify cleanup; an alarm that only sends a notification is insufficient.
- **Keep cache tests intentional:** most structural tests bypass result caching; integrity tests use temporary writable caches and must demonstrate both hits and invalidation. Scientific parity uses cold execution; performance reports cold and warm separately.
- **Final artifact:** rerun required profiles on the actual coordinated candidate wheels and environment, not merely editable source. Historical results do not qualify changed code, dependencies or devices.

## Implementation order

1. Promote the verified offline shared-code selections (F2) and add the client-only remote protocol job (F3). Preserve fail-on-unexpected-skip behavior.
2. Reconcile Jenkins deployment records; add a coordinated UMI revision/install path and exact-head reporting without changing ordinary production jobs implicitly (#16).
3. Define changed-plugin and representative-pair manifests, then add isolated integration profiles (I1–I5) with explicit resource limits.
4. After #6 scope approval, bind each required capability/environment to these rows and run Q1–Q3 on the final candidate (#19/#20).

This document configures no jobs, launches no compute, and authorizes no publication. Release execution remains manual and owned by Kartik.
