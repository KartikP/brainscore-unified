# Release test matrix

Use GitHub Actions for short, isolated checks. Use Jenkins workers for changed plugins, staged datasets, services, and longer CPU/GPU evaluations. Both should consume the same four-repository revision manifest and report the exact tests run.

This matrix defines the checks for [#18](https://github.com/brain-score/unified-model-interface/issues/18). It distinguishes automated checks from scientific qualification. Final support approval belongs to [#6](https://github.com/brain-score/unified-model-interface/issues/6); execution across candidate environments belongs to #19/#20. Roles below identify responsibility, not new assignments.

## Current validation

[Post-merge CI at `cdb5eb1`](https://github.com/KartikP/brainscore-unified/actions/runs/39424650356) passed on October 20, 2026. Both Ubuntu 24.04 and macOS 14 ran the four installed packages outside their source checkouts.

| Installed package | Passing tests on each platform | Scope |
| --- | ---: | --- |
| Core | 791 | Contracts, shared utilities, dispatch, metrics and preflight |
| Vision | 238 | Adapters, cache, transformations, layer mapping and plugin checks |
| Language | 68 | Adapters, legacy scoring, preprocessing and metadata checks |
| Unified | 1,301 | Offline checks; 43 allowed skips and 62 deselected cases |

Each platform passed **2,398 tests**, with no failures or unexpected skips. The dedicated OpenPI client and local instrumentation jobs also passed 42 and 25 tests respectively, with no skips. Those jobs overlap the offline selection; do not add their counts as unique tests. These are automated interface and installation checks, not real-model qualification or a new coverage measurement.

F1 verifies installed files against wheel hashes, checks module import locations, and records versions and skip reasons. Each suite uses temporary home and cache directories. Reports identify the exact revisions and wheels tested; earlier source results do not qualify later wheels. The 31 integration-runner and 14 installation-check tests passed on each platform.

## Capability acceptance

File names below refer to `tests/` in the named repository. Exact selected pytest IDs and skip reasons are emitted by each run. Fixture tests prove interface behavior; real-model measurements prove only their recorded configurations.

| Capability | Automated evidence | Acceptance / further qualification |
| --- | --- | --- |
| Subject, channels and external extensions | Core `test_native_subject.py`, `test_external_extensions.py`, `test_payload_validation.py`; installed partner examples | Declare inputs/outputs, reject invalid payloads, reset and extend without contract edits; independent author trial under #2 and final-candidate review |
| Legacy vision and language | Domain `test_unified_adapter.py`; language `test_legacy_score.py`; core readout/metric tests | Preserve legacy calls and outputs; final eight-case comparison under Q1 |
| Neural scoring and layer mapping | Core unit/composite selection tests; vision layer-mapping classes; unified `test_benchmark_parity.py` | Correct unit ordering, region mapping, fitting and numerical-policy enforcement; real-data Q1 |
| Behavioral and generated answers | `test_api_behavioral.py`, `test_roar_registry.py`, `test_ewok.py`, `test_coggym_benchmark.py` | Correct channels, parser, split/score identity and invalid-answer handling; real ROAR comparison below |
| Multimodal and temporal inputs | Core `test_multimodal.py`, `test_temporal.py`; unified audio/video wrappers and Lahner routing tests | Preserve modalities, order, timestamps and alignment; representative real-model checks below |
| Streaming and environments | Core streaming tests; `test_environment_sessions.py`, `test_gymnasium_harness.py` | Enforce delivery, declared spaces, action horizon, reset and clock semantics; no hard real-time claim |
| Tools and perturbations | `test_experiments.py`, `test_perturbation.py`, `test_scale_activity.py`, `test_attribution_patching.py`, `test_wrapper_cleanup.py` | Baseline and recording-only outputs agree; interventions affect selected activity; attribution estimates are checked against exact patches; state/hooks restore after success, failure and cancellation |
| Records, reasoning and replay | `test_record_validation.py`, `test_reasoning_recorder.py`, `test_production_tools.py` | Preserve exposed responses/activity/provenance; reject corrupt or incomplete records; replay saved measurements and explicitly rerun calls |
| Robotics and OpenPI | `test_robotics_validation.py`, `test_droid_policy.py`; F3 and I4 | Validate action semantics, requests, timeouts and cleanup; trained DROID direct-versus-UMI comparison under Q2 |
| Reliability and performance | Core preflight/cache/memory tests; domain cache-integrity tests; unified extraction tests | Fail cheap invalid prerequisites early, reject stale data and retain errors; real resource budgets under Q3 |
| Installation and score publication | F1 wheel/examples; core `test_leaderboard_eligibility.py`; I3 | Install coordinated artifacts, block ineligible publication and preserve score identity; isolated production flow under #22 |

### Representative real-model checks

These selections make the proposed domain coverage concrete for #6 review. They are acceptance targets, not claims that the current candidate has passed. Pin checkpoints, dataset builds, seeds, preprocessing, layers and settings in the execution manifest before running; retain reference outputs and full failures.

| Domain | Selected comparison | Pass condition / profile |
| --- | --- | --- |
| Vision | ResNet18 on MajajHong2015 public/private × V4/IT | Legacy, adapter and native routes pass the numerical policy; four Q1 cases |
| Language | GPT-2 on Pereira2018 243/384 sentences × linear/ridge | Legacy, adapter and native routes pass all four Q1 cases and paired drift checks; linear is a compatibility reference, not the preferred scientific protocol |
| Behavioral | `clip-vit-b-32` on `Yeatman2021-lexical_decision-image` | I2's `test_roar_benchmark.py` checks fixed splits, above-chance raw accuracy and ceiling/score structure; retain direct-versus-UMI predictions |
| Multimodal | `clip-wav2vec2` on `Lahner2024-fMRI-naturalistic-multimodal` | Both audio and visual inputs reach their branches; reference feature ordering, alignment and scores agree under a budget fixed before the run; I2 |
| Native video | `videomae-base` on `Lahner2024-fMRI-naturalistic-visualROI` | Temporal order is preserved and reference/UMI features and scores agree under a predeclared budget; I2 |
| Perturbation and tools | Pretrained ResNet18, the fixed Notebook 18 images and layer selection | Baseline = recording-only = restored; selected activity changes during intervention, recorded inputs/outputs replay; I2 |
| Embodied | Trained `pi0_fast_droid` on a fixed held-out DROID episode manifest | Same observations and RNG state produce matching direct/UMI actions under a predeclared numerical policy; validate units, horizon, records and cleanup; Q2 |

The [September candidate](qualification/2026-09-16-candidate.md) contains real vision/language measurements. [OpenPI qualification](qualification/2026-10-06-openpi-tools.md) contains trained-policy and LIBERO smoke evidence. Earlier notebooks/demonstrations do not qualify changed candidate packages. CogGym/EWoK remain research integrations; their registrations, archived outputs and parser checks are not leaderboard replication. Physical robots and closed-loop game success are outside the required DROID open-loop gate.

## One execution matrix

**Implemented** means an existing runner executes the selection. **Proposed** means selection, installation or scheduling still needs implementation. Limits in parentheses are existing limits, not measured runtimes. New compute budgets must be set before scheduling.

| ID / check | Selection and purpose | Runner / trigger | Prerequisites and completion criterion | Status / responsible role |
| --- | --- | --- | --- | --- |
| F1 — Coordinated offline baseline | [workspace selection](../brainscore/validation/workspace.py): contracts, adapters, cache integrity, preflight, tools, records, reasoning, synthetic robotics and Gymnasium; plus examples | GitHub Actions, unified PRs and v2 pushes; Ubuntu/macOS CPU (40-minute job; 30 minutes per suite) | Four pinned revisions, constraints, installed candidate wheels and test extra. Verify package files and import locations; every required case runs, with no unexpected skips, empty suites or missing reports | **Implemented**; CI maintainer |
| F2 — Additional shared and legacy tests | Explicit assembly, readout, unit selection, memory, streaming, metric, preprocessing and plugin-selection checks; offline classes selected from mixed files; ROAR registry tests separated from data tests | GitHub Actions, extension to F1 on both CPU platforms | No network, private data or model download. Record selected IDs and timings before promotion; retain needed cases in integration profiles if they cannot run offline | **Implemented**; core/domain maintainers |
| F3 — Remote policy protocol | `tests/test_openpi_remote.py`: loopback requests, timeouts, disconnects, invalid responses, cleanup, concurrent-client handling | GitHub Actions, dedicated Linux CPU job on unified PRs and v2 pushes | Install [pinned client dependencies](../install/openpi-client-tests.txt) and pinned core. All 42 selected cases run, including the 27 skipped in F1. No skip exemptions; local sockets only, no trained weights/JAX/Torch; 15-minute job, 30-second test limit | **Implemented**; robotics/tool maintainer |
| I1 — Changed-plugin integration | Existing vision/language plugin selectors, plus shared plugin installation/import/environment tests. Model PR: declared smoke benchmark; benchmark PR: representative compatible models. Shared-code changes: representative pairs in both domains | Jenkins, relevant PR head; scheduled sweep for shared changes | Coordinated revisions, plugin dependencies, data/checkpoint access and CPU/GPU sizing. Assert nonempty expected tests for changed plugins; setup, OOM, timeout and failed tests cannot qualify a change | Existing domain runners; [targeted integration runner](integration_tests.md), qualification/deployment pending; domain + CI maintainers |
| I2 — Real-model and data integration | Legacy vision transformation/integration tests; language embedding/Hugging Face/container/localization tests; unified `test_text_wrapper.py`, `test_vlm_vision_wrapper.py`, `test_roar_benchmark.py`; representative audio/video adapters | Jenkins, relevant PRs plus scheduled CPU/GPU jobs | Pre-staged, identified weights/data; licensed access where required. Verify loading, output shape/semantics, repeated calls and recorded expected results. Use GPU for cases that require it, not the whole tier | [Eight profiles](../install/integration-tests.json); all eight executable; asset-backed qualification and scheduling pending; domain maintainers |
| I3 — Submission and storage integration | Core/vision/language submission tests, metadata-only flows, compatible/incompatible/failed score persistence | Jenkins, relevant PRs and candidate runs | Disposable database/service fixtures and test credentials. Verify score provenance and failure/N/A behavior; no production writes. Move pure mocked cases to F2 | Existing tests; **isolated coordinated profile proposed**; scoring/database maintainers; #22 |
| I4 — Local policy instrumentation and transport | `test_openpi_instrumentation.py`, `test_libero_transport.py`; OpenPI/JAX internal recording, interventions, reference/UMI transport, and LIBERO evaluator/tool-bridge completion faults | Dedicated pinned CPU environment in GitHub Actions; PRs, v2 pushes and candidate runs | Pinned OpenPI source/JAX dependencies. Remove the two collection/case skip exceptions in this profile. Small-model tests must run and clean up correctly; this is not trained-policy success | **Implemented CPU automation**; robotics/tool maintainer |
| I5 — Atlas-dependent checks | Five skipped LanA tests in `test_lebel2023_language_mask.py` | Jenkins data job, relevant changes and candidate if mask support is selected | Licensed/available atlas, checksum, declared mask convention. All five tests run; missing atlas fails this profile | **Proposed**; language/data maintainer |
| Q1 — Scientific compatibility | All eight `RELEASE_CASES` through [run_parity](../brainscore/validation/run_parity.py); legacy/adapter/native comparisons, four repaired Pereira cases; anchored regression pairs and current-upstream HMAX control | Jenkins scheduled qualification and exact final candidate; supported CPU and NVIDIA environments | Fixed weights/data, FP32 device policy, cold extraction, sufficient RAM. Adapter exactness and native numerical budgets from [numerical policy](numerical_policy.md); retain raw/normalized scores and failed observations. Historical HMAX mismatch stays separate under #21 | Existing manual harness; **scheduled/final-artifact qualification proposed**; scientific/domain maintainers; #19/#20 |
| Q2 — Trained robotics | Reference versus UMI OpenPI/LIBERO recordings, actions, tool cleanup and task success; separate trained-DROID evaluation on fixed episodes | Jenkins approved GPU/simulator runs; relevant changes and candidate | Pinned checkpoint, evaluator, seeds, task/episode manifest and action semantics. Compare reference and UMI on identical cases. LIBERO smoke success does not qualify trained DROID or full-suite reproduction | Recorded LIBERO smoke evidence; **automation and DROID qualification proposed**; robotics/scientific maintainers |
| Q3 — Cache and resource cost | Real CLIP/Pereira cold/warm scoring; changed input/weight invalidation; 3B-class hashing and trained inference; large stimulus-file hashing | Jenkins CPU/GPU performance profile after extraction changes and on candidate | Same model/data/settings across paired runs; isolated caches. Warm unchanged-weight validation: median ≤50 ms, zero weight-byte rereads. Report first hash, full fingerprint, file reads, runtime, memory and score separately; set other budgets before accepting them | Local CPU evidence; **broader qualification proposed**; performance/domain maintainers |
| D1 — Tutorials and figures | Execute offline notebooks/examples; redraw archived figures; run selected measured-data notebooks with supplied assets | GitHub Actions for 15 offline tutorials on Ubuntu/macOS; Jenkins for data/model reruns | No execution errors, correct links/assets, explicit measured versus illustrative outputs. Image rendering is not a new scientific measurement; verify numerical captions against exported metadata | Partner examples and 15 tutorials automated; 03/04/16/18 require separate asset-backed runs; docs/domain maintainers |
| D2 — Local dataset preparation | Every supported benchmark's stimulus set and reference data: authorized prepared download or conversion from author-native files; optional provider-specific retrieval ID | Offline fixture tests on relevant PRs; authorized full-data preparation before release | Follow the [local data acceptance checks](local_data.md#release-requirement-users-can-prepare-the-data-locally) from a fresh directory. Verify local loading, IDs/alignment, reference content and provenance without private caches or model loading. Document unavailable-source exceptions for scope review; native external evaluators may retain their own data format | [Source/fixture coverage tracked](dataset_preparation.md#coverage-tracked-by-release-check-d2); full-data qualification pending; benchmark/data maintainers |
| O1 — Operational/publication checks | One root-hygiene module and eight Algonauts submission-bundler skips; package assets, licenses, repository contents | Maintainer/operations validation; package or submission changes | Run external tools at pinned paths in their owning workspace, or replace obsolete tests with tests of a maintained implementation. Require schema/archive checks for advertised submission paths. Do not classify these as scientific passes | **Proposed routing**; publication/submission maintainers; #7/#13 |

A new model needs targeted integration with existing benchmarks. A new benchmark needs representative models. Neither requires every catalog plugin on every PR. Use the representative comparisons above for the proposed release scope; #6 must approve changes to that selection.

## Close the skip and selection gaps

The current 47 allowed skips in F1 consist of **31 remote OpenPI cases (F3), one local OpenPI module and one LIBERO transport module (I4), five LanA cases (I5), and nine operational entries (O1)**. A module-level skip can hide multiple tests; one reported skip is not one qualified behavior.

The 62 deselected cases in the October 20 run span `test_benchmark_parity_slow.py`, `test_regression_baselines.py`, `test_roar_benchmark.py`, `test_text_wrapper.py` and `test_vlm_vision_wrapper.py`. Route parity/baselines to Q1 and real-wrapper/ROAR cases to I2. ROAR registry assertions now run in F2; model/data cases remain in I2.

For each promoted selection, store exact test IDs, dependencies, environment, expected collection and an owner role. A supported case may be excluded from the fast profile only when its other required profile is identified. In that required profile, missing assets and skips fail; an explicit unsupported scope decision is different from a passing test.

## Shared execution rules

- **Revision set:** start from `install/peer-revisions.json`; for a peer PR, override that peer with its exact head SHA and record all four resulting revisions. The reusable offline workflow accepts this override and verifies checkouts before building. [Coordinated CI](coordinated_ci.md) describes caller pins and the required-check rollout; the existing offline/client callers and merge gates are enabled. New profile changes reach peer callers only after their immutable workflow pins are updated.
- **One test implementation:** reuse the repository runners and domain selectors. GitHub/Jenkins should supply environments and schedules, not separate assertions.
- **Report actual work:** save selected/deselected/skipped IDs and reasons, JUnit or equivalent outcomes, all revisions, wheel hashes, dependency lock, data/checkpoint identifiers, precision/device, cache settings, durations and resource use. A zero exit with absent evidence is a failure. An OOM/UNSTABLE Jenkins outcome cannot satisfy a required release check.
- **Bound expensive runs:** check dependencies, credentials/data availability, disk and expected memory before downloading/loading models. Set a hard runtime/compute limit and verify cleanup; an alarm that only sends a notification is insufficient.
- **Keep cache tests intentional:** most structural tests bypass result caching; integrity tests use temporary writable caches and must demonstrate both hits and invalidation. Scientific parity uses cold execution; performance reports cold and warm separately.
- **Final artifact:** rerun required profiles on the actual coordinated candidate wheels and environment, not merely editable source. Historical results do not qualify changed code, dependencies or devices.

## Implementation order

1. F1/F2 installed-package checks, F3 client checks and I4 local instrumentation are required in all four v2 repositories. Keep caller pins and required status names aligned when changing the workflow.
2. Jenkins coordination and maintainer-only automation controls are complete under #16. Preserve their deployed behavior when adding integration selections.
3. Run the [eight targeted I1/I2 profiles](integration_tests.md) with pinned assets before scheduling repeat jobs. The installed language scoring/CLI profile is executable; fresh Conda environment creation remains unqualified. I4 is automated; I3/I5 remain separate integration work.
4. After #6 scope approval, bind each required capability/environment to these rows and run Q1–Q3 on the final candidate (#19/#20).

The workflow implements F1–F3, I4 and the offline part of D1; other rows retain their stated qualification requirements. This document launches no compute and authorizes no publication. Release execution remains manual and owned by Kartik.

## New-user acceptance — separate pre-release task

Track the final release-package trial under [#40](https://github.com/brain-score/unified-model-interface/issues/40), with evidence reused by #19. The earlier getting-started guide review is #2. Schedule the final trial after package installation and the guide are stable; it does not block implementation of F1.

- Start with a new environment and empty application caches, without editable installs or private cached data.
- Follow the guide to install the candidate, obtain or prepare an authorized dataset, and complete a first benchmark run.
- Build a small tool using the documented extension points.
- Record the candidate revisions, steps, time, questions and workarounds. An unfamiliar developer must complete the agreed workflow without undocumented assistance; blocking findings must be fixed and retried before release.

Existing guide feedback remains useful evidence, but does not replace a trial of the final candidate. New-user acceptance, automated installed-package checks and scientific qualification are separate results.
