# Targeted integration tests

Integration tests check real model dependencies and datasets that do not belong
in the short offline suite. Reuse the existing pytest tests and domain plugin
selectors. Jenkins supplies the environment, assets and time limit.

**Status:** proposed selections for release-matrix I1/I2. The
[selection file](../install/integration-tests.json) is review material; no job
reads it yet. Collecting tests verifies their names, not their scientific results.
This change does not schedule workers or download weights or datasets.
The eight profiles collected **52 test cases** locally on October 15, 2026.
Their full IDs and source revisions are recorded in the selection file. No test
fixtures or models executed during this check; the existing Python environment
was reused with temporary caches and network-dependent loading disabled.

## Select tests from the change

| Change | Selection |
| --- | --- |
| Model plugin | That plugin's own tests and generic model checks, plus one author-declared compatible benchmark |
| Benchmark plugin | That benchmark's own tests, plus a model with a known reference result and a cheap baseline where meaningful |
| Data plugin | Its preparation/loading tests and a benchmark that consumes the changed data |
| Metric plugin | Its metric tests and an affected representative benchmark/model pair |
| Shared domain code | The affected domain profiles below and the coordinated offline suite |
| Core interface or shared extraction code | Both domains' profiles and affected native wrapper profiles |
| Documentation only | Offline examples/notebooks; asset-backed reruns only when the changed example needs them |

Use the union when a change touches several areas. A new plugin needs an explicit
test pair and prerequisites in its review; an unrelated default pair cannot
qualify it. New model sizes and devices need a separate resource assessment.

## First representative selections

Exact pytest arguments, asset names, owner roles and proposed limits are in the
selection file. Roles describe responsibility, not new assignments to people.
The initial targets are Linux CPU, except the Qwen profile, which proposes a
CUDA worker after explicit device assertions are added.

| Profile | What it checks | Assets |
| --- | --- | --- |
| `vision-public-score` | Existing Vision scoring with a cheap pixels baseline | MajajHong2015 public IT data and images |
| `vision-layer-mapping` | AlexNet layer selection, scores and saved mapping files | AlexNet weights; MajajHong2015 public V4/IT data |
| `language-public-score` | Existing Language scoring and a fixed reference result | DistilGPT-2; Futrell2018 data |
| `language-readout` | Reading-time outputs and token handling | DistilGPT-2; test strings supplied by the tests |
| `language-installation` | Existing dependency-installation modes and CLI scoring | Conda; random embeddings; Pereira2018 data |
| `native-text` | CLIP/GPT-2 feature shapes, layers, batching and token alignment | CLIP and GPT-2; test strings |
| `native-vlm` | Qwen image features, patch grouping and repeated outputs | Qwen2.5-VL-3B; generated test images |
| `behavioral-roar` | Fixed train/test split, CLIP readout and score structure | CLIP; authorized Yeatman2021 stimulus/data build |

These small selections complement the eight-case scientific parity matrix.
They do not qualify every plugin, trained DROID, audio/video scoring, or every
supported device. Those retain their acceptance criteria in the
[release matrix](release_test_matrix.md).

## Before enabling a profile

1. Resolve all four source revisions and build the coordinated wheels. Pin the
   model/processor snapshot and dataset build or object versions in the run
   record. A model name alone is not a checkpoint pin.
2. Stage authorized assets before testing. Check their presence, disk space and
   estimated memory before model loading. Give the worker an isolated writable
   cache; do not reuse a developer's private result cache.
3. Collect the exact pytest selection in the proposed environment. Freeze its
   full test IDs and count for that revision; reject empty or changed selections
   until reviewed. Missing imports or collection skips are failures.
4. Run the collected tests with a hard timeout, zero skip exemptions and retained
   JSON/JUnit reports. Record device, precision, dependency versions, duration,
   memory and failures. Missing reports, OOM and timeout are failures.
5. Run against installed packages outside source checkouts, checking import
   locations and wheel hashes. Source-only CLI tests need adjustment before they
   can establish installed-package acceptance.

The proposed limits in the selection file are stopping limits, not measured
budgets or permission to start paid workers. Confirm resource sizing with a
bounded pilot before scheduling. Keep each profile in its own environment when
its dependencies or hardware differ.

## Reuse the current reporting tools

For execution, use `brainscore.validation.required_checks` with a distinct
profile such as `integration-native-text`; these names have no skip exemptions.
Retain the same `qualification/revisions.json` and installed-package identity
evidence used by coordinated CI. Collection-only runs do not use this plugin:
its success rule requires tests to have executed.

Do not pass the offline suite's `-m "unit or integration"` filter: it would
exclude the real-model tests marked `slow`. Preserve only the explicit filters
listed in the selection file. Start with result caching disabled; repeated-output
checks alone do not prove a cache hit. Cache-hit/miss instrumentation belongs to Q3.

## Gaps to resolve during activation

- **Plugin result accounting:** Core's `run_args()` accepts pytest exit code 5
  and overwrites its result dictionary when several test files are supplied.
  Qualify each selected plugin separately and retain every result; harden this
  accounting before treating the combined runner's exit status as acceptance.
- **Installed CLI checks:** Vision/Language CLI tests execute a source directory.
  Adapt those checks to installed entry points before claiming installed CLI
  coverage. The initial Vision profile selects the Python scoring test only.
  Language's dependency-installation modes must also preserve and verify the
  coordinated peer versions rather than silently installing different packages.
- **Subject dispatch:** the tests named `TestTextWrapperIntegration` and
  `TestVLMVisionWrapperIntegration` call wrappers directly. They do not prove
  real-model dispatch through `BrainScoreModel.process()`. Retain the wrapper
  checks and add genuine dispatch checks before closing that acceptance item.
- **GPU selection:** the current Qwen fixture does not explicitly require CUDA.
  A GPU worker alone is not GPU evidence. Record/assert model placement and
  precision before counting a GPU run.
- **Other domains:** add explicit audio/video selections and frozen reference
  outputs; keep isolated submission/database tests under #22 and full scientific
  execution under #19/#20.

Enable the profiles only after these requirements and #6's support scope are
reviewed. Keep Jenkins deployment changes separate from this selection review.
