# Historical support assessment — October 5, 2026

This is a historical planning record. Use the [feature reference](../supported_features.md) for current capabilities and the [release policy](../production_release.md) for release requirements. Statements about pending work below describe the assessment date.

Initial assessment: September 30, 2026. Updated October 5 after the coordinated v2 merges. Proposed scope for [issue #6](https://github.com/brain-score/unified-model-interface/issues/6).

**Draft for review under #6. These are proposed support targets, not an approved release policy.** This document sets the support targets and the evidence needed to claim them. It covers legacy Brain-Score use and the public interfaces for new domains, tools, and experiments. Kartik owns the manual release; this assessment authorizes no publication or service change. Support owners and response commitments remain unassigned.

## 1. Support boundary

- **Required:** part of the initial general-release promise. Missing evidence or a correctness defect blocks that promise; availability in source is insufficient.
- **Experimental:** usable for research, with explicit limits and no general compatibility or scientific-validity guarantee. Promotion requires its own model, data, platform, and protocol evidence.
- **Deferred:** outside the initial support promise. Extension points should let partners build these features without changing the model contract.

A supported result identifies the four package revisions, dependency environment, model/checkpoint, data, protocol, device, precision, and cache configuration. A passing unit test does not qualify every combination of those choices.

### Proposed required interfaces and representative use cases

| Surface | Initial support target | Current evidence and remaining boundary |
| --- | --- | --- |
| Model contract | Minimal `Subject`: identity, input/output channels, `interact(session)`, and reset; `UnifiedModel`/`BrainScoreModel` provide `process` and task/recording helpers | Source tests cover dispatch and lifecycle. Add uncovered session and failure-path assertions before release. |
| Legacy use | Vision `look_at`, language `digest_text`, legacy task/recording protocols, registry loading, existing capability constructor forms | Adapters and selected legacy helpers are tested. Existing deprecation warnings remain. Full plugin compatibility is not established. |
| Vision and language scoring | Fixed ResNet18/MajajHong and GPT-2/Pereira reference cases; documented layer/region mapping and numerical policy | Eight-case candidate evidence exists. Requalify the final coordinated artifacts, especially the repaired language helper on Linux/CUDA. |
| Additional benchmark domains | One identified native behavioral, temporal, multimodal, perturbation, and embodied example for the release checks | Structural tests and earlier demonstrations exist. Select exact model/data/protocol profiles and obtain current real-model evidence; broader catalogs remain experimental. |
| New input and output domains | External channel registration, payload validation, preprocessors, capability dispatch, and session declarations | External-extension tests and the independent example package demonstrate the path. Registration alone does not implement preprocessing or inference. |
| Experiment tools | Observation, model input/layer capture, scoped interventions, and restoration after success or failure | Fixture tests cover public calls and cleanup. Internal capture requires access to a supported model backend. |
| Records and replay | Versioned records of supported payloads, provenance, errors, stored outputs, and offline metric evaluation | Round trips and selected corruption checks pass. Close remaining schema/integrity and session-recording gaps. Replay evaluates saved measurements; it does not guarantee deterministic model re-execution. |
| Generated-response traces | Provider-exposed raw responses and metadata, explicit answer parsing, invalid-response status | Invalid parsing and session dispatch are tested. Add successful-parser and provider-failure assertions. Hidden reasoning is not available unless the provider exposes it. |
| Robotics integration | Recorded DROID observations/actions, explicit action semantics, policy reset/chunks, recording and measurement replay; controlled environment sessions | One real 119-step episode passed using an untrained network. A trained-policy direct-versus-UMI comparison remains required. Controlled feedback simulation is optional after release. |
| Authoring experience | A researcher can add an external tool/domain/experiment using public APIs and the guide | Maintainer-built examples work. Kartik reports Deirdre completed the initial guide trial (#2); its findings still need incorporation. The independent final-candidate trial (1.5.2) remains separate. |

The representative-domain requirement follows [issue #18](https://github.com/brain-score/unified-model-interface/issues/18). Calling a broad catalog experimental does not remove these required examples.

### Compatibility limits

Domain loaders may return adapters. Concrete legacy class identity, private attributes, arbitrary pickles, and every historical plugin dependency environment are outside the contract. `UnifiedModel` extends `Subject` with `process`, task/recording setup, and layer/modality declarations. Session-native implementations can inherit `Subject` directly. `BrainScoreModel` builds on `UnifiedModel`. A plugin without compatibility evidence is unqualified, not necessarily broken.

Intentional corrections remain visible: `EnvironmentSession.next_input` raises while an action is owed; API action history resets at episode start; missing registrations raise `PluginNotFoundError`, while construction errors propagate. Failed reset means the instance must not be reused as clean.

Keep FP32 as the target. Legacy adapter activations must match exactly; scalar scores retain the four-FP32-ULP reduction allowance. Native tolerances apply only to their named profiles, not arbitrary prompts, interventions, or robot actions. The [numerical policy](../numerical_policy.md) defines those budgets. Incompatible public API/schema changes require versioning and migration guidance. The proposed deprecation window remains two minor releases and six months, whichever is later.

## 2. Environments and versions

### Coordinated source snapshot

These are the coordinated code revisions tested after the October 5 merges into `unified-model-interface-v2`. The unified peer manifest names the same three peer revisions. Later documentation-only commits do not change this tested snapshot.

| Repository / distribution | Candidate version | Source commit |
| --- | --- | --- |
| core / `brainscore-core` | 2.4.0rc1 | `434eb311fca090567653716ce1cb337c35646b1b` |
| vision / `brainscore-vision` | 2.4.0rc1 | `591628f09dd7b80808dc4197a5b13eea3f15f6af` |
| language / `brainscore-language` | 2.3.0rc1 | `e635fc8d8ae289a20d4be03cb57b17547286b135` |
| unified / `brainscore` | 0.3.0rc1 | `0a38dbb9a9d697777ada47373cad229c953c0f3f` |

Install the four packages together. Candidate peer versions are unpublished; removing peer pins is not a supported installation workaround. Final package identity and license decisions belong to [issue #8](https://github.com/brain-score/unified-model-interface/issues/8).

### Python and principal libraries

All four packages require **Python >=3.11,<3.12**. Recorded runs used 3.11.15. Package dependency ranges describe resolver permissions, not qualification of every version within them. The shared CPU constraints specify this narrower target:

| Dependency | Coordinated metadata boundary | Shared CPU pin |
| --- | --- | --- |
| NumPy | `<2` | 1.26.4 |
| pandas | `<3` | 2.3.3 |
| xarray | `==2022.3.0` | 2022.3.0 |
| SciPy | No bounded range | 1.17.1 |
| scikit-learn | `>=1.7,<1.8` | 1.7.2 |
| PyTorch | `>=2.6` through vision | 2.13.0 |
| torchvision | `>=0.21` through vision | 0.28.0 |
| Transformers | `>=4.57,<6` | 4.57.6 |
| datasets | No bounded range | 5.0.0 |
| importlib-metadata | `<5` | 4.13.0 |

Use [shared constraints](../../install/v2-constraints.txt) for source integration. The [macOS candidate constraints](../../install/candidate-macos-py311.txt) record a larger historical dependency closure; they are not a Linux/CUDA lock. Optional `robotics-data` declares tensorflow-datasets >=4.9,<5; `api` declares OpenAI >=1,<3 and Anthropic >=0.40,<1. Those ranges do not certify live providers or all versions. Model-specific dependencies need separate profiles. Keep trained robotics policy servers in their own supported environment when their dependencies conflict.

### Operating systems and compute

These profiles await approval under #6. The October 1 meeting did not approve specific OS, Python, or CUDA versions. It kept closed-loop games experimental; the existing release plan retains trained DROID validation as required.

| Environment | Proposed scope | Evidence and release requirement |
| --- | --- | --- |
| macOS ARM64, CPU, Python 3.11 | Required library and scientific profile | Post-merge integration CI passed on macOS 14. Earlier repaired-language evidence is on macOS 26.5.1; September 29 source coverage ran on 27.0.1. Choose one exact OS/build profile and qualify final wheels, examples, and real-data comparisons there. Neither run qualifies every macOS release. |
| Ubuntu 24.04 x86_64, CPU, Python 3.11 | Required library profile, matching the pinned offline CI runner | Current v2 CI passed coordinated installation, installed examples, and source tests using the pinned CPU wheels. Final-artifact scientific qualification remains required. |
| Ubuntu 22.04 x86_64, NVIDIA L4, CUDA 13.0, FP32 | Required first GPU scientific profile | Earlier candidate passed eight fixed cases using Torch 2.13.0+cu130, driver 580.126.09, four CPU threads, TF32 disabled. Requalify the repaired code and final artifacts before claiming support. |
| Ubuntu 22.04 AMD CPU and Amazon Linux 2023 Intel CPU | Historical evidence; additional profiles remain experimental | Earlier candidate passed four language cases on Ubuntu with one thread and four vision cases on Amazon Linux with two. Both used CUDA-enabled Torch on CPU, which does not qualify a CPU-only wheel or Ubuntu 24.04. |
| macOS MPS; other NVIDIA GPUs/CUDA stacks; AMD ROCm; Linux ARM64 | Experimental | No general backend guarantee. Each needs a declared environment, numerical controls, and relevant compatibility cases. |
| Windows, Intel macOS, Python outside 3.11 | Outside initial support | No qualified installation/runtime profile; other Python versions are rejected by package metadata. |

The driver and Ubuntu release above come from the September 15 Linux qualification record; the committed [candidate evidence](../qualification/2026-09-16-candidate.md) records the kernel, package stack, CUDA runtime, and cases. These measurements precede the language cache repair. Its [macOS comparison](../qualification/2026-09-16-upstream-language.md) does not refresh Linux/GPU qualification. The current manual candidate workflow also uses floating OS labels and unconstrained dependency resolution; align it with these targets before using its output as support evidence. This document does not dispatch it.

## 3. How partners extend UMI

The supported authoring path is an **external Python package** explicitly imported before model construction. Repository plugin templates remain useful for catalog contributions, but editing a Brain-Score repository is not required for a tool.

| Contribution | Where and how to implement it | Required checks |
| --- | --- | --- |
| New sensory input | Register a `CatalogEntry` through `brainscore_core.extensions.register_channel`, optionally map stimulus columns, and provide a preprocessor | Shape, units, materialization, malformed payloads, channel/column conflicts. Register a family, not an addressed instance. |
| New output or operation | Register its output channel; subclass `Capability`, implement `handles`/`process`, and call `register_capability` | Enable explicitly through model configuration; return validated `StreamEvent` payloads; test setup, per-model state, reset/reuse, and unsupported combinations. |
| New session | Implement `supports_session(model, channels)` and `interact(model, session)` on the capability | Accept the complete output combination; reject ambiguous handlers; preserve ordering, termination, and error cleanup. |
| Observation or interpretability tool | Use `observe`, `ActivationWindow`, `PerceptWindow`, or `intervene` from the documented tool interfaces | Nested observers, original method restoration, exceptions, scoped handle removal, and matched baseline/intervention conditions. |
| Saved measurement or response trace | Use `RunRecorder`/`RunRecord` or `build_trace_subject` with a provider and parser | Provenance, valid/invalid responses, codec limits, corruption detection, and replay without model calls. |
| New experiment, benchmark, or metric | Pass a model object to a benchmark/score call; optionally register factories in the public benchmark, metric, data, or model registries | Data ordering, splits, ceilings, null controls, metric semantics, and the model's declared capabilities. |
| Robotics policy | Implement the policy's `infer` interface for `DroidPolicy`; provide `ActionSpec`, observation mapping, action source, and reset behavior | Camera/state isolation from targets, action shape/units/frame/bounds, chunk horizon, timestamp provenance, and direct-reference agreement. |

No new `Subject` method, core event-union edit, or `BrainScoreModel` dispatch edit is needed for these paths. Registration does not create execution behavior, and arbitrary custom Python objects are not automatically supported by record codecs. Use supported arrays/events/mappings or implement and qualify a separate storage adapter. Register once during process initialization; use a separate model instance for each concurrent experiment.

Start with the [tool guide](../tool_authoring.md), [independent package](../../examples/partner_tool/README.md), and [DROID guide](../droid_integration.md). The getting-started trial must record every missing instruction or workaround before the guide is repaired.

## 4. Remaining work

Use the [reconciled production backlog in #4](https://github.com/brain-score/unified-model-interface/issues/4#issuecomment-5996495739) for current status, owners, completion criteria, and the mapping of test gaps to existing tasks. Coverage assessment #5 is complete; release-check planning remains under #18, and compatibility/regression execution is shared by #19 and #20. The register replaces the earlier duplicate gap list in this document.

The [September 29 coverage assessment](https://github.com/brain-score/unified-model-interface/issues/5#issuecomment-5912297590), on the earlier revisions listed in that report, recorded 1,945 passed, 34 failed, two collection errors, 14 skipped, and 285 deselected cases. Two failures concern assertions/path portability; 32 require unavailable assets or live-service access. Blocked cases remain failures. Whole-repository line/branch coverage was core 81.4%/72.0%, vision 2.7%/1.3%, language 20.7%/13.1%, and unified
57.7%/50.1%. Large unexecuted plugin catalogs explain the low domain totals; those numbers neither erase legacy gaps nor certify correctness.

The old published HMAX score mismatch is a separate [historical reproducibility task](https://github.com/brain-score/unified-model-interface/issues/21). Current-upstream HMAX V4/IT matched the candidate exactly on the recorded macOS CPU profile. Missing old logs do not block that scoped compatibility result.

## 5. Experimental and deferred scope

Keep closed-loop games, unqualified audio/video/multimodal model combinations, new GPU families, live paid providers, mixed precision, and broad plugin catalogs experimental. A demonstrated wrapper or registration is not a validated scientific result.

The DROID sample establishes data transport and tools, not trained-policy quality. Recorded prediction agreement is not closed-loop task success. Physical robot control, hardware safety, hard real-time deadlines, and calibrated control loops remain harness-specific and outside this initial guarantee. Robotics is broader than the GridGame demonstration.

Defer a replay UI, plugin marketplace, automatic circuit discovery, general asynchronous/distributed scheduling, and exhaustive model-catalog qualification. The required deliverable is a documented, tested way for partners to build tools through stable interfaces; those optional products need not be built first.

## 6. Status of this scope assessment

The inventory is published for review under #6. Kartik reports Deirdre completed the initial guide trial; #2 currently has no findings attached. Incorporate those findings and approve the feature/environment matrix before finalizing the support promise. This does not complete scientific qualification under #19/#20 or the final independent user trial.

### Post-merge integration evidence

[CPU CI run 37329425500](https://github.com/KartikP/brainscore-unified/actions/runs/37329425500) tested the four revisions above on Ubuntu 24.04 and macOS 14. Each platform passed 1,133 source tests and 28 installed-package tests, with 15 skipped and 64 deselected. Builds, dependency checks, and installed examples passed. This is CPU integration evidence, not full scientific or GPU qualification.

### Checks performed for this assessment

On September 30, the Python 3.11 macOS CPU source environment passed:

- **14 core tests:** `test_external_extensions.py` and `test_capabilities.py`.
- **33 unified tests:** `test_production_tools.py`, `test_droid_policy.py`, and `test_robotics_harness.py`.
- **External-package example:** new input/output channels and a capability, observer attachment, saved-output replay, and reset/reuse; run outside the repository using explicit source imports.

The pytest runs forced CPU execution, blocked external sockets, and disabled result-cache reuse. No weights or data were downloaded. The example used a synthetic signal and existing dependencies. These checks confirm the documented extension route; they are not a fresh installed-wheel or independent-author trial.

The [production policy](../production_release.md) covers release evidence, ownership, and recovery. This document is the feature/environment boundary used by that policy; existing qualification reports remain scoped historical evidence.
