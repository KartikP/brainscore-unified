# UMI production release policy and candidate qualification

Status: **candidate implementation; general availability is not approved**.
This is the proposed support policy for maintainer review. It does not assign
people, publish artifacts, establish an SLA, or deploy scoring services.

Scientific parity uses the [versioned numerical policy](numerical_policy.md).
The opt-in CPU and NVIDIA L4 budgets cover fixed GPT-2/Pereira cases. New inputs, models,
interventions and robotics actions require their own numerical validation.
Higher-precision diagnostic agreement does not replace FP32 qualification.

## Coordinated package identities

| Package | Candidate version |
| --- | --- |
| brainscore-core | 2.4.0rc1 |
| brainscore-vision | 2.4.0rc1 |
| brainscore-language | 2.3.0rc1 |
| brainscore | 0.3.0rc1 |

These unpublished candidate versions distinguish UMI from existing release
identities. Unified pins all three peers; vision/language pin the matching core.
Select the final version policy before publication. Build all four artifacts
together and retain hashes, dependency resolution and platform reports.

## Install the locally built candidate

On the qualified macOS/Python 3.11 CPU profile, start with a fresh virtual
environment and the four candidate wheels supplied in the implementation output:

```sh
python3.11 -m venv umi-candidate
source umi-candidate/bin/activate
python -m pip install -c install/candidate-macos-py311.txt /path/to/wheels/*.whl
python -m pip check
python -m brainscore.doctor
python examples/partner_integration.py --out /tmp/new-umi-demo
```

The constraint file fixes the runtime dependency closure used for the local cold
installation. It is a macOS profile, not a universal Linux/CUDA lock. Candidate
wheels have not been uploaded to a package index. Keep the wheels and their hashes
with the constraint file. Do not run an unpinned pip upgrade in a scored experiment.

## Compatibility boundary

Preserve Subject/process, task and recording setup, legacy look_at/digest_text,
legacy capability constructor forms (with existing deprecation warnings), and
supported legacy benchmark protocols. Domain loaders continue returning adapters;
concrete legacy object identity, isinstance checks against old classes, private
attributes and arbitrary pickles are outside this boundary. Python is currently
3.11 only. Do not claim that every plugin's dependencies fit one environment.

The new public extension, observation and record APIs are candidate interfaces
until their gates pass. At general availability, incompatible changes require a
major interface/schema version, migration instructions and release notes. Proposed
deprecation window: at least two minor releases and six months, whichever is later.
Existing deprecated robotics dataclasses remain importable in this candidate.

Intentional corrections: EnvironmentSession.next_input raises if an action is
still owed, instead of returning the end-of-input sentinel; API action history
resets at episode start; missing plugin registrations have a distinct
PluginNotFoundError (also an AssertionError and KeyError for legacy callers),
while construction errors propagate. Legacy behavioral label fallback remains;
use the explicit response-trace path for invalid-response measurements.

New capability/session APIs reject conflicting registrations and unsupported
combinations. Reset attempts component/provider cleanup and reports failures; a
failed reset means the instance is not clean and should not be reused.

## Support levels to qualify

| Surface | Intended support | Evidence required before general availability |
| --- | --- | --- |
| Python 3.11, Linux CPU and macOS CPU | Maintained library profiles | Cold install, pip check, doctor, all selected suites and examples from wheels |
| Linux NVIDIA/CUDA | Maintained scientific profile after qualification | Pinned driver/runtime/checkpoints; complete real-data parity |
| macOS MPS | Experimental pending numerical qualification | Backend-specific tests and parity; CPU pass does not qualify MPS |
| External tools/channels | Maintained extension interfaces | Independent package and unfamiliar-author trial |
| DROID recorded trajectories | Reference robotics integration | Actual staged dataset/checkpoint validation and action semantics review |
| Physical robot control | Harness-specific; currently unqualified | Hardware owner, safety/control validation and closed-loop evaluation |
| Additional community plugins | Maintainer/community declared | Plugin-specific dependency and protocol evidence |

## Required release evidence

1. **B1 compatibility:** all selected source suites; packaged historical baselines;
   cold-cache legacy/adapter/native comparisons for all eight release cases;
   cache-enabled reuse/invalidation tests; current-upstream controls for declared
   model profiles, with historical published-score discrepancies tracked separately.
2. **B2 extension:** external domain/channel/capability round trips, reset/reuse,
   declaration and session-combination checks. No contract edits in the example.
3. **B3 tools:** nested observers, exact method restoration, legacy methods,
   exceptions, scoped intervention and episode state reset.
4. **B4 records:** supported arrays/assemblies/events/files, integrity failures,
   invalid responses and offline measurement replay. Versioned schema policy.
5. **B5 embodiment:** synthetic fixture plus actual DROID episode/policy; a
   controlled feedback environment; declared action semantics and time source.
6. **B6 validation:** actionable loader errors and coherent source/wheel CI.
7. **B7 authoring:** runnable examples and a recorded unfamiliar-author trial.
8. **B8 artifacts:** declared build backend, installed-wheel asset checks,
   dependency resolution, supported-platform jobs, license decision and publication.
9. **B9 operations:** named owners, issue route, response policy, recovery rehearsal;
   service canary/rollback if existing production scoring services adopt the set.

The [HMAX current-master control](qualification/2026-09-16-hmax-master.md) passes
for the recorded macOS CPU V4/IT cases: clean upstream and the UMI candidate
produce identical activation values and scores. The older published-score gap
also occurs upstream. Missing historical logs do not block this compatibility
result; reproducing those older scores remains a separate claim requiring its
own evidence. This does not qualify other HMAX platforms or protocols.

Run source qualification with:

```sh
python -m brainscore.validation.workspace --root /path/to/sibling/repos --out qualification.json
```

It records base commits and a digest of local changes. Separate pytest processes
avoid cross-repository tests-package collisions. Source passes do not certify an
installation. `release-candidate.yml` builds wheels, installs and exercises them
outside the source tree, checks dependencies, runs examples, and then runs all four
source suites on Linux/macOS. Dispatch it with immutable peer commit SHAs after
the coordinated changes are reviewed and committed. The existing offline workflow
is development feedback against peer branches, not release certification.

Run scientific parity against explicitly staged checkpoints and datasets:

```sh
python -m brainscore.validation.run_parity --resnet18 /data/resnet18.pt --gpt2 /data/gpt2 --out parity.json
```

The default covers four vision, two Pereira linear and two Pereira ridge cases.
Unknown selections fail. Selecting linear Pereira requires both cases and enforces
the paired drift guard. A subset result reports `release_complete=false`. Ridge
uses strict defaults until separately reviewed calibration exists; do not loosen
tolerances to obtain a pass. Historical ridge legacy/adapter matches are preserved
in `tests/fixtures/ridge-history.json`; they do not qualify the current native route.
Historical baseline manifests now ship with the tests and cannot silently skip
when missing. Their older 0.01 score bound is historical, not the three-route
release parity policy.

## Ownership and issue handling

Before publication, assign actual people for contract/API, vision compatibility,
language compatibility, tools/records, robotics integration, packaging/release,
and production scoring operations. Until staffed, these roles are **unassigned**.
Use the repository's issue tracker for non-sensitive reproducible reports; the
maintainers must approve and publish a private route for sensitive incidents.
Proposed operating target: triage installation/authoring issues in three business
days; acknowledge score-changing defects within one business day. These targets
become commitments only after owners accept them.

Collect exact versions, platform/backend, model and data revisions, benchmark/
protocol, cache state, minimal reproduction and original traceback. Do not request
API credentials or participant data in public reports. Score-changing defects stop
release; identify affected artifact/protocol ranges, document affected results and
publish corrected guidance. An unreviewed fallback must not silently repair scores.

## Upgrade, recovery and publication

1. Save the last known-good wheel set, full environment constraints, model/data
   manifests, package hashes and protocol documentation in the release record.
2. Install candidates in a separate environment. Run pip check, doctor, a legacy
   example, the robotics/tool demo and the required scientific reports.
3. Compare supported behavior with the previous set. Use distinct cache roots for
   changed model/preprocessing/protocol identities and test intended cache reuse.
4. Rehearse recovery by recreating the retained environment, installing retained
   wheels, reading retained run records and recomputing the reference measurement.
   Do not delete the current environment to test rollback.
5. Publish only after all gates have evidence and owners. Resolve the unified
   package license with its maintainers; do not infer a legal license from peers.
6. If scoring services change, deploy a canary and compare metrics/job failures.
   Roll back the service dependency set on regression; library tests alone do not
   qualify a production service deployment.

No publication, commits, pushes, remote CI dispatch or service changes are part of
this local implementation. These require subsequent explicit authorization and,
for remote CI, committed source identities.

## Deferred scope

Broad model catalogs, replay UI, a marketplace, general asynchronous scheduling,
automatic circuit discovery and hard real-time robotics remain optional follow-on
work. The release criterion is a supported path for partners to build those tools.
