# The subject contract

A subject participates in an experiment through `interact(session)`. The same method works for fixed stimuli and an environment that reacts to the subject.

| Part | Responsibility |
| --- | --- |
| Subject | Declare inputs and outputs, respond, and clear its state on reset. |
| Session | Supply inputs, accept responses, and carry task and recording settings. |
| Benchmark | Define the experiment, measured data, metric and ceiling. |
| Extractor | Prepare model inputs, run the model and package recorded activity. |

## Declarations

A subject declares `identifier`, `in_channels`, `out_channels`, and `required_channels`. Its class does not determine what it can do. `process()` is a convenience on `BrainScoreModel`, not a requirement on `Subject`.

A compatible pair satisfies all three conditions:

- The subject can provide every requested output.
- At least one benchmark input overlaps the subject's inputs.
- The benchmark supplies every input the subject requires.

`check_compatibility` returns a `CompatibilityMatch`. `consumed_inputs` and `dropped_inputs` explain a `full` or `partial` match. Scoring records those fields and `modalities_used`. An incompatible pair raises `CompatibilityError` before evaluation. Use `score_or_na` when a sweep should record N/A instead of raising. N/A is never zero.

For example, a vision subject may participate in an audiovisual experiment. The experiment still declares both streams. The subject ignores audio, and the score identifies the partial match.

## Events

`StreamEvent(channel, payload, t_ms, meta)` carries one input or response. Use `t_ms=None` when there is no physical clock. A trial index is metadata, not milliseconds.

A channel family has one catalog entry. The entry names its payload type, direction, validation and owner. Buffered sessions validate all available inputs before execution. Live sessions validate each future input at delivery; they cannot inspect observations that do not exist yet. Replacing a catalog entry requires `replace=True`.

Keep model tensor preparation inside the extractor or subject. For example, a `vision` event carries original image pixels; a ResNet subject performs its own resize and normalization.

## Recording and changes

The session requests outputs, such as `neural:IT`, and supplies its `task_context`. A model registration maps that region to a layer or unit selection. Benchmarks do not choose model layers at scoring time.

Interventions return a receipt and cleanup. Undoing one intervention leaves others in place. A run has one reset owner; `SessionProtocol` resets before and after each trial.

## Legacy benchmarks

`LegacyBenchmarkFacade` translates legacy methods into session settings. It supports `look_at`, `digest_text`, `start_task`, `start_behavioral_task`, `start_recording`, `start_neural_recording`, and `visual_degrees`. It carries trial counts, time bins and recording type in the session. A subject remains responsible for honoring the requested measurement.

Legacy model adapters and the benchmark facade are permanent. Legacy plugins and scientific scoring formulas are not rewritten to adopt the session contract.

## Retained integration paths

New extractors use a modality-to-callable mapping and one call shape:
`extractor(stimuli, layers=...)`. Preparation belongs inside that callable.
The deprecated `preprocessors` / `activations_model` constructor forms still
work through a separate adapter. Only that adapter inspects old callback
signatures; normal extraction and session routing do not.

Domain loaders identify legacy benchmark calls and scoring supplies the facade
automatically. Benchmarks that declare `required_input_channels` are treated as
session-native unless they explicitly set `subject_api='legacy'`. A benchmark
using the retained typed helpers can declare `subject_api='typed'`.
Register a non-default visual field with `register_visual_field(subject, provider)`
from `brainscore_core.legacy_benchmark`; the provider returns degrees.
`BrainScoreModel` and the vision adapter register their configured field.

Existing environment recordings may contain `EnvironmentStep` directly.
Replay validates those records through `EnvironmentReplaySession`, preserving
the original payload and time. New buffered inputs use
`StreamEvent(channel='observation', payload=step, t_ms=None)` when no physical
clock exists. Live environment sessions retain their typed observation boundary.

These compatibility paths preserve existing integrations. They are not new
requirements on `Subject`.
