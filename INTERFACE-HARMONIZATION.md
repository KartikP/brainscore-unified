# Unified Model Interface harmonization inventory

This inventory covers the public model, recording, delivery-session, adapter,
and activation-wrapper surfaces in `brainscore_core`, `brainscore_vision`,
`brainscore_language`, and `brainscore`. It deliberately separates fixes that
are provably number-preserving from recommendations that could alter an xarray
schema, dispatch choice, cache key, or benchmark result.

## Canonical surface today

- `Subject` is the public contract. `BrainScoreModel` is its concrete
  multimodal implementation; `UnifiedModel` remains a deprecated alias.
- `Subject.identifier` is a property. The vision and language loaders wrap
  legacy models in adapters, so models returned by any public `load_model`
  expose the property even though legacy language `ArtificialSubject` classes
  implement `identifier()` as a method.
- `BrainScoreModel.start_recording(target)` configures subsequent neural
  `process` calls. A target is a mapped region string, a list of mapped regions,
  `'all'`, or a raw layer string.
- `BrainScoreModel.process(input_event, multi_modality=False)` is the direct
  evaluation entry point. `interact(session)` is the delivery-oriented entry
  point and routes from the session's requested output channels.
- `VisionWrapper` is the public vision facade. It selects `PytorchWrapper`,
  `VideoWrapper`, or `VLMVisionWrapper`; `TextWrapper` and `AudioWrapper` remain
  modality-specific public wrappers.

## Findings and decisions

| Area | Current behavior | Decision |
| --- | --- | --- |
| Single versus list recording | A multi-region list adds per-neuroid `region`; a single region string preserves the historical schema without it. Both carry `layer` when the extractor supplies it. | Defer schema unification. Adding `region` to the default single-region path can change xarray indexes/alignment and cannot be proven score-neutral without forbidden benchmark scoring. |
| Multi-region delivery | Batch and streaming drivers now establish one list-valued recording, run once per delivery unit, and demultiplex from explicit `layer`/`region` provenance. | Applied as the confirmed bug fix. Missing provenance raises instead of assigning by position. |
| Convenience helper cardinality | `start_recording` accepts a list, but `stimulus_session`, `StreamingStimulusSetSession`, `WindowedStreamSession`, and `neural_response` publicly type and format `record` as one string. A list passed to these constructors becomes one malformed string-valued channel. | Defer. Supporting several records requires choosing whether `neural_response` returns one combined assembly, a channel mapping, or a new multi-channel result type. |
| Multimodal processing | Direct `process` selects one modality by default and fans out only with `multi_modality=True`; the buffered `interact` driver can infer fan-out from several input-channel events. | Keep the default. Automatic inference in direct `process` would change the return shape and default numerical path. Prefer a future explicit method or enum over another boolean, with deprecation. |
| Identifier form | The unified loaders and both legacy adapters expose `identifier` as a property. The method/property difference remains only when users bypass loaders and instantiate a legacy language `ArtificialSubject` directly. | No runtime change. Public annotations and prose now name `Subject`; the `UnifiedModel` alias remains imported for compatibility. |
| Streaming expression | `StimulusSetSession` is buffered; `StreamingStimulusSetSession` is lazy per stimulus; `WindowedStreamSession` is a bounded window feed; `RealTimeStreamSession` wraps a window feed with lag/drop/error policy. `session.streaming` is the routing protocol used by the latter three and by custom sessions. | Defer renames or a delivery enum. Removing the flag or renaming classes needs a compatibility/deprecation plan; documentation now distinguishes data ownership, delivery unit, and real-time policy. |
| Legacy recording adapters | `VisionModelAdapter` substitutes default time bins when `time_bins` is falsey and delegates list targets to a legacy single-target API. `LanguageModelAdapter` ignores `time_bins` and passes a list target through to `start_neural_recording`. | Defer. Correct translation depends on domain-specific time-bin and multi-region semantics and could change legacy benchmark outputs. |
| Legacy task adapter | `VisionModelAdapter.start_task` accepts both `TaskContext` and the legacy two-argument form although `Subject.start_task` exposes the former. | Keep inside the compatibility adapter; removing the overload would break legacy benchmarks. |
| `recording_type` | `BrainScoreModel` accepts but ignores it, while the language adapter uses it and vision ignores it. | Keep for compatibility and document the concrete behavior; a capability-specific recording-options object is a future API decision. |

## Wrapper inventory

All wrappers put `identifier` on a property and accept a model as the first
argument, but their second argument and output rank necessarily follow the
modality.

| Wrapper | Construction focus | Call/return contract | Inconsistency |
| --- | --- | --- | --- |
| `VisionWrapper` | `preprocessing`, or `processor` plus `kind`; arbitrary strategy kwargs | Delegates to the selected strategy | The facade is canonical, but strategy classes remain directly importable and `kind='auto'` is heuristic. |
| `PytorchWrapper` | image preprocessing, forward kwargs/input key, extractor passthrough args | Flexible extractor call; normally `(presentation, neuroid)` | Accepts open-ended `*args`/`**kwargs`, unlike the strict modality wrappers. |
| `TextWrapper` | tokenizer, token aggregation, max length, batch size | `__call__(stimuli, layers, stimuli_identifier=None)`; 2-D for pooled modes and 3-D for `per_token` | Extra call kwargs are silently ignored; direct list input bypasses the StimulusSet cache path. |
| `VLMVisionWrapper` | processor, patch grouping, patch aggregation | Same core call shape; remaining patch dimensions are flattened into neuroids | Aggregation validity is checked during extraction rather than construction; direct list input bypasses caching. |
| `VideoWrapper` | frame sampling, temporal hooks, optional context windows | Same core call shape; returns `(presentation, time_bin, neuroid)` | Many temporal constructor options are necessarily bespoke; direct list input bypasses caching. |
| `AudioWrapper` | processor, sampling rate, time aggregation, chunk size | Same core call shape; pooled or time-resolved output | Direct list input uses caching unlike the other new wrappers; `audio_input_key` is accepted and stored but not applied. |

Recommended wrapper direction: retain modality-specific configuration, but
standardize the public call signature, unknown-kwarg policy, constructor-time
validation, and direct-list caching behavior in a staged release. Doing this
now could change cache reuse or error timing, so none is included here.

## Documentation drift found

The following prose described behavior the code did not implement. The wording
has been corrected without changing execution:

- The generic streaming driver claimed every delivery was one stimulus and
  score-equivalent to batch delivery, although it also drives windowed and
  real-time sessions, including a policy that drops windows.
- `StreamingStimulusSetSession` claimed the whole stream never existed in
  memory, although it retains the caller's materialized stimulus table; only
  event creation is lazy.
- `TextWrapper.__call__` claimed every output was two-dimensional although
  `per_token` returns a `time_bin` dimension.
- `AudioWrapper` claimed every output was time-resolved although `mean_time` is
  the default pooled path.
- `AudioWrapper` claimed long clips were truncated with a warning and that
  chunking was deferred; the implementation chunks and recombines them.
- `AudioWrapper.audio_input_key` was described as controlling the model-forward
  kwarg, but the current implementation forwards the processor dictionary
  unchanged. It is now explicitly documented as a reserved no-op; implementing
  or deprecating it is deferred because it would change model invocation.
- `VideoWrapper` claimed a custom frame sampler was required although an
  OpenCV-based default exists, and its constructor documentation omitted the
  backbone, context-window, padding, and whole-clip guard options.
- `VLMVisionWrapper.identifier` was described as the cache key without noting
  that `backbone_id` overrides it.
- The unified package prose and annotations still named `UnifiedModel` as the
  primary contract even though it is a deprecated alias of `Subject`.

## Deferred recommendations

1. Add a `records=` multi-channel convenience API with an explicit return type;
   the present `record=` helpers are unambiguously single-channel.
2. Replace `multi_modality: bool` in a future major release with an explicit
   modality-selection operation; inferring it would change the default result.
3. Add single-region `region` provenance only after xarray-alignment and score
   contracts are validated remotely; metadata can affect numerical alignment.
4. Treat `streaming` as an internal delivery-mode protocol and expose one
   documented session factory; class/flag removal requires deprecation.
5. Standardize wrapper call kwargs, validation timing, and cache entry points;
   current differences affect compatibility or cache behavior.
6. Implement or deprecate `AudioWrapper.audio_input_key`; applying it today can
   change which model forward arguments are supplied.
7. Define multi-region and time-bin translation for legacy adapters before
   advertising list recording through them; pass-through is not a guarantee.
8. Keep legacy identifier methods only below adapter boundaries and discourage
   direct construction in new examples; loaders already provide one property.

No recommendation above is included as a runtime change in this harmonization
pass.
