# Feature status

Features are grouped by how much you need to know to use them, and by how settled they are.

## Core: needed by everyone

| Feature | Where |
| --- | --- |
| `Subject`, `participate`, `reset` | [Core concepts 1](../02_core_concepts.md#1-subject), [Lesson 1](../../notebooks/lessons/01_your_first_subject.ipynb) |
| Channels, `StreamEvent`, the channel catalog | [Core concepts 2-3](../02_core_concepts.md#2-channel) |
| `BufferedSession`, `EnvironmentSession` | [Core concepts 4](../02_core_concepts.md#4-session) |
| `StimulusSet`, `MultimodalStimulusSet` (stimuli with several input types) | [Write a benchmark](../guides/write_a_benchmark.md) |
| `Benchmark`, `Score`, compatibility and N/A | [Lesson 2](../../notebooks/lessons/02_your_first_benchmark.ipynb) |
| `BrainScoreModel` with `extractor` and `region_layer_map` | [Wrap your model](../guides/wrap_your_model.md) |
| `VisionWrapper`, `TextWrapper`, `AudioWrapper` | [Class map](../03_class_map.md#extractor-one-per-input-type) |
| Adapters for classic vision and language models; classic benchmarks on any subject | [Existing models](../guides/existing_models.md) |
| `load_model`, `load_benchmark`, `score`, the unified registry | [Score a model](../guides/score_a_model.md) |
| Image-processor pinning (`pin_image_processor`) so preprocessing does not drift between library versions | `brainscore_core.hf_compat` |

## Advanced: for specific study designs

| Feature | Where |
| --- | --- |
| `LayerSelector`, `CompositeSelector`, several regions per run | [Lesson 5](../../notebooks/lessons/05_layers_and_regions.ipynb), [Lesson 8](../../notebooks/lessons/08_choose_units.ipynb) |
| Native input forms: `vision:still`, `vision:clip`, `VLMVideoWrapper`, `attrs['input_adaptation']` | [Core concepts 6](../02_core_concepts.md#6-compatibility) |
| Behavior: `TaskContext` responses (`Choice`, `Scale`, `Text`), reported and implied answers, `reply`, `likelihood`, readout, `prefer_path` | [Behavior](../guides/behavior.md) |
| Perturbation: `Selection`, `StateChange`, `intervention` | [Perturb and select units](../guides/perturb_and_select_units.md) |
| Unit selection: `IndexSelection`, `RandomSelection`, `FunctionalSelection` | same |
| Closed loop: `policy`, `EnvironmentStep`, specs | [Closed loop](../guides/closed_loop.md) |
| `Experiment`, `TaskProtocol`, `SessionProtocol`, `CallableProtocol`, conditions, trials | [Experiments](../guides/experiments.md) |
| Recorders: `RecordNeural`, `RecordBehavior` (one presentation for both); `Intervene` | same |
| Tools: `RecordInputsOutputs`, `RecordActivity`, `Ablate`, `ScaleActivity`, `PatchActivity` | same |
| `TorchInstrumentation`, `RunRecord`, `replay_sessions`, `compare_outputs` | same |
| Timed delivery: `WindowedStreamSession`, `RealTimeStreamSession`, `StreamingStimulusSetSession` | [Class map](../03_class_map.md#sessions-protocols-and-experiments) |
| Temporal helpers: `temporal_bin`, HRF convolution, contiguous-block cross-validation | `brainscore_core.temporal`, [Experiment 10](../../notebooks/experiments/10_story_listening_fmri.ipynb) |
| Score provenance and status (`score.attrs`, N/A scores) | [Score a model](../guides/score_a_model.md) |
| Activation caching keyed by weights, settings and inputs | [Perturb and select units](../guides/perturb_and_select_units.md#perturb) |

## Experimental: usable, but interfaces may change

| Feature | Notes |
| --- | --- |
| `VisionWrapper(kind='vlm')`, `kind='temporal'` and other large-model wrappers | Tested on specific registrations (Qwen-VL, BLIP-2, V-JEPA, VideoMAE). |
| Custom channels, tools and capabilities (`register_channel`, `register_tool`, `register_capability`) | [Extend UMI](../guides/extend.md). Long-term structure still under discussion. |
| `register_input_handler` | Superseded by capabilities; emits a deprecation warning. |
| Robotics harnesses: DROID, LIBERO, OpenPI instrumentation | See [archive](../archive/) for qualification records. |
| Game and multi-agent harnesses: Gymnasium, grid games, `multi_agent_rollout`, `Message` | [Experiment 8](../../notebooks/experiments/08_learning_in_a_closed_loop.ipynb), [Experiment 13](../../notebooks/experiments/13_passing_outputs_between_subjects.ipynb) |
| Attribution and activation patching | `brainscore.experiments.attribution` |
| Reasoning traces (`RecordReasoning`, `build_trace_subject`) | Records only reasoning text a provider exposes. |
| `PerceptWindow`, `ActivationWindow` (low-level capture used by the tools) | Prefer `RecordActivity` in experiments. |
| Brain-surface visualization | `brainscore.visualization` |
| Human participants through the same session interface | Prototype, `brainscore.harnesses.human`. |

## Names

Each part has one current name. The older names in the right column still work but emit a deprecation warning, except `interact`, which was renamed without an alias. Do not use the older names in new code.

| Current | Older name |
| --- | --- |
| `participate(session)` | `interact(session)` |
| `extractor=` | `activations_model=`, `preprocessors=` |
| `answer=` | `generation_fn=` |
| `policy=` | `action_fn=` |
| `intervention=` | `state_change_fn=` |
| `BufferedSession` | `InMemorySession` |
| `session.respond(event)` | `session.emit(event)` |
| `region_layer_map` values with `modality=` | `region_modality_map=` |

`process(...)` is not deprecated, but it is a convenience on `BrainScoreModel` and the adapters, not part of the contract. Benchmarks call `participate`.
