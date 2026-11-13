# Feature status

Features are grouped by how much you need to know to use them, and by how settled they are.

## Core: needed by everyone

| Feature | Where |
| --- | --- |
| `Subject`, `participate`, `reset` | [Core concepts 1](../02_core_concepts.md#1-subject), [Lesson 1](../../notebooks/lessons/01_your_first_subject.ipynb) |
| Channels, `StreamEvent`, the channel catalog | [Core concepts 2-3](../02_core_concepts.md#2-channel) |
| `Session`, `stimulus_session`, `EnvironmentSource`, control events, trial memory | [Core concepts 4](../02_core_concepts.md#4-session) |
| `StimulusSet` (stimuli with a column for each input channel) | [Write a benchmark](../guides/write_a_benchmark.md) |
| `Benchmark`, `Score`, compatibility and N/A | [Lesson 2](../../notebooks/lessons/02_your_first_benchmark.ipynb) |
| `BrainScoreModel` with `extractor` and `region_layer_map` | [Wrap your model](../guides/wrap_your_model.md) |
| `PytorchWrapper`, `VLMVisionWrapper`, `VideoWrapper`, `TextWrapper`, `AudioWrapper` | [Class map](../03_class_map.md#extractor-one-per-input-type) |
| Adapters for classic vision and language models; classic benchmarks on any subject | [Existing models](../guides/existing_models.md) |
| `load_model`, `load_benchmark`, `score`, the unified registry | [Score a model](../guides/score_a_model.md) |
| Image-processor pinning (`pin_image_processor`) so preprocessing does not drift between library versions | `brainscore_core.hf_compat` |

## Advanced: for specific study designs

| Feature | Where |
| --- | --- |
| `LayerSelector`, `CompositeSelector`, several regions per run | [Lesson 5](../../notebooks/lessons/05_layers_and_regions.ipynb), [Lesson 8](../../notebooks/lessons/08_choose_units.ipynb) |
| Native input forms: `vision:image`, `vision:video`, `VLMVideoWrapper`, `attrs['input_adaptation']` | [Core concepts 6](../02_core_concepts.md#6-compatibility) |
| Behavior: `TaskContext` responses (`Choice`, `Scale`, `Text`), reported and implied answers, `reply`, `likelihood`, readout, `prefer_path` | [Behavior](../guides/behavior.md) |
| Perturbation: `Selection`, `StateChange`, `intervention` | [Perturb and select units](../guides/perturb_and_select_units.md) |
| Unit selection: `IndexSelection`, `RandomSelection`, `FunctionalSelection` | same |
| Closed loop: `policy`, rows, `EnvironmentSource`, specs | [Closed loop](../guides/closed_loop.md) |
| `Experiment`, `TaskProtocol`, `SessionProtocol`, `CallableProtocol`, conditions, trials | [Experiments](../guides/experiments.md) |
| Recorders: `RecordNeural`, `RecordBehavior` (one presentation for both); `Intervene` | same |
| Tools: `RecordInputsOutputs`, `RecordActivity`, `Ablate`, `ScaleActivity`, `PatchActivity` | same |
| `TorchInstrumentation`, `RunRecord`, `replay_sessions`, `compare_outputs` | same |
| Timed delivery: `windowed_stream`, `stimulus_session(..., streaming=True)` | [Core concepts 4](../02_core_concepts.md#4-session) |
| Temporal helpers: `temporal_bin`, HRF convolution, contiguous-block cross-validation | `brainscore.temporal`, [Experiment 10](../../notebooks/experiments/10_story_listening_fmri.ipynb) |
| Score provenance and status (`score.attrs`, N/A scores) | [Score a model](../guides/score_a_model.md) |
| Activation caching keyed by weights, settings and inputs | [Perturb and select units](../guides/perturb_and_select_units.md#perturb) |

## Experimental: usable, but interfaces may change

| Feature | Notes |
| --- | --- |
| `VLMVisionWrapper`, `VideoWrapper` and other large-model wrappers | Tested on specific registrations (Qwen-VL, BLIP-2, V-JEPA, VideoMAE). |
| Custom channels, tools and capabilities (`register_channel`, `register_tool`, `register_capability`) | [Extend UMI](../guides/extend.md). Long-term structure still under discussion. |
| Robotics harnesses: DROID, LIBERO, OpenPI instrumentation | See [archive](../archive/) for qualification records. |
| Game and multi-agent harnesses: Gymnasium, grid games, `multi_agent_rollout` (one session per subject for the whole conversation; answers on `behavior:text`, heard on `text`) | [Experiment 8](../../notebooks/experiments/08_learning_in_a_closed_loop.ipynb), [Experiment 13](../../notebooks/experiments/13_passing_outputs_between_subjects.ipynb) |
| Attribution and activation patching | `brainscore.experiments.attribution` |
| Reasoning records (`RecordReasoning`) | Records only reasoning text a subject exposes in the extra data of its answer events (`meta['reasoning']`). A `BrainScoreModel` part cannot put reasoning there yet. |
| `PerceptWindow`, `ActivationWindow` (hooks on a torch network; `TorchInstrumentation` uses `ActivationWindow`) | Prefer `RecordActivity` in experiments. Pass the network, not the subject. |
| `Witness` (a panel per response of a subject, from its sessions) | `brainscore.witness`; watches sessions with `observed`. |
| Brain-surface visualization | `brainscore.visualization` |
| Human participants through the same session interface | Prototype, `brainscore.harnesses.human`. |

## Names

Each part has one name. The older names in the right column were removed without an alias; code that uses them fails with a `TypeError`.

| Current | Removed name |
| --- | --- |
| `participate(session)` | `interact(session)` |
| `extractor=` | `activations_model=`, `preprocessors=` |
| `answer=` | `generation_fn=` |
| `policy=` | `action_fn=` |
| `intervention=` | `state_change_fn=` |
| `region_layer_map` values with `modality=` | `region_modality_map=` |
| `backbone_id=` on the extractor | `backbone_id=` on `BrainScoreModel` |

`process(...)` is not deprecated, but it is a convenience on `BrainScoreModel` and the adapters, not part of the contract. Benchmarks call `participate`.
