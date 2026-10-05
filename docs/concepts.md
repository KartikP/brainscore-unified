# UMI concepts

UMI lets an experiment send inputs to a model, collect responses, and compare them with measured data. Tools can record those interactions or change model activity during an experiment.

Start with [Getting started](getting_started.md). See [Supported features and environments](supported_features.md) for proposed support and remaining validation.

## Subject

A **subject** is the model or policy being evaluated. Import these classes from `brainscore_core.model_interface`:

| Class | Use it for |
| --- | --- |
| `Subject` | A new integration that handles a session directly. Declare `identifier`, `in_channels`, and `out_channels`; implement `interact(session)`. |
| `UnifiedModel` | A `Subject` base with `process`, task/recording setup, layer maps, and modality declarations. |
| `BrainScoreModel` | A ready-made implementation combining extraction wrappers and capabilities. It extends `UnifiedModel`. |

A native `Subject` needs no layer map or `process()` method. Stateful subjects implement `reset()` to clear state between independent runs. `required_channels` can declare inputs that must be present.

Existing Brain-Score vision and language models can use adapters; compatibility evidence is listed in the support matrix.

## Sessions and channels

A **session** delivers inputs and collects outputs. In `interact(session)`, a subject reads `session.next_input()` until it returns `None`, and sends responses through `session.emit(event)`.

A **channel** identifies what an event carries, such as `text` or `neural:IT`. A `StreamEvent` contains a channel, payload, timestamp (`t_ms`), and optional metadata. Input and output declarations let experiments check compatibility before running.

Use an external package to add channels, capabilities, or experiment tools. See [Build tools and integrations](tool_authoring.md).

## Stimuli, and `StimulusSet`

A `StimulusSet` is a pandas table with one row per stimulus. `stimulus_id` links each row to its response. For the `BrainScoreModel.process()` path, recognized columns select the input modality:

| Columns | Modality |
| --- | --- |
| `image_file_name`, `image_path`, `filename` | Vision |
| `video_path` | Vision, with time |
| `sentence`, `text` | Text |
| `audio_path`, `audio_file_name`, `audio_file` | Audio |

When several modalities are present, the default priority is vision, text, then audio. Request `process(stimuli, multi_modality=True)` to use multiple modalities. External channel registration can add recognized columns.

## Assembly

An **assembly** is an xarray array with named dimensions and coordinates. Neural responses commonly have these dimensions:

- `presentation`: the stimuli, identified by `stimulus_id`.
- `neuroid`: recorded units, with layer, region, and unit identifiers.
- `time_bin`: response intervals, when the model returns a time course.

A **neuroid** is one recorded unit. In a CNN, each channel and spatial position can be a separate unit.

Some assembly metadata lives in a pandas MultiIndex. Use `assembly['layer']` to access it; checking only `'layer' in assembly.coords` can miss it. Follow the existing assembly constructors when building benchmark outputs.

## `region_layer_map`

For neural recording, this map selects the model units used to represent a brain region:

```python
region_layer_map = {'V4': 'layer3', 'IT': 'layer4'}
```

On a `BrainScoreModel`, `start_recording('IT')` then selects `layer4`. Choose the mapping using benchmark evidence; it is not automatically a scientifically validated match. Composite selections can combine units from several layers; see the [API reference](umi_api_reference.md#record-a-region).

## Layer path

A layer path is PyTorch's name for a module, such as `layer3.0.conv1`. List the paths on the module passed to the extraction wrapper:

```python
[name for name, _ in backbone.named_modules() if name]
```

Paths are relative to that module. A VLM wrapper may wrap only its vision tower, so its paths can differ from those on the full model. `nn.Sequential` uses numeric names such as `0` and `2` when its children have no explicit names.

## Modalities

For `BrainScoreModel`, preprocessors describe available input modalities. `required_modalities` identifies inputs the model cannot run without. Native subjects declare channels directly.

## Preprocessor vs. `activations_model`

A **preprocessor** prepares inputs: resize images, tokenize text, or resample audio. An **extraction wrapper** runs the model, records selected layers, and packages responses.

Image models commonly put a callable in `preprocessors['vision']` and a wrapper in `activations_model`. Text paths put the complete `TextWrapper` in `preprocessors['text']`; a tokenizer alone is insufficient. When the extraction wrapper already preprocesses, an identity preprocessor avoids doing the work twice.

See [Getting started](getting_started.md#choose-an-extraction-wrapper) for wrapper imports and the [CLIP registration](../brainscore/models/clip_vit_b_32/model.py) for a multimodal example.

## Benchmark and metric

A **benchmark** defines the data and evaluation procedure. It drives a compatible subject through a session or the retained model methods, then returns a score.

A **metric** compares model responses with target measurements. It receives outputs, rather than the model itself.

## Score, raw vs. ceiled

A `Score` holds a value and metadata. **Raw** means the direct metric result. **Ceiled** means adjusted using the benchmark's estimate of measurement reliability. State which form you report; normalization depends on the benchmark.

Some ceiled scores can exceed 1.0. That is not the same as an accuracy above 100%. A **null floor** is the chance or randomized baseline used to interpret a result.

## Finding models and benchmarks

```python
import brainscore
sorted(brainscore.model_registry)
sorted(brainscore.benchmark_registry)
```

These list the unified package's registered entries. Loaders also consult vision and language, so the lists are not a complete catalog of loadable identifiers. Use the registry key to load an entry; the object's reported `identifier` can differ.

## Where to go next

- [Getting started](getting_started.md): installation and a first run.
- [Build tools and integrations](tool_authoring.md): external extensions, recording, and interventions.
- [API reference](umi_api_reference.md): method details and examples.
- [Coming from Brain-Score](from_brain_score.md): existing models and migration.
