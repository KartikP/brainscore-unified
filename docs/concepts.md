# Concepts

The vocabulary, in the order it is encountered. Intended to be read once before the
notebooks and consulted afterwards.

Every term defined here appears in error messages and function signatures, so the
definitions are what make most failures actionable.

---

## The one-sentence version

A model is wrapped as a **Subject**, configured with **what to record**, and given
**stimuli**; it returns an **assembly**, a labelled table of unit responses. A
**benchmark** performs those steps and then compares the result against measured brain
data using a **metric**.

The remainder of this document defines those six terms.

---

## Subject

A model, wrapped so Brain-Score can treat it like an experimental subject.

**Which name to use.** Import these names from `brainscore_core.model_interface`:

| Name | What it is |
| --- | --- |
| `Subject` | The abstract contract consumed by unified benchmarks and adapters. Use for type annotations or a custom implementation |
| `BrainScoreModel` | The concrete `Subject` subclass to instantiate for ordinary registrations. Composes wrappers, recording, and capability callables |
| `UnifiedModel` | The deprecated spelling of the same ABC: `UnifiedModel is Subject`. Retained for existing imports, subclasses, and `isinstance` checks; use `Subject` in new code |

`UnifiedModel` has no separate implementation — `UnifiedModel is Subject` is
literally true — and as of 2026-09-10 nothing inside these four repositories
uses the old spelling except its own definition and a few "formerly" notes. It
is exported solely so that code outside this tree importing
`from brainscore_core import UnifiedModel` keeps working. It is a compatibility
name, not a third kind of model. The permanent vision and
language adapters implement `Subject` too; existing domain plugins stay supported.

A `BrainScoreModel` exposes the contract's operations:

```python
model.start_recording('IT')     # what to measure
model.start_task(task_context)  # what task to perform (behavioral work only)
assembly = model.process(stimuli)   # run it, get responses back
```

`process()` is the **only evaluation method on the interface**. There is no `look_at`, no
`digest_text`, no per-modality entry point. What a model *can* do is decided by which
optional slots were filled at registration, not by which methods exist.

**Consequence.** The same call serves an image model, a language model, a video model
or a VLM. Benchmarks are written against `process()` alone and therefore need no
knowledge of the model type.

**Scope of the rule.** The constraint applies to the interface contract: a benchmark
reaches a model only through `process()`. Calling the underlying `nn.Module` directly
remains valid for other purposes, and notebooks 11 and 14 do so, measuring a lesion's
effect with a plain `net(x)` where the perturbation rather than the recording path is
the subject of the demonstration.

## Stimuli, and `StimulusSet`

The input presented to the subject: a table with one row per stimulus. It subclasses
pandas `DataFrame`, so `len()`, slicing and column access behave conventionally.

The column names are load-bearing. Dispatch decides **which modality** a stimulus set
carries by looking for a *recognized* column:

| Column | Modality |
| --- | --- |
| `image_file_name`, `image_path`, `filename` | vision |
| `video_path` | vision — video is temporal vision, not a separate channel |
| `sentence`, `text` | text |
| `audio_path`, `audio_file_name`, `audio_file` | audio |

When a set carries columns for more than one modality, `MODALITY_PRIORITY`
(`vision`, `text`, `audio`) breaks the tie. To use both towers instead of picking one,
call `process(stimuli, multi_modality=True)`.

A set carrying only `stimulus_id` and unrecognised columns raises *"No recognized
modality columns"*. The error indicates that the modality could not be determined, not
that the data is malformed.

`stimulus_id` identifies each row and is the key by which responses are matched back to
stimuli.

## Assembly

The return value: an `xarray` array with **labelled** axes, usually
`(presentation, neuroid)`, holding one row per stimulus and one column per recorded unit.
It is a numpy array that retains the meaning of every row and column.

- **presentation** — the stimuli, carrying `stimulus_id` and any accompanying metadata.
- **neuroid** — the recorded units, carrying `layer`, `region`, `neuroid_id`.
- **time_bin** — present when the model returns a time course (video, audio, per-token text).

**Metadata storage.** Some metadata is held as a plain coordinate and some as a level
inside a pandas MultiIndex. `.coords` lists only the former:

```python
'layer' in assembly.coords     # False
assembly['layer']              # works fine
```

Metadata should be read through `assembly['name']`, which resolves both forms, and
enumerated with `assembly.indexes['neuroid'].names`. A guard written as
`if 'layer' not in assembly.coords` discards information silently rather than raising.

Assemblies should be constructed with **at least two coordinates per axis**. With one,
the MultiIndex is not created and downstream code cannot locate `stimulus_id`.

## Neuroid

One recorded unit, the model's analogue of a neuron. For a CNN layer of shape
`(channels, height, width)` each `(channel, x, y)` position is a separate neuroid, which
is why unit counts reach the hundreds of thousands. Each carries its source `layer` and,
when several regions are recorded together, its assigned `region`.

## `region_layer_map`

The dictionary that says which model layer stands in for which brain region:

```python
region_layer_map={'V1': 'layer1', 'V2': 'layer2', 'V4': 'layer3', 'IT': 'layer4'}
```

The mapping is asserted by the model author rather than discovered by Brain-Score. Any
region may map to any layer. `start_recording('IT')` then records `layer4`.

Two consequences follow. Passing a **list** records several regions in one forward pass
and tags every unit with its region. An unrecognised *string* is treated as a raw layer
path, which serves as an escape hatch; an unrecognised region inside a *list* raises.

## Layer path

The right-hand side of `region_layer_map` (`'layer4'`, `'encoder.layers.10'`,
`'backbone.blocks.16'`) is a **layer path**: PyTorch's name for a module within the
network. These are determined by how the model was constructed.

**Paths are relative to the module the extraction wrapper wraps, not to the model passed
to `BrainScoreModel`.** The two are frequently the same object, in which case the
distinction does not surface. They differ whenever the wrapper is pointed at a
sub-module, which is the usual arrangement for VLMs: their `forward()` requires every
modality's input simultaneously, so the wrapper takes a single tower.

CLIP is the shipped example. Its registered paths are **absent** from the full model:

```python
model = brainscore.load_model('clip-vit-b-32')
model.region_layer_map['IT']                                  # 'encoder.layers.10'
'encoder.layers.10' in [n for n, _ in model._model.named_modules()]   # False (!)
```

They are relative to the vision tower held by the wrapper, a `CLIPVisionTransformer`.
Paths should therefore be enumerated from **the module the wrapper was constructed on**:

```python
backbone = ...                       # the module passed to VisionWrapper/PytorchWrapper
[name for name, _ in backbone.named_modules() if name]
```

For a `torchvision` ResNet-18 (where wrapper and model are the same object) that starts
`['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer1.0', ...]`; dots are nesting, so
`layer3.0.conv1` is the first conv of the first block of `layer3`.

For a multi-tower registration each tower is enumerated separately against its own
wrapper: vision and text paths occupy different namespaces and may legitimately collide.
CLIP contains an `encoder.layers.10` in both.

**Why some layers are called `'0'`, `'1'`, `'2'`.** `nn.Sequential` does not name its
children, so PyTorch numbers them by position:

```python
net = nn.Sequential(nn.Conv2d(3, 8, 3), nn.ReLU(), nn.Conv2d(8, 16, 3))
[name for name, _ in net.named_modules() if name]      # -> ['0', '1', '2']
```

That is the whole explanation for `Selection(layer='0')` in notebook 11, and for the
`# '0'` / `# '2'` annotations in `templates/new_model/model.py`.

**Selecting a layer.** Enumerating the paths establishes what is available, not which
is appropriate. Two tools assist: `brainscore.tools.auto_register.inspect_model(model)`
proposes a provisional map from the architecture, and the layer-mapping explorer scores
candidate layers against a benchmark, allowing a choice to be made on evidence. The
provisional map is a starting point rather than a result.

## Modalities

The kinds of input a model accepts. Derived rather than declared separately:

```python
model.supported_modalities   # comes from preprocessors.keys()
```

A model with a `'vision'` preprocessor supports vision. No separate list is maintained
and no `hasattr` checks are used. `required_modalities` is the stricter statement that a
model *requires* an input: GPT-2 requires text, whereas CLIP requires nothing specific
because it accepts either.

## Preprocessor vs. `activations_model`

Two distinct responsibilities:

- **preprocessor** — a plain callable, one per modality, converting raw stimuli into the
  model's expected input format: resize and normalize, tokenize, resample.
- **activations_model** — the wrapper that runs the forward pass, hooks the requested
  layers, batches, caches, and packages the result as an assembly.

The wrapper performs the extraction. Three are available, selected by input type:
`VisionWrapper` for images, VLMs and video, `TextWrapper`, and `AudioWrapper`.

**Relationship to `PytorchWrapper`.** `VisionWrapper` is a facade: it inspects the
model and dispatches to `PytorchWrapper` for standard image models,
`VLMVisionWrapper` for VLMs whose patches arrive concatenated rather than stacked, or
`VideoWrapper` for native-temporal models. A vision registration therefore requires only
the one name.

Several notebooks use `PytorchWrapper` directly, since naming the concrete class makes
the individual stages visible. Both forms are correct. For registration `VisionWrapper`
is shorter and is what the `auto_register` scaffolder emits, which accounts for the
difference in naming between this page and those notebooks.

The preprocessor follows the same division. Where the wrapper already performs
preprocessing, as it does throughout these notebooks, the preprocessor is the identity
function:

```python
preprocessors = {'vision': lambda stimuli: stimuli}
```

This is not a placeholder: it records that preprocessing has already occurred in the
wrapper.

**Vision and text are wired differently.** A multimodal registration typically takes this
form:

```python
preprocessors={'vision': preprocessing,   # a bare callable; the wrapper is separate
               'text': text_wrapper},     # the whole TextWrapper goes IN here
activations_model=activations_model,      # the vision wrapper lives here
```

For vision the preprocessor and the extraction wrapper are separate objects. For text
the `TextWrapper` is itself the preprocessor entry, performing both preparation and
extraction. A tokenizer alone is not a text preprocessor.
`brainscore/models/clip_vit_b_32/model.py` is the maintained example.

## Benchmark

Data together with a scoring procedure. Given a subject it configures recording, calls
`process()`, compares the output against measurements, and returns a `Score`. Benchmarks
reach models **only** through the three operations above, which is what allows any model
to run against any compatible benchmark.

## Metric

The comparison itself: two assemblies in, one number out. A metric never receives the
model, only its output, which keeps metrics reusable across models and benchmarks.

## Score, raw vs. ceiled

A number with attached metadata. Two forms exist, and conflating them is the most
common reporting error:

- **raw** — the metric's direct output.
- **ceiled** — raw divided by the **ceiling**, the maximum attainable given noise in the
  data, usually estimated by split-half reliability.

Ceiled scores are the comparable form. **A ceiled score can exceed 1.0**, indicating that
the model predicted held-out data better than one half of the data predicts the other.
This is uncommon but legitimate.

Raw and ceiled numbers must not appear in the same table or on the same axis, and the
form in use must always be stated.

## Null floor

The score obtained in the absence of signal, measured either as a chance baseline or as
a random-weight model of the same architecture. A result counts only if it exceeds its
null. Nulls should be measured first on any new benchmark: a model that does not exceed
random features is reporting noise regardless of its absolute value.

---

## Seeing what exists

```python
import brainscore
sorted(brainscore.model_registry)        # 27 models defined in this package
sorted(brainscore.benchmark_registry)    # 30 benchmarks defined in this package
```

**These lists are incomplete.** `load_model` and
`load_benchmark` check this package first, then fall back to the vision and language
registries — and those populate *lazily*, one plugin at a time, the first time something
in them is loaded. So:

```python
'MajajHong2015public.IT-pls-unified' in brainscore.benchmark_registry   # False
brainscore.load_benchmark('MajajHong2015public.IT-pls-unified')         # works fine
```

Inspecting the registry and concluding that the documented example benchmark does not
exist is a natural but incorrect inference. The vision registry is empty until something
is loaded from it; after a single `load_benchmark` call it holds ten MajajHong entries.

No single call currently enumerates everything reachable. The registries should therefore
be read as the set this package defines, not the set that can be loaded.

---

## Supporting terms

**Hook.** A callback PyTorch invokes when a module produces output. Wrappers attach one
to each requested layer, so a single forward pass captures intermediate activations
without modifying the model. This is the mechanism by which activations are obtained;
nothing is re-implemented or re-run per layer.

**Registration.** Making a model or benchmark loadable by name. Concretely: a directory
under `brainscore/models/<name>/` whose `__init__.py` adds an entry to `model_registry`,
together with one `from . import <name>` line in the parent `__init__.py`. No database
or decorator is involved. `templates/new_model/` is a working example.

**Pandas MultiIndex.** An index with several named levels rather than a single column of
labels, allowing one axis to carry `stimulus_id`, `object_name` and `subject`
simultaneously. Brain-Score constructs one on the `presentation` and `neuroid` axes so
that every row and column retains its metadata. Two consequences follow: the levels do
not appear in `.coords`, so `assembly['name']` should be used, and an axis with a single
coordinate is not promoted to a MultiIndex, which is why assemblies require at least two
coordinates per axis.

**Tower.** One modality's path through a multimodal model. CLIP has a vision tower and a
text tower. Recording both towers means extracting from each rather than allowing
`MODALITY_PRIORITY` to select one.

**Reading a benchmark identifier.** `MajajHong2015public.IT-pls-unified` decomposes as:

| Part | Meaning |
| --- | --- |
| `MajajHong2015` | the dataset, by first author and year |
| `public` | the openly available split (some data is access-controlled) |
| `IT` | the brain region measured |
| `pls` | the metric — PLS regression from model units onto neural units |
| `unified` | scored through `process()` rather than the legacy per-domain path; regression-validated to give the same numbers, so the difference is plumbing |

**Identifier mismatch.** The name used to load a benchmark and the name the object
reports are not always the same string:

```python
b = brainscore.load_benchmark('MajajHong2015public.IT-pls-unified')
b.identifier            # -> 'MajajHong2015.IT.public-pls-unified'   (NOT loadable)
```

The registry key is the loadable form. `benchmark.identifier` is what appears on
results, so both spellings occur in practice, and a reported identifier cannot be assumed
to work as an argument to `load_benchmark`.

---

## Where to go next

- **Use it:** `docs/getting_started.md`, then the notebooks in `notebooks/README.md`.
- **Extend it:** `EXTENDING.md` and the runnable skeletons in `templates/`.
- **Look something up:** `docs/umi_api_reference.md`.
- **Coming from classic Brain-Score:** `docs/from_brain_score.md`.
