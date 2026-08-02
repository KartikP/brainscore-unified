# Concepts

The vocabulary, in the order you meet it. Read once before the notebooks; come back when
a word stops making sense.

Every term here appears in error messages and function signatures, so knowing them turns
most failures into something you can act on.

---

## The one-sentence version

You wrap a model as a **Subject**, tell it **what to record**, hand it **stimuli**, and
get back an **assembly** — a labelled table of unit responses. A **benchmark** does
exactly that and then compares the result against real brain data with a **metric**.

Everything below is detail on those six words.

---

## Subject

A model, wrapped so Brain-Score can treat it like an experimental subject. In code it is
`BrainScoreModel`, and it has essentially three verbs:

```python
model.start_recording('IT')     # what to measure
model.start_task(task_context)  # what task to perform (behavioral work only)
assembly = model.process(stimuli)   # run it, get responses back
```

`process()` is the **only** evaluation method. There is no `look_at`, no `digest_text`,
no per-modality entry point. What a model *can* do is decided by which optional slots
were filled at registration, not by which methods exist.

**Why it matters:** the same call works for an image model, a language model, a video
model, or a VLM. Benchmarks are written against `process()` alone, so a benchmark never
has to know what kind of model it is scoring.

## Stimuli, and `StimulusSet`

What you show the subject: a table with one row per stimulus. It is a pandas DataFrame
subclass, so `len()`, slicing, and column access work as you would expect.

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

A set with only `stimulus_id` and columns you invented raises *"No recognized modality
columns"*. That error means "I cannot tell what kind of data this is", not "your data is
malformed".

`stimulus_id` identifies each row and is how responses get matched back to stimuli.

## Assembly

What comes back: an `xarray` array with **labelled** axes, usually
`(presentation, neuroid)` — one row per stimulus, one column per recorded unit. Think of
it as a numpy array that remembers what every row and column means.

- **presentation** — the stimuli, carrying `stimulus_id` and whatever metadata came with them.
- **neuroid** — the recorded units, carrying `layer`, `region`, `neuroid_id`.
- **time_bin** — present when the model returns a time course (video, audio, per-token text).

**The gotcha worth knowing up front.** Some metadata lives as a plain coordinate and some
as a level inside a pandas MultiIndex. `.coords` only lists the first kind, so:

```python
'layer' in assembly.coords     # False
assembly['layer']              # works fine
```

Read metadata with `assembly['name']`, which finds both. Use
`assembly.indexes['neuroid'].names` to see what is actually there. Writing
`if 'layer' not in assembly.coords` is the mistake this warning exists to prevent — it
silently drops information rather than raising.

Related: build assemblies with **at least two coordinates per axis**. With only one, the
MultiIndex is never created and downstream code cannot find `stimulus_id`.

## Neuroid

One recorded unit — the model's analogue of a neuron. For a CNN layer of shape
`(channels, height, width)` every `(channel, x, y)` position is its own neuroid, which is
why unit counts reach the hundreds of thousands. Each carries the `layer` it came from
and, when you record several regions at once, the `region` it was assigned to.

## `region_layer_map`

The dictionary that says which model layer stands in for which brain region:

```python
region_layer_map={'V1': 'layer1', 'V2': 'layer2', 'V4': 'layer3', 'IT': 'layer4'}
```

This is a **commitment you make**, not something Brain-Score discovers. Any region can
map to any layer. `start_recording('IT')` then means "record `layer4`".

Two things follow. Passing a **list** records several regions in one forward pass, and
tags every unit with its region. Passing an unknown *string* is treated as a raw layer
path (an escape hatch); passing an unknown region inside a *list* fails fast.

## Modalities

Which kinds of input a model accepts. Derived — never declared twice:

```python
model.supported_modalities   # comes from preprocessors.keys()
```

If a model has a `'vision'` preprocessor, it supports vision. There is no separate list
to keep in sync, and no `hasattr` checks anywhere. `required_modalities` is the stricter
statement that a model *needs* an input (GPT-2 requires text; CLIP requires nothing in
particular because it can take either).

## Preprocessor vs. `activations_model`

Two different jobs, easy to confuse:

- **preprocessor** — a plain callable, one per modality, turning raw stimuli into what
  the model eats (resize and normalize, tokenize, resample). Simple functions.
- **activations_model** — the wrapper that runs the forward pass, hooks the requested
  layers, batches, caches, and packages the result as an assembly.

The wrapper does the heavy lifting. Pick the one that matches your input: `VisionWrapper`
(images, VLMs, video), `TextWrapper`, `AudioWrapper`.

**Why the examples say `PytorchWrapper` instead.** `VisionWrapper` is a front door: it
inspects your model and dispatches to `PytorchWrapper` (plain image models),
`VLMVisionWrapper` (VLMs, whose patches arrive concatenated rather than stacked), or
`VideoWrapper` (native-temporal models). So for a vision model you normally need only
that one name.

The notebooks reach past it and use `PytorchWrapper` directly, because seeing the
concrete class makes the moving parts visible while you are learning. Both are correct.
If you are registering a model, `VisionWrapper` is the shorter path — it is also what
the `auto_register` scaffolder emits. If you are reading a notebook and wondering why the
name differs from this page, that is why.

The preprocessor question follows the same seam. When the wrapper already handles
preprocessing — which it does in every notebook here — the preprocessor is the identity
function:

```python
preprocessors = {'vision': lambda stimuli: stimuli}
```

That looks like a placeholder and is not. It says "the wrapper did it."

## Benchmark

Data plus a scoring procedure. Given a subject, it configures recording, calls
`process()`, compares the output to real measurements, and returns a `Score`. Benchmarks
talk to models **only** through the three verbs above, which is what lets any model run
against any compatible benchmark.

## Metric

The comparison itself: two assemblies in, a number out. Metrics never see the model —
only its output — which keeps them reusable across models and benchmarks.

## Score, raw vs. ceiled

A number with metadata attached. Two forms, and confusing them is the most common
reporting error:

- **raw** — the metric's direct output.
- **ceiled** — raw divided by the **ceiling**, the best any model could do given noise in
  the data. Usually estimated by split-half reliability.

Ceiled scores are the comparable ones. **A ceiled score can exceed 1.0** — that means the
model predicted held-out data better than one half of the data predicts the other, which
is unusual but legitimate.

Never put raw and ceiled numbers in the same table or on the same axis. Always say which
one you are showing.

## Null floor

What the score would be with no real signal — a chance baseline, or a random-weight model
with the same architecture. A score only counts if it clears its null. Run the nulls
*first* on any new benchmark; a model that fails to beat random features is reporting
noise, however respectable the absolute number looks.

---

## Where to go next

- **Use it:** `docs/getting_started.md`, then the notebooks in `notebooks/README.md`.
- **Extend it:** `EXTENDING.md` and the runnable skeletons in `templates/`.
- **Look something up:** `docs/umi_api_reference.md`.
- **Coming from classic Brain-Score:** `docs/from_brain_score.md`.
