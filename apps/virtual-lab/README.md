# UMI virtual lab

A local GUI for loading subjects, running existing benchmarks, presenting your
own stimuli, collecting neural/behavioral responses, and applying declared
interventions. The pixel laboratory shows real Python execution. The calibration
circuit is one optional experiment, not the application's scope.

## Start

Use the Python 3.11 UMI environment. From this directory:

```bash
cd frontend
npm ci
npm run build
cd ..
python lab.py
```

Open http://127.0.0.1:8765. This workspace already has the frontend built and a
GUI-local runtime configuration; restarting requires only `python lab.py`.

## Load a subject

Type part of a name in **Find a model** (for example, `qwen`, `gpt`, or
`resnet 50`). Matching models appear immediately below, grouped by registry
source. Select one and click **Load subject**. Clear the search to browse the
full dropdown. Search ignores case and punctuation and never loads weights.
This calls the existing unified `load_model`, including its
vision/language fallbacks. It may download weights or require model-specific
dependencies and credentials. The registry list does not certify those assets.
The dropdown preserves the loaded model when the catalog finishes loading.

The worker releases the old subject before loading the next one. There is one
live model, not one copy per tab or run. A failed load leaves the loader usable.

The same identifier generates the same avatar using SHA-256. Color, accessories,
and a nine-pixel chest emblem are cosmetic. Input/output badges and available
tools come from declarations, not the avatar or a model-class test.

## Choose a workflow

### Stimuli

1. Select a declared output: `response_trace` for generated text,
   `neural:<region>` for activations, or `behavior` for labeled task responses.
2. Paste text, one stimulus per line; import a CSV; upload images, audio, or
   video; or select a registered stimulus-set identifier.
3. For behavior, provide an instruction and response labels, or expand the
   fitting-stimuli section and supply a labeled CSV for a fitted readout.
4. Click **Run experiment**. The app creates a session and calls `interact`.

The fitted-readout CSV needs a `label` column and the appropriate stimulus
columns. The behavioral path can be explicitly set to generation or readout.
For a download-free first run, load `chance-baseline`, leave the two sample
sentences and `yes,no` labels, and run. Its returned choice values are 0.5/0.5.
This is a null control that ignores stimulus content. Equal maximal choice
values are displayed as **No preference**, rather than arbitrarily choosing the
first label. Behavioral rows include their stimulus text, and refreshing restores
the displayed run's inputs and settings. For question answering, load an
instruction-following model and make the task instruction describe that task;
the registered GPT-2 feature/readout model is not an instruction-following model.

The `qwen2.5-0.5b-instruct` registration is a small local instruction-following
option. In this workspace its pinned checkpoint is downloaded under `.runtime`
and selected through `BRAINSCORE_QWEN25_05B_PATH`. Elsewhere its normal model
download pre-flight applies. Choose `response_trace`, enter questions one per
line, and supply a task instruction. The GUI builds generation-request events
and displays the returned text verbatim, without forcing it into a label. The
same model also exposes `neural:language_system` and intervention channels.
Its neural mapping is a provisional last-layer hypothesis, not a validated
brain-region fit. This small model can give incorrect factual answers.

CSV columns include `stimulus_id`, `sentence`, `image_path`, `audio_path`, and
`video_path`. Media columns reference files uploaded through this app. Uploading
files generates corresponding CSV rows. Add aligned text or label columns for
a multimodal experiment. Uploads are capped at 128 MiB per file; CSV/JSON requests
at 1 MiB. Use registered stimulus sets for larger datasets.

Full and partial matches run; the compatibility result and dropped streams are
saved with the experiment. The GUI does not invent missing inputs. It previews
at most 20 behavioral rows or 16 neural values. Complete labeled assemblies
remain on disk and in the artifact export.

### Benchmark

Select **Benchmark**, enter an existing identifier, and run. The app calls
`brainscore.score` with the loaded subject and registered benchmark, including
the scorer's compatibility and memory checks. It does not rewrite the benchmark's
stimuli, fitting, cross-validation, metric, or ceiling.

Missing data, dependencies, credentials, and incompatible pairs are reported.
Incompatibility is N/A, never zero. Exploratory tools are disabled in this mode.
Use an existing intervention benchmark or a custom session when the protocol
requires perturbation. Benchmark artifacts retain the score and metadata.

### Session JSON

Choose a declared JSON-compatible input and output channel, edit the event list,
and run. The editor supports generation, conditional log probabilities, and new
JSON-native families whose catalog declares their payloads. Guidance comes from
the catalog; a new JSON channel requires no frontend dispatch change.

For a causal-LM subject declaring these channels:

```json
[
  {
    "channel": "conditional_logprob_request",
    "payload": {
      "contexts": ["The capital of France is", "The capital of France is"],
      "targets": [" Paris", " London"],
      "reduction": "sum"
    },
    "t_ms": null
  }
]
```

Raw answers, validity, and parse errors from response traces are retained. The
editor does not coerce dictionaries into environment or other typed dataclasses;
those need their benchmark or an explicit Python harness.

### Calibration

**Calibration circuit** loads an engineered eight-unit PyTorch circuit with two
signal units. Its preset demonstrates recording, first-N-unit ablation, a seeded
size-matched random control, and restoration. These are not scientific benchmark
scores. Controls are disjoint when the dose permits; overlap is reported.

## Recording and interventions

The behavior-log tool controls persistent session input/output recording.
Selecting a neural output collects its activation assembly. The activation
probe shortcut selects the first declared neural output; use the output selector
to choose another region.

For a custom session, expand **Intervention**. Enter a target address, optional
unit indices, and zero/scale/drive settings. The subject must declare the
corresponding lesion/stimulation input and perturbation output. The app uses
the existing typed state-change helper and selective cleanup, not private model
inspection. Runs have baseline, intervention, and restored conditions.
Addresses follow UMI's common naming convention; `:input` addresses an input site
when the registered handler supports it.

The core experiment loop owns resets. Every intervention has selective undo in
`finally`. Restoration compares full outputs and coordinates, not just the
preview, using bounded temporary buffers.

Stopping waits for the current independent trial; a benchmark runs as one
operation and stops after that operation. The app does not interrupt an in-flight
GPU call. Closing the server requests cleanup, then terminates its isolated
worker if necessary after five seconds.

## Artifacts and headless execution

Each `.lab-runs/<id>/` contains `plan.json`, `summary.json`, and experiment
artifacts. Uploaded media are copied into that run's `inputs` directory. Download
the plan, summary, or complete archive. Registered stimulus sets retain their
identifier and require their data source when re-run.

```bash
python lab.py --run /path/to/extracted/run/plan.json --output ./direct-runs
```

Keep the archive's `inputs` directory beside its plan for media replays. Plans
store structured settings, never executable Python or model weights.

## Custom profiles and typed harnesses

`--factory your_module:create_profile` selects a trusted Python factory returning
`LabProfile`. This remains the integration point for typed environments, explicit
instrumentation, and scientific protocols beyond the editors. A profile supplies
its subject, session factory, declarations, tools, summary, and provenance.

`umi_lab.huggingface:create_profile` is an optional causal-LM profile using the
existing UMI builder. `UMI_LAB_MODEL` selects a local/cached checkpoint,
`UMI_LAB_TARGET` its explicit site, `UMI_LAB_UNITS` the output width, and
`UMI_LAB_DEVICE` its device. Offline loading is the default;
`UMI_LAB_OFFLINE=0` permits downloading the chosen checkpoint. Its preset is an
exploratory completion task, not a registered benchmark.

## Runtime and performance

The optional `.lab-runtime.json` prepends app-local dependency paths before model
imports. This workspace uses a detached core worktree at `bdda009` and
scikit-learn 1.7.2 under `.runtime/`. This resolves the mismatched editable core
and shared environment for the GUI without switching another repo's branch or
changing its install. These local runtime files are ignored by Git. Elsewhere,
use an aligned UMI environment or `python lab.py --core /path/to/matching/core`.

The result cache defaults to app-local storage unless `RESULTCACHING_HOME` is
set. The server never imports model libraries. Weights remain in one persistent
spawned worker. Registry discovery is a startup operation and the catalog is
fetched separately, not repeated in progress polling. No tensor enters browser
state.

The frontend is plain TypeScript and Canvas 2D, without a runtime framework,
WebGL, CDN, or external fonts. Unchanged idle frames are skipped; active animation
is capped at four frames per second. Progress uses bounded nonblocking messages.

The budget is **25% added warm experiment runtime** against the same model,
experiment, recording tools, and artifacts without GUI telemetry. Recording
itself has a scientific cost on both sides. Browser and GPU memory are separate
qualification requirements. See [performance qualification](PERFORMANCE.md).

```bash
python lab.py benchmark --pairs 9 --repeats 10
PYTHONPATH=backend python -m pytest tests -q
cd frontend
npm run check
npm run build
```

The timing check warms both paths, alternates order, verifies encoded outputs,
and polls the HTTP endpoint ten times more frequently than the normal UI. It
exits unsuccessfully above 25%. Large-model GPU qualification is model-specific.

The server binds only to loopback, checks Host/Origin and a per-launch token,
accepts no browser-supplied Python code, and serves only built assets and known
run exports. It is a local single-user app.

## Coverage limits

This workbench does not yet cover every extension visually. Gaps include live
environment setup/playback outside registered benchmarks, functional-localizer
design, layer-mapping search, activation-patch reference management, multi-region
selection, parameter sweeps, and run-history comparison. Those remain Python or
profile workflows. A model's declared channels must actually work; the GUI
reports failures and does not invent an adapter based on its class. Some plugins
also need separate environments or external services.
