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

Open http://127.0.0.1:8765. Use matching unified, core, vision and language
revisions from `install/peer-revisions.json`; a stale editable install can expose
different channels or experiment APIs. The frontend build is app-local.

The Lab panel reports activity from public experiment events and counts completed
repetitions across all conditions. A repetition can process the entire stimulus
table as one batch; this is not a per-token or per-image completion estimate.
Neural recording is requested during that same presentation, rather than in a
second forward pass. Benchmarks without intermediate progress use an
indeterminate indicator. Progress messages contain only bounded metadata, never
activation tensors, and are throttled within each condition.

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

**Allow model downloads** is on by default in the Subject panel. It permits
missing model weights to download without the disk-space preflight. Turn it off
to restore the standard UMI download guard. The setting applies to the next
worker operation, including model loads and runs, and survives page refreshes
within this server session. It resets to on when the server restarts.

This uses `BRAINSCORE_SKIP_MODEL_DOWNLOAD_CHECK` in the lab worker only. It does
not change the parent shell, the library defaults, or Hugging Face offline
settings. The switch does not cancel a download in progress or estimate whether
a model fits in RAM. It is disabled during loads, validation and experiments.

The Subject panel reports model download progress from actual byte callbacks.
Direct checkpoint downloads use `urlretrieve` report hooks; Hugging Face downloads
use byte-oriented `tqdm` counters. The display reports the current file, received
bytes, known total size and completed file count. Unknown sizes remain
indeterminate. A completed file does not imply that the full model is ready;
other files and model initialization may follow. Cached loads can remain in the
preparation phase without a network download. The observer exists only during
subject loads and restores its hooks on success or failure.

### Videos with sound

For a subject with an audio input, video uploads automatically extract the first
soundtrack when **Extract the soundtrack for subjects with audio input** is on.
The stimulus row retains `video_path` and gains `audio_path`: both are explicit
inputs to one trial. V-JEPA and Wav2Vec2 receive their respective streams. This
preparation does not fuse their representations or change their declarations.

For videos already in the table, use **Extract audio from current video rows**
in the Stimuli step, then validate the changed protocol again. Explicit audio
paths already in a row take precedence. Silent videos remain video-only and show
an explanation; the importer never invents a silent audio input.

Extraction keeps the original video and creates a mono, 16 kHz, PCM16 WAV file.
It uses the container's first audio stream, shifts timestamps relative to the
container start and retains leading gaps. The row records the source video and
conversion settings. The artifact bundle copies both media files, and the Python
export uses those same files. Derived WAV files are cached for repeated use.

The lab uses FFmpeg from the system or from the `imageio-ffmpeg` wheel included in
the lab's dependencies. No model runs during extraction. Operations are serial,
time-limited to 120 seconds, and reject audio outputs at the 128 MiB limit rather
than silently keeping truncated audio. Extraction does not run during an active
model operation. Video-only subjects retain the original upload behavior unless
the user explicitly extracts a soundtrack.

The same identifier generates the same avatar using SHA-256. Color, accessories,
and a nine-pixel chest emblem are cosmetic. Input/output badges and available
tools come from declarations, not the avatar or a model-class test.

## Choose a workflow

### Protocol builder (default)

The builder follows the workflow in `notebooks/lessons/10_run_an_experiment.ipynb`
and `docs/guides/experiments.md`:

1. **Subject:** name the protocol and load a registered subject. Its declarations
   determine compatible tasks, recording regions and interventions.
2. **Stimuli:** paste text, import CSV, upload images/audio/video, or choose a
   registered stimulus set. Keep experimental factors as CSV columns.
3. **Task and response:** choose plain viewing, choices, ratings, written answers,
   or legacy labeled behavior. For choices, implied probabilities and reported
   confidence are separate options. Select automatic routing, generation,
   likelihood or a fitted readout. Readouts require separate labeled fitting
   examples; their CSV and media import controls are in this step. Specify an
   explicit trial-part order for multimodal tasks, such as `text,image`.
4. **Recordings:** choose behavior, one or more declared neural regions, or both.
   The task and recorders are independent: neural activity can be measured during
   a task without also saving its answers. Joint recordings describe the same
   presentation. A replay log is optional and separate from measurement saving.
5. **Conditions:** name independent subject states, optionally attach a zero,
   scale or drive intervention with an explicit target and units, and choose
   repetitions. Choose whether a readout is fitted on the current or intact
   network in step 3. Session memory is an explicit requirement, when needed.
6. **Review and run:** inspect the complete configuration and click **Validate
   setup**. Validation builds a real `TaskProtocol` with the selected tools
   and calls `Experiment.validate()` without inference. **Run protocol** becomes
   available after success. Changing the draft or subject requires validation
   again; task-specific model checks also run when the session starts.

Execution uses `Experiment.run()`, `RecordNeural`, `RecordBehavior` and conditional
`Intervene` tools. The experiment owns resets and cleanup. Each condition and
repetition has its own saved measurements. The result selectors inspect these
artifacts without running the subject again. Color scales persist across
inspected conditions for each measurement, expanding when a larger value appears.

**Export Python** produces an editable UMI program before any inference.
**Save config JSON** preserves the draft and subject identifier for machine
replay. Uploaded file references in a draft belong to this workspace; after
execution use **All artifacts** for the Python program, configuration and copied
inputs together. Registered stimulus sets
retain their identifier and require their data source on replay.

Declared channels are the authority. A subject declaring only `response_trace`
does not automatically support `behavior:text`; use **Quick session** for its
generation-request path. Some legacy adapters accept only a single recording
channel in a session; unsupported combinations fail explicitly. Functional
localizers, adaptive environments and custom analysis/scoring still use Python
protocols or their existing benchmarks. This builder configures static
`TaskProtocol` experiments rather than generating arbitrary executable code.

### Quick session

1. Select a declared output: `response_trace` for generated text,
   `neural:<region>` for activations, or `behavior` for labeled task responses.
2. Paste text, one stimulus per line; import a CSV; upload images, audio, or
   video; or select a registered stimulus-set identifier.
   For media, choose **Images, audio or video** under **Stimulus source**, then
   click **Choose files** in the upload area. The uploaded rows remain editable
   as CSV, and switching sources preserves each source's text during the visit.
3. For behavior, provide an instruction and response labels, or expand the
   fitting-stimuli section and supply a labeled CSV for a fitted readout.
4. Click **Run experiment**. The app creates a session and calls `participate`.

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

For a subject declaring `generation_request` and `response_trace`, choose
`response_trace`, enter questions one per line, and supply a task instruction.
The GUI builds generation-request events and displays the returned text verbatim,
without forcing it into a label. Neural recording and intervention availability
come from the loaded subject's declarations. Checkpoint setup and scientific
mapping notes belong to that model's registration documentation.

CSV columns include `stimulus_id`, `sentence`, `image_path`, `audio_path`, and
`video_path`. Media columns reference files uploaded through this app. Uploading
files generates corresponding CSV rows. Add aligned text or label columns for
a multimodal experiment. Uploads are capped at 128 MiB per file; CSV/JSON requests
at 1 MiB. Use registered stimulus sets for larger datasets.

Full and partial matches run; the compatibility result and dropped streams are
saved with the experiment. The GUI does not invent missing inputs. Complete
labeled assemblies remain on disk and in the artifact export.

### Visualize results

The result cards have **Plot** and **Data** views. Behavioral plots show raw
choice values for a selected stimulus, including ties and missing values.
Generated text uses exact-response counts, preserving case, punctuation, and
parse errors. These plots summarize the recorded preview, not accuracy.

For registered neural outputs, the heatmap shows **every recorded neuroid** for
the selected stimulus, wrapped into rows in recording order. It is not an
anatomical map. Use the stimulus number to inspect other presentations, zoom
to enlarge the cells, scroll to pan, and click or use arrow keys to inspect a
neuron. For temporal outputs, use **Time bin** to scrub through recorded bins.
The slider and direct bin entry show native start/end times when available.
The selected stimulus remains selected across time changes. Conditions and
inspected time bins share one symmetric color scale, which expands when a larger
value is found. Other tensor axes use index zero and the fixed slice is reported;
values are not pooled or normalized.

The full heatmap reads existing `.npy` artifacts with memory mapping in pages
of at most 2,048 values, independently of the inference worker. It performs no
additional model forwards. The viewer's numeric buffer is capped at 128 MiB;
oversized recordings fail explicitly rather than silently omitting neurons.
The **Data** tab retains a small labeled numerical preview, and the complete
recording is available in **All artifacts**. Calibration probe plots show up
to eight existing recording events with at most 24 units per event; the default
comparison circuit has eight units, all of which fit.

The **Neural view** selector adds these temporal views for recorded time axes:

- **Neurons across time:** rows are neurons and columns are time bins in native
  order. Select the first neuron, first time bin and page sizes. Every neuron
  and bin is reachable; displayed ranges are explicit. Columns have equal widths,
  even when their physical intervals differ. Click or use arrow keys for exact
  values and time bounds. Missing values are gray.
- **Neuron response traces:** enter up to eight distinct neuron indices,
  separated by commas. Indices start at zero. The horizontal axis uses native
  physical intervals when present, or bin indices when bounds are unavailable.
  Each physical bin appears as a constant segment; missing values and gaps break
  traces. No smoothing, interpolation or normalization is applied.

Temporal windows read at most 65,536 values per request, with at most 1,024 bins
and 2,048 neurons. Reduce the page sizes if a request exceeds this limit. The
controls label direct neuron indices as zero-based and page starts as one-based.
Audio and video channels retain independent time grids; the GUI does not create
a shared clock. Existing saved experiments can use these views without a rerun.

The selector also provides three stimulus comparisons within the same channel,
condition, repetition and selected time bin:

- **Difference between two stimuli:** choose A and B to see all units as A minus
  B, plus the 20 largest absolute changes. The difference uses its own symmetric
  color range. Persistent offsets cancel, while missing values remain missing.
- **Stimulus variation:** show up to 32 units ranked by their range across all
  stimuli, with each unit's mean across those stimuli subtracted. This removes
  constant feature offsets for display without dividing by variance. It is a
  selected descriptive view; raw and difference heatmaps retain every unit.
- **Pairwise distances:** compare all stimulus pairs with RMS activation
  difference or correlation distance (1 minus Pearson r). Each pair reports its
  jointly finite unit count. Constant vectors have undefined correlation. Shared
  feature outliers may dominate correlation, so RMS is the default. Absolute RMS
  scales should not be compared across different regions.

Comparisons load only when selected, use saved paged arrays, and never invoke a
model. The browser explicitly refuses comparisons exceeding 64 stimuli, 65,536
units or 1,000,000 values rather than silently sampling. Other axes use the
same reported fixed slice as the raw view. A few images provide descriptive
comparisons, not statistical evidence of category selectivity.

The **Replay** tab displays a saved stimulus beside every recorded neural region.
Choose its condition, repetition and stimulus, then select **Play**. The time
slider seeks through the same clip. Playback uses saved data only and never
loads or runs the subject. Switch to Results to pause playback.

The player resolves each neural presentation by stimulus ID. Each region uses
its own native physical time bounds, so audio and video can change at different
rates. Gaps and overlapping intervals stay explicit. Outputs without a temporal
axis appear as static responses; temporal outputs without physical bounds cannot
be synchronized to media. These timestamps locate model output bins, not causal
online responses. Models can use information from other parts of the clip.

Video plays with its saved model audio when an audio input exists. Image and
text stimuli remain visible while a recorded physical clock advances. The saved
audio owns the clock while it plays; the silent video follows it. Replay seeks
audio only at an explicit start or seek action, not on every animation frame.
Missing or unsupported media produces an explicit message; the time slider can still
inspect available neural data. No duration is invented for static data without
a media or physical clock.

Replay buffers the selected stimulus before playback, with a combined limit of
4,000,000 neural values across regions. Every recorded neuron is included.
Oversized selections use the paged Results views instead. Color scales remain
fixed across time within each region and can expand across inspected stimuli or
conditions. Different regions have separate scales. The layout is not anatomical.

Frontend checks: `cd frontend && npm test`. Backend checks:
`PYTHONPATH=backend python -m pytest tests -q`. Use the pinned UMI environment;
the pre-existing local scikit-learn 1.5.2 installation remains outside its
runtime compatibility bounds and is not qualified for benchmark scoring.

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

Choose **Event JSON**, select a declared JSON-compatible input and output channel, edit the event list,
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

In the protocol builder, the behavior-log and activation-probe tools mirror the
recordings selected in step 4, and ablation opens the conditions setup. In Quick
session, behavior-log controls persistent session input/output recording, and
the output selector chooses the measurement. The activation-probe shortcut
selects the first declared neural output.

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

Stopping a protocol waits for the current repetition and cleanup. Other sessions
stop after their current independent trial; a benchmark runs as one operation
and stops after that operation. The app does not interrupt an in-flight
GPU call. Closing the server requests cleanup, then terminates its isolated
worker if necessary after five seconds.

## Artifacts and headless execution

Each `.lab-runs/<id>/` contains `plan.json`, `summary.json`, and experiment
artifacts. Uploaded media are copied into that run's `inputs` directory. Download
the **Python script**, configuration, summary, or complete archive. The archive
also contains `experiment.py`, generated directly from the saved configuration.
Old runs gain this export on download without running their subject again.
Registered stimulus sets retain their
identifier and require their data source when re-run.

Extract **All artifacts** and run or edit the program:

```bash
python experiment.py --output ./my-rerun
python experiment.py --validate-only
```

The Python program shows the subject, stimuli, task/response, recorders,
interventions, conditions and repetitions as public UMI calls. For a registered
protocol or benchmark it needs UMI v2 and model/data dependencies, not a running
webserver or the lab package. Downloading only the script works for text inputs;
for uploaded media, keep the archive's `inputs` directory beside it or pass
`--inputs /path/to/extracted/inputs`. It creates a new output directory and never
overwrites the downloaded recordings. Validation loads the model but performs
no inference. Model weights are not included in the archive.

Calibration and custom-profile exports explicitly require the app-local profile
factory and `umi_lab` to be importable. New profile runs record that factory;
older exports without it require `--factory module:create_profile` (for the
calibration circuit, `--factory umi_lab.demo:create_profile`).

The JSON is the machine-readable configuration used to reproduce the exact
settings, automate runs, or compare plans. It is not a Python program and does
not itself contain the measured arrays. It can be replayed through the lab CLI:

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
imports. These local runtime files are ignored by Git. Use an aligned UMI
environment or `python lab.py --core /path/to/matching/core`; ensure the unified
checkout and its other peers also match the pinned revisions.

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
