# Sun 2025: watch a clip and report motion

This UMI experiment presents controlled short clips and asks for the direction
and speed of motion at the indicated location/time. It compares model reports
with the three participants in Sun et al. (2025). See the
[data plugin](../../brainscore/data/sun2025/README.md) for sources and preparation.

The [interactive notebook](../../notebooks/experiments/15_sun2025_motion_and_activity_replay.ipynb)
walks through model selection, joint recording, human comparison, and synchronized
stimulus/activity replay. Run mode starts a new experiment; replay mode opens a
saved experiment without loading a model. The viewer provides play/pause, frame
stepping, a time slider, speed controls, and a standalone HTML export. It uses
the recording's time-bin coordinates and never changes the saved activity.

The [registered benchmark](../../brainscore/benchmarks/sun2025/README.md),
`Sun2025-motion-vector-error`, wraps this same procedure with the human
reference and a mean vector-error metric. It runs through `brainscore.score`.
Its full-study registration requires all 1,120 trials; smaller studies can
compose `Sun2025MotionBenchmark(protocol, reference, recorders=recorders)`.
Call `benchmark.score_measurements(result.behavior(measure='text'))` to score
existing recordings without presenting the clips again.

## The experiment, with both instruments visible

The registered data plugin supplies stimuli and human data separately. Any
subject declaring clip input and text replies can do the numerical-report task.
Only attach neural recorders for regions that the subject declares.

    import brainscore
    from brainscore.experiments import Experiment, RecordBehavior, RecordNeural
    from brainscore.experiments.sun2025 import make_protocol, analyze

    # 1. Choose a subject with declared clip, reply and optional neural outputs.
    subject = ...

    # 2. Load the stimuli and define the task and protocol.
    stimuli = brainscore.load_stimulus_set('Sun2025-second-order-motion', root=data_root)
    protocol = make_protocol(stimuli)

    # 3. Attach the instruments.
    tools = [RecordBehavior(), RecordNeural(['V1'])]

    # 4. Validate and run. One presentation provides both measurements.
    experiment = Experiment(subject=subject, protocol=protocol,
                            tools=tools, output_dir=run_root)
    experiment.validate()
    result = experiment.run()

    # 5. Inspect the answers and activity.
    behavior = result.behavior(measure='text')
    neural = result.neural('V1')

    # 6. Compare with separately loaded human reports.
    human = brainscore.load_dataset('Sun2025-second-order-motion', root=data_root)
    trials, comparison, summary = analyze(behavior, stimuli, human)

One row is a trial: a clip followed by one report containing both quantities.
One repetition is one independent session, reset before and after by the
protocol. A full session contains 1,120 shuffled trials. The subject's declared
memory policy applies within the session. The demonstration subject has no
cross-trial memory. A pilot with one scene has 28 trials.

## Native video example

run.py contains the complete workflow, including a no-motion control before
the real subject. qwen_video.py builds a small Qwen2.5-VL-3B subject with a
native video extractor, a reply part and an explicitly named V1 recording
mapping. It preserves all 16 frames at 30 Hz in this subject's video processor.
It is a separately named example subject; existing model registrations and
their default preprocessing are unchanged. The layer mapping is illustrative.

The checkpoint must already be in the Hugging Face cache at the pinned revision
listed in qwen_video.py; the example never downloads weights implicitly.
Run remotely, from the repository root:

    python examples/sun2025/run.py \
        --data /data/sun2025/prepared-pilot \
        --output /data/sun2025/runs/qwen-pilot

The output contains the original UMI recordings under model/, a control run
under no-motion-control/, raw answers and parse status in motion-reports.csv,
per-participant comparisons in human-comparison.csv, and summary/provenance JSON.
RecordNeural captures the prompt-reading forward pass, before generation.

For an existing run, reanalyze.py writes a new analysis directory without
changing the raw recordings or earlier summary. validate_run.py checks matching
neural/behavioral IDs, temporal patch count, row-order invariance, source hashes
and checkpoint weight hashes. The validation directory contains the pilot's
receipts and explains the explicit JSON-fence parser revision.

## What the comparison means

The response method is adapted: humans adjusted a comparison animation; the
model writes a JSON object with direction_deg and speed_px_per_frame.
Direction is counterclockwise from right. Speed uses the original 600-pixel
clip resolution, per frame at 30 Hz. The study parser accepts only finite,
in-range numeric fields with that exact schema, as either a plain JSON object
or one JSON code fence containing the whole answer. It records the format and
parser version, keeps the raw reply, and never guesses or changes a number.
UMI records raw text first. Study-specific parsing happens during analysis.

The analysis reverses flips, then averages all four vectors per scene and
modulation. It reports angular error, absolute speed error and vector endpoint
error against each participant. Invalid flips invalidate their entire comparison
group; coverage is always reported. Zero-speed vectors have no defined direction
and contribute no angular error, with the angular comparison count reported.
These are descriptive raw errors, without a fitted mapping or ceiling. The
registered benchmark uses mean vector endpoint error as its score (lower is
better), retains direction/speed diagnostics, and requires valid answers for
every trial. It has no automatic leaderboard eligibility. One-scene pilots
cannot establish an effect across scenes.

Neural data is collected during the same task for inspection. The study's open
human reference is behavioral, so this example does not claim neural alignment.

## Checks

Run remotely with the same pinned peer repositories as the unified branch:

    SUN2025_PREPARED=/data/sun2025/prepared-pilot \
      python -m pytest tests/test_sun2025.py tests/test_task_protocol.py tests/test_import_hygiene.py -q

The real-source test checks all prepared clips, frame counts, dimensions, frame
rate, aperture background and absence of answer columns. Other checks cover
invalid reports, unflipping, identity alignment, missing trials, reference
ordering, data-plugin ownership and joint recording through participate.

The notebook and replay were also validated remotely: both a fresh four-trial
Qwen run and opening a saved recording passed in fresh notebook kernels.
Seventeen replay and notebook-validator checks passed. A Chromium check covered
frame/bin alignment, playback, seeking, raw values, narrow layout and the
notebook's sandboxed iframe. See [the receipts](validation/README.md).

To repeat notebook execution, use a new output directory:

    python examples/sun2025/validate_notebook.py \
        --data /data/sun2025/prepared-pilot \
        --saved-run /data/sun2025/runs/qwen-pilot/model \
        --output /data/sun2025/notebook-validation

For the exported viewer, install Playwright and its Chromium runtime on the
validation worker, then run:

    python examples/sun2025/validate_replay_browser.py \
        --html /path/to/exported-activity-replay.html \
        --output /data/sun2025/browser-validation

The repository-wide offline notebook validator excludes this notebook because
it needs prepared study data plus a cached checkpoint or saved recordings.
The separate validator above executes the notebook without rewriting its cells.

The current core documents response subclassing but rejects subclass names in
TaskContext. The example uses the public Text response plus a study parser;
no core patch is required. This finding is recorded in the design note.

The generic task runner also emits a plain vision channel for video rows. This
study's small ClipTaskProtocol extension declares and emits vision:clip, so a
still-only subject fails before inference. It inherits the ordinary task,
recorder and reset lifecycle. This experiment defines a native-clip condition;
a frame-based model condition would need its own explicitly identified protocol.
