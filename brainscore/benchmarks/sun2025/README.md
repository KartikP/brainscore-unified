# Sun 2025 motion-perception benchmark

Registered name: **Sun2025-motion-vector-error**, version 1.
The subject views a short clip and reports perceived motion direction and speed.
The benchmark compares those reports with the study's three human participants.
Its score is mean motion-vector error in displayed pixels per frame: **lower is
better**. Zero means identical vectors, not chance performance. There is no
ceiling normalization or automatic leaderboard eligibility.

## Read the code like a laboratory study

| Laboratory part | UMI component | Where to look |
| --- | --- | --- |
| Stimuli and human observations | Registered stimulus set and reference assembly | `brainscore/data/sun2025/` |
| Experimental procedure | `make_protocol`: instructions, clip trials, session and reset rules | `brainscore/experiments/sun2025/experiment.py` |
| Recording equipment | `RecordBehavior`, optionally `RecordNeural` | Passed visibly as `recorders` |
| Running the experiment | `Experiment(subject, protocol, tools=recorders)` | `benchmark.py`, `_run` |
| Comparing measurements | `motion_vector_error` | `metric.py`, reusing the experiment's analysis |
| Benchmark | Procedure + equipment + human reference + scoring rule | `Sun2025MotionBenchmark` |

The benchmark does not own another trial loop. It runs the existing protocol
through `Experiment`, reads the recorded answers and applies its metric.
The same `score_measurements` method scores recordings collected earlier.

## Score the complete registered study

Prepare all 40 scenes using the [data plugin](../../data/sun2025/README.md).
Source acquisition, packaging and data registration belong there. Loading this
benchmark never downloads data or model weights.

```python
import brainscore

subject = brainscore.load_model('your-registered-video-model')
benchmark = brainscore.load_benchmark(
    'Sun2025-motion-vector-error', root='/data/sun2025/prepared',
    output_dir='/data/sun2025/runs/model-benchmark')
score = brainscore.score(subject, benchmark)
print(float(score), score.attrs['units'])
print(score.attrs['summary'])
```

The subject must declare `vision:clip` input and `behavior:text` output. Neural
outputs are optional. To record activity during scoring, pass
`recorders=[RecordBehavior(), RecordNeural(['V1'])]` when loading the benchmark;
the subject must declare the requested region. Both instruments measure the
same presentations. Select a new output directory for each run. If omitted,
temporary experiment files are cleaned up; the returned Score retains the
behavioral measurements, parsed reports and participant errors.

Setting `BRAINSCORE_SUN2025_ROOT` also supports the identifier-only call:
`brainscore.score(subject, 'Sun2025-motion-vector-error')`.

## Turn an existing experiment into a benchmark

Keep the protocol, recorders and human reference you already assembled:

```python
from brainscore.benchmarks.sun2025.benchmark import Sun2025MotionBenchmark

benchmark = Sun2025MotionBenchmark(
    protocol, reference, recorders=recorders,
    output_dir='/data/sun2025/runs/new-benchmark-run')

# Compare measurements from the experiment you already ran; no new inference.
score = benchmark.score_measurements(result.behavior(measure='text'))

# To evaluate another subject, run the same procedure through normal scoring.
another_score = brainscore.score(another_subject, benchmark)
```

This constructor gives custom procedures and subsets a distinct identifier.
The registered full-study benchmark requires 40 scenes x seven motion types x
four flips, one condition, one repetition and the three matched participants.
It refuses partial prepared data before inference. Pilots remain useful through
the composition above and cannot be mistaken for the full registered study.

## Scoring and failure policy

The metric converts direction and speed into x/y vectors, reverses each spatial
flip, averages all four model vectors within a scene/motion-type group, and
computes Euclidean distance to each participant's averaged vector. The scalar
is the mean distance across all 280 groups and three participants. Separate
speed error and angular error are diagnostics in the summary. Zero-speed
vectors have no direction; angular diagnostics report their available count.

All requested trials must return valid numerical reports. Missing or invalid
reports produce `NaN`, `status='failed'`, a reason and coverage information.
Available comparisons can still be inspected, but are not the benchmark's
headline score. Raw answers and parsing errors remain accessible. Incompatible
subjects fail pre-flight with `CompatibilityError`; orchestration can use
`score_or_na` to represent that as N/A. No failure becomes zero error.

## Interpretation and validation

The rendered clips and numerical reports adapt the human motion-matching task;
see the data and experiment guides for the exact transformations. This metric
is the integration's explicit descriptive comparison, not a reproduction of
the paper's reported model statistic. There is no human neural reference.
Optional activity recordings and their replay support inspection only.

`tests/test_sun2025_benchmark.py` checks registered scoring on a complete
synthetic design, the no-motion control, agreement with direct experiment
analysis, valid/invalid reports, identity alignment, resets, optional neural
recording and the previously collected 28-trial model pilot. Synthetic tests
validate the implementation; they are not a full-study scientific result.
Run all checks remotely. The real-source test in `tests/test_sun2025.py` is
marked slow and requires `SUN2025_PREPARED`; offline CI selects the remaining
unit and integration checks.
