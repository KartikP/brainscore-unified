# Extending the Unified Model Interface

[Build tools and integrations](docs/guides/extend.md) | [Support matrix](docs/reference/features.md) | [Release qualification](docs/archive/production_release.md)

Extend the unified interface through five registries and Capability.

For a model registration, instantiate `BrainScoreModel`, the concrete `Subject`
subclass that composes wrappers, recording, and capability callables. For benchmark
type annotations or a custom implementation, use the abstract `Subject` contract.
Both are importable from `brainscore_core.model_interface`. A native `Subject`
declares identity and input/output channels and implements `participate(session)`;
it does not need `process()` or a layer map. `UnifiedModel` extends `Subject`
with `process`, task/recording setup, and layer/modality declarations.
`BrainScoreModel` builds on that base. See [Concepts](docs/02_core_concepts.md#1-subject).

> Companion: the interactive contract map at
> [brain-score.github.io/public/UMI/architecture.html](https://brain-score.github.io/public/UMI/architecture.html).

## Registries

Everything is a factory registered under a string identifier. All five registries live
in `brainscore/__init__.py`:

```python
data_registry:         Dict[str, Callable[[], DataAssembly]]
stimulus_set_registry: Dict[str, Callable[[], StimulusSet]]
metric_registry:       Dict[str, Callable[[], Metric]]
benchmark_registry:    Dict[str, Callable[[], Benchmark]]
model_registry:        Dict[str, Callable[[], Subject]]
```

Each `load_*` checks this package first, then falls back to the `brainscore_vision` and
`brainscore_language` registries, so legacy domain plugins remain loadable by id. Note
that those domain registries populate lazily — see
[Finding models and benchmarks](docs/guides/score_a_model.md#find-models-and-benchmarks) before concluding something is
missing because it is not in a registry dict.

A plugin is a subpackage whose `__init__.py` adds a factory to the
matching registry; the parent package imports it so registration runs on load. Load by id:

```python
import brainscore
data      = brainscore.load_dataset("your-data")
stimuli   = brainscore.load_stimulus_set("your-stimuli")
metric    = brainscore.load_metric('your-metric')
benchmark = brainscore.load_benchmark('your-benchmark')
score     = brainscore.score('your-model', 'your-benchmark')
```

`load_*` checks the unified registry first, then the `brainscore_vision` /
`brainscore_language` registries.

---

## Extension seams

The model template builds a `BrainScoreModel`; the benchmark template scores any `Subject` through typed helpers such as `neural_response`, which call `participate(session)`. For a session-native integration, implement `Subject.participate(session)` and use a benchmark or protocol that consumes that contract. A bare `Subject` is not required to implement `process()` or region mapping.

| Seam | Lives in | Implements | Contract | Template |
|------|----------|---------------|----------|----------|
| **Model** | `brainscore/models/<name>/` | `get_model() -> BrainScoreModel` | `Subject.participate(session)`; `BrainScoreModel.process(input_event)` is a typed convenience, not the contract | `templates/new_model/` |
| **Benchmark** | `brainscore/benchmarks/<name>/` | a `BenchmarkBase` subclass | `__call__(subject) -> Score` | `templates/new_benchmark/` |
| **Metric** | `brainscore/metrics/<name>/` | a `Metric` subclass | `__call__(assembly1, assembly2) -> Score` | `templates/new_metric/` |
| **Data / Stimulus set** | `brainscore/data/<name>/` | loaders registered in `data_registry` / `stimulus_set_registry` | return a `StimulusSet` and a `DataAssembly` | `templates/new_data/` |
| **Capability** | constructor slots on `BrainScoreModel`, or your own package | a function (`answer` / `policy` / `intervention`) or a registered `Capability` | see below | `templates/new_capability/` |

Reference registered data from a benchmark with `load_dataset` /
`load_stimulus_set`.

### Seam 1 — a new model

A model is one `BrainScoreModel` construction. The optional slots decide which dispatch branches it
can take (see the architecture map at https://brain-score.github.io/public/UMI/architecture.html for the full router):

```python
from brainscore import BrainScoreModel
from brainscore import model_registry

def get_model():
    return BrainScoreModel(
        identifier='your-model',
        model=backbone,                       # nn.Module, or None for output-only
        region_layer_map={'IT': 'blocks.16'}, # brain region -> layer path
        extractor={'vision': wrapper},        # wrapper owns input preparation and recording
        # optional capability slots:
        # answer=..., policy=..., intervention=...,
        # behavioral_readout_layer=..., backbone_id=...,
    )

model_registry['your-model'] = get_model
```

Then add `from . import your_name` to `brainscore/models/__init__.py`. Run `auto_register` to
infer the wrapper, layers, and a provisional `region_layer_map`, then refine with the
layer-mapping explorer.

Factories that fetch multi-gigabyte checkpoints should use the local pre-download
guard before any loader call; see [model downloads](docs/archive/model_downloads.md) and
`brainscore/models/blip2_opt_2_7b/model.py` for the pattern.

Custom extraction providers should expose output-affecting state through `cache_config()`.
Disable result caching if that state cannot be represented. Make `.data` alias or
NumPy weight edits between scoring runs, because they bypass within-run change
tracking. See [activation caching](docs/archive/caching.md) for supported paths and limits.

### Seam 2 — a new benchmark

A benchmark drives the subject model and scores it. Subclass `BenchmarkBase`, implement
`__call__(subject)`:

```python
from brainscore_core.benchmarks import BenchmarkBase, ceil_score
from brainscore_core.metrics import Score
from brainscore_core.streaming_helpers import neural_response
from brainscore import benchmark_registry, load_metric, load_stimulus_set

class YourBenchmark(BenchmarkBase):
    # Checked before any compute, so a mismatched model is N/A instead of failing mid-run.
    required_input_channels = {'vision'}
    requested_output_channels = {'neural:IT'}
    uses_session = True

    def __init__(self):
        super().__init__(identifier='your-benchmark', ceiling=Score(1.0),  # or your estimate
                         version=1, parent='neural', bibtex=BIBTEX)
        self._assembly = load_assembly()        # target measurements
        self._stimulus_set = load_stimulus_set('your-stimuli')
        self._metric = load_metric('your-metric')

    def __call__(self, subject):
        # Runs a session through subject.participate; works for any Subject.
        predictions = neural_response(
            subject=subject,
            stimulus_set=self._stimulus_set,
            record='IT',
            time_bins=[(70, 170)],
        )                                                    # -> NeuroidAssembly
        raw = self._metric(predictions, self._assembly)
        return ceil_score(raw, self.ceiling)

benchmark_registry['your-benchmark'] = lambda: YourBenchmark()
```

Then add `from . import your_name` to `brainscore/benchmarks/__init__.py`. For naturalistic /
temporal data, reuse `brainscore/temporal.py` (`temporal_bin`, `hrf_convolve`,
`contiguous_block_cv`, `window_plan`) — see `benchmarks/lahner2024` and `benchmarks/algonauts2025`
as worked examples. For a clip longer than a video model's native temporal window, set
`VideoWrapper(..., context_window_ms=...)` to tile it into windows and stitch the per-window
time-resolved features into one clip-time sequence. Set `max_clip_ms` for a fail-fast.
Before sharing a new benchmark, complete the
[benchmark addition checklist](docs/archive/benchmark_addition_checklist.md). Record the data boundary,
null floor, ceiling, modality ablations, timing assumptions, score attrs, and EC2 evidence.

### Seam 3 — a new metric

A metric compares two assemblies and returns a `Score`. Subclass `Metric`:

```python
from brainscore_core.metrics import Metric, Score
from brainscore import metric_registry

class YourMetric(Metric):
    def __call__(self, assembly1, assembly2) -> Score:
        value = compare(assembly1, assembly2)   # the alignment computation
        score = Score(value)
        score.attrs['raw'] = ...                 # optional: per-unit / per-fold detail
        return score

metric_registry['your-metric'] = lambda: YourMetric()
```

Then register it in `brainscore/metrics/__init__.py`. A reference lives at
`brainscore/metrics/topographic.py` — `load_metric('topographic-alignment')` — which scores a model's
spatial unit layout against cortical topography (the correlation-vs-distance profile). Copy
`templates/new_metric/` to start. Keep predictivity, RSA, topographic, and SCA-style metrics
separate.

### Seam 4 — configure or extend a capability

There are two related extension levels:

1. **Configure an existing capability for a model.** Pass a function into a
   `BrainScoreModel` constructor slot; the already-registered capability
   reaches it.
2. **Add a new capability.** In your own package, define typed payloads,
   register their channels with `register_channel`, subclass
   `brainscore.capabilities.Capability` and register it once with
   `register_capability`. Models opt in through `capability_config`. No core
   edit is needed.

The constructor slots available to model authors are:

| Slot | Fires on | Signature |
|------|----------|-----------|
| `answer` | `StimulusSet` + `TaskContext.instruction` | `(*, stimulus_row, instruction, label_set) -> str`, called with keywords |
| `policy` | `EnvironmentStep` / `Message` | `(env_step) -> EnvironmentResponse` |
| `intervention` | `StateChange` | `(state_change) -> (PerturbationApplied, cleanup)` |

An `answer` function receives its arguments by keyword, so use these names:

```python
def answer(*, stimulus_row, instruction, label_set):
    reply = generate(stimulus_row, instruction)   # your model
    return reply  # a reply outside label_set is recorded as invalid, not as label_set[0]
```

`templates/new_capability/` shows both levels, with a runnable registered
capability that runs through `subject.participate(session)`.

Builders: a function that builds a part is called `build_<part>` and returns
it, for example `build_policy()` returns the function passed as `policy=`. The
ready-made builders work the same way: `build_api_generation_fn(...)` returns an
`answer`, `build_api_action_fn(...)` a `policy`, and
`build_pytorch_ablation_fn(model)` an `intervention`.

The interface evaluates checkpoints; it does not train. Register the checkpoint.
For a discovery or analysis pipeline, work directly with `process()` outputs — see
`notebooks/04_multiregion_geometry.ipynb` for an end-to-end analysis that never
produces a `Score`.
