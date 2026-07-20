# Extending the Unified Model Interface

Extend the unified interface through five registries and Capability.

> Companion: the interactive contract map at
> [`website/architecture.html`](website/architecture.html).

## Registries

Everything is a factory registered under a string identifier. Five registries —
`model`/`benchmark`/`metric` live in `brainscore/__init__.py`; `data`/`stimulus_set`
live in the domain repos (`brainscore_vision`/`_language`):

```python
data_registry:         Dict[str, Callable[[], DataAssembly]]   # in brainscore_vision/_language
stimulus_set_registry: Dict[str, Callable[[], StimulusSet]]    # in brainscore_vision/_language
metric_registry:       Dict[str, Callable[[], Metric]]
benchmark_registry:    Dict[str, Callable[[], Benchmark]]
model_registry:        Dict[str, Callable[[], Subject]]
```

A plugin is a subpackage whose `__init__.py` adds a factory to the
matching registry; the parent package imports it so registration runs on load. Load by id:

```python
import brainscore
data      = brainscore.load_dataset('your-data')          # via vision/language registry
stimuli   = brainscore.load_stimulus_set('your-stimuli')  #   "
metric    = brainscore.load_metric('your-metric')
benchmark = brainscore.load_benchmark('your-benchmark')
score     = brainscore.score('your-model', 'your-benchmark')
```

`load_*` checks the unified registry first, then the `brainscore_vision` /
`brainscore_language` registries.

---

## Extension seams

| Seam | Lives in | You implement | Contract | Template |
|------|----------|---------------|----------|----------|
| **Model** | `brainscore/models/<name>/` | `get_model() -> BrainScoreModel` | `process(input_event) -> OutputEvent` | `templates/new_model/` |
| **Benchmark** | `brainscore/benchmarks/<name>/` | a `BenchmarkBase` subclass | `__call__(candidate) -> Score` | `templates/new_benchmark/` |
| **Metric** | `brainscore/metrics/<name>/` | a `Metric` subclass | `__call__(assembly1, assembly2) -> Score` | `templates/new_metric/` |
| **Data / Stimulus set** | `brainscore_vision/_language` `data/<name>/` | a loader registered in `data_registry` / `stimulus_set_registry` | returns a `DataAssembly` / `StimulusSet` | (domain-repo pattern) |
| **Capability** | constructor slots on `BrainScoreModel` | a callable (`generation_fn` / `action_fn` / `state_change_fn`) | see below | `templates/new_capability/` |

Reference registered data from a benchmark with `load_dataset` /
`load_stimulus_set`.

### Seam 1 — a new model

A model is one `BrainScoreModel` construction. The optional slots decide which dispatch branches it
can take (see `website/architecture.html` for the full router):

```python
from brainscore_core.model_interface import BrainScoreModel
from brainscore import model_registry

def get_model():
    return BrainScoreModel(
        identifier='your-model',
        model=backbone,                       # nn.Module, or None for output-only
        region_layer_map={'IT': 'blocks.16'}, # brain region -> layer path
        preprocessors={'vision': preprocess},  # modality -> callable (defines supported_modalities)
        activations_model=wrapper,            # PytorchWrapper / Text / VLMVision / Video / Audio
        # optional capability slots:
        # generation_fn=..., action_fn=..., state_change_fn=...,
        # behavioral_readout_layer=..., region_modality_map=..., backbone_id=...,
    )

model_registry['your-model'] = get_model
```

Then add `from . import your_name` to `brainscore/models/__init__.py`. Run `auto_register` to
infer the wrapper, layers, and a provisional `region_layer_map`, then refine with the
layer-mapping explorer.

### Seam 2 — a new benchmark

A benchmark drives the candidate model and scores it. Subclass `BenchmarkBase`, implement
`__call__(candidate)`:

```python
from brainscore_core.benchmarks import BenchmarkBase
from brainscore_core.metrics import Score
from brainscore import benchmark_registry, load_metric

class YourBenchmark(BenchmarkBase):
    def __init__(self):
        super().__init__(identifier='your-benchmark', ceiling=ceiling_score,
                         version=1, parent='neural', bibtex=BIBTEX)
        self._assembly = load_assembly()        # target measurements
        self._metric = load_metric('your-metric')

    def __call__(self, candidate):
        candidate.start_recording('IT', time_bins=[(70, 170)])
        predictions = candidate.process(self._stimulus_set)   # -> NeuroidAssembly
        raw = self._metric(predictions, self._assembly)
        return ceil_score(raw, self.ceiling)

benchmark_registry['your-benchmark'] = lambda: YourBenchmark()
```

Then add `from . import your_name` to `brainscore/benchmarks/__init__.py`. For naturalistic /
temporal data, reuse `core/brainscore_core/temporal.py` (`temporal_bin`, `hrf_convolve`,
`contiguous_block_cv`, `window_plan`) — see `benchmarks/lahner2024` and `benchmarks/algonauts2025`
as worked examples. For a clip longer than a video model's native temporal window, set
`VideoWrapper(..., context_window_ms=...)` to tile it into windows and stitch the per-window
time-resolved features into one clip-time sequence. Set `max_clip_ms` for a fail-fast.
Before sharing a new benchmark, complete the
[benchmark addition checklist](docs/benchmark_addition_checklist.md). Record the data boundary,
null floor, ceiling, modality ablations, timing assumptions, score attrs, and EC2 evidence.

### Seam 3 — a new metric

A metric compares two assemblies and returns a `Score`. Subclass `Metric`:

```python
from brainscore_core.metrics import Metric, Score
from brainscore import metric_registry

class YourMetric(Metric):
    def __call__(self, assembly1, assembly2) -> Score:
        value = compare(assembly1, assembly2)   # your alignment computation
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

1. **Configure an existing capability for a model.** Pass a callable into a
   `BrainScoreModel` constructor slot;
   `process()` reaches it through the already-registered framework capability.
2. **Add a new framework dispatch capability.** Subclass
   `brainscore_core.capabilities.Capability` and register that class with
   `register_capability` for a new UMI dispatch path.

The constructor-callable slots available to model authors are:

| Slot | Fires on | Signature |
|------|----------|-----------|
| `generation_fn` | `StimulusSet` + `TaskContext.instruction` | `(stimulus_row, instruction, label_set) -> str` |
| `action_fn` | `EnvironmentStep` / `Message` | `(env_step) -> EnvironmentResponse` |
| `state_change_fn` | `StateChange` | `(state_change) -> (PerturbationApplied, cleanup)` |

A new input/output event or dispatch behavior belongs at the second level. See
`core/brainscore_core/capabilities/` for the framework registry and
`templates/new_capability/` for model-level callable examples.

The interface evaluates checkpoints; it does not train. Register the checkpoint.
For a discovery or analysis pipeline, use `process()` outputs and see
`scripts/yeatman_sweep/`.
