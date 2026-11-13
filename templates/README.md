# Extension templates

Starting points for all five extension seams: model, benchmark, metric, data and
capability. Each folder is a runnable skeleton with `TODO` markers, a `README.md`, and
tests that define the contract. `../docs/02_core_concepts.md` covers the vocabulary and
`../EXTENDING.md` the contracts.

| Template | Seam | Implements | Loadable via |
|----------|------|---------------|--------------|
| `new_model/` | Model | `get_model() -> BrainScoreModel` | `load_model('id')` |
| `new_benchmark/` | Benchmark | a `BenchmarkBase` subclass | `load_benchmark('id')` |
| `new_metric/` | Metric | a `Metric` subclass | `load_metric('id')` |
| `new_data/` | Data / Stimulus set | loaders for stimuli and measurements | `load_stimulus_set('id')` / `load_dataset('id')` |
| `new_capability/` | Capability | an `answer`/`policy`/`intervention` function, or a registered `Capability` | `capability_config` on a model |

## Workflow

1. Copy the relevant folder into `brainscore/{models,benchmarks,metrics,data}/<your_name>/`
   (a capability lives in your own package; call its `register()` once).
2. Fill the `TODO`s.
3. Register it (`__init__.py`) and add `from . import <your_name>` to the parent package's `__init__.py`.
4. Adapt and run the template's test.

The core package is deliberately small: it ships the extension seams and a few reference
integrations, and the catalog grows through contributed plugins. `../EXTENDING.md`
documents the contract for each seam.

Naming: a function that builds a part is called `build_<part>` and returns it, for
example `build_policy()` returns the function passed as `policy=`. Constructor slots
are `extractor`, `answer`, `policy` and `intervention` (and `reply` and `likelihood`
for answers). Three ready-made builders still carry the names of removed parts:
`build_api_generation_fn(...)` returns an `answer`, `build_api_action_fn(...)` a
`policy`, and `build_pytorch_ablation_fn(model)` an `intervention`.
