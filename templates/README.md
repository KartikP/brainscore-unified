# Extension templates

Starting points for all five extension seams: model, benchmark, metric, data and
capability. Each folder is a runnable skeleton with `TODO` markers, a `README.md`, and
tests that define the contract. `../docs/concepts.md` covers the vocabulary and
`../EXTENDING.md` the contracts.

| Template | Seam | Implements | Loadable via |
|----------|------|---------------|--------------|
| `new_model/` | Model | `get_model() -> BrainScoreModel` | `load_model('id')` |
| `new_benchmark/` | Benchmark | a `BenchmarkBase` subclass | `load_benchmark('id')` |
| `new_metric/` | Metric | a `Metric` subclass | `load_metric('id')` |
| `new_data/` | Data / Stimulus set | loaders for stimuli and measurements | `load_stimulus_set('id')` / `load_dataset('id')` |
| `new_capability/` | Capability | a `generation_fn`/`action_fn`/`state_change_fn` closure | (wired into a model) |

## Workflow

1. Copy the relevant folder into `brainscore/{models,benchmarks,metrics,data}/<your_name>/`
   (capabilities have no registry — wire the closure into a model instead).
2. Fill the `TODO`s.
3. Register it (`__init__.py`) and add `from . import <your_name>` to the parent package's `__init__.py`.
4. Adapt and run the template's test.

The core package is deliberately small: it ships the extension seams and a few reference
integrations, and the catalog grows through contributed plugins. `../EXTENDING.md`
documents the contract for each seam.
