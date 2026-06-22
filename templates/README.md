# Extension templates

Copy-paste starting points for the four *code* seams (Model / Benchmark / Metric / Capability). Data and
stimulus sets are also seams but use the domain-repo `data_registry` / `stimulus_set_registry` pattern
(no template here) — see `../EXTENDING.md`. Each folder is a skeleton with `TODO` markers + a `README.md`
+ tests-as-spec. Read `../EXTENDING.md` first for the contracts, and `../website/architecture.html` for
the interactive contract/dispatch map.

| Template | Seam | You implement | Loadable via |
|----------|------|---------------|--------------|
| `new_model/` | Model | `get_model() -> BrainScoreModel` | `load_model('id')` |
| `new_benchmark/` | Benchmark | a `BenchmarkBase` subclass | `load_benchmark('id')` |
| `new_metric/` | Metric | a `Metric` subclass | `load_metric('id')` |
| `new_capability/` | Capability | a `generation_fn`/`action_fn`/`state_change_fn` closure | (wired into a model) |

## Workflow

1. Copy the relevant folder into `brainscore/{models,benchmarks,metrics}/<your_name>/`
   (capabilities have no registry — wire the closure into a model instead).
2. Fill the `TODO`s.
3. Register it (`__init__.py`) and add `from . import <your_name>` to the parent package's `__init__.py`.
4. Adapt and run the template's test.

The backbone stays small on purpose: it ships the seams + a few reference integrations, and the
catalog grows from the community. See `../EXTENDING.md` § "What we deliberately leave to you".
