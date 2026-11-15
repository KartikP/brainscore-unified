# Extension templates

Starting points for the extension seams: model, benchmark, metric, data, tool and
capability. Each folder is a runnable skeleton with `TODO` markers, a `README.md`, and
tests that define the contract. `../docs/02_core_concepts.md` covers the vocabulary and
`../EXTENDING.md` the contracts.

| Template | Seam | Implements | Loadable via |
|----------|------|---------------|--------------|
| `new_model/` | Model | `get_model() -> BrainScoreModel` | `load_model('id')` |
| `new_benchmark/` | Benchmark | a `BenchmarkBase` subclass | `load_benchmark('id')` |
| `new_metric/` | Metric | a `Metric` subclass | `load_metric('id')` |
| `new_data/` | Data / Stimulus set | loaders for stimuli and measurements | `load_stimulus_set('id')` / `load_dataset('id')` |
| `new_tool/` | Tool | a `Tool` subclass (one module) | `load_tool('id')` |
| `new_capability/` | Capability | an `answer`/`policy`/`intervention` function, or a registered `Capability` | `capability_config` on a model |

## Workflow

1. Copy the relevant folder into `brainscore/{models,benchmarks,metrics,data}/<your_name>/`
   (a tool is one module, `brainscore/tools/<your_name>.py`; a capability lives in your own
   package; call its `register()` once).
2. Fill the `TODO`s.
3. Register it (`__init__.py`). The folder registers by its presence: every subpackage of
   `brainscore/{models,benchmarks,data,metrics}/` is imported with `brainscore`, so no other
   file changes. A plugin in another package (another repository) registers when you import that package before you load it. A tool in Brain-Score is one line in `_TOOLS` of
   `brainscore/tools/_registry.py`; a tool in your own package registers with
   `register_tool(identifier, factory)` when you import that package.
4. Move the template's test to `tests/test_<your_name>.py`, adapt it and run it. Every
   plugin's test lives in `tests/`; the template keeps its test beside it only so the
   template can be tried in place.

The core package is deliberately small: it ships the extension seams and a few reference
integrations, and the catalog grows through contributed plugins. `../EXTENDING.md`
documents the contract for each seam.

Naming: a function that builds a part is called `build_<part>` and returns it, for
example `build_policy()` returns the function passed as `policy=`. Constructor slots
are `extractor`, `answer`, `policy` and `intervention` (and `reply` and `likelihood`
for answers). Three ready-made builders still carry the names of removed parts:
`api_answer(...)` returns an `answer`, `api_policy(...)` a
`policy`, and `pytorch_intervention(model)` an `intervention`.
