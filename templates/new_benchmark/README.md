# Template: a new benchmark

A benchmark drives a candidate model through the unified interface and returns a ceiled `Score`.
See `EXTENDING.md` (Seam 2) for the contract.

## Use it

1. Copy this folder to `brainscore/benchmarks/<your_name>/`.
2. Edit `benchmark.py` — implement `_load_assembly()` (your measurements + stimulus_set),
   `_ceiling()`, and `__call__(candidate)`. Pick your metric via `load_metric(...)`.
3. Pick your identifier in `benchmark_registry['<your-id>']`.
4. Add `from . import <your_name>` to `brainscore/benchmarks/__init__.py`.
5. Adapt `test_benchmark.py` and run it.

```python
import brainscore
score = brainscore.score('<some-model>', '<your-id>')
```

## The interface contract

Touch the candidate ONLY through `start_recording` / `start_task` / `process` — never
`look_at`/`digest_text`. That's what lets any compliant model run on your benchmark unchanged.

## Worked references (read these)

- `brainscore/benchmarks/roar_yeatman2021/` — behavioral (generation + readout paths).
- `brainscore/benchmarks/induced_dyslexia/` — perturbation (`process(StateChange)` + random null).
- `brainscore/benchmarks/lahner2024/` — naturalistic fMRI encoding (temporal kit, ROI variants).
- `brainscore/benchmarks/algonauts2025/` — multimodal movie fMRI (video+audio+text, banded ridge).
