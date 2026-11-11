# Template: a new benchmark

This template evaluates a subject through task/recording helpers and returns a ceiling-adjusted `Score`. Benchmarks without a reliability ceiling can return raw scores instead; state the score definition explicitly.
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

This template records responses with `neural_response(subject=..., stimulus_set=..., record='IT', time_bins=...)`, which runs a session through `subject.participate(session)`. Any `Subject` can be scored: a `BrainScoreModel`, a legacy adapter, or a subject that implements only `participate`. For behavioral tasks use `behavioral_response(subject, task_context)`. Do not call `start_recording`, `start_task` or `process` directly; a session-native `Subject` does not promise them.
Before opening a PR, complete the
[benchmark addition checklist](../../docs/archive/benchmark_addition_checklist.md) and attach the evidence
for data-plugin loading, null floor, ceiling, modality ablations, timing assumptions, score attrs,
and any EC2-only verification.

## Worked references (read these)

- `brainscore/benchmarks/roar_yeatman2021/` — behavioral (generation + readout paths).
- `brainscore/benchmarks/induced_dyslexia/` — perturbation (`apply_state_change` + random null).
- `brainscore/benchmarks/lahner2024/` — naturalistic fMRI encoding (temporal kit, ROI variants).
- `brainscore/benchmarks/algonauts2025/` — multimodal movie fMRI (video+audio+text, banded ridge).
