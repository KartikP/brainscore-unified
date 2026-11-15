# Template: a new benchmark

This template evaluates a subject through task/recording helpers and returns a `Score`: the raw value divided by a measured ceiling, or the raw value when the ceiling is `not_estimated()` (as shipped). Never use `Score(1.0)` to mean "no ceiling".
See `EXTENDING.md` (Seam 2) for the contract.

## Use it

1. Copy this folder to `brainscore/benchmarks/<your_name>/`.
2. Edit `benchmark.py`. Implement `_load_assembly()` (your measurements + stimulus_set),
   `_ceiling()`, and `__call__(subject)`. Pick your metric via `load_metric(...)`.
3. Pick your identifier in `benchmark_registry['<your-id>']`.
4. Nothing else to edit: the folder registers by its presence.
5. Move `test_benchmark.py` to `tests/test_<your_name>.py`, adapt it and run it.

## Variants

A variant of a benchmark is one more registry line: `benchmark_registry['<id>-<suffix>'] = partial(YourBenchmark, <arguments>)` (or `lambda: YourBenchmark(<arguments>)`). Put the arguments that make the variant in that line, for example `channels=('text',)` for a variant that sends text. The registry line names the identifier too: a class with variants takes it as an argument, `identifier='<id>-<suffix>'`, as ROAR, induced dyslexia and Algonauts2025 do, and does not build it from a table of suffixes. Do not write a subclass or a factory function that only passes arguments. Write a subclass only when it adds behavior. The identifier of a variant stays fixed once registered.

From the `templates/` folder, the template benchmark scores the template model
offline:

```python
import brainscore
import new_model, new_benchmark      # registers 'your-model' and 'your-benchmark'

score = brainscore.score('your-model', 'your-benchmark')
print(score.attrs['metric'], score.attrs['normalized'])   # linear_predictivity False (no ceiling estimated yet)
```

## The interface contract

This template records responses with `neural_response(subject, stimuli, ['neural:IT'], time_bins=...)`, which runs a session through `subject.participate(session)`. Any `Subject` can be scored: a `BrainScoreModel`, a legacy adapter, or a subject that implements only `participate`. For behavioral tasks use `behavioral_response(subject, stimuli, task)`. Do not call `start_recording`, `start_task` or `process` directly; a session-native `Subject` does not promise them.

The benchmark keeps the table it presents as `stimulus_set` (and a behavioral task as `task`); pre-flight reads its input channels from them. It declares its requested output channels (`requested_output_channels`) and its `trial_memory` (`'none'` or `'history'`). A benchmark whose inputs do not come from one kept table declares them in `required_input_channels`. `score()` refuses a benchmark with neither.

Before opening a PR, complete the
[benchmark addition checklist](../../docs/archive/benchmark_addition_checklist.md) and attach the evidence
for data-plugin loading, null floor, ceiling, modality ablations, timing assumptions, score attrs,
and any EC2-only verification.

## Worked references (read these)

- `brainscore/benchmarks/roar_yeatman2021/`: behavioral (generation + readout paths).
- `brainscore/benchmarks/induced_dyslexia/`: intervention (`apply_state_change` + random null).
- `brainscore/benchmarks/lahner2024/`: naturalistic fMRI encoding (temporal kit, ROI variants).
- `brainscore/benchmarks/algonauts2025/`: multimodal movie fMRI (video+audio+text, banded ridge).
