# Template: a new benchmark

This template evaluates a subject through task/recording helpers and returns a ceiling-adjusted `Score`. Benchmarks without a reliability ceiling can return raw scores instead; state the score definition explicitly.
See `EXTENDING.md` (Seam 2) for the contract.

## Use it

1. Copy this folder to `brainscore/benchmarks/<your_name>/`.
2. Edit `benchmark.py`. Implement `_load_assembly()` (your measurements + stimulus_set),
   `_ceiling()`, and `__call__(subject)`. Pick your metric via `load_metric(...)`.
3. Pick your identifier in `benchmark_registry['<your-id>']`.
4. Add `from . import <your_name>` to `brainscore/benchmarks/__init__.py`.
5. Adapt `test_benchmark.py` and run it.

## Variants

A variant of a benchmark is one more registry line: `benchmark_registry['<id>-<suffix>'] = partial(YourBenchmark, <arguments>)` (or `lambda: YourBenchmark(<arguments>)`). Put the arguments that make the variant in that line, for example `channels=('text',)` for a variant that sends text. Do not write a subclass or a factory function that only passes arguments. Write a subclass only when it adds behavior. The identifier of a variant stays fixed once registered.

```python
import brainscore
score = brainscore.score('<some-model>', '<your-id>')
```

## The interface contract

This template records responses with `neural_response(subject, stimuli, ['neural:IT'], time_bins=...)`, which runs a session through `subject.participate(session)`. Any `Subject` can be scored: a `BrainScoreModel`, a legacy adapter, or a subject that implements only `participate`. For behavioral tasks use `behavioral_response(subject, stimuli, task)`. Do not call `start_recording`, `start_task` or `process` directly; a session-native `Subject` does not promise them.

The benchmark declares its input channels (`required_input_channels`), its requested output channels (`requested_output_channels`) and its `trial_memory` (`'none'` or `'history'`). `score()` refuses a benchmark that declares no inputs.

Before opening a PR, complete the
[benchmark addition checklist](../../docs/archive/benchmark_addition_checklist.md) and attach the evidence
for data-plugin loading, null floor, ceiling, modality ablations, timing assumptions, score attrs,
and any EC2-only verification.

## Worked references (read these)

- `brainscore/benchmarks/roar_yeatman2021/`: behavioral (generation + readout paths).
- `brainscore/benchmarks/induced_dyslexia/`: intervention (`apply_state_change` + random null).
- `brainscore/benchmarks/lahner2024/`: naturalistic fMRI encoding (temporal kit, ROI variants).
- `brainscore/benchmarks/algonauts2025/`: multimodal movie fMRI (video+audio+text, banded ridge).
