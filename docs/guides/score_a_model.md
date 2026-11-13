# Score a model

## A first score, with no download

`GridGame-reach-5x5` is a closed-loop benchmark: the subject moves on a 5 x 5 grid to
reach a goal, on 15 boards that the benchmark generates from fixed seeds. It needs no
data file and no model weights, so it runs on any machine in a few seconds. It has no
measured brain or behavior data: the score is the success rate divided by that of a
greedy oracle (the ceiling). It shows the scoring path, not alignment with the brain.

A subject that moves at random is the floor of this benchmark:

```python
import brainscore
from brainscore import BrainScoreModel
from brainscore.harnesses.grid_game import random_action_policy

subject = BrainScoreModel(identifier='random-moves', policy=random_action_policy(seed=0))
score = brainscore.score(subject, 'GridGame-reach-5x5')
print(float(score), score.attrs['success_rate'], score.attrs['random_floor'])
# 0.2 0.2 0.2
```

The random subject reaches the goal on 3 of the 15 boards, the same rate as the floor
that the benchmark measures with its own random policy. A subject with no `policy` part,
such as `chance-baseline`, does not give the `motor` output, so this pair is N/A.

## A model on brain data

This example downloads the CLIP weights and the MajajHong2015 data, and a full run is
slow on a laptop:

```python
import brainscore

score = brainscore.score('clip-vit-b-32', 'MajajHong2015public.IT-pls-unified')
print(float(score), score.attrs.get('raw'))
```

`score` accepts registered names or objects you built yourself. The random policy
keeps its random state between runs, so build a new subject for a new score:

```python
benchmark = brainscore.load_benchmark('GridGame-reach-5x5')
subject = BrainScoreModel(identifier='random-moves', policy=random_action_policy(seed=0))
score = brainscore.score(subject, benchmark)
```

What happens:

1. **Compatibility check.** The subject's declared channels are compared with what the benchmark provides and requests. An incompatible pair raises `CompatibilityError` and is reported as N/A, not zero. See [Core concepts, section 6](../02_core_concepts.md#6-compatibility).
2. **Memory check.** One stimulus is run to estimate the memory the full benchmark needs. Pass `check_mem=False` to skip it.
3. **The benchmark runs** the subject through sessions and returns a `Score`. If the benchmark leaves an intervention active on the subject, `score` fails.

Every unified benchmark score carries the same attributes: `raw` (the metric value before any ceiling normalization), `ceiling` (the reference, a `Score`), `normalized` (whether the score is `raw` divided by a measured ceiling; when it is `False` the score is the raw value, and the ceiling is NaN with `attrs['status'] == 'not_estimated'`, or `Score(1.0)` for LeBel2023, the induced-dyslexia benchmarks and every Lahner2024 variant except `Lahner2024-fMRI-naturalistic-visualROI`), `metric`, `benchmark_version`, `benchmark_identifier` and `compatibility`. A raw score and a normalized score are not on one scale.

## Find models and benchmarks

```python
brainscore.list_models()         # identifier and source (unified, vision or language)
brainscore.list_benchmarks()
```

These list every identifier `load_model` and `load_benchmark` can find, across the unified, vision and language codebases. Listing downloads nothing and imports no plugin.

## Downloads and runtime

The first run downloads model weights and benchmark data. Large checkpoints stop before downloading and report their size; set `BRAINSCORE_SKIP_MODEL_DOWNLOAD_CHECK=1` to allow them. Full benchmark runs of large models need a GPU machine.

## Reading a score

- **Raw** is the metric value. **Ceiled** divides by the benchmark's ceiling and can exceed 1.
- Compare a model with a **null baseline** that fits the benchmark before interpreting it: a random policy for a closed-loop benchmark (above), `random-vit-b-32` (CLIP ViT-B/32 with random weights) for image or text benchmarks, `chance-baseline` for a task on the plain `behavior` channel. [Lesson 5](../../notebooks/lessons/05_layers_and_regions.ipynb) shows why.
- The score's `attrs` record provenance: compatibility result, behavioral path, model and benchmark identifiers, the subject's traits (`subject_traits`) and the trial memory (`trial_memory`).
