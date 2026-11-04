# Score a model

```python
import brainscore

score = brainscore.score('clip-vit-b-32', 'MajajHong2015public.IT-pls-unified')
print(float(score), score.attrs.get('raw'))
```

`score` accepts registered names or objects you built yourself:

```python
score = brainscore.score(my_model, my_benchmark)
```

What happens:

1. **Compatibility check.** The subject's declared channels are compared with what the benchmark provides and requests. An incompatible pair raises `CompatibilityError` and is reported as N/A, not zero. See [Core concepts, section 6](../02_core_concepts.md#6-compatibility).
2. **Memory check.** One stimulus is run to estimate the memory the full benchmark needs. Pass `check_mem=False` to skip it.
3. **The benchmark runs** the subject through sessions and returns a `Score`.

## Find models and benchmarks

```python
sorted(brainscore.model_registry)
sorted(brainscore.benchmark_registry)
```

These list entries registered in the unified package. `load_model` and `load_benchmark` also find classic vision and language entries, such as `'alexnet'` or `'MajajHong2015.IT-pls'`, which are not in these lists.

## Downloads and runtime

The first run downloads model weights and benchmark data. Large checkpoints stop before downloading and report their size; set `BRAINSCORE_SKIP_MODEL_DOWNLOAD_CHECK=1` to allow them. Full benchmark runs of large models need a GPU machine.

## Reading a score

- **Raw** is the metric value. **Ceiled** divides by the benchmark's ceiling and can exceed 1.
- Compare a model with a **null baseline** (chance, or the same architecture with random weights) before interpreting it. [Lesson 5](../../notebooks/lessons/05_layers_and_regions.ipynb) shows why.
- The score's `attrs` record provenance: compatibility result, behavioral path, model and benchmark identifiers.
