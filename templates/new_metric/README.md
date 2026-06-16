# Template: a new metric

A metric is a new **alignment axis** — it compares two assemblies and returns a `Score`.
This is the seam the field is moving along (beyond linear predictivity → representational,
topographic, behavioral, causal). See `EXTENDING.md` (Seam 3) for the full contract.

## Use it

1. Copy this folder to `brainscore/metrics/<your_name>/`.
2. Edit `metric.py` — implement `__call__(assembly1, assembly2) -> Score`. Document the coords
   each assembly must carry, and fail loudly on a mismatch.
3. Edit `__init__.py` — pick your identifier in `metric_registry['<your-id>']`.
4. Add `from . import <your_name>` to `brainscore/metrics/__init__.py`.
5. Adapt `test_metric.py` and run it (`pytest brainscore/metrics/<your_name>/test_metric.py`).

```python
import brainscore
metric = brainscore.load_metric('<your-id>')
score  = metric(model_assembly, brain_assembly)
```

## Worked reference

`brainscore/metrics/topographic.py` (`load_metric('topographic-alignment')`) is a complete,
non-trivial example: it scores a model's spatial unit layout against cortical topography via the
correlation-vs-distance profile. Read it alongside this template.

## Keep axes separate

Don't overload one metric to mean several things. Predictivity, RSA, topographic alignment, and
SCA-style representational alignment are *distinct* metrics that answer distinct questions; register
them separately.
