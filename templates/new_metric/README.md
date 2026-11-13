# Template: a new metric

A metric is a new alignment axis. It compares two assemblies and returns a `Score`.
This is the seam the field is moving along (beyond linear predictivity → representational,
topographic, behavioral, causal). See `EXTENDING.md` (Seam 3) for the full contract.

## Use it

1. Copy this folder to `brainscore/metrics/<your_name>/`.
2. Edit `metric.py`. Implement `__call__(assembly1, assembly2) -> Score`. Document the coords
   each assembly must carry, and fail loudly on a mismatch.
3. Edit `__init__.py`. Pick your identifier in `metric_registry['<your-id>']`.
4. Add `from . import <your_name>` to `brainscore/metrics/__init__.py`.
5. Adapt `test_metric.py` and run it (`pytest brainscore/metrics/<your_name>/test_metric.py`).

From the `templates/` folder, with the data template for an assembly:

```python
import brainscore
import new_data, new_metric          # registers 'your-measurements' and 'your-metric'

measured = brainscore.load_dataset('your-measurements')
metric = brainscore.load_metric('your-metric')
print(round(float(metric(measured, measured)), 6))   # identical assemblies: 1.0
```

In a benchmark, the first argument is the model's response and the second the measured data.

## Worked reference

`brainscore/metrics/topographic.py` (`load_metric('topographic-alignment')`) is a complete,
non-trivial example: it scores a model's spatial unit layout against cortical topography via the
correlation-vs-distance profile. Read it alongside this template.

## Keep axes separate

Don't overload one metric to mean several things. Predictivity, RSA, topographic alignment, and
SCA-style representational alignment are *distinct* metrics that answer distinct questions; register
them separately.
