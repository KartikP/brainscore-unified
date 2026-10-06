# Streaming figures

These figures come from the executed session examples in notebook 15. Timing uses a simulated clock; it is not a hardware speed measurement.

- `input_delivery.png`: calls per input, partial windows, and bounded input buffering.
- `clock_policies.png`: delivered and skipped windows under each clock policy. Numbers inside bars identify windows.
- `measurements.json`: timing outcomes and a hash of the notebook code used to generate them.

From the repository root, in the notebook environment:

```bash
python notebooks/figure_sources/render_streaming_figures.py
```

The script executes the notebook's code before drawing. Regenerate after changing the example's settings.
