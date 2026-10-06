# Trained OpenPI measurements

These records come from the trained `pi05_libero` policy on an NVIDIA L4. Inputs were saved from a LIBERO reference evaluation. The four `records/` runs repeat the same ten observations with matched starting random state; they do not themselves rerun the simulator.

| Provenance | Value |
| --- | --- |
| Source qualification | `openpi-gpu-qualification-2026-10-06/evidence/same-input-host-selection` |
| OpenPI | `215abfb217dbac7d5f1273282331b9b1866c0479` |
| Runtime | Python 3.11.9, JAX 0.5.3, Flax 0.10.2, NVIDIA L4 |
| Checkpoint/normalization tree digest | `a6423f21c698393718047e96c05e0d13026d002f6788a6f05c22f750597083c1` |
| Intervention | Unit 0 of `action_out_proj`, denoising iterations 0 and 1 |
| Procedure | `examples/libero/qualify_tools.py::qualify` |

- `records/`: complete baseline, recording, ablation and restored experiment records, with input/output/activity arrays and checksums.
- `qualification.json`: exact-equality and intervention checks.
- `recording-sites.json`: additional ten-input checks for each of the three recording sites.
- `recording-comparison.json`, `ablation-comparison.json`: separate ten-task LIBERO Spatial smoke comparisons. The `reference` column means baseline; the `umi` column means the named tool condition.
- `recording-trace-comparison.json`: all 223 closed-loop baseline/recording calls have identical observations and actions.
- `activity.png`, `actions.png`: measured first-call values from the bundled records. The camera panel is the actual first input image. Action values are LIBERO controller commands, not joint angles or measured movements.

Rerender without GPU inference:

```bash
python notebooks/figure_sources/render_openpi_figures.py
```

To rerun inference, use the full pinned OpenPI runtime and `examples/libero/qualify_tools.py` with the trained checkpoint and original raw call record. Always write to a new directory. The general UMI notebook environment is sufficient for reading these records, but not for loading the policy.

[Qualification scope, discovered failure and correction](../../../docs/qualification/2026-10-06-openpi-tools.md).
