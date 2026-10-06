# OpenPI tool qualification — October 6, 2026

**Passed for the tested trained-policy configuration after a recording-selection fix.** This is a ten-task simulator smoke check, not full LIBERO reproduction or physical-robot qualification.

## Configuration

| Item | Tested value |
| --- | --- |
| Policy | Trained `pi05_libero` |
| OpenPI | `215abfb217dbac7d5f1273282331b9b1866c0479` |
| LIBERO | `f78abd68ee283de9f9be3c8f7e2a9ad60246e95c` |
| GPU | NVIDIA L4, 23 GB reported memory |
| Policy runtime | Python 3.11.9, JAX 0.5.3, Flax 0.10.2; original OpenPI mixed precision |
| Checkpoint/normalization tree digest | `a6423f21c698393718047e96c05e0d13026d002f6788a6f05c22f750597083c1` |
| Selection | Unit 0; denoising iterations 0 and 1 |
| Simulator | LIBERO Spatial, one fixed initial-state trial per task, ten tasks |
| Compiler control | Fresh policy processes; simulator routes share baseline GPU autotuning results |

The policy server retains OpenPI's Transformers 4.53.2, outside the general UMI scoring profile. Qualification covers this isolated policy-tool runtime; it does not qualify legacy vision/language scoring in that environment. Keep the normal UMI scoring environment separate.

## Results

| Check | Result |
| --- | --- |
| Local trained inference: baseline, recording, ablation, restored | Ten inputs per route; recorded/restored actions match baseline exactly; selected activity zeroed; ablation changes actions |
| Remote trained inference | Ten inputs per route; recorded/restored actions match direct inference exactly within the controlled run |
| Selected-unit recording at each exposed site | Ten inputs per site; all three sites preserve actions exactly |
| LIBERO baseline | 10/10 successes |
| LIBERO recording-only | 10/10 successes; all 223 calls have identical observations and action chunks to baseline |
| LIBERO ablation | 10/10 successes; success alone does not measure the intervention's effect |
| CPU OpenPI tests after the fix | 36 passed |
| Unified offline regressions | 843 passed, 23 skipped, 64 excluded by the slow/private-access filter |

After the first compiled call, median policy inference time was 187 ms for baseline, 214 ms with recording, and 216 ms with ablation plus before/after recording. These measurements cover this configuration, not real-time robot control.

The three tested recording sites are `embed_suffix`, `PaliGemma.llm.suffix`, and `action_out_proj`. The intervention checks and simulator ablation use `action_out_proj`. Other selections, policies, GPUs and task suites need their own qualification.

## Failure found and corrected

The first trained run failed recording-only equality: the largest action difference was approximately `0.002197` in controller coordinates. Selecting one recorded unit inside the JAX graph changed the result. Direct inference, plain recompilation, a no-op probe and whole-projection recording served as controls.

Recording now transfers the site output and selects units on the host. The original exact-equality checks then passed; tolerances were not widened. This transfers the whole selected site's output even when only a few units are saved.

Independent default GPU processes also differed numerically (up to `0.003051` in a saved cross-check). Fresh simulator processes therefore reused the baseline's GPU autotuning results. Equality claims above apply to their documented runtime controls, not arbitrary GPU/compiler settings.

## Reproduce and inspect

- [Local same-input qualification](../../examples/libero/qualify_tools.py)
- [Remote same-input qualification](../../examples/libero/qualify_remote_tools.py)
- [Evaluator-to-tool-server bridge](../../examples/libero/serve_tool_bridge.py)
- [Tool setup and supported sites](../policy_instrumentation.md)
- [Notebook 19](../../notebooks/19_openpi_experiment_toolbox.ipynb): saved trained measurements, exact checks and an optional live call
- [Bundled records and provenance](../../notebooks/assets/openpi/README.md)

The notebook includes complete ten-input records for baseline, recording, ablation and restoration, plus simulator comparison summaries. The full internal evidence archive also contains per-trial results, videos, call/activity records, runtime manifests, driver sources and the initial failed run. All 4,280 remote evidence/log files were copied and hash-verified before requesting shutdown.

## Limits

Ten trials do not establish a population success rate or full benchmark parity. A targeted ablation can alter predictions while all ten tasks still succeed. No physical robot ran. There is no claim of general JAX support or general UMI scoring compatibility in the OpenPI server environment.

Earlier transport-only qualification is separate: its full Spatial comparison was 496/500 successes for reference and 488/500 for UMI. Its 221-call smoke replay matched exactly, but those full closed-loop outcomes were not equivalent. Do not use that result to inflate the scope of this new instrumentation smoke check.
