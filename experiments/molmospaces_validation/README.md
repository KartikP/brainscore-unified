# MolmoSpaces validation — π0-FAST schema round-trip

Reproduces a published MolmoSpaces-Bench number (π0-FAST DROID, MS-Pick, **22.0% ± 2.6%**) two ways:

1. **Native** — using molmospaces' stock `PiPolicyEvalConfig`. Anchors against the public leaderboard.
2. **Wrapped** — same checkpoint, but routed through `BrainScoreModel(action_fn=...)` consuming our `EnvironmentStep` schema. Tests that the unified-interface schema is end-to-end lossless.

If both match the public 22.0% within the ±2.6% CI, the schema is validated against an externally-published embodied benchmark.

## Reference

- Leaderboard: https://molmospaces.allen.ai/leaderboard?bench=pick (MS-Pick task, π0-FAST DROID row, 22.0% ± 2.6%)
- Source: https://github.com/allenai/molmospaces (Apache 2.0)
- Checkpoint: `gs://openpi-assets/checkpoints/pi0_fast_droid_jointpos`

## Files

| File | Purpose |
|---|---|
| `wrapper_adapter.py` | `UnifiedInterfacePolicy(InferencePolicy)` — bridges molmospaces' policy ABC to our `BrainScoreModel(action_fn=...)`. Lives on EC2; copied here for review. |
| `eval_native.sh` | Documented native eval command (anchor against public number). |
| `eval_wrapped.sh` | Wrapped eval command — same checkpoint, schema in the loop. |
| `compare.py` | Prints the three-way comparison: published / native / wrapped. |
