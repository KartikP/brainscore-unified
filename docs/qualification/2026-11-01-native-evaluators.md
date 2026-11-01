# Native evaluator comparison — November 1, 2026

## Outcome

CogGym passed a matched text experiment. MolmoSpaces passed fixed-input routing
checks but did not pass exact whole-rollout equivalence. Neither test reproduces
a published leaderboard score.

## CogGym

Qwen3.5-4B answered Levine2020Logic/exp3 through direct CogGym, the UMI runner,
and the registered Brain-Score benchmark. All 16 requests and trial results
matched across routes (48 fresh calls). Raw R² was 0.006981193559190357 in each.
The low score is not evidence of strong model performance; equality is the
engineering result. The registered benchmark delegates to the authors' evaluator.

Settings: one repetition, 512 tokens, temperature 1, thinking disabled, matched
per-request sampling seeds and one loaded model. This does not qualify the
five-repetition default profile or image/video experiments.

## MolmoSpaces

The trained pi0_fast_droid_jointpos policy controlled a Franka arm on two original
MS-Pick episodes: pick up a kettle and a remote. Upstream owned observations,
action execution and scoring; episodes had a 500-step limit.

| Execution | Successes | Calls |
|---|---:|---:|
| Native, separate server | 1/2 | 99 |
| UMI, separate server | 0/2 | 126 |
| UMI with recording, separate server | 0/2 | 126 |

Three saved observations replayed twice natively, once through UMI, and once
through UMI with recording produced identical actions on one loaded server.
However, two native executions on separate servers produced different first
actions for identical inputs (maximum difference 0.012180331778673281).
This is evidence of native repeatability variation, not proof of its cause.

A second control shared one policy server across three one-episode kettle runs.
All failed the task (0/1; 63 calls each), but their trajectories differed.
Saved simulator model binaries, camera poses and initial physics states matched.
Between native and UMI, post-action physics and executed actions matched for
indices 0–15; both first differed at index 16. Camera images had already differed
at call 0 by one channel value of one gray level. Thus matching final success
counts does not establish identical execution.

The pinned FAST implementation uses greedy decoding by default. Its policy removes
an explicit temperature override before calling sample_actions. The cause of
cross-process action differences and fixed-state image differences is unresolved;
no compiler, renderer or adapter defect has yet been established.

## Evidence and reproducibility

The experiment used fresh isolated Python environments, reused immutable
simulation assets where available, and downloaded the text checkpoint anew.
Nine recorded-state clips passed 1920 × 1080, checksum and decoded-frame checks;
27 screenshots and complete calls/states were saved. These clips are visual
replays, not new policy evaluations. Internal neural activity was not recorded.

Source pins and compact machine-readable comparisons accompany this document in
`2026-11-01-native-evaluators.json`. Full evidence is retained outside the package
in the `runner-parity-2026-11-01` qualification bundle; videos, model observations
and simulator assets are not committed or redistributed here.

## Follow-up

1. Compare preprocessing, generated tokens and actions across native policy
   processes with independent and shared compilation caches.
2. Render a fixed saved state repeatedly within and across renderer contexts.
3. Establish native repeatability before repeating the matched UMI rollout or
   expanding to published-score replication.

New diagnostic probes are prepared but have not yet been GPU-qualified.

## Follow-up completed

The [native repeatability investigation](2026-11-01-molmospaces-repeatability.md)
reproduced kernel-selection and multisample-rendering variation independently
of UMI. It documents the tested controls and their limits; full-rollout
equivalence and published-score replication are still not established.
