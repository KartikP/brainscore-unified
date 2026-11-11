# MolmoSpaces repeatability investigation

## Finding

We isolated two upstream sources of variation consistent with our small
MolmoSpaces pilot: **GPU kernel selection changes policy outputs, and multisample
anti-aliasing changes repeated renders of a fixed scene.** Both were reproduced
without Brain-Score in the execution path. This does not establish full-rollout
UMI equivalence or reproduce a published score.

This follows the [native evaluator comparison](2026-11-01-native-evaluators.md).
Compact measurements and source provenance are recorded in
[the investigation summary](2026-11-01-molmospaces-repeatability.json).

## 1. Same observation, different native policy output

We loaded the trained `pi0_fast_droid_jointpos` checkpoint in nine fresh native
OpenPI processes. Each processed three saved observations three times: 81 calls.
We saved the preprocessed inputs, generated action tokens and decoded actions.
No Brain-Score subject, simulator or model-output cache was involved.

| Control | Result |
|---|---|
| Repeat within each process | Inputs, tokens and actions match exactly |
| Independent native processes | Inputs match; tokens and actions can differ |
| Three processes sharing reference GPU kernel choices | All three observations match exactly |
| Reload a different process's kernel choices | Its alternate tokens and actions recur exactly |

Two independently tuned processes selected different GPU implementations for
9 and 11 of the same 23 recorded operations. Timings were excluded from this
comparison. The process with 11 changed choices produced different actions for
two observations; loading those choices in a fresh process reproduced those
alternate actions. The largest observed action difference was 0.0188086584.
Different choices need not change every answer: the other recorded process
changed nine choices but matched all three reference outputs.

The FAST policy uses greedy decoding here. Random answer sampling is not needed
for this variation. Reusing XLA autotune results stabilizes these tested calls;
it does not guarantee identical behavior across different GPUs or software stacks.

## 2. Same physics state, different rendered image

We loaded saved simulator models, fixed physics states and exterior camera poses.
There was no robot motion, policy or Brain-Score call. The initial 27 renders
already varied by two channel values of one gray level. We then rendered each
setting 90 times across three contexts in each of three processes.

| Rendering setting | Kettle: distinct images / 90 | Remote: distinct images / 90 |
|---|---:|---:|
| Original multisample anti-aliasing | 4 | 4 |
| Multisample anti-aliasing disabled | 1 | 1 |
| Shadows disabled | 2 | Not tested |
| Reflections disabled | 4 | Not tested |
| Dithering disabled | 4 | Not tested |

This isolates the tested variation to the multisample rendering path. It does
not identify the precise driver or shader operation. The probe renders a saved
exterior view at 224 × 224; it does not reproduce every part of the original
camera preprocessing pipeline.

**Do not silently disable anti-aliasing for replication.** It changes 16,657 of
150,528 channel values in the kettle image and 11,423 in the remote image.
That is a different visual input, even though it is more repeatable.

## What changes for qualification

- Preserve the original evaluator's rendering settings for published-score work.
- Save and reuse GPU autotune choices for paired native/UMI checks on the same
  hardware and pinned stack. These files contain execution choices, not answers.
- Use identical saved observations to test routing, recording and interventions
  exactly. Test live simulator performance separately, accounting for native
  variation instead of treating every trajectory difference as an adapter bug.
- If a deterministic-rendering variant is needed, name and report that variant;
  do not present its score as the original benchmark result.

No UMI contracts, benchmark defaults, model weights or scores were changed.
A new full native-versus-UMI rollout and a larger score reproduction remain
separate qualification work.

## Environment and evidence

NVIDIA L4; driver 580.126.09; JAX 0.5.3; MuJoCo 3.5.0. Environments and model
checkpoint were rebuilt/downloaded on fresh scratch storage. Saved observations
and simulator models were reused. FAST's separately downloaded tokenizer revision
is `ec4d7aa71691cac0b8bed6942be45684db2110f4`.

The first policy attempt stopped before inference because upstream imports pytest
at runtime. Installing the original environment's pytest 9.2.0 resolved it; the
failed log is retained. Diagnostic scripts and 846 remote evidence files were
copied locally and SHA-256 verified in the `runner-parity-2026-11-01/investigation`
bundle. Raw observations and simulator assets are not distributed with the package.
EC2 shutdown was requested after retrieval, well before the two-hour deadline.
