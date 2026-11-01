# LIBERO recording and randomness

## What was tested

A trained `pi05_libero` policy controlled a simulated robot taking a black bowl
from a drawer and placing it on a plate. LIBERO judged success; MuJoCo simulated
the task. Brain-Score recorded component 0 of `action_out_proj` at denoising
iterations 0 and 1.

An earlier ten-task run scored 10/10 without recording and 9/10 with recording.
These controls tested whether recording caused the failed drawer task.

| Starting policy random state | Recording off | Recording on |
|---|---|---|
| After 87 prior calls | Success; 23 calls, 115 steps | Same |
| After 88 prior calls | Failure; 44 calls, 220 steps | Same |

All inputs, actions and saved simulator states matched exactly within each
pair: 67 paired calls, 134 total. The original successful and failed episodes
were also reproduced exactly. Four targeted saved-input/random-state probes,
each evaluated off/on/on/off, matched all 16 archived action chunks exactly.
The recording calls captured the requested activity.

## What this explains

The earlier runs first differed at call 25: two color values of one wrist-camera
pixel changed by 1 out of 255. That episode needed one extra policy call.
OpenPI advances its random sequence on each call and the evaluator does not
reset it between episodes. The later drawer task therefore began at different
positions in that sequence despite identical initial observations.

The controlled reruns show that the drawer-task outcome followed the random
state, not recording. This is evidence for the tested policy and recording
selection, not a guarantee for all models or equal long-run success rates.

The first camera difference remains unexplained. Four fixed-action replays,
with and without added delays, produced identical states and images, all
matching the original recording image. Repeated rendering of a fixed state was
also identical. Original full simulator states and raw camera frames were not
saved at the divergence point.

## Reproduction identity

- Core: `83f7539bc06ceb436a071cef5de4d2420aced3ea`
- Unified: `e6b9d1200aab69c52ee2d538c6ca636a2f707701`
- OpenPI: `215abfb217dbac7d5f1273282331b9b1866c0479`
- LIBERO: `f78abd68ee283de9f9be3c8f7e2a9ad60246e95c`
- Checkpoint tree SHA-256: `a6423f21c698393718047e96c05e0d13026d002f6788a6f05c22f750597083c1`
- Fresh EC2 L4 environments: Python 3.11.9 / JAX 0.5.3 / Flax 0.10.2 for
  inference; Python 3.8.20 / MuJoCo 3.2.3 for simulation. Original GPU autotuning
  choices were reused. No output tolerance was introduced.

## Qualification change

`examples/libero/run_tools.sh` accepts `UMI_PAIRED_SEED`; the bridge accepts
`--paired-seed`. These opt-in controls supply reproducible sampling noise for
each task/trial/call through the existing OpenPI connection. Extra calls in one
episode cannot shift another episode's noise. Reports record the profile and
per-call noise checksums; comparisons reject different profiles.

Default evaluation retains upstream randomness. The paired profile uses NumPy
PCG64 rather than JAX's default noise generator and must not be described as
upstream benchmark replication. It leaves model weights, observation processing,
action scheduling and success scoring unchanged.

**Validation boundary:** the GPU evidence above used matched JAX random states.
It did not run the new explicit-noise launcher. That launcher has offline and
transport coverage; its trained-policy GPU/simulator qualification is pending.
See the [LIBERO guide](../../examples/libero/README.md#pair-randomness-when-comparing-tools)
for usage and interpretation.
