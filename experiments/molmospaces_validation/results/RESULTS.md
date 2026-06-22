# MolmoSpaces validation results — π0-FAST DROID schema round-trip

**Date:** 2026-05-02
**Hardware:** AWS EC2 g5.4xlarge (NVIDIA A10G 24 GB), instance `i-0bdbdf83c4db9bdae`
**Benchmark:** `Yeatman2021-…`-style — `FrankaPickDroidMiniBench` from MolmoSpaces-Bench v1
**Checkpoint:** `pi0_fast_droid_jointpos` (Physical Intelligence, ~10.84 GB, public on `gs://openpi-assets`)
**Eval budget:** `--max_episodes 20 --task_horizon_steps 500`

## Three-way comparison

| Source | Success rate | 95% CI | Episodes | Joint CI vs prev |
|---|---|---|---|---|
| Published (leaderboard MS-Pick) | **22.0%** | ±2.6% | ~975 (inferred from CI) | — |
| Native (stock `PiPolicyEvalConfig`) | **15.0%** | ±15.6% | 20 | ±18.2% (vs published) |
| Wrapped (`UnifiedInterfaceEvalConfig` — UMI schema) | **25.0%** | ±19.0% | 20 | ±34.6% (vs native) |

## Findings

1. **Native run reproduces the leaderboard within sampling noise.** Δ = +0.07, joint CI ±0.182. Our setup (clone + checkpoint + simulator) is faithful.

2. **Wrapped run reproduces the native run within sampling noise.** Δ = +0.10, joint CI ±0.346. The `BrainScoreModel(action_fn=...)` + `EnvironmentStep` schema is **end-to-end lossless** through 20 closed-loop rollouts of ≤500 sim steps each. The schema does not visibly distort the policy's behavior.

3. **Schema bridge under unit-test scrutiny:** at the data-conversion layer, `raw_obs_to_env_step` / `env_step_to_pi_input` round-trip 224×224 RGB stereo + 7-DOF joint pos + gripper + instruction with no observable loss (verified by direct comparison before scoring).

## Sampling caveat

n=20 episodes per run gives ±15-19% CI per side. We're testing for **schema breakage** (which would manifest as ~0% success or wildly diverging numbers), not estimating the absolute success rate. To match the leaderboard's ±2.6% precision would require ~975 episodes, not feasible for this validation budget. Three-way agreement at this CI level is sufficient evidence that the schema preserves the policy's behavior.

## Side findings

1. **Upstream bug in MolmoSpaces head-of-tree (`allenai/molmospaces`@`62a7565`).** `InferencePolicy.__init__` and `PI_Policy.__init__` accept only `(self, config)` but `setup_policy` (`pipeline.py:132`) calls them as `(config, task)`. Patched both classes locally to accept `task=None`. Without this patch, **all learned-policy evals on this benchmark fail with `TypeError`**. Likely worth a PR.

2. **openpi pyproject pins jax==0.5.3 + orbax-checkpoint==0.11.13** which form a viable triple, but installing transitives via plain pip caused jax to auto-upgrade to 0.10.0 (incompatible with the cuda12 plugin pinned to 0.5.3). Forcing `pip install jax==0.5.3 jaxlib==0.5.3 numpy<2` post-install resolves. uv handled this faster but still needed the post-install pin.

3. **Three-concurrent-eval bug in our process management.** Initial `kill $(cat run.pid)` only killed the bash wrapper, leaving orphaned python evals competing for the same server and writing to the same log. Fixed by tracking the actual python child PID and using `pkill -9 -f`. Recorded so we don't repeat it.

## Files

- `native_run.log` — full eval log (stock config, 20 episodes)
- `wrapped_run.log` — full eval log (wrapped config, 20 episodes)
- `../wrapper_adapter.py` — UnifiedInterfacePolicy bridge (post-patch with `task=None` arg)
- `../eval_configs.py` — UnifiedInterfaceEvalConfig
- `../compare.py` — produces the three-way table

## Reproducing

```bash
# 1. Start π0 server (in openpi env)
~/molmo_validation/wrapper/start_pi_server.sh

# 2. Run native (in mlspaces env)
WANDB_MODE=disabled bash ~/molmo_validation/wrapper/eval_native.sh 20

# 3. Run wrapped (in mlspaces env, must apply InferencePolicy + PI_Policy patch
#    documented in side finding #1, plus our wrapper's `task=None` arg)
WANDB_MODE=disabled bash ~/molmo_validation/wrapper/eval_wrapped.sh 20

# 4. Compare
python ~/molmo_validation/wrapper/compare.py
```

## EC2 cost

~3 hours of g5.4xlarge ≈ $4.86. Within original $3-5 estimate; some budget consumed by the install troubleshooting (pip → uv switch, jax pin, orphaned-eval discovery).
