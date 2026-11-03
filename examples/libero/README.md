# Reproduce a trained policy on LIBERO

**Transport results:** both routes passed 10/10 smoke trials and 221 saved requests matched exactly. The full Spatial run later completed: reference 496/500 successes, UMI 488/500. Those closed-loop outcomes were not equivalent.

**Historical internal-tools check (October 6, 2026):** a separate ten-task check passed after a GPU recording fix. Baseline, recording and ablation each succeeded on 10/10 tasks; baseline and recording matched all 223 observations/action chunks exactly. See the [qualification report](../../docs/qualification/2026-10-06-openpi-tools.md).

For the design and steps to reuse in another benchmark, read
[Connect a robotics benchmark](../../docs/robotics_benchmark_integration.md).

Use the public `pi05_libero` checkpoint with OpenPI's LIBERO evaluator. The
reference evaluator keeps control of tasks, initial states, image preprocessing,
action scheduling, and success scoring. A separate Python 3.11 bridge passes
its prepared requests through `BrainScoreModel` and records requests/results.
LIBERO keeps its upstream Python 3.8 environment; OpenPI inference has its own
environment. No core contract or scientific scoring dependencies change.

## What the three runs establish

1. **Reference:** official evaluator, recording proxy, original trained policy.
2. **Replay:** identical saved requests, in order, through UMI and a fresh policy
   server. Every predicted action must match exactly (`atol=rtol=0`). A mismatch
   stops the script for investigation; do not widen tolerance after seeing it.
3. **UMI:** official evaluator controls the simulator through the UMI bridge.
   Compare per-trial outcomes and success rates against the reference.

The reference proxy records calls but does not dispatch them through a subject.
Both routes use the same network and recording boundary. Runtime overhead is
therefore not a comparison against an uninstrumented server.

OpenPI's websocket client `reset()` does not reset server RNG state. Restart the
policy server before each run. Replaying saved outputs needs no inference;
`replay.py` instead reruns inference to test equivalence on identical inputs.
This first integration covers the selected policy, not arbitrary recurrent
policies or robot hardware. The bridge returns a complete action chunk; the
reference evaluator chooses and applies actions. It does not exercise UMI's
`EnvironmentSession` driver or internal model interventions.

## Pinned sources

| Component | Revision / identifier |
| --- | --- |
| OpenPI | `215abfb217dbac7d5f1273282331b9b1866c0479` |
| LIBERO submodule | `f78abd68ee283de9f9be3c8f7e2a9ad60246e95c` |
| Policy config | `pi05_libero` |
| Checkpoint | `gs://openpi-assets/checkpoints/pi05_libero` |
| UMI peers | `install/peer-revisions.json` |

Hash downloaded checkpoint and normalization files before interpreting results.
The checkpoint URL alone is not immutable provenance. Preserve the prepared
source bundle's manifest, image IDs, dependency freezes, and GPU/driver details.

## Prepare on Linux with NVIDIA Docker support

Stage the four repositories under one directory (`core`, `vision`, `language`,
`unified`), including this uncommitted candidate. Keep OpenPI separately.
Run these commands from the directory containing the four repositories:

```bash
git clone https://github.com/Physical-Intelligence/openpi.git openpi
git -C openpi checkout 215abfb217dbac7d5f1273282331b9b1866c0479
git -C openpi submodule update --init third_party/libero
docker build -t umi-libero-policy -f openpi/scripts/docker/serve_policy.Dockerfile openpi
python3 unified/examples/libero/prepare_runtime.py --openpi openpi --out runtime-base.Dockerfile
docker build -t umi-libero-runtime-base -f runtime-base.Dockerfile openpi
docker build -t umi-libero-runtime -f unified/examples/libero/runtime.Dockerfile .
docker build -t umi-libero-bridge -f unified/examples/libero/bridge.Dockerfile .
bash unified/examples/libero/run_pair.sh "$PWD/openpi" "$PWD/smoke-results" libero_spatial 1
```

Use a dedicated host with ports 8000 and 8001 free. Keep the policy port blocked
from external access. The bridge binds to localhost. Each `serve_bridge.py`
process serves one evaluator connection and refuses any later one (close code
1013); concurrent clients are not supported. Run one client per policy server:
it holds its random state across requests.

One trial per task is a smoke test, not a benchmark reproduction. After reviewing
it, repeat with a new output directory and `50` trials per task: 500 episodes per
route for the ten-task Spatial suite. Use `libero_object`, `libero_goal`, and
`libero_10` for later suites. Do not present a single suite as the four-suite mean.

Set an independent host shutdown deadline before starting paid compute. The
script does not start, stop, or authorize AWS resources. Build/download time
counts against the run budget. Preserve outputs on persistent storage.

`prepare_runtime.py` repairs the upstream container's package installation:
Python 3.8 source builds need compatible setuptools, and the LIBERO requirements
include packages whose transitive dependencies must be resolved. The generated
Dockerfile preserves explicitly pinned runtime versions. It leaves the upstream
checkout and evaluation code unchanged; save the resulting dependency freeze.

## Review the evidence

- `reference/report.json` and `umi/report.json`: complete episode count, errors,
  settings, successes, and success rate. All attempted trials remain counted.
- `*/trials.jsonl`: task, trial index, initial-state hash, outcome, and errors.
- `replay.json`: same-input action agreement; any mismatched call fails.
- `comparison.json`: paired outcome differences and descriptive success intervals.
- `*-checkpoint.json`: content hashes, required to agree across the three runs.
- `*-record/`: inputs and full action chunks in checksum-verified UMI records.
  These are transport captures (see below), not evidence that an evaluation finished.
- `*/videos/`: a distinct video for each completed trial.
- `*.log`, `*-freeze.txt`, `images.json`, `nvidia-smi.txt`: execution evidence.

**Transport captures.** A `serve_bridge.py` record closes `complete` only if its
one evaluator connection was closed by the evaluator before the bridge stopped,
and every request has its recorded reply. A stop during a call, a stop while the
evaluator is still connected, no connection at all, or a policy failure closes it
`failed`; a failed call is never retried. An evaluator that crashes still closes
its connection, so `complete` means "capture complete", not "evaluation complete".
`replay.py`, `toolbox_calls.py`, `qualify_tools.py` and `qualify_remote_tools.py`
therefore take `--report`, the evaluator output directory of the same run, and
refuse a record unless that report is complete, matches the record's route, uses
the pinned revisions and settings, covers every declared episode in order with a
success or step-limit ending (the tool bridge's receipt rules), and accounts for
exactly the record's calls, which must alternate request and reply. Records from
the bridge before this check are refused. No run identifier links the two on this
route, so keep each run's evaluator output and record together.

The evaluator writes a failed report if upstream logs an error, even though the
original evaluation loop catches inference exceptions. Interrupted runs and
missing trials must not be reported as completed evaluations. Collect confidence
intervals and paired outcome differences before comparing scientific results.

The authors report 98.8% Spatial success for this checkpoint. Treat that as a
reference to reproduce, not a guaranteed local result. See the
[official protocol and results](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/examples/libero/README.md).
This work supplements the required trained DROID check; it does not replace it.

## Diagnosing variation across server restarts

The first GPU smoke run completed 10/10 reference trials, but fresh-server action
replay was not exact. A reference-only repeat also differed. These initial runs
remain failed exact-reproducibility checks, not accepted numerical tolerances.

Set `UMI_XLA_AUTOTUNE_CACHE=1` to save the reference server's GPU autotuning
results and require subsequent servers to reuse them. Flags and the tuning file
are retained with the evidence. This is a separate execution profile that must
pass its own reference, replay, and UMI runs before being called qualified.
The exact-action gate remains unchanged. See
[OpenXLA's determinism guidance](https://openxla.org/xla/determinism).

For a larger run using the already tested compiler choices, also set
`UMI_XLA_AUTOTUNE_SOURCE=/path/to/smoke-results/autotune/results.textproto`.
The file is copied into the new evidence directory and all three servers load
it with complete-cache enforcement. Keep it with the GPU/runtime provenance;
it is not a portable cache for arbitrary devices or software versions.

To repeat the smaller smoke corpus alongside a full benchmark run, set
`UMI_REPLAY_RECORD=/path/to/smoke-results/reference-record` together with
`UMI_REPLAY_REPORT=/path/to/smoke-results/reference`, that run's evaluator output.
The script refuses the override without it. This saves inference
time without reducing benchmark trials. `replay-source.txt` and `replay.json`
identify the selected corpus and call count. Do not describe that result as a
replay of every full-benchmark request. Use the same checkpoint and saved tuning
profile that produced the selected corpus.

## Internal policy tools

`serve_tools.py` adds the common `RecordActivity` and `Ablate` tools to the pinned
OpenPI JAX sampler. `qualify_tools.py` compares baseline, recorded, ablated and
restored inference on saved requests. See [policy instrumentation](../../docs/policy_instrumentation.md)
for the target names, client example and cleanup rules.

The providers have 36 CPU checks plus trained NVIDIA L4 same-input and remote checks. Selected-unit recording at all three sites preserved actions exactly on ten inputs each. The separate ten-task simulator check uses shared GPU autotuning results; it is not full benchmark reproduction.

To run the optional tests, add these dependencies to an isolated Python 3.11
UMI test environment. `OPENPI_SOURCE` must point to the pinned checkout above:

```bash
python -m pip install -r examples/libero/requirements-tools-cpu.txt
OPENPI_SOURCE=/path/to/openpi python -m pytest \
  tests/test_openpi_instrumentation.py \
  tests/test_openpi_remote.py -q
```

The CPU requirements cover the test fixture. Loading a full trained checkpoint
also requires OpenPI's complete runtime (including its LeRobot dependencies),
checkpoint/normalization files, and suitable compute. The server imports those
through OpenPI's own policy factory.

### Use the official evaluator with tools

Who owns what on each route:

| Route | Loop, resets, scheduling, scoring | Inference and its random state | Recording and internal tools |
|---|---|---|---|
| `run_pair.sh` (`serve_bridge.py`) | LIBERO evaluator (`evaluate.py`) | upstream policy server | bridge records requests and chunks only; no internal tools |
| `run_tools.sh` (`serve_tool_bridge.py`) | LIBERO evaluator (`evaluate.py`) | `serve_tools.py` | bridge runs `Experiment` with `RecordInputsOutputs`, `RecordActivity`, `Ablate`; tools act inside each inference request |
| `qualify_tools.py` / `qualify_remote_tools.py` | the script (saved requests, no simulator) | local or `serve_tools.py` policy | same tools, on identical inputs |

**Closed-loop inputs are not matched after an intervention.** On the tool route the
simulator responds to the actions the policy returns. Once an ablation changes an
action, every later observation differs from the baseline run, so per-call
comparison of internals or actions across conditions is not meaningful there.
`compare.py` then reports a descriptive difference in task outcomes, which is not a
parity check. Use `qualify_tools.py` or `qualify_remote_tools.py` for same-input
comparisons. They gate on exact action parity of the recording and restored runs,
on zeroed selected units that were nonzero before ablation, and on finite
outputs; whether the ablation changed actions is reported under `effects`, not
required. A recording-only closed-loop run sees the baseline inputs only while its
actions stay identical.

`run_tools.sh` drives this route from pre-staged inputs. It downloads nothing and
starts no cloud resources:

```bash
UMI_POLICY_PYTHON=/path/to/openpi-runtime/bin/python \
UMI_BRIDGE_PYTHON=/path/to/umi-env/bin/python \
UMI_EVALUATOR_PYTHON=/path/to/libero-runtime/bin/python \
bash unified/examples/libero/run_tools.sh /path/to/openpi /path/to/local/pi05_libero \
  /path/to/new-evidence libero_spatial 1 7200 recording ablation
```

The sixth argument bounds each evaluator run in seconds; the run is stopped and the
script fails when it is exceeded. `UMI_READY_SECONDS` (default 1200) bounds server
start-up. For each condition (baseline first) it starts a fresh `serve_tools.py`
and bridge, runs `evaluate.py`, then stops both; it compares each tool condition
with the baseline using `compare.py`. On exit or interruption it stops only the
processes it started. The evidence directory must not exist. Interpreter versions,
dependency freezes and revisions are saved alongside the logs. The checkpoint
directory must already contain the weights and normalization files.

The manual steps it performs:

Start `serve_tools.py --port 8000 --steps 0 1` (its default port, 8001, is the bridge's). In the UMI/OpenPI client environment, run:

```bash
python examples/libero/serve_tool_bridge.py \
  --upstream ws://127.0.0.1:8000 \
  --mode recording \
  --suite libero_spatial --trials-per-task 50 --execution-horizon 5 \
  --out /path/to/new/experiment-records
```

Add `--suite`, `--trials-per-task` and `--execution-horizon 5` (the evaluator's `replan_steps`); the bridge refuses an evaluator whose start message disagrees. Run the evaluator only through `evaluate.py` on port 8001, from a unified checkout: it loads the standard-library `brainscore/experiments/backends/socket_deadline.py` by path, without installing brainscore (`run_pair.sh` mounts the checkout read-only). The bridge uses the same `LiberoChunkPolicy` and common experiment tools; LIBERO still owns physics, observations, scheduling and success scoring. `--mode baseline` omits internal tools; `--mode ablation` additionally silences projection unit 0. Restart the policy server for each matched condition.

**Completion.** On one connection the bridge sends a hello naming its UMI run, the evaluator declares its suite, tasks and step limits, and each request carries its call index and episode. After writing `report.json`, the evaluator sends a receipt: per episode, chunks received, actions whose `env.step` returned, wait steps, errors and how it ended. The bridge accepts it only if every episode in the declared order ended by success or the step limit, its calls match the calls served, and executed actions fill whole chunks except possibly the last. A receipt is evaluator testimony, checked for consistency; it is not independent proof. A served chunk is not evidence that its actions ran.

The bridge acknowledges `complete` only after `Experiment.run` has saved its records; the evaluator writes the answer to `finish.json` and exits 0 only for a complete report and acknowledgment. A disconnect, idle timeout (`--idle-timeout`, 900 s), rejected receipt or failed save marks the bridge record failed. A lost or late inference reply closes the evaluator connection; nothing is resent, and the remaining episodes fail.

**Transport limits.** Operational defaults, not measured budgets.

| Connection | Limit | Default |
|---|---|---|
| Evaluator → bridge | `--connect-timeout`: connecting | 120 s |
| | `--infer-timeout`, `--control-timeout`, `--finish-timeout`: one whole exchange (send plus reply) | 600 / 60 / 600 s |
| Bridge → evaluator | `--idle-timeout`: each wait for, and each send to, the evaluator | 900 s |
| Bridge → policy (`OpenPIPolicyClient`) | connecting, and each whole request | 300 s |
| Policy server (`serve_tools.py`) | each reply send | 300 s |
| Any close | separate cleanup allowance, added after the limit above | about 10 s |

- When a limit runs out, the connection is terminated and nothing is resent. A stalled close cannot keep the bridge locked.
- These limits cover transport only. They do not cancel an inference already running on the policy server, and a request that timed out may still have been processed there. Nothing bounds model inference, record or artifact writing, or a whole experiment.
- `evaluate.py` and the helper it loads are checked for Python 3.8 syntax only; they have not been run under Python 3.8. To run the evaluator subprocess tests with another interpreter, set `UMI_EVALUATOR_PYTHON` (default: the test interpreter). It receives only the pinned `openpi-client` on `PYTHONPATH` and needs numpy, pytest, websockets, msgpack and Pillow.

Qualify a tool-bridge run in the UMI environment, passing each tool run's experiment directory:

```bash
python examples/libero/compare.py --reference baseline-eval --umi recording-eval \
  --reference-experiment baseline-records/connection-1 \
  --umi-experiment recording-records/connection-1 --out comparison.json
```

This verifies the manifests, artifact hashes, call record, receipt and report digest, and joins every recorded call to its episode. If the bridge record is complete but the acknowledgment was lost, the run is refused unless `--reconcile-unacknowledged` is given; the result is then labelled `tool-bridge-reconciled-unacknowledged`. Reports from `serve_bridge.py` are labelled `transport`.

`compare.py` reads only the evaluator reports for `transport` sides; it does not open `serve_bridge.py` captures.

For matched GPU processes, preserve baseline compilation choices with `XLA_FLAGS=--xla_gpu_dump_autotune_results_to=/path/to/autotune.textproto`. Subsequent conditions load that file with `--xla_gpu_load_autotune_results_from=/path/to/autotune.textproto --xla_gpu_require_complete_aot_autotune_results=true`. Set these before starting the policy process and retain the file with the run evidence.

`qualify_remote_tools.py` takes the same `--checkpoint`, `--record`, `--report` and `--out` paths as the local qualification script, and tests direct-versus-remote inference before simulator use. Both scripts record the checkpoint content hash and the source evidence, write `qualification.json` with `status: failed` and the error when a check fails, and use explicit checks that still run under `python -O`.

The `run_tools.sh` baseline is the UMI tool route **without tools**, not the native
reference from `run_pair.sh`. Its intervention targets unit 0 of `action_out_proj`
at denoising iterations 0 and 1 of the default ten iterations. Closed-loop outcome
differences are descriptive; use the same-input checks to establish action parity.
