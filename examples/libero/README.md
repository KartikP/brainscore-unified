# Reproduce a trained policy on LIBERO

**Status: trained-policy smoke checks passed with cached GPU autotuning; full
Spatial evaluation is in progress.** Both routes completed 10/10 smoke trials;
221 same-input inference requests matched exactly. This is not yet a completed
full benchmark qualification.

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
from external access. The bridge binds to localhost. Run only one client per
server. The policy server holds its random state across requests.

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
- `*/videos/`: a distinct video for each completed trial.
- `*.log`, `*-freeze.txt`, `images.json`, `nvidia-smi.txt`: execution evidence.

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
`UMI_REPLAY_RECORD=/path/to/smoke-results/reference-record`. This saves inference
time without reducing benchmark trials. `replay-source.txt` and `replay.json`
identify the selected corpus and call count. Do not describe that result as a
replay of every full-benchmark request. Use the same checkpoint and saved tuning
profile that produced the selected corpus.
