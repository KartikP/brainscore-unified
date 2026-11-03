# MIRAGE UMI integration qualification

Date: 2026-11-02. Scope: preserve the released model's predictions while using
UMI sessions, experiment recording and input replay. No human-data score was
computed.

## Revisions and dependency

| Component | Tested revision |
|---|---|
| unified, PR #43 | `abf74bd55a48bd6585d08ccd3320dd30a1e9ceca` |
| core | `15dcc3c6cbfea16565052abd2663b8de4bf1b203` |
| vision | `eca24202bb7b188c1115d3b140b96abc0fa848c7` |
| language | `62e2dcfe17daa81ecdf939f6e075d9edd33389ca` |
| upstream MIRAGE source | `c594e3124e2a9cb7b8a26a750b373e583a7fb297` |
| HF MIRAGE encoder | `b38a252c28e5416336423f4eda0c42d7e1df4e9c` |
| Qwen3-Omni-30B-A3B-Thinking | `2f443cfc4c54b14a815c0e2bb9a9d6cbcd9a748b` |

Peer pins were verified against `install/peer-revisions.json` before work.
The GPU qualification used the then-unmerged facade candidate above. PR #43
has since merged. The demo is restacked onto integration branch revision
`529237a` with merged peer pins: core
`0fbb419ff0ba461e6b1e217c845e0fc61d8096a1`, vision
`1cdda82f38ae49b331fcc4b7257263aca802ea43`, language
`a9e572d0a93945f01ed49255a4a8ffd6b37ad2a7`.
The GPU results retain their original revision scope; they were not rerun after
restacking.

The encoder is 4,492,439,044 bytes, verified on EC2 with SHA-256
`8ee898b4d83466a72192577a492e89bececdd563d0b2fd82f1679dfd1dbce69d`.
The published config leaves backbone revisions unset. Asset preparation pins
all three modalities to the same explicit Qwen revision, without other config
changes. Original config SHA-256:
`fb306fb902a186ca25ed9c8c5e2d2cbf293a3d9b073373e4efbb226da945c461`;
resolved config SHA-256:
`5e34c4508125ca455dd1b33c2fc498b2b1505732e0f96cf30e08a4ea31b8732c`.
Both direct and UMI inference use the resolved configuration.

## Implementation and contract

`brainscore/models/mirage/` contains an explicitly configured subject and a
Python subprocess runtime. No core, legacy model, benchmark, channel catalog
or dependency baseline was changed. The model is not automatically registered;
its runtime paths and trained subject head must be supplied explicitly.

- Inputs declare `vision`, `audio`, `text`; all three events are required per
  clip. Vision and audio reference the same container, as the native API requires.
  Text is the joined transcript, with validated word/onset/duration metadata;
  an empty transcript is explicit. Unconsumed channels are ignored.
- `participate(session)` responds after each complete clip. Inference uses the
  unmodified upstream `InferenceRequest` and `run_inference` API.
- `neural:Schaefer1000` carries a `NeuroidAssembly` of predicted BOLD, with
  native TR and parcel ordering, clip identity, trained subject identity and
  provenance. These are predictions of BOLD, not extracted neural activations.
- Native TR is 1.49 s. Custom time bins are rejected before inference. Native
  149 s windows, -4.47 s window shift and duration trimming are preserved.
  No extra HRF, normalization or fitted readout is applied.
- The worker has a finite timeout, explicit errors and temporary-file cleanup.
  Timeout or interruption kills the process group, including decoder children.
  Each clip uses a fresh process; no GPU state survives a call, and reset is
  stateless. Model startup cost is paid for every clip. Native window batching
  is configurable; this qualification used batch size 1.
- Existing `Experiment`, `RecordInputsOutputs`, `replay_sessions` and
  `compare_outputs` provide recording and comparison. Replay reruns native
  inference from saved inputs; it does not serve saved responses. File staging
  restores the media suffix lost by content-addressed recording filenames.

## Environments

Upstream requires Python `>=3.12,<3.13` and torch `>=2.7,<2.8`. The dedicated
EC2 worker used Ubuntu 24.04, one RTX PRO 6000 Blackwell Server Edition GPU
(96 GB), driver 595.112.07, CUDA 12.8, Python 3.12.3, torch 2.7.1+cu128,
transformers 4.57.6, NumPy 1.26.4 and x-transformers 2.11.15. The separate CPU
UMI host used Python 3.11.14, NumPy 1.26.4 and xarray 2022.3.0. Full native
and host package freezes are in the evidence archive. Upstream source was
imported by `PYTHONPATH`; training dependencies were not installed.

No quantization, sharding or source patch was used. Native CUDA defaults were
shared: TF32 matmul false, cuDNN TF32 true; backbone config dtype bfloat16,
final outputs float32. A BF16 CUDA smoke check passed before model execution.
The worker checks source revision, tracked source changes and weight checksum.
It records full config, packages, input hashes and integration source hashes.

## Reproduction commands

Export the exact pinned core and this unified tree under `/opt/mirage-run`,
then use the example preparation scripts on a new dedicated GPU worker:

```sh
bash examples/mirage_umi/setup_ec2.sh
env HF_HOME=/opt/mirage-run/cache/huggingface \
  /opt/mirage-run/native/bin/python examples/mirage_umi/download_assets.py \
  --asset-dir /opt/mirage-run/assets
bash examples/mirage_umi/prepare_clip.sh
/opt/mirage-run/host/bin/python examples/mirage_umi/prepare_synthetic_clip.py \
  --out /opt/mirage-run/stimuli/synthetic
```

The pinned backbone must already be in the cache before offline qualification.

Actual first-case command, from the exported unified directory:

```sh
env PYTHONPATH=/opt/mirage-run/core:/opt/mirage-run/unified \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 RESULTCACHING_DISABLE=1 \
  AWS_EC2_METADATA_DISABLED=true PYTHONDONTWRITEBYTECODE=1 \
  /opt/mirage-run/host/bin/python examples/mirage_umi/run.py \
  --mirage-python /opt/mirage-run/native/bin/python \
  --source-dir /opt/mirage-run/mirage \
  --run-dir /opt/mirage-run/assets/encoder \
  --work-dir /opt/mirage-run/cache \
  --video /opt/mirage-run/stimuli/clip.mp4 \
  --transcript /opt/mirage-run/stimuli/transcript.json \
  --out /opt/mirage-run/results/qualification-01 \
  --device cuda \
  --subject sub-01
```

The second command substitutes `stimuli/synthetic/clip.mp4`,
`stimuli/synthetic/transcript.json`, `results/qualification-02` and `sub-02`.
The native comparison invokes `python -m brain_enc.cli.infer_fmri` independently
with the same paths, subject index, device and batch size 1.

## Results

| Case | Head | Shape and dtype | Native = UMI | Original = replay |
|---|---|---|---|---|
| Big Buck Bunny public sample, 4 s A/V, no transcript | sub-01 | (3, 1000), float32 | exact | exact |
| Synthetic green square video and synthesized speech, timed words | sub-02 | (3, 1000), float32 | exact | exact |

There were six actual released-model inference executions: direct, recorded
UMI and replay for each case. Arrays, dtype, parcel order and TR order were
checked without a tolerance. Replay succeeded after deleting the example's
private original video copy. GPU memory returned to zero after inference.
Different inputs and heads prevent attributing differences to the head alone.

[The compact qualification record](../examples/mirage_umi/qualification.json)
contains array/input hashes, source hashes and execution settings. Full arrays,
recordings, both configs, stimuli, logs and freezes are archived outside Git.

### Tests

- New adapter/runtime tests: **23 passed**, locally and on the EC2 Python 3.11
  host. Includes malformed inputs, ordering, reactive sessions, repeat/reset,
  wrong head, invalid native outputs, recording after original input removal,
  worker failure, missing artifacts, timeout and decoder-child cleanup.
- Relevant pinned-core tests: **71 passed** (`test_native_subject.py`,
  `test_streaming.py`, `test_io_catalog.py`, `test_channel_compatibility.py`).
- Full unified offline unit/integration selection: **1,620 passed, 48 skipped,
  63 deselected**, 75 warnings, 134.37 s. Local host Python was 3.11.15.

Run the offline tests from the unified checkout with the pinned peer sources
available on `PYTHONPATH` and the Python 3.11 host environment active:

```sh
env UMI_TEST_CPU_ONLY=1 RESULTCACHING_DISABLE=1 HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 AWS_EC2_METADATA_DISABLED=true \
  AWS_SHARED_CREDENTIALS_FILE=/dev/null AWS_CONFIG_FILE=/dev/null \
  BS_INSTALL_DEPENDENCIES=no PYTHONDONTWRITEBYTECODE=1 \
  python -m pytest tests/ -p no:cacheprovider \
  -m 'unit or integration' -q -ra --tb=short --timeout=60
```

The 48 skips cover missing optional OpenPI source/client, maintainer-only
operational tools, protected EWoK archives and absent LanA atlas. The 63
unselected tests are outside this offline marker selection. This does not
replace full peer-repo suites or paid/model/data regression qualification.

### Post-restack verification

After restacking onto `529237ada8c243cab0ad67cb53abde350342f4ea` and
its merged peer pins, the full offline unified selection passed: **1,609 passed,
48 skipped, 63 deselected**, 75 warnings, 153.57 s. Relevant core tests again
passed: **71 passed**. The unified total includes all 32 adapter/timing tests.
Production MIRAGE model/runtime/worker hashes still match the GPU qualification.

The test runner needs local process inspection and loopback sockets for
subprocess and transport tests. A sandboxed attempt denied these operations;
the rerun passed without source changes. The same optional-data skips and
offline scope apply. GPU inference was not rerun after restacking.

## Evidence

Compact public results are in `examples/mirage_umi/qualification.json`.
The full run archive is retained outside this repository; it contains native
arrays, experiment recordings, configurations, stimuli, logs and package freezes.
Archive size: 1,245,231 bytes. Verified SHA-256:
`3cc9ad9c0fa8b7dfe99db57c99b5daae763390ab23d67f99b2103eec3039e5df`.
The raw archive is not distributed with this example.

## Limits and remaining work

- These short clips exercise mostly padded native windows. They establish
  integration fidelity, not predictive accuracy or generalization.
- MIRAGE is bidirectional within its 149 s context, not a causal online model.
  Only the released four subject heads (`sub-01`, `sub-02`, `sub-03`, `sub-05`)
  are supported; two were exercised here. Parcel indices are retained without
  inventing anatomical names or atlas mappings.
- The published checkpoint has training/model-selection exposure to Friends
  and Movie10. No published held-out result was reproduced and no human data
  was invented. A direct-BOLD benchmark must preserve fitting/evaluation splits
  and avoid fitting a second readout or applying another HRF to these outputs.
- Existing Algonauts benchmarks fit feature-to-BOLD mappings, so this change
  deliberately adds no benchmark registration or leaderboard score. No
  contract change is required for the integration demonstrated here.
- Longer clips, other two heads, broader runtime configurations and another
  GPU architecture remain unqualified. Exact equality is demonstrated only
  under the recorded execution settings, not promised across hardware.
- GPU qualification on the merged facade peer revisions remains future work;
  the original exact-equality evidence applies to the pins listed above.
