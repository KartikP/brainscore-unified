# MIRAGE UMI overhead

Measured 2026-11-02 on the dedicated MIRAGE EC2 worker. The current integration
added approximately **5–6%** to median end-to-end native CLI latency for these
two four-second clips. Per-call encoder checkpoint verification took about
**2.81 s**. Host work outside the runtime callable took approximately **3 ms**
for the bare UMI session and **13 ms** for the recorded experiment.

The production adapter and native inference sources stayed unchanged. These
measurements describe the existing implementation before optimization.

## End-to-end measurements

Seconds, median (minimum–maximum), three measured repetitions per cell:

| Condition | Public audiovisual clip, sub-01 | Synthetic speech/video, sub-02 |
|---|---:|---:|
| Direct native CLI | 52.378 (52.176–52.778) | 46.069 (45.919–46.169) |
| UMI session | 55.133 (54.582–55.535) | 48.825 (48.478–48.927) |
| UMI experiment with recording | 55.044 (54.994–55.494) | 48.835 (48.788–48.986) |

Relative to the direct median, UMI added **5.26%** and **5.98%**, respectively.
The recorded experiment added **5.09%** and **6.00%**. The smaller recorded median
on the public clip reflects native-run variability; recording has positive
measured host cost. Three repetitions do not establish a confidence interval.

The driver also preserves within-block paired percentages in `summary.json`.
Those summarize ratios within each repetition and need not equal the ratios of
medians above. Every raw sample and its execution position is retained.

## Where the time went

Median phase measurements:

| Phase | Public UMI session | Public recorded | Synthetic UMI session | Synthetic recorded |
|---|---:|---:|---:|---:|
| Encoder checkpoint verification, seconds | 2.812 | 2.808 | 2.809 | 2.812 |
| Host work outside runtime callable, milliseconds | 3.089 | 13.203 | 3.043 | 12.658 |
| Native inference including model loading, seconds | 50.498 | 50.380 | 44.188 | 44.169 |

Direct native inference phase medians were 50.624 s and 44.305 s. The native CLI
does not perform the integration worker's extra full-file SHA-256 verification
of the 4,492,439,044-byte encoder checkpoint. That sequential check accounts for
most additional work in the current adapter. No verification was disabled.

The approximately 3 ms host measurement includes session handling, validation,
labeled assembly construction and reset. It excludes all work inside
`MirageRuntime`, including staging, worker execution and result deserialization.
It therefore measures a narrow part of the integration, not its total overhead.
The approximately 13 ms recorded measurement includes that host work plus
Experiment orchestration, input/output recording and result serialization. This
experiment does not isolate the recorder from the Experiment runner.

Native imports took approximately 0.88 s per process. Initial CUDA
synchronization/context initialization took approximately 0.16 s in direct calls
and 0.20 s in integration calls. Remaining wall time outside the timed worker
body was approximately 0.70–0.74 s. That residual combines process startup,
shutdown and host transport/work; it is not an isolated startup measurement.
Component medians should not be summed to reconstruct a median total.

## Method and limits

- One initialized Python 3.11 UMI host launches a fresh native Python 3.12
  process for every call in every condition. All conditions use the same
  benchmark timing launcher and unmodified native inference computation.
- Each input has one excluded successful warmup per condition, followed by
  three cyclic orders: direct/session/recorded, session/recorded/direct and
  recorded/direct/session. Each condition occupies every position once.
- Filesystem caches remain warm. No system/model cache is cleared. Native
  model loading still occurs on every call. This does not measure a persistent
  loaded-model API, cold application startup, or pure GPU-kernel latency.
- `perf_counter` times the complete call through array return to the host.
  CUDA synchronization brackets the native inference phase. Narrow wrappers
  time native imports, checkpoint verification and inference, then write
  sidecar JSON. Timing diagnostics do not enter the prediction assemblies.
- The direct condition executes the native CLI with phase timers. UMI uses the
  qualified `MirageSubject.participate` (named `interact` when measured) and
  `MirageRuntime`; the recorded condition adds existing Experiment and
  RecordInputsOutputs tools. No model-class
  dispatch, new Subject method, core dependency or benchmark registration was
  introduced.
- Inputs, head, CUDA settings, precision and batch size match across conditions
  within each case. Batch size is 1, backbone dtype bfloat16 and output float32.
  Different inputs and heads prevent comparing the two cases as a head effect.
- These are small media files and (3, 1000) output arrays. Recording cost can
  grow with media/assembly size. Training, fitting, benchmark scoring, memory
  pre-flight and human-data evaluation are outside this measurement.

### Startup limitation

The first attempt's initial direct warmup exceeded a 300 s timeout after the
instance restart. The process group cleaned up and GPU memory returned to zero.
At inspection, the process waited on file pages, disk reads had reached about
23 GB and the GPU was idle. This established a disk-bound loading delay.

The retry used a 1,200 s timeout. Its first successful warmup took 320.852 s
and matched the original predictions exactly. It followed the earlier partial
load, so it is not an independent cold-start measurement. Both that warmup and
the failed attempt are excluded from the 18 reported samples. The failed log,
all six successful warmups and every measured sample remain in the archive.

## Environment and fidelity

Same tested facade candidate and peer pins as the
[integration qualification](mirage_umi_integration.md): unified
`abf74bd55a48bd6585d08ccd3320dd30a1e9ceca`, core
`15dcc3c6cbfea16565052abd2663b8de4bf1b203`, vision
`eca24202bb7b188c1115d3b140b96abc0fa848c7`, language
`62e2dcfe17daa81ecdf939f6e075d9edd33389ca`. This remains qualification of that
candidate revision, with no qualification claim for subsequent facade changes.

Native MIRAGE source: `c594e3124e2a9cb7b8a26a750b373e583a7fb297`.
Encoder revision: `b38a252c28e5416336423f4eda0c42d7e1df4e9c`.
Qwen revision: `2f443cfc4c54b14a815c0e2bb9a9d6cbcd9a748b`.
Resolved config SHA-256:
`5e34c4508125ca455dd1b33c2fc498b2b1505732e0f96cf30e08a4ea31b8732c`.
Full package freezes, config and source hashes accompany the evidence.

Hardware: g7e.4xlarge, RTX PRO 6000 Blackwell 96 GB, Ubuntu 24.04,
driver 595.112.07, CUDA 12.8. Native Python 3.12.3, torch 2.7.1+cu128,
transformers 4.57.6. Host Python 3.11.14, NumPy 1.26.4, xarray 2022.3.0.

All **24 successful calls** (18 measured, six warmups) matched the original
qualification's full array hashes and float32 dtype exactly. UMI TR, parcel
and trained-subject labels passed their ordering checks. No numerical tolerance
was introduced. Qualified model/runtime/worker source hashes are unchanged.

- Focused adapter and timing tests: **32 passed**, locally and on EC2. The nine
  new tests cover balanced orders, paired arithmetic, warmup exclusion and
  output-preserving CLI/worker phase timers using explicit synthetic fixtures.
- Relevant pinned-core tests: **71 passed**.
- Full local offline unit/integration selection: **1,629 passed, 48 skipped,
  63 deselected**, 75 warnings. The focused unified tests are included in this
  total. Skips concern optional OpenPI, maintainer-only tools, protected EWoK
  data and the absent LanA atlas. Other model/data regression tiers did not run.

Post-restack offline verification on the merged facade passed **1,609 tests**
with 48 skipped and 63 deselected; **71 core tests** also passed. See the
[integration report](mirage_umi_integration.md#post-restack-verification) for
merged pins, commands and the sandbox limitation. GPU timings were not rerun.

## Reproduction

From the exported unified tree, with the existing qualified assets available:

```sh
env PYTHONPATH=/opt/mirage-run/core:/opt/mirage-run/unified \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 RESULTCACHING_DISABLE=1 \
  AWS_EC2_METADATA_DISABLED=true PYTHONDONTWRITEBYTECODE=1 \
  /opt/mirage-run/host/bin/python examples/mirage_umi/measure_overhead.py \
  --mirage-python /opt/mirage-run/native/bin/python \
  --source-dir /opt/mirage-run/mirage \
  --run-dir /opt/mirage-run/assets/encoder \
  --work-dir /opt/mirage-run/cache \
  --stimuli-dir /opt/mirage-run/stimuli \
  --out /opt/mirage-run/results/overhead-02 \
  --repetitions 3 \
  --timeout-s 1200
```

The output directory must be new. Six repetitions additionally exercise the
reversed orders. Each condition launches actual inference, so choose repetition
counts with the runtime cost in mind. The scripts do not provision instances.

[Compact results](../examples/mirage_umi/performance.json) retain medians, ranges,
phase measurements and source hashes. Full evidence is retained outside this
repository in a 1,805,727-byte archive, verified against SHA-256:
`595bde7d7842b74f08e37ee7f383a17dcc210a151bddba2da03f173e6974db79`.
It includes raw samples, recordings, phase sidecars, native logs, source snapshots,
package freezes, system details and the incomplete first attempt. The raw
archive, CPU test logs and JUnit files are not distributed with this example.

## Performance options

No performance optimization was applied. The first candidate is amortizing
checkpoint verification over a verified immutable asset/worker lifetime. A
persistent worker could also amortize imports, but upstream run_inference itself
loads the extractor and brain encoder per call; keeping those models resident
would require additional native integration and its own fidelity tests.
