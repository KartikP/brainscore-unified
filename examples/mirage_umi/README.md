# MIRAGE native prediction fidelity

This example preserves the released MIRAGE encoder and Qwen3-Omni raw-media
pipeline. It does not score a human benchmark or fit a new readout.

Use a Python 3.11 UMI host and a separate Python 3.12 upstream environment.
Install MIRAGE's pinned upstream requirements in the worker, including torch
2.7 and a transformers build exposing Qwen3-Omni. Do not change UMI's baseline.
Keep model assets, caches and generated output outside Git on a storage volume
with sufficient free space.

### Runtime boundary

- `runtime.py` runs in the Python 3.11 UMI host. It stages one clip, starts the
  configured Python executable, enforces the whole-process timeout, terminates
  the process group on timeout/interruption, reads predictions and provenance,
  and removes temporary files.
- `worker.py` runs in the separate Python 3.12 environment. It checks the pinned
  MIRAGE source and checkpoint, calls upstream `run_inference`, and writes the
  output array and provenance for the host.

Both processes run on the same machine. They exchange temporary files; no EC2
management or network service is involved. A fresh worker loads the model for
each clip. This keeps MIRAGE's Python and GPU dependencies out of the UMI host.

The run directory must contain the pinned public `model.safetensors` and
`config.yaml`. Set `data.text.revision`, `data.audio.revision` and
`data.vision.revision` to the same explicit Qwen Hugging Face commit before
running. Record the original and resolved configuration checksums. Model weights
are verified against the published checkpoint's SHA-256 before UMI inference.

```sh
python examples/mirage_umi/run.py \
  --mirage-python /path/to/mirage-py312/bin/python \
  --source-dir /path/to/pinned/mirage \
  --run-dir /path/to/pinned/weights \
  --work-dir /path/to/task/cache \
  --video /path/to/short.mp4 \
  --transcript /path/to/transcript.json \
  --out /path/to/fresh/results \
  --device cuda \
  --subject sub-01
```

The transcript is a JSON list of `{word, onset, duration}` records, with times
in seconds. `[]` explicitly requests upstream's no-transcript path. The three
input events refer to one complete clip; video and audio reference the same
container, and text contains the joined timed words. The subject responds as
soon as that clip is complete, before requesting another clip.

The example runs the upstream CLI directly, runs UMI with recording, requires
exact dtype/value/ordering equality, then removes its private stimulus copy and
replays the recording by rerunning native inference from copied input assets.
Arrays, model provenance, logs and complete recordings are generated under the
chosen output directory. Source files and source media supplied by the user
are not changed. Failure leaves diagnostic artifacts rather than a passing
summary.

Outputs are parcel-wise predicted BOLD on the native 1.49 s row grid. They are
not IT activations. Relative times label the native output grid; no claim about
actual acquisition onset is made. The upstream encoder processes complete
149 s windows with bidirectional context. This is an offline clip-level model.

Existing UMI Algonauts benchmarks fit ridge readouts and apply their own HRF
alignment. Evaluating MIRAGE's fixed readout requires a separate direct-BOLD
protocol, with training/model-selection/test separation. Do not pass these
predictions through the existing fitting pipeline and call that paper parity.

## Verified released-model run

On 2026-11-02, two GPU cases passed exact native/UMI/replay equality:

| Input | Trained head | Output |
|---|---|---|
| 4 s public audiovisual clip, empty transcript | sub-01 | float32, 3 TRs × 1,000 parcels |
| 4 s synthetic video and speech, timed words | sub-02 | float32, 3 TRs × 1,000 parcels |

These cases test different inputs and heads; they do not isolate a head effect.
[qualification.json](qualification.json) records hashes and execution settings.
Full arrays and recordings are archived outside Git. Replay reruns the model
from recorded inputs; it does not return saved responses.

`setup_ec2.sh`, `download_assets.py`, `prepare_clip.sh` and
`prepare_synthetic_clip.py` prepare a new dedicated worker and the two inputs.
Run setup on a suitably provisioned Ubuntu 24.04 GPU instance with the pinned
core and this unified source exported under `/opt/mirage-run/{core,unified}`.
The setup script does not launch or stop an instance. Use `--help` on the Python
scripts for asset and input destinations. Keep `HF_HOME` and the cache paths
consistent with `run.py`. The complete commands, environment details and
scientific limits are in the
[integration report](../../docs/archive/mirage_umi_integration.md).

Upstream: [source](https://github.com/epflneuroailab/mirage) and
[released encoder](https://huggingface.co/epfl-neuroai/mirage).

## Runtime overhead

The [performance report](../../docs/archive/mirage_umi_performance.md) measures native
CLI, UMI session and recorded-experiment latency under matched execution settings.
The current adapter adds approximately 5–6% on the two short clips. Its per-call
encoder checkpoint verification takes about 2.81 s; host work outside the runtime
call takes about 3 ms, or 13 ms with the experiment and recording tools.

Use `measure_overhead.py --help` for the reproducible timing driver.
`timing_worker.py` adds benchmark-only phase timers around the unchanged native
CLI or qualified worker. [performance.json](performance.json) contains compact
results; the full samples and recordings remain outside Git. The measurements
use an initialized host and fresh native workers with warm filesystem caches.
