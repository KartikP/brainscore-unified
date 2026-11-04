# Coordinated UMI candidate qualification, September 16, 2026

The recorded Linux CPU vision gap is closed. General availability remains open.

## Qualified source and wheels

| Repository | Source commit | Distribution |
| --- | --- | --- |
| core | `4bb7a31d31ddd8109e0474238967f7c2ba69c8b6` | `brainscore_core-2.4.0rc1-py3-none-any.whl` |
| vision | `bed1612461c994901ecd7df80e0804a619d72a87` | `brainscore_vision-2.4.0rc1-py3-none-any.whl` |
| language | `a4dfe9d8e8080c5cd5d06a08bf20885bc01e4f65` | `brainscore_language-2.3.0rc1-py3-none-any.whl` |
| unified | `2e61867fa09dc89793f35090c73270d3a9f9f0ca` | `brainscore-0.3.0rc1-py3-none-any.whl` |

All 3,408 packaged Python and benchmark CSV files matched these source revisions
byte for byte. The [machine-readable record](2026-09-16-candidate.json) contains
wheel hashes, checkpoint hashes, route scores, metric-input hashes and actual
execution environments. This comparison does not rebuild wheels or establish
reproducible build metadata. The documentation commit adds this record without
changing package code.

## Scientific results

| Recorded profile | Cases | Result |
| --- | --- | --- |
| Ubuntu, NVIDIA L4, FP32 | Four vision and four language | 8/8 passed |
| Ubuntu, AMD CPU, FP32, one compute thread | Four language | 4/4 passed |
| Amazon Linux 2023, Intel CPU, FP32, two compute threads | Four vision | 4/4 passed |

Within each vision profile, legacy, adapter and native activations, raw scores
and normalized scores agree exactly. Native language routes pass the explicitly
versioned CPU or L4 [numerical policy](../numerical_policy.md); adapter activation
equality remains exact. CPU vision retains its unchanged historical policy.
Persistent activation caching was disabled and unstaged downloads were refused.

The GPU matrix combines independently audited completed cases from several run
reports. The initial `cuda` report contains two completed vision cases but its
whole selection was interrupted; it is not presented as a completed eight-case
run. Separate completed reports supply the other six cases. Original report
hashes and completion flags are retained in the machine-readable record.

Linux CPU language and vision used different hardware, operating systems and
thread counts. Both used CUDA-enabled Torch explicitly on CPU; a separate
CPU-only wheel is not qualified. Fixed ResNet18/GPT-2 results do not qualify
HMAX, every model, another GPU family or arbitrary inputs.

## Other evidence

The Linux/CUDA run passed 1,054 selected source tests, installed dependency
checks and both partner demos; 111 copied evidence files were verified. The CPU
vision follow-up passed 20 installed CLI tests and dependency checks, with all
52 copied evidence files independently verified. Initial setup errors and
historical numerical failures remain in the archives and are not counted as
passes. Both paid runs stopped within their approved caps; the GPU instance's
original type was restored.

Full logs and staged inputs remain outside this repository in the workspace's
`work/umi-production/artifacts/aws-qualification-2026-09-15` and
`work/umi-production/artifacts/linux-cpu-vision-2026-09-16` directories. The
committed record contains observations and hashes, not stimulus data or model
weights. No remote release CI, publication or deployment is implied.

## Remaining gates

Investigate HMAX provenance and fixed-layer legacy/adapter behavior before
attributing its drift to layer selection or dependencies. Real DROID integration,
an independent researcher trial, final supported profiles, licensing, ownership
and recovery decisions remain open. See the [production policy](../production_release.md).

### Subsequent HMAX control

The [clean current-master comparison](2026-09-16-hmax-master.md) now matches the
UMI candidate exactly for both default mapped HMAX CPU cases. The older
published-score gap also occurs upstream. Historical logs are not required for
that direct compatibility result. This later source qualification is separate
from the fixed candidate wheel set identified above; its identities are unchanged.
