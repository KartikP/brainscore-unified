# Qwen2.5-VL precision configurations

`qwen2.5-vl-3b-bf16` loads Qwen/Qwen2.5-VL-3B-Instruct in bfloat16 at revision
`66285546d2b821cf421d4f5eb2576359d3770cd3`. Select that identifier in the lab's
model search, or use `brainscore.load_model('qwen2.5-vl-3b-bf16')`.

The original `qwen2.5-vl-3b`, `qwen2.5-vl-7b` and their `-vwfa` variants retain
their float16 loading, checkpoint selection, cache identifiers and behavior.
The BF16 registration has its own model and extraction-cache identity. It
declares the base 3B model's channels; it does not enable VWFA interventions.

## Why a separate registration

On two 2048 by 3072 RGB images on the tested Mac/MPS/SDPA stack, native Hugging
Face and UMI reproduce the same float16 failure on one image: the final visual
block's down-projection overflows before the merger propagates NaNs into the
language network. The BF16 control remains finite; the measured projection
magnitude of 75,776 exceeds float16's maximum of 65,504. The checkpoint declares
bfloat16, while the historical registration explicitly narrows it to float16.

Changing precision can change valid answers too. A new identifier keeps earlier
scores and cached recordings reproducible. These two images do not qualify
benchmark scores, CUDA, every backend, or every future stimulus.

## Provenance and resources

Each BF16 response includes checkpoint/revision, weight-shard SHA-256 hashes,
effective dtype/device, Torch/Transformers versions, image-processor settings,
and generation configuration. Assemblies carry this in `attrs['provenance']`;
StreamEvents also carry it in `meta['provenance']`. The metadata decorator
delegates channels, tasks, resets and participation to the original subject.
It shares assembly values and does not normalize, clip or replace NaNs.
Weight hashes are computed once at load time, with bounded read buffers.

BF16 and FP16 both use two bytes per weight. This change addresses numerical
range, not model memory use: the full-resolution two-image diagnostic reached
roughly 28 GiB combined physical footprint on this Mac. Run only one model copy
and use an appropriate explicit memory budget.

## Two-image registration check

The new registration was checked on the same two original-resolution images on
Apple M3 Pro / MPS / SDPA, Torch 2.13.0 and Transformers 4.57.6. Native generation,
UMI behavior-only, UMI behavior with all five neural regions, and an Experiment
with saved replay artifacts returned identical answers and generated token IDs.
All 18 clean/probe/recheck comparisons matched input tensors and generation
settings. Both images received normal Yes answers; all observed activations and
all saved neural arrays were finite. The 24 saved assemblies retained the BF16
identifier, revision, dtype/device and matching weight-shard hashes.

The registration run peaked at 21.03 GiB under a 32 GiB physical-footprint guard;
the earlier focused diagnostic's larger peak was about 28 GiB. All model
processes were released afterward. The 126 targeted unit/lab checks passed;
two HTTP checks required rerunning with loopback socket permission. Existing
scikit-learn environment drift remains outside this check of generated replies.
