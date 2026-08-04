# Image-processor pinning

Model registrations pin which image-processor implementation they use, via
`brainscore_core.hf_compat.pin_image_processor`. This page records why, and which
scores it moves.

## Why

transformers rebound the image-processor class names. `CLIPImageProcessor` is the
PIL implementation in 4.57 and the torchvision ("fast") one in 5.x, with the PIL
version renamed to `CLIPImageProcessorPil`. The **default** therefore changes across
the upgrade even though the model is untouched.

Measured on CLIP ViT-B/32, before the forward pass:

| comparison | difference |
|---|---|
| `pixel_values`, 4.57 default vs 5.14 default | 3.0e-2 |
| vision output, 4.57 default vs 5.14 default | 1.8e-2 |
| `pixel_values`, 4.57 default vs 5.14 `use_fast=False` | **0.0** |
| either explicit setting, across versions | **0.0** |

Each version is internally deterministic (rerunning gives a difference of exactly
0), so this is a real change, not noise. `attn_implementation` was `sdpa` in both,
so attention is not the cause. Only the *default* moved; both explicit settings are
stable, which is why pinning fixes it.

## Why not just pass `use_fast=False`

Because on a processor class it does not only affect images:

```
AutoProcessor.from_pretrained(clip)                  image=CLIPImageProcessor  tok=CLIPTokenizerFast
AutoProcessor.from_pretrained(clip, use_fast=False)  image=CLIPImageProcessor  tok=CLIPTokenizer
```

It swaps the **tokenizer** to the slow Python implementation as well, changing the
text path and slowing it down. `pin_image_processor` replaces only the
image-processor component and leaves the tokenizer alone.

Audio feature extractors (`Wav2Vec2FeatureExtractor`, `SeamlessM4TFeatureExtractor`)
ignore `use_fast` — it is image-only — so those sites are deliberately untouched.
`pin_image_processor` returns anything without an `image_processor` attribute
unchanged, so callers do not need to branch.

## Effect on scores

Pinning selects PIL. Under transformers 4.57 most models already resolved to PIL, so
pinning is a no-op for them. **Qwen2.5-VL is the exception**: transformers already
flipped `Qwen2VLImageProcessor` to fast-by-default, and pinning moves it back.

| model | default under 4.57 | pinned | scores affected |
|---|---|---|---|
| CLIP ViT-B/32 | `CLIPImageProcessor` (PIL) | same | no — verified identical |
| BLIP-2 OPT-2.7B | `BlipImageProcessor` (PIL) | same | no — verified identical |
| ConvNeXtV2-tiny | `ConvNextImageProcessor` (PIL) | same | no |
| VideoMAE base | `VideoMAEImageProcessor` (PIL) | same | no |
| **Qwen2.5-VL-3B** | **`Qwen2VLImageProcessorFast`** | `Qwen2VLImageProcessor` | **yes** |

Qwen's `pixel_values` shift by max 1.5e-2 (mean 6.3e-5), so every Qwen2.5-VL score
in the repository moves — by how much is measured below.

### Measured at the score level

Qwen2.5-VL-3B scored on `MajajHong2015public.IT-pls-unified` both ways, each arm in
its own process with activation caches cleared between them:

| arm | image processor | score | extraction |
|---|---|---|---|
| default | `Qwen2VLImageProcessorFast` | 0.3152492311 | 125 s |
| pinned | `Qwen2VLImageProcessor` (PIL) | 0.3151280291 | 132 s |

**Delta −0.000121, or 0.038% relative.** A 1.5e-2 shift in pixels becomes a 1.2e-4
shift in score. At the precision scores are reported the number does not move:
0.3152 → 0.3151 at four decimals, 0.315 → 0.315 at three.

So the documented Qwen figures stand as published, and the eventual repository-wide
move to the fast processors is also a sub-0.001 change rather than a re-baselining
exercise. The pinning is cheap insurance, not a trade-off.

Two harness errors had to be fixed before this number was trustworthy, both of which
produced a plausible-looking 0.000000 delta:

- Both arms silently replayed one cached activation set, because the cache
  directories were listed by name and the VLM wrapper's was missed. Arms now clear
  by suffix, and an arm finishing faster than re-extraction takes is reported as
  invalid.
- Patching `pin_image_processor` in-process did not reach the model registrations,
  which bind the name via `from ... import` at import time; the first arm's patch
  then persisted in `sys.modules` into the second. Each arm now runs in its own
  process.

## The longer-term choice

Pinning to PIL preserves continuity but freezes us on an implementation transformers
has deprecated and will eventually delete. The alternative is to adopt fast
everywhere and re-baseline, which aligns with upstream and is faster, but shifts
published numbers.

The current position is deliberate: pin now so the transformers upgrade changes no
vision score, and treat moving to fast as a separate decision rather than a side
effect of a dependency bump. The measurement above makes that later move cheap —
0.038% on the one model affected — so it is a scheduling question, not a
re-baselining one.
