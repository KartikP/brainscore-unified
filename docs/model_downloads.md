# Large model downloads

The shipped BLIP-2, Qwen-VL, Qwen3.6-27B, and V-JEPA factories check their local
checkpoint cache before downloading. This includes their multimodal and ablation
variants; even random V-JEPA v1 variants currently load a checkpoint before
randomizing it. Smaller models and legacy domain factories keep their existing
loading behaviour.

A cold or partially populated cache raises `ModelDownloadRequired` before the
first download. Like the [local-data convention](local_data.md), the message names
the model, why it needs fetching, where to get it, where it will land, and how to
proceed. It also reports free disk on the cache filesystem and an approximate
budget: the full checkpoint size plus 10% and 1 GB of headroom. It never prompts
for terminal input, so headless jobs fail immediately rather than hanging.

For a managed CI/EC2 run where downloading is intended, set this before loading:

```bash
export BRAINSCORE_SKIP_MODEL_DOWNLOAD_CHECK=1
```

This bypasses both the opt-in and the disk estimate and restores the existing
loader behaviour. Hugging Face offline settings still apply. Unset the variable
to restore the guard. Alternatively, populate the checkpoint cache yourself.

Hugging Face destinations follow the installed Transformers cache configuration
(`HF_HUB_CACHE`, `HF_HOME`, `XDG_CACHE_HOME`, or the default
`~/.cache/huggingface/hub`; a `TRANSFORMERS_CACHE` override takes precedence where
supported by the installed Transformers version). Set these before starting
Python. The exact path checked is passed to the loaders. A complete cached
weight set and config loads with
`local_files_only=True`, including processor loads, so an upstream revision
cannot silently fetch new weights. Missing processor metadata may still need
preparing locally or an explicit opt-out.

V-JEPA v1 uses `vjepa_v1/vitl16.pth.tar` under `HF_HOME`, otherwise
`XDG_CACHE_HOME`, otherwise `~/.cache`. Its existing nonempty checkpoint is reused.

Approximate checkpoint sizes used by the guard (decimal GB):

| Checkpoint | GB | Basis |
| --- | ---: | --- |
| BLIP-2 OPT-2.7B | 15.5 | [Published weight shards](https://huggingface.co/Salesforce/blip2-opt-2.7b/tree/main); budgets one format, not both |
| Qwen2.5-VL-3B | 7.6 | [Published weight shards](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/tree/main), rounded up |
| Qwen3.6-27B | 55.6 | [Published checkpoint](https://huggingface.co/Qwen/Qwen3.6-27B/tree/main) |
| V-JEPA v2 ViT-L | 1.3 | [Transformers weights](https://huggingface.co/facebook/vjepa2-vitl-fpc64-256/tree/main), excluding the original training checkpoint |
| V-JEPA v1 ViT-L | 5.2 | Approximate full [training checkpoint](https://github.com/facebookresearch/jepa#model-zoo), including optimizer state |

These are conservative full-download budgets, not exact remaining bytes for a
partial cache. They can drift with upstream files and exclude auxiliary audio
towers, allocator memory, and later benchmark assets. Loading weights in float16
does not halve a checkpoint stored in float32. Disk availability does not prove
that a model fits in RAM or accelerator memory.

The current activation wrappers already select CUDA, then MPS, then CPU. This
guard leaves placement and precision unchanged, including explicit float32
registrations; it does not claim to validate accelerator support or runtime fit.
