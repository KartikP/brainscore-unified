"""Qwen3-VL-2B-Instruct: a vision part, a patch merger, a Qwen3 causal language model.

Layout: the vision part takes the processor's flattened patches (``image_grid_thw``),
so ``VLMVisionWrapper`` records pictures and ``VLMVideoWrapper`` records clips through
the same blocks. The language model also receives vision features from several vision
depths (deepstack); the depth hypothesis does not model this.

Regions come from the depth hypothesis over each part's blocks (no layer search).
The weights load in bfloat16 (Qwen in float16 overflows in its last vision block).
"""
from brainscore.model_helpers.vlm import build_vlm_subject


def get_model(identifier: str):
    assert identifier == 'qwen3-vl-2b'
    return build_vlm_subject(
        identifier, 'Qwen/Qwen3-VL-2B-Instruct', 'Qwen3VLForConditionalGeneration',
        vision=('model.visual', 'blocks'),                # module, then its block list
        language=('model.language_model', 'layers'),
        dtype='bfloat16',
        download_gb=4.3,
        video=True,                                       # the processor takes clips natively
        probabilities='readout',                          # fitted on the task's fitting stimuli
    )
