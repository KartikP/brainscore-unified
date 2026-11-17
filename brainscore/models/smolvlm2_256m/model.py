"""SmolVLM2-256M-Video-Instruct: a SigLIP vision part, a pixel-shuffle projector, SmolLM2-135M.

The smallest model of the set. Layout: the vision part takes whole pictures as
``(B, C, H, W)``, so ``PytorchWrapper`` with ``LayerPCA(1000)`` records it. Image
splitting is off: each picture is one input, resized to the processor's square
``max_image_size``. The subject takes no clips (no clip extractor fits this layout), and
its vision regions are recorded from pictures alone, not during a written answer
(``PytorchWrapper`` has no live recording).

Regions come from the depth hypothesis over each part's blocks (no layer search).
"""
from brainscore.model_helpers.vlm import build_vlm_subject


def get_model(identifier: str):
    assert identifier == 'smolvlm2-256m'
    return build_vlm_subject(
        identifier, 'HuggingFaceTB/SmolVLM2-256M-Video-Instruct', 'SmolVLMForConditionalGeneration',
        vision=('model.vision_model', 'encoder.layers'),  # module, then its block list
        language=('model.text_model', 'layers'),
        dtype='bfloat16',
        download_gb=1.0,                                  # the Hub files are float32
        image_processor_kwargs={'do_image_splitting': False},  # one whole image per picture
        probabilities='readout',                          # fitted on the task's fitting stimuli
    )
