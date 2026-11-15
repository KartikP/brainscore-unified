"""
BLIP-2 OPT-2.7B registered as a BrainScoreModel — second VLM with a different
architecture from CLIP and Qwen.

Architecture: ViT-G/14 vision encoder + Q-Former + OPT-2.7B causal LM. This
is a third VLM topology (encoder + adapter + decoder), distinct from CLIP's
dual-encoder pair and Qwen's flattened-patch native VLM.

Vision input is standard (batch, C, H, W) — wraps cleanly with PytorchWrapper,
unlike Qwen which needed VLMVisionWrapper. This validates that the unified
interface scales across VLM families with very different vision input layouts:
- CLIP: dual encoder, vision_model.encoder.layers.{N}
- Qwen: flattened patches, VLMVisionWrapper
- BLIP-2: standard ViT, PytorchWrapper

Layer counts:
  vision_model.encoder.layers.{0-38}             — 39 ViT-G blocks
  language_model.model.decoder.layers.{0-31}     — 32 OPT decoder layers
"""

import functools
import re
from typing import List

import torch
from PIL import Image
from transformers import AutoProcessor, Blip2ForConditionalGeneration

from brainscore.model_helpers.text_wrapper import TextWrapper
from brainscore import BrainScoreModel
from brainscore_vision.model_helpers.activations.pca import LayerPCA
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_core.hf_compat import pin_image_processor
from brainscore.models._shared import _load_preprocess_images


REGION_LAYER_MAP = {
    # Vision regions — relative to vision_model (PytorchWrapper root)
    'V1': 'encoder.layers.4',
    'V2': 'encoder.layers.10',
    'V4': 'encoder.layers.20',
    'IT': 'encoder.layers.34',
    # Language region — relative to language_model.model.decoder (TextWrapper root)
    'language_system': 'layers.28',
}


def _make_answer(blip_model, blip_processor, max_new_tokens: int = 8):
    """Generation callable for BLIP-2 instruction-following.

    BLIP-2 OPT-2.7B isn't as instruction-tuned as Qwen-VL (no RLHF), so we
    prompt it in a captioning-style template that it handles better:
    "Question: {instruction} Answer:" → model completes with label.
    """
    device = next(blip_model.parameters()).device

    def generate(stimulus_row, instruction: str, label_set: List[str]) -> str:
        image = Image.open(stimulus_row['image_file_name']).convert('RGB')
        # BLIP-2 OPT responds best to "Question: ... Answer:" template
        prompt = (f"Question: {instruction} "
                  f"Choose from: {', '.join(label_set)}. Answer:")
        inputs = blip_processor(text=prompt, images=image,
                                return_tensors='pt')
        # Explicit per-tensor device move. BatchFeature.to(device) does not
        # reliably move all tensors on MPS; BLIP-2's generate() accesses
        # input_ids inside get_input_embeddings() and needs it on the same
        # device as the embedding weights.
        model_dtype = next(blip_model.parameters()).dtype
        inputs = {
            k: (v.to(device, dtype=model_dtype) if torch.is_tensor(v) and v.is_floating_point()
                else (v.to(device) if torch.is_tensor(v) else v))
            for k, v in inputs.items()
        }

        with torch.no_grad():
            generated = blip_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        response = blip_processor.tokenizer.decode(
            generated[0], skip_special_tokens=True).strip().lower()
        # BLIP-2 echoes the prompt sometimes; strip it
        if 'answer:' in response:
            response = response.split('answer:')[-1].strip()

        for lbl in label_set:
            if response == lbl.lower():
                return lbl
        for lbl in label_set:
            if re.search(rf'\b{re.escape(lbl.lower())}\b', response):
                return lbl
        return response  # unparseable — caller defaults to label_set[0]

    return generate


def load_blip2(identifier: str):
    """BLIP-2 OPT-2.7B in float16 and its processor (image part pinned): the one load of the family.

    ``identifier`` names the registration in the download check.
    """
    from brainscore.models._downloads import hf_preflight
    download = hf_preflight(identifier, 'Salesforce/blip2-opt-2.7b', 15.5)
    blip_model = Blip2ForConditionalGeneration.from_pretrained(
        'Salesforce/blip2-opt-2.7b',
        torch_dtype=torch.float16,
        **download,
    )
    # Pin the image-processor implementation: transformers 5 rebinds the class
    # names, moving the default from PIL to torchvision and shifting pixels.
    blip_processor = AutoProcessor.from_pretrained('Salesforce/blip2-opt-2.7b', **download)
    blip_processor = pin_image_processor(blip_processor, 'Salesforce/blip2-opt-2.7b', **download)
    return blip_model, blip_processor


def picture_tower(blip_model, blip_processor, identifier: str, backbone_id=None) -> PytorchWrapper:
    """The picture extractor of every BLIP-2 registration: the vision model, the image loader, LayerPCA(1000).

    ``identifier`` labels the activations; ``backbone_id`` names the stored
    path of the legacy wrapper (its ``identifier`` when None).
    """
    # BLIP-2's vision_model can be called standalone (pixel_values ->
    # last_hidden_state), so the full model wrapper isn't needed.
    vision_wrapper = PytorchWrapper(
        identifier=identifier,
        backbone_id=backbone_id,
        model=blip_model.vision_model,
        preprocessing=functools.partial(_load_preprocess_images, processor=blip_processor),
    )
    # ViT-G hook outputs (batch, 257, 1408) flatten to 361K features per image.
    # That's too many for sklearn PLS at scoring time. LayerPCA(1000) is the
    # convention used by LayerSelection / Layer Mapping Explorer — it fits a
    # PCA on 1000 ImageNet validation images and reduces hook output to 1000D.
    # First-time overhead is ~1-2 min per layer; result cached to disk.
    LayerPCA.hook(vision_wrapper, n_components=1000)
    return vision_wrapper


def get_model(identifier: str) -> BrainScoreModel:
    assert identifier == 'blip2-opt-2.7b'
    blip_model, blip_processor = load_blip2(identifier)
    vision_wrapper = picture_tower(blip_model, blip_processor, identifier)

    # Text: TextWrapper wraps the OPT decoder. Causal LM → last aggregation.
    # Wrap language_model.model.decoder so layer paths are 'layers.{N}'.
    text_wrapper = TextWrapper(
        model=blip_model.language_model.model.decoder,
        tokenizer=blip_processor.tokenizer,
        identifier=f'{identifier}-text',
    )

    # BLIP-2 is composed of vision_model + qformer + language_projection +
    # language_model. Each wrapper moves only its sub-module to device, so
    # qformer/language_projection can end up on a different device than
    # vision_model and language_model. Force-align to one device for
    # end-to-end generate() to work.
    from brainscore.model_helpers._device import select_device
    blip_model = blip_model.to(select_device())

    # Generation path (BLIP-2's OPT decoder does instruction-style completion).
    answer = _make_answer(blip_model, blip_processor)

    return BrainScoreModel(
        identifier=identifier,
        model=blip_model,
        region_layer_map=REGION_LAYER_MAP,
        extractor={
            'vision': vision_wrapper,
            'text': text_wrapper,
        },
        answer=answer,
        # Readout path also available — PCA-1000 features from 'IT' layer.
        behavioral_readout_layer='encoder.layers.34',
        # The answer function and the readout can both answer a legacy task; the answer function owns it.
        capability_config={'behavior-sessions': {'behavior': 'generation'}},
    )
