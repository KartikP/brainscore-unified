"""
Qwen2.5-VL-3B registered as a BrainScoreModel -- a true VLM where the vision
encoder feeds into a causal language model.

Uses VLMVisionWrapper for the vision path (it handles Qwen's flattened-patch
layout with image_grid_thw metadata) and TextWrapper for the text path
(causal mode with last aggregation). Both are class-based, cached via
@store_xarray, and symmetric in their calling convention.

Architecture:
  model.visual.blocks.{0-31}          -- 32 vision transformer blocks
  model.language_model.layers.{0-35}  -- 36 decoder layers

Region layer paths are relative to each wrapper's sub-model root:
  vision paths    → relative to qwen_model.model.visual
  language paths  → relative to qwen_model.model.language_model
"""

import re
import hashlib
import json
from pathlib import Path
from typing import List

import torch
from PIL import Image

from brainscore import BrainScoreModel
from brainscore_core.contract import Subject
from brainscore_core.hf_compat import pin_image_processor


REGION_LAYER_MAP = {
    # Vision regions -- relative to model.visual (VLMVisionWrapper root)
    'V1': 'blocks.2',
    'V2': 'blocks.6',
    'V4': 'blocks.14',
    'IT': 'blocks.28',
    # Language region -- relative to model.language_model (TextWrapper root)
    'language_system': 'layers.28',
}

# A declared search population, not an independently validated brain mapping.
# Unit selectivity is determined by the benchmark's real/pseudo localizer.
VWFA_LAYER = 'blocks.28'
ABLATION_VARIANT = 'qwen2.5-vl-3b-vwfa'

# Sizes share the vision tower (32 blocks, so the vision regions are the same)
# and differ in the decoder. The 7B language layer is provisional: ~2/3 of its
# 28 decoder layers, as layers.28 is for the 3B's 36.
SIZES = {
    'qwen2.5-vl-3b': ('Qwen/Qwen2.5-VL-3B-Instruct', 7.6, 'layers.28'),
    'qwen2.5-vl-7b': ('Qwen/Qwen2.5-VL-7B-Instruct', 16.6, 'layers.18'),
}
VARIANTS = {f'{size}-vwfa': size for size in SIZES}
BF16_IDENTIFIER = 'qwen2.5-vl-3b-bf16'
BF16_REVISION = '66285546d2b821cf421d4f5eb2576359d3770cd3'


def _bf16_provenance(model, processor, download):
    import transformers
    from transformers.utils.hub import cached_file

    hf_id = SIZES['qwen2.5-vl-3b'][0]
    def cached(filename):
        return Path(cached_file(hf_id, filename, revision=BF16_REVISION,
                    cache_dir=download.get('cache_dir'), local_files_only=True))

    index = json.loads(cached('model.safetensors.index.json').read_text())
    hashes = {}
    for filename in sorted(set(index['weight_map'].values())):
        digest = hashlib.sha256()
        with cached(filename).open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        hashes[filename] = digest.hexdigest()
    return {
        'identifier': BF16_IDENTIFIER, 'checkpoint': hf_id, 'revision': BF16_REVISION,
        'weights_sha256': hashes, 'dtype': str(model.dtype), 'device': str(model.device),
        'torch': torch.__version__, 'transformers': transformers.__version__,
        'image_processor': processor.image_processor.to_dict(),
        'generation_config': model.generation_config.to_dict(),
        'reply_generation': {'max_new_tokens': 64, 'do_sample': False},
    }


def _make_answer(qwen_model, qwen_processor, max_new_tokens: int = 8):
    """Create a generation callable for instruction-following behavioral
    tasks. Used by BrainScoreModel when TaskContext carries an
    ``instruction`` (not just fitting_stimuli).

    Signature (the ``answer`` part of BrainScoreModel):
        fn(stimulus_row, instruction, label_set) -> predicted_label
    """
    device = next(qwen_model.parameters()).device

    def generate(stimulus_row, instruction: str, label_set: List[str]) -> str:
        image = Image.open(stimulus_row['image_file_name']).convert('RGB')
        messages = [{
            'role': 'user',
            'content': [
                {'type': 'image', 'image': image},
                {'type': 'text',  'text': instruction +
                 f' Answer with exactly one of: {", ".join(label_set)}.'},
            ],
        }]
        prompt = qwen_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = qwen_processor(
            text=[prompt], images=[image], return_tensors='pt', padding=True,
        ).to(device)

        with torch.no_grad():
            generated = qwen_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        # Strip the prompt tokens, decode the tail
        input_len = inputs['input_ids'].shape[1]
        response_ids = generated[0, input_len:]
        response = qwen_processor.tokenizer.decode(
            response_ids, skip_special_tokens=True).strip().lower()

        # Robust parsing: prefer exact match, then substring match, then
        # fall back to first label (BrainScoreModel emits a warning).
        for lbl in label_set:
            if response == lbl.lower():
                return lbl
        for lbl in label_set:
            if re.search(rf'\b{re.escape(lbl.lower())}\b', response):
                return lbl
        return response  # unparseable — caller will default to label_set[0]

    return generate


def load_qwen(identifier: str, size: str):
    """A Qwen2.5-VL of ``size`` (``SIZES``), on one device, and its processor (image part pinned).

    The one load of the family: float16, or bfloat16 at ``BF16_REVISION`` for
    ``BF16_IDENTIFIER``. ``identifier`` names the registration in the download
    check. Returns the model, the processor and the download settings.
    """
    bf16 = identifier == BF16_IDENTIFIER
    hf_id, download_gb, _ = SIZES[size]
    from brainscore.models._downloads import hf_preflight
    selection = {'revision': BF16_REVISION} if bf16 else {}
    download = hf_preflight(identifier, hf_id, download_gb, **selection)
    if bf16:
        download = {**download, 'revision': BF16_REVISION}

    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        hf_id,
        torch_dtype=torch.bfloat16 if bf16 else torch.float16,
        **download,
    )
    # Pin the image-processor implementation: transformers 5 rebinds the class
    # names, moving the default from PIL to torchvision and shifting pixels.
    # The whole model on one device: the 7B's output head is not tied to its
    # input embeddings, so moving the towers alone would leave it behind.
    from brainscore.model_helpers._device import select_device
    qwen_model = qwen_model.to(select_device())
    qwen_processor = AutoProcessor.from_pretrained(hf_id, **download)
    qwen_processor = pin_image_processor(qwen_processor, hf_id, **download)
    return qwen_model, qwen_processor, download


def picture_tower(qwen_model, qwen_processor, identifier: str):
    """The picture extractor of every Qwen2.5-VL registration: the visual tower, ``<identifier>-vision``."""
    from brainscore.model_helpers.vlm_vision_wrapper import VLMVisionWrapper
    # VLMVisionWrapper handles Qwen's flattened-patch layout by default
    # (image_grid_thw segments the patches back to images).
    return VLMVisionWrapper(model=qwen_model.model.visual, processor=qwen_processor,
                            identifier=f'{identifier}-vision')


def get_model(identifier: str) -> Subject:
    bf16 = identifier == BF16_IDENTIFIER
    size = 'qwen2.5-vl-3b' if bf16 else VARIANTS.get(identifier, identifier)
    assert size in SIZES, identifier
    language_layer = SIZES[size][2]
    from brainscore.model_helpers.text_wrapper import TextWrapper

    qwen_model, qwen_processor, download = load_qwen(identifier, size)
    vision_wrapper = picture_tower(qwen_model, qwen_processor, identifier)

    # Text: TextWrapper wraps model.language_model (causal decoder).
    # last aggregation with attention_mask handles padded batches.
    text_wrapper = TextWrapper(
        model=qwen_model.model.language_model,
        tokenizer=qwen_processor.tokenizer,
        identifier=f'{identifier}-text',
    )

    # Qwen-VL is instruction-following, so expose its generate() as the
    # behavioral-task answer path. Benchmarks that pass a TaskContext
    # with an `instruction` and `label_set` (e.g., ROAR) will use this
    # path; others fall back to the logistic readout configured below.
    answer = _make_answer(qwen_model, qwen_processor)
    # Raw replies and candidate likelihoods for TaskContext(response=...) tasks:
    # ratings, written answers and implied probabilities.
    from brainscore.model_helpers.replies import chat_parts
    reply, likelihood = chat_parts(qwen_model, qwen_processor)

    layers = dict(REGION_LAYER_MAP, language_system=language_layer)
    variant_kwargs = {}
    if identifier in VARIANTS:
        from brainscore.perturbation import build_pytorch_ablation_fn
        layers['VWFA'] = VWFA_LAYER
        # Selection paths are relative to the extraction root. The same
        # visual module is also used by the full model's generate().
        variant_kwargs = dict(intervention=build_pytorch_ablation_fn(qwen_model.model.visual))
    subject = BrainScoreModel(
        identifier=identifier,
        model=qwen_model,
        region_layer_map=layers,
        extractor={
            'vision': vision_wrapper,
            'text': text_wrapper,
        },
        answer=answer,
        reply=reply,
        likelihood=likelihood,
        # Also enable the readout path so Qwen can be compared on the same
        # behavioral benchmark both ways. Uses the same vision layer as 'IT'.
        behavioral_readout_layer='blocks.28',
        # Two parts give a legacy answer and a rating natively; generation owns them. The
        # likelihood derives the probabilities by softmax (the readout gives them natively).
        capability_config={'behavior-sessions': {
            'behavior': 'generation', 'behavior:rating': 'generation',
            'behavior:probabilities': ('likelihood', 'softmax')}},
        **variant_kwargs,
    )
    if not bf16:
        return subject

    # The checkpoint's BF16 range avoids the reproduced final-visual-block
    # FP16 overflow. Keep this scientific setting explicit in every response.
    from brainscore.model_helpers.provenance import ProvenanceSubject
    return ProvenanceSubject(subject, _bf16_provenance(qwen_model, qwen_processor, download))
