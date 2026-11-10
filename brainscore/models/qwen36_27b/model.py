"""Qwen3.6-27B registered as a BrainScoreModel: vision tower (images and clips),
language decoder, replies, likelihoods and interventions from one checkpoint.

``Qwen3_5ForConditionalGeneration`` holds both towers:

  model.visual.blocks.{0-26}          -- 27 vision transformer blocks
  model.language_model.layers.{0-63}  -- 64 decoder layers

Region paths are relative to the wrapper that records them: vision regions to
``model.visual`` (``VisionWrapper(kind='vlm')``), the language region to
``model.language_model`` (``TextWrapper``). The language decoder is the same
module the earlier text-only registration wrapped, under the same identifier
and cache key, so text recordings are unchanged.

Needs transformers 5 (``qwen3_5``) and about 56 GB of accelerator memory in
bf16; weights are placed automatically across the available devices.
"""

from brainscore.brainscore_model import BrainScoreModel
from brainscore import LayerSelector

HF_IDENTIFIER = 'Qwen/Qwen3.6-27B'

# Provisional. The language layer comes from a LeBel 2023 layer sweep, where
# ~2/3 depth sat inside the useful range. The vision layers are placed at the
# same relative depths as the Qwen2.5-VL registration and have not been swept;
# refine them with the layer-mapping explorer before quoting vision scores.
REGION_LAYER_MAP = {
    'V1': 'blocks.2',
    'V2': 'blocks.5',
    'V4': 'blocks.12',
    'IT': 'blocks.23',
    'language_system': 'layers.40',
}
TEXT_REGIONS = ('language_system',)

# Share of each accelerator given over to weights; the rest holds activations.
WEIGHT_FRACTION = 0.78


def _placement_budget(torch):
    """Per-device memory budget for ``device_map='auto'``, or None on CPU."""
    if not torch.cuda.is_available():
        return None
    budget = {
        index: int(torch.cuda.get_device_properties(index).total_memory
                   * WEIGHT_FRACTION)
        for index in range(torch.cuda.device_count())
    }
    budget['cpu'] = '64GiB'      # spill target if the weights still do not fit
    return budget


def _tower_root(visual, language_model):
    """One module whose attribute paths match the region map, so an
    intervention named by a region's layer (``blocks.N`` or ``layers.N``)
    reaches the right tower. The modules are shared, not copied, so the
    change is seen by recording and by ``generate()`` alike."""
    import torch
    return torch.nn.ModuleDict({'blocks': visual.blocks,
                                'layers': language_model.layers})


def _make_answer(model, processor, max_new_tokens: int = 16):
    """``answer(stimulus_row, instruction, label_set) -> label`` for behavioral
    tasks. Thinking is switched off so the reply is the label itself."""
    import torch
    from PIL import Image
    from brainscore.model_helpers.api_behavioral import (
        IMAGE_COLUMNS, TEXT_COLUMNS, _build_instruction, _find_column, _parse_label)

    def answer(stimulus_row, instruction, label_set):
        prompt = _build_instruction(instruction, label_set)
        image_column = _find_column(stimulus_row, IMAGE_COLUMNS)
        content, images = [], None
        if image_column:
            images = [Image.open(stimulus_row[image_column]).convert('RGB')]
            content.append({'type': 'image', 'image': images[0]})
        else:
            text_column = _find_column(stimulus_row, TEXT_COLUMNS)
            if text_column:
                prompt = f'{stimulus_row[text_column]}\n\n{prompt}'
        content.append({'type': 'text', 'text': prompt})
        text = processor.apply_chat_template(
            [{'role': 'user', 'content': content}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        inputs = processor(text=[text], images=images, return_tensors='pt',
                           padding=True).to(model.device)
        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                       do_sample=False)
        reply = processor.tokenizer.decode(
            generated[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        label = _parse_label(reply, label_set)
        return reply.strip() if label is None else label   # unparsed: caller warns

    return answer


def get_model(identifier: str) -> BrainScoreModel:
    assert identifier == 'qwen3.6-27b'
    from brainscore.models._downloads import hf_preflight
    download = hf_preflight(identifier, HF_IDENTIFIER, 55.6)
    # Imported here, not at module scope: importing brainscore must not pull in
    # torch or transformers, which is asserted by tests/test_import_hygiene.py.
    import torch
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
    from brainscore_core.hf_compat import pin_image_processor
    from brainscore.model_helpers.text_wrapper import TextWrapper
    from brainscore.model_helpers.vision_wrapper import VisionWrapper
    from brainscore.perturbation import build_pytorch_ablation_fn

    # 56 GB at bf16, so placement is automatic. Left to itself accelerate packs
    # each device to the brim and the first forward pass then OOMs on
    # activations; reserving a fraction of each device keeps room for them.
    full = Qwen3_5ForConditionalGeneration.from_pretrained(
        HF_IDENTIFIER, dtype=torch.bfloat16, device_map='auto',
        max_memory=_placement_budget(torch), **download)
    full.eval()
    visual, language_model = full.model.visual, full.model.language_model

    processor = AutoProcessor.from_pretrained(HF_IDENTIFIER, **download)
    processor = pin_image_processor(processor, HF_IDENTIFIER, **download)
    tokenizer = processor.tokenizer
    tokenizer.padding_side = 'left'              # keeps the last position real
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    vision = VisionWrapper(
        model=visual,
        kind='vlm',
        processor=processor,
        identifier=f'{identifier}-vision',
        image_input_key='pixel_values',
        forward_kwargs_map={'grid_thw': 'image_grid_thw'},
        patch_count_fn=lambda out: [int(t * h * w) for t, h, w in out['image_grid_thw']],
        layer_aggregation='mean_patches',
        batch_size=4,
    )
    text = TextWrapper(
        model=language_model,
        tokenizer=tokenizer,
        identifier=f'{identifier}-text',
        backbone_id='qwen3.6-27b',               # shareable across registrations
        layer_aggregation='last_token',
        max_length=128,
    )

    # Clips go through the model's own video processor (vision:clip); stills
    # through the image path above (vision:still). Both record the same tower.
    from brainscore.model_helpers.vlm_video_wrapper import VLMVideoWrapper
    from brainscore.model_helpers.replies import chat_parts
    video = VLMVideoWrapper(visual, processor, identifier=f'{identifier}-video')
    reply, likelihood = chat_parts(full, processor, chat_template_kwargs={'enable_thinking': False})

    return BrainScoreModel(
        identifier=identifier,
        model=full,
        region_layer_map={
            region: LayerSelector(name=layer,
                                  modality='text' if region in TEXT_REGIONS else 'vision')
            for region, layer in REGION_LAYER_MAP.items()
        },
        extractor={'vision': vision, 'video': video, 'text': text},
        visual_degrees=8,
        answer=_make_answer(full, processor),
        reply=reply,
        likelihood=likelihood,
        behavioral_readout_layer=REGION_LAYER_MAP['IT'],
        intervention=build_pytorch_ablation_fn(_tower_root(visual, language_model)),
    )
