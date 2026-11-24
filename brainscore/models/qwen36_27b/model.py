"""Qwen3.6-27B registered as a BrainScoreModel: vision tower (images and clips),
language decoder, replies, likelihoods and interventions from one checkpoint.

``Qwen3_5ForConditionalGeneration`` holds both towers:

  model.visual.blocks.{0-26}          -- 27 vision transformer blocks
  model.language_model.layers.{0-63}  -- 64 decoder layers

Region paths are relative to the wrapper that records them: vision regions to
``model.visual`` (``VLMVisionWrapper``), the language region to
``model.language_model`` (``TextWrapper``). The language decoder is the same
module the earlier text-only registration wrapped, under the same identifier
and cache key, so text recordings are unchanged.

Needs transformers 5 or later (``qwen3_5``) and about 56 GB of accelerator memory in
bf16; weights are placed automatically across the available devices. Not loadable in
the reference environment (transformers 4.57.6): the load says so.
"""

from brainscore.brainscore_model import BrainScoreModel

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
    """``answer(stimulus_row, instruction, label_set) -> text`` for behavioral
    tasks: the row's picture and text, then the prompt. Thinking is switched off
    so the reply is short; the generated text is returned exactly."""
    import torch
    from PIL import Image
    from brainscore.model_helpers.api_behavioral import (
        answer_text, row_parts)

    def answer(stimulus_row, instruction, label_set):
        prompt = answer_text(instruction, label_set)
        image_column, text_column = row_parts(stimulus_row, 'Qwen3.6-27B answer')
        content, images = [], None
        if image_column:
            images = [Image.open(stimulus_row[image_column]).convert('RGB')]
            content.append({'type': 'image', 'image': images[0]})
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
        return reply   # the generated text, exactly: BrainScoreModel parses it

    return answer


def get_model(identifier: str) -> BrainScoreModel:
    assert identifier == 'qwen3.6-27b'
    from brainscore.models._downloads import hf_preflight
    download = hf_preflight(identifier, HF_IDENTIFIER, 55.6)
    # Imported here, not at module scope: importing brainscore must not pull in
    # torch or transformers, which is asserted by tests/test_package/test_import_hygiene.py.
    import torch
    try:
        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
    except ImportError as error:
        import transformers
        raise ImportError(
            f"Model {identifier!r} needs transformers 5 or later (Qwen3_5ForConditionalGeneration); "
            f"the installed transformers is {transformers.__version__}. It is not loadable in the "
            f"reference environment (transformers 4.57.6).") from error
    from brainscore_core.hf_compat import pin_image_processor
    from brainscore.model_helpers.text_wrapper import TextWrapper
    from brainscore.model_helpers.vlm_vision_wrapper import VLMVisionWrapper
    from brainscore.model_helpers.intervention import pytorch_intervention
    from brainscore.selection import LayerSelector

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

    vision = VLMVisionWrapper(
        model=visual,
        processor=processor,
        identifier=f'{identifier}-vision',
    )
    text = TextWrapper(
        model=language_model,
        tokenizer=tokenizer,
        identifier=f'{identifier}-text',
    )

    # Clips go through the model's own video processor (vision:video); stills
    # through the image path above (vision:image). Both record the same tower.
    from brainscore.model_helpers.vlm_video_wrapper import VLMVideoWrapper
    from brainscore.model_helpers.replies import chat_parts
    video = VLMVideoWrapper(visual, processor, identifier=f'{identifier}-video')
    reply, likelihood = chat_parts(full, processor, chat_template_kwargs={'enable_thinking': False})

    return BrainScoreModel(
        identifier=identifier,
        model=full,
        # The picture tokens enter the language model, so its layer is reached from text and from a
        # picture in an answering pass (rule 2.5); the text extractor records it.
        region_layer_map=dict(REGION_LAYER_MAP, language_system=LayerSelector(
            name=REGION_LAYER_MAP['language_system'], modality=('text', 'vision'))),
        extractor={'vision': vision, 'video': video, 'text': text},
        answer=_make_answer(full, processor),
        reply=reply,
        likelihood=likelihood,
        behavioral_readout_layer=REGION_LAYER_MAP['IT'],
        # Generation answers the rating (reply and readout both give it natively). Bare
        # behavior carries the requested format's answer and takes no statement. The readout
        # gives the probabilities natively (fitted on the task's fitting stimuli).
        answer_owners={
            'behavior:rating': 'generation',
            'behavior:probabilities': 'readout'},
        intervention=pytorch_intervention(_tower_root(visual, language_model)),
    )
