"""Register a Hugging Face vision-language model (a vision part, a projector, a causal LM) in one call.

    model_registry['my-vlm'] = lambda: build_vlm_subject(
        'my-vlm', 'Org/My-VLM-2B', 'MyVLMForConditionalGeneration',
        vision=('model.visual', 'blocks'), language=('model.language_model', 'layers'),
        dtype='bfloat16', download_gb=4.3, video=True)

The subject records the vision part (pictures, and clips when ``video``) and the
language part; it writes replies and scores options (``chat_parts``) and fits a
readout on its IT layer. Its regions come from the stated depth hypothesis
(``brainscore.tools.auto_register``), not from a layer search; the score's
provenance says so (``region_layer_map_status``).
"""
from functools import partial
from typing import Any, Mapping, Optional, Tuple, Union

from brainscore.tools.auto_register import (
    DEFAULT_AUDIO_REGIONS, DEFAULT_LANGUAGE_REGIONS, DEFAULT_VISION_REGIONS, DEPTH_HYPOTHESIS, space_layers)

# Share of each accelerator given over to weights under device_map='auto'; the rest holds activations.
WEIGHT_FRACTION = 0.78


def _placement_budget(torch):
    """Per-device memory budget for ``device_map='auto'``, or None on CPU."""
    if not torch.cuda.is_available():
        return None
    budget = {index: int(torch.cuda.get_device_properties(index).total_memory * WEIGHT_FRACTION)
              for index in range(torch.cuda.device_count())}
    budget['cpu'] = '64GiB'      # spill target if the weights still do not fit
    return budget


def _load(identifier: str, hf_id: str, model_class: str, dtype: str, download_gb: float,
          revision: Optional[str], device_map: Optional[Union[str, dict]]):
    """The checkpoint (``transformers.<model_class>``) and its processor, image part pinned."""
    import torch
    import transformers
    from brainscore_core.hf_compat import pin_image_processor
    from brainscore.models._downloads import hf_preflight
    download = hf_preflight(identifier, hf_id, download_gb, revision=revision)
    placement = {}
    if device_map is not None:
        placement = {'device_map': device_map}
        if device_map == 'auto':
            placement['max_memory'] = _placement_budget(torch)
    model = getattr(transformers, model_class).from_pretrained(
        hf_id, dtype=getattr(torch, dtype), **placement, **download)
    if device_map is None:
        from brainscore.model_helpers._device import select_device
        model = model.to(select_device())
    processor = transformers.AutoProcessor.from_pretrained(hf_id, **download)
    return model.eval(), pin_image_processor(processor, hf_id, **download)


def _blocks(model, part: Tuple[str, str]):
    """``(the part's module, the paths of its blocks relative to that module)``."""
    root, blocks = part
    count = len(model.get_submodule(f'{root}.{blocks}'))
    return model.get_submodule(root), [f'{blocks}.{index}' for index in range(count)]


def build_vlm_subject(identifier: str, hf_id: str, model_class: str, *,
                      vision: Tuple[str, str], language: Tuple[str, str], dtype: str, download_gb: float,
                      pixels: Optional[Mapping[str, Any]] = None, video: bool = False,
                      audio: Optional[Tuple[str, str, Any]] = None,
                      probabilities: Union[str, Tuple[str, str]] = 'readout',
                      device_map: Optional[Union[str, dict]] = None, revision: Optional[str] = None,
                      chat_template_kwargs: Optional[Mapping] = None,
                      provenance: Optional[Mapping[str, Any]] = None):
    """A subject for the vision-language checkpoint ``hf_id``, loaded as ``transformers.<model_class>``.

    :param vision: ``(module path in the loaded model, path of its block list in that module)``.
    :param language: the same for the causal language model (recorded by ``TextWrapper``).
    :param dtype: the precision the weights load in (``'bfloat16'``, ``'float16'``, ``'float32'``).
    :param download_gb: the approximate checkpoint size, for the download check.
    :param pixels: the settings of the processor's image part that give each picture whole as
        ``(C, H, W)`` (``{}`` when its defaults do). Given, the vision module takes those pixels
        and ``PytorchWrapper`` with ``LayerPCA(1000)`` records it; left out, the module takes the
        processor's flattened patches (Qwen-VL layout) and ``VLMVisionWrapper`` records it.
    :param video: also record clips through the processor's video part (``VLMVideoWrapper``,
        Qwen-VL layout only).
    :param audio: ``(module path, block list path, extractor)`` of an audio encoder.
    :param probabilities: the part that gives ``behavior:probabilities``: ``'readout'`` (fitted
        on the task's fitting stimuli) or ``('likelihood', 'softmax')``.
    :param device_map: passed to ``from_pretrained``; ``'auto'`` spreads a large model over the
        accelerators with room left for activations. ``None``: one device.
    :param provenance: the registration's own facts, added to the score's provenance.
    """
    if pixels is not None and video:
        raise ValueError('video=True records clips through the flattened-patch video processor '
                         '(VLMVideoWrapper); no clip extractor fits a vision part given pixels=.')
    from brainscore import BrainScoreModel
    from brainscore.model_helpers.provenance import ProvenanceSubject
    from brainscore.model_helpers.replies import chat_parts
    from brainscore.model_helpers.text_wrapper import TextWrapper
    from brainscore.selection import LayerSelector

    model, processor = _load(identifier, hf_id, model_class, dtype, download_gb, revision, device_map)
    vision_module, vision_blocks = _blocks(model, vision)
    language_module, language_blocks = _blocks(model, language)
    region_layer_map = space_layers(vision_blocks, DEFAULT_VISION_REGIONS)
    depths = dict(DEFAULT_VISION_REGIONS, **DEFAULT_LANGUAGE_REGIONS)
    # The picture tokens enter the language model, so its layer is reached from text and
    # from a picture in an answering pass; the text extractor records it.
    region_layer_map['language_system'] = LayerSelector(
        name=space_layers(language_blocks, DEFAULT_LANGUAGE_REGIONS)['language_system'],
        modality=('text', 'vision'))

    if pixels is None:
        from brainscore.model_helpers.vlm_vision_wrapper import VLMVisionWrapper
        image = VLMVisionWrapper(model=vision_module, processor=processor, identifier=f'{identifier}-vision')
    else:
        from brainscore_vision.model_helpers.activations.pca import LayerPCA
        from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
        from brainscore.models._shared import _load_preprocess_images
        image = PytorchWrapper(model=vision_module, identifier=f'{identifier}-vision', preprocessing=partial(
            _load_preprocess_images, processor=processor.image_processor, **pixels))
        # Every patch token is kept, too many features for PLS: reduce each layer to 1000.
        LayerPCA.hook(image, n_components=1000)
    extractor = {'vision': image, 'text': TextWrapper(
        model=language_module, tokenizer=processor.tokenizer, identifier=f'{identifier}-text')}
    if video:
        from brainscore.model_helpers.vlm_video_wrapper import VLMVideoWrapper
        extractor['video'] = VLMVideoWrapper(vision_module, processor, identifier=f'{identifier}-video')
    if audio is not None:
        root, blocks, extractor['audio'] = audio
        _, audio_blocks = _blocks(model, (root, blocks))
        depths.update(DEFAULT_AUDIO_REGIONS)
        region_layer_map.update({region: LayerSelector(name=layer, modality='audio') for region, layer
                                 in space_layers(audio_blocks, DEFAULT_AUDIO_REGIONS).items()})

    reply, likelihood = chat_parts(model, processor, chat_template_kwargs=chat_template_kwargs)
    subject = BrainScoreModel(
        identifier, model, region_layer_map, extractor=extractor,
        reply=reply, likelihood=likelihood,       # the reply also answers a plain choice
        behavioral_readout_layer=region_layer_map['IT'],
        # Generation answers the rating (reply and readout both give it natively); the
        # registration states the part that gives the probabilities.
        capability_config={'behavior-sessions': {'behavior:rating': 'generation',
                                                 'behavior:probabilities': probabilities}})
    status = ', '.join(f'{region} {depth:.2f}' for region, depth in depths.items())
    return ProvenanceSubject(subject, {
        'checkpoint': hf_id, 'revision': revision or 'not pinned', 'dtype': dtype, 'device_map': device_map,
        'region_layer_map_status': f'{DEPTH_HYPOTHESIS}: {status}; no layer search',
        **dict(provenance or {})})
