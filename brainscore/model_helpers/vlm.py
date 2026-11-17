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
    DEPTH_HYPOTHESIS, DEPTH_TABLE, depth_layers)

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
    # The class first: a class this transformers lacks is the first thing to fix, before any download.
    try:
        checkpoint_class = getattr(transformers, model_class)
    except AttributeError:
        raise ImportError(f'{identifier!r} needs transformers.{model_class}, which transformers '
                          f'{transformers.__version__} does not have.') from None
    download = hf_preflight(identifier, hf_id, download_gb, revision=revision)
    placement = {}
    if device_map is not None:
        placement = {'device_map': device_map}
        if device_map == 'auto':
            placement['max_memory'] = _placement_budget(torch)
    model = checkpoint_class.from_pretrained(
        hf_id, dtype=getattr(torch, dtype), **placement, **download)
    if device_map is None:
        from brainscore.model_helpers._device import select_device
        model = model.to(select_device())
    processor = transformers.AutoProcessor.from_pretrained(hf_id, **download)
    return model.eval(), pin_image_processor(processor, hf_id, **download)


def _placed(module) -> Optional[str]:
    """The one device where the loaded part's parameters are, so that its extractor keeps it there;
    None when they are spread over devices or not loaded (the extractor then leaves them as placed)."""
    devices = {parameter.device for parameter in module.parameters()}
    if len(devices) != 1 or next(iter(devices)).type == 'meta':
        return None
    return str(next(iter(devices)))


# The settings that give one whole image for each picture, by family; named in the refusal.
WHOLE_IMAGE_SETTINGS = "SmolVLM: do_image_splitting=False; Gemma 3: do_pan_and_scan=False; InternVL: crop_to_patches=False"


def _whole_images(paths, image_processor, settings):
    """Each picture as one whole image ``(C, H, W)``, by the image processor with the stated
    settings; a processor that gives several tiles or crops for one picture is refused."""
    import numpy as np
    from PIL import Image
    rows = []
    for path in paths:
        with Image.open(path) as image:
            pixels = image_processor(images=[image.convert('RGB')], return_tensors='pt', **settings)['pixel_values']
        shape = tuple(pixels.shape)
        if len(shape) == 5 and shape[:2] == (1, 1):        # one sample with one image (SmolVLM)
            pixels = pixels[0]
        if pixels.ndim != 4 or pixels.shape[0] != 1:
            raise ValueError(f'{type(image_processor).__name__} gives pixel_values of shape {shape} for one '
                             f'picture with the settings {settings}, not one whole image. State the setting '
                             f'that turns tiles and crops off in the registration ({WHOLE_IMAGE_SETTINGS}).')
        rows.append(pixels.float().numpy())
    return np.concatenate(rows)


def _blocks(model, part: Tuple[str, str]):
    """``(the part's module, the paths of its blocks relative to that module)``."""
    root, blocks = part
    count = len(list(model.get_submodule(f'{root}.{blocks}').children()))
    return model.get_submodule(root), [f'{blocks}.{index}' for index in range(count)]


def _intervention(model, *parts):
    """``pytorch_intervention`` on one module that holds each part's blocks under the first name
    of its block path, so a change names a layer as the region map does. The modules are
    shared, not copied: recording and generation see the change."""
    import torch
    from brainscore.model_helpers.intervention import pytorch_intervention
    roots = {}
    for part in parts:
        if part is None:
            continue
        root, blocks = part[0], part[1]
        name = blocks.split('.')[0]
        if name in roots:
            raise ValueError(f'Two parts have block paths that start with {name!r}, so a change named '
                             f'by a layer would be ambiguous; pass intervention=False.')
        roots[name] = model.get_submodule(f'{root}.{name}')
    return pytorch_intervention(torch.nn.ModuleDict(roots))


def build_vlm_subject(identifier: str, hf_id: str, model_class: str, *,
                      vision: Tuple[str, str], language: Tuple[str, str], dtype: str, download_gb: float,
                      image_processor_kwargs: Optional[Mapping[str, Any]] = None, video: bool = False,
                      audio: Optional[Tuple[str, str, Any]] = None,
                      probabilities: Union[str, Tuple[str, str]] = 'readout',
                      device_map: Optional[Union[str, dict]] = None, revision: Optional[str] = None,
                      chat_template_kwargs: Optional[Mapping] = None, intervention: bool = True,
                      provenance: Optional[Mapping[str, Any]] = None):
    """A subject for the vision-language checkpoint ``hf_id``, loaded as ``transformers.<model_class>``.

    :param vision: ``(module path in the loaded model, path of its block list in that module)``.
    :param language: the same for the causal language model (recorded by ``TextWrapper``).
    :param dtype: the precision the weights load in (``'bfloat16'``, ``'float16'``, ``'float32'``).
    :param download_gb: the approximate checkpoint size, for the download check.
    :param image_processor_kwargs: the settings of the processor's image part that give each
        picture as one whole image (``{}`` when its defaults do). Given, the vision module takes those pixels
        and ``PytorchWrapper`` with ``LayerPCA(1000)`` records it; left out, the module takes the
        processor's flattened patches (Qwen-VL layout) and ``VLMVisionWrapper`` records it.
    :param video: also record clips through the processor's video part (``VLMVideoWrapper``,
        Qwen-VL layout only).
    :param audio: ``(module path, block list path, extractor)`` of an audio encoder.
    :param probabilities: the part that gives ``behavior:probabilities``: ``'readout'`` (fitted
        on the task's fitting stimuli) or ``('likelihood', 'softmax')``.
    :param device_map: passed to ``from_pretrained``; ``'auto'`` spreads a large model over the
        accelerators with room left for activations. ``None``: one device.
    :param intervention: wire ``pytorch_intervention`` over the parts' block lists, so the subject
        takes ``lesion``, ``stimulation`` and ``pharmacological`` and answers ``perturbation``; a
        change names a layer as its region does (``blocks.10``, ``layers.2``). ``False`` for a
        family whose parts' block lists start with the same name.
    :param provenance: the registration's own facts, added to the score's provenance.
    """
    if image_processor_kwargs is not None and video:
        raise ValueError('video=True records clips through the flattened-patch video processor '
                         '(VLMVideoWrapper); no clip extractor fits a vision part given image_processor_kwargs=.')
    from brainscore import BrainScoreModel
    from brainscore.model_helpers.provenance import ProvenanceSubject
    from brainscore.model_helpers.replies import chat_parts
    from brainscore.model_helpers.text_wrapper import TextWrapper
    from brainscore.selection import LayerSelector

    model, processor = _load(identifier, hf_id, model_class, dtype, download_gb, revision, device_map)
    vision_module, vision_blocks = _blocks(model, vision)
    language_module, language_blocks = _blocks(model, language)
    region_layer_map = depth_layers(vision_blocks, DEPTH_TABLE['vision'])
    depths = dict(DEPTH_TABLE['vision'], **DEPTH_TABLE['language'])
    # The picture tokens enter the language model, so its layer is reached from text and
    # from a picture in an answering pass; the text extractor records it.
    region_layer_map['language_system'] = LayerSelector(
        name=depth_layers(language_blocks, DEPTH_TABLE['language'])['language_system'],
        modality=('text', 'vision'))

    if image_processor_kwargs is None:
        from brainscore.model_helpers.vlm_vision_wrapper import VLMVisionWrapper
        image = VLMVisionWrapper(model=vision_module, processor=processor, identifier=f'{identifier}-vision',
                                 device=_placed(vision_module))
    else:
        from brainscore_vision.model_helpers.activations.pca import LayerPCA
        from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
        device = _placed(vision_module)
        if device is None:
            raise ValueError(f'{identifier!r}: the vision part is spread over devices (or not loaded), and '
                             f'PytorchWrapper moves its module to one device; place the vision part on one '
                             f'device in device_map.')
        image = PytorchWrapper(model=vision_module, identifier=f'{identifier}-vision', device=device,
                               preprocessing=partial(_whole_images, image_processor=processor.image_processor,
                                                     settings=dict(image_processor_kwargs)))
        # Every patch token is kept, too many features for PLS: reduce each layer to 1000.
        LayerPCA.hook(image, n_components=1000)
    extractor = {'vision': image, 'text': TextWrapper(
        model=language_module, tokenizer=processor.tokenizer, identifier=f'{identifier}-text',
        device=_placed(language_module))}
    if video:
        from brainscore.model_helpers.vlm_video_wrapper import VLMVideoWrapper
        extractor['video'] = VLMVideoWrapper(vision_module, processor, identifier=f'{identifier}-video')
    if audio is not None:
        root, blocks, extractor['audio'] = audio
        _, audio_blocks = _blocks(model, (root, blocks))
        depths.update(DEPTH_TABLE['audio'])
        region_layer_map.update({region: LayerSelector(name=layer, modality='audio') for region, layer
                                 in depth_layers(audio_blocks, DEPTH_TABLE['audio']).items()})

    parts = {'intervention': _intervention(model, vision, language, audio)} if intervention else {}
    reply, likelihood = chat_parts(model, processor, chat_template_kwargs=chat_template_kwargs)
    subject = BrainScoreModel(
        identifier, model, region_layer_map, extractor=extractor, **parts,
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
