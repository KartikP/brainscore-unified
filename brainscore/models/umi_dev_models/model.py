"""The vision-language models of the UMI development study: one ``get_model`` for every entry of
``model_configs.json``, by the rules of ``README.md``. Everything of the study is in this folder."""
import json
import re
from decimal import ROUND_HALF_UP, Decimal
from functools import partial
from pathlib import Path

from brainscore.model_helpers.provenance import ProvenanceSubject

MODEL_CONFIGS = json.loads((Path(__file__).parent / 'model_configs.json').read_text())
# The packages the study needs beyond the reference environment, with their versions: stated once, in
# requirements.txt. A model names the ones its processor or forward uses (``packages`` in its values).
REQUIREMENTS = [re.split(r'[<>=!~;\[ ]', line.strip())[0] for line in
                (Path(__file__).parent / 'requirements.txt').read_text().splitlines()
                if line.strip() and not line.startswith('#')]

# The depth hypothesis: a region's layer is the block at f * (N - 1) of a part's N blocks,
# rounded half up. No data chose these values; a change of a value takes a new version word.
DEPTH_HYPOTHESIS = 'depth hypothesis v1'
DEPTH_TABLE = {
    'vision': {'V1': 0.10, 'V2': 0.25, 'V4': 0.50, 'IT': 0.90, 'VWFA': 0.90},
    'language': {'language_system': 0.67},
}
# Share of each accelerator given over to weights under device_map='auto'; the rest holds activations.
WEIGHT_FRACTION = 0.78


def depth_layers(block_paths, depths):
    """The block of each region of ``depths`` (region -> relative depth) among ``block_paths``."""
    n = len(block_paths)
    return {} if n == 0 else {region: block_paths[int((Decimal(str(f)) * (n - 1)).quantize(Decimal(1), ROUND_HALF_UP))]
                              for region, f in depths.items()}


class _StudySubject(ProvenanceSubject):
    """The study's provenance on every response, and the one input the policy acts on: it reads the
    observation's frame, so ``motor`` is answered from an observation only. The study records the
    language layer in a passive session only: a session that asks for it under a task (the planner runs
    the task even when only neural outputs are requested) or together with an answer is refused before
    any compute."""

    # The readout reads the IT layer of the vision part, fitted on pictures: these formats come from pictures.
    READOUT = ('behavior:probabilities', 'behavior:rt')

    @property
    def output_inputs(self):
        inputs = dict(self._subject.output_inputs)
        if 'motor' in self._subject.out_channels:
            inputs['motor'] = frozenset({'observation'})
        inputs.update({output: frozenset({'vision:image'}) for output in self.READOUT if output in self._subject.out_channels})
        return inputs

    def participate(self, session):
        outputs, task = set(session.outputs or ()), session.task
        answering = (task is not None and task.task_type != 'passive') or any(
            o.startswith('behavior') or o == 'motor' for o in outputs)
        shown = self._shown(session) if outputs & set(self.READOUT) else None
        if shown is not None and 'vision:image' not in shown:
            from brainscore_core.compatibility import ExecutionNotFeasibleError
            raise ExecutionNotFeasibleError(
                f"'{self.identifier}' answers {sorted(outputs & set(self.READOUT))} from a readout of its vision "
                f"part, fitted on pictures; these trials show {sorted(shown)}, no picture. Give pictures.")
        if 'neural:language_system' in outputs and answering:
            from brainscore_core.compatibility import ExecutionNotFeasibleError
            raise ExecutionNotFeasibleError(
                f"'{self.identifier}' records neural:language_system in a passive session only (the study's "
                f"rule: one session for regions, one for an answer). Ask for it in its own passive session.")
        super().participate(session)

    def _shown(self, session):
        """The inputs this session delivers to the model, as its task presents them: the table's, or the
        events of a listed source with no table (read here and given back as the same list). None for a
        stream with no table: its inputs are known only when they arrive."""
        from brainscore_core.responses import PART_CHANNELS
        from brainscore_core.streaming import FixedSource
        from brainscore_core.streaming_helpers import task_input_channels
        task = session.task
        if session.stimuli is not None:
            shown = task_input_channels(session.stimuli, task)
        elif session.source.listed:
            events = list(iter(session.source.next_input, None))
            session.source = FixedSource(events)
            shown = {event.channel for event in events}
            if getattr(task, 'present', None) is not None:
                shown &= {PART_CHANNELS.get(entry.split(':', 1)[0]) for entry in task.present}
        else:
            return None
        return set(shown) & set(self.in_channels)


def _refuse(identifier, config):
    """A package the model uses that is missing, or a class this transformers lacks: named before any
    download. A package the model names must be a line of requirements.txt."""
    import importlib
    import importlib.util
    import transformers
    from brainscore import LoadError
    utils = importlib.import_module('transformers.utils')
    for package in config['packages']:
        if package not in REQUIREMENTS:
            raise ValueError(f"'{identifier}' names the package '{package}', which requirements.txt does not state.")
        # The check the processor itself makes (transformers.utils.is_<package>_available), else an import.
        available = getattr(utils, f'is_{package}_available', None)
        if not (available() if available else importlib.util.find_spec(package)):
            raise LoadError(f"'{identifier}' needs the package '{package}', which is not installed. Next step: "
                            f"pip install -r brainscore/models/umi_dev_models/requirements.txt, or install "
                            f"'{package}' yourself.")
    if not hasattr(transformers, config['model_class']):
        raise LoadError(f"'{identifier}' needs transformers.{config['model_class']}, which transformers "
                        f"{transformers.__version__} does not have.")


def _load(identifier, config):
    """The processor (image part pinned), then the checkpoint, placed as ``device_map`` states: a fault
    of the processor shows before the large download."""
    import torch
    import transformers
    from brainscore_core.hf_compat import pin_image_processor
    from brainscore.models._downloads import hf_preflight
    pinned = {} if config['revision'] == 'not pinned' else {'revision': config['revision']}
    download = hf_preflight(identifier, config['hf_id'], config['download_gb'], **pinned)
    processor = pin_image_processor(transformers.AutoProcessor.from_pretrained(config['hf_id'], **download),
                                    config['hf_id'], **download)
    placement = {}
    if config['device_map'] is not None:
        placement['device_map'] = config['device_map']
        if config['device_map'] == 'auto' and torch.cuda.is_available():
            placement['max_memory'] = {index: int(torch.cuda.get_device_properties(index).total_memory * WEIGHT_FRACTION)
                                       for index in range(torch.cuda.device_count())}
    model = getattr(transformers, config['model_class']).from_pretrained(
        config['hf_id'], dtype=getattr(torch, config['dtype']), **placement, **download)
    if config['device_map'] is None:
        from brainscore.model_helpers.device import select_device
        model = model.to(select_device())
    return model.eval(), processor


def _one_device(identifier, module, part):
    """The one device of a loaded part; its extractor is given it, so nothing is moved. Offloaded or
    spread weights are refused: v2's extractors would move them or send inputs to meta."""
    devices = {parameter.device for parameter in module.parameters()}
    if len(devices) != 1 or next(iter(devices)).type == 'meta':
        raise ValueError(f"'{identifier}': the {part} part is on {sorted(map(str, devices))}; this study "
                         f"needs each part on one accelerator (no offload). Give a device_map or a "
                         f"max_memory that holds it on one.")
    return str(next(iter(devices)))


def _blocks(model, part):
    """``(the part's module, the paths of its blocks relative to that module)``."""
    root, blocks = part
    count = len(list(model.get_submodule(f'{root}.{blocks}').children()))
    return model.get_submodule(root), [f'{blocks}.{index}' for index in range(count)]


def _whole_images(paths, image_processor, settings):
    """Each picture as one whole image ``(C, H, W)`` by the image processor with the stated settings;
    a processor that gives several tiles or crops for one picture is refused."""
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
                             f'picture with {settings}, not one whole image; state the setting that turns '
                             f'tiles and crops off in image_processor_kwargs.')
        rows.append(pixels.float().numpy())
    return np.concatenate(rows)


class _StatedProcessor:
    """The model's processor for the answering parts, called with the stated whole-image settings, so
    that a model sees the same picture when it answers as when it is recorded (a parent processor can
    have its own default, as InternVL's crop_to_patches). The settings are also set on its image processor,
    because a processor can drop a setting it does not declare (SmolVLM's do_image_splitting).
    Everything else is the processor's own."""

    def __init__(self, processor, settings):
        self._processor, self._settings = processor, dict(settings)
        for key, value in self._settings.items():
            if hasattr(processor.image_processor, key):
                setattr(processor.image_processor, key, value)

    def __getattr__(self, name):
        return getattr(self._processor, name)

    def __call__(self, *args, **kwargs):
        return self._processor(*args, **{**self._settings, **kwargs})


def _pool_tokens(activity, kernel):
    """Patch tokens ``(B, N, C)`` on a square grid, row by row, as the mean of each ``kernel`` x ``kernel``
    block of the grid: ``(B, N / kernel**2, C)``."""
    import math
    count, n, width = activity.shape
    side = math.isqrt(n)
    if side * side != n or side % kernel:
        raise ValueError(f'{n} patch tokens are not a square grid with a side divisible by {kernel}.')
    cells = side // kernel
    return activity.reshape(count, cells, kernel, cells, kernel, width).mean((2, 4)).reshape(count, cells * cells, width)


def _pixel_extractor(token_pool=None):
    """``PytorchWrapper`` that exports a bfloat16 activation as float32 (NumPy has no bfloat16), through
    the extractors' own hook helper; the network computes in bfloat16, other dtypes export unchanged.
    With ``token_pool``, each layer's patch tokens are pooled (``_pool_tokens``) before any PCA."""
    from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
    from brainscore.model_helpers.extractors import hook, as_numpy

    class Float32Export(PytorchWrapper):
        def register_hook(self, layer, layer_name, target_dict):
            def keep(output, name=layer_name):
                activity = as_numpy(output)
                target_dict[name] = activity if token_pool is None else _pool_tokens(activity, token_pool)
            return hook(layer, layer_name, keep)
    return Float32Export


def _reply_client(reply):
    """The model's own ``reply`` as an ``api_policy`` client: ``(model, system, messages, settings)`` -> text,
    so it is asked as an API model is. A picture arrives as base64 data."""
    import base64
    import io
    from brainscore_core.responses import Prompt, Text

    def local_model(model, system, messages, settings):
        parts, names = ([('text', system)], ['instruction']) if system else ([], [])
        for part in messages[0]['content']:
            parts.append(('image', io.BytesIO(base64.b64decode(part['data']))) if part['type'] == 'image'
                         else ('text', part['text']))
            names.append(part['type'])
        return reply(Prompt(parts=tuple(parts), names=tuple(names), response=Text(
            max_tokens=settings.get('max_tokens'), temperature=settings.get('temperature') or 0)))
    return local_model


def _intervention(model, vision, language):
    """``pytorch_intervention`` on one module that holds each part's blocks under the first name of its
    block path, so a change names a layer as the region map does (the modules are shared)."""
    import torch
    from brainscore.model_helpers.intervention import pytorch_intervention
    roots = {}
    for root, blocks in (vision, language):
        name = blocks.split('.')[0]
        if name in roots:
            raise ValueError(f'The two parts have block paths that start with {name!r}: a change named by a '
                             f'layer would be ambiguous.')
        roots[name] = model.get_submodule(f'{root}.{name}')
    return pytorch_intervention(torch.nn.ModuleDict(roots))


def get_model(identifier: str):
    from brainscore import BrainScoreModel
    from brainscore.model_helpers.api_behavioral import api_policy
    from brainscore.model_helpers.replies import chat_parts
    from brainscore.model_helpers.text_wrapper import TextWrapper
    from brainscore.selection import LayerSelector

    config = MODEL_CONFIGS[identifier]
    _refuse(identifier, config)
    model, processor = _load(identifier, config)
    vision_module, vision_blocks = _blocks(model, config['vision'])
    language_module, language_blocks = _blocks(model, config['language'])
    region_layer_map = depth_layers(vision_blocks, DEPTH_TABLE['vision'])
    language_layer = depth_layers(language_blocks, DEPTH_TABLE['language'])['language_system']
    # Recorded from text, in a session without an answer (README: one session for regions, one for an answer).
    region_layer_map['language_system'] = LayerSelector(name=language_layer, modality='text')
    pixels = config['image_processor_kwargs']
    if pixels is None:          # the processor's flattened patches (Qwen-VL layout)
        from brainscore.model_helpers.vlm_vision_wrapper import VLMVisionWrapper
        image = VLMVisionWrapper(model=vision_module, processor=processor, identifier=f'{identifier}-vision',
                                 device=_one_device(identifier, vision_module, 'vision'))
    else:                       # whole pictures as (B, C, H, W); every patch token kept (or pooled), so PCA to 1000
        from brainscore_vision.model_helpers.activations.pca import LayerPCA
        pool = config.get('token_pool')
        image = _pixel_extractor(pool)(
            model=vision_module, identifier=f'{identifier}-vision' + (f'-pool{pool}' if pool else ''),
            device=_one_device(identifier, vision_module, 'vision'),
            preprocessing=partial(_whole_images, image_processor=processor.image_processor, settings=pixels))
        LayerPCA.hook(image, n_components=1000)
    extractor = {'vision': image, 'text': TextWrapper(
        model=language_module, tokenizer=processor.tokenizer, identifier=f'{identifier}-text',
        device=_one_device(identifier, language_module, 'language'))}
    if config['video']:
        from brainscore.model_helpers.vlm_video_wrapper import VLMVideoWrapper
        # The stated frame sampling (the Qwen authors' tool), on the processor, so a recording and an
        # answer sample a clip alike.
        for key in ('fps', 'min_frames', 'max_frames', 'do_sample_frames') if config.get('video_sampling') else ():
            setattr(processor.video_processor, key, config['video_sampling'][key])
        extractor['video'] = VLMVideoWrapper(vision_module, processor, identifier=f'{identifier}-video')
    reply, likelihood = chat_parts(model, processor if pixels is None else _StatedProcessor(processor, pixels))
    # A lesion addresses module units; a layer recorded as PCA components cannot select them.
    parts = {} if pixels is not None else {'intervention': _intervention(model, config['vision'], config['language'])}
    subject = BrainScoreModel(
        identifier, model, region_layer_map, extractor=extractor, reply=reply, likelihood=likelihood,
        behavioral_readout_layer=region_layer_map['IT'],
        policy=api_policy(_reply_client(reply), identifier), policy_in_channels=('observation', 'instruction', 'reward'),
        capability_config={'behavior-sessions': {'behavior:rating': 'generation', 'behavior:probabilities': 'readout'}},
        **parts)
    depths = ', '.join(f'{region} {f:.2f}' for table in DEPTH_TABLE.values() for region, f in table.items())
    return _StudySubject(subject, {
        'study': 'umi_dev_models', 'checkpoint': config['hf_id'], 'revision': config['revision'],
        'dtype': config['dtype'],
        'device_map': config['device_map'],
        'region_layer_map_status': f'{DEPTH_HYPOTHESIS}: {depths}; no layer search',
        'readout_layer': f"{region_layer_map['IT']} (the IT layer of the vision part)",
        'intervention': 'on' if parts else 'off: the vision layers are recorded as LayerPCA components',
        **({'token_pool': f"mean of each {config['token_pool']} x {config['token_pool']} block of the patch "
                          f"grid, before LayerPCA"} if config.get('token_pool') else {}),
        **({'video_sampling': "fps {fps}, min_frames {min_frames}, max_frames {max_frames}, do_sample_frames {do_sample_frames} ({source})".format(**config['video_sampling'])}
           if config.get('video_sampling') else {})})
