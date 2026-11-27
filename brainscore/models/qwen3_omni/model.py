"""Qwen3-Omni-30B-A3B-Thinking (thinker only): one forward pass on the clip and the sound of a row together.

The thinker of ``Qwen/Qwen3-Omni-30B-A3B-Thinking`` takes the picture and the sound of a row as one interleaved
token sequence (``use_audio_in_video=True``: the picture and sound tokens are put in time order). The extractor
records, in that one pass, every thinker state (``embed`` and ``layers.0`` to ``layers.47``, width 2048), each
pooled three ways and kept apart: the mean over the picture tokens (``/picture``), the mean over the sound tokens
(``/sound``) and the last token (``/last``). It also records four blocks of the vision encoder
(``visual.blocks.<i>/mean``, width 1152) and four layers of the audio encoder (``audio_tower.layers.<i>/mean``,
width 1280), each the mean over its tokens.

Input of a row: the clip (``video_path``) and, when the benchmark gives it, the sound (``audio_path``, a WAV at
16 kHz, mono). ``FRAMES`` frames are taken evenly from the clip and given with ``fps = FRAMES / clip seconds``,
so the time positions of the picture tokens match the sound. Without a sound column, the row is the clip alone.
Text of a row (``TEMPLATE``): the model's own chat template for one user turn that holds the clip, followed by the
start of the assistant turn; ``/last`` is the last token of that start (the newline after ``assistant``).

The region map is committed (``REGION_LAYER_MAP``). It was chosen by the rule in the README on Friends films
that the registered Friends score does not use; nothing is searched at run time.
"""
import numpy as np

from brainscore import BrainScoreModel
from brainscore.selection import LayerSelector

IDENTIFIER = 'qwen3-omni-30b-a3b-thinking'
WEIGHTS = 'Qwen/Qwen3-Omni-30B-A3B-Thinking'
REVISION = None               # set to the commit of the weights used by the recorded run (README)
FRAMES = 8                    # frames taken evenly from a clip; 2 temporal patches each give 4 picture groups
SOUND_RATE_HZ = 16000
THINKER_LAYERS = 48
POOLS = ('picture', 'sound', 'last')
VISUAL_BLOCKS = (0, 8, 17, 26)        # 4 evenly spaced of the 27 vision-encoder blocks
AUDIO_LAYERS = (0, 10, 21, 31)        # 4 evenly spaced of the 32 audio-encoder layers
TEMPLATE = ('<|im_start|>user\n<|vision_start|><|video_pad|><|vision_end|><|im_end|>\n'
            '<|im_start|>assistant\n')

# The committed map (README, "Layer rule"). V1, V4 and IT read the clip, A1 reads the sound: a session without
# sound that asks A1 is N/A in pre-flight (``output_inputs``). The regions that read the clip come first, so the
# memory probe (a clip without sound) records one of them.
REGION_LAYER_MAP = {
    'V1': LayerSelector('layers.11/picture', modality='video'),
    'V4': LayerSelector('layers.23/picture', modality='video'),
    'IT': LayerSelector('layers.35/picture', modality='video'),
    'A1': LayerSelector('layers.23/sound', modality='audio'),
}


def layer_names() -> list:
    """Every layer path the extractor records, thinker states first."""
    states = ['embed'] + [f'layers.{i}' for i in range(THINKER_LAYERS)]
    return ([f'{state}/{pool}' for state in states for pool in POOLS]
            + [f'visual.blocks.{i}/mean' for i in VISUAL_BLOCKS]
            + [f'audio_tower.layers.{i}/mean' for i in AUDIO_LAYERS])


def read_sound(path) -> np.ndarray:
    """The WAV at ``path`` as float32 in [-1, 1]; refuses a rate other than 16 kHz or more than one channel."""
    from scipy.io import wavfile
    rate, samples = wavfile.read(str(path))
    if rate != SOUND_RATE_HZ or samples.ndim != 1:
        raise ValueError(f'{path}: the sound is {rate} Hz with shape {samples.shape}; the extractor takes the '
                         f'mono {SOUND_RATE_HZ} Hz WAV that the benchmark prepares. Convert it first.')
    if samples.dtype.kind == 'i':
        return samples.astype(np.float32) / np.float32(np.iinfo(samples.dtype).max + 1)
    return samples.astype(np.float32)


class OmniThinkerExtractor:
    """The pooled thinker states (and four encoder blocks of each tower) of each row, from one forward pass.

    Limit: a pooled layer name (``layers.<i>/picture``) is a recording address only; an intervention cannot
    address it.
    Limit: one row per forward pass; a row has no context from the rows before it.
    Limit: the vision encoder sits on one GPU (``device_map='auto'`` places it whole).
    """

    independent_layers = True         # one pass gives every layer; a layer depends on that layer alone

    def __init__(self, identifier: str, model, processor, clip_seconds: float = 1.49) -> None:
        self.identifier = identifier
        self._model = model
        self._processor = processor
        self._clip_seconds = clip_seconds
        config = model.config.thinker_config
        self._video_token, self._audio_token = config.video_token_id, config.audio_token_id
        self._hooked = {}
        from brainscore.model_helpers.extractors import get_layer, hook
        for path in [f'visual.blocks.{i}' for i in VISUAL_BLOCKS] + [f'audio_tower.layers.{i}' for i in AUDIO_LAYERS]:
            hook(get_layer(model.thinker, path), path, self._keep(f'{path}/mean'))

    def _keep(self, name):
        def keep(output):
            hidden = output[0] if isinstance(output, tuple) else output
            self._hooked[name] = hidden.reshape(-1, hidden.shape[-1]).float().mean(0).cpu().numpy()
        return keep

    def layer_names(self) -> list:
        return layer_names()

    def pooled(self, video_path, audio_path=None) -> dict:
        """``{layer path: vector}`` for every layer of one row (``audio_path`` None: the clip alone)."""
        import torch
        from brainscore.model_helpers.video_wrapper import default_uniform_frame_sampler
        frames = np.stack(default_uniform_frame_sampler(video_path, FRAMES))
        sound = None if audio_path is None else read_sound(audio_path)
        inputs = self._processor(text=TEMPLATE, videos=[frames], audio=None if sound is None else [sound],
                                 use_audio_in_video=sound is not None, fps=len(frames) / self._clip_seconds,
                                 return_tensors='pt')
        ids = inputs['input_ids'][0]
        spans = {'picture': ids == self._video_token, 'sound': ids == self._audio_token}
        if not spans['picture'].any() or (sound is not None and not spans['sound'].any()):
            raise ValueError(f'{video_path}: the processor gave {int(spans["picture"].sum())} picture and '
                             f'{int(spans["sound"].sum())} sound tokens; a row needs at least one of each input '
                             'it has. Check the clip and the sound files.')
        device = self._model.thinker.get_input_embeddings().weight.device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        self._hooked.clear()
        with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            out = self._model.thinker(**inputs, output_hidden_states=True, use_audio_in_video=sound is not None)
        states = out.hidden_states
        if len(states) != THINKER_LAYERS + 1:
            raise ValueError(f'the thinker gave {len(states)} hidden states; expected {THINKER_LAYERS + 1}.')
        record = dict(self._hooked)
        record.update({f'tokens/{pool}': np.array([mask.sum()], np.float32) for pool, mask in spans.items()})
        for index, state in enumerate(states):
            name = 'embed' if index == 0 else f'layers.{index - 1}'
            hidden = state[0].float()
            for pool, mask in spans.items():
                if mask.any():
                    record[f'{name}/{pool}'] = hidden[mask.to(hidden.device)].mean(0).cpu().numpy()
            record[f'{name}/last'] = hidden[-1].cpu().numpy()
        return record

    def __call__(self, stimuli, layers):
        from brainscore_core.assembly_builder import make_assembly, make_layer_neuroid_coords
        layers = list(layers)
        unknown = sorted(set(layers) - set(layer_names()))
        if unknown:
            raise ValueError(f'{unknown} are not layers of {self.identifier}; a layer path is <state>/<pool>, '
                             f'for example layers.23/picture (state embed or layers.0 to layers.47, pool '
                             f'{"/".join(POOLS)}), or an encoder block <visual.blocks.i|audio_tower.layers.i>/mean.')
        has_sound = 'audio_path' in stimuli.columns
        if not has_sound and any(layer.endswith('/sound') or layer.startswith('audio_tower.') for layer in layers):
            raise ValueError(f'{[l for l in layers if l.endswith("/sound") or l.startswith("audio_tower.")]} read '
                             'the sound, and the rows have no audio_path. Give the sound, or ask a picture layer.')
        rows = [self.pooled(video, audio) for video, audio in zip(
            stimuli['video_path'], stimuli['audio_path'] if has_sound else [None] * len(stimuli))]
        values = np.concatenate([np.stack([row[layer] for row in rows]) for layer in layers], axis=1)
        neuroid = {}
        for layer in layers:
            for name, (_, coord) in make_layer_neuroid_coords(self.identifier, layer, len(rows[0][layer])).items():
                neuroid.setdefault(name, []).extend(coord)
        presentation = {column: ('presentation', list(stimuli[column])) for column in stimuli.columns}
        return make_assembly(values, coords={**presentation, **{n: ('neuroid', v) for n, v in neuroid.items()}},
                             dims=['presentation', 'neuroid'])


def build_model(identifier: str, extractor: OmniThinkerExtractor) -> BrainScoreModel:
    """The subject: one extractor for the clip and the sound; the clip is required, the sound is optional.

    The one extractor is given under each input key, not under the tuple key ``('video', 'audio')``: on the tuple
    key, a session with both inputs calls the extractor once per region (the fused path of the dispatch does not
    share the call), here once for all regions. Each call gets the whole row, so the sound reaches the picture
    tokens too.
    Limit: this goes when the fused path of the dispatch shares one call between regions.
    """
    return BrainScoreModel(identifier=identifier, region_layer_map=REGION_LAYER_MAP,
                           extractor={'video': extractor, 'audio': extractor}, required_channels=['vision:video'])


def get_model(identifier: str) -> BrainScoreModel:
    if identifier != IDENTIFIER:
        raise ValueError(f'{identifier!r} is not registered here; this folder registers {IDENTIFIER!r}.')
    import torch
    import transformers
    from transformers import AutoProcessor, Qwen3OmniMoeForConditionalGeneration
    from brainscore.model_helpers.provenance import ProvenanceSubject
    processor = AutoProcessor.from_pretrained(WEIGHTS, revision=REVISION)
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        WEIGHTS, revision=REVISION, dtype=torch.bfloat16, device_map='auto').eval()
    subject = build_model(identifier, OmniThinkerExtractor(identifier, model, processor))
    return ProvenanceSubject(subject, {
        'identifier': identifier, 'checkpoint': WEIGHTS, 'revision': getattr(model.config, '_commit_hash', None),
        'dtype': 'bfloat16', 'device_map': 'auto', 'torch': torch.__version__,
        'transformers': transformers.__version__, 'frames_per_clip': FRAMES, 'template': TEMPLATE,
        'use_audio_in_video': 'when the row has sound', 'pools': list(POOLS)})
