"""LFM3-Omni-2B (no speech output): one forward pass on the clip and the sound of a row together, and written
answers to text, pictures and clips.

``LiquidAI/LFM3-Omni-2B`` runs its own code (``trust_remote_code``, pinned by ``REVISION``): a vision encoder
(14 layers), an audio encoder (17 layers) and a language model of 24 layers (width 2048) that mixes short
convolutions and attention. With ``use_audio_in_video``, its processor puts the frames of a clip in groups of two
(each group with its time stamp) and the sound of each group right after it. The extractor records, in that one
pass, every state of the language model (``embed`` and ``layers.0`` to ``layers.23``), each pooled three ways
and kept apart: the mean over the picture tokens (``/picture``), the mean over the sound tokens (``/sound``) and
the last token (``/last``). It also records four layers of the vision encoder
(``vision_tower.layers.<i>/mean``) and four of the audio encoder (``audio_encoder.lfm.layers.<i>/mean``), each the
mean over its tokens.

Input of a row: the clip (``video_path``) and, when the benchmark gives it, the sound (``audio_path``, a WAV at
16 kHz, mono). ``FRAMES`` frames are taken evenly from the clip and given as decoded frames with their metadata
(``fps = FRAMES / clip seconds``), so the processor does not sample again and the time stamps match the sound.
Without a sound column, the row is the clip alone; without a clip column, the sound alone. Text of a row: one
user turn that holds the clip (and its sound), with no system turn, then the start of the assistant turn with an
empty thinking block (``<think></think>``, as the model's template writes it); ``/last`` is its last token.

Environment: the model code needs transformers ``TRANSFORMERS`` or later (5.x, inside the packages' bound), in a
clone of the coordinated environment (README, "Environment"). The load refuses an older version.

Written answers: the shared chat parts for the likelihood; the reply on the same inputs through the model's own
``generate`` (``generation_mode='text'``), greedy unless the task states its settings.

The region map (``REGION_LAYER_MAP``) is a placeholder until the layer rule of the README runs on the selection
films; nothing is searched at run time.
"""
import numpy as np

from brainscore import BrainScoreModel
from brainscore.selection import LayerSelector

IDENTIFIER = 'lfm3-omni-2b'
WEIGHTS = 'LiquidAI/LFM3-Omni-2B'
REVISION = 'c01718a568e53ae936e6b5e1ff5ab2de7d7f388b'   # LFM Open License v1.0, not gated; its own code
TRANSFORMERS = (5, 20, 2)     # the lowest version the model card names
FRAMES = 8                    # frames taken evenly from a clip; groups of 2 give 4 time-stamped picture groups
SOUND_RATE_HZ = 16000
LAYERS = 24
POOLS = ('picture', 'sound', 'last')
VISUAL_LAYERS = (0, 4, 9, 13)         # 4 evenly spaced of the 14 vision-encoder layers, first and last included
AUDIO_LAYERS = (0, 5, 11, 16)         # 4 evenly spaced of the 17 audio-encoder layers, first and last included
HEAD = '<|startoftext|><|im_start|>user\n'
TAIL = '<|im_end|>\n<|im_start|>assistant\n<think></think>'
CLIP, SOUND = '<|video|>', '<|audio|>'
ANSWER_INPUTS = ('text', 'vision:image', 'vision:video')

# Placeholder map: the middle layer of each band of the rule (README, "Layer rule"), until the rule runs on the
# selection films friends_s01e06a to s01e10b. Every region is a layer of the language model, after the point
# where the inputs meet, so it states both inputs (rule 2.5). A1 reads the sound tokens: a request for A1
# without sound is refused before any pass. The regions of the clip come first, so the memory probe (a clip
# without sound) records one of them.
_BOTH = ('video', 'audio')
REGION_LAYER_MAP = {
    'V1': LayerSelector('layers.4/last', modality=_BOTH),
    'V4': LayerSelector('layers.12/last', modality=_BOTH),
    'IT': LayerSelector('layers.20/last', modality=_BOTH),
    'A1': LayerSelector('layers.12/sound', modality=_BOTH),
}


def layer_names() -> list:
    """Every layer path the extractor records, language-model states first."""
    states = ['embed'] + [f'layers.{i}' for i in range(LAYERS)]
    return ([f'{state}/{pool}' for state in states for pool in POOLS]
            + [f'vision_tower.layers.{i}/mean' for i in VISUAL_LAYERS]
            + [f'audio_encoder.lfm.layers.{i}/mean' for i in AUDIO_LAYERS])


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


def clip_metadata(clip_seconds: float):
    """The metadata of ``FRAMES`` frames taken evenly from a clip of ``clip_seconds``: the processor reads the
    time stamps from it."""
    from transformers.video_utils import VideoMetadata
    return VideoMetadata(total_num_frames=FRAMES, fps=FRAMES / clip_seconds, duration=clip_seconds,
                         frames_indices=list(range(FRAMES)))


class Lfm3OmniExtractor:
    """The pooled language-model states (and four layers of each encoder) of each row, one row a pass.

    Limit: a pooled layer name (``layers.<i>/picture``) is a recording address only; an intervention cannot
    address it.
    Limit: a row has no context from the rows before it.
    Limit: one row a pass (the batch test of this model has not run).
    """

    independent_layers = True         # one pass gives every layer; a layer depends on that layer alone

    def __init__(self, identifier: str, model, processor, clip_seconds: float = 1.49) -> None:
        self.identifier = identifier
        self._model = model
        self._processor = processor
        self._clip_seconds = clip_seconds
        config = model.config
        self._video_token, self._audio_token = config.video_token_id, config.audio_token_id
        self._hooked = {}
        from brainscore.model_helpers.extractors import get_layer, hook
        for path in ([f'vision_tower.layers.{i}' for i in VISUAL_LAYERS]
                     + [f'audio_encoder.lfm.layers.{i}' for i in AUDIO_LAYERS]):
            hook(get_layer(model.model, path), path, self._keep(f'{path}/mean'))

    def _keep(self, name):
        def keep(output):
            hidden = output[0] if isinstance(output, tuple) else output
            self._hooked.setdefault(name, []).append(hidden.reshape(-1, hidden.shape[-1]).float())
        return keep

    def layer_names(self) -> list:
        return layer_names()

    def pooled(self, rows) -> list:
        """``[{layer path: vector}]`` for every layer of each row of ``rows`` (``(video_path, audio_path)``; an
        ``audio_path`` of None: the clip alone; a ``video_path`` of None: the sound alone), one row a pass."""
        return [self._row(video, audio) for video, audio in rows]

    def _row(self, video_path, audio_path) -> dict:
        import torch
        from brainscore.model_helpers.video_wrapper import default_uniform_frame_sampler
        call = {}
        if video_path is not None:
            call.update(videos=[np.stack(default_uniform_frame_sampler(video_path, FRAMES))],
                        video_metadata=[clip_metadata(self._clip_seconds)], do_sample_frames=False)
        if audio_path is not None:
            call['audio'] = [read_sound(audio_path)]
        if video_path is not None and audio_path is not None:
            call['use_audio_in_video'] = True
        text = HEAD + (CLIP if video_path is not None else '') + (SOUND if audio_path is not None else '') + TAIL
        inputs = self._processor(text=[text], return_tensors='pt', **call)
        ids = inputs['input_ids'][0]
        spans = {'picture': ids == self._video_token, 'sound': ids == self._audio_token}
        counts = {pool: int(mask.sum()) for pool, mask in spans.items()}
        if (video_path is not None and not counts['picture']) or (audio_path is not None and not counts['sound']):
            raise ValueError(f'{video_path}, {audio_path}: the processor gave {counts["picture"]} picture and '
                             f'{counts["sound"]} sound tokens; a row needs at least one of each input it has. '
                             'Check the clip and the sound files.')
        device = self._model.get_input_embeddings().weight.device
        inputs = {key: value.to(device) if hasattr(value, 'to') else value for key, value in inputs.items()
                  if key != 'video_metadata'}
        self._hooked.clear()
        with torch.no_grad():
            out = self._model(**inputs, output_hidden_states=True)
        states = out.hidden_states
        if len(states) != LAYERS + 1:
            raise ValueError(f'the language model gave {len(states)} hidden states; expected {LAYERS + 1}.')
        record = {name: torch.cat(parts).mean(0).cpu().numpy() for name, parts in self._hooked.items()}
        record.update({f'tokens/{pool}': np.array([float(count)], np.float32) for pool, count in counts.items()})
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
                             f'for example layers.12/picture (state embed or layers.0 to layers.{LAYERS - 1}, '
                             f'pool {"/".join(POOLS)}), or an encoder layer '
                             '<vision_tower.layers.i|audio_encoder.lfm.layers.i>/mean.')
        has_sound = 'audio_path' in stimuli.columns
        if not has_sound and any(layer.endswith('/sound') or layer.startswith('audio_') for layer in layers):
            raise ValueError(f'{[l for l in layers if l.endswith("/sound") or l.startswith("audio_")]} read the '
                             'sound, and the rows have no audio_path. Give the sound, or ask a picture layer.')
        has_clip = 'video_path' in stimuli.columns
        if not has_clip and any(layer.endswith('/picture') or layer.startswith('vision_') for layer in layers):
            raise ValueError(f'{[l for l in layers if l.endswith("/picture") or l.startswith("vision_")]} read '
                             'the clip, and the rows have no video_path. Give the clip, or ask a sound layer.')
        none = [None] * len(stimuli)
        rows = self.pooled(list(zip(stimuli['video_path'] if has_clip else none,
                                    stimuli['audio_path'] if has_sound else none)))
        values = np.concatenate([np.stack([row[layer] for row in rows]) for layer in layers], axis=1)
        neuroid = {}
        for layer in layers:
            for name, (_, coord) in make_layer_neuroid_coords(self.identifier, layer, len(rows[0][layer])).items():
                neuroid.setdefault(name, []).extend(coord)
        presentation = {column: ('presentation', list(stimuli[column])) for column in stimuli.columns}
        return make_assembly(values, coords={**presentation, **{n: ('neuroid', v) for n, v in neuroid.items()}},
                             dims=['presentation', 'neuroid'])


def answer_parts(model, processor, *, max_new_tokens: int = 64):
    """``reply`` and ``likelihood``: the likelihood of the shared chat parts, and a reply on the same inputs
    (``chat_inputs``) through the model's own ``generate``, which returns its sequences with their lengths."""
    from brainscore_core.responses import Text
    from brainscore.model_helpers.replies import chat_inputs, chat_parts, generation_settings
    _, likelihood = chat_parts(model, processor, max_new_tokens=max_new_tokens)

    def reply(prompt) -> str:
        import torch
        inputs = chat_inputs(model, processor, prompt)
        settings = generation_settings(prompt, max_new_tokens)
        if isinstance(prompt.response, Text) and prompt.response.seed is not None:
            torch.manual_seed(prompt.response.seed)
        with torch.no_grad():
            out = model.generate(**inputs, **settings, generation_mode='text')
        start = inputs['input_ids'].shape[1]
        return processor.decode(out.sequences[0, start:int(out.sequence_lengths[0])],
                                skip_special_tokens=True).strip()

    return reply, likelihood


def build_model(identifier: str, extractor: Lfm3OmniExtractor, reply, likelihood) -> BrainScoreModel:
    """The subject: one extractor for the clip and the sound, and the written answers.

    The extractor goes under the tuple key ``('video', 'audio')``: a session with both inputs runs it once for all
    regions, and each call gets the whole row. No input is required: the model takes text alone, a picture alone,
    a clip alone or a sound alone.

    Limit: no speech output: the catalog has no speech output channel, and the answers ask for text only.
    Not wired yet: a written answer with sound. The shared chat parts refuse a sound part at the first trial that
    shows one, before the forward pass; a benchmark that sends only sound and asks for an answer is N/A.
    Not wired yet: a picture alone to the regions. The layers are chosen on films; a picture map needs its own
    selection. A benchmark that sends only text or pictures and asks for a region is N/A.
    Not wired yet: the ``language_system`` region. No layer was chosen on language data.
    Not wired yet: the intervention part. The pooled names are not module addresses.
    Not wired yet: screen acts. The coordinate grid of this model is not measured.
    """
    return BrainScoreModel(
        identifier=identifier, region_layer_map=REGION_LAYER_MAP, extractor={_BOTH: extractor},
        in_channels=('text', 'vision:image', 'instruction'), reply=reply, likelihood=likelihood,
        answer_owners={'behavior:probabilities': ('likelihood', 'softmax')},
        output_inputs={output: ANSWER_INPUTS for output in (
            'behavior:choice', 'behavior:rating', 'behavior:text', 'behavior:log_probabilities',
            'behavior:probabilities', 'behavior:rt')})


def get_model(identifier: str) -> BrainScoreModel:
    if identifier != IDENTIFIER:
        raise ValueError(f'{identifier!r} is not registered here; this folder registers {IDENTIFIER!r}.')
    import torch
    import transformers
    version = tuple(int(part) for part in transformers.__version__.split('.')[:3] if part.isdigit())
    if version < TRANSFORMERS:
        raise ValueError(f'{IDENTIFIER} needs transformers {".".join(map(str, TRANSFORMERS))} or later, and this '
                         f'environment has {transformers.__version__}: the model code uses its newer processor '
                         'parts (README, "Environment"). Run it in the clone with the newer transformers.')
    from transformers import AutoModelForMultimodalLM, AutoProcessor
    from brainscore.model_helpers.provenance import ProvenanceSubject
    processor = AutoProcessor.from_pretrained(WEIGHTS, revision=REVISION, trust_remote_code=True)
    model = AutoModelForMultimodalLM.from_pretrained(
        WEIGHTS, revision=REVISION, trust_remote_code=True, dtype=torch.bfloat16,
        device_map='cuda' if torch.cuda.is_available() else 'cpu').eval()
    subject = build_model(identifier, Lfm3OmniExtractor(identifier, model, processor),
                          *answer_parts(model, processor))
    return ProvenanceSubject(subject, {
        'identifier': identifier, 'checkpoint': WEIGHTS, 'revision': REVISION,
        'licence': 'LFM Open License v1.0 (research use)',
        'remote_code': f'{WEIGHTS}@{REVISION} (modeling, processing, configuration, codec)',
        'dtype': 'bfloat16', 'torch': torch.__version__, 'transformers': transformers.__version__,
        'frames_per_clip': FRAMES, 'frame_sampling': 'decoded frames with metadata, no sampling in the processor',
        'batch': 1, 'template': HEAD + CLIP + SOUND + TAIL, 'use_audio_in_video': 'when the row has both',
        'pools': list(POOLS), 'encoder_layers': {'vision': list(VISUAL_LAYERS), 'audio': list(AUDIO_LAYERS)},
        'answers': 'shared chat parts for the likelihood; own generate for the reply; greedy unless the task '
                   'states settings; empty thinking block from the template'})
