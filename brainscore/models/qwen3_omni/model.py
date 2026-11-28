"""Qwen3-Omni-30B-A3B-Thinking (thinker only): one forward pass on the clip and the sound of a row together, and
written answers to text, pictures and clips.

The thinker of ``Qwen/Qwen3-Omni-30B-A3B-Thinking`` takes the picture and the sound of a row as one interleaved
token sequence (``use_audio_in_video=True``: the picture and sound tokens are put in time order). The extractor
records, in that one pass, every thinker state (``embed`` and ``layers.0`` to ``layers.47``, width 2048), each
pooled three ways and kept apart: the mean over the picture tokens (``/picture``), the mean over the sound tokens
(``/sound``) and the last token (``/last``). It also records four blocks of the vision encoder
(``visual.blocks.<i>/mean``, width 1152) and four layers of the audio encoder (``audio_tower.layers.<i>/mean``,
width 1280), each the mean over its tokens.

Input of a row: the clip (``video_path``) and, when the benchmark gives it, the sound (``audio_path``, a WAV at
16 kHz, mono). ``FRAMES`` frames are taken evenly from the clip and given with ``fps = FRAMES / clip seconds``,
so the time positions of the picture tokens match the sound. Without a sound column, the row is the clip alone;
without a clip column, the row is the sound alone (``SOUND_TEMPLATE``; only the sound layers can be read).
Text of a row (``TEMPLATE``): the model's own chat template for one user turn that holds the clip, followed by the
start of the assistant turn; ``/last`` is the last token of that start (the newline after ``assistant``).

Written answers (``reply``, ``likelihood``): the shared chat parts on the thinker, with thinking switched off
(``CHAT_TEMPLATE``).

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
BATCH = 8                     # rows per forward pass (the expert layers run once per pass: 0.39 s a row, 2.6 alone)
SOUND_RATE_HZ = 16000
THINKER_LAYERS = 48
POOLS = ('picture', 'sound', 'last')
VISUAL_BLOCKS = (0, 8, 17, 26)        # 4 evenly spaced of the 27 vision-encoder blocks
AUDIO_LAYERS = (0, 10, 21, 31)        # 4 evenly spaced of the 32 audio-encoder layers
TEMPLATE = ('<|im_start|>user\n<|vision_start|><|video_pad|><|vision_end|><|im_end|>\n'
            '<|im_start|>assistant\n')
SOUND_TEMPLATE = ('<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n'
                  '<|im_start|>assistant\n')    # a row with a sound and no clip
# The written answers: the model's chat template with thinking switched off (its template writes an empty
# reasoning block), so a reply is the answer itself.
CHAT_TEMPLATE = {'enable_thinking': False}
# The inputs a written answer can show (the shared chat parts: text, pictures, clips).
ANSWER_INPUTS = ('text', 'vision:image', 'vision:video')

# The committed map (README, "Layer rule"), chosen on the selection films friends_s01e06a to s01e10b (median r over
# parcels, one band each: layers.14/last 0.168, layers.26/last 0.169, layers.32/last 0.160, layers.23/sound 0.163).
# V1, V4 and IT are recorded from the clip; with sound, the sound reaches their tokens too (the picture and sound
# tokens are interleaved and the last token follows both), so they state both inputs (tuple modality). A1 reads the
# sound tokens: a session without sound that asks A1 is N/A in pre-flight. The regions of the clip come first, so
# the memory probe (a clip without sound) records one of them.
REGION_LAYER_MAP = {
    'V1': LayerSelector('layers.14/last', modality=('video', 'audio')),
    'V4': LayerSelector('layers.26/last', modality=('video', 'audio')),
    'IT': LayerSelector('layers.32/last', modality=('video', 'audio')),
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


def _audio_per_row(tower) -> None:
    """Make ``tower`` (the audio encoder) encode the sounds of a packed batch one at a time, then join them."""
    forward = tower.forward

    def per_row(input_features, feature_lens=None, aftercnn_lens=None):
        if feature_lens is None or len(feature_lens) < 2:
            return forward(input_features, feature_lens=feature_lens)
        from transformers.modeling_outputs import BaseModelOutput
        import torch
        parts, start = [], 0
        for i, length in enumerate(feature_lens.tolist()):
            parts.append(forward(input_features[:, start:start + length],
                                 feature_lens=feature_lens[i:i + 1]).last_hidden_state)
            start += length
        return BaseModelOutput(last_hidden_state=torch.cat(parts))
    tower.forward = per_row


class OmniThinkerExtractor:
    """The pooled thinker states (and four encoder blocks of each tower) of each row, from one forward pass.

    Limit: a pooled layer name (``layers.<i>/picture``) is a recording address only; an intervention cannot
    address it.
    Limit: a row has no context from the rows before it.
    Limit: rows run in batches of ``BATCH`` in table order (a batch needs rows with one token count; otherwise its
    rows run one by one). The audio encoder runs one row at a time inside a batch (``_audio_per_row``): in
    transformers 4.57 its attention keeps the packed rows of a batch apart only under FlashAttention 2, so under the
    default attention the sounds of a batch would attend to each other. The thinker values of a row still differ
    slightly with its batch (bf16 kernels): per-unit r over 24 Friends scans, layers.23, picture 0.999, sound 0.995,
    last 0.980 (medians). The same run twice is identical.
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
        _audio_per_row(model.thinker.audio_tower)
        from brainscore.model_helpers.extractors import get_layer, hook
        for path in [f'visual.blocks.{i}' for i in VISUAL_BLOCKS] + [f'audio_tower.layers.{i}' for i in AUDIO_LAYERS]:
            hook(get_layer(model.thinker, path), path, self._keep(f'{path}/mean'))

    def _keep(self, name):
        def keep(output):
            hidden = output[0] if isinstance(output, tuple) else output
            self._hooked.setdefault(name, []).append(hidden.reshape(-1, hidden.shape[-1]).float())
        return keep

    def layer_names(self) -> list:
        return layer_names()

    def pooled(self, rows) -> list:
        """``[{layer path: vector}]`` for every layer of each row of ``rows`` (``(video_path, audio_path)``; an
        ``audio_path`` of None: the clip alone), in batches of ``BATCH``."""
        records = []
        for start in range(0, len(rows), BATCH):
            records += self._batch(rows[start:start + BATCH])
        return records

    def _batch(self, rows) -> list:
        import torch
        from brainscore.model_helpers.video_wrapper import default_uniform_frame_sampler
        with_clip = rows[0][0] is not None
        frames = [np.stack(default_uniform_frame_sampler(video, FRAMES)) for video, _ in rows] if with_clip else None
        sounds = [None if audio is None else read_sound(audio) for _, audio in rows]
        with_sound = sounds[0] is not None
        if any((s is not None) != with_sound for s in sounds):
            raise ValueError('a batch mixes rows with and without sound; give every row a sound or none.')
        together = with_sound and with_clip
        inputs = self._processor(text=[TEMPLATE if with_clip else SOUND_TEMPLATE] * len(rows), videos=frames,
                                 audio=sounds if with_sound else None, use_audio_in_video=together,
                                 fps=FRAMES / self._clip_seconds, return_tensors='pt')
        if 'attention_mask' in inputs and not bool(inputs['attention_mask'].all()) and len(rows) > 1:
            return [record for row in rows for record in self._batch([row])]     # rows of different lengths
        ids = inputs['input_ids']
        spans = {'picture': ids == self._video_token, 'sound': ids == self._audio_token}
        counts = {pool: mask.sum(1) for pool, mask in spans.items()}
        if (with_clip and (counts['picture'] == 0).any()) or (with_sound and (counts['sound'] == 0).any()):
            raise ValueError(f'{[v for v, _ in rows]}: the processor gave {counts["picture"].tolist()} picture and '
                             f'{counts["sound"].tolist()} sound tokens; a row needs at least one of each input it '
                             'has. Check the clip and the sound files.')
        device = self._model.thinker.get_input_embeddings().weight.device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        self._hooked.clear()
        with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            out = self._model.thinker(**inputs, output_hidden_states=True, use_audio_in_video=together)
        states = out.hidden_states
        if len(states) != THINKER_LAYERS + 1:
            raise ValueError(f'the thinker gave {len(states)} hidden states; expected {THINKER_LAYERS + 1}.')
        records = []
        for b in range(len(rows)):
            # the encoders pack the rows of a batch one after the other; rows of a batch have equal lengths
            record = {}
            for name, parts in self._hooked.items():
                hidden = torch.cat(parts)
                record[name] = hidden.reshape(len(rows), -1, hidden.shape[-1])[b].mean(0).cpu().numpy()
            record.update({f'tokens/{pool}': np.array([float(count[b])], np.float32) for pool, count in counts.items()})
            for index, state in enumerate(states):
                name = 'embed' if index == 0 else f'layers.{index - 1}'
                hidden = state[b].float()
                for pool, mask in spans.items():
                    if mask[b].any():
                        record[f'{name}/{pool}'] = hidden[mask[b].to(hidden.device)].mean(0).cpu().numpy()
                record[f'{name}/last'] = hidden[-1].cpu().numpy()
            records.append(record)
        return records

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
        has_clip = 'video_path' in stimuli.columns
        if not has_clip and any(layer.endswith('/picture') or layer.startswith('visual.') for layer in layers):
            raise ValueError(f'{[l for l in layers if l.endswith("/picture") or l.startswith("visual.")]} read '
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


def build_model(identifier: str, extractor: OmniThinkerExtractor, reply, likelihood) -> BrainScoreModel:
    """The subject: one extractor for the clip and the sound, and the written answers of the thinker.

    The extractor goes under the tuple key ``('video', 'audio')``: a session with both inputs runs it once for all
    regions, and each call gets the whole row, so the sound reaches the picture tokens too. A row with a sound and
    no clip runs the sound alone (``A1``). ``reply`` and ``likelihood`` are the shared chat parts
    (``chat_parts``) on the thinker: text, pictures and clips in, with the task's instruction as the system turn;
    choices, ratings and texts out, and the log
    probability of each option. No input is required: the model takes text alone, a picture alone, a clip alone
    or a sound alone.

    Limit: the regions are recorded from the clip and the sound (the layers were chosen on films with sound); a
    benchmark that sends only text or pictures and asks for a region is N/A.
    Limit: a written answer takes no sound: the shared chat parts refuse a sound part at the first trial that
    shows one, before the forward pass; a benchmark that sends only sound and asks for an answer is N/A.
    Limit: in a written answer, the time positions of a clip's picture tokens use the processor's default of one
    frame a second, not the clip's rate (the shared chat parts give the rate as metadata, which this processor
    does not read).
    Limit: no speech output: this checkpoint has no talker (``enable_audio_output`` is false in its config).
    Limit: no ``language_system`` region: no layer of the thinker was chosen on language data.
    Limit: no screen acts: the act form of the study models reads positions on the 0-1000 grid measured for
    Qwen3-VL; the grid of this model is not measured.
    Limit: no intervention part is wired.
    """
    return BrainScoreModel(
        identifier=identifier, region_layer_map=REGION_LAYER_MAP, extractor={('video', 'audio'): extractor},
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
    from transformers import AutoProcessor, Qwen3OmniMoeForConditionalGeneration
    from brainscore.model_helpers.provenance import ProvenanceSubject
    from brainscore.model_helpers.replies import chat_parts
    processor = AutoProcessor.from_pretrained(WEIGHTS, revision=REVISION)
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        WEIGHTS, revision=REVISION, dtype=torch.bfloat16, device_map='auto').eval()
    subject = build_model(identifier, OmniThinkerExtractor(identifier, model, processor),
                          *chat_parts(model.thinker, processor, chat_template_kwargs=CHAT_TEMPLATE))
    return ProvenanceSubject(subject, {
        'identifier': identifier, 'checkpoint': WEIGHTS, 'revision': getattr(model.config, '_commit_hash', None),
        'dtype': 'bfloat16', 'device_map': 'auto', 'torch': torch.__version__,
        'transformers': transformers.__version__, 'frames_per_clip': FRAMES, 'template': TEMPLATE,
        'use_audio_in_video': 'when the row has a clip and a sound', 'pools': list(POOLS),
        'answers': f'chat template with {CHAT_TEMPLATE}, greedy unless the task states settings'})
