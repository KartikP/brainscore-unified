"""Qwen2.5-Omni-7B and -3B (thinker only): one forward pass on the clip and the sound of a row together, and written
answers to text, pictures and clips.

The thinker of ``Qwen/Qwen2.5-Omni-7B`` or ``Qwen/Qwen2.5-Omni-3B`` (``Qwen2_5OmniThinkerForConditionalGeneration``:
no talker, no speaker file) takes the picture and the sound of a row as one token sequence (``use_audio_in_video=True``). A scan of
1.49 s is shorter than one 2 s chunk of the processor, so all picture tokens come first, then all sound tokens;
their time positions (25 a second) overlap. The extractor records, in that one pass, every thinker state
(``embed`` and ``layers.0`` to the last layer: 28 layers of width 3584 in the 7B, 36 of width 2048 in the 3B), each pooled three ways and kept apart: the mean over
the picture tokens (``/picture``), the mean over the sound tokens (``/sound``) and the last token (``/last``). It
also records four blocks of the vision encoder (``visual.blocks.<i>/mean``) and four layers of the audio encoder
(``audio_tower.layers.<i>/mean``), each the mean over its tokens.

Input of a row: the clip (``video_path``) and, when the benchmark gives it, the sound (``audio_path``, a WAV at
16 kHz, mono). ``FRAMES`` frames are taken evenly from the clip and given with ``fps = FRAMES / clip seconds``,
so the time positions of the picture tokens match the sound. Without a sound column, the row is the clip alone;
without a clip column, the row is the sound alone (``SOUND_TEMPLATE``; only the sound layers can be read).
Text of a row (``TEMPLATE``): one user turn that holds the clip, with no system turn (the model's template adds
"You are a helpful assistant." when none is given; the recording leaves it out), then the start of the assistant
turn; ``/last`` is the last token of that start (the newline after ``assistant``). The picture size is the
processor's own limit (``PICTURE_LIMIT``).

Written answers (``reply``, ``likelihood``): the shared chat parts on the thinker. This model has no thinking mode.

The region map of each identifier (``MODELS``) is a placeholder until the layer rule of the README runs on the selection
films; nothing is searched at run time.
"""
import numpy as np

from brainscore import BrainScoreModel
from brainscore.selection import LayerSelector

FRAMES = 8                    # frames taken evenly from a clip; 2 temporal patches each give 4 picture groups
BATCH = 8                     # rows per forward pass
SOUND_RATE_HZ = 16000
POOLS = ('picture', 'sound', 'last')
VISUAL_BLOCKS = (0, 10, 21, 31)       # 4 evenly spaced of the 32 vision-encoder blocks (both sizes), first and last
AUDIO_LAYERS = (0, 10, 21, 31)        # 4 evenly spaced of the 32 audio-encoder layers (both sizes), first and last
PICTURE_LIMIT = 'the processor default: 128 x 28 x 28 to 768 x 28 x 28 pixels a frame'
TEMPLATE = ('<|im_start|>user\n<|vision_bos|><|VIDEO|><|vision_eos|><|im_end|>\n'
            '<|im_start|>assistant\n')
SOUND_TEMPLATE = ('<|im_start|>user\n<|audio_bos|><|AUDIO|><|audio_eos|><|im_end|>\n'
                  '<|im_start|>assistant\n')    # a row with a sound and no clip
# The inputs a written answer can show (the shared chat parts: text, pictures, clips).
ANSWER_INPUTS = ('text', 'vision:image', 'vision:video')



def _placeholder_map(layers: int) -> dict:
    """The middle layer of each band of the rule (README, "Layer rule"), until the rule runs on the selection
    films friends_s01e06a to s01e10b.

    V1, V4 and IT are recorded from the clip; with sound, the sound reaches their tokens too (the last token
    follows both), so they state both inputs (tuple modality). A1 is a thinker layer after the point where the
    inputs meet, so it states both inputs too (rule 2.5); its layer reads the sound tokens, so a request for A1
    without sound is refused before any pass. The regions of the clip come first, so the memory probe (a clip
    without sound) records one of them."""
    third = -(-layers // 3)                       # V1: 0 .. ceil(N/3)-1, V4: .. ceil(2N/3)-1, IT: the rest
    two_thirds = -(-2 * layers // 3)
    both = ('video', 'audio')
    return {
        'V1': LayerSelector(f'layers.{third // 2}/last', modality=both),
        'V4': LayerSelector(f'layers.{(third + two_thirds) // 2}/last', modality=both),
        'IT': LayerSelector(f'layers.{(two_thirds + layers) // 2}/last', modality=both),
        'A1': LayerSelector(f'layers.{layers // 2}/sound', modality=both),
    }


# Each identifier: weights, the pinned revision, the licence, the thinker depth and the region map.
MODELS = {
    'qwen2.5-omni-7b': {'weights': 'Qwen/Qwen2.5-Omni-7B', 'revision': 'ae9e1690543ffd5c0221dc27f79834d0294cba00',
                        'licence': 'Apache-2.0', 'layers': 28, 'region_layer_map': _placeholder_map(28)},
    'qwen2.5-omni-3b': {'weights': 'Qwen/Qwen2.5-Omni-3B', 'revision': 'f75b40e3da2003cdd6e1829b1f420ca70797c34e',
                        'licence': 'Qwen Research License (research use; not for commercial use)', 'layers': 36,
                        'region_layer_map': _placeholder_map(36)},
}


def layer_names(layers: int) -> list:
    """Every layer path the extractor records for a thinker of ``layers`` layers, thinker states first."""
    states = ['embed'] + [f'layers.{i}' for i in range(layers)]
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
    """The pooled thinker states (and four encoder blocks of each encoder) of each row, from one forward pass.

    Limit: a pooled layer name (``layers.<i>/picture``) is a recording address only; an intervention cannot
    address it.
    Limit: a row has no context from the rows before it.
    Limit: rows run in batches of ``BATCH`` in table order (a batch needs rows with one token count; otherwise its
    rows run one by one). The thinker values of a row can differ slightly with its batch (bf16 kernels); the
    difference is measured before a scored pass and stated in the README.
    """

    independent_layers = True         # one pass gives every layer; a layer depends on that layer alone

    def __init__(self, identifier: str, model, processor, clip_seconds: float = 1.49) -> None:
        self.identifier = identifier
        self._model = model
        self._processor = processor
        self._clip_seconds = clip_seconds
        config = model.config.thinker_config
        self._layers = config.text_config.num_hidden_layers
        self._video_token, self._audio_token = config.video_token_index, config.audio_token_index
        self._hooked = {}
        from brainscore.model_helpers.extractors import get_layer, hook
        for path in [f'visual.blocks.{i}' for i in VISUAL_BLOCKS] + [f'audio_tower.layers.{i}' for i in AUDIO_LAYERS]:
            hook(get_layer(model.thinker, path), path, self._keep(f'{path}/mean'))

    def _keep(self, name):
        def keep(output):
            hidden = output[0] if isinstance(output, tuple) else output
            self._hooked.setdefault(name, []).append(hidden.reshape(-1, hidden.shape[-1]).float())
        return keep

    def layer_names(self) -> list:
        return layer_names(self._layers)

    def pooled(self, rows) -> list:
        """``[{layer path: vector}]`` for every layer of each row of ``rows`` (``(video_path, audio_path)``; an
        ``audio_path`` of None: the clip alone; a ``video_path`` of None: the sound alone), in batches of
        ``BATCH``."""
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
        inputs = {key: value.to(device) if hasattr(value, 'to') else value for key, value in inputs.items()}
        self._hooked.clear()
        with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            out = self._model.thinker(**inputs, output_hidden_states=True, use_audio_in_video=together)
        states = out.hidden_states
        if len(states) != self._layers + 1:
            raise ValueError(f'the thinker gave {len(states)} hidden states; expected {self._layers + 1}.')
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
        unknown = sorted(set(layers) - set(self.layer_names()))
        if unknown:
            raise ValueError(f'{unknown} are not layers of {self.identifier}; a layer path is <state>/<pool>, '
                             f'for example layers.14/picture (state embed or layers.0 to '
                             f'layers.{self._layers - 1}, pool {"/".join(POOLS)}), or an encoder block '
                             '<visual.blocks.i|audio_tower.layers.i>/mean.')
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
    regions, and each call gets the whole row, so the sound reaches the picture tokens too. ``reply`` and
    ``likelihood`` are the shared chat parts (``chat_parts``) on the thinker: text, pictures and clips in, with
    the task's instruction as the system turn; choices, ratings and texts out, and the log probability of each
    option. No input is required: the model takes text alone, a picture alone, a clip alone or a sound alone.

    Limit: no speech output: the thinker class has no talker, and the catalog has no speech output channel.
    Not wired yet: a written answer with sound. The shared chat parts refuse a sound part at the first trial that
    shows one, before the forward pass; a benchmark that sends only sound and asks for an answer is N/A. This is
    work on the shared chat parts.
    Not wired yet: a picture alone to the regions. The layers are chosen on films; a picture map needs its own
    selection. A benchmark that sends only text or pictures and asks for a region is N/A.
    Not wired yet: the ``language_system`` region. No layer was chosen on language data; a later step applies
    the same rule on a language benchmark.
    Not wired yet: the intervention part. Hooks on module paths would work, but the pooled names are not module
    addresses.
    Not wired yet: screen acts. The coordinate grid of this model is not measured.
    Limit: in a written answer, the time positions of a clip's picture tokens use the processor's default rate,
    not the clip's rate (the shared chat parts give the rate as metadata, which this processor does not read).
    """
    return BrainScoreModel(
        identifier=identifier, region_layer_map=MODELS[identifier]['region_layer_map'], extractor={('video', 'audio'): extractor},
        in_channels=('text', 'vision:image', 'instruction'), reply=reply, likelihood=likelihood,
        answer_owners={'behavior:probabilities': ('likelihood', 'softmax')},
        output_inputs={output: ANSWER_INPUTS for output in (
            'behavior:choice', 'behavior:rating', 'behavior:text', 'behavior:log_probabilities',
            'behavior:probabilities', 'behavior:rt')})


def get_model(identifier: str) -> BrainScoreModel:
    if identifier not in MODELS:
        raise ValueError(f'{identifier!r} is not registered here; this folder registers {sorted(MODELS)}.')
    spec = MODELS[identifier]
    import torch
    import transformers
    from transformers import AutoProcessor, Qwen2_5OmniThinkerForConditionalGeneration
    from types import SimpleNamespace
    from brainscore.model_helpers.provenance import ProvenanceSubject
    from brainscore.model_helpers.replies import chat_parts
    processor = AutoProcessor.from_pretrained(spec['weights'], revision=spec['revision'])
    thinker = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        spec['weights'], revision=spec['revision'], dtype=torch.bfloat16, device_map='auto').eval()
    model = SimpleNamespace(thinker=thinker, config=SimpleNamespace(thinker_config=thinker.config))
    subject = build_model(identifier, OmniThinkerExtractor(identifier, model, processor),
                          *chat_parts(thinker, processor))
    return ProvenanceSubject(subject, {
        'identifier': identifier, 'checkpoint': spec['weights'], 'revision': spec['revision'],
        'licence': spec['licence'],
        'class': 'Qwen2_5OmniThinkerForConditionalGeneration (thinker only)',
        'dtype': 'bfloat16', 'device_map': 'auto', 'torch': torch.__version__,
        'transformers': transformers.__version__, 'frames_per_clip': FRAMES, 'batch': BATCH,
        'picture_limit': PICTURE_LIMIT, 'template': TEMPLATE, 'sound_template': SOUND_TEMPLATE,
        'use_audio_in_video': 'when the row has a clip and a sound', 'pools': list(POOLS),
        'encoder_blocks': {'visual': list(VISUAL_BLOCKS), 'audio': list(AUDIO_LAYERS)},
        'answers': 'shared chat parts, model chat template (no thinking mode), greedy unless the task states '
                   'settings'})
