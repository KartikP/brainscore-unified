"""MiniCPM-o 4.5 (no speech part): one forward pass on the clip and the sound of a row together, and written
answers to text, pictures and clips.

``openbmb/MiniCPM-o-4_5`` runs its own code (``trust_remote_code``, pinned by ``REVISION``): a SigLIP vision
encoder (27 blocks) with a resampler, a Whisper audio encoder (24 layers) and a Qwen3 language model (36 layers,
width 4096). It reads a film as units of one second: one frame, then the sound of that second. A scan of 1.49 s
is two units: the frame at 0 s and the sound from 0 to 1 s, then the frame at 1 s and the sound from 1 to 1.49 s.
The extractor records, in that one pass, every state of the language model (``embed`` and ``layers.0`` to
``layers.35``), each pooled three ways and kept apart: the mean over the picture tokens (``/picture``), the mean
over the sound tokens (``/sound``) and the last token (``/last``). It also records four blocks of the vision
encoder (``vpm.encoder.layers.<i>/mean``) and four layers of the audio encoder (``apm.layers.<i>/mean``), each the
mean over its tokens.

Input of a row: the clip (``video_path``) and, when the benchmark gives it, the sound (``audio_path``, a WAV at
16 kHz, mono). Without a sound column, the units hold the frames alone; without a clip column, the sound alone.
Each frame is one slice (``max_slice_nums=1``). Text of a row: one user turn that holds the units, with no system
turn, then the start of the assistant turn with thinking off (``<think>\\n\\n</think>\\n\\n``); ``/last`` is the
last token of that start.

The model's forward is called as its own code calls it (``forward(data)``), one row at a time. Its config sets
``stream_input``, which allows one row with sound per pass; and with that switch off, its code fills the sound of
the last row of a batch only. So rows never run in batches here.

Written answers (``reply``, ``likelihood``): the model's own input path (``answer_parts``), greedy, with thinking
off, with the task's instruction as the system turn.

Environment: the model code runs on transformers 4.51.0 (``TRANSFORMERS``), the version its card pins, in a
clone of the coordinated environment (README, "Environment"); on 4.57 its sound path and its generation step
fail. The load refuses another version.

The region map (``REGION_LAYER_MAP``) is a placeholder until the layer rule of the README runs on the selection
films; nothing is searched at run time.
"""
import math

import numpy as np

from brainscore import BrainScoreModel
from brainscore.selection import LayerSelector

IDENTIFIER = 'minicpm-o-4.5'
WEIGHTS = 'openbmb/MiniCPM-o-4_5'
REVISION = 'fb09628e98b1f55d8d5b863e0eccfcdd11bff499'   # Apache-2.0, not gated; its own code at this revision
TRANSFORMERS = '4.51.0'        # the version the model code runs on (README, "Environment")
SOUND_RATE_HZ = 16000
UNIT_SECONDS = 1.0            # the model's unit: one frame, then the sound of that second
LAYERS = 36
POOLS = ('picture', 'sound', 'last')
VISUAL_BLOCKS = (0, 9, 17, 26)        # 4 evenly spaced of the 27 SigLIP blocks, first and last included
AUDIO_LAYERS = (0, 8, 15, 23)         # 4 evenly spaced of the 24 Whisper layers, first and last included
ANSWER_FRAMES = 8             # frames of a clip shown in a written answer, each one slice
PICTURE = '<image>./</image>'
SOUND = '<audio>./</audio>'
HEAD = '<|im_start|>user\n'
TAIL = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'     # thinking off, as its template writes it
TERMINATORS = ('<|im_end|>', '<|endoftext|>')
ANSWER_INPUTS = ('text', 'vision:image', 'vision:video')

# Placeholder map: the middle layer of each band of the rule (README, "Layer rule"), until the rule runs on the
# selection films friends_s01e06a to s01e10b. Every region is a layer of the language model, after the point
# where the inputs meet, so it states both inputs (rule 2.5). A1 reads the sound tokens: a request for A1
# without sound is refused before any pass. The regions of the clip come first, so the memory probe (a clip
# without sound) records one of them.
_BOTH = ('video', 'audio')
REGION_LAYER_MAP = {
    'V1': LayerSelector('layers.6/last', modality=_BOTH),
    'V4': LayerSelector('layers.18/last', modality=_BOTH),
    'IT': LayerSelector('layers.30/last', modality=_BOTH),
    'A1': LayerSelector('layers.18/sound', modality=_BOTH),
}


def layer_names() -> list:
    """Every layer path the extractor records, language-model states first."""
    states = ['embed'] + [f'layers.{i}' for i in range(LAYERS)]
    return ([f'{state}/{pool}' for state in states for pool in POOLS]
            + [f'vpm.encoder.layers.{i}/mean' for i in VISUAL_BLOCKS]
            + [f'apm.layers.{i}/mean' for i in AUDIO_LAYERS])


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


def frames_at(path, seconds) -> list:
    """The RGB frames of the clip at ``path`` nearest to each time in ``seconds`` (the last frame at the most)."""
    import cv2
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise IOError(f'cv2 could not open {path}')
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    for second in seconds:
        capture.set(cv2.CAP_PROP_POS_FRAMES, min(int(round(second * fps)), total - 1))
        ok, frame = capture.read()
        if not ok:
            raise IOError(f'{path}: no frame at {second} s')
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    return frames


def clip_seconds(path) -> float:
    import cv2
    capture = cv2.VideoCapture(str(path))
    seconds = capture.get(cv2.CAP_PROP_FRAME_COUNT) / (capture.get(cv2.CAP_PROP_FPS) or 30.0)
    capture.release()
    return seconds


def units(video_path, audio_path):
    """The text, pictures and sounds of one row: one unit a second (a frame, then the sound of that second)."""
    from PIL import Image
    sound = None if audio_path is None else read_sound(audio_path)
    seconds = len(sound) / SOUND_RATE_HZ if sound is not None else clip_seconds(video_path)
    count = max(1, math.ceil(seconds / UNIT_SECONDS - 1e-6))
    starts = [k * UNIT_SECONDS for k in range(count)]
    pictures = ([Image.fromarray(frame) for frame in frames_at(video_path, starts)]
                if video_path is not None else [])
    step = int(UNIT_SECONDS * SOUND_RATE_HZ)
    sounds = [sound[k * step:(k + 1) * step] for k in range(count)] if sound is not None else []
    text = HEAD + ''.join((PICTURE if pictures else '') + (SOUND if sounds else '') for _ in starts) + TAIL
    return text, pictures, sounds


def _spans(bounds, length):
    """A mask over ``length`` positions that is true inside each ``[start, end)`` of ``bounds``."""
    import torch
    mask = torch.zeros(length, dtype=torch.bool)
    for start, end in (bounds.tolist() if hasattr(bounds, 'tolist') else bounds):
        mask[int(start):int(end)] = True
    return mask


class MiniCPMOExtractor:
    """The pooled language-model states (and four blocks of each encoder) of each row, one row a pass.

    Limit: a pooled layer name (``layers.<i>/picture``) is a recording address only; an intervention cannot
    address it.
    Limit: a row has no context from the rows before it.
    Limit: a scan is units of one second, each with one frame: a scan of 1.49 s shows 2 frames.
    Limit: one row a pass (the model's own code allows one row with sound a pass).
    """

    independent_layers = True         # one pass gives every layer; a layer depends on that layer alone

    def __init__(self, identifier: str, model, processor) -> None:
        self.identifier = identifier
        self._model = model
        self._processor = processor
        self._hooked = {}
        from brainscore.model_helpers.extractors import get_layer, hook
        for path in ([f'vpm.encoder.layers.{i}' for i in VISUAL_BLOCKS]
                     + [f'apm.layers.{i}' for i in AUDIO_LAYERS]):
            hook(get_layer(model, path), path, self._keep(f'{path}/mean'))

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
        text, pictures, sounds = units(video_path, audio_path)
        inputs = self._processor([text], [pictures] if pictures else None, [sounds] if sounds else None,
                                 [[0] * len(sounds)] if sounds else None, max_slice_nums=1, return_tensors='pt')
        ids = inputs['input_ids']
        length = ids.shape[1]
        spans = {'picture': _spans(inputs['image_bound'][0], length),
                 'sound': _spans(inputs['audio_bounds'][0], length)}
        counts = {pool: int(mask.sum()) for pool, mask in spans.items()}
        if (pictures and not counts['picture']) or (sounds and not counts['sound']):
            raise ValueError(f'{video_path}, {audio_path}: the processor gave {counts["picture"]} picture and '
                             f'{counts["sound"]} sound tokens; a row needs at least one of each input it has. '
                             'Check the clip and the sound files.')
        device = self._model.llm.get_input_embeddings().weight.device
        data = inputs.to(device=device)
        data['position_ids'] = torch.arange(length, device=device)[None]
        self._hooked.clear()
        with torch.no_grad():
            out = self._model(data, output_hidden_states=True)
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
                             f'for example layers.18/picture (state embed or layers.0 to layers.{LAYERS - 1}, '
                             f'pool {"/".join(POOLS)}), or an encoder block <vpm.encoder.layers.i|apm.layers.i>'
                             '/mean.')
        has_sound = 'audio_path' in stimuli.columns
        if not has_sound and any(layer.endswith('/sound') or layer.startswith('apm.') for layer in layers):
            raise ValueError(f'{[l for l in layers if l.endswith("/sound") or l.startswith("apm.")]} read the '
                             'sound, and the rows have no audio_path. Give the sound, or ask a picture layer.')
        has_clip = 'video_path' in stimuli.columns
        if not has_clip and any(layer.endswith('/picture') or layer.startswith('vpm.') for layer in layers):
            raise ValueError(f'{[l for l in layers if l.endswith("/picture") or l.startswith("vpm.")]} read the '
                             'clip, and the rows have no video_path. Give the clip, or ask a sound layer.')
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
    """``reply`` and ``likelihood`` on the model's own input path (its processor, its picture and sound
    embedding), as ``chat_parts`` gives them for a standard chat model.

    The task's instruction (the text part named ``instruction``) is the system turn; the other parts are one user
    turn in their order: text as text, a picture as one picture, a clip as ``ANSWER_FRAMES`` frames taken evenly,
    each one slice. The reply is greedy unless the task states its generation settings, with thinking off. The
    likelihood is the log probability of each option as the whole reply (summed over its tokens); under a
    ``verbatim`` choice it scores ``f"{text} {option}"`` as plain text on the language model.

    Not wired yet: a sound part, earlier trials as earlier turns, and a ``Segments`` task. A sound part is refused
    at the first trial that shows one, before any pass.
    """
    from brainscore_core.compatibility import ExecutionNotFeasibleError
    from brainscore_core.responses import Choice, Text
    from brainscore.model_helpers.replies import generation_settings, plain_text_scores
    tokenizer = processor.tokenizer

    def embedded(prompt):
        import torch
        from PIL import Image
        from brainscore.model_helpers.video_wrapper import default_uniform_frame_sampler
        if prompt.history:
            raise ExecutionNotFeasibleError(f'{IDENTIFIER} takes no earlier trials as earlier turns.')
        names = prompt.names or (None,) * len(prompt.parts)
        system, shown, pictures, clip = [], [], [], False
        for (kind, value), name in zip(prompt.parts, names):
            if kind == 'text' and name == 'instruction':
                system.append(value)
            elif kind == 'text':
                shown.append(value)
            elif kind == 'image':
                pictures.append(Image.open(value).convert('RGB'))
                shown.append(PICTURE)
            elif kind == 'video':
                clip = True
                for frame in default_uniform_frame_sampler(value, ANSWER_FRAMES):
                    pictures.append(Image.fromarray(frame))
                    shown.append(PICTURE)
            else:
                raise ExecutionNotFeasibleError(f'{IDENTIFIER} answers here cannot be shown a {kind!r} part')
        messages = ([{'role': 'system', 'content': '\n\n'.join(system)}] if system else []) + [
            {'role': 'user', 'content': '\n'.join(shown)}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False, use_tts_template=False)
        inputs = processor([text], [pictures] if pictures else None, None, None,
                           max_slice_nums=1 if clip else None, return_tensors='pt')
        device = model.llm.get_input_embeddings().weight.device
        data = inputs.to(device=device)
        with torch.no_grad():
            embeds, _ = model.get_vllm_embedding(data)
        return embeds, data['attention_mask']

    def reply(prompt) -> str:
        import torch
        embeds, mask = embedded(prompt)
        settings = generation_settings(prompt, max_new_tokens)
        if isinstance(prompt.response, Text) and prompt.response.seed is not None:
            torch.manual_seed(prompt.response.seed)
        with torch.no_grad():
            generated = model.llm.generate(inputs_embeds=embeds, attention_mask=mask, pad_token_id=0,
                                           eos_token_id=[tokenizer.convert_tokens_to_ids(t) for t in TERMINATORS],
                                           **settings)
        return tokenizer.decode(generated[0], skip_special_tokens=True).strip()

    def likelihood(prompt, candidates) -> dict:
        import torch
        if isinstance(prompt.response, Choice) and prompt.response.verbatim:
            return plain_text_scores(model.llm, tokenizer, prompt, candidates)
        embeds, mask = embedded(prompt)
        length = embeds.shape[1]
        scores = {}
        for candidate in candidates:
            ids = torch.tensor([tokenizer(candidate, add_special_tokens=False)['input_ids']], device=embeds.device)
            if ids.shape[1] == 0:
                raise ValueError(f'option {candidate!r} has no tokens to score')
            with torch.no_grad():
                joined = torch.cat([embeds, model.llm.get_input_embeddings()(ids).to(embeds.dtype)], dim=1)
                full = torch.cat([mask, torch.ones_like(ids, dtype=mask.dtype)], dim=1)
                logits = model.llm(inputs_embeds=joined, attention_mask=full).logits[
                    0, length - 1:length - 1 + ids.shape[1]].to(ids.device)
            scores[candidate] = float(logits.float().log_softmax(-1).gather(-1, ids[0, :, None]).sum())
        return scores

    return reply, likelihood


def build_model(identifier: str, extractor: MiniCPMOExtractor, reply, likelihood) -> BrainScoreModel:
    """The subject: one extractor for the clip and the sound, and the written answers.

    The extractor goes under the tuple key ``('video', 'audio')``: a session with both inputs runs it once for all
    regions, and each call gets the whole row. No input is required: the model takes text alone, a picture alone,
    a clip alone or a sound alone.

    Limit: no speech output: the speech part is not loaded (``init_tts=False``), and the catalog has no speech
    output channel.
    Not wired yet: a written answer with sound, earlier trials as earlier turns, and a ``Segments`` task
    (``answer_parts``). A benchmark that sends only sound and asks for an answer is N/A.
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
    import os
    if os.environ.get('USE_FLAGOS'):
        raise ValueError('USE_FLAGOS is set: the model code of MiniCPM-o would then replace the RMSNorm of '
                         'transformers for every model in this process. Unset it, then load the model.')
    import torch
    import transformers
    if transformers.__version__ != TRANSFORMERS:
        raise ValueError(f'{IDENTIFIER} runs on transformers {TRANSFORMERS}, and this environment has '
                         f'{transformers.__version__}: the model code fails on another version (README, '
                         '"Environment"). Run it in the clone with transformers ' + TRANSFORMERS + '.')
    from transformers import AutoModel, AutoProcessor
    from brainscore.model_helpers.provenance import ProvenanceSubject
    processor = AutoProcessor.from_pretrained(WEIGHTS, revision=REVISION, trust_remote_code=True)
    model = AutoModel.from_pretrained(WEIGHTS, revision=REVISION, trust_remote_code=True,
                                      torch_dtype=torch.bfloat16, attn_implementation='sdpa', init_vision=True,
                                      init_audio=True, init_tts=False).eval()
    if torch.cuda.is_available():
        model = model.cuda()
    subject = build_model(identifier, MiniCPMOExtractor(identifier, model, processor),
                          *answer_parts(model, processor))
    return ProvenanceSubject(subject, {
        'identifier': identifier, 'checkpoint': WEIGHTS, 'revision': REVISION, 'licence': 'Apache-2.0',
        'remote_code': f'{WEIGHTS}@{REVISION} (modeling, processing, configuration)', 'init_tts': False,
        'dtype': 'bfloat16', 'attention': 'sdpa', 'torch': torch.__version__,
        'transformers': transformers.__version__, 'unit_seconds': UNIT_SECONDS, 'max_slice_nums': 1,
        'batch': 1, 'template': HEAD + '<units>' + TAIL, 'thinking': 'off', 'pools': list(POOLS),
        'encoder_blocks': {'vision': list(VISUAL_BLOCKS), 'audio': list(AUDIO_LAYERS)},
        'answers': f'own input path, greedy unless the task states settings, a clip as {ANSWER_FRAMES} frames'})
