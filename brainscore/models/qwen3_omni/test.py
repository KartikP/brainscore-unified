"""Offline tests of the Qwen3-Omni registration with a stand-in thinker and processor (no weights)."""
import numpy as np
import pytest

VIDEO, AUDIO = 7, 9
PICTURE_TOKENS, SOUND_TOKENS = 4, 2


class _Processor:
    """Text, then the picture tokens (and the sound tokens), then two text tokens."""

    def __init__(self):
        self.calls = []

    def __call__(self, text, videos, audio=None, use_audio_in_video=False, fps=None, return_tensors=None):
        import torch
        self.calls.append({'sound': audio is not None, 'together': use_audio_in_video, 'fps': fps})
        rows = len(videos if videos is not None else audio)
        self.calls[-1].update(frames=len(videos[0]) if videos is not None else 0, rows=rows, template=text[0])
        ids = ([1] + ([VIDEO] * PICTURE_TOKENS if videos is not None else [])
               + ([AUDIO] * SOUND_TOKENS if audio is not None else []) + [2, 3])
        return {'input_ids': torch.tensor([ids] * rows), 'attention_mask': torch.ones(rows, len(ids))}


class _Thinker:
    def __init__(self):
        import torch
        self.visual = torch.nn.Module()
        self.visual.blocks = torch.nn.ModuleList([torch.nn.Identity() for _ in range(27)])
        self.audio_tower = torch.nn.Module()
        self.audio_tower.layers = torch.nn.ModuleList([torch.nn.Identity() for _ in range(32)])
        self._embedding = torch.nn.Embedding(16, 4)
        self.calls = 0

    def get_input_embeddings(self):
        return self._embedding

    def __call__(self, input_ids, output_hidden_states, use_audio_in_video, attention_mask=None):
        import torch
        from types import SimpleNamespace
        self.calls += 1
        rows, n = input_ids.shape
        for block in self.visual.blocks:          # the encoders pack the rows of a batch one after the other
            block(torch.arange(rows, dtype=torch.float32).repeat_interleave(5)[:, None].expand(rows * 5, 1152))
        if (input_ids == AUDIO).any():
            for layer in self.audio_tower.layers:
                layer((torch.full((rows * 3, 1280), 2.0),))
        # row b, state k, position p: every value is 1000 b + 100 k + p
        states = tuple((1000.0 * torch.arange(rows)[:, None] + 100.0 * k + torch.arange(n, dtype=torch.float32)
                        )[:, :, None].expand(rows, n, 2048) for k in range(49))
        return SimpleNamespace(hidden_states=states)


def _subject(prompts=None):
    """The subject with a stand-in thinker; its reply and likelihood keep each prompt's parts in ``prompts``."""
    from types import SimpleNamespace
    from .model import OmniThinkerExtractor, build_model
    thinker = _Thinker()
    model = SimpleNamespace(thinker=thinker, config=SimpleNamespace(
        thinker_config=SimpleNamespace(video_token_id=VIDEO, audio_token_id=AUDIO)))
    processor = _Processor()
    extractor = OmniThinkerExtractor('made-up-omni', model, processor)
    prompts = [] if prompts is None else prompts

    def reply(prompt):
        prompts.append([kind for kind, _ in prompt.parts] + list(prompt.names or ()))
        return 'B'

    def likelihood(prompt, candidates):
        prompts.append([kind for kind, _ in prompt.parts] + list(prompt.names or ()))
        return {candidate: -float(i) for i, candidate in enumerate(candidates)}
    return build_model('made-up-omni', extractor, reply, likelihood), extractor, thinker, processor


def _rows(tmp_path, n=2, sound=True):
    import cv2
    from scipy.io import wavfile
    from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
    rows = []
    for i in range(n):
        clip = tmp_path / f'scan-{i}.mp4'
        writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*'mp4v'), 20, (32, 32))
        for _ in range(30):
            writer.write(np.full((32, 32, 3), 40 * i, np.uint8))
        writer.release()
        row = {'stimulus_id': f'scan-{i}', 'video_path': str(clip)}
        if sound:
            wav = tmp_path / f'scan-{i}.wav'
            wavfile.write(wav, 16000, np.zeros(23840, np.int16))
            row['audio_path'] = str(wav)
        rows.append(row)
    table = StimulusSet(rows)
    table.identifier = f'made-up-scans-{n}-{sound}'
    table.stimulus_paths = dict(zip(table['stimulus_id'], table['video_path']))
    return table


def test_four_regions_come_from_one_pass_per_row_with_the_pools_kept_apart(tmp_path):
    from brainscore_core.streaming_helpers import stimulus_session
    subject, extractor, thinker, processor = _subject()
    session = stimulus_session(_rows(tmp_path), [f'neural:{r}' for r in ('V1', 'V4', 'IT', 'A1')], None)
    subject.participate(session)
    assert thinker.calls == 1 and processor.calls[0]['rows'] == 2   # one batched pass for four regions
    assert processor.calls[0]['together'] and processor.calls[0]['frames'] == 8
    sound = session.collect('neural:A1').transpose('presentation', 'neuroid').values
    picture = session.collect('neural:V1').transpose('presentation', 'neuroid').values
    # layers.23 is state 24, sound tokens at positions 5, 6; layers.14 is state 15, last token at position 8
    rows = 1000 * np.arange(2)[:, None]
    assert np.allclose(sound, rows + 100 * 24 + 5.5) and np.allclose(picture, rows + 100 * 15 + 8)


def test_without_sound_the_clip_regions_run_and_a1_is_refused_before_any_pass(tmp_path):
    from brainscore_core.streaming_helpers import stimulus_session
    subject, extractor, thinker, processor = _subject()
    assert subject.output_inputs['neural:A1'] == {'audio'}
    assert subject.output_inputs['neural:IT'] == {'vision:video', 'audio'}
    session = stimulus_session(_rows(tmp_path, sound=False), ['neural:IT'], None)
    subject.participate(session)
    assert not processor.calls[0]['together'] and thinker.calls == 1
    with pytest.raises(Exception):
        subject.participate(stimulus_session(_rows(tmp_path, sound=False), ['neural:A1'], None))
    assert thinker.calls == 1


def test_the_friends_benchmarks_with_and_without_sound_accept_it():
    from brainscore_core.compatibility import check_compatibility
    from brainscore.benchmarks.algonauts2025.benchmark import Algonauts2025
    subject = _subject()[0]
    both = Algonauts2025(1, channels=('vision:video', 'audio'), identifier='made-up-friends-av')
    clip = Algonauts2025(1, channels=('vision:video',), identifier='made-up-friends-v')
    assert check_compatibility(subject, both).consumed_inputs == {'vision:video', 'audio'}
    assert check_compatibility(subject, clip).consumed_inputs == {'vision:video'}


def test_the_memory_probe_order_records_a_clip_region_from_a_clip_alone(tmp_path):
    from brainscore_core.streaming_helpers import neural_response
    subject = _subject()[0]
    first = next(iter(subject.region_layer_map))
    response = neural_response(subject, _rows(tmp_path, n=1, sound=False), [f'neural:{first}'])
    assert response.shape[-1] == 2048


def test_a_layer_that_is_not_recorded_and_a_sound_at_another_rate_are_refused(tmp_path):
    from scipy.io import wavfile
    from .model import read_sound
    subject, extractor, thinker, processor = _subject()
    with pytest.raises(ValueError, match='not layers'):
        extractor(_rows(tmp_path, n=1), ['layers.48/picture'])
    wav = tmp_path / 'other.wav'
    wavfile.write(wav, 44100, np.zeros(100, np.int16))
    with pytest.raises(ValueError, match='16000'):
        read_sound(wav)
    assert thinker.calls == 0


def test_the_encoder_means_are_kept_per_row_in_a_batch(tmp_path):
    subject, extractor, thinker, processor = _subject()
    records = extractor.pooled([(v, a) for v, a in zip(_rows(tmp_path, n=3)['video_path'], _rows(tmp_path, n=3)['audio_path'])])
    assert [float(r['visual.blocks.8/mean'][0]) for r in records] == [0.0, 1.0, 2.0]
    assert [float(r['tokens/sound'][0]) for r in records] == [SOUND_TOKENS] * 3


def test_the_audio_encoder_takes_the_sounds_of_a_batch_one_at_a_time():
    import torch
    from types import SimpleNamespace
    from .model import _audio_per_row
    seen = []

    class Tower(torch.nn.Module):
        def forward(self, input_features, feature_lens=None, aftercnn_lens=None):
            seen.append(input_features.shape[1])
            # each output token holds the sum of every input frame it was given: rows must not meet
            return SimpleNamespace(last_hidden_state=input_features.sum(1, keepdim=True).T.expand(2, -1))
    tower = Tower()
    _audio_per_row(tower)
    features = torch.cat([torch.full((1, 3), 1.0), torch.full((1, 5), 10.0)], dim=1)
    out = tower(features, feature_lens=torch.tensor([3, 5])).last_hidden_state
    assert seen == [3, 5] and out[:2, 0].tolist() == [3.0, 3.0] and out[2:, 0].tolist() == [50.0, 50.0]


def test_a_wrong_identifier_is_refused():
    from .model import get_model
    with pytest.raises(ValueError, match='registers'):
        get_model('qwen3-omni-made-up')


def _table(tmp_path, **columns):
    """One row with the given columns; ``image_path`` a small picture file."""
    from PIL import Image
    from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
    row = {'stimulus_id': 'trial-0', **columns}
    if 'image_path' in columns:
        Image.fromarray(np.full((8, 8, 3), 90, np.uint8)).save(columns['image_path'])
    table = StimulusSet([row])
    table.identifier = 'made-up-trials-' + '-'.join(sorted(columns))
    return table


def test_it_declares_every_input_and_requires_none():
    subject = _subject()[0]
    assert {'text', 'vision:image', 'vision:video', 'audio', 'instruction'} <= subject.in_channels
    assert subject.required_channels == set()
    assert {'behavior:choice', 'behavior:rating', 'behavior:text', 'behavior:probabilities'} <= subject.out_channels
    assert subject.output_inputs['behavior:choice'] == {'text', 'vision:image', 'vision:video'}


@pytest.mark.parametrize('columns, shown', [
    ({'text': 'Is a cat an animal?'}, ['text', 'instruction']),
    ({'image_path': 'picture.png'}, ['image', 'instruction']),
    ({'video_path': 'clip'}, ['video', 'instruction']),
])
def test_text_a_picture_or_a_clip_alone_gets_a_written_answer(tmp_path, columns, shown):
    from brainscore_core.contract import TaskContext
    from brainscore_core.responses import Choice
    from brainscore_core.streaming_helpers import stimulus_session
    columns = {k: (str(tmp_path / v) if k.endswith('_path') else v) for k, v in columns.items()}
    if 'video_path' in columns:
        columns['video_path'] = _rows(tmp_path, n=1, sound=False)['video_path'][0]
    prompts = []
    subject, extractor, thinker, processor = _subject(prompts)
    for output in ('behavior:choice', 'behavior:probabilities'):
        session = stimulus_session(_table(tmp_path, **columns), [output],
                                   TaskContext(instruction='Answer A or B.', response=Choice(['A', 'B'])))
        subject.participate(session)
        assert len(session.collect(output)['presentation']) == 1
    assert all(set(shown) <= set(parts) for parts in prompts) and len(prompts) == 2
    assert thinker.calls == 0          # the answer parts ran; no recording pass


def test_a_sound_alone_records_a1_with_the_sound_template(tmp_path):
    from brainscore_core.streaming_helpers import stimulus_session
    from .model import SOUND_TEMPLATE
    subject, extractor, thinker, processor = _subject()
    rows = _rows(tmp_path)
    session = stimulus_session(rows[['stimulus_id', 'audio_path']].pipe(_as_table, rows), ['neural:A1'], None)
    subject.participate(session)
    assert processor.calls[0]['template'] == SOUND_TEMPLATE and not processor.calls[0]['together']
    sound = session.collect('neural:A1').transpose('presentation', 'neuroid').values
    # no picture tokens: the sound tokens are at positions 1, 2 of layers.23 (state 24)
    assert np.allclose(sound, 1000 * np.arange(2)[:, None] + 100 * 24 + 1.5)
    # a clip region from a sound alone is refused at the session's start, before any pass
    from brainscore_core.compatibility import ExecutionNotFeasibleError
    with pytest.raises(ExecutionNotFeasibleError):
        subject.participate(stimulus_session(rows[['stimulus_id', 'audio_path']].pipe(_as_table, rows),
                                             ['neural:V1'], None))
    assert thinker.calls == 1


def _as_table(frame, like):
    from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
    table = StimulusSet(frame.reset_index(drop=True))
    table.identifier = like.identifier + '-sound'
    return table


def test_a_request_it_cannot_serve_is_refused_before_any_pass(tmp_path):
    from types import SimpleNamespace
    from brainscore_core.compatibility import CompatibilityError, check_channel_compatibility
    from brainscore_core.contract import TaskContext
    from brainscore_core.responses import Choice
    from brainscore_core.streaming_helpers import stimulus_session
    prompts = []
    subject, extractor, thinker, processor = _subject(prompts)
    for inputs, output in [({'vision:image'}, 'neural:IT'), ({'text'}, 'neural:V1'), ({'audio'}, 'behavior:choice'),
                           ({'text'}, 'neural:language_system'), ({'vision:image'}, 'motor')]:
        with pytest.raises(CompatibilityError):
            check_channel_compatibility(subject, SimpleNamespace(
                identifier='made-up', required_input_channels=inputs, requested_output_channels={output}))
    # text with a sound passes pre-flight; the shared chat parts refuse the sound part at the first trial,
    # before any forward pass
    from brainscore.model_helpers.replies import chat_parts
    from .model import CHAT_TEMPLATE, build_model
    subject = build_model('made-up-omni', extractor, *chat_parts(thinker, processor, chat_template_kwargs=CHAT_TEMPLATE))
    rows = _rows(tmp_path, n=1)
    table = _as_table(rows[['stimulus_id', 'audio_path']].assign(text='Say A or B.'), rows)
    task = TaskContext(response=Choice(['A', 'B']), present=('audio:audio_path', 'text:text'))
    with pytest.raises(Exception, match="'audio' part"):
        subject.participate(stimulus_session(table, ['behavior:choice'], task))
    assert thinker.calls == 0 and not processor.calls
