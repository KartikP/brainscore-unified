"""Offline tests of the Qwen2.5-Omni registrations with a stand-in thinker and processor (no weights)."""
import numpy as np
import pytest

VIDEO, AUDIO = 7, 9
PICTURE_TOKENS, SOUND_TOKENS = 4, 2
WIDTH = 16


class _Processor:
    """Text, then the picture tokens (and the sound tokens), then two text tokens."""

    def __init__(self):
        self.calls = []

    def __call__(self, text, videos, audio=None, use_audio_in_video=False, fps=None, return_tensors=None):
        import torch
        rows = len(videos if videos is not None else audio)
        self.calls.append({'sound': audio is not None, 'together': use_audio_in_video, 'fps': fps, 'rows': rows,
                           'frames': len(videos[0]) if videos is not None else 0, 'template': text[0]})
        ids = ([1] + ([VIDEO] * PICTURE_TOKENS if videos is not None else [])
               + ([AUDIO] * SOUND_TOKENS if audio is not None else []) + [2, 3])
        return {'input_ids': torch.tensor([ids] * rows), 'attention_mask': torch.ones(rows, len(ids))}


class _Thinker:
    def __init__(self, layers):
        import torch
        self.layers = layers
        self.visual = torch.nn.Module()
        self.visual.blocks = torch.nn.ModuleList([torch.nn.Identity() for _ in range(32)])
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
        if (input_ids == VIDEO).any():            # the encoders pack the rows of a batch one after the other
            for block in self.visual.blocks:
                block(torch.arange(rows, dtype=torch.float32).repeat_interleave(5)[:, None].expand(rows * 5, 8))
        if (input_ids == AUDIO).any():
            for layer in self.audio_tower.layers:
                layer((torch.full((rows * 3, 8), 2.0),))
        # row b, state k, position p: every value is 1000 b + 100 k + p
        states = tuple((1000.0 * torch.arange(rows)[:, None] + 100.0 * k + torch.arange(n, dtype=torch.float32)
                        )[:, :, None].expand(rows, n, WIDTH) for k in range(self.layers + 1))
        return SimpleNamespace(hidden_states=states)


def _subject(identifier='qwen2.5-omni-7b', prompts=None):
    """The subject with a stand-in thinker; its reply and likelihood keep each prompt's parts in ``prompts``."""
    from types import SimpleNamespace
    from .model import MODELS, OmniThinkerExtractor, build_model
    thinker = _Thinker(MODELS[identifier]['layers'])
    model = SimpleNamespace(thinker=thinker, config=SimpleNamespace(thinker_config=SimpleNamespace(
        video_token_index=VIDEO, audio_token_index=AUDIO,
        text_config=SimpleNamespace(num_hidden_layers=MODELS[identifier]['layers']))))
    processor = _Processor()
    extractor = OmniThinkerExtractor(identifier, model, processor)
    prompts = [] if prompts is None else prompts

    def reply(prompt):
        prompts.append([kind for kind, _ in prompt.parts] + list(prompt.names or ()))
        return 'B'

    def likelihood(prompt, candidates):
        prompts.append([kind for kind, _ in prompt.parts] + list(prompt.names or ()))
        return {candidate: -float(i) for i, candidate in enumerate(candidates)}
    return build_model(identifier, extractor, reply, likelihood), extractor, thinker, processor


def _rows(tmp_path, n=2, sound=True, clip=True):
    import cv2
    from scipy.io import wavfile
    from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
    rows = []
    for i in range(n):
        row = {'stimulus_id': f'scan-{i}'}
        if clip:
            path = tmp_path / f'scan-{i}.mp4'
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 20, (32, 32))
            for _ in range(30):
                writer.write(np.full((32, 32, 3), 40 * i, np.uint8))
            writer.release()
            row['video_path'] = str(path)
        if sound:
            wav = tmp_path / f'scan-{i}.wav'
            wavfile.write(wav, 16000, np.zeros(23840, np.int16))
            row['audio_path'] = str(wav)
        rows.append(row)
    table = StimulusSet(rows)
    table.identifier = f'made-up-scans-{n}-{sound}-{clip}'
    table.stimulus_paths = dict(zip(table['stimulus_id'], table['video_path' if clip else 'audio_path']))
    return table


def _map(identifier):
    from .model import MODELS
    return {region: selector.name for region, selector in MODELS[identifier]['region_layer_map'].items()}


@pytest.mark.parametrize('identifier', ['qwen2.5-omni-7b', 'qwen2.5-omni-3b'])
def test_four_regions_come_from_one_pass_per_row_with_the_pools_kept_apart(tmp_path, identifier):
    from brainscore_core.streaming_helpers import stimulus_session
    subject, extractor, thinker, processor = _subject(identifier)
    session = stimulus_session(_rows(tmp_path), [f'neural:{r}' for r in ('V1', 'V4', 'IT', 'A1')], None)
    subject.participate(session)
    assert thinker.calls == 1 and processor.calls[0]['rows'] == 2   # one batched pass for four regions
    assert processor.calls[0]['together'] and processor.calls[0]['frames'] == 8
    layers = _map(identifier)
    state = {region: int(layer.split('/')[0].split('.')[1]) + 1 for region, layer in layers.items()}
    rows = 1000 * np.arange(2)[:, None]
    # picture tokens at positions 1 to 4, sound tokens at 5 and 6, the last token at 8
    assert np.allclose(session.collect('neural:A1').transpose('presentation', 'neuroid').values,
                       rows + 100 * state['A1'] + 5.5)
    assert np.allclose(session.collect('neural:V1').transpose('presentation', 'neuroid').values,
                       rows + 100 * state['V1'] + 8)


def test_the_placeholder_maps_follow_the_depth_partition_of_the_rule():
    assert _map('qwen2.5-omni-7b') == {'V1': 'layers.5/last', 'V4': 'layers.14/last', 'IT': 'layers.23/last',
                                       'A1': 'layers.14/sound'}
    assert _map('qwen2.5-omni-3b') == {'V1': 'layers.6/last', 'V4': 'layers.18/last', 'IT': 'layers.30/last',
                                       'A1': 'layers.18/sound'}


def test_without_sound_the_clip_regions_run_and_a1_is_refused_before_any_pass(tmp_path):
    from brainscore_core.streaming_helpers import stimulus_session
    subject, extractor, thinker, processor = _subject()
    assert subject.output_inputs['neural:A1'] == {'vision:video', 'audio'}     # a thinker layer: rule 2.5
    assert subject.output_inputs['neural:IT'] == {'vision:video', 'audio'}
    session = stimulus_session(_rows(tmp_path, sound=False), ['neural:IT'], None)
    subject.participate(session)
    assert not processor.calls[0]['together'] and thinker.calls == 1
    with pytest.raises(Exception, match='audio_path'):
        subject.participate(stimulus_session(_rows(tmp_path, sound=False), ['neural:A1'], None))
    assert thinker.calls == 1


def test_a_sound_alone_runs_with_the_sound_template_or_is_refused_before_any_pass(tmp_path):
    from brainscore_core.compatibility import ExecutionNotFeasibleError
    from brainscore_core.streaming_helpers import stimulus_session
    subject, extractor, thinker, processor = _subject()
    for region in ('V1', 'A1'):
        with pytest.raises(ExecutionNotFeasibleError):
            subject.participate(stimulus_session(_rows(tmp_path, clip=False), [f'neural:{region}'], None))
    assert thinker.calls == 0
    # the extractor itself reads a sound alone with the sound template
    from .model import SOUND_TEMPLATE
    records = extractor.pooled([(None, a) for a in _rows(tmp_path, clip=False)['audio_path']])
    assert processor.calls[0]['template'] == SOUND_TEMPLATE and not processor.calls[0]['together']
    assert float(records[1]['layers.14/sound'][0]) == 1000 + 100 * 15 + 1.5


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
    assert response.shape[-1] == WIDTH


def test_a_layer_that_is_not_recorded_and_a_sound_at_another_rate_are_refused(tmp_path):
    from scipy.io import wavfile
    from .model import read_sound
    subject, extractor, thinker, processor = _subject()
    with pytest.raises(ValueError, match='not layers'):
        extractor(_rows(tmp_path, n=1), ['layers.28/picture'])
    wav = tmp_path / 'other.wav'
    wavfile.write(wav, 44100, np.zeros(100, np.int16))
    with pytest.raises(ValueError, match='16000'):
        read_sound(wav)
    assert thinker.calls == 0


def test_the_encoder_means_are_kept_per_row_in_a_batch(tmp_path):
    subject, extractor, thinker, processor = _subject()
    rows = _rows(tmp_path, n=3)
    records = extractor.pooled(list(zip(rows['video_path'], rows['audio_path'])))
    assert [float(r['visual.blocks.10/mean'][0]) for r in records] == [0.0, 1.0, 2.0]
    assert [float(r['tokens/sound'][0]) for r in records] == [SOUND_TOKENS] * 3


def test_a_wrong_identifier_is_refused():
    from .model import get_model
    with pytest.raises(ValueError, match='registers'):
        get_model('qwen2.5-omni-made-up')


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
    for output in ('behavior:choice', 'behavior:log_probabilities', 'behavior:probabilities', 'behavior:rt'):
        assert subject.output_inputs[output] == {'text', 'vision:image', 'vision:video'}


@pytest.mark.parametrize('output', ['behavior:choice', 'behavior:probabilities', 'behavior:log_probabilities',
                                    'behavior:rt'])
def test_each_declared_answer_is_produced(tmp_path, output):
    from brainscore_core.contract import TaskContext
    from brainscore_core.responses import Choice
    from brainscore_core.streaming_helpers import stimulus_session
    subject, extractor, thinker, processor = _subject()
    if output not in subject.out_channels:
        pytest.skip(f'{output} is not declared')
    # log probabilities are given for a verbatim choice; a reaction time when the response asks for one
    response = Choice(['A', 'B'], verbatim=output == 'behavior:log_probabilities',
                      reaction_time='surprisal' if output == 'behavior:rt' else None)
    task = (TaskContext(response=response) if response.verbatim
            else TaskContext(instruction='Answer A or B.', response=response))
    session = stimulus_session(_table(tmp_path, text='Is a cat an animal?'), [output], task)
    subject.participate(session)
    assert len(session.collect(output)['presentation']) == 1


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
    subject, extractor, thinker, processor = _subject(prompts=prompts)
    session = stimulus_session(_table(tmp_path, **columns), ['behavior:choice'],
                               TaskContext(instruction='Answer A or B.', response=Choice(['A', 'B'])))
    subject.participate(session)
    assert len(session.collect('behavior:choice')['presentation']) == 1
    assert all(set(shown) <= set(parts) for parts in prompts) and prompts
    assert thinker.calls == 0          # the answer parts ran; no recording pass


def test_a_request_it_cannot_serve_is_refused_before_any_pass():
    from types import SimpleNamespace
    from brainscore_core.compatibility import CompatibilityError, check_channel_compatibility
    subject, extractor, thinker, processor = _subject()
    for inputs, output in [({'vision:image'}, 'neural:IT'), ({'text'}, 'neural:V1'), ({'audio'}, 'behavior:choice'),
                           ({'text'}, 'neural:language_system'), ({'vision:image'}, 'motor')]:
        with pytest.raises(CompatibilityError):
            check_channel_compatibility(subject, SimpleNamespace(
                identifier='made-up', required_input_channels=inputs, requested_output_channels={output}))
    assert thinker.calls == 0 and not processor.calls
