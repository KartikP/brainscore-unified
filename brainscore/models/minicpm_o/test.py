"""Offline tests of the MiniCPM-o 4.5 registration with a stand-in model and processor (no weights)."""
import numpy as np
import pytest

PICTURE_TOKENS, SOUND_TOKENS = 3, 2       # a unit: tokens of its frame, then of its sound
WIDTH = 16


class _Inputs(dict):
    def to(self, device=None):
        return self


class _Processor:
    """Two text tokens, then each unit's picture tokens and sound tokens, then two text tokens."""

    def __init__(self):
        self.calls = []

    def __call__(self, texts, images=None, audios=None, audio_parts=None, max_slice_nums=None, return_tensors=None):
        import torch
        from .model import PICTURE, SOUND
        text = texts[0]
        self.calls.append({'text': text, 'pictures': len(images[0]) if images else 0,
                           'sounds': [len(a) for a in audios[0]] if audios else [], 'slices': max_slice_nums})
        ids, pictures, sounds, position = [1, 1], [], [], 2
        for token in text.replace(PICTURE, '\0P').replace(SOUND, '\0S').split('\0')[1:]:
            count = PICTURE_TOKENS if token.startswith('P') else SOUND_TOKENS
            (pictures if token.startswith('P') else sounds).append([position, position + count])
            ids += [5] * count
            position += count
        ids += [2, 3]
        return _Inputs(input_ids=torch.tensor([ids]), attention_mask=torch.ones(1, len(ids), dtype=torch.bool),
                       image_bound=[torch.tensor(pictures).reshape(-1, 2)],
                       audio_bounds=[torch.tensor(sounds).reshape(-1, 2)])


class _Model:
    def __init__(self):
        import torch
        self.vpm = torch.nn.Module()
        self.vpm.encoder = torch.nn.Module()
        self.vpm.encoder.layers = torch.nn.ModuleList([torch.nn.Identity() for _ in range(27)])
        self.apm = torch.nn.Module()
        self.apm.layers = torch.nn.ModuleList([torch.nn.Identity() for _ in range(24)])
        self.llm = torch.nn.Module()
        self.llm.embed = torch.nn.Embedding(8, 4)
        self.llm.get_input_embeddings = lambda: self.llm.embed
        self.calls = []

    def __call__(self, data, output_hidden_states=False):
        import torch
        from types import SimpleNamespace
        n = data['input_ids'].shape[1]
        self.calls.append(n)
        if len(data['image_bound'][0]):
            for layer in self.vpm.encoder.layers:
                layer((torch.full((len(data['image_bound'][0]), 5, 8), 1.0),))
        if len(data['audio_bounds'][0]):
            for layer in self.apm.layers:
                layer((torch.full((1, 7, 8), 2.0),))
        # state k, position p: every value is 100 k + p
        states = tuple((100.0 * k + torch.arange(n, dtype=torch.float32))[None, :, None].expand(1, n, WIDTH)
                       for k in range(37))
        return SimpleNamespace(hidden_states=states)


def _subject():
    from .model import MiniCPMOExtractor, build_model
    model, processor = _Model(), _Processor()
    extractor = MiniCPMOExtractor('minicpm-o-4.5', model, processor)

    def reply(prompt):
        return 'B'

    def likelihood(prompt, candidates):
        return {candidate: -float(i) for i, candidate in enumerate(candidates)}
    return build_model('minicpm-o-4.5', extractor, reply, likelihood), extractor, model, processor


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
            for _ in range(30):                   # 1.5 s at 20 frames a second
                writer.write(np.full((32, 32, 3), 40 * i, np.uint8))
            writer.release()
            row['video_path'] = str(path)
        if sound:
            wav = tmp_path / f'scan-{i}.wav'
            wavfile.write(wav, 16000, np.zeros(23840, np.int16))          # 1.49 s
            row['audio_path'] = str(wav)
        rows.append(row)
    table = StimulusSet(rows)
    table.identifier = f'made-up-scans-{n}-{sound}-{clip}'
    table.stimulus_paths = dict(zip(table['stimulus_id'], table['video_path' if clip else 'audio_path']))
    return table


def test_a_scan_of_1_49_s_is_two_units_of_a_frame_then_its_sound(tmp_path):
    from .model import PICTURE, SOUND
    subject, extractor, model, processor = _subject()
    rows = _rows(tmp_path, n=1)
    record = extractor.pooled([(rows['video_path'][0], rows['audio_path'][0])])[0]
    call = processor.calls[0]
    assert call['text'].count(PICTURE + SOUND) == 2 and call['pictures'] == 2 and call['slices'] == 1
    assert call['sounds'] == [16000, 7840]
    assert 'system' not in call['text'] and call['text'].endswith('<think>\n\n</think>\n\n')
    assert float(record['tokens/picture'][0]) == 2 * PICTURE_TOKENS
    assert float(record['tokens/sound'][0]) == 2 * SOUND_TOKENS


def test_four_regions_come_from_one_pass_per_row_with_the_pools_kept_apart(tmp_path):
    from brainscore_core.streaming_helpers import stimulus_session
    subject, extractor, model, processor = _subject()
    session = stimulus_session(_rows(tmp_path), [f'neural:{r}' for r in ('V1', 'V4', 'IT', 'A1')], None)
    subject.participate(session)
    assert len(model.calls) == 2                   # one pass a row for four regions
    # positions: 2-4 picture, 5-6 sound, 7-9 picture, 10-11 sound, last 13; layers.18 is state 19
    sound = session.collect('neural:A1').transpose('presentation', 'neuroid').values
    assert np.allclose(sound, 100 * 19 + np.mean([5, 6, 10, 11]))
    picture = session.collect('neural:V1').transpose('presentation', 'neuroid').values
    assert np.allclose(picture, 100 * 7 + 13)


def test_without_sound_the_clip_regions_run_and_a1_is_refused_before_any_pass(tmp_path):
    from brainscore_core.streaming_helpers import stimulus_session
    subject, extractor, model, processor = _subject()
    assert subject.output_inputs['neural:A1'] == {'vision:video', 'audio'}     # a language-model layer: rule 2.5
    session = stimulus_session(_rows(tmp_path, sound=False), ['neural:IT'], None)
    subject.participate(session)
    assert processor.calls[0]['sounds'] == [] and processor.calls[0]['pictures'] == 2   # 1.5 s: two units
    with pytest.raises(Exception, match='audio_path'):
        subject.participate(stimulus_session(_rows(tmp_path, sound=False), ['neural:A1'], None))
    assert len(model.calls) == 2


def test_a_sound_alone_is_refused_at_the_session_start_and_the_extractor_reads_it(tmp_path):
    from brainscore_core.compatibility import ExecutionNotFeasibleError
    from brainscore_core.streaming_helpers import stimulus_session
    subject, extractor, model, processor = _subject()
    for region in ('V1', 'A1'):
        with pytest.raises(ExecutionNotFeasibleError):
            subject.participate(stimulus_session(_rows(tmp_path, clip=False), [f'neural:{region}'], None))
    assert not model.calls
    record = extractor.pooled([(None, _rows(tmp_path, n=1, clip=False)['audio_path'][0])])[0]
    assert processor.calls[0]['pictures'] == 0 and 'layers.3/picture' not in record


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


def test_a_layer_that_is_not_recorded_is_refused_before_any_pass(tmp_path):
    subject, extractor, model, processor = _subject()
    with pytest.raises(ValueError, match='not layers'):
        extractor(_rows(tmp_path, n=1), ['layers.36/picture'])
    assert not model.calls


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
    from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
    subject, extractor, model, processor = _subject()
    table = StimulusSet([{'stimulus_id': 'trial-0', 'text': 'Is a cat an animal?'}])
    table.identifier = 'made-up-trials-text'
    response = Choice(['A', 'B'], verbatim=output == 'behavior:log_probabilities',
                      reaction_time='surprisal' if output == 'behavior:rt' else None)
    task = (TaskContext(response=response) if response.verbatim
            else TaskContext(instruction='Answer A or B.', response=response))
    session = stimulus_session(table, [output], task)
    subject.participate(session)
    assert len(session.collect(output)['presentation']) == 1
    assert not model.calls             # the answer parts ran; no recording pass


def test_the_answer_parts_refuse_a_sound_part_before_any_pass(tmp_path):
    from types import SimpleNamespace
    from brainscore_core.compatibility import ExecutionNotFeasibleError
    from brainscore_core.responses import Choice, Prompt
    from .model import answer_parts
    model, processor = _Model(), _Processor()
    processor.tokenizer = SimpleNamespace()
    reply, likelihood = answer_parts(model, processor)
    prompt = Prompt(parts=(('audio', 'x.wav'), ('text', 'A or B?')), names=(None, None), response=Choice(['A', 'B']))
    with pytest.raises(ExecutionNotFeasibleError, match="'audio' part"):
        reply(prompt)
    with pytest.raises(ExecutionNotFeasibleError, match="'audio' part"):
        likelihood(prompt, ['A', 'B'])
    assert not model.calls and not processor.calls


def test_a_request_it_cannot_serve_is_refused_before_any_pass():
    from types import SimpleNamespace
    from brainscore_core.compatibility import CompatibilityError, check_channel_compatibility
    subject, extractor, model, processor = _subject()
    for inputs, output in [({'vision:image'}, 'neural:IT'), ({'text'}, 'neural:V1'), ({'audio'}, 'behavior:choice'),
                           ({'text'}, 'neural:language_system'), ({'vision:image'}, 'motor')]:
        with pytest.raises(CompatibilityError):
            check_channel_compatibility(subject, SimpleNamespace(
                identifier='made-up', required_input_channels=inputs, requested_output_channels={output}))
    assert not model.calls and not processor.calls


def test_a_wrong_identifier_the_global_norm_switch_and_another_transformers_are_refused(monkeypatch):
    import transformers
    from .model import get_model
    with pytest.raises(ValueError, match='registers'):
        get_model('minicpm-o-made-up')
    monkeypatch.setenv('USE_FLAGOS', '1')
    with pytest.raises(ValueError, match='USE_FLAGOS'):
        get_model('minicpm-o-4.5')
    monkeypatch.delenv('USE_FLAGOS')
    monkeypatch.setattr(transformers, '__version__', '4.57.6')
    with pytest.raises(ValueError, match='4.51.0'):
        get_model('minicpm-o-4.5')
