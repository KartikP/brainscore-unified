"""Offline tests of the LFM3-Omni-2B registration with a stand-in model and processor (no weights)."""
import numpy as np
import pytest

VIDEO, AUDIO = 7, 9
GROUP_PICTURE, GROUP_SOUND = 2, 1       # tokens of one frame group, then of its sound
WIDTH = 16


class _Processor:
    """Two text tokens; with a clip, each group of two frames (its picture tokens, then its sound tokens when the
    sound comes with the clip); a sound alone as one run; then two text tokens."""

    def __init__(self):
        self.calls = []

    def __call__(self, text, return_tensors=None, videos=None, video_metadata=None, do_sample_frames=None,
                 audio=None, use_audio_in_video=False):
        import torch
        self.calls.append({'text': text[0], 'frames': len(videos[0]) if videos is not None else 0,
                           'fps': video_metadata[0].fps if video_metadata else None, 'sample': do_sample_frames,
                           'sound': audio is not None, 'together': use_audio_in_video})
        ids = [1, 1]
        if videos is not None:
            for _ in range(len(videos[0]) // 2):
                ids += [VIDEO] * GROUP_PICTURE + ([AUDIO] * GROUP_SOUND if use_audio_in_video else [])
        if audio is not None and not use_audio_in_video:
            ids += [AUDIO] * 3
        ids += [2, 3]
        return {'input_ids': torch.tensor([ids]), 'attention_mask': torch.ones(1, len(ids))}


class _Model:
    def __init__(self):
        import torch
        from types import SimpleNamespace
        self.config = SimpleNamespace(video_token_id=VIDEO, audio_token_id=AUDIO)
        self.model = torch.nn.Module()
        self.model.vision_tower = torch.nn.Module()
        self.model.vision_tower.layers = torch.nn.ModuleList([torch.nn.Identity() for _ in range(14)])
        self.model.audio_encoder = torch.nn.Module()
        self.model.audio_encoder.lfm = torch.nn.Module()
        self.model.audio_encoder.lfm.layers = torch.nn.ModuleList([torch.nn.Identity() for _ in range(17)])
        self._embedding = torch.nn.Embedding(16, 4)
        self.calls = []

    def get_input_embeddings(self):
        return self._embedding

    def __call__(self, input_ids, output_hidden_states=False, attention_mask=None):
        import torch
        from types import SimpleNamespace
        n = input_ids.shape[1]
        self.calls.append(n)
        if (input_ids == VIDEO).any():
            for layer in self.model.vision_tower.layers:
                layer(torch.full((20, 8), 1.0))
        if (input_ids == AUDIO).any():
            for layer in self.model.audio_encoder.lfm.layers:
                layer((torch.full((1, 6, 8), 2.0),))
        # state k, position p: every value is 100 k + p
        states = tuple((100.0 * k + torch.arange(n, dtype=torch.float32))[None, :, None].expand(1, n, WIDTH)
                       for k in range(25))
        return SimpleNamespace(hidden_states=states)


def _subject():
    from .model import Lfm3OmniExtractor, build_model
    model, processor = _Model(), _Processor()
    extractor = Lfm3OmniExtractor('lfm3-omni-2b', model, processor)

    def reply(prompt):
        return 'B'

    def likelihood(prompt, candidates):
        return {candidate: -float(i) for i, candidate in enumerate(candidates)}
    return build_model('lfm3-omni-2b', extractor, reply, likelihood), extractor, model, processor


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


def test_a_scan_gives_8_frames_with_their_clock_and_the_sound_after_each_group(tmp_path):
    from .model import CLIP, SOUND
    subject, extractor, model, processor = _subject()
    rows = _rows(tmp_path, n=1)
    record = extractor.pooled([(rows['video_path'][0], rows['audio_path'][0])])[0]
    call = processor.calls[0]
    assert call['frames'] == 8 and call['sample'] is False and call['together']
    assert abs(call['fps'] - 8 / 1.49) < 1e-9
    assert CLIP + SOUND in call['text'] and 'system' not in call['text'] and call['text'].endswith('<think></think>')
    assert float(record['tokens/picture'][0]) == 4 * GROUP_PICTURE
    assert float(record['tokens/sound'][0]) == 4 * GROUP_SOUND


def test_four_regions_come_from_one_pass_per_row_with_the_pools_kept_apart(tmp_path):
    from brainscore_core.streaming_helpers import stimulus_session
    subject, extractor, model, processor = _subject()
    session = stimulus_session(_rows(tmp_path), [f'neural:{r}' for r in ('V1', 'V4', 'IT', 'A1')], None)
    subject.participate(session)
    assert len(model.calls) == 2                   # one pass a row for four regions
    # groups at 2-4, 5-7, 8-10, 11-13 (two picture tokens, then one sound token); last at 15; layers.12 is state 13
    sound = session.collect('neural:A1').transpose('presentation', 'neuroid').values
    assert np.allclose(sound, 100 * 13 + np.mean([4, 7, 10, 13]))
    picture = session.collect('neural:V1').transpose('presentation', 'neuroid').values
    assert np.allclose(picture, 100 * 5 + 15)


def test_without_sound_the_clip_regions_run_and_a1_is_refused_before_any_pass(tmp_path):
    from brainscore_core.streaming_helpers import stimulus_session
    subject, extractor, model, processor = _subject()
    assert subject.output_inputs['neural:A1'] == {'vision:video', 'audio'}     # a language-model layer: rule 2.5
    subject.participate(stimulus_session(_rows(tmp_path, sound=False), ['neural:IT'], None))
    assert not processor.calls[0]['together'] and not processor.calls[0]['sound']
    with pytest.raises(Exception, match='audio_path'):
        subject.participate(stimulus_session(_rows(tmp_path, sound=False), ['neural:A1'], None))
    assert len(model.calls) == 2


def test_a_sound_alone_is_refused_at_the_session_start_and_the_extractor_reads_it(tmp_path):
    from brainscore_core.compatibility import ExecutionNotFeasibleError
    from brainscore_core.streaming_helpers import stimulus_session
    from .model import CLIP, SOUND
    subject, extractor, model, processor = _subject()
    for region in ('V1', 'A1'):
        with pytest.raises(ExecutionNotFeasibleError):
            subject.participate(stimulus_session(_rows(tmp_path, clip=False), [f'neural:{region}'], None))
    assert not model.calls
    record = extractor.pooled([(None, _rows(tmp_path, n=1, clip=False)['audio_path'][0])])[0]
    assert SOUND in processor.calls[0]['text'] and CLIP not in processor.calls[0]['text']
    assert float(record['tokens/sound'][0]) == 3 and 'layers.3/picture' not in record


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
        extractor(_rows(tmp_path, n=1), ['layers.24/picture'])
    assert not model.calls


def test_it_declares_every_input_and_requires_none():
    subject = _subject()[0]
    assert {'text', 'vision:image', 'vision:video', 'audio', 'instruction'} <= subject.in_channels
    assert subject.required_channels == set()
    for output in ('behavior:choice', 'behavior:log_probabilities', 'behavior:probabilities', 'behavior:rt'):
        assert subject.output_inputs[output] == {'text', 'vision:image', 'vision:video'}


@pytest.mark.parametrize('output', ['behavior:choice', 'behavior:probabilities', 'behavior:log_probabilities',
                                    'behavior:rt'])
def test_each_declared_answer_is_produced(output):
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


def test_a_wrong_identifier_and_an_older_transformers_are_refused(monkeypatch):
    import transformers
    from .model import get_model
    with pytest.raises(ValueError, match='registers'):
        get_model('lfm3-omni-made-up')
    monkeypatch.setattr(transformers, '__version__', '4.57.6')
    with pytest.raises(ValueError, match='5.20.2'):
        get_model('lfm3-omni-2b')
