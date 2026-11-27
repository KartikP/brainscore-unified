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
        self.calls.append({'frames': len(videos[0]), 'rows': len(videos), 'sound': audio is not None,
                           'together': use_audio_in_video, 'fps': fps})
        ids = [1] + [VIDEO] * PICTURE_TOKENS + ([AUDIO] * SOUND_TOKENS if audio is not None else []) + [2, 3]
        return {'input_ids': torch.tensor([ids] * len(videos)), 'attention_mask': torch.ones(len(videos), len(ids))}


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
        if use_audio_in_video:
            for layer in self.audio_tower.layers:
                layer((torch.full((rows * 3, 1280), 2.0),))
        # row b, state k, position p: every value is 1000 b + 100 k + p
        states = tuple((1000.0 * torch.arange(rows)[:, None] + 100.0 * k + torch.arange(n, dtype=torch.float32)
                        )[:, :, None].expand(rows, n, 2048) for k in range(49))
        return SimpleNamespace(hidden_states=states)


def _subject():
    from types import SimpleNamespace
    from .model import OmniThinkerExtractor, build_model
    thinker = _Thinker()
    model = SimpleNamespace(thinker=thinker, config=SimpleNamespace(
        thinker_config=SimpleNamespace(video_token_id=VIDEO, audio_token_id=AUDIO)))
    processor = _Processor()
    extractor = OmniThinkerExtractor('made-up-omni', model, processor)
    return build_model('made-up-omni', extractor), extractor, thinker, processor


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
    # layers.23 is state 24; sound tokens at positions 5, 6; layers.11 is state 12, picture tokens at 1 to 4
    rows = 1000 * np.arange(2)[:, None]
    assert np.allclose(sound, rows + 100 * 24 + 5.5) and np.allclose(picture, rows + 100 * 12 + 2.5)


def test_without_sound_the_clip_regions_run_and_a1_is_refused_before_any_pass(tmp_path):
    from brainscore_core.streaming_helpers import stimulus_session
    subject, extractor, thinker, processor = _subject()
    assert subject.output_inputs['neural:A1'] == {'audio'}
    assert subject.output_inputs['neural:IT'] >= {'vision:video'}
    assert subject.required_channels == {'vision:video'}
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


def test_a_wrong_identifier_is_refused():
    from .model import get_model
    with pytest.raises(ValueError, match='registers'):
        get_model('qwen3-omni-made-up')
