"""The load of each registered identifier is checked by the generic plugin suite
(``brainscore/model_helpers/generic_plugin_tests.py``)."""
import socket

import pytest


@pytest.mark.parametrize('identifier', ['clip-wav2vec', 'vjepa1-wav2vec2', 'random-vjepa1-wav2vec2'])
def test_a_wrong_identifier_is_refused_before_any_tower(identifier, monkeypatch):
    from .model import get_model

    def no_network(*args, **kwargs):
        raise OSError('the network is blocked in this check')
    monkeypatch.setenv('HF_HUB_OFFLINE', '1')
    monkeypatch.setattr(socket.socket, 'connect', no_network)
    with pytest.raises(ValueError, match='list_models'):
        get_model(identifier)


class _Tower:
    """A made-up tower: each layer answers, at each of ``steps`` time bins, the row number plus the step,
    on 4 units scaled by the layer's place."""

    def __init__(self, layers, steps):
        self._layers, self._steps = list(layers), steps
        self.calls = 0

    def layer_names(self):
        return list(self._layers)

    def __call__(self, stimuli, layers):
        from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
        import numpy as np
        self.calls += 1
        rows = np.arange(len(stimuli))[:, None, None] + np.arange(self._steps)[None, :, None]
        values = np.concatenate([rows * np.ones((1, 1, 4)) * (self._layers.index(layer) + 1)
                                 for layer in layers], axis=2)
        return NeuroidAssembly(values, dims=['presentation', 'time_bin', 'neuroid'], coords={
            'stimulus_id': ('presentation', list(stimuli['stimulus_id'])),
            'path': ('presentation', list(stimuli['stimulus_id'])),
            'time_bin_idx': ('time_bin', list(range(self._steps))),
            'layer': ('neuroid', [layer for layer in layers for _ in range(4)]),
            'neuroid_id': ('neuroid', [f'{layer}.{u}' for layer in layers for u in range(4)])})


def _fused(video_steps=3, audio_steps=5):
    from .model import fused_model
    video = _Tower([f'model.encoder.layers.{i}' for i in range(12)], video_steps)
    audio = _Tower([f'encoder.layers.{i}' for i in range(12)], audio_steps)
    return fused_model('made-up-fused', video, audio), video, audio


def _media(tmp_path, n=3):
    from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
    rows = []
    for i in range(n):
        paths = {f'{kind}_path': tmp_path / f'tr-{i}.{suffix}'
                 for kind, suffix in (('video', 'mp4'), ('image', 'png'), ('audio', 'wav'))}
        for path in paths.values():
            path.write_bytes(b'')
        rows.append({'stimulus_id': f'tr-{i}', **{k: str(v) for k, v in paths.items()}})
    table = StimulusSet(rows)
    table.identifier = f'made-up-media-{n}'
    table.stimulus_paths = dict(zip(table['stimulus_id'], table['image_path']))
    return table


def test_cortex_is_recorded_from_both_towers_in_one_session(tmp_path):
    import numpy as np
    from brainscore_core.streaming_helpers import neural_response
    from .model import CORTEX_LAYERS
    subject, video, audio = _fused()
    response = neural_response(subject, _media(tmp_path), ['neural:cortex'])
    layers = list(dict.fromkeys(np.asarray(response['layer'].values).astype(str)))
    assert layers == list(CORTEX_LAYERS) and len(layers) == 12
    assert {layer.partition('/')[0] for layer in layers} == {'video', 'audio'}
    assert video.calls == 1 and audio.calls == 1
    values = np.asarray(response.transpose('presentation', 'neuroid').values)
    # each tower's time bins are averaged: row r, steps 0..S-1 -> r + (S - 1) / 2, times the layer's place
    first_audio = np.asarray(response['layer'].values).astype(str) == 'audio/encoder.layers.1'
    assert np.allclose(values[:, first_audio], (np.arange(3)[:, None] + 2.0) * 2)
    first_video = np.asarray(response['layer'].values).astype(str) == 'video/model.encoder.layers.1'
    assert np.allclose(values[:, first_video], (np.arange(3)[:, None] + 1.0) * 2)


def test_the_mario_cortex_benchmark_runs_it_and_a_benchmark_without_the_sound_is_not_applicable():
    import numpy as np
    import brainscore
    from brainscore_core.compatibility import check_compatibility
    from brainscore.benchmarks.paugam2025.benchmark import Paugam2025Mario
    subject, video, audio = _fused()
    mario = Paugam2025Mario(('sub-01',), identifier='Paugam2025-fMRI-mario-cortex', root='/nothing-packaged-here')
    assert check_compatibility(subject, mario).consumed_inputs == {'vision:video', 'audio'}
    assert subject.in_channels >= {'vision:video', 'audio'} and 'neural:cortex' in subject.out_channels

    class Offers(Paugam2025Mario):
        required_input_channels = frozenset({'vision:video', 'vision:image'})    # no sound
    score = brainscore.score(subject, Offers(('sub-01',), identifier='made-up-no-sound',
                                             root='/nothing-packaged-here'), check_mem=False)
    assert np.isnan(float(score)) and video.calls == 0 and audio.calls == 0
