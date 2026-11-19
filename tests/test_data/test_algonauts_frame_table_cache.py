"""The Friends table name identifies the input files, so every cache answers the current input.

Some stores key on the table name and the file paths only (the temporal helper of
old vision models). Others also hash the file bytes (the unified wrappers). The
table name must serve both: equal files in equal order share one name, so
participants share one stored entry; new bytes at an old path, or other rows,
give another name, so a store that keys on the name alone answers the current
frames. Tiny synthetic frames, stand-in networks, a temporary store.
"""
from collections import OrderedDict

import numpy as np
import pandas as pd
import pytest
import result_caching
from PIL import Image

from brainscore.data.algonauts2025 import per_scan_stimuli
from tests.test_model_helpers.test_wrapper_cache_guard import vlm_vision_wrapper
from tests.test_model_helpers.test_wrapper_cache_sharing import _storage, store  # noqa: F401 (fixture)


def _films(root, movies=('m1',)):
    return pd.DataFrame({'stimulus_id': list(movies), 'video_path': [str(root / f'{m}.mkv') for m in movies]})


def _frame(root, movie, t, value):
    path = root / 'frames' / movie / f'{t:04d}.jpg'
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new('RGB', (16, 16), (value, value, value)).save(path)


def _table(root, participant, n_scans):
    return per_scan_stimuli(_films(root), {'m1': n_scans}, ['frames'], 'friends', participant, root)


def _name_only_reader():
    """A stand-in store that keys on the table name alone: the mean pixel of each frame."""
    stored = {}

    def read(table, cached):
        if not cached or table.identifier not in stored:
            stored[table.identifier] = np.array([np.asarray(Image.open(path)).mean()
                                                 for path in table['image_file_name']])
        return stored[table.identifier] if cached else stored.pop(table.identifier)
    return read


def _temporal_reader(monkeypatch, tmp_path):
    """The temporal helper of old vision models, with a stand-in network and a temporary store."""
    from brainscore_vision.model_helpers.activations.temporal.core import TemporalInferencer
    from brainscore_vision.model_helpers.activations.temporal.model import ActivationWrapper

    class StandIn(ActivationWrapper):
        def get_activations(self, inputs, layers):
            return OrderedDict((layer, np.stack(inputs)) for layer in layers)

    model = StandIn('stand-in', lambda video: video.to_numpy()[:, :8, :8], inferencer_cls=TemporalInferencer,
                    fps=1, layer_activation_format={'layer1': 'THWC'}, max_workers=1, batch_size=4)
    monkeypatch.setattr(_storage(model._extractor._from_paths_stored), '_storage_directory', str(tmp_path / 'store'))
    temporal = 'brainscore_vision.model_helpers.activations.temporal.'

    def read(table, cached):
        monkeypatch.setattr(result_caching, 'is_enabled', lambda key: cached and key.startswith(temporal))
        return model(table, layers=['layer1']).values.copy()
    return read


@pytest.fixture(params=['name_only_stand_in', 'vision_temporal_helper'])
def reader(request, monkeypatch, tmp_path):
    if request.param == 'name_only_stand_in':
        return _name_only_reader()
    return _temporal_reader(monkeypatch, tmp_path)


def test_a_store_keyed_on_the_name_answers_a_frame_replaced_at_the_same_path(reader, tmp_path):
    _frame(tmp_path, 'm1', 1, 0)
    reader(_table(tmp_path, 1, 1), cached=True)
    _frame(tmp_path, 'm1', 1, 255)
    warm = reader(_table(tmp_path, 2, 1), cached=True)
    assert np.array_equal(warm, reader(_table(tmp_path, 2, 1), cached=False))


def test_a_store_keyed_on_the_name_answers_a_participant_with_other_rows(reader, tmp_path):
    _frame(tmp_path, 'm1', 1, 0)
    _frame(tmp_path, 'm1', 2, 255)
    reader(_table(tmp_path, 1, 2), cached=True)
    warm = reader(_table(tmp_path, 2, 1), cached=True)
    assert np.array_equal(warm, reader(_table(tmp_path, 2, 1), cached=False))


def test_participants_with_equal_frames_share_one_stored_entry(store, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for t, value in ((1, 30), (2, 200)):
        _frame(tmp_path, 'm1', t, value)
    first, second = _table(tmp_path, 1, 2), _table(tmp_path, 2, 2)
    assert first.identifier == second.identifier
    wrapper = vlm_vision_wrapper()
    store.install(wrapper)
    wrapper(first, layers=['blocks.0'])
    entries = sorted(store.folder.rglob('*.pkl'))
    warm = wrapper(second, layers=['blocks.0'])
    assert sorted(store.folder.rglob('*.pkl')) == entries
    store.enabled = False
    assert np.array_equal(warm.values, wrapper(second, layers=['blocks.0']).values)
