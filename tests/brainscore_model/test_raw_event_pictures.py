"""A picture stream with no stimulus table reaches the old vision extractor.

The session rebuilds the table from the events; the pictures must be in a column
the old vision extractor (``stimulus_paths_of``) reads, as they are in a table run.
"""
import numpy as np
import pytest
import torch
from PIL import Image

from brainscore import BrainScoreModel
from brainscore_core.streaming import Session, StreamEvent

pytestmark = pytest.mark.unit


def test_a_raw_picture_stream_reaches_the_old_vision_extractor(tmp_path, monkeypatch):
    from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: False)
    monkeypatch.setenv('RESULTCACHING_DISABLE', '1')
    paths = []
    for i in range(2):
        path = tmp_path / f'p{i}.png'
        Image.fromarray(np.full((8, 8, 3), 40 * i, np.uint8)).save(path)
        paths.append(str(path))

    def load(files):
        return np.stack([np.asarray(Image.open(f), np.float32).transpose(2, 0, 1) / 255 for f in files])
    network = torch.nn.Sequential(torch.nn.Conv2d(3, 2, 3), torch.nn.Flatten())
    subject = BrainScoreModel('raw-pictures', extractor={
        'vision': PytorchWrapper(network, load, identifier='raw-pictures-net', batch_size=2)},
        region_layer_map={'IT': '0'})
    events = [StreamEvent('vision:image', path, float(i), {'stimulus_id': f'p{i}'})
              for i, path in enumerate(paths)]
    session = Session(events, ['neural:IT'])
    subject.participate(session)
    assert session.collect('neural:IT').sizes['presentation'] == 2
