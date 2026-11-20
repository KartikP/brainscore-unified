"""The Friends table name identifies the input files, so every cache answers the current input.

Some stores key on the table name and the file paths only (the temporal helper of
old vision models). Others also hash the file bytes (the unified wrappers). The
table name must serve both: equal files in equal order share one name, so
participants share one stored entry; new bytes at an old path, or other rows,
give another name, so a store that keys on the name alone answers the current
frames.

A store reads its folder (``RESULTCACHING_HOME``) when its module is imported, so
each run with the store on is a child process with its own empty folder; the
store is switched off with ``RESULTCACHING_DISABLE``. Tiny synthetic frames,
stand-in networks.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from brainscore.data.algonauts2025 import per_scan_stimuli

_REPOSITORY = Path(__file__).resolve().parents[2]


def _films(root, movies=('m1',)):
    return pd.DataFrame({'stimulus_id': list(movies), 'video_path': [str(root / f'{m}.mkv') for m in movies]})


def _frame(root, movie, t, value):
    path = root / 'frames' / movie / f'{t:04d}.jpg'
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new('RGB', (16, 16), (value, value, value)).save(path)


def _table(root, participant, n_scans):
    return per_scan_stimuli(_films(root), {'m1': n_scans}, ['frames'], 'friends', participant, root)


def _child(step, root, storage='on'):
    """Run ``step(root)`` of this module in a child process whose store is the empty folder ``root/store``."""
    root = Path(root)
    env = {**os.environ, 'RESULTCACHING_HOME': str(root / 'store'),
           'RESULTCACHING_DISABLE': '0' if storage == 'on' else '1',
           'PYTHONPATH': os.pathsep.join([str(_REPOSITORY), *sys.path])}
    subprocess.run([sys.executable, '-c', f'from {__name__} import {step}; {step}({str(root)!r})'],
                   cwd=_REPOSITORY, env=env, check=True)
    return json.loads((root / f'{step}.json').read_text())


def _entries(root):
    return sorted(str(path) for path in (Path(root) / 'store').rglob('*') if path.is_file())


# -- a store keyed on the name answers the current frames ---------------------

def _name_only_reader():
    """A stand-in store that keys on the table name alone: the mean pixel of each frame."""
    stored = {}

    def read(table, cached):
        if not cached or table.identifier not in stored:
            stored[table.identifier] = np.array([np.asarray(Image.open(path)).mean()
                                                 for path in table['image_file_name']])
        return stored[table.identifier] if cached else stored.pop(table.identifier)
    return read


def _temporal_reader():
    """The temporal helper of old vision models with a stand-in network (child process only)."""
    from collections import OrderedDict
    from brainscore_vision.model_helpers.activations.temporal.core import TemporalInferencer
    from brainscore_vision.model_helpers.activations.temporal.model import ActivationWrapper

    class StandIn(ActivationWrapper):
        def get_activations(self, inputs, layers):
            return OrderedDict((layer, np.stack(inputs)) for layer in layers)

    model = StandIn('stand-in', lambda video: video.to_numpy()[:, :8, :8], inferencer_cls=TemporalInferencer,
                    fps=1, layer_activation_format={'layer1': 'THWC'}, max_workers=1, batch_size=4)

    def read(table, cached):
        os.environ['RESULTCACHING_DISABLE'] = '0' if cached else '1'
        return model(table, layers=['layer1']).values.copy()
    return read


def _replaced_frame(read, root):
    _frame(root, 'm1', 1, 0)
    read(_table(root, 1, 1), cached=True)
    _frame(root, 'm1', 1, 255)
    warm = read(_table(root, 2, 1), cached=True)
    return bool(np.array_equal(warm, read(_table(root, 2, 1), cached=False)))


def _other_rows(read, root):
    _frame(root, 'm1', 1, 0)
    _frame(root, 'm1', 2, 255)
    read(_table(root, 1, 2), cached=True)
    warm = read(_table(root, 2, 1), cached=True)
    return bool(np.array_equal(warm, read(_table(root, 2, 1), cached=False)))


def temporal_replaced_frame(root):
    root = Path(root)
    (root / 'temporal_replaced_frame.json').write_text(json.dumps(_replaced_frame(_temporal_reader(), root)))


def temporal_other_rows(root):
    root = Path(root)
    (root / 'temporal_other_rows.json').write_text(json.dumps(_other_rows(_temporal_reader(), root)))


def test_a_store_keyed_on_the_name_answers_a_frame_replaced_at_the_same_path(tmp_path):
    assert _replaced_frame(_name_only_reader(), tmp_path / 'stand-in')
    assert _child('temporal_replaced_frame', tmp_path / 'temporal')


def test_a_store_keyed_on_the_name_answers_a_participant_with_other_rows(tmp_path):
    assert _other_rows(_name_only_reader(), tmp_path / 'stand-in')
    assert _child('temporal_other_rows', tmp_path / 'temporal')


# -- participants with equal frames share one stored entry --------------------

FORWARDS = []


class _Network(torch.nn.Module):
    """A stand-in vision encoder: the mean colour of each image through one layer; counts its forward calls."""

    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([torch.nn.Linear(3, 2)])

    def forward(self, pixel_values):
        FORWARDS.append(len(pixel_values))
        return self.blocks[0](pixel_values)


class _Processor:
    def __call__(self, images=None, return_tensors='pt'):
        return {'pixel_values': torch.tensor(np.stack(
            [np.asarray(image, dtype=np.float32).mean(axis=(0, 1)) for image in images]))}


def participants_share_one_entry(root):
    from brainscore.model_helpers.vlm_vision_wrapper import VLMVisionWrapper
    root = Path(root)
    torch.manual_seed(0)
    wrapper = VLMVisionWrapper(_Network(), _Processor(), identifier='stand-in', device='cpu')
    seen = {}

    def run(name, participant):
        FORWARDS.clear()
        values = wrapper(_table(root, participant, 2), layers=['blocks.0']).values
        seen[name] = {'entries': len(_entries(root)), 'forwards': len(FORWARDS), 'values': values.tolist()}

    for t, value in ((1, 30), (2, 200)):
        _frame(root, 'm1', t, value)
    run('participant 1', 1)
    run('participant 2', 2)
    _frame(root, 'm1', 2, 90)
    run('changed file', 2)
    (root / 'participants_share_one_entry.json').write_text(json.dumps(seen))


def test_participants_with_equal_frames_share_one_stored_entry(tmp_path):
    seen = _child('participants_share_one_entry', tmp_path)
    first, second, changed = seen['participant 1'], seen['participant 2'], seen['changed file']
    assert first['entries'] == 1 and first['forwards'] > 0
    assert second['entries'] == 1 and second['forwards'] == 0
    assert second['values'] == first['values']
    assert changed['entries'] == 2 and changed['forwards'] > 0
