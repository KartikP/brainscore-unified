"""Every registered model builds with stand-in parts, and each kind of answer it declares has a way.

The table tool (``tests/ways_to_answer.py``) runs in its own process, because it replaces
``transformers``, the extractors and the checkpoint loaders for the whole process. The
assertions are subset facts: a new model or a new way does not break them.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

UNIFIED = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def rows():
    result = subprocess.run([sys.executable, '-m', 'tests.ways_to_answer', '--json'], cwd=UNIFIED,
                            capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr[-2000:]
    return json.loads(result.stdout)


def test_every_registered_model_builds_with_stand_ins(rows):
    assert {identifier: row['error'] for identifier, row in rows.items() if 'error' in row} == {}


def test_every_declared_kind_of_answer_has_a_way(rows):
    for identifier, row in rows.items():
        assert all(row['answers'].values()), identifier


def test_a_qwen25_vl_model_has_writing_likelihood_and_readout(rows):
    row = rows['qwen2.5-vl-3b']
    assert {'writing', 'likelihood', 'readout'} <= set(row['ways'])
    assert {'text': 'writing', 'log_probabilities': 'likelihood'}.items() <= row['answers'].items()


def test_clip_answers_by_its_readout(rows):
    row = rows['clip-vit-b-32']
    assert 'readout' in row['ways']
    assert row['answers']['choice'] == 'readout'
    assert {'V1', 'IT', 'language_system'} <= set(row['regions'])
