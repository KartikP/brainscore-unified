"""Every registered model has the row its folder states, and the page shows the stated rows.

The first two tests read the rows of every registered plugin; the others use a made-up
registration in a temporary folder and read no real plugin. The checks are
subset facts: a new plugin or a new row does not break them.
"""
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.ways_to_answer import cells, check_statement, declared_row, markdown, stated_rows

pytestmark = pytest.mark.unit

PAGE = Path(__file__).resolve().parents[1] / 'docs' / 'reference' / 'ways_to_answer.md'
ROW = {'inputs': ['text'], 'regions': [], 'ways': [], 'answers': {'choice': 'own code'}}


@pytest.fixture
def made_up(tmp_path, monkeypatch):
    """A registry with one identifier, registered by a plugin folder in a temporary place."""
    folder = tmp_path / 'made_up_plugin'
    folder.mkdir()
    (folder / '__init__.py').write_text('def load():\n    raise AssertionError("nothing is built")\n')
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, 'made_up_plugin', raising=False)   # each test its own folder
    return folder, {'made-up-model': importlib.import_module('made_up_plugin').load}


def test_every_registered_model_has_a_stated_row():
    from brainscore import model_registry
    stated_rows(model_registry)


def test_each_model_on_the_page_shows_its_stated_row():
    from brainscore import model_registry
    rows = stated_rows(model_registry)
    for line in PAGE.read_text().splitlines():
        if line.startswith('| `'):
            models, shown = line[2:].split(' | ', 1)
            for identifier in models.split(', '):
                assert cells(rows[identifier.strip('`')]) == shown, identifier


def test_a_made_up_registration_with_its_row_is_listed(made_up):
    folder, registry = made_up
    (folder / 'ways_to_answer.json').write_text(json.dumps({'made-up-model': ROW}))
    assert stated_rows(registry) == {'made-up-model': ROW}
    assert f'| `made-up-model` | {cells(ROW)}' in markdown(stated_rows(registry))


def test_a_made_up_registration_without_its_row_names_the_folder_and_the_line(made_up):
    _, registry = made_up
    with pytest.raises(ValueError) as error:
        stated_rows(registry)
    message = str(error.value)
    assert "'made-up-model' (folder made_up_plugin)" in message
    assert 'made_up_plugin/ways_to_answer.json' in message and 'check_statement' in message


def test_check_statement_compares_a_subject_of_its_own_code(made_up):
    folder, registry = made_up
    subject = SimpleNamespace(identifier='made-up-model', in_channels={'text'}, out_channels={'behavior:choice'})
    (folder / 'ways_to_answer.json').write_text(json.dumps({'made-up-model': ROW}))
    check_statement(subject, registry=registry)
    (folder / 'ways_to_answer.json').write_text(json.dumps({'made-up-model': {**ROW, 'inputs': ['audio']}}))
    with pytest.raises(AssertionError, match='its subject declares'):
        check_statement(subject, registry=registry)


def test_check_statement_reads_the_wired_parts(made_up):
    from brainscore import BrainScoreModel
    folder, registry = made_up
    subject = BrainScoreModel('made-up-model', in_channels={'text'}, likelihood=lambda prompt, options: {})
    row = declared_row(subject, subject)
    assert 'likelihood' in row['ways'] and row['answers']['log_probabilities'] == 'likelihood'
    (folder / 'ways_to_answer.json').write_text(json.dumps({'made-up-model': row}))
    check_statement(subject, subject, registry=registry)
    with pytest.raises(AssertionError, match='its subject declares'):
        check_statement(subject, registry=registry)      # stated with parts, checked as own code
