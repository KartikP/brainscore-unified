"""CI must test the PR head, including fork PRs, with its pinned peers."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from brainscore.validation.revisions import resolve, verify


PINS = {'core': 'a' * 40, 'vision': 'b' * 40, 'language': 'c' * 40}
UNIFIED = 'd' * 40
HEAD = 'e' * 40


def candidate_record(candidate='core', **kwargs):
    arguments = dict(candidate=candidate, repository=f'contributor/{candidate}',
                     revision=HEAD, unified_repository='KartikP/brainscore-unified',
                     unified_revision=UNIFIED)
    arguments.update(kwargs)
    return resolve(PINS, **arguments)


@pytest.mark.parametrize('candidate', ['core', 'vision', 'language'])
def test_only_the_candidate_uses_the_fork_head(candidate):
    record = candidate_record(candidate)
    assert record['checkouts'][candidate] == {
        'repository': f'contributor/{candidate}', 'revision': HEAD,
    }
    for peer in PINS.keys() - {candidate}:
        assert record['checkouts'][peer] == {
            'repository': f'brain-score/{peer}', 'revision': PINS[peer],
        }
    assert record['checkouts']['unified']['revision'] == UNIFIED


def test_unified_fork_uses_its_own_manifest_and_head():
    record = candidate_record('unified', unified_repository='contributor/unified',
                              unified_revision=HEAD)
    assert record['checkouts']['unified']['revision'] == HEAD
    assert record['checkouts']['core']['revision'] == PINS['core']


@pytest.mark.parametrize('values', [
    {'candidate': 'unknown'}, {'revision': 'main'}, {'revision': 'e' * 7},
    {'repository': 'fork/repo\ncore_revision=main'},
    {'repository': 'https://example.com/repo'}, {'unified_revision': 'v2'},
    {'candidate': 'unified'},
])
def test_invalid_candidate_fails_before_checkout(values):
    with pytest.raises(ValueError):
        candidate_record(**values)


def test_invalid_pin_cannot_be_hidden_by_an_override():
    with pytest.raises(ValueError, match='full lowercase'):
        resolve({**PINS, 'core': 'main'}, candidate='core', repository='fork/core',
                revision=HEAD, unified_repository='KartikP/brainscore-unified',
                unified_revision=UNIFIED)


def test_checkout_mismatch_blocks_the_build(tmp_path, monkeypatch):
    record = candidate_record()
    def git_head(command, **kwargs):
        name = Path(command[2]).name
        return record['checkouts'][name]['revision'] + '\n'
    monkeypatch.setattr(subprocess, 'check_output', git_head)
    verify(record, tmp_path, list(record['checkouts']))
    monkeypatch.setattr(subprocess, 'check_output', lambda *args, **kwargs: 'f' * 40)
    with pytest.raises(ValueError, match='requested .* checked out'):
        verify(record, tmp_path, ['core'])


def test_bootstrap_cli_needs_no_installed_brainscore_package(tmp_path):
    import brainscore.validation.revisions as revisions

    manifest = tmp_path / 'pins.json'
    manifest.write_text(json.dumps(PINS))
    record, output = tmp_path / 'record.json', tmp_path / 'outputs'
    result = subprocess.run(
        [sys.executable, '-I', '-S', revisions.__file__, '--manifest', str(manifest),
         '--record', str(record)], capture_output=True, text=True,
        env={**os.environ, 'CANDIDATE': 'language',
             'CANDIDATE_REPOSITORY': 'fork/language', 'CANDIDATE_REVISION': HEAD,
             'UNIFIED_REPOSITORY': 'KartikP/brainscore-unified',
             'UNIFIED_REVISION': UNIFIED, 'GITHUB_OUTPUT': str(output)},
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(record.read_text())['checkouts']['language']['revision'] == HEAD
    assert f'language_revision={HEAD}\n' in output.read_text()
