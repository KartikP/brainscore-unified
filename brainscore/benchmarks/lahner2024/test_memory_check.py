"""The memory check of Lahner2024 reads the inputs the way the run reads them.

The probe shows the subject one clip as the run shows it: the clip on ``vision:video``,
a frame of it on ``vision:image`` for a subject without clips, and its sound on
``audio``. Synthetic data behind the registered loaders (``tests/lahner_synthetic.py``).
"""
import logging

import pytest

import brainscore
from brainscore_core import memory
from tests.lahner_synthetic import Recorder, install


@pytest.mark.parametrize('identifier, inputs, regions, shown', [
    ('Lahner2024-fMRI-naturalistic', {'vision:video'}, ('IT',), ['vision:video']),
    ('Lahner2024-fMRI-naturalistic', {'vision:image'}, ('IT',), ['vision:image']),
    ('Lahner2024-fMRI-naturalistic-multimodal', {'vision:video', 'audio'}, ('IT', 'A1'),
     ['audio', 'vision:video']),
    ('Lahner2024-fMRI-naturalistic-multimodal', {'audio'}, ('IT', 'A1'), ['audio']),
], ids=['clip', 'frame', 'clip-and-sound', 'sound'])
def test_the_probe_shows_the_input_of_the_run(monkeypatch, tmp_path, caplog,
                                              identifier, inputs, regions, shown):
    install(monkeypatch, tmp_path)
    benchmark = brainscore.load_benchmark(identifier)
    subject = Recorder(inputs, regions=regions)
    monkeypatch.setattr(memory, 'get_host_available_memory', lambda: 64_000_000_000)
    with caplog.at_level(logging.INFO, logger='brainscore_core.memory'):
        assert memory.check_memory(subject, benchmark) == 'estimated'
    assert [channels for channels, _, _, _ in subject.seen] == [shown]
    assert len(subject.seen[0][2]) == 1                              # one presentation
    assert not any(record.exc_info for record in caplog.records)
    assert any('DECLARED-grounded' in record.getMessage() for record in caplog.records)


class _FailsFirstSession(Recorder):
    """Gives no answer in its first session (the probe); answers every later one."""

    def participate(self, session):
        if not self.seen:
            self.seen.append(None)
            raise RuntimeError('no width in the probe')
        super().participate(session)


def test_a_score_whose_memory_check_did_not_run_says_so(monkeypatch, tmp_path, caplog):
    """A probe that cannot run is one line with no traceback, and the score records it."""
    install(monkeypatch, tmp_path)
    benchmark = brainscore.load_benchmark('Lahner2024-fMRI-naturalistic')
    with caplog.at_level(logging.WARNING, logger='brainscore_core.memory'):
        score = brainscore.score(_FailsFirstSession({'vision:video'}), benchmark)
    warned = [record for record in caplog.records if record.name == 'brainscore_core.memory']
    assert len(warned) == 1 and warned[0].exc_info is None
    assert 'no width in the probe' in score.attrs['memory_check']
