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


class _ReadsImagePaths(Recorder):
    """Answers ``IT`` from its pictures, read as the old vision extractor reads them
    (``stimulus_paths_of``); answers ``A1`` from the sound. Its regions are in the
    order of a model's ``region_layer_map`` (the picture region first)."""

    @property
    def region_layer_map(self):
        return dict.fromkeys(self._regions)

    def participate(self, session):
        from brainscore_vision.model_helpers.activations.core import stimulus_paths_of
        if 'neural:IT' in session.outputs:
            stimulus_paths_of(session.stimuli)
        super().participate(session)


@pytest.mark.parametrize('identifier, inputs, regions', [
    ('Lahner2024-fMRI-naturalistic', {'vision:image'}, ('IT',)),
    ('Lahner2024-fMRI-naturalistic-multimodal', {'vision:image', 'audio'}, ('IT', 'A1')),
], ids=['frame', 'frame-and-sound'])
def test_the_probe_names_its_pictures_for_every_extractor(monkeypatch, tmp_path,
                                                          identifier, inputs, regions):
    """The probe gives its frame in a column the old vision extractor reads too, so the
    width comes from the first region and not from a later one after a failure."""
    install(monkeypatch, tmp_path)
    benchmark = brainscore.load_benchmark(identifier)
    subject = _ReadsImagePaths(inputs, regions=regions)
    monkeypatch.setattr(memory, 'get_host_available_memory', lambda: 64_000_000_000)
    assert memory.check_memory(subject, benchmark) == 'estimated'
    assert [outputs for _, outputs, _, _ in subject.seen] == [['neural:IT']]
