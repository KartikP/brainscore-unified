"""Lahner2024 shows a subject clips when it declares it takes them (vision:video),
and frames sampled from each clip otherwise. The route comes from declarations,
never from raw extractor keys."""
import pytest

from brainscore.benchmarks.lahner2024 import Lahner2024BOLDMoments
from tests.lahner_synthetic import Recorder, install


@pytest.mark.parametrize('in_channels, shown, pipeline', [
    ({'vision:video'}, 'vision:video', 'video_native'),
    ({'vision:image'}, 'vision:image', 'frame_aggregation'),
    ({'vision:image', 'vision:video'}, 'vision:video', 'video_native'),
])
def test_the_declared_form_decides_what_is_shown(monkeypatch, tmp_path, in_channels, shown, pipeline):
    install(monkeypatch, tmp_path)
    subject = Recorder(in_channels)
    score = Lahner2024BOLDMoments()(subject)
    assert [inputs for inputs, _, _, _ in subject.seen] == [[shown]]
    assert score.attrs['pipeline'] == pipeline
