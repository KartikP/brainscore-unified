"""The Lahner2024 CSV sidecars go through the one fetch function that checks the sha1."""
import hashlib

import pytest

from brainscore_core.supported_data_standards.brainio import fetch

pytestmark = pytest.mark.unit

EVENTS = b'stimulus_id,onset\nstimulus_0001,0.0\n'


@pytest.fixture
def downloads(tmp_path, monkeypatch):
    """No network: the fetcher writes the given bytes where it would put the download."""
    monkeypatch.setattr(fetch, '_local_data_path', str(tmp_path))
    seen = []

    def download(fetcher, body):
        seen.append((fetcher.bucketname, fetcher.relative_path, fetcher.extra_args))
        with open(fetcher.output_filename, 'wb') as file:
            file.write(body)

    def install(body):
        monkeypatch.setattr(fetch.BotoFetcher, 'download_boto', lambda self: download(self, body))
        return seen
    return install


def _events_with_sha1(monkeypatch, sha1):
    import brainscore.data.lahner2024 as lahner
    monkeypatch.setattr(lahner, 'TIMERESOLVED_EVENTS_SHA1', sha1)
    return lahner.load_lahner2024_timeresolved_events()


def test_the_sidecar_is_read_from_its_versioned_key(downloads, monkeypatch):
    import brainscore.data.lahner2024 as lahner
    seen = downloads(EVENTS)
    events = _events_with_sha1(monkeypatch, hashlib.sha1(EVENTS).hexdigest())
    assert list(events['stimulus_id']) == ['stimulus_0001']
    assert seen == [('brainscore-storage',
                     'brainscore-vision/benchmarks/Lahner2024-fMRI/Lahner2024-fMRI-timeresolved-events.csv',
                     {'VersionId': lahner.TIMERESOLVED_EVENTS_VERSION_ID})]


def test_changed_bytes_are_refused(downloads, monkeypatch):
    downloads(EVENTS + b'stimulus_0002,1.0\n')
    with pytest.raises(ValueError, match='SHA1 mismatch'):
        _events_with_sha1(monkeypatch, hashlib.sha1(EVENTS).hexdigest())
