"""Tests for the user-supplied-asset registry.

The point of this layer is that a missing file produces instructions rather
than a stack trace, so what is tested is mostly the message.
"""

import pathlib

import pytest

from brainscore.data import local

pytestmark = pytest.mark.unit


@pytest.fixture
def asset(tmp_path, monkeypatch):
    declared = local.LocalAsset(
        name='test-asset', env_var='BRAINSCORE_TEST_ASSET',
        default_path='nowhere/at/all.pkl', kind='stimuli',
        why_local='Licensed film, not ours to hand out.',
        source='https://example.invalid/data',
        obtain='The whole thing, about 2 GB.',
        prepare='python -m brainscore.data.example.prepare --root <root>',
        used_by=['Example-benchmark'])
    monkeypatch.setitem(local.REGISTRY, declared.name, declared)
    monkeypatch.delenv(declared.env_var, raising=False)
    return declared


class TestResolution:
    def test_environment_overrides_the_default(self, asset, tmp_path, monkeypatch):
        monkeypatch.setenv(asset.env_var, str(tmp_path / 'elsewhere.pkl'))
        assert asset.resolved() == tmp_path / 'elsewhere.pkl'

    def test_default_is_under_home(self, asset):
        assert asset.resolved() == pathlib.Path.home() / asset.default_path

    def test_present_file_resolves(self, asset, tmp_path, monkeypatch):
        target = tmp_path / 'here.pkl'
        target.write_bytes(b'x')
        monkeypatch.setenv(asset.env_var, str(target))
        assert local.path(asset.name) == target

    def test_unknown_asset_names_the_known_ones(self):
        with pytest.raises(KeyError, match='known:'):
            local.path('no-such-asset')


class TestTheMessage:
    """A user who hits this has to be able to act on it without reading code."""

    def test_missing_asset_explains_itself(self, asset):
        with pytest.raises(local.LocalDataMissing) as caught:
            local.path(asset.name)
        message = str(caught.value)
        for expected in (asset.source, asset.env_var, asset.obtain,
                         asset.why_local, asset.prepare, 'Example-benchmark'):
            assert expected in message, f'message omits {expected!r}'

    def test_it_is_a_filenotfounderror(self, asset):
        """Callers that already catch FileNotFoundError keep working."""
        assert issubclass(local.LocalDataMissing, FileNotFoundError)
        with pytest.raises(FileNotFoundError):
            local.path(asset.name)


class TestStatus:
    def test_reports_presence_without_raising(self, asset):
        rows = {row['name']: row for row in local.status()}
        assert rows[asset.name]['present'] is False
        assert rows[asset.name]['kind'] == 'stimuli'

    def test_format_lists_every_asset(self, asset):
        rendered = local.format_status()
        for row in local.status():
            assert row['name'] in rendered
        assert 'present' in rendered


class TestManifest:
    """Every declared asset must be actionable, not just declared."""

    def test_real_assets_carry_a_source_and_a_reason(self):
        assert local.REGISTRY, 'no assets declared'
        for name, asset in local.REGISTRY.items():
            assert asset.source.startswith('http'), f'{name} has no source'
            assert asset.why_local.strip(), f'{name} does not say why it is local'
            assert asset.obtain.strip(), f'{name} does not say what to download'
            assert asset.used_by, f'{name} is needed by nothing'

    def test_names_are_stable_identifiers(self):
        for name in local.REGISTRY:
            assert name == name.lower() and ' ' not in name
