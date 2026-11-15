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
    monkeypatch.setenv('HOME', str(tmp_path))  # the default folder is never the user's own
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

    def test_a_tilde_means_the_home_folder(self, asset, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv(asset.env_var, '~/frames')
        assert asset.resolved() == pathlib.Path.home() / 'frames'

    def test_a_tilde_with_a_literal_folder_here_is_refused(self, asset, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / '~' / 'frames').mkdir(parents=True)
        monkeypatch.setenv(asset.env_var, '~/frames')
        with pytest.raises(ValueError, match='ambiguous') as caught:
            asset.resolved()
        assert str(pathlib.Path.home() / 'frames') in str(caught.value)
        assert str(tmp_path / '~' / 'frames') in str(caught.value)
        row, = [row for row in local.status() if row['name'] == asset.name]
        assert row['present'] is False and row['path'].startswith('refused:')

    @pytest.mark.parametrize('value', ['', '  '])
    def test_an_empty_variable_is_refused(self, asset, monkeypatch, value):
        monkeypatch.setenv(asset.env_var, value)
        with pytest.raises(ValueError, match=f'{asset.env_var} is set but empty.*unset it'):
            asset.resolved()

    def test_an_empty_ewok_variable_is_refused_not_read_as_this_folder(self, monkeypatch):
        from brainscore.data.ewok.data import load_prepared
        monkeypatch.setenv('BRAINSCORE_EWOK_ROOT', '')
        with pytest.raises(ValueError, match='BRAINSCORE_EWOK_ROOT is set but empty'):
            load_prepared()

    def test_an_empty_friends_variable_is_refused_not_read_as_the_default(self, monkeypatch):
        from brainscore.data.algonauts2025 import default_data_root
        monkeypatch.setenv('BRAINSCORE_ALGONAUTS_PREPARED', '')
        with pytest.raises(ValueError, match='BRAINSCORE_ALGONAUTS_PREPARED is set but empty'):
            default_data_root()

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

    def test_names_are_stable_identifiers(self):
        for name in local.REGISTRY:
            assert name == name.lower() and ' ' not in name


class TestFriendsLocation:
    """The Friends benchmarks read two folders; each has one BRAINSCORE_ variable."""

    def test_two_assets_with_prefixed_variables(self):
        from brainscore.data.algonauts2025 import ALGONAUTS_PREPARED, ALGONAUTS_ROOT
        assert ALGONAUTS_ROOT.env_var == 'BRAINSCORE_ALGONAUTS_ROOT'
        assert ALGONAUTS_PREPARED.env_var == 'BRAINSCORE_ALGONAUTS_PREPARED'

    def test_data_code_reads_the_prepared_asset(self, tmp_path, monkeypatch):
        from brainscore.data.algonauts2025 import default_data_root
        monkeypatch.delenv('BRAINSCORE_ALGONAUTS_PREPARED', raising=False)
        assert default_data_root() == pathlib.Path.home() / '.brainio/algonauts2025'
        monkeypatch.setenv('BRAINSCORE_ALGONAUTS_PREPARED', str(tmp_path))
        assert default_data_root() == tmp_path

    def test_unprefixed_variable_is_ignored(self, tmp_path, monkeypatch):
        from brainscore.data.algonauts2025 import default_data_root
        monkeypatch.delenv('BRAINSCORE_ALGONAUTS_PREPARED', raising=False)
        monkeypatch.setenv('ALGONAUTS_DATA_ROOT', str(tmp_path))
        assert default_data_root() != tmp_path

    @pytest.mark.parametrize('identifier', [
        'Algonauts2025-friends-sub01', 'Algonauts2025-friends-sub05',
        'Algonauts2025-friends-sub02', 'Algonauts2025-friends-sub03',
        'Algonauts2025-friends-sub01-video',
        'Algonauts2025-friends-sub01-video-audio'])
    def test_precheck_covers_friends_identifiers(self, identifier, tmp_path, monkeypatch):
        monkeypatch.setenv('BRAINSCORE_ALGONAUTS_ROOT', str(tmp_path / 'native'))
        monkeypatch.setenv('BRAINSCORE_ALGONAUTS_PREPARED', str(tmp_path / 'prepared'))
        with pytest.raises(local.LocalDataMissing):
            local.check_benchmark_assets(identifier)
        (tmp_path / 'native').mkdir()
        with pytest.raises(local.LocalDataMissing, match='algonauts2025-prepared'):
            local.check_benchmark_assets(identifier)
        (tmp_path / 'prepared').mkdir()
        local.check_benchmark_assets(identifier)



def test_records_name_exactly_the_registered_benchmarks_that_read_them():
    """The pre-check in score() reads ``used_by``; a missing identifier skips the
    check, an extra one refuses a benchmark that does not need the asset."""
    from brainscore import benchmark_registry
    registered = set(benchmark_registry)

    def family(prefix):
        return {identifier for identifier in registered if identifier.startswith(prefix)}

    expected = {
        'algonauts2025-root': family('Algonauts2025-'),
        'algonauts2025-prepared': family('Algonauts2025-'),
        'coggym-checkout': family('CogGym.'),
        'ewok-core-1.0': family('EWoK-core-1.0-'),
        'lana-atlas': {'LeBel2023-UTS03-encoding-languagemask'},
        'lebel2023-pickle': set(),       # the registered variants read the hosted files
        # Needed only by the route a subject takes; the benchmark checks them when read.
        'lahner2024-frames': set(),
        'lahner2024-sound': set(),
    }
    none = {'lebel2023-pickle', 'lahner2024-frames', 'lahner2024-sound'}
    assert all(identifiers for name, identifiers in expected.items() if name not in none)
    assert {name: set(asset.used_by) for name, asset in local.REGISTRY.items()} == expected
    for name, asset in local.REGISTRY.items():
        assert len(asset.used_by) == len(set(asset.used_by)), f'{name} repeats an identifier'


def test_the_asset_list_needs_no_ml_import():
    """Each data plugin declares its assets in its own ``__init__.py``; listing them
    (``python -m brainscore.data``) imports no ML library."""
    import subprocess
    import sys
    code = (
        "import sys\n"
        "from brainscore.data.__main__ import main\n"
        "main([])\n"
        "loaded = [m for m in ('torch', 'transformers', 'cv2', 'sklearn') if m in sys.modules]\n"
        "assert not loaded, loaded\n")
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    for name in local.REGISTRY:
        assert name in result.stdout


def test_a_second_asset_with_the_same_name_is_refused(asset):
    with pytest.raises(ValueError, match='already declared'):
        local.register(asset)
