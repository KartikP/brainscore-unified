"""Tests for the user-supplied-asset registry.

The point of this layer is that a missing file produces instructions rather
than a stack trace, so what is tested is mostly the message.
"""

import pathlib

import pytest

from brainscore.data import local
from tests.heavy_import import heavy_import_message

pytestmark = pytest.mark.unit


@pytest.fixture
def asset(tmp_path, monkeypatch):
    declared = local.LocalAsset(
        name='test-asset', kind='stimuli',
        why_local='Licensed film, not ours to hand out.',
        source='https://example.invalid/data',
        obtain='The whole thing, about 2 GB.',
        prepare='python -m brainscore.data.example.prepare --root <root>',
        used_by=['Example-benchmark'])
    monkeypatch.setitem(local.REGISTRY, declared.name, declared)
    monkeypatch.setenv('BRAINSCORE_DATA_HOME', str(tmp_path / 'data'))  # never the user's own folder
    return declared


class TestResolution:
    def test_an_asset_is_under_the_data_home_by_its_name(self, asset, tmp_path):
        assert asset.resolved() == tmp_path / 'data' / 'test-asset'

    def test_the_default_data_home_is_in_the_home_folder(self, asset, monkeypatch):
        monkeypatch.delenv('BRAINSCORE_DATA_HOME')
        assert asset.resolved() == pathlib.Path.home() / 'brainscore-data' / 'test-asset'

    @pytest.mark.parametrize('value', ['', '   ', 'relative', '.', '~relative-user-name-that-is-absent'])
    def test_an_empty_or_relative_data_home_is_refused_and_makes_no_folder(
            self, asset, tmp_path, monkeypatch, value):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('BRAINSCORE_DATA_HOME', value)
        before = sorted(tmp_path.iterdir())
        with pytest.raises(ValueError, match='BRAINSCORE_DATA_HOME'):
            asset.resolved()
        with pytest.raises(ValueError, match='BRAINSCORE_DATA_HOME'):
            from brainscore.data import package_data
            package_data('ewok', source=tmp_path)
        assert sorted(tmp_path.iterdir()) == before

    def test_a_tilde_data_home_is_the_home_folder(self, asset, monkeypatch):
        monkeypatch.setenv('BRAINSCORE_DATA_HOME', '~/elsewhere')
        assert asset.resolved() == pathlib.Path.home() / 'elsewhere' / 'test-asset'

    def test_present_file_resolves(self, asset):
        asset.resolved().parent.mkdir(parents=True)
        asset.resolved().write_bytes(b'x')
        assert local.path(asset.name) == asset.resolved()

    def test_a_link_at_the_place_is_read(self, asset, tmp_path):
        (tmp_path / 'other-drive').mkdir()
        asset.resolved().parent.mkdir(parents=True)
        asset.resolved().symlink_to(tmp_path / 'other-drive')
        assert local.path(asset.name) == asset.resolved()

    def test_unknown_asset_names_the_known_ones(self):
        with pytest.raises(KeyError, match='known:'):
            local.path('no-such-asset')


def test_every_registered_asset_resolves_under_the_data_home(tmp_path, monkeypatch):
    monkeypatch.setenv('BRAINSCORE_DATA_HOME', str(tmp_path))
    for name, asset in local.REGISTRY.items():
        assert asset.resolved() == tmp_path / name


class TestTheMessage:
    """A user who hits this has to be able to act on it without reading code."""

    def test_missing_asset_explains_itself(self, asset):
        with pytest.raises(local.LocalDataMissing) as caught:
            local.path(asset.name)
        message = str(caught.value)
        for expected in (asset.source, str(asset.resolved()), 'BRAINSCORE_DATA_HOME', asset.obtain,
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
    """The Friends benchmarks read one folder: the prepared files."""

    def test_data_code_reads_the_prepared_asset(self, tmp_path, monkeypatch):
        from brainscore.data.algonauts2025 import default_data_root
        monkeypatch.setenv('BRAINSCORE_DATA_HOME', str(tmp_path))
        assert default_data_root() == tmp_path / 'algonauts2025-prepared'

    @pytest.mark.parametrize('identifier', [
        'Algonauts2025-friends-sub01', 'Algonauts2025-friends-sub05',
        'Algonauts2025-friends-sub02', 'Algonauts2025-friends-sub03',
        'Algonauts2025-friends-sub01-video',
        'Algonauts2025-friends-sub01-video-audio'])
    def test_precheck_covers_friends_identifiers(self, identifier, tmp_path, monkeypatch):
        monkeypatch.setenv('BRAINSCORE_DATA_HOME', str(tmp_path))
        with pytest.raises(local.LocalDataMissing, match='algonauts2025-prepared'):
            local.check_benchmark_assets(identifier)
        (tmp_path / 'algonauts2025-prepared').mkdir()
        local.check_benchmark_assets(identifier)


def test_records_name_only_registered_benchmarks_and_every_known_reader():
    """The pre-check in score() reads ``used_by``; a missing identifier skips the
    check, an extra one refuses a benchmark that does not need the asset.

    For every asset: ``used_by`` names only registered benchmarks, with no repeats.
    For the assets below, whose readers are known: the asset is still declared and
    ``used_by`` still lists each known reader. A new asset or a new reader needs no line here."""
    from brainscore import benchmark_registry
    registered = set(benchmark_registry)
    for name, asset in local.REGISTRY.items():
        assert set(asset.used_by) <= registered, \
            f'{name}: used_by names unregistered benchmarks {sorted(set(asset.used_by) - registered)}'
        assert len(asset.used_by) == len(set(asset.used_by)), f'{name} repeats an identifier'

    def family(prefix):
        return {identifier for identifier in registered if identifier.startswith(prefix)}

    known = {
        'algonauts2025-prepared': family('Algonauts2025-'),
        'coggym-checkout': family('CogGym.'),
        'ewok-core-1.0': family('EWoK-core-1.0-'),
        'lana-atlas': {'LeBel2023-UTS03-encoding-languagemask'},
        # Needed only by the route a subject takes; the benchmark checks them when read.
        'lahner2024-frames': set(),
        'lahner2024-sound': set(),
    }
    none = {'lahner2024-frames', 'lahner2024-sound'}
    assert all(identifiers for name, identifiers in known.items() if name not in none)
    for name, readers in known.items():
        assert name in local.REGISTRY, f'the local asset {name!r} is no longer declared'
        missing = sorted(readers - set(local.REGISTRY[name].used_by))
        assert not missing, f'{name}: used_by no longer lists its known readers {missing}'


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
    assert result.returncode == 0, heavy_import_message('Listing the local assets', result.stderr)
    for name in local.REGISTRY:
        assert name in result.stdout


def test_a_second_asset_with_the_same_name_is_refused(asset):
    with pytest.raises(ValueError, match='already declared'):
        local.register(asset)


def test_no_distributed_text_names_a_deleted_location_variable():
    """The pages under docs/archive are history and say so; every other text gives the data home."""
    deleted = ('BRAINSCORE_HOME', 'BRAINSCORE_ALGONAUTS_PREPARED', 'BRAINSCORE_ALGONAUTS_ROOT',
               'BRAINSCORE_EWOK_ROOT', 'BRAINSCORE_COGGYM_CHECKOUT', 'BRAINSCORE_LAHNER_FRAMES_DIR',
               'BRAINSCORE_LAHNER_SOUND_DIR', 'BRAINSCORE_LANA_ATLAS', 'BRAINSCORE_LEBEL_CACHE',
               'BRAINSCORE_LEBEL_PICKLE')
    root = pathlib.Path(__file__).resolve().parents[1]
    found = []
    for path in root.rglob('*'):
        relative = path.relative_to(root).as_posix()
        if (not path.is_file() or path.suffix not in ('.py', '.md', '.ipynb', '.yml', '.yaml', '.toml', '.txt')
                or relative.startswith(('docs/archive/', '.git/')) or path == pathlib.Path(__file__).resolve()):
            continue
        text = path.read_text(errors='ignore')
        found += [f'{relative}: {name}' for name in deleted if name in text]
    assert not found, found


@pytest.mark.parametrize('document', ['brainscore/data/lebel2023/README.md', 'notebooks/figure_sources/README.md',
                                      'notebooks/assets/whole_brain/README.md',
                                      'notebooks/archive/16_whole_brain_encoding.ipynb'])
def test_the_lebel_source_command_places_the_pickle_where_the_loader_reads_it(document, tmp_path):
    """A data home that starts with ~ is expanded as everywhere else."""
    import json
    import os
    import re
    import subprocess
    import sys
    root = pathlib.Path(__file__).resolve().parents[1]
    text = (root / document).read_text()
    if document.endswith('.ipynb'):
        text = ''.join(line for cell in json.loads(text)['cells'] for line in cell['source'])
    command = re.search(r'^SOURCE=.*?mv [^\n]*', text, flags=re.M | re.S).group(0)
    pickle = tmp_path / 'download' / 'assembly_lebel_uts03.pkl'
    pickle.parent.mkdir()
    pickle.write_bytes(b'pickle')
    command = re.sub(r'/path/to/\S+\.pkl', str(pickle), command)
    environment = dict(os.environ, HOME=str(tmp_path / 'user'), BRAINSCORE_DATA_HOME='~/data',
                       PATH=f'{pathlib.Path(sys.executable).parent}{os.pathsep}{os.environ["PATH"]}')
    subprocess.run(['bash', '-c', command], env=environment, cwd=tmp_path, check=True)
    assert (tmp_path / 'user' / 'data' / 'sources' / 'lebel2023' / 'assembly_lebel_uts03.pkl').read_bytes() == b'pickle'
