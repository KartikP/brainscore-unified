"""``package_data(name)``: every data plugin says how its files are packaged from the source.

The first two tests are an intended gate: a new data plugin fails them until it
registers a ``DataBuilder`` and its README carries the ``## Rebuild`` block of that
declaration, and every builder has a tiny made-up source below (also a gate). The
others run each builder through the public command on that source, with sockets,
network programs (git, aws) and S3 replaced by stand-ins whose uploads fail.
"""
import dataclasses
import hashlib
import importlib.util
import pathlib
import re
import shlex
import socket
import sys
import types

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import brainscore
import brainscore.data
from brainscore import data_builder_registry
from brainscore.data import DataBuilder, package_data
from brainscore.data import local
from brainscore.data.local import LocalDataMissing

pytestmark = pytest.mark.unit

DATA = pathlib.Path(brainscore.data.__file__).parent
PLUGINS = sorted(p for p in DATA.iterdir() if (p / '__init__.py').exists() and not p.name.startswith('_'))


def _rebuild_blocks(readme: pathlib.Path) -> dict:
    """``{name: block}`` of the ``### <name>`` blocks of a README's ``## Rebuild`` section."""
    text = readme.read_text() if readme.exists() else ''
    if '## Rebuild' not in text:
        return {}
    section = re.split(r'^## (?!Rebuild)', text.split('## Rebuild', 1)[1], flags=re.M)[0]
    return {block.split('\n', 1)[0].strip(): ('### ' + block).strip()
            for block in section.split('### ')[1:]}


@pytest.mark.parametrize('plugin', PLUGINS, ids=lambda p: p.name)
def test_every_data_plugin_states_how_its_data_is_packaged(plugin):
    # Intended gate: a data plugin without these fails, and the message says what to add.
    blocks = _rebuild_blocks(plugin / 'README.md')
    assert blocks, (
        f'{plugin.name} does not say how its data is packaged. Register a DataBuilder in '
        f'brainscore/data/{plugin.name}/__init__.py (data_builder_registry[<name>] = DataBuilder(...): '
        f'the builder command, or cannot=..., or nothing to package), then add a section '
        f"'## Rebuild' to its README.md with DataBuilder.readme(<name>). See docs/guides/data.md.")
    for name, block in blocks.items():
        assert name in data_builder_registry, f'{plugin.name}/README.md names {name!r}, which no plugin registers'
        expected = data_builder_registry[name]().readme(name)
        assert block == expected, (
            f'{plugin.name}/README.md: the Rebuild block of {name} differs from its declaration. '
            f'Paste this block:\n\n{expected}')


def test_every_declaration_is_in_a_readme_and_answers_package_data():
    documented = set().union(*(_rebuild_blocks(plugin / 'README.md') for plugin in PLUGINS))
    for name, factory in data_builder_registry.items():
        builder = factory()
        assert name in documented, f'{name} has no Rebuild block in its plugin README.md'
        assert isinstance(builder, DataBuilder)
        assert not (builder.build and builder.cannot), f'{name} has a builder and says it cannot'
        if builder.build:
            module = builder.build.split('python -m ', 1)[1].split()[0]
            assert importlib.util.find_spec(module) is not None, f'{name}: {module} does not exist'


class _S3:
    """A stand-in S3 client: lists and downloads the objects a test gives; any upload fails."""

    def __init__(self, objects):
        self.objects = objects

    def get_paginator(self, operation):
        objects = self.objects
        return types.SimpleNamespace(paginate=lambda Bucket, Prefix: [
            {'Contents': [{'Key': key} for key in sorted(objects) if key.startswith(Prefix)]}])

    def download_file(self, bucket, key, path):
        pathlib.Path(path).write_bytes(self.objects[key])

    def __getattr__(self, name):           # upload_file, put_object, head_object, ...
        raise AssertionError(f'package_data called S3 {name}: nothing may be uploaded')


_NETWORK_PROGRAMS = {'git', 'aws', 'curl', 'wget', 'pip', 'hf', 'huggingface-cli', 'datalad', 'ssh',
                     'scp', 'rsync', 'gsutil'}


class _Offline:
    """Stand-ins a test fills: S3 objects, and programs other than ffmpeg (git, aws)."""

    def __init__(self):
        self.objects, self.programs = {}, {}


@pytest.fixture
def offline(monkeypatch, tmp_path):
    """No network: sockets fail, S3 is a stand-in whose uploads fail, and a program that
    reaches the network (git, aws, ...) runs only as a stand-in the test gives. The default source and output root
    is a temporary folder."""
    import subprocess
    stand_ins = _Offline()

    def refuse(*args, **kwargs):
        raise AssertionError('package_data used the network')
    monkeypatch.setattr(socket.socket, 'connect', refuse)
    monkeypatch.setattr(socket, 'create_connection', refuse)
    real_run = subprocess.run

    def run(argv, *args, **kwargs):
        program = pathlib.Path(str(argv[0] if isinstance(argv, (list, tuple)) else argv)).name
        if program in stand_ins.programs:
            return stand_ins.programs[program](list(map(str, argv)))
        if program in _NETWORK_PROGRAMS:
            raise AssertionError(f'package_data ran {argv}')
        return real_run(argv, *args, **kwargs)
    monkeypatch.setattr(subprocess, 'run', run)
    client = lambda *args, **kwargs: _S3(stand_ins.objects)
    boto3 = types.ModuleType('boto3')
    boto3.client = client
    boto3.session = types.SimpleNamespace(Config=lambda **kwargs: None, UNSIGNED=None)
    botocore = types.ModuleType('botocore')
    botocore.UNSIGNED = None
    botocore_config = types.ModuleType('botocore.config')
    botocore_config.Config = lambda **kwargs: None
    for name, module in (('boto3', boto3), ('botocore', botocore), ('botocore.config', botocore_config)):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setenv('BRAINSCORE_DATA_HOME', str(tmp_path / 'home'))
    for asset in local.REGISTRY.values():          # every local asset folder is a temporary one
        monkeypatch.setenv(asset.env_var, str(tmp_path / 'assets' / asset.name))
    return stand_ins


def _tree(folder):
    """Every file under ``folder`` with its sha1: a source must not change."""
    return {str(p.relative_to(folder)): _sha1(p) for p in sorted(folder.rglob('*')) if p.is_file()}


@pytest.mark.parametrize('folder', ['absent', 'empty'])
def test_a_missing_source_says_what_to_get_and_where_to_put_it(tmp_path, offline, folder):
    needs_source = [name for name, factory in data_builder_registry.items()
                    if factory().build and not factory().fetches]
    assert needs_source
    for name in needs_source:
        builder = data_builder_registry[name]()
        source = tmp_path / 'home' / 'sources' / name      # the one rule, the same for every plugin
        if folder == 'empty':
            source.mkdir(parents=True, exist_ok=True)
        with pytest.raises(LocalDataMissing) as missing:
            package_data(name, output=tmp_path / 'out' / name)
        message = str(missing.value)
        for fact in (builder.source, builder.access, builder.layout, builder.needs, str(source)):
            assert fact in message, f'{name}: the error lacks {fact!r}'
        assert not (tmp_path / 'out' / name).exists()


def test_a_plugin_with_no_builder_says_so(tmp_path):
    for name, factory in data_builder_registry.items():
        builder = factory()
        if builder.cannot:
            with pytest.raises(NotImplementedError) as refused:
                package_data(name, output=tmp_path / name)
            assert builder.cannot in str(refused.value)
        elif not builder.build:
            assert package_data(name, output=tmp_path / name) is None
        assert not (tmp_path / name).exists()


def _sha1(path):
    return hashlib.sha1(path.read_bytes()).hexdigest()


def test_ewok_packages_from_the_source_folder_of_the_one_rule(tmp_path, monkeypatch, capsys, offline):
    # Synthetic text only; no EWoK sentence is committed to the tests.
    monkeypatch.setenv('BRAINSCORE_DATA_HOME', str(tmp_path))
    source = tmp_path / 'sources' / 'ewok'
    source.mkdir(parents=True)
    pd.DataFrame([dict(Domain='synthetic', MetaTemplateID='1', TemplateID=str(i), Version='0',
                       Context1=f'context A {i}', Context2=f'context B {i}',
                       Target1=f'target A {i}', Target2=f'target B {i}') for i in range(3)]
                 ).to_csv(source / 'native.csv', index=False)
    output = package_data('ewok')
    assert output == tmp_path / 'packaged' / 'ewok'
    printed = capsys.readouterr().out
    assert re.search(rf"items\.json .*sha1 {_sha1(output / 'items.json')} .*items 3", printed)
    assert _sha1(output / 'manifest.json') in printed


def _made_up_rajalingham():
    """A tiny 2-AFC data set: 6 objects, 11 images each, 5 distractors, 4 trials per condition."""
    objects = list('abcdef')
    rows = []
    for o, sample in enumerate(objects):
        for i in range(11):
            for distractor in objects:
                if distractor != sample:
                    rows += [(f'{sample}{i}', sample, distractor,
                              sample if (t + i + o) % (2 + i % 3) else distractor) for t in range(4)]
    image, sample, distractor, choice = map(list, zip(*rows))
    assembly = xr.DataArray(np.array(choice, dtype=object), dims=['presentation'], coords={
        'image_id': ('presentation', image), 'sample_obj': ('presentation', sample),
        'dist_obj': ('presentation', distractor)})
    stimuli = types.SimpleNamespace(stimulus_paths={i: f'{i}.png' for i in set(image)})
    return types.SimpleNamespace(load_dataset=lambda identifier: assembly,
                                 load_stimulus_set=lambda identifier: stimuli)


def test_rajalingham_prints_sha1_and_counts_and_names_the_files_its_tool_reads(
        tmp_path, monkeypatch, capsys, offline):
    from brainscore.data.rajalingham2018_2afc import TRIALS_CSV
    monkeypatch.setitem(sys.modules, 'brainscore_vision', _made_up_rajalingham())
    first = package_data('rajalingham2018-2afc', output=tmp_path / 'first')
    printed = capsys.readouterr().out
    for file in ('trials.csv', 'tokens.csv'):
        rows = len(pd.read_csv(first / file))
        assert re.search(rf'{file} .*sha1 {_sha1(first / file)} .*rows {rows}\)', printed)
    assert 'differs from the registered sha1' in printed          # made-up data is not the registered data
    # An equal rebuild is told about the files the scoring tool reads, not a download cache.
    registered = _sha1(first / 'trials.csv')
    original = data_builder_registry['rajalingham2018-2afc']()
    monkeypatch.setitem(data_builder_registry, 'rajalingham-fixture',
                        lambda: dataclasses.replace(original, sha1={'trials.csv': registered}))
    package_data('rajalingham-fixture', output=tmp_path / 'second')
    printed = capsys.readouterr().out
    assert 'equals the registered sha1' in printed
    assert TRIALS_CSV.parent.relative_to(DATA.parents[1]).as_posix() in printed
    assert 'BRAINIO_HOME' not in printed


def _gifti(path, n_tr):
    """A synthetic surface time series: ``n_tr`` arrays over a full fsaverage hemisphere."""
    import nibabel as nib
    arrays = [nib.gifti.GiftiDataArray(np.full(10242 + 5, t, dtype=np.float32)) for t in range(n_tr)]
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.gifti.GiftiImage(darrays=arrays), str(path))


def _lahner_source(source):
    """One run of ds005165 laid out as the bucket, with a file of the user beside it."""
    func = source / 'derivatives/versionB/fmriprep/sub-01/ses-02/func'
    stem = 'sub-01_ses-02_task-test_run-1'
    for hemi in 'LR':
        _gifti(func / f'{stem}_hemi-{hemi}_space-fsaverage_bold.func.gii', n_tr=4)
    events = source / 'sub-01/ses-02/func' / f'{stem}_events.tsv'
    events.parent.mkdir(parents=True)
    pd.DataFrame({'onset': [0.0, 4.0], 'duration': [3.0, 3.0], 'trial_type': ['test', 'oddball'],
                  'stim_file': ['test/0001.mp4', 'n/a']}).to_csv(events, sep='\t', index=False)
    (source / 'derivatives/versionB/fmriprep/sub-01/my-notes.txt').write_text('mine')
    (source / 'run_records.pkl').write_bytes(b'an older checkpoint of mine')
    return f'ds005165/derivatives/versionB/fmriprep/sub-01/ses-02/func/{stem}_hemi-L_space-fsaverage_bold.func.gii'


@pytest.mark.parametrize('listed', [True, False], ids=['builds', 'fails'])
def test_lahner_time_series_reads_the_source_and_never_changes_it(tmp_path, offline, listed):
    source = tmp_path / 'ds005165'
    key = _lahner_source(source)
    before = _tree(source)
    # The bucket listing names the run (or nothing); a download of a file the source holds fails.
    def aws(argv):
        if argv[1:3] != ['s3', 'ls']:
            pytest.fail(f'downloaded a file the source holds: {argv}')
        named = listed and argv[-1].endswith('/sub-01/') and 'task-test' in key
        return types.SimpleNamespace(stdout=f'2026-01-01 00:00:00 1 {key}\n' if named else '', returncode=0)
    offline.programs['aws'] = aws
    built = [tmp_path / 'out']
    if listed:
        output = package_data('lahner2024-timeresolved', source=source, output=built[0])
        assert (output / 'assy_Lahner2024-fMRI-timeresolved.nc').exists()
    else:
        with pytest.raises(RuntimeError):
            package_data('lahner2024-timeresolved', source=source, output=built[0])
        assert not built[0].exists()
    assert _tree(source) == before
    assert not list(tmp_path.glob('.brainscore-*'))


def _algonauts_source(root, participants=(1, 2, 3, 5), film=b''):
    """A tiny competition tree: one Friends film of 3 scans, one S7 and one OOD film."""
    import h5py
    for participant in participants:
        sub = f'sub-{participant:02d}'
        (root / 'fmri' / sub / 'func').mkdir(parents=True)
        with h5py.File(root / 'fmri' / sub / 'func' / f'{sub}_task-friends_s1_bold.h5', 'w') as h5:
            h5['ses-001_task-s01e01a'] = np.full((3, 1000), participant, dtype=np.float32)
        counts = root / 'fmri' / sub / 'target_sample_number'
        counts.mkdir()
        np.save(counts / f'{sub}_friends-s7_fmri_samples.npy', {'s07e01a': 2})
        np.save(counts / f'{sub}_ood_fmri_samples.npy', {'chaplin1': 2})
    films = {'friends/s1/friends_s01e01a.mkv': film, 'friends/s7/friends_s07e01a.mkv': b'',
             'ood/chaplin/task-chaplin1_video.mkv': b''}
    for relative, content in films.items():
        (root / 'stimuli/movies' / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / 'stimuli/movies' / relative).write_bytes(content)
    (root / 'stimuli/movies/movie10').mkdir()


def test_algonauts_fails_as_a_whole_when_a_participant_lacks_its_input(tmp_path, offline):
    _algonauts_source(tmp_path / 'native', participants=(1, 3, 5))
    with pytest.raises(LocalDataMissing) as missing:
        package_data('algonauts2025', source=tmp_path / 'native', output=tmp_path / 'out')
    assert 'algonauts2025_friends_sub02.nc' in str(missing.value)
    assert data_builder_registry['algonauts2025']().access in str(missing.value)
    assert not (tmp_path / 'out').exists()


@pytest.mark.skipif(__import__('shutil').which('ffmpeg') is None, reason='ffmpeg is needed')
def test_algonauts_builds_every_file_the_registered_benchmarks_read(tmp_path, offline):
    import subprocess
    from brainscore.data.algonauts2025 import per_scan_stimuli, scans_per_stimulus
    film = tmp_path / 'film.mkv'
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'lavfi', '-i', 'testsrc=duration=5:size=64x48:rate=25',
                    '-f', 'lavfi', '-i', 'sine=duration=5:sample_rate=16000', '-shortest',
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-c:a', 'aac', str(film)], check=True)
    _algonauts_source(tmp_path / 'native', film=film.read_bytes())
    output = package_data('algonauts2025', source=tmp_path / 'native', output=tmp_path / 'out')
    # Read the result as the benchmarks do: the registered loaders, then the media table of each variant.
    for participant, media in ((1, ['frames']), (1, ['clips']), (1, ['clips', 'sound']), (2, ['frames']),
                               (3, ['frames']), (5, ['frames'])):
        assembly = brainscore.load_dataset(f'Algonauts2025-friends-sub{participant:02d}', root=output)
        films = brainscore.load_stimulus_set('Algonauts2025-friends', root=output)
        table = per_scan_stimuli(films, scans_per_stimulus(assembly), media, 'friends', participant, output)
        assert len(table) == assembly.sizes['presentation'] == 3


@pytest.mark.skipif(__import__('shutil').which('ffmpeg') is None, reason='ffmpeg is needed')
def test_the_algonauts_recovery_command_reads_the_folder_the_benchmark_names(tmp_path, offline):
    import subprocess
    from brainscore.data.__main__ import main
    film = tmp_path / 'film.mkv'
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'lavfi', '-i', 'testsrc=duration=5:size=64x48:rate=25',
                    '-f', 'lavfi', '-i', 'sine=duration=5:sample_rate=16000', '-shortest',
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-c:a', 'aac', str(film)], check=True)
    # The competition files are where the benchmark's pre-check looks for them.
    _algonauts_source(local.REGISTRY['algonauts2025-root'].resolved(), film=film.read_bytes())
    prepared = local.REGISTRY['algonauts2025-prepared'].resolved()
    with pytest.raises(LocalDataMissing) as missing:
        local.check_benchmark_assets('Algonauts2025-friends-sub01')
    # Run the command the message gives, as written, with the folder it names.
    command = str(missing.value).split('Then convert it:')[1].split('\n')[1].strip()
    assert main(shlex.split(command)[3:]) == 0
    local.check_benchmark_assets('Algonauts2025-friends-sub01')
    assert brainscore.load_dataset('Algonauts2025-friends-sub01', root=prepared).sizes['presentation'] == 3


def _clips(folder, monkeypatch):
    """Two tiny hosted clips, one with sound and one without, as ``load_stimulus_set`` gives them."""
    import subprocess
    from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
    folder.mkdir()
    paths = {'stimulus_0001': folder / '0001.mp4', 'stimulus_0002': folder / '0002.mp4'}
    for path, sound in zip(paths.values(), (True, False)):
        inputs = ['-f', 'lavfi', '-i', 'testsrc=duration=3:size=64x48:rate=25']
        if sound:
            inputs += ['-f', 'lavfi', '-i', 'sine=duration=3:sample_rate=16000']
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', *inputs, '-shortest', '-c:v', 'libx264',
                        '-pix_fmt', 'yuv420p', str(path)], check=True)
    stimuli = StimulusSet(pd.DataFrame({'stimulus_id': list(paths)}))
    stimuli.identifier = 'BOLDMoments'
    stimuli.stimulus_paths = {key: str(path) for key, path in paths.items()}
    monkeypatch.setattr(brainscore, 'load_stimulus_set', lambda identifier: stimuli)
    return paths


@pytest.mark.skipif(__import__('shutil').which('ffmpeg') is None, reason='ffmpeg is needed')
def test_the_lahner_sound_manifest_names_files_that_exist_after_the_build(tmp_path, monkeypatch, offline):
    import json
    _clips(tmp_path / 'clips', monkeypatch)
    output = package_data('lahner2024-sound', output=tmp_path / 'sound')
    manifest = json.loads((output / 'manifest.json').read_text())
    assert len(manifest['entries']) == 2
    for entry in manifest['entries']:
        assert (output / entry['audio_path']).is_file()


# --- every builder through the public command --------------------------------------
# A tiny made-up source for each builder: (source folder or None when the script fetches it).

def _ewok_source(tmp_path, monkeypatch, offline):
    source = tmp_path / 'ewok'
    source.mkdir()
    pd.DataFrame([dict(Domain='synthetic', MetaTemplateID='1', TemplateID='0', Version='0',
                       Context1='context A', Context2='context B', Target1='target A', Target2='target B')]
                 ).to_csv(source / 'native.csv', index=False)
    return source


def _lebel_source(tmp_path, monkeypatch, offline):
    import pickle
    n_tr = 6
    story = types.SimpleNamespace(brain_data=np.arange(n_tr * 4, dtype=np.float32).reshape(n_tr, 4),
                                  tr_times=np.arange(n_tr + 15) * 2.0, words=['one', 'two', 'three'],
                                  data_times=[21.0, 23.0, 25.0])
    source = tmp_path / 'lebel'
    source.mkdir()
    (source / 'assembly_lebel_uts03.pkl').write_bytes(
        pickle.dumps(types.SimpleNamespace(stories=['story'], story_data={'story': story})))
    return source


def _algonauts_film_source(tmp_path, monkeypatch, offline):
    import subprocess
    film = tmp_path / 'film.mkv'
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'lavfi', '-i', 'testsrc=duration=5:size=64x48:rate=25',
                    '-f', 'lavfi', '-i', 'sine=duration=5:sample_rate=16000', '-shortest',
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-c:a', 'aac', str(film)], check=True)
    _algonauts_source(tmp_path / 'native', film=film.read_bytes())
    return tmp_path / 'native'


def _coggym_source(tmp_path, monkeypatch, offline):
    def git(argv):
        if argv[1] == 'clone':                 # a local stand-in of the repository
            (pathlib.Path(argv[-1]) / 'evaluation').mkdir(parents=True, exist_ok=True)
            (pathlib.Path(argv[-1]) / 'evaluation' / 'public_manifest.json').write_text('[]')
        elif argv[3] != 'checkout':
            raise AssertionError(f'package_data ran {argv}')
        return types.SimpleNamespace(returncode=0, stdout='', stderr='')
    offline.programs['git'] = git


def _lahner_clips_source(tmp_path, monkeypatch, offline):
    _clips(tmp_path / 'clips', monkeypatch)


def _lahner_series_source(tmp_path, monkeypatch, offline):
    key = _lahner_source(tmp_path / 'ds005165')

    def aws(argv):
        assert argv[1:3] == ['s3', 'ls'], argv
        named = argv[-1].endswith('/sub-01/')
        return types.SimpleNamespace(stdout=f'2026-01-01 00:00:00 1 {key}\n' if named else '', returncode=0)
    offline.programs['aws'] = aws
    return tmp_path / 'ds005165'


def _lahner_motion_source(tmp_path, monkeypatch, offline):
    tsv = pd.DataFrame({column: [0.0, 0.1] for column in (
        'trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z', 'csf', 'white_matter')})
    tsv['framewise_displacement'] = ['n/a', 0.2]
    offline.objects['ds005165/derivatives/versionB/fmriprep/sub-01/ses-02/func/'
                    'sub-01_ses-02_task-test_run-1_desc-confounds_timeseries.tsv'] = \
        tsv.to_csv(sep='\t', index=False).encode()


def _rajalingham_source(tmp_path, monkeypatch, offline):
    monkeypatch.setitem(sys.modules, 'brainscore_vision', _made_up_rajalingham())


SOURCES = {'ewok': _ewok_source, 'lebel2023': _lebel_source, 'algonauts2025': _algonauts_film_source,
           'coggym': _coggym_source, 'lahner2024-frames': _lahner_clips_source,
           'lahner2024-sound': _lahner_clips_source, 'lahner2024-timeresolved': _lahner_series_source,
           'lahner2024-motion': _lahner_motion_source, 'rajalingham2018-2afc': _rajalingham_source}
UPLOAD_FLAGS = {'lebel2023': '--upload', 'lahner2024-timeresolved': '--upload-to-s3',
                'lahner2024-motion': '--upload-to-s3'}


def test_every_builder_has_a_made_up_source_here():
    # Intended gate: a new builder gets a tiny made-up source in SOURCES, so the test below runs it.
    builders = {name for name, factory in data_builder_registry.items() if factory().build}
    assert builders <= set(SOURCES), f'add a tiny made-up source to SOURCES for {sorted(builders - set(SOURCES))}'


def _package(name, tmp_path, monkeypatch, offline):
    from brainscore.data.__main__ import main
    source = SOURCES[name](tmp_path, monkeypatch, offline)
    arguments = ['package', name, '--output', str(tmp_path / 'out')]
    return main(arguments + (['--source', str(source)] if source else []))


@pytest.mark.skipif(__import__('shutil').which('ffmpeg') is None, reason='ffmpeg is needed')
@pytest.mark.parametrize('name', sorted(SOURCES))
def test_every_builder_runs_through_the_public_command_and_uploads_nothing(name, tmp_path, monkeypatch, offline):
    assert _package(name, tmp_path, monkeypatch, offline) == 0
    written = [p for p in (tmp_path / 'out').rglob('*') if p.is_file() and p.name != 'build.json']
    assert written, f'{name} wrote nothing'


@pytest.mark.parametrize('name', sorted(UPLOAD_FLAGS))
def test_a_planted_upload_fails(name, tmp_path, monkeypatch, offline):
    builder = data_builder_registry[name]()
    monkeypatch.delitem(data_builder_registry, name)
    monkeypatch.setitem(data_builder_registry, name,
                        lambda: dataclasses.replace(builder, build=f'{builder.build} {UPLOAD_FLAGS[name]}'))
    with pytest.raises(AssertionError, match='nothing may be uploaded'):
        _package(name, tmp_path, monkeypatch, offline)
    assert not (tmp_path / 'out').exists()


def test_a_rebuilt_hosted_file_is_read_where_its_instruction_puts_it(tmp_path, monkeypatch, offline):
    from brainscore.data.lebel2023 import hosted
    from brainscore_core.supported_data_standards.brainio.s3 import get_path
    assert _package('lebel2023', tmp_path, monkeypatch, offline) == 0
    built = tmp_path / 'out' / f'stimulus_{hosted.TR_STIMULI_ID}.csv'
    # Follow the declared instruction: $BRAINIO_HOME/<sha1>/<file name>, then load with no download.
    from brainscore_core.supported_data_standards.brainio import fetch
    monkeypatch.setattr(fetch, 'get_local_data_path', lambda: str(tmp_path / 'brainio'))   # BRAINIO_HOME
    sha1 = _sha1(built)
    (tmp_path / 'brainio' / sha1).mkdir(parents=True)
    (tmp_path / 'brainio' / sha1 / built.name).write_bytes(built.read_bytes())
    assert '$BRAINIO_HOME/<sha1>/<file name>' in data_builder_registry['lebel2023']().writes
    path = get_path(hosted.TR_STIMULI_ID, 'csv', hosted.BUCKET, hosted.VERSION_ID['TR stimuli'], sha1,
                    filename_prefix='stimulus_', folder_name=hosted.FOLDER)
    assert _sha1(pathlib.Path(path)) == sha1


@pytest.mark.skipif(__import__('shutil').which('ffmpeg') is None, reason='ffmpeg is needed')
def test_algonauts_fails_as_a_whole_when_an_assembly_names_an_absent_film(tmp_path, offline):
    import h5py
    source = _algonauts_film_source(tmp_path, None, offline)
    # The fMRI of every participant also holds a second film, whose file the source lacks.
    for h5_path in source.glob('fmri/sub-0*/func/*.h5'):
        with h5py.File(h5_path, 'a') as h5:
            h5['ses-001_task-s01e02a'] = np.zeros((3, 1000), dtype=np.float32)
    before = _tree(source)
    with pytest.raises(LocalDataMissing) as missing:
        package_data('algonauts2025', source=source, output=tmp_path / 'out')
    assert 'friends_s01e02a' in str(missing.value)
    assert not (tmp_path / 'out').exists()
    assert not list(tmp_path.glob('.brainscore-*'))
    assert _tree(source) == before
