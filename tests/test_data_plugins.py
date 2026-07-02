"""C1 data-plugin separation guards for naturalistic benchmarks."""

from pathlib import Path

import brainscore


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_lahner_data_registry_entries_present():
    assert 'BOLDMoments' in brainscore.stimulus_set_registry
    for identifier in (
        'Lahner2024-fMRI',
        'Lahner2024-fMRI-timeresolved',
        'Lahner2024-fMRI-timeresolved-events',
        'Lahner2024-fMRI-timeresolved-motion',
    ):
        assert identifier in brainscore.data_registry


def test_algonauts_data_registry_entries_present():
    for split in ('friends', 'friends-s7', 'ood'):
        assert f'Algonauts2025-{split}' in brainscore.stimulus_set_registry
        for subject in (1, 2, 3, 5):
            assert (
                f'Algonauts2025-{split}-sub{subject:02d}'
                in brainscore.data_registry
            )


def test_roar_data_registry_entries_present():
    assert 'Yeatman2021' in brainscore.stimulus_set_registry
    assert 'Yeatman2021' in brainscore.data_registry


def test_roar_data_plugin_loads_from_s3_and_postprocesses(monkeypatch):
    import numpy as np
    import pandas as pd
    import xarray as xr
    from brainscore.data import roar_yeatman2021 as roar_data
    from brainscore_core.supported_data_standards.brainio import s3
    from brainscore_core.supported_data_standards.brainio.stimuli import (
        StimulusSet,
    )

    raw = StimulusSet(pd.DataFrame({
        'label': [1, 0],
        'stimulus_id': ['word-1', 'pseudo-1'],
        'realpseudo': ['real', 'pseudo'],
        'image_filename': ['raw-a.png', 'raw-b.png'],
        'word': ['table', 'blark'],
        'filename': ['roar_0001.png', 'roar_0002.png'],
    }))
    raw.identifier = 'Yeatman2021'
    raw.stimulus_paths = {
        'word-1': '/cache/roar_0001.png',
        'pseudo-1': '/cache/roar_0002.png',
    }
    calls = []

    def fake_load_stimulus_set_from_s3(**kwargs):
        calls.append(('stimulus', kwargs))
        return raw

    def fake_load_assembly_from_s3(**kwargs):
        calls.append(('assembly', kwargs))
        data = xr.DataArray(
            np.array([1, 0]),
            dims=['presentation'],
            coords={
                'stimulus_id': ('presentation', ['word-1', 'pseudo-1']),
                'correct': ('presentation', [1, 0]),
            },
            name='data',
            attrs={'stimulus_set_identifier': 'Yeatman2021'},
        )
        return kwargs['cls'](data=data)

    monkeypatch.setattr(
        s3, 'load_stimulus_set_from_s3', fake_load_stimulus_set_from_s3)
    monkeypatch.setattr(
        s3, 'load_assembly_from_s3', fake_load_assembly_from_s3)

    stimulus_set = roar_data.load_yeatman2021_stimulus_set()
    assembly = roar_data.load_yeatman2021_assembly()

    assert calls[0] == ('stimulus', {
        'identifier': 'Yeatman2021',
        'bucket': roar_data.BUCKET,
        'csv_sha1': roar_data.STIMULUS_CSV_SHA1,
        'zip_sha1': roar_data.STIMULUS_ZIP_SHA1,
        'csv_version_id': roar_data.STIMULUS_CSV_VERSION_ID,
        'zip_version_id': roar_data.STIMULUS_ZIP_VERSION_ID,
    })
    assert calls[1][0] == 'assembly'
    assert calls[1][1]['identifier'] == 'Yeatman2021'
    assert calls[1][1]['bucket'] == roar_data.BUCKET
    assert calls[1][1]['sha1'] == roar_data.ASSEMBLY_SHA1
    assert calls[1][1]['version_id'] == roar_data.ASSEMBLY_VERSION_ID
    assert calls[1][1]['merge_stimulus_set_meta'] is False

    assert list(stimulus_set.columns) == [
        'stimulus_id',
        'image_file_name',
        'sentence',
        'image_label',
        'numeric_label',
        'word',
        'realpseudo',
    ]
    assert stimulus_set['image_label'].tolist() == ['real', 'pseudo']
    assert stimulus_set['numeric_label'].tolist() == [1, 0]
    assert stimulus_set['sentence'].tolist() == ['table', 'blark']
    assert stimulus_set['image_file_name'].tolist() == [
        '/cache/roar_0001.png',
        '/cache/roar_0002.png',
    ]
    assert isinstance(assembly, xr.Dataset)
    assert list(assembly.data_vars) == ['data']


def test_lahner_runtime_benchmarks_resolve_data_via_registry(monkeypatch):
    from brainscore.benchmarks.lahner2024 import benchmark as lahner_glm
    from brainscore.benchmarks.lahner2024 import benchmark_timeresolved as lahner_tr

    calls = []

    def fake_load_dataset(identifier, *args, **kwargs):
        calls.append(('dataset', identifier, kwargs))
        return f'data:{identifier}'

    def fake_load_stimulus_set(identifier, *args, **kwargs):
        calls.append(('stimulus', identifier, kwargs))
        return f'stimulus:{identifier}'

    monkeypatch.setattr(lahner_glm, 'load_dataset', fake_load_dataset)
    monkeypatch.setattr(lahner_glm, 'load_stimulus_set', fake_load_stimulus_set)
    benchmark = lahner_glm.Lahner2024BOLDMoments()
    assert benchmark.assembly == 'data:Lahner2024-fMRI'
    assert benchmark.stimulus_set == 'stimulus:BOLDMoments'

    monkeypatch.setattr(lahner_tr, 'load_dataset', fake_load_dataset)
    timeresolved = lahner_tr.Lahner2024BOLDMoments_timeresolved(
        apply_motion_regression=True)
    assert timeresolved.events == 'data:Lahner2024-fMRI-timeresolved-events'
    assert timeresolved.motion == 'data:Lahner2024-fMRI-timeresolved-motion'

    assert ('dataset', 'Lahner2024-fMRI', {}) in calls
    assert ('stimulus', 'BOLDMoments', {}) in calls
    assert (
        'dataset',
        'Lahner2024-fMRI-timeresolved-events',
        {},
    ) in calls
    assert (
        'dataset',
        'Lahner2024-fMRI-timeresolved-motion',
        {},
    ) in calls


def test_algonauts_benchmark_resolves_data_via_registry(monkeypatch, tmp_path):
    from brainscore.benchmarks.algonauts2025 import benchmark as algonauts

    calls = []

    def fake_load_dataset(identifier, *args, **kwargs):
        calls.append(('dataset', identifier, kwargs))
        return f'data:{identifier}'

    def fake_load_stimulus_set(identifier, *args, **kwargs):
        calls.append(('stimulus', identifier, kwargs))
        return f'stimulus:{identifier}'

    monkeypatch.setattr(algonauts, 'load_dataset', fake_load_dataset)
    monkeypatch.setattr(algonauts, 'load_stimulus_set', fake_load_stimulus_set)

    benchmark = algonauts.Algonauts2025Friends(
        subject=1, assembly_root=tmp_path)
    assert benchmark.assembly == 'data:Algonauts2025-friends-sub01'
    assert benchmark.stimulus_set == 'stimulus:Algonauts2025-friends'
    assert calls == [
        ('dataset', 'Algonauts2025-friends-sub01', {'root': tmp_path}),
        ('stimulus', 'Algonauts2025-friends', {'root': tmp_path}),
    ]


def test_roar_benchmark_resolves_data_via_registry(monkeypatch):
    import numpy as np
    import pandas as pd
    import xarray as xr
    from brainscore.benchmarks.roar_yeatman2021 import benchmark as roar
    from brainscore_core.supported_data_standards.brainio.stimuli import (
        StimulusSet,
    )

    rows = []
    for label, label_name in ((1, 'real'), (0, 'pseudo')):
        for idx in range(250):
            stimulus_id = f'{label_name}-{idx:03d}'
            rows.append({
                'stimulus_id': stimulus_id,
                'image_file_name': f'/fake/{stimulus_id}.png',
                'sentence': stimulus_id,
                'image_label': label_name,
                'numeric_label': label,
                'word': stimulus_id,
                'realpseudo': label_name,
            })
    stimulus_set = StimulusSet(pd.DataFrame(rows))
    stimulus_set.identifier = 'Yeatman2021'
    stimulus_set.stimulus_paths = {
        row['stimulus_id']: row['image_file_name'] for row in rows
    }
    assembly = xr.Dataset({
        'data': ('presentation', np.array([row['numeric_label']
                                           for row in rows])),
        'correct': ('presentation', np.array([row['numeric_label']
                                              for row in rows])),
        'stimulus_id': ('presentation', np.array([row['stimulus_id']
                                                  for row in rows])),
    })
    calls = []

    def fake_load_stimulus_set(identifier, *args, **kwargs):
        calls.append(('stimulus', identifier, kwargs))
        return stimulus_set

    def fake_load_dataset(identifier, *args, **kwargs):
        calls.append(('dataset', identifier, kwargs))
        return assembly

    monkeypatch.setattr(roar, 'load_stimulus_set', fake_load_stimulus_set)
    monkeypatch.setattr(roar, 'load_dataset', fake_load_dataset)

    benchmark = roar.Yeatman2021LexicalDecision(modality='vision')

    assert benchmark._human_accuracy == 1.0
    assert calls == [
        ('stimulus', 'Yeatman2021', {}),
        ('dataset', 'Yeatman2021', {}),
    ]


def test_benchmark_package_does_not_inline_s3_loader_metadata():
    forbidden = (
        'STIMULUS_ID',
        '_SHA1',
        '_VERSION_ID',
        'load_assembly_from_s3',
        'load_stimulus_set_from_s3',
        'ROAR_DATA_DIR',
        '/Users/kartik',
        'Brain-Score Unified/data',
        's3://',
    )
    offenders = {}
    for path in (REPO_ROOT / 'brainscore' / 'benchmarks').rglob('*.py'):
        text = path.read_text()
        hits = [pattern for pattern in forbidden if pattern in text]
        if hits:
            offenders[str(path.relative_to(REPO_ROOT))] = hits

    assert not offenders


def test_benchmark_package_does_not_contain_prep_or_driver_scripts():
    forbidden_names = set()
    for benchmark_dir in (
        REPO_ROOT / 'brainscore' / 'benchmarks' / 'lahner2024',
        REPO_ROOT / 'brainscore' / 'benchmarks' / 'algonauts2025',
    ):
        forbidden_names.update(
            path.name for path in benchmark_dir.glob('prepare_*.py'))
        forbidden_names.update(
            path.name for path in benchmark_dir.glob('score_*.py'))
        if (benchmark_dir / 'submit_codabench.py').exists():
            forbidden_names.add('submit_codabench.py')

    assert not forbidden_names
