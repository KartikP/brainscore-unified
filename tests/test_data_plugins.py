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


def test_benchmark_package_does_not_inline_s3_loader_metadata():
    forbidden = (
        'STIMULUS_ID',
        '_SHA1',
        '_VERSION_ID',
        'load_assembly_from_s3',
        'load_stimulus_set_from_s3',
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
