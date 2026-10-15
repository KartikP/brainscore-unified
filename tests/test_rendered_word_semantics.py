"""Offline checks of paired images, retrieval, and the real UMI extraction path."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from PIL import Image
import pytest
from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet

from brainscore.benchmarks.rendered_word_semantics import RenderedWordRetrieval
from brainscore.benchmarks.rendered_word_semantics.benchmark import retrieval_ranks


@pytest.fixture
def manifest(tmp_path):
    rows = []
    for word_index, word in enumerate(['cat', 'dog', 'table', 'chair']):
        for slot in range(2):
            for condition in ['word', 'scramble']:
                path = tmp_path / f'{word}-{slot}-{condition}.png'
                Image.new('RGB', (16, 16), (word_index * 50, slot * 40, 90)).save(path)
                rows.append(dict(word=word, slot=slot, condition=condition, image_path=str(path)))
    return pd.DataFrame(rows)


class FeatureSubject:
    """Known features with a background offset that paired subtraction removes."""

    supported_modalities = {'vision'}

    def __init__(self, manifest, *, reverse=False, bad_ids=False, zero=False):
        vectors = {'cat': [1, 0], 'dog': [0.9, 0.1], 'table': [-1, 0], 'chair': [-0.9, -0.1]}
        self.features = {}
        for row in manifest.itertuples():
            background = np.array([10, -20], dtype=np.float32) * (row.slot + 1)
            signal = np.array(vectors[row.word], dtype=np.float32)
            self.features[row.image_path] = background + (
                signal if row.condition == 'word' and not zero else 0
            )
        self.reverse = reverse
        self.bad_ids = bad_ids
        self.calls = 0

    def start_recording(self, target):
        assert target == 'word_representation'

    def process(self, stimuli):
        self.calls += 1
        ids = list(stimuli['stimulus_id'])
        values = np.asarray([self.features[str(stimuli.get_stimulus(identifier))] for identifier in ids])
        if self.reverse:
            ids, values = ids[::-1], values[::-1]
        if self.bad_ids:
            ids = [ids[0]] * len(ids)
        return NeuroidAssembly(
            values,
            dims=['presentation', 'neuroid'],
            coords={
                'stimulus_id': ('presentation', ids),
                'condition': ('presentation', ['test'] * len(ids)),
                'neuroid_id': ('neuroid', ['u0', 'u1']),
                'layer': ('neuroid', ['summary', 'summary']),
            },
        )


def benchmark(manifest, **kwargs):
    return RenderedWordRetrieval(manifest, [('cat', 'dog'), ('table', 'chair')], top_k=1, **kwargs)


def test_paired_subtraction_centering_and_retrieval(manifest):
    subject = FeatureSubject(manifest, reverse=True)
    task = benchmark(manifest, batch_size=3)
    score = task(subject)
    assert float(score) == 1
    assert score.attrs['ranks'] == [0, 0]
    assert score.attrs['chance'] == pytest.approx(1 / 3)
    assert subject.calls == 3  # The last batch is partial.
    expected = []
    for word in task.words:
        paired = []
        for slot in task.slots:
            rows = manifest[(manifest.word == word) & (manifest.slot == slot)]
            values = {
                row.condition: subject.features[row.image_path].astype(np.float16).astype(np.float32)
                for row in rows.itertuples()
            }
            paired.append(values['word'] - values['scramble'])
        expected.append(np.mean(paired, axis=0))
    expected = np.asarray(expected)
    expected -= expected.mean(axis=0)
    np.testing.assert_array_equal(score.attrs['directions'], expected)
    assert task.leaderboard_eligible is False
    assert task.ceiling is None


def test_row_order_and_batch_size_do_not_change_results(manifest):
    a = benchmark(manifest, batch_size=1)
    b = benchmark(manifest.sample(frac=1, random_state=2), batch_size=7)
    assert a.identifier == b.identifier
    np.testing.assert_array_equal(
        a(FeatureSubject(manifest)).attrs['directions'],
        b(FeatureSubject(manifest)).attrs['directions'],
    )


def test_rank_excludes_query_and_favors_tied_partner():
    directions = np.array([[1, 0], [1, 0], [0, 1], [-1, 0]])
    words = ['a', 'b', 'c', 'd']
    assert retrieval_ranks(directions, words, [('a', 'b'), ('a', 'c'), ('a', 'd')]) == [0, 1, 2]


def test_all_vocabulary_items_remain_candidates():
    directions = np.array([[1, 0], [0, 1], [1, 0.1], [1, 0.2]])
    assert retrieval_ranks(
        directions, ['query', 'partner', 'distractor', 'pseudoword'], [('query', 'partner')],
    ) == [2]


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'unequal_slots', 'condition', 'nan', 'slot'])
def test_bad_manifests_fail_before_model_calls(manifest, change):
    if change == 'missing':
        manifest = manifest.iloc[1:]
    elif change == 'duplicate':
        manifest = pd.concat([manifest, manifest.iloc[:1]])
    elif change == 'unequal_slots':
        manifest = manifest[~((manifest.word == 'cat') & (manifest.slot == 1))]
    elif change == 'condition':
        manifest.loc[0, 'condition'] = 'clean'
    elif change == 'nan':
        manifest.loc[0, 'word'] = None
    else:
        manifest['slot'] = manifest['slot'] + 0.5
    with pytest.raises(ValueError):
        benchmark(manifest)


@pytest.mark.parametrize('pairs', [[], [('cat', 'cat')], [('cat', 'absent')], [('cat', 'dog')] * 2])
def test_invalid_pairs_are_rejected(manifest, pairs):
    with pytest.raises(ValueError):
        RenderedWordRetrieval(manifest, pairs, top_k=1)


def test_subsets_labels_and_changed_pixels_get_different_identifiers(manifest):
    original = benchmark(manifest).identifier
    assert benchmark(manifest[manifest.slot == 0]).identifier != original
    assert RenderedWordRetrieval(manifest, [('cat', 'chair')], top_k=1).identifier != original
    Image.new('RGB', (16, 16), 'white').save(manifest.iloc[0].image_path)
    assert benchmark(manifest).identifier != original


def test_changed_image_is_rejected_before_inference(manifest):
    task = benchmark(manifest)
    subject = FeatureSubject(manifest)
    Image.new('RGB', (16, 16), 'black').save(manifest.iloc[0].image_path)
    with pytest.raises(ValueError, match='changed after preflight'):
        task(subject)
    assert subject.calls == 0


def test_text_only_subject_is_rejected(manifest):
    with pytest.raises(ValueError, match='vision'):
        benchmark(manifest)(SimpleNamespace(supported_modalities={'text'}))


@pytest.mark.parametrize('kwargs,match', [({'bad_ids': True}, 'stimulus IDs'), ({'zero': True}, 'nonzero')])
def test_invalid_outputs_do_not_produce_scores(manifest, kwargs, match):
    with pytest.raises(ValueError, match=match):
        benchmark(manifest)(FeatureSubject(manifest, **kwargs))


@pytest.fixture
def author_data(tmp_path, monkeypatch):
    """Build a native-format manifest without downloading copyrighted images."""
    from brainscore.benchmarks.rendered_word_semantics import data

    repo = tmp_path / 'authors'
    (repo / 'lexicon').mkdir(parents=True)
    vocabulary = []
    for number in range(60):
        for role in ['country', 'capital']:
            vocabulary.append(dict(
                list='country_capital', item=f'pair{number}',
                item_role=role, word=f'{role}{number}',
            ))
    for number in range(1441):
        vocabulary.append(dict(list='other', item=str(number), item_role='word', word=f'other{number}'))
    vocabulary = pd.DataFrame(vocabulary)
    vocabulary.to_csv(repo / 'lexicon/vocabulary.tsv', sep='\t', index=False)
    rows = pd.MultiIndex.from_product(
        [vocabulary.word, range(96), ['word', 'scramble']],
        names=['word', 'slot', 'condition'],
    ).to_frame(index=False)
    rows['image_path'] = '/another/machine/data/renders/test.png'
    monkeypatch.setattr(data.pd, 'read_parquet', lambda path: rows.copy())
    monkeypatch.setattr(
        data.subprocess, 'check_output',
        lambda command, **kwargs: data.AUTHOR_REVISION if 'rev-parse' in command else '',
    )
    return repo, tmp_path / 'local-data', data


def test_author_data_keeps_all_candidates_and_relocates_paths(author_data):
    repo, root, data = author_data
    rows, pairs = data.load_author_data(repo, root, slot_limit=1)
    assert len(rows) == 2 * 1561
    assert len(pairs) == 60
    assert set(rows.slot) == {0}
    assert set(rows.image_path) == {str(root / 'renders/test.png')}


@pytest.mark.parametrize('problem', ['revision', 'dirty', 'traversal', 'slots', 'vocabulary'])
def test_author_data_rejects_changed_or_invalid_inputs(author_data, monkeypatch, problem):
    repo, root, data = author_data
    kwargs = {}
    if problem == 'revision':
        monkeypatch.setattr(data.subprocess, 'check_output', lambda *args, **kwargs: 'wrong')
    elif problem == 'dirty':
        monkeypatch.setattr(
            data.subprocess, 'check_output',
            lambda command, **kwargs: data.AUTHOR_REVISION if 'rev-parse' in command else ' M src/models.py',
        )
    elif problem in {'traversal', 'vocabulary'}:
        rows = data.pd.read_parquet(None)
        if problem == 'traversal':
            rows['image_path'] = '/data/renders/../../outside.png'
        else:
            rows = rows[rows.word != 'country0']
        monkeypatch.setattr(data.pd, 'read_parquet', lambda path: rows)
    else:
        kwargs['slot_limit'] = 0
    with pytest.raises(ValueError):
        data.load_author_data(repo, root, **kwargs)


@pytest.mark.integration
@pytest.mark.parametrize('model_type', ['dinov2', 'dinov3_vit'])
def test_real_dino_architecture_through_process_matches_direct_features(manifest, monkeypatch, model_type):
    """Random tiny DINO models check extraction mechanics, not learned semantics."""
    import torch
    from transformers import AutoConfig, AutoModel
    from brainscore.benchmarks.rendered_word_semantics.model import build_vision_subject

    monkeypatch.setenv('RESULTCACHING_DISABLE', '1')
    torch.manual_seed(4)
    model = AutoModel.from_config(AutoConfig.for_model(
        model_type,
        image_size=16,
        patch_size=4,
        hidden_size=12,
        num_hidden_layers=2,
        num_attention_heads=3,
        intermediate_size=24,
    )).eval()

    def processor(*, images, return_tensors):
        return {'pixel_values': torch.stack([
            torch.tensor(np.asarray(image).copy()).permute(2, 0, 1).float() / 255
            for image in images
        ])}

    subject = build_vision_subject(
        encoder=model, processor=processor, identifier='random-dino-test', batch_size=2,
    )
    paths = manifest['image_path'].iloc[:3].tolist()
    stimuli = StimulusSet({'stimulus_id': ['a', 'b', 'c'], 'image_file_name': paths})
    stimuli.stimulus_paths = dict(zip(stimuli['stimulus_id'], map(Path, paths)))
    stimuli.identifier = 'three-local-images'
    subject.start_recording('word_representation')
    activity = subject.process(stimuli)
    with torch.no_grad():
        images = [Image.open(path).convert('RGB') for path in paths]
        pixels = processor(images=images, return_tensors='pt')['pixel_values']
        reference = model(pixel_values=pixels.to(next(model.parameters()).device)).last_hidden_state[:, 0]
    # Match by stimulus ID; UMI is allowed to reorder presentations.
    order = [list(activity.stimulus_id.values).index(identifier) for identifier in stimuli.stimulus_id]
    np.testing.assert_allclose(activity.values[order], reference.cpu().numpy(), atol=1e-6, rtol=1e-6)
    assert subject.supported_modalities == {'vision'}
    assert not any(module._forward_hooks for module in model.modules())
    subject.reset()


@pytest.mark.slow
def test_matches_pinned_author_analysis_on_identical_features(manifest, monkeypatch):
    """Opt-in reference check; needs the public author checkout, no weights or data."""
    import os
    import subprocess
    from brainscore.benchmarks.rendered_word_semantics.data import AUTHOR_REVISION

    location = os.environ.get('UMI_WORD_AUTHOR_REPO')
    if not location:
        pytest.skip('Set UMI_WORD_AUTHOR_REPO to the pinned public author checkout')
    repo = Path(location).resolve()
    assert subprocess.check_output(
        ['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True,
    ).strip() == AUTHOR_REVISION
    assert not subprocess.check_output(
        ['git', '-C', str(repo), 'status', '--porcelain', '--untracked-files=no'], text=True,
    ).strip()
    monkeypatch.syspath_prepend(str(repo))
    from src.analysis import directions, lexical

    task = benchmark(manifest, batch_size=3)
    subject = FeatureSubject(manifest)
    score = task(subject)
    features = np.stack([
        subject.features[row.image_path] for row in task.rows.itertuples()
    ]).astype(np.float16)[:, None, :]
    monkeypatch.setattr(directions, 'manifest', lambda ds=None: task.rows)
    monkeypatch.setattr(directions, 'word_order', lambda ds=None: tuple(task.words))
    monkeypatch.setattr(
        directions, 'load_cache', lambda key, ds=None: (features, np.array([1]), 1),
    )
    reference = directions.word_directions('same-features', center=True)
    np.testing.assert_array_equal(score.attrs['directions'], reference)
    ranks = lexical.partner_ranks(
        lexical.similarity(reference),
        dict(zip(task.words, range(len(task.words)))),
        task.pairs,
    )
    np.testing.assert_array_equal(score.attrs['ranks'], ranks)


@pytest.mark.integration
def test_cli_writes_measured_artifacts_and_refuses_overwrite(tmp_path, monkeypatch):
    """Exercise the runnable example without network access or trained weights."""
    import importlib.util
    import json
    import sys
    import torch
    import transformers

    path = Path(__file__).parents[1] / 'examples/rendered_word_semantics.py'
    spec = importlib.util.spec_from_file_location('rendered_words_example', path)
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    rows = []
    for number in range(12):
        for condition in ['word', 'scramble']:
            path = tmp_path / f'{number}-{condition}.png'
            color = (number * 20, 50, 80) if condition == 'word' else (0, 0, 0)
            Image.new('RGB', (16, 16), color).save(path)
            rows.append(dict(word=f'word{number}', slot=0, condition=condition, image_path=str(path)))
    monkeypatch.setattr(
        example, 'load_author_data',
        lambda **kwargs: (pd.DataFrame(rows), [('word0', 'word1')]),
    )
    config = transformers.Dinov2Config(
        image_size=16, patch_size=4, hidden_size=12,
        num_hidden_layers=1, num_attention_heads=3,
    )

    class Processor:
        def __call__(self, *, images, return_tensors):
            return {'pixel_values': torch.stack([
                torch.tensor(np.asarray(image).copy()).permute(2, 0, 1).float() / 255
                for image in images
            ])}

        def to_dict(self):
            return {'test_processor': True}

    monkeypatch.setattr(transformers.AutoConfig, 'from_pretrained', lambda *args, **kwargs: config)
    monkeypatch.setattr(transformers.AutoImageProcessor, 'from_pretrained', lambda *args, **kwargs: Processor())
    monkeypatch.setattr(
        transformers.AutoModel, 'from_pretrained',
        lambda *args, **kwargs: transformers.Dinov2Model(config).eval(),
    )
    monkeypatch.setenv('RESULTCACHING_DISABLE', '1')
    output = tmp_path / 'run'
    monkeypatch.setattr(sys, 'argv', [
        'rendered_word_semantics.py', '--author-repo', str(tmp_path),
        '--data-root', str(tmp_path), '--revision', 'a' * 40,
        '--output-dir', str(output),
    ])
    example.main()
    result = json.loads((output / 'result.json').read_text())
    assert result['status'] == 'complete'
    assert result['pilot'] is True
    assert result['paper_replication_verified'] is False
    assert result['queries'][0]['rank'] >= 1
    with np.load(output / 'word_directions.npz', allow_pickle=False) as arrays:
        assert arrays['directions'].shape == (12, 12)
        assert len(arrays['words']) == 12
    with pytest.raises(SystemExit):
        example.main()
