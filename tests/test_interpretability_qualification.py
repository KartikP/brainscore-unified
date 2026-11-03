"""Qualification gates run offline in CI; trained checkpoints run explicitly."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture(autouse=True)
def qualification_imports(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'examples'))
    monkeypatch.setenv('RESULTCACHING_DISABLE', '1')


class Tokenizer:
    """Fixture tokenization; this is not a pretrained tokenizer."""
    def __call__(self, texts, **kwargs):
        ids = [[1 + sum(word.encode()) % 127 for word in text.split()] for text in texts]
        length = max(map(len, ids))
        return {'input_ids': torch.tensor([row + [0] * (length - len(row)) for row in ids]),
                'attention_mask': torch.tensor([[1] * len(row) + [0] * (length - len(row)) for row in ids])}


class Encoder(torch.nn.Module):
    """Small contextual encoder fixture for testing the qualification procedure."""
    def __init__(self):
        super().__init__()
        with torch.random.fork_rng():
            torch.manual_seed(12)
            self.embedding = torch.nn.Embedding(128, 8)
            self.final_norm = torch.nn.LayerNorm(8)

    def forward(self, input_ids, attention_mask):
        values = self.embedding(input_ids) * attention_mask[..., None]
        return SimpleNamespace(last_hidden_state=self.final_norm(values.cumsum(1)))


def test_complete_qualification_gates_on_small_fixture(tmp_path):
    from interpretability_qualification import check_model
    report = check_model(tmp_path, Encoder(), Tokenizer(), 'final_norm', {'fixture': True})
    assert report['checks'] == 'passed'
    assert report['n_conditions'] == 9 and report['n_discovery'] == 32 and report['n_evaluation'] == 8
    assert report['max_absolute_errors']['failure_restoration'] == 0
    assert report['max_absolute_errors']['exact_patch'] == 0
    assert report['evidence']['experiment/inputs_outputs/manifest.json']
    assert all(0 <= value <= 1 for value in report['diagnostics'].values())


@pytest.mark.parametrize('observed, expected', [([float('nan')], [0.]), ([1.], [0.]), ([0., 1.], [0.])])
def test_incorrect_or_unmeasured_values_fail_qualification(observed, expected):
    from interpretability_qualification import check_close
    with pytest.raises(RuntimeError, match='Qualification check failed'):
        check_close(observed, expected, 'deliberately invalid')


def test_native_reference_detects_an_extractor_output_mismatch(tmp_path):
    from interpretability_qualification import check_model

    class DifferentOutput(Encoder):
        def forward(self, **kwargs):
            output = super().forward(**kwargs)
            # The wrapper records final_norm; a claimed final output must agree.
            return SimpleNamespace(last_hidden_state=output.last_hidden_state + 0.25)

    with pytest.raises(RuntimeError, match='native/session: numerical equality'):
        check_model(tmp_path, DifferentOutput(), Tokenizer(), 'final_norm', {'fixture': True})


def test_missing_checkpoint_retains_failed_report_before_loading(tmp_path):
    from interpretability_qualification import qualify
    with pytest.raises(FileNotFoundError):
        qualify(tmp_path / 'out', 'gpt2', tmp_path / 'absent')
    report = json.loads((tmp_path / 'out' / 'qualification.json').read_text())
    assert report['status'] == 'failed' and 'results' not in report and 'identity' not in report
    assert len(report['driver_sha256']) == 64


def test_discovery_and_evaluation_do_not_overlap():
    from interpretability_qualification import sentences
    discovery, labels, evaluation, held_out = sentences()
    assert len(discovery) == len(labels) == len(set(discovery)) == 32
    assert len(evaluation) == len(held_out) == len(set(evaluation)) == 8
    assert not set(discovery) & set(evaluation)


@pytest.mark.slow
@pytest.mark.parametrize('key', ['gpt2', 'clip'])
def test_pinned_pretrained_encoders_qualify(tmp_path, key):
    from huggingface_hub.constants import HF_HUB_CACHE
    from interpretability_qualification import MODELS, qualify
    manifest = MODELS[key]
    cache_name = 'models--' + manifest['repository'].replace('/', '--')
    snapshot = Path(HF_HUB_CACHE) / cache_name / 'snapshots' / manifest['revision']
    if key == 'gpt2' and not snapshot.exists():
        snapshot = Path(HF_HUB_CACHE) / 'models--gpt2' / 'snapshots' / manifest['revision']
    snapshot = Path(os.environ.get(f'UMI_INTERPRETABILITY_{key.upper()}_SNAPSHOT', snapshot))
    directory = Path(os.environ.get('UMI_INTERPRETABILITY_OUTPUT', tmp_path)) / key
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(2)
        report = qualify(directory, key, snapshot)
    finally:
        torch.set_num_threads(previous)
    assert report['status'] == 'passed'
    assert report['identity']['revision'] == manifest['revision']
    assert report['results']['checks'] == 'passed'
