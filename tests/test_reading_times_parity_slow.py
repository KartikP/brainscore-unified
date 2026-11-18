"""Parity of the reading times of unified ``gpt2-small`` with the old ``gpt2`` on Futrell2018.

Real weights and data (slow tier, cloud). Both registrations load the checkpoint ``gpt2``;
the test refuses any snapshot other than the qualified one (``GPT2_SNAPSHOT``) and a tokenizer
whose Futrell2018 ids differ from the qualified ones, and it needs both models on one device
with one attention implementation. The evidence comes from the two scored runs themselves
(``brainscore.score``): the token ids fed to every forward pass of each model, and the
predictions the facade hands to Futrell2018. It compares those ids, the NaN mask, the dtype
and the bytes of every finite value, and the raw and normalized scores. It writes what it ran
to ``READING_TIMES_PARITY_OUT`` (default ``reading-times-parity.json``). No model is moved.
"""
import hashlib
import json
import os

import numpy as np
import pytest

pytestmark = pytest.mark.slow

# The qualified snapshot of the Hub repository 'gpt2' (model and tokenizer files), and the digest
# of the ids its tokenizer gives for the joined Futrell2018 text (12,370 tokens; the fast and the
# slow GPT-2 tokenizer give the same ids). A later snapshot fails the test instead of passing.
GPT2_SNAPSHOT = '607a30d783dfa663caf39e06633721c8d4cfcd7e'
FUTRELL_IDS = (12370, 'e975c59f453e6f5acfa40054e69078f2')


def _digest(arrays) -> str:
    """One digest of a sequence of id arrays, their lengths included."""
    md5 = hashlib.md5()
    for array in arrays:
        md5.update(np.int64(len(array)).tobytes())
        md5.update(np.asarray(array, dtype=np.int64).tobytes())
    return md5.hexdigest()


def test_gpt2_small_reading_times_equal_the_old_gpt2_on_futrell2018(monkeypatch):
    import torch
    import transformers
    import brainscore
    from brainscore import model_registry
    from brainscore_core.legacy_facade import LegacyBenchmarkFacade, READING_TIME_POLICY
    from brainscore_core.text import prepare_context
    from brainscore_language import load_benchmark, load_model

    benchmark = load_benchmark('Futrell2018-pearsonr')
    words = list(benchmark.data['word'].values)
    old = load_model('gpt2')                       # the language adapter of the old model
    reference = old.legacy_model                   # the old HuggingfaceSubject
    new = model_registry['gpt2-small']()
    parts = new._segment_likelihood.__self__       # the shared causal-model parts
    snapshots = [reference.basemodel.config._commit_hash, parts.model.config._commit_hash]
    assert snapshots == [GPT2_SNAPSHOT, GPT2_SNAPSHOT], snapshots
    for tokenizer in (reference.tokenizer, parts.tokenizer):
        ids = np.asarray(tokenizer(prepare_context(words))['input_ids'], dtype=np.int64)
        assert (len(ids), hashlib.md5(ids.tobytes()).hexdigest()) == FUTRELL_IDS

    # What each model is fed during its scored run, and what Futrell2018 receives.
    fed = {'old': [], 'new': []}

    def recorder(name):
        def record(module, args, kwargs):
            ids = kwargs['input_ids'] if 'input_ids' in kwargs else args[0]
            fed[name].append(ids.detach().cpu().numpy().ravel())
        return record

    reference.basemodel.register_forward_pre_hook(recorder('old'), with_kwargs=True)
    parts.model.register_forward_pre_hook(recorder('new'), with_kwargs=True)
    received = []
    digest_text = LegacyBenchmarkFacade.digest_text

    def recording_digest_text(self, text, *args, **kwargs):
        output = digest_text(self, text, *args, **kwargs)
        received.append(output['behavior'])
        return output

    monkeypatch.setattr(LegacyBenchmarkFacade, 'digest_text', recording_digest_text)
    old_score = brainscore.score(old, benchmark, check_mem=False)
    new_score = brainscore.score(new, benchmark, check_mem=False)
    old_values, new_values = (assembly.values for assembly in received)
    old_nan, new_nan = np.isnan(old_values), np.isnan(new_values)
    finite = ~old_nan
    record = {
        'transformers': transformers.__version__, 'torch': torch.__version__, 'snapshots': snapshots,
        'tokenizers': [type(reference.tokenizer).__name__, type(parts.tokenizer).__name__],
        'devices': [str(next(reference.basemodel.parameters()).device),
                    str(next(parts.model.parameters()).device)],
        'attention': [reference.basemodel.config._attn_implementation,
                      parts.model.config._attn_implementation],
        'parts': len(words), 'tokens': FUTRELL_IDS[0],
        'window': reference.basemodel.config.max_position_embeddings,
        'forward_passes': [len(fed['old']), len(fed['new'])],
        'fed_ids': [_digest(fed['old']), _digest(fed['new'])],
        'predictions': [len(old_values), len(new_values)],
        'nan_masks_equal': bool(np.array_equal(old_nan, new_nan)),
        'dtypes': [str(old_values.dtype), str(new_values.dtype)],
        'finite_bytes_equal': bool(old_values.dtype == new_values.dtype and np.array_equal(old_nan, new_nan)
                                   and old_values[finite].tobytes() == new_values[finite].tobytes()),
        'raw': [float(old_score.attrs['raw']), float(new_score.attrs['raw'])],
        'score': [float(old_score), float(new_score)],
        'new_answer_policy': new_score.attrs.get('answer_policy'),
        'new_n_invalid': new_score.attrs.get('n_invalid'),
    }
    with open(os.environ.get('READING_TIMES_PARITY_OUT', 'reading-times-parity.json'), 'w') as f:
        json.dump(record, f, indent=1)
    assert len(received) == 2                            # one prediction call in each scored run
    assert min(record['forward_passes']) > 0            # the hooks saw the forwards of both runs
    assert record['tokens'] > record['window']          # the passage crosses the window
    assert record['devices'][0] == record['devices'][1] and record['attention'][0] == record['attention'][1]
    assert record['fed_ids'][0] == record['fed_ids'][1]
    assert record['nan_masks_equal'] and record['dtypes'][0] == record['dtypes'][1]
    assert record['finite_bytes_equal']
    assert READING_TIME_POLICY in (record['new_answer_policy'] or '')
    assert record['new_n_invalid'] == int(new_nan.any())   # the one passage trial
    assert record['raw'][0] == record['raw'][1] and record['score'][0] == record['score'][1]
