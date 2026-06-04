"""TextWrapper per-token timestamps — the continuous-transcript path.

The pure word→token alignment helper is tested directly. The wrapper
integration (word_id capture + attach_timestamps) is tested offline with a
tiny WordLevel fast tokenizer + a 1-layer embedding model — no model download.
"""
import os

import numpy as np
import pytest

from brainscore.model_helpers.text_wrapper import (
    TextWrapper, align_word_times_to_tokens)


@pytest.fixture(autouse=True)
def _no_result_cache(monkeypatch):
    # the per-token path goes through @store_xarray; disable disk caching so the
    # offline tests don't depend on a writable ~/.result_caching.
    monkeypatch.setenv('RESULTCACHING_DISABLE', '1')


# ── pure helper ────────────────────────────────────────────────────

class TestAlignWordTimesToTokens:
    def test_one_token_per_word(self):
        s, e = align_word_times_to_tokens(
            [0, 1, 2], [0.0, 100.0, 200.0], [100.0, 100.0, 100.0])
        assert list(s) == [0.0, 100.0, 200.0]
        assert list(e) == [100.0, 200.0, 300.0]

    def test_subword_tokens_share_word_window(self):
        # word 1 split into two subword tokens → both get word 1's window
        s, e = align_word_times_to_tokens(
            [0, 1, 1, 2], [0.0, 100.0, 300.0], [100.0, 200.0, 50.0])
        assert list(s) == [0.0, 100.0, 100.0, 300.0]
        assert list(e) == [100.0, 300.0, 300.0, 350.0]

    def test_special_tokens_bridged(self):
        # leading + trailing special tokens (word_id None) borrow neighbours
        s, e = align_word_times_to_tokens(
            [None, 0, 1, None], [10.0, 20.0], [5.0, 5.0])
        assert s[0] == 10.0          # bridged from first real word's start
        assert e[-1] == 25.0         # bridged from last real word's end
        assert s[1] == 10.0 and e[2] == 25.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match='same length'):
            align_word_times_to_tokens([0, 1], [0.0], [0.0, 1.0])


# ── wrapper integration (offline) ──────────────────────────────────

def _tiny_fast_tokenizer():
    tokenizers = pytest.importorskip('tokenizers')
    from transformers import PreTrainedTokenizerFast
    vocab = {'[PAD]': 0, '[UNK]': 1, 'the': 2, 'cat': 3, 'sat': 4,
             'on': 5, 'mat': 6, 'dog': 7, 'ran': 8}
    tok = tokenizers.Tokenizer(
        tokenizers.models.WordLevel(vocab=vocab, unk_token='[UNK]'))
    tok.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=tok, unk_token='[UNK]', pad_token='[PAD]')


class _TinyModel:
    """1-layer embedding 'model' with a hookable submodule named ``enc``."""
    def __init__(self, vocab=9, hidden=4):
        import torch
        from torch import nn
        self._t = torch
        self.enc = nn.Embedding(vocab, hidden)

    def eval(self):
        self.enc.eval()
        return self

    def __call__(self, input_ids=None, attention_mask=None, **kw):
        return self.enc(input_ids)


@pytest.fixture
def tiny_wrapper():
    import torch
    from torch import nn
    tok = _tiny_fast_tokenizer()
    model = _TinyModel()

    class Net(nn.Module):
        def __init__(self, enc):
            super().__init__()
            self.enc = enc

        def forward(self, input_ids=None, attention_mask=None, **kw):
            return self.enc(input_ids)

    net = Net(model.enc)
    return TextWrapper(model=net, tokenizer=tok, identifier='tiny',
                       layer_aggregation='per_token', max_length=64)


def _stim(sentence):
    from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
    ss = StimulusSet({'stimulus_id': ['s0'], 'sentence': [sentence]})
    ss.identifier = 'tiny-stim'
    return ss


def test_per_token_shape_and_axes(tiny_wrapper):
    asm = tiny_wrapper(_stim('the cat sat on mat'), layers=['enc'])
    assert set(asm.dims) == {'presentation', 'time_bin', 'neuroid'}
    assert asm.sizes['time_bin'] == 5            # 5 words → 5 tokens (WordLevel)
    assert 'token_position' in asm.coords
    assert list(asm['token_position'].values) == [0, 1, 2, 3, 4]
    assert 'word_id' in asm.coords


def test_attach_timestamps(tiny_wrapper):
    asm = tiny_wrapper(_stim('the cat sat'), layers=['enc'])
    onsets = [0.0, 100.0, 250.0]
    durations = [100.0, 150.0, 50.0]
    timed = TextWrapper.attach_timestamps(asm, onsets, durations)
    assert 'time_bin_start_ms' in timed.coords
    assert 'time_bin_end_ms' in timed.coords
    start = np.asarray(timed['time_bin_start_ms'].values).reshape(-1)
    end = np.asarray(timed['time_bin_end_ms'].values).reshape(-1)
    assert list(start) == [0.0, 100.0, 250.0]
    assert list(end) == [100.0, 250.0, 300.0]


def test_attach_timestamps_without_word_id_raises():
    import xarray as xr
    asm = xr.DataArray(np.zeros((1, 2, 3)),
                       dims=['presentation', 'time_bin', 'neuroid'])
    with pytest.raises(ValueError, match='word_id'):
        TextWrapper.attach_timestamps(asm, [0.0], [1.0])
