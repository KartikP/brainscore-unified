"""The text extractor cuts a text at the model's own window unless the registration states a limit."""
import pytest
import torch

from brainscore.model_helpers.text_wrapper import TextWrapper

pytestmark = pytest.mark.unit


class _Tokenizer:
    def __init__(self, model_max_length):
        self.model_max_length = model_max_length


def _network(positions=None):
    network = torch.nn.Linear(1, 1)
    if positions is not None:
        from transformers import GPT2Config
        network.config = GPT2Config(n_positions=positions, n_embd=8, n_layer=1, n_head=1)
    return network


def test_a_tokenizer_limit_is_the_window():
    assert TextWrapper(_network(48), _Tokenizer(40), device='cpu')._max_length == 40


def test_the_smaller_real_limit_is_the_window():
    # A text longer than the decoder's positions cannot be read, whatever the tokenizer states.
    assert TextWrapper(_network(32), _Tokenizer(40), device='cpu')._max_length == 32


def test_a_placeholder_limit_gives_the_position_limit_of_the_configuration():
    # Hugging Face states int(1e30) when a tokenizer has no limit.
    assert TextWrapper(_network(48), _Tokenizer(int(1e30)), device='cpu')._max_length == 48


def test_no_stated_limit_is_refused_with_the_remedy():
    with pytest.raises(ValueError, match='State max_length') as error:
        TextWrapper(_network(), _Tokenizer(int(1e30)), identifier='no-window', device='cpu')
    assert 'no-window' in str(error.value)


def test_a_stated_limit_wins():
    assert TextWrapper(_network(48), _Tokenizer(40), max_length=16, device='cpu')._max_length == 16


def test_no_tokenizer_needs_no_limit():
    assert TextWrapper(_network(), None, device='cpu')._max_length is None


def test_the_limit_alone_changes_the_cache_key(monkeypatch):
    """One network, one tokenizer, one input: only the limit differs, and the real key differs."""
    from brainscore.model_helpers._extractors import cache_signature
    from tests.test_model_helpers.test_wrapper_cache_guard import _TextNet, _eighths, _tokenizer
    monkeypatch.delenv('RESULTCACHING_DISABLE', raising=False)     # the key is made only with the store on
    network, tokenizer = _eighths(_TextNet(), 1), _tokenizer()
    window, stated = (TextWrapper(network, tokenizer, max_length=limit, device='cpu') for limit in (None, 32))
    assert window.cache_config()['settings']['_max_length'] == 64
    keys = [cache_signature(wrapper, '_from_texts_stored', ['the cat'], 'texts-1') for wrapper in (window, stated)]
    assert None not in keys and keys[0] != keys[1]


class _Seen(torch.nn.Module):
    """Keeps the token ids of its last pass."""

    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(9, 2)

    def forward(self, input_ids=None, attention_mask=None, **_):
        self.seen = input_ids.tolist()
        return self.embed(input_ids)


def test_a_long_text_keeps_its_newest_tokens():
    from tests.test_model_helpers.test_wrapper_cache_guard import _tokenizer
    network, tokenizer = _Seen(), _tokenizer()
    TextWrapper(network, tokenizer, max_length=2, device='cpu')(['the cat sat'], ['embed'])
    assert network.seen == [tokenizer.convert_tokens_to_ids(['cat', 'sat'])]


class _OutOfMemory(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(9, 2)

    def forward(self, input_ids=None, attention_mask=None, **_):
        raise torch.OutOfMemoryError('out of memory')


def test_an_out_of_memory_batch_names_the_limit_and_the_remedy():
    from tests.test_model_helpers.test_wrapper_cache_guard import _tokenizer
    wrapper = TextWrapper(_OutOfMemory(), _tokenizer(), identifier='big', max_length=48, device='cpu')
    with pytest.raises(torch.OutOfMemoryError) as error:
        wrapper(['the cat sat', 'dog'], ['embed'])
    assert 'out of memory' in str(error.value)   # the original error, not a new one
    note = ' '.join(error.value.__notes__)
    for fragment in ('big', 'limit 48', '3 tokens', 'smaller max_length'):
        assert fragment in note


def test_an_out_of_memory_batch_without_token_ids_keeps_the_memory_error():
    class _Embeddings:
        model_max_length = 8

        def __call__(self, texts, **_):
            return {'inputs_embeds': torch.zeros(len(texts), 3, 2)}

    class _Network(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(9, 2)

        def forward(self, **_):
            raise torch.OutOfMemoryError('out of memory')

    wrapper = TextWrapper(_Network(), _Embeddings(), identifier='big', device='cpu')
    with pytest.raises(torch.OutOfMemoryError) as error:
        wrapper(['the cat sat'], ['embed'])
    assert '3 tokens' in ' '.join(error.value.__notes__)
