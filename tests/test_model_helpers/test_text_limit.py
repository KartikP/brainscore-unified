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


def test_the_limit_is_in_the_recorded_settings_and_the_cache_key():
    short, long = (TextWrapper(_network(), _Tokenizer(limit), device='cpu') for limit in (512, 1024))
    assert short.cache_config()['settings']['_max_length'] == 512
    assert short.cache_config() != long.cache_config()
