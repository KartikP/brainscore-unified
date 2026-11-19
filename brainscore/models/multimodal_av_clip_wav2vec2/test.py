import socket

from pathlib import Path

import pytest

from brainscore import load_model
from brainscore_core.plugin_management.parse_plugin_changes import get_plugin_ids

HERE = Path(__file__).parent


@pytest.mark.parametrize('identifier', get_plugin_ids('models', [HERE.name], str(HERE.parents[1])))
def test_has_identifier(identifier):
    model = load_model(identifier)
    assert model.identifier == identifier


@pytest.mark.parametrize('identifier', ['clip-wav2vec', 'vjepa1-wav2vec2', 'random-vjepa1-wav2vec2'])
def test_a_wrong_identifier_is_refused_before_any_tower(identifier, monkeypatch):
    from .model import get_model

    def no_network(*args, **kwargs):
        raise OSError('the network is blocked in this check')
    monkeypatch.setenv('HF_HUB_OFFLINE', '1')
    monkeypatch.setattr(socket.socket, 'connect', no_network)
    with pytest.raises(ValueError, match='list_models'):
        get_model(identifier)
