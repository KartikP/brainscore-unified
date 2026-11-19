"""The load of each registered identifier is checked by the generic plugin suite
(``brainscore/model_helpers/generic_plugin_tests.py``)."""
import socket

import pytest


@pytest.mark.parametrize('identifier', ['clip-wav2vec', 'vjepa1-wav2vec2', 'random-vjepa1-wav2vec2'])
def test_a_wrong_identifier_is_refused_before_any_tower(identifier, monkeypatch):
    from .model import get_model

    def no_network(*args, **kwargs):
        raise OSError('the network is blocked in this check')
    monkeypatch.setenv('HF_HUB_OFFLINE', '1')
    monkeypatch.setattr(socket.socket, 'connect', no_network)
    with pytest.raises(ValueError, match='list_models'):
        get_model(identifier)
