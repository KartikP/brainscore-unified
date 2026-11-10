"""Contract conformance for ``participate(session)`` (Contract Rules v3, rule 1).

A subject that implements only ``participate`` must run through every public
path: the static typed helpers, the legacy-benchmark facade and a closed-loop
environment. An added capability that claims the same outputs as a built-in
part runs instead of it. No class carries a second evaluation method.
"""

import numpy as np
import pandas as pd
import pytest

from brainscore import capabilities
from brainscore.capabilities import Capability, register_capability
from brainscore import BrainScoreModel
from brainscore_core.streaming import StreamEvent
from brainscore_core.streaming_helpers import table_session
from brainscore_core.streaming_helpers import (
    neural_response, stimulus_session,
)
from brainscore_core.supported_data_standards.brainio.assemblies import (
    NeuroidAssembly,
)
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet


def _images(n=2):
    stimuli = StimulusSet(pd.DataFrame({
        'stimulus_id': [f's{i}' for i in range(n)],
        'image_path': [f'/tmp/s{i}.png' for i in range(n)],
    }))
    stimuli.identifier = f'participate-images-{n}'
    return stimuli


def _neural(n):
    return NeuroidAssembly(np.arange(n, dtype=float)[:, None],
                           coords={'stimulus_id': ('presentation', [f's{i}' for i in range(n)]),
                                   'neuroid_id': ('neuroid', [0])},
                           dims=['presentation', 'neuroid'])


def test_concrete_subject_has_no_second_evaluation_method():
    for cls in (BrainScoreModel, Capability):
        assert not hasattr(cls, 'interact'), cls


@pytest.fixture
def registry(monkeypatch):
    saved = dict(capabilities.capability_registry)
    yield
    capabilities.capability_registry.clear()
    capabilities.capability_registry.update(saved)


def test_added_capability_takes_precedence_over_the_built_in_part(registry):
    calls = []

    class Custom(Capability):
        identifier = 'participate-precedence-test'
        output_channels = {'neural:IT'}

        def handles(self, model, event, **kwargs):
            return False

        def process(self, model, event, **kwargs):
            raise NotImplementedError

        def supports_session(self, model, channels):
            return set(channels) == {'neural:IT'}

        def participate(self, model, session):
            while session.next_input() is not None:
                pass
            calls.append(model.identifier)
            session.respond(StreamEvent('neural:IT', _neural(len(session.stimulus_set)), None))

    def extractor(stimuli, layers=None, **kwargs):
        pytest.fail('built-in neural part ran instead of the added capability')

    register_capability(Custom())
    model = BrainScoreModel('precedence', region_layer_map={'IT': 'layer'},
                            extractor={'vision': extractor})
    assert model.check_session_support({'neural:IT'}).identifier == Custom.identifier
    out = neural_response(model, _images(2), record='IT')
    assert calls == ['precedence'] and out.sizes['presentation'] == 2


def _features(stimuli, layers=None, **kwargs):
    return _neural(len(stimuli)).assign_coords(layer=('neuroid', [layers[0]]))


@pytest.mark.parametrize('streaming, meta', [
    (False, [{'driver': 'interact', 'record': 'IT'}]),
    (True, [{'driver': 'interact_streaming', 'record': 'IT', 'stream_index': i, 'streaming': True}
            for i in range(2)]),
])
def test_typed_drivers_keep_the_recorded_pre_rename_driver_labels(streaming, meta):
    """Response meta written before the rename must still compare equal.

    ``driver`` is a record-format label, not the method name; saved records
    compare meta exactly (``compare_outputs``), so the label is frozen.
    """
    model = BrainScoreModel('labels', region_layer_map={'IT': 'layer'},
                            extractor={'vision': _features})
    session = table_session(_images(2), ['neural:IT'])
    session.streaming = streaming
    model.participate(session)
    assert [event.meta for event in session.emitted] == meta
