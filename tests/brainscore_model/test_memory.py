"""The memory pre-flight probes the feature width through participate, for every subject."""
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from brainscore import BrainScoreModel
from brainscore_core.memory import MemoryError, check_memory
from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet


def _stimuli():
    stimuli = StimulusSet(pd.DataFrame({'stimulus_id': ['a', 'b'],
                                        'image_path': ['/tmp/a.png', '/tmp/b.png']}))
    stimuli.identifier = 'probe-images'
    return stimuli


def _probed_width(subject, **declarations):
    """Run check_memory with no host memory; the refusal names the probed width."""
    benchmark = SimpleNamespace(identifier='bench-ridge', stimulus_set=_stimuli(), **declarations)
    with patch('brainscore_core.memory.get_host_available_memory', return_value=1):
        with pytest.raises(MemoryError) as refusal:
            check_memory(subject, benchmark)
    return str(refusal.value)


class TestProbeThroughParticipate:
    """The feature-width probe runs one session through participate for every subject."""

    def test_participate_only_subject_is_probed_from_its_declarations(self):
        from brainscore_core.contract import Subject
        from brainscore_core.streaming import StreamEvent

        class Native(Subject):
            identifier = 'native'
            in_channels = {'vision'}
            out_channels = {'neural:IT'}
            required_channels = set()
            sessions = []

            def participate(self, session):
                self.sessions.append(tuple(session.outputs))
                n = 0
                while session.next_input() is not None:
                    n += 1
                session.respond(StreamEvent('neural:IT', NeuroidAssembly(
                    np.zeros((n, 7)), dims=['presentation', 'neuroid'],
                    coords={'stimulus_id': ('presentation', list(range(n))),
                            'neuroid_id': ('neuroid', list(range(7)))}), None))

        subject = Native()
        assert 'n_features: 7 ' in _probed_width(subject)
        assert subject.sessions == [('neural:IT',)]

    def test_typed_model_is_probed_the_same_way(self):
        layers_seen = []

        def extractor(stimuli, layers=None, **kwargs):
            layers_seen.append(list(layers))
            n = len(stimuli)
            return NeuroidAssembly(np.zeros((n, 5)), dims=['presentation', 'neuroid'],
                                   coords={'stimulus_id': ('presentation', list(stimuli['stimulus_id'])),
                                           'neuroid_id': ('neuroid', list(range(5))),
                                           'layer': ('neuroid', [layers[0]] * 5)})

        model = BrainScoreModel('typed', region_layer_map={'V4': 'early', 'IT': 'late'},
                                extractor={'vision': extractor})
        assert 'n_features: 5 ' in _probed_width(model)
        assert layers_seen == [['early']]                 # one region, in region map order
        assert 'n_features: 5 ' in _probed_width(model, region='IT')
        assert layers_seen[-1] == ['late']                # the benchmark's region

    def test_no_neural_output_means_no_probe(self, caplog):
        import logging
        behavioral = BrainScoreModel('policy', policy=lambda step: None)
        benchmark = SimpleNamespace(identifier='bench-ridge', stimulus_set=_stimuli())
        # No width can be probed, so the check is skipped (loudly) even with no memory.
        with patch('brainscore_core.memory.get_host_available_memory', return_value=1), \
                caplog.at_level(logging.WARNING, logger='brainscore_core.memory'):
            check_memory(behavioral, benchmark)
        assert any('could NOT run' in record.getMessage() for record in caplog.records)
