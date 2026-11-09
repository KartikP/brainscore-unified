"""The memory pre-flight starts no layer search for a legacy model with a lazy map.

A legacy vision ``ModelCommitment`` with no committed ``region_layer_map``
searches for its layers at the first use of a region. On a benchmark that
declares no region and no requested channels (for example a legacy behavioral
benchmark), the memory probe must not start that search: it probes only the
subject's ``probe_safe_regions``, which for a lazy map are its committed layers.

This test needs core and vision together (core ``probe_safe_regions`` in the
memory probe, vision's adapter declaring it), so it lives in unified, which pins
both. Only the layer search and the activations are stubs; no weights or
benchmark data are loaded.
"""
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from brainscore_core.compatibility import ensure_legacy_benchmark_modalities
from brainscore_core.legacy_facade import mark_legacy_benchmark
from brainscore_core.memory import check_memory
from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
from brainscore_vision.compat.unified_adapter import VisionModelAdapter
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment, RegionLayerMap


def _stimuli(n=2):
    ids = [f's{i}' for i in range(n)]
    stimuli = StimulusSet(pd.DataFrame({'stimulus_id': ids,
                                        'image_path': [f'{i}.png' for i in ids],
                                        'object_name': ['cat', 'dog'][:n]}))
    stimuli.identifier = 'synthetic-lazy-regions'
    return stimuli


class _CountingLayerSelection:
    """Stands in for LayerSelection: one call is one layer search."""

    def __init__(self, layer='l2'):
        self.layer = layer
        self.searches = []

    def __call__(self, selection_identifier, benchmark):
        self.searches.append(selection_identifier)
        return self.layer


class _ActivationsModel:
    """Stands in for the activations model."""
    identifier = 'stub-activations'

    def __init__(self):
        self._extractor = MagicMock()

    def __call__(self, stimuli, layers, number_of_trials=1, require_variance=None):
        n = len(stimuli)
        return NeuroidAssembly(
            np.arange(n * 2 * len(layers), dtype=float).reshape(n, 2 * len(layers)),
            coords={'stimulus_id': ('presentation', list(stimuli['stimulus_id'])),
                    'object_name': ('presentation', list(stimuli['object_name'])),
                    'neuroid_id': ('neuroid', [f'{layer}.{i}' for layer in layers for i in range(2)]),
                    'layer': ('neuroid', [layer for layer in layers for _ in range(2)])},
            dims=['presentation', 'neuroid'])


class _LegacyBehavioralBenchmark:
    """A legacy behavioral benchmark: it declares no region and no requested channels."""
    identifier = 'Rajalingham2018-i2n'

    def __init__(self):
        self.stimulus_set = _stimuli()


def test_memory_probe_on_a_behavioral_benchmark_runs_no_search():
    model = ModelCommitment('lazy-regions-test-no-such-plugin', _ActivationsModel(), layers=['l1', 'l2'])
    region_layer_map = model.layer_model.region_layer_map
    assert isinstance(region_layer_map, RegionLayerMap) and len(region_layer_map) == 0
    selection = _CountingLayerSelection()
    region_layer_map.layer_selection = selection

    benchmark = mark_legacy_benchmark(
        ensure_legacy_benchmark_modalities(_LegacyBehavioralBenchmark(), {'vision'}))
    check_memory(VisionModelAdapter(model), benchmark)

    assert selection.searches == []
