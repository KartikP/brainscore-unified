
from brainscore_core.compatibility import (
    check_compatibility,
)


# ── Helpers ──────────────────────────────────────────────────────────


class FakeBenchmark:

    def __init__(self, identifier='test-bench', required_modalities=None,
                 available_modalities=None, accepted_modalities=None, region=None):
        self.identifier = identifier
        if required_modalities is not None:
            self.required_modalities = required_modalities
        if available_modalities is not None:
            self.available_modalities = available_modalities
        if accepted_modalities is not None:
            self.accepted_modalities = accepted_modalities
        if region is not None:
            self.region = region


# ── Required modalities (hard gate) ─────────────────────────────────


# ── Accepted modalities (any-of gate) ───────────────────────────────


# ── Deprecated benchmark.available_modalities ────────────────────────


# ── Region mapping ──────────────────────────────────────────────────


# ── Combined scenarios ──────────────────────────────────────────────


# ── Model-side required modalities (hard gate) ──────────────────────


# ── Available-modalities API on UnifiedModel ────────────────────────


class TestDeclaredChannelsRun:
    """Contract v3 rule 2.4: every declared channel passes pre-flight and runs."""

    def _model(self):
        from brainscore import BrainScoreModel
        return BrainScoreModel(
            'all-parts', region_layer_map={'IT': 'layer'},
            extractor={'vision': lambda stimuli, layers: None},
            behavioral_readout_layer='layer',
            action_fn=lambda step: None,
            state_change_fn=lambda change: (None, lambda: None))

    def test_every_part_declares_registered_channels(self):
        model = self._model()
        assert model.out_channels == {'neural:IT', 'behavior', 'behavior:choice',
                                      'behavior:probabilities', 'behavior:rating', 'behavior:rt',
                                      'motor', 'perturbation'}
        assert {'vision', 'observation', 'lesion', 'stimulation'} <= model.in_channels
        for output in sorted(model.out_channels):
            bench = FakeBenchmark(identifier=output)
            bench.requested_output_channels = {output}
            bench.required_input_channels = {'vision'}
            check_compatibility(model, bench)  # registered and produced

    def test_every_declared_output_has_one_session_handler(self):
        model = self._model()
        for output in sorted(model.out_channels):
            assert model.check_session_support({output}) is not None
