


# ── Helpers ──────────────────────────────────────────────────────────




# ── get_available_memory ─────────────────────────────────────────────


# ── get_peak_memory ──────────────────────────────────────────────────


# ── reset_peak_memory ────────────────────────────────────────────────


# ── _detect_metric_category ──────────────────────────────────────────


# ── estimate_metric_memory ───────────────────────────────────────────


# ── check_memory ─────────────────────────────────────────────────────


class TestProbeThroughParticipate:
    """The feature-width probe runs one session through participate for every subject."""

    def _stimuli(self):
        import pandas as pd
        from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
        stimuli = StimulusSet(pd.DataFrame({'stimulus_id': ['a', 'b'],
                                            'image_path': ['/tmp/a.png', '/tmp/b.png']}))
        stimuli.identifier = 'probe-images'
        return stimuli

    def test_participate_only_subject_is_probed_from_its_declarations(self):
        from types import SimpleNamespace
        import numpy as np
        from brainscore_core.memory import _probe_feature_dim, _probe_regions
        from brainscore_core.model_interface import Subject
        from brainscore_core.streaming import StreamEvent
        from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly

        class Native(Subject):
            identifier = 'native'
            in_channels = {'vision'}
            out_channels = {'neural:IT'}
            required_channels = set()

            def participate(self, session):
                n = 0
                while session.next_input() is not None:
                    n += 1
                session.respond(StreamEvent('neural:IT', NeuroidAssembly(
                    np.zeros((n, 7)), dims=['presentation', 'neuroid'],
                    coords={'stimulus_id': ('presentation', list(range(n))),
                            'neuroid_id': ('neuroid', list(range(7)))}), None))

        benchmark = SimpleNamespace(identifier='bench')
        assert _probe_regions(Native(), benchmark) == ['IT']
        width, region = _probe_feature_dim(Native(), benchmark, self._stimuli())
        assert (width, region) == (7, 'IT')

    def test_typed_model_is_probed_the_same_way(self):
        from types import SimpleNamespace
        import numpy as np
        from brainscore_core.memory import _probe_feature_dim, _probe_regions
        from brainscore import BrainScoreModel
        from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly

        def extractor(stimuli, layers=None, **kwargs):
            n = len(stimuli)
            return NeuroidAssembly(np.zeros((n, 5)), dims=['presentation', 'neuroid'],
                                   coords={'stimulus_id': ('presentation', list(stimuli['stimulus_id'])),
                                           'neuroid_id': ('neuroid', list(range(5))),
                                           'layer': ('neuroid', [layers[0]] * 5)})

        model = BrainScoreModel('typed', region_layer_map={'V4': 'early', 'IT': 'late'},
                                extractor={'vision': extractor})
        benchmark = SimpleNamespace(identifier='bench')
        # Region map order kept; the declared neural:all (every mapped region) comes last.
        assert _probe_regions(model, benchmark) == ['V4', 'IT', 'all']
        width, region = _probe_feature_dim(model, benchmark, self._stimuli())
        assert (width, region) == (5, 'V4')
        width, region = _probe_feature_dim(model, SimpleNamespace(identifier='it', region='IT'),
                                           self._stimuli())
        assert (width, region) == (5, 'IT')

    def test_no_neural_output_means_no_probe(self):
        from types import SimpleNamespace
        from brainscore_core.memory import _probe_feature_dim, _probe_regions
        from brainscore import BrainScoreModel
        behavioral = BrainScoreModel('policy', policy=lambda step: None)
        assert _probe_regions(behavioral, SimpleNamespace(identifier='b')) == [None]
        assert _probe_feature_dim(behavioral, SimpleNamespace(identifier='b'), self._stimuli()) == (None, None)
