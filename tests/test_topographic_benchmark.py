"""End-to-end wiring test for the topographic-organization axis (offline, synthetic).

Proves the benchmark drives a candidate through process(), obtains unit positions, scores via
the registered topographic-alignment metric, and subtracts the shuffle-coordinate null — and
that a topographic layout clears the null while a non-topographic one does not. A real
registered instance swaps the synthetic brain/candidate for a surface fMRI assembly + a real
topographic model (EC2); the wiring under test is identical.
"""
import numpy as np
import pandas as pd

from brainscore_core.contract import Subject
from brainscore_core.streaming import StreamEvent
from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
from brainscore.benchmarks.topographic import TopographicBenchmark


def _assembly(seed=0, n_units=64, n_stim=30, smooth=True, with_coords=True):
    """A (presentation, neuroid) assembly on a 2-D grid. smooth -> responses vary
    smoothly with position (topographic); with_coords -> carries tissue_x/tissue_y."""
    rng = np.random.RandomState(seed)
    side = int(round(n_units ** 0.5)); n_units = side * side
    gx, gy = np.meshgrid(np.linspace(0, 1, side), np.linspace(0, 1, side))
    pos = np.stack([gx.ravel(), gy.ravel()], axis=1)
    if smooth:
        resp = np.zeros((n_stim, n_units))
        for s in range(n_stim):
            freq = rng.randn(2) * 4.0; phase = rng.uniform(0, 2 * np.pi)
            resp[s] = np.sin(pos @ freq + phase) + 0.05 * rng.randn(n_units)
    else:
        resp = rng.randn(n_stim, n_units)          # no spatial structure
    coords = {'stimulus_id': ('presentation', [f's{i}' for i in range(n_stim)]),
              'neuroid_id': ('neuroid', list(range(n_units)))}
    if with_coords:
        coords['tissue_x'] = ('neuroid', pos[:, 0])
        coords['tissue_y'] = ('neuroid', pos[:, 1])
    return NeuroidAssembly(resp, coords=coords, dims=['presentation', 'neuroid'])


class _StubModel(Subject):
    """Participate-only candidate: answers neural:IT with a fixed assembly."""
    identifier = 'participate-only-topographic'
    in_channels = {'text'}
    out_channels = {'neural:IT'}

    def __init__(self, assembly):
        self.assembly = assembly
        self.requests = []

    def participate(self, session):
        self.requests.append(tuple(session.requested_output_channels))
        while session.next_input() is not None:
            pass
        session.respond(StreamEvent('neural:IT', self.assembly, None, {}))


def _stimuli(n_stim=30):
    stimuli = StimulusSet(pd.DataFrame({
        'stimulus_id': [f's{i}' for i in range(n_stim)],
        'sentence': [f'stimulus {i}' for i in range(n_stim)]}))
    stimuli.identifier = 'topographic-demo-stimuli'
    return stimuli


def _benchmark(brain):
    return TopographicBenchmark('topographic-demo', brain_assembly=brain,
                                stimulus_set=_stimuli(), region='IT')


class TestTopographicBenchmark:
    def test_topographic_model_clears_the_null(self):
        brain = _assembly(seed=1)                                  # topographic target
        model = _StubModel(_assembly(seed=2))                      # topographic, with coords
        score = _benchmark(brain)(model)
        assert score.attrs['raw'] > score.attrs['null']            # spatial signal present
        assert float(score) > 0.1                                  # raw - null

    def test_nontopographic_model_near_null(self):
        brain = _assembly(seed=1)
        # no spatial structure + no coords -> benchmark assigns a neutral grid -> flat r(d)
        model = _StubModel(_assembly(seed=3, smooth=False, with_coords=False))
        score = _benchmark(brain)(model)
        assert abs(float(score)) < 0.3                             # ~ at the floor

    def test_topographic_beats_nontopographic(self):
        brain = _assembly(seed=1)
        topo = float(_benchmark(brain)(_StubModel(_assembly(seed=2))))
        flat = float(_benchmark(brain)(_StubModel(
            _assembly(seed=3, smooth=False, with_coords=False))))
        assert topo > flat

    def test_score_carries_raw_and_null(self):
        brain = _assembly(seed=1)
        score = _benchmark(brain)(_StubModel(_assembly(seed=2)))
        assert 'raw' in score.attrs and 'null' in score.attrs and 'ceiling' in score.attrs

    def test_runs_a_participate_only_subject_and_keeps_its_tissue_coords(self):
        from brainscore import load_metric
        brain = _assembly(seed=1)
        model = _StubModel(_assembly(seed=2))
        assert not hasattr(model, 'process') and not hasattr(model, 'start_recording')
        score = _benchmark(brain)(model)
        assert model.requests == [('neural:IT',)]
        # The subject's own positions were scored, not the fallback grid.
        expected = float(load_metric('topographic-alignment')(model.assembly, brain).values)
        assert score.attrs['raw'] == expected


def test_brainscore_model_session_response_keeps_tissue_coords_like_process():
    """BrainScoreModel returns the same assembly, extra neuroid coords included,
    through neural_response as through start_recording + process."""
    from brainscore_core.model_interface import BrainScoreModel
    from brainscore_core.streaming_helpers import neural_response

    template = _assembly(seed=2)

    def extract(stimuli, layers):
        ids = list(stimuli['stimulus_id'])
        values = template.values      # same 30 stimuli, in table order
        return NeuroidAssembly(values, dims=['presentation', 'neuroid'], coords={
            'stimulus_id': ('presentation', ids),
            'neuroid_id': ('neuroid', list(range(values.shape[1]))),
            'layer': ('neuroid', [layers[0]] * values.shape[1]),
            'tissue_x': ('neuroid', template['tissue_x'].values),
            'tissue_y': ('neuroid', template['tissue_y'].values)})

    model = BrainScoreModel('topographic-fixture', region_layer_map={'IT': 'sheet'},
                            extractor={'text': extract})
    model.start_recording('IT')
    direct = model.process(_stimuli())
    model.reset()
    through_session = neural_response(model, _stimuli(), record='IT')
    assert through_session.identical(direct)
    np.testing.assert_array_equal(through_session['tissue_x'].values, template['tissue_x'].values)
