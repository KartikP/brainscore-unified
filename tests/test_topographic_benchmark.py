"""End-to-end wiring test for the topographic-organization axis (offline, synthetic).

Proves the benchmark drives a candidate through process(), obtains unit positions, scores via
the registered topographic-alignment metric, and subtracts the shuffle-coordinate null — and
that a topographic layout clears the null while a non-topographic one does not. A real
registered instance swaps the synthetic brain/candidate for a surface fMRI assembly + a real
topographic model (EC2); the wiring under test is identical.
"""
import numpy as np

from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
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


class _StubModel:
    """Minimal candidate: start_recording is a no-op; process returns a fixed assembly."""
    def __init__(self, assembly):
        self._assembly = assembly
    def start_recording(self, region, **kw):
        pass
    def process(self, stimulus_set, **kw):
        return self._assembly


def _benchmark(brain):
    return TopographicBenchmark('topographic-demo', brain_assembly=brain,
                                stimulus_set=None, region='IT')


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
