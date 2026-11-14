"""score() returns the N/A result for a pair that is not applicable, before any compute."""
import math

import pytest

import brainscore
from brainscore_core.contract import Subject
from brainscore_core.memory import MemoryError
from brainscore_core.metrics import Score

pytestmark = pytest.mark.unit


class _Recorder(Subject):
    """A subject that records every call to participate."""

    identifier = 'recorder'
    in_channels = frozenset()
    out_channels = frozenset()
    required_channels = frozenset()

    def __init__(self, in_channels, out_channels):
        self.in_channels = frozenset(in_channels)
        self.out_channels = frozenset(out_channels)
        self.calls = []

    def participate(self, session):
        self.calls.append(session)

    def reset(self):
        pass


class _Benchmark:
    identifier = 'toy-neural'
    required_input_channels = frozenset({'vision:image'})
    requested_output_channels = frozenset({'neural:IT'})

    def __init__(self):
        self.runs = 0

    def __call__(self, subject):
        self.runs += 1
        return Score(0.25)


def test_incompatible_pair_is_na_and_runs_nothing():
    subject, benchmark = _Recorder({'vision:image'}, {'behavior'}), _Benchmark()
    result = brainscore.score(subject, benchmark, check_mem=False)
    assert math.isnan(float(result))
    assert result.attrs['status'] == 'N/A'
    assert result.attrs['missing_channels'] == ['neural:IT']
    assert "'neural:IT'" in result.attrs['reason']
    assert result.attrs['model_identifier'] == 'recorder'
    assert result.attrs['benchmark_identifier'] == 'toy-neural'
    assert subject.calls == [] and benchmark.runs == 0


def test_na_comes_before_the_memory_check(monkeypatch):
    import brainscore_core.memory as memory
    monkeypatch.setattr(memory, 'check_memory', lambda *a, **k: pytest.fail('memory checked'))
    result = brainscore.score(_Recorder({'text'}, {'neural:IT'}), _Benchmark(), check_mem=True)
    assert result.attrs['status'] == 'N/A'
    assert result.attrs['missing_channels'] == ['vision:image']


def test_memory_failure_still_raises(monkeypatch):
    import brainscore_core.memory as memory

    def refuse(*args, **kwargs):
        raise MemoryError('not enough memory')
    monkeypatch.setattr(memory, 'check_memory', refuse)
    benchmark = _Benchmark()
    with pytest.raises(MemoryError, match='not enough memory'):
        brainscore.score(_Recorder({'vision:image'}, {'neural:IT'}), benchmark, check_mem=True)
    assert benchmark.runs == 0


def test_compatible_pair_score_is_unchanged():
    benchmark = _Benchmark()
    result = brainscore.score(_Recorder({'vision:image'}, {'neural:IT'}), benchmark, check_mem=False)
    assert float(result) == 0.25 and benchmark.runs == 1
    assert 'status' not in result.attrs
    assert result.attrs['compatibility']['match'] == 'full'


# ── N/A only before compute (the run's sessions are watched) ─────────────────

import numpy as np

from brainscore.brainscore_model import BrainScoreModel
from brainscore_core.compatibility import CompatibilityError
from brainscore_core.contract import TaskContext
from brainscore_core.responses import Choice
from brainscore_core.streaming import Session, StreamEvent
from brainscore_core.streaming_helpers import stimulus_session
from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet

STIMULI = StimulusSet({'stimulus_id': ['a', 'b'], 'sentence': ['cat', 'dog']})
STIMULI.identifier = 'not-applicable-toy'
CHOICE = TaskContext(instruction='Cat or dog?', response=Choice(['cat', 'dog']))


class _CountingExtractor:
    def __init__(self):
        self.calls = 0

    def __call__(self, stimuli, layers, **kwargs):
        self.calls += 1
        ids = list(stimuli['stimulus_id'])
        return NeuroidAssembly(np.ones((len(ids), 2)), dims=['presentation', 'neuroid'], coords={
            'stimulus_id': ('presentation', ids), 'neuroid_id': ('neuroid', ['f0', 'f1']),
            'layer': ('neuroid', [layers[0]] * 2)})


def _recording_model():
    """Records IT passively; has no part that answers a task."""
    extractor = _CountingExtractor()
    return BrainScoreModel(identifier='toy-recorder', extractor={'text': extractor},
                           region_layer_map={'IT': 'late'}), extractor


class _TextBenchmark:
    identifier = 'toy-text'
    required_input_channels = frozenset({'text'})
    requested_output_channels = frozenset({'neural:IT'})
    stimulus_set = STIMULI

    def __init__(self, run):
        self.run = run

    def __call__(self, subject):
        return self.run(subject)


def test_a_task_no_part_answers_is_na_with_zero_compute_without_the_memory_probe():
    model, extractor = _recording_model()
    benchmark = _TextBenchmark(lambda s: s.participate(stimulus_session(STIMULI, ['neural:IT'], CHOICE)))
    result = brainscore.score(model, benchmark, check_mem=False)
    assert result.attrs['status'] == 'N/A' and 'Choice' in result.attrs['reason']
    assert extractor.calls == 0


def test_with_the_memory_probe_only_the_probe_computes_before_the_na():
    # The residual: the task is known only inside the benchmark, so the probe
    # (one stimulus) runs before the first session refuses. The result is N/A.
    model, extractor = _recording_model()
    benchmark = _TextBenchmark(lambda s: s.participate(stimulus_session(STIMULI, ['neural:IT'], CHOICE)))
    result = brainscore.score(model, benchmark, check_mem=True)
    assert result.attrs['status'] == 'N/A'
    assert extractor.calls == 1          # the memory probe, one stimulus


def test_a_later_session_that_refuses_is_an_error():
    model, extractor = _recording_model()

    def run(subject):
        subject.participate(stimulus_session(STIMULI, ['neural:IT']))
        subject.participate(stimulus_session(STIMULI, ['neural:IT'], CHOICE))
    with pytest.raises(RuntimeError, match=r"after the run computed.*\['neural:IT'\]") as caught:
        brainscore.score(model, _TextBenchmark(run), check_mem=False)
    assert isinstance(caught.value.__cause__, CompatibilityError)
    assert extractor.calls == 1


def test_a_compatibility_error_from_the_metric_after_responses_is_an_error():
    model, _ = _recording_model()

    def run(subject):
        subject.participate(stimulus_session(STIMULI, ['neural:IT']))
        raise CompatibilityError('metric cannot compare these responses')
    with pytest.raises(RuntimeError, match='after the run computed'):
        brainscore.score(model, _TextBenchmark(run), check_mem=False)


class _Lesionable(Subject):
    identifier = 'lesionable'
    in_channels = frozenset({'lesion'})
    out_channels = frozenset({'perturbation'})
    required_channels = frozenset()

    def participate(self, session):
        from brainscore_core.events import PerturbationApplied
        while (event := session.next_input()) is not None:
            change = event.payload
            session.respond(StreamEvent('perturbation', PerturbationApplied(
                handle_id='h1', target=change.target, perturbation=change.perturbation), event.t_ms))

    def reset(self):
        pass


def test_an_intervention_left_active_before_a_refusal_is_an_error():
    from brainscore_core.events import Perturbation, Selection, StateChange

    class _Benchmark:
        identifier = 'toy-lesion'
        required_input_channels = frozenset({'lesion'})
        requested_output_channels = frozenset({'perturbation'})

        def __call__(self, subject):
            change = StateChange(kind='lesion', target=Selection(layer='late'),
                                 perturbation=Perturbation(kind='zero'))
            subject.participate(Session([StreamEvent('lesion', change, 0.0)], ['perturbation']))
            raise CompatibilityError('cannot answer the next session')
    with pytest.raises(RuntimeError, match=r"\['h1'\] are still active"):
        brainscore.score(_Lesionable(), _Benchmark(), check_mem=False)


def test_a_runtime_error_passes_through():
    def crash(subject):
        raise RuntimeError('model crashed')
    with pytest.raises(RuntimeError, match='model crashed'):
        brainscore.score(_recording_model()[0], _TextBenchmark(crash), check_mem=False)


# ── A malformed declaration is a defect, never N/A ───────────────────────────

@pytest.mark.parametrize('fields', [
    {'in_channels': frozenset({'text'})},                                   # a field benchmarks do not read
    {'required_input_channels': frozenset()},                               # no inputs
    {'required_input_channels': frozenset({'neural:IT'})},                  # an output as an input
    {'required_input_channels': frozenset({'vision'})},                     # names no visual form
    {'required_input_channels': frozenset({'text:'})},                      # bad syntax
], ids=['misspelled', 'empty', 'direction', 'form', 'syntax'])
def test_a_malformed_benchmark_declaration_raises(fields):
    from types import SimpleNamespace
    benchmark = SimpleNamespace(identifier='malformed', requested_output_channels=frozenset({'neural:IT'}),
                                **fields)
    subject = _Recorder({'text'}, {'neural:IT'})
    with pytest.raises(ValueError):
        brainscore.score(subject, benchmark, check_mem=False)
    assert subject.calls == []


def test_an_inconsistent_output_inputs_raises():
    class _Listing(_Recorder):
        output_inputs = {'neural:IT': frozenset({'audio'})}    # an input it does not declare
    subject = _Listing({'vision:image'}, {'neural:IT'})
    with pytest.raises(ValueError, match='output_inputs'):
        brainscore.score(subject, _Benchmark(), check_mem=False)


# ── Support refusals of the legacy paths are N/A through brainscore.score ────

def test_a_neural_only_subject_on_a_legacy_behavioral_task_is_na():
    from brainscore_core.legacy_facade import mark_legacy_benchmark
    subject = _Recorder({'text'}, {'neural:language_network'})

    class _Legacy:
        identifier = 'legacy-reading-times'
        required_modalities = {'text'}

        def __call__(self, candidate):
            candidate.start_behavioral_task('reading_times')
            candidate.digest_text(['the', 'cat'])
    result = brainscore.score(subject, mark_legacy_benchmark(_Legacy()), check_mem=False)
    assert result.attrs['status'] == 'N/A' and result.attrs['missing_channels'] == ['behavior']
    assert subject.calls == []


def test_a_legacy_vision_model_asked_for_behavior_and_a_region_together_is_na():
    from unittest.mock import MagicMock, PropertyMock
    from brainscore_vision.compat.unified_adapter import VisionModelAdapter
    legacy = MagicMock()
    type(legacy).identifier = PropertyMock(return_value='mock-legacy')
    legacy.visual_degrees.return_value = 8
    legacy.layer_model.region_layer_map = {'IT': 'layer4'}
    images = StimulusSet({'stimulus_id': ['s0'], 'image_path': ['s0.png']})
    images.identifier = 'legacy-images'

    class _Joint:
        identifier = 'joint'
        required_input_channels = frozenset({'vision:image'})
        requested_output_channels = frozenset({'behavior', 'neural:IT'})

        def __call__(self, subject):
            subject.participate(stimulus_session(images, ['behavior', 'neural:IT']))
    result = brainscore.score(VisionModelAdapter(legacy), _Joint(), check_mem=False)
    assert result.attrs['status'] == 'N/A'
    assert result.attrs['missing_channels'] == ['behavior', 'neural:IT']
    legacy.look_at.assert_not_called()
