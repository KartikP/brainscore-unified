"""Registration delegates evaluation and preserves failures and raw scores."""
import json
import math
from collections import Counter
from types import SimpleNamespace

import pytest

import brainscore
from brainscore.benchmarks.coggym import EXPERIMENTS
from brainscore.benchmarks.coggym.benchmark import CogGymBenchmark
from brainscore.experiments import Experiment, RecordInputsOutputs
from brainscore.harnesses import coggym
from brainscore_core.compatibility import CompatibilityError
from brainscore_core.contract import Subject

from .test_coggym import image_part, reference, make_subject, reply_with_trial

pytestmark = pytest.mark.unit


@pytest.fixture
def scored_reference(reference, monkeypatch):
    root, cli = reference
    _, analysis = coggym._load_reference(root)
    def analyze(paths, *args, **kwargs):
        runs = [json.loads(path.read_text()) for path in paths]
        row = {'model': runs[0]['model'], 'experiment': runs[0]['experiment'],
               'r2': 0.375, 'n_items': 3, 'n_artifacts': len(runs)}
        return [row], [], []
    analysis.analyze = analyze
    return root, analysis


@pytest.mark.parametrize('invalid', [False, True])
def test_slider_validation_preserves_native_results_and_gates_score(scored_reference, tmp_path, invalid):
    root, _ = scored_reference
    cli, _ = coggym._load_reference(root)
    trials = cli._selected_trials(None, None, False)
    for trial in trials:
        trial.queries = [SimpleNamespace(
            tag='rating', type='single-slider', slider_config={'min': 0, 'max': 100}, option=[],
        )]
    native = [{
        'trial_id': trial.id,
        'response': '{"answer": <number between 0 and 100>}' if invalid and i == 0 else '{"answer": 42}',
        'parsed': {'rating': {'value': 0 if invalid and i == 0 else 42}},
        'scores': {'rating': {'scorable': True}},
    } for i, trial in enumerate(trials)]
    cli._run_experiment = lambda *args: native
    runner = coggym.CogGymRunner(root, model='fixture', repetitions=1)
    direct = runner.run(
        SimpleNamespace(complete_with_metadata=lambda *args: None),
        output_dir=tmp_path / 'direct', reset=lambda repetition: None,
    )
    assert direct['experiments'][0]['r2'] == 0.375
    validation = direct['response_validation']
    assert validation['status'] == ('invalid' if invalid else 'valid')
    assert validation['checked_queries'] == 3
    assert validation['invalid_queries'] == int(invalid)
    saved = json.loads((tmp_path / 'direct/run-001.json').read_text())
    assert saved['trials'] == native
    assert saved['umi']['response_validation'] == validation['repetitions'][0]
    benchmark = brainscore.load_benchmark(
        'CogGym.Hu2023Fine.exp1', checkout=root, repetitions=1,
        reset=lambda repetition: None, output_dir=tmp_path / 'registered',
    )
    assert benchmark.version == 1
    if invalid:
        with pytest.raises(ValueError, match='slider validation rejected 1 answers'):
            benchmark(make_subject(reply_with_trial))
        assert json.loads((tmp_path / 'registered/experiment.json').read_text())['status'] == 'failed'
    else:
        score = benchmark(make_subject(reply_with_trial))
        assert float(score) == 0.375
        assert score.attrs['response_validation'] == validation
    assert json.loads((tmp_path / 'registered/coggym/run-001.json').read_text())['trials'] == native


def test_all_public_experiments_registered_lazily():
    ids = {name for name in brainscore.benchmark_registry if name.startswith('CogGym.')}
    assert ids == {f'CogGym.{path.replace("/", ".")}' for path in EXPERIMENTS}
    assert Counter(EXPERIMENTS.values()) == {'text': 11, 'image': 6, 'video': 6}


def test_loading_requires_explicit_checkout_before_model_use(monkeypatch):
    monkeypatch.delenv('BRAINSCORE_COGGYM_CHECKOUT', raising=False)
    with pytest.raises(FileNotFoundError, match='BRAINSCORE_COGGYM_CHECKOUT'):
        brainscore.load_benchmark('CogGym.Hu2023Fine.exp1')


def test_environment_checkout_and_explicit_provider_reset(scored_reference, monkeypatch):
    root, _ = scored_reference
    monkeypatch.setenv('BRAINSCORE_COGGYM_CHECKOUT', str(root))
    benchmark = brainscore.load_benchmark('CogGym.Hu2023Fine.exp1')
    with pytest.raises(ValueError, match='reset=provider.reset'):
        benchmark(make_subject(reply_with_trial))
    assert math.isnan(float(benchmark.ceiling))


def test_registered_scoring_preserves_raw_score_and_repetitions(scored_reference, tmp_path, monkeypatch):
    root, _ = scored_reference
    resets, calls = [], []
    def capture(prompt):
        calls.append(prompt)
        return reply_with_trial(prompt)
    benchmark = brainscore.load_benchmark(
        'CogGym.Hu2023Fine.exp1', checkout=root, reset=resets.append,
        repetitions=2, max_tokens=1234, output_dir=tmp_path / 'scored',
    )
    monkeypatch.setenv('RESULTCACHING_HOME', str(tmp_path / 'cache'))
    score = brainscore.score(make_subject(capture), benchmark, check_mem=False)
    assert float(score) == float(score.attrs['raw']) == 0.375
    assert score.attrs['normalized'] is False
    assert score.attrs['ceiling'].attrs['status'] == 'not_estimated'
    assert score.attrs['metric'] == 'pearson_r_squared'
    assert score.attrs['benchmark_version'] == 1
    assert score.attrs['compatibility']['match'] == 'full'
    assert score.attrs['configuration']['repetitions'] == 2
    assert resets == [1, 2]
    assert len(calls) == 6
    # The benchmark's settings reach the reply on the task's Text response.
    assert all(call.response.max_tokens == 1234 for call in calls)
    assert all(call.response.temperature == 1.0 for call in calls)
    assert score.attrs['input_adaptation']['kind'] == 'subject_prompt'
    assert 'system text' in score.attrs['input_adaptation']['reason']
    assert score.attrs['benchmark_identifier'] == benchmark.identifier
    assert '.repeats-2.tokens-1234.temperature-1.0' in benchmark.identifier
    assert (tmp_path / 'scored/coggym/analysis.json').is_file()


def test_protocol_and_plain_runner_produce_identical_reference_artifacts(scored_reference, tmp_path):
    root, _ = scored_reference
    benchmark = brainscore.load_benchmark('CogGym.Hu2023Fine.exp1', checkout=root)
    direct = coggym.CogGymRunner(root, experiment='Hu2023Fine/exp1', model='fixture', max_tokens=512, repetitions=5)
    from types import SimpleNamespace
    def complete(system, messages, model, temperature, max_tokens):
        return {'text': messages[0]['content'], 'reasoning': None, 'token_usage': None}
    direct.run(SimpleNamespace(complete_with_metadata=complete),
               output_dir=tmp_path / 'direct', reset=lambda repetition: None)
    result = Experiment(
        subject=make_subject(reply_with_trial),
        protocol=benchmark.protocol(model='fixture', reset=lambda repetition: None),
        tools=[RecordInputsOutputs()], output_dir=tmp_path / 'registered',
    ).run()
    plain = json.loads((tmp_path / 'direct/run-001.json').read_text())
    wrapped = json.loads((result.directory / 'coggym/run-001.json').read_text())
    assert wrapped == plain
    assert float(result.value) == 0.375
    assert sum(event['kind'] == 'input' and event['payload'].channel == 'text'
               for event in result.record.events()) == 15


@pytest.mark.parametrize('value', [None, float('nan'), float('inf')])
def test_undefined_score_fails_and_preserves_results(scored_reference, tmp_path, value):
    root, analysis = scored_reference
    # None is serialized upstream; nonfinite values also must never become scores.
    analysis.analyze = lambda *args, **kwargs: ([{
        'model': 'fixture', 'experiment': 'Hu2023Fine/exp1', 'r2': value,
    }], [], [])
    benchmark = brainscore.load_benchmark('CogGym.Hu2023Fine.exp1', checkout=root,
        reset=lambda repetition: None, output_dir=tmp_path / 'run')
    with pytest.raises(ValueError):
        benchmark(make_subject(reply_with_trial))
    assert (tmp_path / 'run/coggym/run-001.json').is_file()
    assert json.loads((tmp_path / 'run/experiment.json').read_text())['status'] == 'failed'


def test_feature_only_model_rejected_before_requests(scored_reference):
    root, _ = scored_reference
    benchmark = brainscore.load_benchmark('CogGym.Hu2023Fine.exp1', checkout=root,
        reset=lambda repetition: None)
    # score() runs the one pre-flight; the benchmark checks nothing itself.
    with pytest.raises(CompatibilityError):
        brainscore.score(brainscore.load_model('chance-baseline'), benchmark, check_mem=False)


def test_configuration_is_not_discarded_on_domain_fallback():
    with pytest.raises(TypeError, match='unified registry'):
        brainscore.load_benchmark('some-domain-benchmark', repetitions=2)


def test_video_registration_preserves_still_image_instructions(reference):
    root, cli = reference
    (root / 'evaluation/public_manifest.json').write_text(json.dumps({
        'experiments': [{'path': 'Fu2025Hierarchical/exp1'}],
    }))
    cli.build_messages = lambda trial, experiment: ('system', [{
        'role': 'user', 'content': [{'type': 'text', 'text': 'instructions'},
                                 image_part(), {'type': 'video', 'path': '/clip.mp4'}],
    }])
    benchmark = brainscore.load_benchmark('CogGym.Fu2025Hierarchical.exp1', checkout=root)
    protocol = benchmark.protocol(model='fixture', reset=lambda repetition: None)
    assert set(protocol.metadata['modalities']) == {'text', 'image', 'video'}
    # The still images and the clips are both required inputs.
    assert benchmark.required_input_channels == {'instruction', 'text', 'vision:image', 'vision:video'}


class _ParticipateOnlySubject(Subject):
    """A contract subject: declarations, participate and reset; no process()."""

    identifier = "participate-only"
    in_channels = frozenset({"instruction", "text"})
    out_channels = frozenset({"behavior:text"})

    def __init__(self):
        self.requests = []

    def participate(self, session):
        from brainscore_core.streaming import StreamEvent
        while (event := session.next_input()) is not None:
            self.requests.append(event.payload)
            session.respond(StreamEvent('behavior:text', event.payload, event.t_ms, dict(event.meta)))

    def reset(self):
        pass


def test_participate_only_subject_is_scored(scored_reference, tmp_path):
    root, _ = scored_reference
    subject = _ParticipateOnlySubject()
    benchmark = brainscore.load_benchmark(
        'CogGym.Hu2023Fine.exp1', checkout=root, reset=lambda repetition: None,
        output_dir=tmp_path / 'run')
    score = benchmark(subject)
    assert float(score) == 0.375
    assert len(subject.requests) == 15
    events = _record_events(tmp_path / "run")
    inputs = [event['payload'] for event in events
              if event['kind'] == 'input' and event['payload'].channel == 'text']
    outputs = [event['payload'] for event in events if event['kind'] == 'output']
    assert len(inputs) == len(outputs) == 15
    assert all(output.channel == 'behavior:text' for output in outputs)


def test_reset_is_required_before_any_request(scored_reference):
    root, _ = scored_reference
    subject = _ParticipateOnlySubject()
    benchmark = brainscore.load_benchmark('CogGym.Hu2023Fine.exp1', checkout=root)
    with pytest.raises(ValueError, match='reset=provider.reset'):
        benchmark(subject)
    assert subject.requests == []


def _record_events(directory):
    from brainscore.experiments import read_events
    return read_events(directory)


def test_declares_the_instruction_and_text_in_and_a_written_answer_out(scored_reference):
    from tests.benchmark_declarations import assert_declares, assert_preflight
    root, _ = scored_reference
    benchmark = brainscore.load_benchmark('CogGym.Hu2023Fine.exp1', checkout=root,
                                          reset=lambda repetition: None)
    assert_declares(benchmark, {'instruction', 'text'}, {'behavior:text'})
    assert_preflight(benchmark)
