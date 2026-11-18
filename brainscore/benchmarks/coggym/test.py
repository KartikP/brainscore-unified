"""Registration delegates evaluation and preserves failures and raw scores."""
import json
import math
from types import SimpleNamespace

import pytest

import brainscore
from brainscore.benchmarks.coggym import EXPERIMENTS
from brainscore.benchmarks.coggym.benchmark import CogGymBenchmark
from brainscore.experiments import Experiment, RecordInputsOutputs
from brainscore.harnesses import coggym
from brainscore_core.compatibility import CompatibilityError
from brainscore_core.contract import Subject

from brainscore import LoadError

from tests.coggym_stand_in import ANALYZE_ROW, default_place, write_ready_folder
from tests.test_harnesses.test_coggym import image_part, reference, make_subject, reply_with_trial

pytestmark = pytest.mark.unit


@pytest.fixture
def scored_reference(reference, monkeypatch):
    root, stand_in = reference
    stand_in.analyze(ANALYZE_ROW)
    return root, stand_in


_SLIDER = "SimpleNamespace(tag='rating', type='single-slider', slider_config={'min': 0, 'max': 100}, option=[])"
_CHOICE = "SimpleNamespace(tag='pick', type='multi-choice', slider_config=None, option=['A', 'B'])"


def _native(i, state, query=_SLIDER):
    """CogGym's result entry for trial ``i`` with an answer ``valid``, ``invalid`` or ``missing``."""
    if query == _CHOICE:
        index = -1 if state == 'missing' else 0
        return {'trial_id': f'trial-{i}', 'response': 'A' if index == 0 else 'neither',
                'parsed': {'pick': {'selected_index': index}},
                'scores': {'pick': {'scorable': index == 0}}}
    response, value = {'invalid': ('{"answer": <number between 0 and 100>}', 0),   # read from a placeholder
                       'missing': ('I cannot say.', None)}.get(state, ('{"answer": 42}', 42))
    return {'trial_id': f'trial-{i}', 'response': response, 'parsed': {'rating': {'value': value}},
            'scores': {'rating': {'scorable': value is not None}}}


def _runs(stand_in, *runs, query=_SLIDER, trials=10):
    """CogGym gives one result list for each repetition: ``runs`` are sets of (index, state)."""
    native = [[_native(i, dict(run).get(i, 'valid'), query) for i in range(trials)] for run in runs]
    stand_in.cli(f'''
TRIALS = [f'trial-{{i}}' for i in range({trials})]


def _selected_trials(*args, **kwargs):
    return [SimpleNamespace(id=trial, queries=[{query}]) for trial in TRIALS]


def _run_experiment(*args, **kwargs):
    RUNS.append(1)
    return {native!r}[len(RUNS) - 1]
''')
    return native


def _slider_run(stand_in, invalid=(), missing=(), trials=10):
    """One repetition of ``trials`` slider trials: answers by index invalid, missing or valid."""
    run = {(i, 'invalid') for i in invalid} | {(i, 'missing') for i in missing}
    return _runs(stand_in, run, trials=trials)[0]


def _registered(root, tmp_path, repetitions=1):
    return brainscore.load_benchmark('CogGym.Hu2023Fine.exp1', folder=root, repetitions=repetitions,
                                     reset=lambda repetition: None, output_dir=tmp_path / 'registered')


def test_slider_check_leaves_native_results_unchanged(scored_reference, tmp_path):
    root, stand_in = scored_reference
    native = _slider_run(stand_in, invalid={0}, missing={1})
    direct = coggym.CogGymRunner(root, model='fixture', repetitions=1).run(
        SimpleNamespace(complete_with_metadata=lambda *args: None),
        output_dir=tmp_path / 'direct', reset=lambda repetition: None,
    )
    assert direct['experiments'][0]['r2'] == 0.375
    saved = json.loads((tmp_path / 'direct/run-001.json').read_text())
    assert saved['trials'] == native
    assert saved['umi']['response_validation'] == direct['response_validation']['repetitions'][0]


def test_one_invalid_answer_of_many_gives_a_score_with_the_counts(scored_reference, tmp_path):
    root, stand_in = scored_reference
    _slider_run(stand_in, invalid={0})
    score = _registered(root, tmp_path)(make_subject(reply_with_trial))
    assert float(score) == 0.375                          # CogGym's own number
    assert (score.attrs['n_valid'], score.attrs['n_invalid'], score.attrs['n_missing']) == (9, 1, 0)
    assert score.attrs['result_validity'] == 'valid'      # 1 of 10 is not above 10 percent
    assert 'coggym-slider-v1' in score.attrs['answer_policy']


def test_a_missing_answer_is_counted_missing(scored_reference, tmp_path):
    root, stand_in = scored_reference
    _slider_run(stand_in, missing={3})
    score = _registered(root, tmp_path)(make_subject(reply_with_trial))
    assert (score.attrs['n_invalid'], score.attrs['n_missing']) == (0, 1)
    assert score.attrs['result_validity'] == 'valid'


@pytest.mark.parametrize('r2', ['0.375', 'None'])
def test_a_share_above_the_limit_is_returned_marked_invalid(scored_reference, tmp_path, r2):
    root, stand_in = scored_reference
    _slider_run(stand_in, invalid={0}, missing={1}, trials=5)
    stand_in.analyze(f'''
def analyze(*args, **kwargs):
    return [{{'model': 'fixture', 'experiment': 'Hu2023Fine/exp1', 'r2': {r2}}}], [], []
''')
    score = _registered(root, tmp_path)(make_subject(reply_with_trial))   # raises nothing
    assert score.attrs['result_validity'] == 'invalid'
    assert score.attrs['result_validity_reason']
    assert (score.attrs['n_invalid'], score.attrs['n_missing']) == (1, 1)
    if r2 == 'None':      # no fit: the result is still returned, marked invalid
        assert math.isnan(float(score))
    else:
        assert float(score) == 0.375
    assert json.loads((tmp_path / 'registered/experiment.json').read_text())['status'] != 'failed'


def test_a_result_for_another_experiment_is_a_fault_of_the_run(scored_reference, tmp_path):
    root, stand_in = scored_reference
    stand_in.analyze('''
def analyze(*args, **kwargs):
    return [{'model': 'fixture', 'experiment': 'Hu2023Fine/exp2', 'r2': 0.5}], [], []
''')
    with pytest.raises(RuntimeError, match='requested model and experiment'):
        _registered(root, tmp_path)(make_subject(reply_with_trial))


def test_public_experiments_are_registered_lazily():
    ids = {name for name in brainscore.benchmark_registry if name.startswith('CogGym.')}
    assert ids == {f'CogGym.{path.replace("/", ".")}' for path in EXPERIMENTS}
    # Representative registrations: a text slider experiment and a video choice experiment.
    assert EXPERIMENTS['Hu2023Fine/exp1'] == 'text'
    assert EXPERIMENTS['Fu2025Hierarchical/exp2'] == 'video'


def test_with_installs_off_a_missing_folder_stops_the_load_before_model_use(monkeypatch, tmp_path):
    # No folder at the place of CogGym's folders, and installs are off: one clear error.
    monkeypatch.setenv('BRAINSCORE_ENVIRONMENTS_HOME', str(tmp_path / 'none'))
    monkeypatch.setenv('BS_INSTALL_DEPENDENCIES', 'no')
    with pytest.raises(LoadError, match='BS_INSTALL_DEPENDENCIES'):
        brainscore.load_benchmark('CogGym.Hu2023Fine.exp1')
    with pytest.raises(LoadError, match='BS_INSTALL_DEPENDENCIES'):     # the named score path too
        brainscore.score(make_subject(reply_with_trial), 'CogGym.Hu2023Fine.exp1', check_mem=False)
    assert not (tmp_path / 'none').exists()


def test_the_default_folder_and_explicit_provider_reset(reference, tmp_path, monkeypatch):
    # A ready folder at the default place under BRAINSCORE_ENVIRONMENTS_HOME: no folder= needed.
    monkeypatch.setenv('BRAINSCORE_ENVIRONMENTS_HOME', str(tmp_path / 'folders'))
    write_ready_folder(default_place(tmp_path / 'folders'), analyze=ANALYZE_ROW)
    benchmark = brainscore.load_benchmark('CogGym.Hu2023Fine.exp1')
    with pytest.raises(ValueError, match='reset=adapter.reset'):
        benchmark(make_subject(reply_with_trial))
    assert math.isnan(float(benchmark.ceiling))


def test_registered_scoring_preserves_raw_score_and_repetitions(scored_reference, tmp_path, monkeypatch):
    root, _ = scored_reference
    resets, calls = [], []
    def capture(prompt):
        calls.append(prompt)
        return reply_with_trial(prompt)
    benchmark = brainscore.load_benchmark(
        'CogGym.Hu2023Fine.exp1', folder=root, reset=resets.append,
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
    # The stated policy and its limit; the stand-in's trials ask no slider question.
    assert score.attrs['result_validity'] == 'valid'
    assert "CogGym's own fit" in score.attrs['answer_policy']
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
    benchmark = brainscore.load_benchmark('CogGym.Hu2023Fine.exp1', folder=root)
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


@pytest.mark.parametrize('value', [float('nan'), float('inf')])
def test_a_nonfinite_fit_fails_and_preserves_results(scored_reference, tmp_path, value):
    root, stand_in = scored_reference
    # CogGym writes no fit as None; a nonfinite value must never become a score.
    stand_in.analyze(f'''
def analyze(*args, **kwargs):
    return [{{'model': 'fixture', 'experiment': 'Hu2023Fine/exp1', 'r2': {value if value is None else f"float('{value}')"}}}], [], []
''')
    benchmark = brainscore.load_benchmark('CogGym.Hu2023Fine.exp1', folder=root,
        reset=lambda repetition: None, output_dir=tmp_path / 'run')
    with pytest.raises(ValueError):
        benchmark(make_subject(reply_with_trial))
    assert (tmp_path / 'run/coggym/run-001.json').is_file()
    assert json.loads((tmp_path / 'run/experiment.json').read_text())['status'] == 'failed'


def test_feature_only_model_rejected_before_requests(scored_reference):
    root, _ = scored_reference
    benchmark = brainscore.load_benchmark('CogGym.Hu2023Fine.exp1', folder=root,
        reset=lambda repetition: None)
    # score() runs the one pre-flight; the benchmark checks nothing itself.
    result = brainscore.score(brainscore.load_model('chance-baseline'), benchmark, check_mem=False)
    assert result.attrs['status'] == 'N/A'


def test_configuration_is_not_discarded_on_domain_fallback():
    with pytest.raises(TypeError, match='unified registry'):
        brainscore.load_benchmark('some-domain-benchmark', repetitions=2)


def test_video_registration_preserves_still_image_instructions(reference):
    root, stand_in = reference
    (root / 'checkout/evaluation/public_manifest.json').write_text(json.dumps({
        'experiments': [{'path': 'Fu2025Hierarchical/exp1'}],
    }))
    content = [{'type': 'text', 'text': 'instructions'}, image_part(), {'type': 'video', 'path': '/clip.mp4'}]
    stand_in.cli(f'''
def build_messages(trial, experiment):
    return 'system', [{{'role': 'user', 'content': {content!r}}}]
''')
    benchmark = brainscore.load_benchmark('CogGym.Fu2025Hierarchical.exp1', folder=root)
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
        'CogGym.Hu2023Fine.exp1', folder=root, reset=lambda repetition: None,
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
    benchmark = brainscore.load_benchmark('CogGym.Hu2023Fine.exp1', folder=root)
    with pytest.raises(ValueError, match='reset=adapter.reset'):
        benchmark(subject)
    assert subject.requests == []


def _record_events(directory):
    from brainscore.experiments import read_events
    return read_events(directory)


def test_declares_the_instruction_and_text_in_and_a_written_answer_out(scored_reference):
    from tests.benchmark_declarations import assert_declares, assert_preflight
    root, _ = scored_reference
    benchmark = brainscore.load_benchmark('CogGym.Hu2023Fine.exp1', folder=root,
                                          reset=lambda repetition: None)
    assert_declares(benchmark, {'instruction', 'text'}, {'behavior:text'})
    assert_preflight(benchmark)


class _NoRawAnswer(Subject):
    """M9-01, the reviewer's case: written answer '42' with no raw answer. Core stores the
    trial as missing (a None raw answer is no answer), although answer_valid is True."""
    identifier = 'no-raw-answer'
    in_channels = {'text', 'instruction', 'vision:image'}
    out_channels = {'behavior:text'}
    required_channels = set()

    def participate(self, session):
        import numpy as np
        from brainscore_core.response_formats import response_record
        from brainscore_core.streaming import StreamEvent
        from brainscore_core.supported_data_standards.brainio.assemblies import BehavioralAssembly
        ids = []
        while (event := session.next_input()) is not None:
            if event.meta.get('stimulus_id') is not None and event.meta['stimulus_id'] not in ids:
                ids.append(event.meta['stimulus_id'])
        session.respond(StreamEvent('behavior:text', BehavioralAssembly(
            np.array(['42'] * len(ids), dtype=object), dims=['presentation'], coords={
                'stimulus_id': ('presentation', ids), 'raw_answer': ('presentation', [None] * len(ids)),
                'answer_valid': ('presentation', [True] * len(ids))},
            attrs={'response': response_record('behavior:text', 'test')}), None))

    def reset(self):
        pass


def test_a_text_answer_stored_as_missing_does_not_stop_the_run(scored_reference, tmp_path):
    """Contract Rule 6.2: a missing written answer reaches CogGym as a call with no reply."""
    root, _ = scored_reference
    benchmark = brainscore.load_benchmark('CogGym.Hu2023Fine.exp1', folder=root,
                                          reset=lambda repetition: None, repetitions=2,
                                          output_dir=tmp_path / 'scored')
    score = brainscore.score(_NoRawAnswer(), benchmark, check_mem=False)
    assert float(score) == 0.375
    run = json.loads((tmp_path / 'scored' / 'coggym' / 'run-001.json').read_text())
    assert all('error' in trial and 'response' not in trial for trial in run['trials'])


def test_an_evaluator_that_cannot_import_a_module_is_one_clear_load_error(reference):
    root, _ = reference
    (root / 'checkout/evaluation/providers.py').write_text('import a_package_the_folder_lacks\n')
    with pytest.raises(LoadError) as refused:
        brainscore.load_benchmark('CogGym.Hu2023Fine.exp1', folder=root)
    message = str(refused.value)
    assert 'CogGym.Hu2023Fine.exp1' in message and 'a_package_the_folder_lacks' in message
    assert str(root) in message


@pytest.mark.parametrize('marker', ['[]', '{"revision": "a1cd9df1118fec80eba7463de237d1497d77e041"}', 'not json'])
def test_a_folder_with_an_incomplete_readiness_record_is_one_clear_load_error(reference, marker):
    root, _ = reference
    if marker.startswith('{'):           # the right commit and requirements, but no package list
        record = json.loads((root / 'ready.json').read_text())
        del record['freeze']
        marker = json.dumps(record)
    (root / 'ready.json').write_text(marker)
    with pytest.raises(LoadError, match='not a ready CogGym folder'):
        brainscore.load_benchmark('CogGym.Hu2023Fine.exp1', folder=root)


class _Answers(Subject):
    """Reads every input; answers every trial except those in ``omit`` (no answer event)."""

    identifier = 'some-answers'
    in_channels = frozenset({'instruction', 'text'})
    out_channels = frozenset({'behavior:text'})

    def __init__(self, omit=lambda trial: True):
        self.omit = omit

    def participate(self, session):
        from brainscore_core.streaming import StreamEvent
        while (event := session.next_input()) is not None:
            if event.channel == 'text' and not self.omit(event.payload):
                session.respond(StreamEvent('behavior:text', '{"answer": 42}', event.t_ms, dict(event.meta)))

    def reset(self):
        pass


def _no_row(stand_in):
    stand_in.analyze("def analyze(*args, **kwargs):\n    return [], [], []\n")


def _trial_queries(stand_in, scorable=True):
    """The stand-in's three trials ask one slider question; CogGym reads the JSON answer.
    ``scorable=False``: CogGym has no human data for the question, so no answer is scorable."""
    stand_in.cli(f"""
def _selected_trials(*args, **kwargs):
    return [SimpleNamespace(id=trial, queries=[{_SLIDER}]) for trial in TRIALS]


def _run_experiment(experiment, selected, provider, *args, **kwargs):
    import json
    results = []
    for trial in selected:
        try:
            raw = provider.complete_with_metadata(*build_messages(trial, experiment), *args)
        except Exception as error:      # CogGym's entry of a failed call
            results.append({{'trial_id': trial.id, 'error': str(error), 'model': args[0]}})
            continue
        value = json.loads(raw['text'])['answer']
        results.append({{'trial_id': trial.id, 'response': raw['text'],
                         'parsed': {{'rating': {{'value': value}}}}, 'scores': {{'rating': {{'scorable': {scorable}}}}}}})
    return results
""")


def _saved_score(tmp_path):
    from brainscore.experiments.runner import ExperimentResult
    return ExperimentResult.open(tmp_path / 'registered').value


def test_all_answers_missing_and_no_row_is_a_result_marked_invalid(scored_reference, tmp_path):
    root, stand_in = scored_reference
    _trial_queries(stand_in)
    _no_row(stand_in)          # CogGym makes no row when no answer is scorable
    score = _registered(root, tmp_path)(_Answers())
    assert math.isnan(float(score))
    assert (score.attrs['n_trials'], score.attrs['n_missing']) == (3, 3)
    assert score.attrs['result_validity'] == 'invalid'
    assert 'no fit' in score.attrs['result_validity_reason']
    # Each answer keeps core's reason, as in core's stored row in the saved record.
    run = json.loads((tmp_path / 'registered/coggym/run-001.json').read_text())
    reasons = [row['response_reason'] for row in run['umi']['core_answers']]
    assert len(reasons) == 3 and all(row['response_state'] == 'missing' for row in run['umi']['core_answers'])
    assert score.attrs['response_validation']['reason_counts'] == {reasons[0]: 3}
    saved = _saved_score(tmp_path)
    assert (saved.attrs['n_trials'], saved.attrs['n_missing']) == (3, 3)
    assert saved.attrs['result_validity'] == 'invalid'
    assert saved.attrs['result_validity_reason'] == score.attrs['result_validity_reason']


def test_an_absent_row_that_the_counts_do_not_explain_is_a_fault_of_the_run(scored_reference, tmp_path):
    root, stand_in = scored_reference
    _trial_queries(stand_in)
    _no_row(stand_in)
    with pytest.raises(RuntimeError, match='scorable'):
        _registered(root, tmp_path)(_Answers(omit=lambda trial: False))


def test_valid_answers_with_no_human_data_and_no_row_are_a_result_marked_invalid(scored_reference, tmp_path):
    """Zero scorable queries: CogGym makes no row, and that is the experiment's result."""
    root, stand_in = scored_reference
    _trial_queries(stand_in, scorable=False)
    _no_row(stand_in)
    score = _registered(root, tmp_path)(_Answers(omit=lambda trial: False))
    assert math.isnan(float(score))
    assert (score.attrs['n_trials'], score.attrs['n_valid']) == (3, 3)
    assert score.attrs['result_validity'] == 'invalid'
    assert 'scorable' in score.attrs['result_validity_reason']


def test_invalid_sliders_that_coggym_scores_need_a_row(scored_reference, tmp_path):
    """Every slider fails the strict check, but CogGym read numbers and scores them: no row is
    then a fault of the analysis, not a result."""
    root, stand_in = scored_reference
    _slider_run(stand_in, invalid=range(10))
    _no_row(stand_in)
    with pytest.raises(RuntimeError, match='scorable'):
        _registered(root, tmp_path)(make_subject(reply_with_trial))


def test_omitted_answer_events_are_counted_missing(scored_reference, tmp_path):
    root, stand_in = scored_reference
    _trial_queries(stand_in)
    score = _registered(root, tmp_path)(_Answers(omit=lambda trial: trial == 'trial-1'))
    assert float(score) == 0.375
    assert (score.attrs['n_trials'], score.attrs['n_valid'], score.attrs['n_missing']) == (3, 2, 1)
    assert score.attrs['result_validity'] == 'invalid'    # 1 of 3 is above 10 percent


def test_constant_answers_give_no_fit_marked_invalid(scored_reference, tmp_path):
    root, stand_in = scored_reference
    _slider_run(stand_in)
    stand_in.analyze("""
def analyze(*args, **kwargs):
    return [{'model': 'fixture', 'experiment': 'Hu2023Fine/exp1', 'r2': None, 'n_items': 10}], [], []
""")
    score = _registered(root, tmp_path)(make_subject(reply_with_trial))
    assert math.isnan(float(score))
    assert (score.attrs['n_valid'], score.attrs['n_invalid'], score.attrs['n_missing']) == (10, 0, 0)
    assert score.attrs['result_validity'] == 'invalid'
    assert 'variance' in score.attrs['result_validity_reason']
    saved = _saved_score(tmp_path)
    assert saved.attrs['result_validity'] == 'invalid'
    assert 'variance' in saved.attrs['result_validity_reason']


def test_a_choice_only_experiment_counts_every_answer(scored_reference, tmp_path):
    root, stand_in = scored_reference
    _runs(stand_in, {(0, 'missing'), (1, 'missing')}, query=_CHOICE)
    score = _registered(root, tmp_path)(make_subject(reply_with_trial))
    assert (score.attrs['n_trials'], score.attrs['n_valid'], score.attrs['n_missing']) == (10, 8, 2)
    assert score.attrs['result_validity'] == 'invalid'
    assert score.attrs['response_validation']['reason_counts'] == {'not_read_by_coggym': 2}


@pytest.mark.parametrize('runs, validity', [
    (({(1, 'missing')}, {(1, 'missing'), (2, 'invalid')}), 'invalid'),    # 3 of 20
    (({(1, 'missing'), (2, 'invalid')}, set()), 'valid'),                 # 2 of 20, one repetition at 20 percent
])
def test_the_share_is_over_all_answers_of_all_repetitions(scored_reference, tmp_path, runs, validity):
    root, stand_in = scored_reference
    _runs(stand_in, *runs)
    score = _registered(root, tmp_path, repetitions=2)(make_subject(reply_with_trial))
    assert score.attrs['n_trials'] == 20
    assert score.attrs['result_validity'] == validity
    saved = _saved_score(tmp_path)
    assert (saved.attrs['n_invalid'], saved.attrs['n_missing']) == (score.attrs['n_invalid'], score.attrs['n_missing'])
    assert saved.attrs['result_validity'] == validity
