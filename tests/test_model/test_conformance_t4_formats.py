"""Conformance test T4: one format, different implementations (Response Format Standard 11.1).

Each pseudo benchmark of ``tests/pseudo_benchmarks.py`` requests one format and is
answered by every producer that can give it: a reply part, a likelihood part, a
readout, a custom subject that answers cells and a custom subject that answers one
assembly. Each run goes through ``brainscore.score`` with objects (pre-flight, N/A,
then the benchmark call). The stored answer has the format's schema, the score is a
known value, the benchmark code is the same for every producer, and only the response
record differs (standard 4.3). The pseudo benchmarks are made-up data, never registered.
"""
import inspect
import math

import numpy as np
import pytest

import brainscore
from brainscore_core.compatibility import ExecutionNotFeasibleError
from brainscore_core.contract import TaskContext
from brainscore_core.responses import Choice, Scale
from brainscore_core.streaming_helpers import stimulus_session
from tests import pseudo_benchmarks as pseudo

pytestmark = pytest.mark.unit

PRODUCERS = {
    'reply': pseudo.reply_model,
    'likelihood': pseudo.likelihood_model,
    'readout': pseudo.readout_model,
    'cells': lambda calls=None: pseudo.CellSubject(),
    'assembly': lambda calls=None: pseudo.AssemblySubject(),
}

# (origin, method, conversion, source_format) of each producer's answer (standard 4.3).
NATIVE_GENERATION = ('native', 'generation', None, None)
NATIVE_EXTERNAL = ('native', 'external', None, None)
NATIVE_LOOKUP = ('native', 'lookup', None, None)
LINK = ('derived', 'link', None, 'behavior:probabilities')
RECORDS = {
    'behavior:choice': {
        'reply': NATIVE_GENERATION,
        'likelihood': ('derived', 'likelihood', 'softmax_argmax', 'behavior:log_probabilities'),
        'readout': ('derived', 'readout', 'argmax', 'behavior:probabilities'),
        'cells': NATIVE_EXTERNAL,
        'assembly': ('derived', 'lookup', 'argmax', 'behavior:probabilities')},
    'behavior:probabilities': {
        'likelihood': ('derived', 'likelihood', 'softmax', 'behavior:log_probabilities'),
        'readout': ('native', 'readout', None, None),
        'cells': NATIVE_EXTERNAL, 'assembly': NATIVE_LOOKUP},
    'behavior:rating': {
        'reply': NATIVE_GENERATION, 'readout': ('native', 'readout', None, None),
        'cells': NATIVE_EXTERNAL, 'assembly': NATIVE_LOOKUP},
    'behavior:rt': {'likelihood': LINK, 'readout': LINK, 'cells': NATIVE_EXTERNAL, 'assembly': NATIVE_LOOKUP},
    'behavior:text': {'reply': NATIVE_GENERATION, 'cells': NATIVE_EXTERNAL, 'assembly': NATIVE_LOOKUP},
}
# The literal score of each (format, producer). float32 answers keep their float32 rounding.
SCORES = {
    'behavior:choice': dict.fromkeys(RECORDS['behavior:choice'], 0.75),
    'behavior:probabilities': {'likelihood': 0.7749999761581421, 'readout': 0.5012450218200684,
                               'cells': 0.775, 'assembly': 0.7749999761581421},
    'behavior:rating': {'reply': 0.967821581748397, 'cells': 0.967821581748397,
                        'assembly': 0.967821581748397},
    'behavior:rt': dict.fromkeys(RECORDS['behavior:rt'], 0.9967631883679134),
    'behavior:text': dict.fromkeys(RECORDS['behavior:text'], 0.75),
}
# Producers that cannot give a format: N/A from the declarations (no rating from a
# likelihood, no text from a likelihood or a readout, no probabilities or time from a reply).
NOT_DECLARED = [(fmt, producer) for fmt in pseudo.BENCHMARKS for producer in PRODUCERS
                if producer not in RECORDS[fmt]]
SCHEMA = {   # dims, dtype
    'behavior:choice': (('presentation', 'choice'), np.float32),
    'behavior:probabilities': (('presentation', 'choice'), np.float32),
    'behavior:rating': (('presentation',), np.float64),
    'behavior:rt': (('presentation',), np.float64),
    'behavior:text': (('presentation',), object),
}


def _spied(subject, strip_record=False):
    """Keep every session the subject takes part in. With ``strip_record`` the stored
    answer the benchmark collects has no response record, so a metric that read it fails."""
    sessions, participate = [], subject.participate

    def spy(session):
        sessions.append(session)
        if strip_record:
            collect = session.collect

            def bare(channel=None):
                answer = collect(channel).copy()
                answer.attrs = {k: v for k, v in answer.attrs.items() if k != 'response'}
                return answer
            session.collect = bare
        return participate(session)
    subject.participate = spy
    return sessions


def _score(subject, benchmark):
    return brainscore.score(subject, benchmark, check_mem=False)


def _cases():
    return [(fmt, producer) for fmt, by in RECORDS.items() for producer in by
            if not (fmt == 'behavior:rating' and producer == 'readout')]


# --- every format, every producer that can give it --------------------------------------

@pytest.mark.parametrize('fmt, producer', _cases(), ids='{0[0]}-{0[1]}'.format)
def test_each_producer_answers_the_same_benchmark(fmt, producer):
    subject = PRODUCERS[producer]()
    sessions = _spied(subject)
    score = _score(subject, pseudo.BENCHMARKS[fmt]())
    assert float(score) == pytest.approx(SCORES[fmt][producer], rel=1e-6)   # readout: sklearn rounding
    assert (score.attrs['n_invalid'], score.attrs['n_missing']) == (0, 0)
    answer = sessions[0].collect(fmt)
    dims, dtype = SCHEMA[fmt]
    if (fmt, producer) in (('behavior:probabilities', 'cells'), ('behavior:probabilities', 'readout')):
        dtype = np.float64   # float64 cells and the readout's rows; the format keeps the digits given
    assert tuple(answer.dims) == dims and answer.values.dtype == np.dtype(dtype)
    assert list(answer['stimulus_id'].values) == pseudo.IDS
    if 'choice' in dims:
        assert sorted(answer['choice'].values) == sorted(pseudo.OPTIONS)
    assert list(answer['response_state'].values) == ['valid'] * len(pseudo.IDS)
    assert list(answer['response_reason'].values) == [''] * len(pseudo.IDS)
    assert all(isinstance(raw, str) and raw for raw in answer['raw_answer'].values)
    record = answer.attrs['response']
    assert record['format'] == fmt
    assert (record['origin'], record['method'], record['conversion'],
            record['source_format']) == RECORDS[fmt][producer]
    assert score.attrs['response'] == (record,)          # the score copies the record


@pytest.mark.parametrize('fmt, producer', _cases(), ids='{0[0]}-{0[1]}'.format)
def test_the_metric_never_reads_the_response_record(fmt, producer):
    subject = PRODUCERS[producer]()
    _spied(subject, strip_record=True)
    assert float(_score(subject, pseudo.BENCHMARKS[fmt]())) == pytest.approx(SCORES[fmt][producer], rel=1e-6)


@pytest.mark.parametrize('benchmark', sorted(pseudo.BENCHMARKS.values(), key=lambda b: b.__name__),
                         ids=lambda b: b.__name__)
def test_the_benchmark_code_names_no_producer(benchmark):
    """One class runs for every producer; its code reads no record and names no part."""
    source = inspect.getsource(benchmark)
    for word in ('attrs', "'response'", "'method'", 'likelihood', 'readout', 'generation', 'reply'):
        assert word not in source, word


def test_the_records_differ_only_where_the_standard_says():
    """Per format: the producers give records that differ in origin, method, conversion and
    source format, and in nothing that a metric reads."""
    for fmt, by in RECORDS.items():
        assert len(set(by.values())) >= 2, fmt
    assert {RECORDS['behavior:choice'][p][2] for p in RECORDS['behavior:choice']} == {
        None, 'argmax', 'softmax_argmax'}


def test_a_reaction_time_link_says_its_source_and_a_measured_time_its_units():
    records = {}
    for producer in ('likelihood', 'readout', 'cells'):
        subject = PRODUCERS[producer]()
        sessions = _spied(subject)
        _score(subject, pseudo.PseudoResponseTime())
        records[producer] = sessions[0].collect('behavior:rt').attrs['response']['parameters']
    assert records['likelihood'] == {'link': 'surprisal', 'source_method': 'likelihood', 'units': 'monotone'}
    assert records['readout'] == {'link': 'surprisal', 'source_method': 'readout', 'units': 'monotone'}
    assert records['cells'] == {'units': 'ms'}


def test_a_readout_choice_records_its_own_column_order():
    """Stated exception of standard 2.2 (1): a readout's argmax keeps the readout's order."""
    subject = pseudo.readout_model()
    sessions = _spied(subject)
    _score(subject, pseudo.PseudoChoice())
    assert sessions[0].collect('behavior:choice').attrs['response']['parameters'] == {
        'tie': 'first', 'option_order': 'readout'}


def test_the_image_variant_runs_the_same_metric(tmp_path):
    for subject in (pseudo.reply_model(in_channels=('vision:image',)), pseudo.CellSubject()):
        assert float(_score(subject, pseudo.PseudoChoice('vision:image', tmp_path))) == 0.75
    calls = []
    score = _score(pseudo.readout_model(calls), pseudo.PseudoChoice('vision:image', tmp_path))
    assert score.attrs['status'] == 'N/A' and 'takes in none of the inputs' in score.attrs['reason']
    assert calls == []


# --- a producer that cannot answer --------------------------------------------------------

@pytest.mark.parametrize('fmt, producer', NOT_DECLARED, ids='{0[0]}-{0[1]}'.format)
def test_a_format_the_producer_does_not_declare_is_na_before_any_call(fmt, producer):
    calls = []
    subject = PRODUCERS[producer](calls)
    score = _score(subject, pseudo.BENCHMARKS[fmt]())
    assert score.attrs['status'] == 'N/A' and math.isnan(float(score))
    assert 'cannot produce output channels' in score.attrs['reason']
    assert calls == []


@pytest.mark.parametrize('fmt', [f for f in pseudo.BENCHMARKS if f != 'behavior:text'])
def test_a_subject_that_declares_only_text_is_na(fmt):
    subject = pseudo.text_only_subject()
    score = _score(subject, pseudo.BENCHMARKS[fmt]())
    assert score.attrs['status'] == 'N/A' and math.isnan(float(score))
    assert subject.calls == []


def test_a_statement_that_cannot_serve_the_task_is_refused_before_any_call():
    """The readout owns the probabilities but the experiment has no fitting trials."""
    calls = []
    with pytest.raises(ExecutionNotFeasibleError, match='fitting_stimuli'):
        _score(pseudo.readout_model(calls), pseudo.PseudoProbabilities(fitting=False))
    assert calls == []


# --- invalid and missing trials -----------------------------------------------------------

def _run(subject, benchmark):
    sessions = _spied(subject)
    score = _score(subject, benchmark)
    return score, sessions[0].collect(next(iter(benchmark.requested_output_channels)))


def _trial(answer, stimulus_id):
    row = answer.sel(presentation=answer['stimulus_id'].values == stimulus_id)
    raw = row['raw_answer'].item()
    raw = None if raw is None or raw != raw else raw          # a missing row keeps a null
    return (row['response_state'].item(), raw, row['response_reason'].item())


def test_an_unparsed_choice_reply_is_invalid_and_counts_as_wrong():
    score, answer = _run(pseudo.reply_model(replies={'s0': 'maybe'}), pseudo.PseudoChoice())
    assert _trial(answer, 's0') == ('invalid', 'maybe', 'no label in the reply')
    assert np.isnan(answer.values[0]).all()               # nothing filled
    assert float(score) == 0.625 and score.attrs['n_invalid'] == 1


def test_a_missing_choice_is_a_missing_row_and_counts_as_wrong():
    subject = pseudo.CellSubject(skip={'s3'})
    score, answer = _run(subject, pseudo.PseudoChoice())
    assert _trial(answer, 's3') == ('missing', None, 'no answer event')
    assert float(score) == 0.625 and score.attrs['n_missing'] == 1


@pytest.mark.parametrize('subject, reason', [
    (pseudo.reply_model(replies={'s0': '9'}), '9 is outside 1 to 7'),
    (pseudo.CellSubject(cells={('behavior:rating', 's0'): 9.0}), '9.0 is outside 1 to 7'),
    (pseudo.CellSubject(cells={('behavior:rating', 's0'): 3.5}), '3.5 is not on the scale step 1'),
], ids=['reply off the scale', 'cell off the scale', 'cell off the step'])
def test_a_rating_off_the_scale_is_invalid_and_left_out(subject, reason):
    score, answer = _run(subject, pseudo.PseudoRating())
    state, raw, why = _trial(answer, 's0')
    assert (state, why) == ('invalid', reason)
    assert raw in ('9', '9.0', '3.5')                     # the reply text, or the cell as given
    assert float(score) == pytest.approx(0.971693634218629, rel=1e-9)   # the seven valid trials
    assert score.attrs['n_invalid'] == 1


def test_a_missing_rating_is_left_out():
    score, answer = _run(pseudo.CellSubject(skip={'s0'}), pseudo.PseudoRating())
    assert _trial(answer, 's0')[0] == 'missing'
    assert float(score) == pytest.approx(0.971693634218629, rel=1e-9)
    assert score.attrs['n_missing'] == 1


def test_a_probability_row_that_does_not_sum_to_one_is_invalid_and_left_out():
    subject = pseudo.CellSubject(cells={('behavior:probabilities', 's0'): np.array([0.5, 0.4])})
    score, answer = _run(subject, pseudo.PseudoProbabilities())
    state, raw, reason = _trial(answer, 's0')
    assert (state, raw) == ('invalid', '[0.5, 0.4]')
    assert reason.startswith('the probabilities sum to 0.9') and reason.endswith(', not 1')
    assert float(score) == pytest.approx(5.3 / 7, rel=1e-6)   # five 0.9 rows and two 0.4 rows
    assert score.attrs['n_invalid'] == 1


def test_a_negative_reaction_time_is_invalid_and_left_out():
    subject = pseudo.CellSubject(cells={('behavior:rt', 's0'): -5.0})
    score, answer = _run(subject, pseudo.PseudoResponseTime())
    assert _trial(answer, 's0') == ('invalid', '-5.0', 'negative')
    assert score.attrs['n_invalid'] == 1
    assert float(score) == pytest.approx(0.9968025347626165, rel=1e-9)


def test_text_the_benchmark_cannot_parse_is_valid_but_counts_as_wrong():
    """Any string is a valid F5 answer; the benchmark's own parse decides (rule 3.9)."""
    score, answer = _run(pseudo.reply_model(replies={'s0': 'I am not sure', 's1': None}), pseudo.PseudoText())
    assert _trial(answer, 's0') == ('valid', 'I am not sure', '')
    assert _trial(answer, 's1') == ('missing', None, 'no answer')
    assert float(score) == 0.5 and score.attrs['n_unparsed'] == 2 and score.attrs['n_missing'] == 1


# --- a malformed payload is a contract error, never an answer -------------------------------

@pytest.mark.parametrize('fmt, cell, message', [
    ('behavior:probabilities', np.array([0.2, 0.3, 0.5]), '3 values for 2 options'),
    ('behavior:rating', 'seven', 'a cell is one number, not str'),
    ('behavior:choice', 1, 'a cell is a str, not int'),
], ids=['probabilities of the wrong shape', 'rating as text', 'choice as a number'])
def test_a_malformed_cell_raises_and_nothing_is_stored(fmt, cell, message):
    subject = pseudo.CellSubject(cells={(fmt, 's0'): cell})
    sessions = _spied(subject)
    with pytest.raises(ValueError, match=message):
        _score(subject, pseudo.BENCHMARKS[fmt]())
    assert sessions[0].emitted == []


# --- joint requests --------------------------------------------------------------------------

CHOICE_TASK = TaskContext(instruction='Is this a real word?', response=Choice(pseudo.OPTIONS),
                          fitting_stimuli=pseudo.FITTING)
SCALE_TASK = TaskContext(instruction='How sure?', response=Scale(1, 7, 1),
                         fitting_stimuli=pseudo.FITTING_RATINGS)


def _session(subject, outputs, task):
    session = stimulus_session(pseudo.text_stimuli(), outputs, task)
    subject.participate(session)
    return session


@pytest.mark.parametrize('producer', ['likelihood', 'readout'])
def test_choice_and_probabilities_from_one_part_are_served_and_agree(producer):
    session = _session(PRODUCERS[producer](), ['behavior:choice', 'behavior:probabilities'], CHOICE_TASK)
    choice, probabilities = (session.collect(c) for c in ('behavior:choice', 'behavior:probabilities'))
    chosen = [choice['choice'].values[i] for i in choice.values.argmax(axis=1)]
    likeliest = [probabilities['choice'].values[i] for i in probabilities.values.argmax(axis=1)]
    assert chosen == likeliest


def test_choice_and_probabilities_from_a_reply_alone_are_refused_before_compute():
    calls = []
    with pytest.raises(ExecutionNotFeasibleError, match=r"it does not produce \['behavior:probabilities'\]"):
        _session(pseudo.reply_model(calls), ['behavior:choice', 'behavior:probabilities'], CHOICE_TASK)
    assert calls == []


@pytest.mark.parametrize('producer', sorted(PRODUCERS))
@pytest.mark.parametrize('task, missing', [(CHOICE_TASK, 'behavior:rating'), (SCALE_TASK, 'behavior:choice')],
                         ids=['choice task', 'scale task'])
def test_choice_and_rating_are_refused_when_the_session_is_built(producer, task, missing):
    """Core refuses a format its task's response does not name when the session is built
    (a ValueError: the request is malformed), for every producer; no joint choice and
    rating exists (standard 5)."""
    with pytest.raises(ValueError, match=rf"\['{missing}'\] requested, but the task's response "
                                         rf"{type(task.response).__name__} answers only on"):
        stimulus_session(pseudo.text_stimuli(), ['behavior:choice', 'behavior:rating'], task)


# --- findings, as decided --------------------------------------------------------------------

def test_a_readout_rating_answers_on_the_stated_scale():
    score, answer = _run(pseudo.readout_model(), pseudo.PseudoRating())
    assert score.attrs['n_invalid'] == 0 and not math.isnan(float(score))


def _readout_rating(scale, fitting=None):
    task = TaskContext(instruction='How sure?', response=scale,
                       fitting_stimuli=pseudo.FITTING_RATINGS if fitting is None else fitting)
    session = stimulus_session(pseudo.text_stimuli(), ['behavior:rating'], task)
    pseudo.readout_model().participate(session)
    return session.collect('behavior:rating')


def test_a_readout_rating_is_put_on_the_nearest_step_and_keeps_its_regression_value():
    answer = _readout_rating(Scale(1, 7, 1))
    assert answer.values.tolist() == [6.0, 2.0, 5.0, 2.0, 6.0, 3.0, 6.0, 2.0]
    assert list(answer['raw_answer'].values) == [
        '5.777777777777778', '2.2222222222222223', '4.888888888888889', '2.2222222222222223',
        '5.777777777777778', '3.111111111111111', '5.777777777777778', '2.2222222222222223']
    assert set(answer['response_state'].values) == {'valid'}
    assert answer.attrs['response']['parameters'] == {'rounding': 'nearest step, ties to the higher step'}


def test_a_readout_rating_outside_the_bounds_is_not_clipped():
    answer = _readout_rating(Scale(1, 5, 1))
    assert _trial(answer, 's0') == ('invalid', '5.777777777777778', '5.777777777777778 is outside 1 to 5')
    assert answer.values[0] == 5.777777777777778
    assert _trial(answer, 's1')[0] == 'valid' and answer.values[1] == 2.0


def test_a_readout_rating_on_a_scale_without_a_step_is_unchanged():
    answer = _readout_rating(Scale(1, 7))
    assert answer.values.tolist()[:3] == [5.777777777777778, 2.2222222222222223, 4.888888888888889]
    assert answer.attrs['response']['parameters'] == {}


# Constant fitting labels make the readout give that value for every trial (the cases of the
# review of the findings chunk: float rounding must not leave the bounds or the step).
@pytest.mark.parametrize('label, scale, stored, state', [
    (0.29, Scale(0.1, 0.3, 0.1), 0.3, 'valid'),
    (2.25, Scale(1.1, 7.1, 0.1), 2.3, 'valid'),
    (100000000.29, Scale(100000000.0, 100000001.0, 0.1), 100000000.3, 'valid'),
    (0.31, Scale(0.1, 0.3, 0.1), 0.31, 'invalid'),
    (6.5, Scale(1, 7, 1), 7.0, 'valid'),
    (7.0, Scale(1, 7, 1), 7.0, 'valid'),
    (0.1, Scale(-1e15, 1, 0.1), 0.1, 'valid'),
    (-3 + 16 * 0.3, Scale(-3, -3 + 16 * 0.3, 0.3), -3 + 16 * 0.3, 'valid'),   # high 1.7999999999999998
    (7.29, Scale(0.1, 0.1 + 24 * 0.3, 0.3), 0.1 + 24 * 0.3, 'valid'),
], ids=['top of a decimal scale', 'decimal tie', 'large magnitude', 'above the bounds',
        'integer tie', 'integer top', 'a far low bound', 'high one rounding below a point',
        'raw near a high one rounding off'])
def test_a_constant_readout_rating_lands_on_the_grid(label, scale, stored, state):
    answer = _readout_rating(scale, pseudo._fitting([label] * 4, f'pseudo-fit-constant-{label}'))
    assert set(answer.values.tolist()) == {stored}
    assert set(answer['response_state'].values) == {state}


def test_a_tie_goes_to_the_higher_step():
    from brainscore._model.behavioral import _on_scale_steps
    assert _on_scale_steps([2.5, 1.25, 6.75, 0.5, float('nan')], Scale(1, 7, 0.5))[0][:3] == [2.5, 1.5, 7.0]
    rated, parameters = _on_scale_steps([2.5, 3.5, 0.5], Scale(1, 7, 1))
    assert rated == [3.0, 4.0, 0.5] and parameters == {'rounding': 'nearest step, ties to the higher step'}


def test_a_text_answer_records_no_parser():
    # Nobody turned the text into the format; the benchmark, not the record, says it parses.
    subject = pseudo.reply_model()
    sessions = _spied(subject)
    _score(subject, pseudo.PseudoText())
    assert sessions[0].collect('behavior:text').attrs['response']['parsed_by'] is None


# --- behavior:log_probabilities (F6): a pseudo benchmark and every kind of producer --------

def test_log_probabilities_from_a_likelihood_part_score_the_known_value():
    calls = []
    score, answer = _run(pseudo.likelihood_model(calls), pseudo.PseudoLogProbabilities())
    assert float(score) == 0.75 and score.attrs['n_invalid'] == 0     # 6 of 8 agree with people
    assert calls == ['likelihood'] * len(pseudo.IDS)
    assert answer.values.dtype == np.float64
    assert answer.values[0].tolist() == [-0.10536051565782628, -2.302585092994046]   # log 0.9, log (1 - 0.9)
    record = answer.attrs['response']
    assert (record['format'], record['origin'], record['method'], record['category']) == (
        'behavior:log_probabilities', 'native', 'likelihood', 'model-derived')
    assert record['parameters'] == pseudo.LOG_PROBABILITY_SCHEMA


def test_log_probabilities_from_a_custom_assembly_subject_score_the_known_value():
    subject = pseudo.LogProbabilityAssemblySubject()
    score, answer = _run(subject, pseudo.PseudoLogProbabilities())
    assert float(score) == 0.75 and score.attrs['n_invalid'] == 0
    assert answer.attrs['response']['method'] == 'lookup'


def test_log_probabilities_as_cells_are_malformed_and_nothing_is_stored():
    # Point m: the format is an assembly with its record only; five schema fields cannot
    # default from a cell.
    subject = pseudo.CellSubject(out_channels={'behavior:log_probabilities'}, in_channels={'text'})
    sessions = _spied(subject)
    with pytest.raises(ValueError, match='an assembly with its response record'):
        _score(subject, pseudo.PseudoLogProbabilities())
    assert sessions[0].emitted == []


@pytest.mark.parametrize('producer', ['reply', 'readout', 'text only'])
def test_a_subject_that_does_not_declare_log_probabilities_is_na(producer):
    calls = []
    subject = {'reply': lambda: pseudo.reply_model(calls), 'readout': lambda: pseudo.readout_model(calls),
               'text only': pseudo.text_only_subject}[producer]()
    score = _score(subject, pseudo.PseudoLogProbabilities())
    assert score.attrs['status'] == 'N/A' and math.isnan(float(score))
    assert calls == []


def test_a_pseudo_ewok_runs_the_real_benchmark_on_made_up_items(tmp_path):
    calls = []
    benchmark = pseudo.pseudo_ewok(tmp_path)
    sessions = _spied(subject := pseudo.marker_likelihood_model(calls))
    score = _score(subject, benchmark)
    assert float(score) == 0.5                          # version 0: 1.0, version 1: 0.0
    assert score.attrs['version_scores'] == {'0': 1.0, '1': 0.0}
    assert calls == ['likelihood'] * 6                  # six contexts, one pass each
    answers = [session.collect('behavior:log_probabilities') for session in sessions]
    assert all(answer.attrs['response']['parameters']['layout'] == 'continuation' for answer in answers)


@pytest.mark.parametrize('scale', [Scale(1, 7, 1), Scale(0.1, 0.3, 0.1), Scale(1.1, 7.1, 0.1),
                                   Scale(-1e15, 1, 0.1), Scale(100000000.0, 100000001.0, 0.1),
                                   Scale(0, 1000000000000001, 1), Scale(-5, 5, 0.25),
                                   Scale(-3, -3 + 16 * 0.3, 0.3), Scale(0.1, 0.1 + 24 * 0.3, 0.3)],
                         ids=str)
def test_every_rounded_readout_rating_is_on_core_s_grid(scale):
    """One grid for the readout and core: each stored value inside the bounds is on the step."""
    from brainscore._model.behavioral import _on_scale_steps
    from brainscore_core.responses import _on_step
    rng = np.random.default_rng(0)
    raw = list(rng.uniform(scale.low, scale.high, 500)) + [scale.low, scale.high]
    rated, _ = _on_scale_steps(raw, scale)
    assert all(scale.low <= v <= scale.high and _on_step(v, scale.low, scale.step) for v in rated)


# --- the vision adapter as a producer ---------------------------------------------------
# A legacy vision model behind the permanent adapter answers the image variants of the
# choice and probabilities benchmarks with the same benchmark code; every other format is
# N/A before any call (the adapter declares no other behavior format, and the legacy model
# takes only pictures).

@pytest.mark.parametrize('fmt, benchmark, record, expected', [
    ('behavior:choice', lambda folder: pseudo.PseudoChoice('vision:image', folder),
     ('derived', 'legacy', 'argmax', 'behavior:probabilities'), 0.75),
    ('behavior:probabilities', lambda folder: pseudo.PseudoProbabilities(channel='vision:image', folder=folder),
     ('native', 'legacy', None, None), 0.775),
], ids=['choice', 'probabilities'])
def test_the_vision_adapter_answers_the_same_benchmark(tmp_path, fmt, benchmark, record, expected):
    calls = []
    subject = pseudo.legacy_vision_model(calls)
    sessions = _spied(subject)
    score = _score(subject, benchmark(tmp_path))
    assert float(score) == expected
    stored = sessions[0].collect(fmt)
    dims, dtype = SCHEMA[fmt]
    if fmt == 'behavior:probabilities':
        dtype = np.float64   # the legacy model's own rows; the format keeps the digits given
    assert stored.dims == dims and stored.values.dtype == dtype
    answer = stored.attrs['response']
    assert (answer['origin'], answer['method'], answer['conversion'], answer['source_format']) == record
    assert calls == pseudo.IDS


@pytest.mark.parametrize('fmt', ['behavior:rating', 'behavior:rt', 'behavior:text', 'behavior:log_probabilities'])
def test_the_vision_adapter_is_na_for_every_other_format(fmt):
    calls = []
    benchmark = (pseudo.PseudoLogProbabilities() if fmt == 'behavior:log_probabilities'
                 else pseudo.BENCHMARKS[fmt]())
    score = _score(pseudo.legacy_vision_model(calls), benchmark)
    assert score.attrs['status'] == 'N/A' and math.isnan(float(score))
    assert calls == []
