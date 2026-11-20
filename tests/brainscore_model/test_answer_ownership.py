"""Rule 4.11: the registration states which part owns an answer output several parts can give.

One eligible part answers without a statement; several need an owner, checked
at construction. Each output's owner stands on its own. The answers of the
registered part sets are pinned by ``test_answer_baseline.py``.
"""
import ast
from pathlib import Path

import pytest

import brainscore
from brainscore import BrainScoreModel
from brainscore_core.compatibility import ExecutionNotFeasibleError
from brainscore_core.contract import TaskContext
from brainscore_core.responses import Choice, Scale, Text
from brainscore_core.streaming_helpers import stimulus_session
from tests.architecture_support import plugin_test
from tests.brainscore_model.test_behavior_paths import FITTING, STIMULI, _features

pytestmark = pytest.mark.unit

ARGUMENTS = ('answer', 'reply', 'likelihood', 'behavioral_readout_layer')


def _model(*arguments, owners=None, calls=None):
    calls = [] if calls is None else calls
    parts = {
        'answer': lambda stimulus_row, instruction, label_set: calls.append('answer') or label_set[0],
        'reply': lambda prompt: calls.append('reply') or 'cat',
        'likelihood': lambda prompt, labels: calls.append('likelihood') or {
            label: -float(i) for i, label in enumerate(labels)},
        'behavioral_readout_layer': 'late',
    }
    config = {} if owners is None else {'capability_config': {'behavior-sessions': owners}}

    def extract(stimuli, layers, **kwargs):
        calls.append('extract')
        return _features(stimuli, layers, **kwargs)
    return BrainScoreModel('owned', extractor={'text': extract}, in_channels={'text'},
                           **{name: parts[name] for name in arguments}, **config)


def _run(model, outputs, task):
    session = stimulus_session(STIMULI, outputs, task)
    model.participate(session)
    return session


CHOICE = TaskContext(instruction='Which?', response=Choice(['cat', 'dog']),
                     fitting_stimuli=FITTING)


def test_one_part_is_selected_with_a_statement_of_its_derived_format():
    calls = []
    model = _model('likelihood', calls=calls, owners={'behavior:choice': ('likelihood', 'softmax_argmax')})
    choice = _run(model, ['behavior:choice'], CHOICE).collect('behavior:choice')
    assert choice.attrs['response']['method'] == 'likelihood' and set(calls) == {'likelihood'}


def test_a_derived_format_without_a_statement_is_not_declared_and_is_refused_before_compute():
    calls = []
    model = _model('likelihood', calls=calls)
    assert 'behavior:choice' not in model.out_channels
    with pytest.raises(ExecutionNotFeasibleError, match=r"does not produce \['behavior:choice'\]"):
        _run(model, ['behavior:choice'], CHOICE)
    assert calls == []


def test_two_native_parts_without_an_owner_are_refused_at_construction():
    with pytest.raises(ValueError, match=r"\['generation', 'readout'\] give 'behavior:rating' natively"
                                         r".*capability_config"):
        _model('reply', 'behavioral_readout_layer')


def test_one_native_part_owns_a_format_without_a_statement():
    calls = []
    task = TaskContext(instruction='Which?', response=Choice(['cat', 'dog']))
    choice = _run(_model('reply', 'likelihood', calls=calls), ['behavior:choice'], task)
    assert choice.collect('behavior:choice').attrs['response']['method'] == 'generation'
    assert 'likelihood' not in calls


def test_two_parts_with_a_stated_owner_answer_through_the_owner():
    calls = []
    model = _model('reply', 'likelihood', owners={'behavior:choice': ('likelihood', 'softmax_argmax')},
                   calls=calls)
    choice = _run(model, ['behavior:choice'], CHOICE).collect('behavior:choice')
    assert choice.attrs['response']['method'] == 'likelihood' and 'reply' not in calls


# The statement of each part for behavior:probabilities: the likelihood derives them.
PROBABILITIES = {'likelihood': ('likelihood', 'softmax'), 'readout': 'readout'}


@pytest.mark.parametrize('probabilities', ['likelihood', 'readout'])
def test_each_output_has_its_own_owner(probabilities):
    """Text by generation, choice by generation, probabilities by the stated part."""
    calls = []
    model = _model('reply', 'likelihood', 'behavioral_readout_layer', calls=calls, owners={
        'behavior:choice': 'generation', 'behavior:rating': 'generation',
        'behavior:probabilities': PROBABILITIES[probabilities]})
    session = _run(model, ['behavior:choice', 'behavior:probabilities'], CHOICE)
    assert session.collect('behavior:choice').attrs['response']['method'] == 'generation'
    assert session.collect('behavior:probabilities').attrs['response']['method'] == probabilities
    text = _run(model, ['behavior:text'], TaskContext(instruction='Say it.', response=Text()))
    assert text.collect('behavior:text').values.tolist() == ['cat'] * len(STIMULI)


@pytest.mark.parametrize('owners, message', [
    ({'behavior:choice': 'readout'}, r"names 'readout' for 'behavior:choice', but no "
                                     r"behavioral_readout_layer= \(with an extractor\) part is wired"),
    ({'behavior:text': 'likelihood'}, r"'likelihood' does not give 'behavior:text'"),
    ({'behavior:rt': 'likelihood'}, r"follows the owner of 'behavior:probabilities'"),
])
def test_an_owner_that_cannot_give_the_output_is_refused(owners, message):
    with pytest.raises(ValueError, match=message):
        _model('reply', 'likelihood', owners={'behavior:choice': 'generation', **owners})


def test_a_task_no_part_can_serve_stays_not_feasible():
    model = _model('answer', 'behavioral_readout_layer', owners={})
    task = TaskContext(response=Choice(['cat', 'dog'], verbatim=True), fitting_stimuli=FITTING)
    with pytest.raises(ExecutionNotFeasibleError, match='verbatim'):
        _run(model, ['behavior:choice'], task)


# --- each output's owner stands on its own ------------------------------------

CROSS = {'readout choice': {'behavior:choice': ('readout', 'argmax'),
                            'behavior:probabilities': ('likelihood', 'softmax')},
         'likelihood choice': {'behavior:choice': ('likelihood', 'softmax_argmax'),
                               'behavior:probabilities': 'readout'}}


@pytest.mark.parametrize('settings, channel', [({}, 'behavior:probabilities'),
                                               ({'reaction_time': 'surprisal'}, 'behavior:rt')],
                         ids=['probabilities', 'reaction time'])
@pytest.mark.parametrize('owners', sorted(CROSS))
def test_a_choice_and_probabilities_from_different_owners_are_refused_before_compute(owners, settings, channel):
    """The executor cannot take a likelihood or readout choice and the probabilities
    from the other part; neither owner is replaced, the session is refused."""
    calls = []
    model = _model('likelihood', 'behavioral_readout_layer', owners=CROSS[owners], calls=calls)
    task = TaskContext(instruction='Which?', response=Choice(['cat', 'dog'], **settings),
                       fitting_stimuli=FITTING)
    outputs = ['behavior:choice', channel]
    with pytest.raises(ExecutionNotFeasibleError, match='in one session'):
        _run(model, outputs, task)
    assert calls == []        # no extraction, no readout fit, no likelihood call


@pytest.mark.parametrize('owners', sorted(CROSS))
def test_a_choice_alone_answers_through_its_own_owner(owners):
    calls = []
    model = _model('likelihood', 'behavioral_readout_layer', owners=CROSS[owners], calls=calls)
    task = TaskContext(instruction='Which?', response=Choice(['cat', 'dog']), fitting_stimuli=FITTING)
    choice = _run(model, ['behavior:choice'], task).collect('behavior:choice')
    assert choice.attrs['response']['method'] == CROSS[owners]['behavior:choice'][0]


@pytest.mark.parametrize('settings, channel', [({}, 'behavior:probabilities'),
                                               ({'reaction_time': 'surprisal'}, 'behavior:rt')],
                         ids=['probabilities', 'reaction time'])
@pytest.mark.parametrize('probabilities', ['likelihood', 'readout'])
def test_a_generated_choice_takes_probabilities_from_their_owner(probabilities, settings, channel):
    calls = []
    model = _model('reply', 'likelihood', 'behavioral_readout_layer', calls=calls, owners={
        'behavior:choice': 'generation', 'behavior:rating': 'generation',
        'behavior:probabilities': PROBABILITIES[probabilities]})
    task = TaskContext(instruction='Which?', response=Choice(['cat', 'dog'], **settings),
                       fitting_stimuli=FITTING)
    session = _run(model, ['behavior:choice', channel], task)
    assert session.collect('behavior:choice').attrs['response']['method'] == 'generation'
    record = session.collect(channel).attrs['response']
    # A reaction time is a link from the probabilities; its record names their method.
    assert (record['parameters']['source_method'] if channel == 'behavior:rt'
            else record['method']) == probabilities
    assert ('likelihood' in calls) == (probabilities == 'likelihood')


# --- no choice among parts that do not own the output ------------------------

def test_several_parts_that_do_not_own_the_output_are_refused_before_compute():
    """The owner (generation, answer=) cannot serve raw text; likelihood and readout
    both could, and neither owns the choice: no other part answers it."""
    calls = []
    model = _model('answer', 'likelihood', 'behavioral_readout_layer', calls=calls,
                   owners={'behavior:probabilities': ('likelihood', 'softmax')})
    task = TaskContext(response=Choice(['cat', 'dog'], verbatim=True), fitting_stimuli=FITTING)
    with pytest.raises(ExecutionNotFeasibleError, match=r"'behavior:choice' is owned by 'generation', "
                                                        r"which cannot serve this task \(task settings: verbatim\)"):
        _run(model, ['behavior:choice'], task)
    assert calls == []


QWEN = ('answer', 'reply', 'likelihood', 'behavioral_readout_layer')
BLIP2 = ('answer', 'behavioral_readout_layer')   # an answer part and a readout (blip2-opt-2.7b before its reply and likelihood)
CAUSAL_LM = ('reply', 'likelihood')

# --- the registered part sets ------------------------------------------------

REGISTERED_PART_SETS = {('behavioral_readout_layer',), BLIP2, QWEN, CAUSAL_LM, ('answer', 'reply')}
# The statement each registered part set carries in the library (CLIP, random ViT and
# GPT-2; BLIP-2; the Qwen family; causal_lm; the API models), checked by the scan below.
REGISTERED_STATEMENTS = {
    ('behavioral_readout_layer',): {'behavior:choice': ('readout', 'argmax')},
    BLIP2: {},
    QWEN: {'behavior:rating': 'generation',
           'behavior:probabilities': ('likelihood', 'softmax')},
    CAUSAL_LM: {'behavior:probabilities': ('likelihood', 'softmax')},
    ('answer', 'reply'): {},
}
# The Qwen family states one of two parts for the probabilities (rule 4.11): the readout
# (qwen2.5-vl-3b, -7b, -3b-bf16, qwen3.6-27b) or the likelihood (the -vwfa variants), the
# statement above.
QWEN_READOUT_STATEMENT = {**REGISTERED_STATEMENTS[QWEN], 'behavior:probabilities': 'readout'}
# blip2-opt-2.7b wires the same four parts and states that the readout keeps the rating; the
# scan holds that registration, and only it, to this statement.
BLIP2_REGISTRATION = 'models/blip2_opt_2_7b/model.py'
BLIP2_STATEMENT = {'behavior:rating': 'readout'}
# clip-vit-b-32-zero-shot wires an answer part and a readout, and states that its answer
# part chooses by similarity (rule 4.11).
ZERO_SHOT_REGISTRATION = 'models/clip_vit_b_32/model.py'
ZERO_SHOT_STATEMENT = {'behavior:choice': 'similarity'}
API = ('answer', 'reply')
READOUT = ('behavioral_readout_layer',)

OPTIONS = STIMULI.copy()
OPTIONS['options'] = [['cat', 'dog']] * len(STIMULI)
OPTIONS.identifier = 'behavior-paths-options'
# The task forms the library's benchmarks and harnesses send beyond the numeric
# baseline: EWoK (verbatim choice; log probabilities of the options of each row),
# CogGym (free text).
EWOK_CHOICE = TaskContext(response=Choice(['1', '2'], verbatim=True))
EWOK_LOGPROBS = TaskContext(response=Choice('options', verbatim=True))
FREE_TEXT = TaskContext(instruction='Say it.', response=Text())
# (part set, task, stimuli, requested channel, method in the response record or None when refused)
TASK_FORMS = {
    'qwen ewok choice': (QWEN, EWOK_CHOICE, STIMULI, 'behavior:choice', 'generation'),
    'causal lm ewok choice': (CAUSAL_LM, EWOK_CHOICE, STIMULI, 'behavior:choice', 'generation'),
    'api ewok choice': (API, EWOK_CHOICE, STIMULI, 'behavior:choice', 'generation'),
    'readout ewok choice': (READOUT, EWOK_CHOICE, STIMULI, 'behavior:choice', None),
    'qwen ewok logprobs': (QWEN, EWOK_LOGPROBS, OPTIONS, 'behavior:log_probabilities', 'likelihood'),
    'causal lm ewok logprobs': (CAUSAL_LM, EWOK_LOGPROBS, OPTIONS, 'behavior:log_probabilities',
                                'likelihood'),
    'qwen free text': (QWEN, FREE_TEXT, STIMULI, 'behavior:text', 'generation'),
    'causal lm free text': (CAUSAL_LM, FREE_TEXT, STIMULI, 'behavior:text', 'generation'),
    'api free text': (API, FREE_TEXT, STIMULI, 'behavior:text', 'generation'),
}


@pytest.mark.parametrize('case', sorted(TASK_FORMS))
def test_the_owner_answers_each_task_form_a_registered_benchmark_sends(case):
    parts, task, stimuli, channel, method = TASK_FORMS[case]
    calls = []
    model = _model(*parts, owners=REGISTERED_STATEMENTS[parts], calls=calls)
    session = stimulus_session(stimuli, [channel], task)
    if method is None:
        with pytest.raises(ExecutionNotFeasibleError):
            model.participate(session)
        assert calls == []
        return
    model.participate(session)
    assert session.collect(channel).attrs['response']['method'] == method


# --- the registrations state the owners in the table ---------------------------

ARGUMENTS_WITH_OUTPUTS = ('answer', 'reply', 'likelihood', 'behavioral_readout_layer')
# The statement of every part set the library builds: the registered part sets, a
# model with answer= alone, and models with no answer part.
LIBRARY_STATEMENTS = {**REGISTERED_STATEMENTS, (): {}, ('answer',): {}}
# Constructor calls whose keywords come partly from ``**`` expansion: the scan cannot
# see those keywords. Each is listed with what the expansion carries.
EXPANDED = {
    'models/qwen25_vl_3b/model.py': 'variant_kwargs: intervention only',
    'tools/auto_register.py': 'brainscore_model_kwargs: the caller\'s keywords, checked by the constructor',
}


def _registrations():
    """Every BrainScoreModel call in the library: its answer arguments and stated owners.

    Returns the readable calls, the files whose calls take keywords through ``**``
    expansion, and the files whose ``capability_config`` is not a literal (a capability
    made from the template names its identifier). The scan cannot read the last two
    groups; the constructor checks them."""
    root = Path(brainscore.__file__).parent
    found, expanded, not_literal = [], set(), set()
    for path in sorted(path for path in root.rglob('*.py') if not plugin_test(path)):
        where = path.relative_to(root).as_posix()
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
            if isinstance(node, ast.Call) and 'BrainScoreModel' in (
                    getattr(node.func, 'id', None), getattr(node.func, 'attr', None)):
                if any(keyword.arg is None for keyword in node.keywords):
                    expanded.add(where)
                keywords = {k.arg: k.value for k in node.keywords if k.arg}
                wired = [name for name in ARGUMENTS_WITH_OUTPUTS if name in keywords
                         and not (isinstance(keywords[name], ast.Constant) and keywords[name].value is None)]
                try:
                    config = (ast.literal_eval(keywords['capability_config'])
                              if 'capability_config' in keywords else {})
                except ValueError:
                    not_literal.add(where)
                    continue
                found.append((where, wired, config.get('behavior-sessions', {})))
    return found, expanded, not_literal


def _answering_parts(model):
    """The part that answers each response format the model declares (or the refusal), through
    ``participate``: what a user of the registration sees."""
    from brainscore_core.responses import Scale, Text
    from tests.brainscore_model.stage2_answers_baseline import RATINGS
    tasks = {'behavior:choice': CHOICE, 'behavior:probabilities': CHOICE,
             'behavior:rating': TaskContext(instruction='How much?', response=Scale(1, 7), fitting_stimuli=RATINGS),
             'behavior:text': TaskContext(instruction='Say it.', response=Text())}
    parts = {}
    for channel, task in tasks.items():
        if channel not in model.out_channels:
            continue
        try:
            parts[channel] = _run(model, [channel], task).collect(channel).attrs['response']['method']
        except ExecutionNotFeasibleError as error:
            parts[channel] = type(error).__name__
    return parts


def test_every_registration_answers_each_format_with_the_part_of_its_part_set():
    """Every ``BrainScoreModel(...)`` call of the library, helpers included, whose part set is
    in ``LIBRARY_STATEMENTS``: its statement is accepted, and through ``participate`` each
    format it declares is answered by the part that the part set's statement gives (the Qwen
    family: either probabilities statement). A statement that only names an owner the part set
    already has is valid. The Qwen registrations are also built by their registry loaders
    (``tests/test_models/test_qwen_probability_statements.py``), the readout-only ones by their builders
    (``tests/test_models/test_readout_registrations.py``)."""
    registrations, expanded, not_literal = _registrations()
    assert set(EXPANDED) <= expanded, 'a call listed in EXPANDED no longer expands keywords: remove its line'
    assert {'models/qwen36_27b/model.py', 'models/blip2_opt_2_7b/model.py',
            'model_helpers/causal_lm.py'} <= {r[0] for r in registrations} | not_literal
    for where, wired, stated in registrations:
        model = _model(*wired, owners=stated)                 # construction accepts it
        if tuple(wired) not in LIBRARY_STATEMENTS:
            continue
        references = [LIBRARY_STATEMENTS[tuple(wired)]] + ([QWEN_READOUT_STATEMENT] if tuple(wired) == QWEN else [])
        if where == BLIP2_REGISTRATION:
            # BLIP-2 wires the Qwen part set but keeps ratings with the readout; the Qwen
            # registrations keep their own references.
            references = [BLIP2_STATEMENT]
        if where == ZERO_SHOT_REGISTRATION and tuple(wired) == BLIP2:
            references = [ZERO_SHOT_STATEMENT]
        found = _answering_parts(model)
        assert any(found == _answering_parts(_model(*wired, owners=reference)) for reference in references), (where, found)



# --- every registration: the method an answer records is the declared one (rule 4.11) ---

# The methods a part of a BrainScoreModel states (rule 4.11, Response Format Standard 4.3).
DECLARED_METHODS = {'generation', 'likelihood', 'readout', 'similarity'}
# The method each part declares when no statement names another; 'extract' is the readout.
PART_METHODS = {'answer': 'generation', 'reply': 'generation', 'likelihood': 'likelihood',
                'extract': 'readout'}
METHOD_FORMS = {
    'shared labels': ('behavior:choice', CHOICE, STIMULI),
    'labels per row': ('behavior:choice', TaskContext(response=Choice('options'), fitting_stimuli=FITTING),
                       OPTIONS),
    'probabilities': ('behavior:probabilities', CHOICE, STIMULI),
    'rating': ('behavior:rating', TaskContext(instruction='How much?', response=Scale(1, 7)), STIMULI),
    'text': ('behavior:text', FREE_TEXT, STIMULI),
}


def test_every_registration_records_the_declared_method_of_the_part_that_answered():
    """For every ``BrainScoreModel(...)`` call the scan reads (no list of plugins): each stated
    method is in the vocabulary, and through ``participate`` each answer records the declared
    method of the part that was called (shared labels and labels from a column alike).
    Limit: a registration whose statement is not a literal (the Qwen family looks it up in a
    table) is not read by the scan; its record is checked through its registry loader
    (``tests/test_models/test_qwen_probability_statements.py``)."""
    registrations, _, _ = _registrations()
    for where, wired, stated in registrations:
        declared = {value if isinstance(value, str) else value[0] for value in stated.values()}
        assert declared <= DECLARED_METHODS, (where, declared - DECLARED_METHODS)
        for form, (channel, task, stimuli) in METHOD_FORMS.items():
            calls = []
            model = _model(*wired, owners=stated, calls=calls)
            if channel not in model.out_channels:
                continue
            session = stimulus_session(stimuli, [channel], task)
            try:
                model.participate(session)
            except ExecutionNotFeasibleError:
                continue
            called = next(part for part in PART_METHODS if part in calls)
            expected = ('similarity' if called == 'answer' and 'similarity' in declared
                        else PART_METHODS[called])
            assert session.collect(channel).attrs['response']['method'] == expected, (where, form, called)
