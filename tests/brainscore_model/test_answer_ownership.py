"""Rule 4.11: the registration states which part owns an answer output several parts can give.

One eligible part answers without a statement; several need an owner, checked
at construction. Each output's owner stands on its own. The parity tests
compare with the selection of the base commit 280f49c, captured in
``tests/fixtures/fixed_order_baseline.json`` by ``fixed_order_baseline.py``.
"""
import ast
import json
from pathlib import Path

import pytest

import brainscore
from brainscore import BrainScoreModel
from brainscore_core.compatibility import ExecutionNotFeasibleError
from brainscore_core.contract import TaskContext
from brainscore_core.responses import Choice, Scale, Text
from brainscore_core.streaming_helpers import stimulus_session
from tests.brainscore_model.test_behavior_paths import FITTING, STIMULI, _features
from tests.brainscore_model.fixed_order_baseline import FIXTURE, task_from
from tests.fixed_order import fixed_order_owners

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


CHOICE = TaskContext(instruction='Which?', response=Choice(['cat', 'dog'], probabilities=True),
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
    model = _model('answer', 'behavioral_readout_layer', owners=fixed_order_owners('answer', 'behavioral_readout_layer'))
    task = TaskContext(instruction='Which?', response=Choice(['cat', 'dog'], confidence=Scale(1, 5)),
                       fitting_stimuli=FITTING)
    with pytest.raises(ExecutionNotFeasibleError, match='cannot give a Choice answer with reported confidence'):
        _run(model, ['behavior:choice'], task)


# --- finding 1: each output's owner stands on its own --------------------------

CROSS = {'readout choice': {'behavior:choice': ('readout', 'argmax'),
                            'behavior:probabilities': ('likelihood', 'softmax')},
         'likelihood choice': {'behavior:choice': ('likelihood', 'softmax_argmax'),
                               'behavior:probabilities': 'readout'}}


@pytest.mark.parametrize('settings', [{'probabilities': True}, {'reaction_time': 'surprisal'}],
                         ids=['probabilities', 'reaction time'])
@pytest.mark.parametrize('owners', sorted(CROSS))
def test_a_choice_and_probabilities_from_different_owners_are_refused_before_compute(owners, settings):
    """The executor cannot take a likelihood or readout choice and the probabilities
    from the other part; neither owner is replaced, the session is refused."""
    calls = []
    model = _model('likelihood', 'behavioral_readout_layer', owners=CROSS[owners], calls=calls)
    task = TaskContext(instruction='Which?', response=Choice(['cat', 'dog'], **settings),
                       fitting_stimuli=FITTING)
    outputs = ['behavior:choice', 'behavior:probabilities' if 'probabilities' in settings else 'behavior:rt']
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


@pytest.mark.parametrize('settings', [{'probabilities': True}, {'reaction_time': 'surprisal'}],
                         ids=['probabilities', 'reaction time'])
@pytest.mark.parametrize('probabilities', ['likelihood', 'readout'])
def test_a_generated_choice_takes_probabilities_from_their_owner(probabilities, settings):
    calls = []
    model = _model('reply', 'likelihood', 'behavioral_readout_layer', calls=calls, owners={
        'behavior:choice': 'generation', 'behavior:rating': 'generation',
        'behavior:probabilities': PROBABILITIES[probabilities]})
    task = TaskContext(instruction='Which?', response=Choice(['cat', 'dog'], **settings),
                       fitting_stimuli=FITTING)
    channel = 'behavior:probabilities' if 'probabilities' in settings else 'behavior:rt'
    session = _run(model, ['behavior:choice', channel], task)
    assert session.collect('behavior:choice').attrs['response']['method'] == 'generation'
    record = session.collect(channel).attrs['response']
    # A reaction time is a link from the probabilities; its record names their method.
    assert (record['parameters']['source_method'] if channel == 'behavior:rt'
            else record['method']) == probabilities
    assert ('likelihood' in calls) == (probabilities == 'likelihood')


# --- finding 2: no choice among parts that do not own the output -------------

def test_several_parts_that_do_not_own_the_output_are_refused_before_compute():
    """The owner (generation, answer=) cannot serve raw text; likelihood and readout
    both could, and neither owns the choice: no other part answers it (C4)."""
    calls = []
    model = _model('answer', 'likelihood', 'behavioral_readout_layer', calls=calls,
                   owners=fixed_order_owners('answer', 'likelihood', 'behavioral_readout_layer'))
    task = TaskContext(response=Choice(['cat', 'dog'], raw_text=True), fitting_stimuli=FITTING)
    with pytest.raises(ExecutionNotFeasibleError, match=r"'behavior:choice' is owned by 'generation', "
                                                        r"which cannot serve this task \(task settings: raw_text\)"):
        _run(model, ['behavior:choice'], task)
    assert calls == []


LEGACY_NO_INSTRUCTION = TaskContext(task_type='probabilities', label_set=['cat', 'dog'],
                                    fitting_stimuli=FITTING)
QWEN = ('answer', 'reply', 'likelihood', 'behavioral_readout_layer')
BLIP2 = ('answer', 'behavioral_readout_layer')
CAUSAL_LM = ('reply', 'likelihood')

# The cases where a registered model answers bare behavior through a part that does
# not own it, because the task form excludes the owner and exactly one other part
# can serve it: a legacy task without an instruction (as the facade sends) goes to
# the readout. Only bare behavior keeps this fallback (C4); a format channel is
# answered by its owner or refused.
TASK_FORM_CASES = {
    'qwen legacy without instruction': (QWEN, LEGACY_NO_INSTRUCTION, ['behavior'], 'readout'),
    'blip2 legacy without instruction': (BLIP2, LEGACY_NO_INSTRUCTION, ['behavior'], 'readout'),
}


@pytest.mark.parametrize('case', sorted(TASK_FORM_CASES))
def test_the_one_part_that_serves_a_task_form_answers_for_a_registered_model(case):
    parts, task, outputs, source = TASK_FORM_CASES[case]
    owners = fixed_order_owners(*parts)
    assert owners[outputs[0]] != source                  # not the owner of the output
    session = _run(_model(*parts, owners=owners), outputs, task)
    assert session.collect(outputs[0]).attrs['response']['method'] == source


# --- finding 3: parity with the base commit ----------------------------------

BASELINE = json.loads(Path(FIXTURE).read_text(encoding='utf-8'))
REGISTERED_PART_SETS = {('behavioral_readout_layer',), BLIP2, QWEN, CAUSAL_LM, ('answer', 'reply')}
# The statement each registered part set carries in the library (CLIP, random ViT and
# GPT-2; BLIP-2; the Qwen family; causal_lm; the API models), checked by the scan below.
REGISTERED_STATEMENTS = {
    ('behavioral_readout_layer',): {'behavior:choice': ('readout', 'argmax')},
    BLIP2: {'behavior': 'generation'},
    QWEN: {'behavior': 'generation', 'behavior:rating': 'generation',
           'behavior:probabilities': ('likelihood', 'softmax')},
    CAUSAL_LM: {'behavior:probabilities': ('likelihood', 'softmax')},
    ('answer', 'reply'): {},
}
# The only cases where the owner rule differs from the base commit: a raw-text Choice,
# whose choice is owned by answer= (generation without reply), which cannot serve raw
# text; the base commit let the likelihood or the readout answer it. Since C4 no part
# answers a format it does not own, so these are refused. Of the registered part sets
# only BLIP-2 (answer, readout) is among them, with shared labels (a readout reads no
# per-row options).
REFUSED_NOW = {(parts, (('labels', labels), ('probabilities', p), ('raw_text', True), ('reaction_time', rt)))
               for parts, all_labels in ((BLIP2, ('shared',)),
                                         (('answer', 'likelihood'), ('shared', 'per_row')),
                                         (('answer', 'likelihood', 'behavioral_readout_layer'),
                                          ('shared', 'per_row')))
               for labels in all_labels for p in (False, True) for rt in (None, 'surprisal')}


def _key(case):
    task = case['task']
    return (tuple(case['parts']), tuple(sorted((k, v) for k, v in task.items() if k in (
        'labels', 'probabilities', 'raw_text', 'reaction_time'))))


def test_the_baseline_covers_every_part_combination_and_task_form():
    assert BASELINE['base'] == '280f49c'
    assert len({tuple(case['parts']) for case in BASELINE['cases']}) == 15
    assert len(BASELINE['cases']) == 15 * 80


@pytest.mark.parametrize('parts', sorted({tuple(case['parts']) for case in BASELINE['cases']}),
                         ids='+'.join)
def test_the_fixed_order_owners_choose_what_the_base_commit_chose(monkeypatch, parts):
    model = _model(*parts, owners=fixed_order_owners(*parts))
    monkeypatch.setattr(model._behavioral, 'fit_behavioral_readout', lambda stimuli: None)
    differences = set()
    for case in (case for case in BASELINE['cases'] if tuple(case['parts']) == parts):
        try:
            model.start_task(task_from(case['task'], FITTING))
            outcome = [model._behavioral.response_path, model._behavioral.probability_source]
        except ExecutionNotFeasibleError:
            outcome = 'refused'
        if outcome != case['outcome']:
            assert outcome == 'refused' and case['task']['raw_text'] and not case['task']['confidence'], case
            differences.add(_key(case))
    assert differences == {key for key in REFUSED_NOW if key[0] == parts}
    if parts in REGISTERED_PART_SETS and parts != BLIP2:
        assert not differences


SOURCE = {'generation': {'answer': 'answer', 'reply': 'reply'}, 'likelihood': 'likelihood', 'readout': 'readout'}
RATINGS = FITTING.copy()
RATINGS['label'] = [6.0, 2.0, 6.0, 2.0]
RATINGS.identifier = 'behavior-paths-ratings'
CHOICE_FORM = {'kind': 'choice', 'labels': 'shared', 'probabilities': False, 'reaction_time': None,
               'raw_scores': False, 'raw_text': False, 'confidence': False, 'fitting': True}
SESSIONS = {   # baseline task descriptions and the outputs the session requests
    'legacy answer': ({'kind': 'legacy', 'instruction': True, 'label_set': ['cat', 'dog'], 'fitting': True},
                      ['behavior']),
    'legacy readout': ({'kind': 'legacy', 'instruction': False, 'label_set': ['cat', 'dog'], 'fitting': True},
                       ['behavior']),
    'choice': (dict(CHOICE_FORM, probabilities=True), ['behavior:choice', 'behavior:probabilities']),
    'raw scores': (dict(CHOICE_FORM, raw_scores=True, fitting=False),
                   ['behavior:choice', 'behavior:probabilities']),
    'rating': ({'kind': 'scale', 'fitting': True}, ['behavior:rating']),
}


# Sessions that request a format the part set does not declare are refused at
# session start (C3). Every registered part set declares the formats of SESSIONS.
UNDECLARED = set()


@pytest.mark.parametrize('session_name', sorted(SESSIONS))
@pytest.mark.parametrize('parts', sorted(REGISTERED_PART_SETS), ids='+'.join)
def test_registered_part_sets_emit_the_sources_the_base_commit_chose(parts, session_name):
    """What a session emits (the source of each answer and the parts called) follows the base selection."""
    description, outputs = SESSIONS[session_name]
    case = next(c for c in BASELINE['cases'] if tuple(c['parts']) == parts and c['task'] == description)
    task = task_from(description, RATINGS if description['kind'] == 'scale' else FITTING)
    calls = []
    model = _model(*parts, owners=REGISTERED_STATEMENTS[parts], calls=calls)
    if case['outcome'] == 'refused' or (parts, session_name) in UNDECLARED:
        with pytest.raises(ExecutionNotFeasibleError):
            _run(model, outputs, task)
        assert calls == []
        return
    path, source = case['outcome']
    # A legacy task is answered by answer=; a response task by reply= when wired.
    generator = 'reply' if 'reply' in parts and task.response is not None else 'answer'
    session = _run(model, outputs, task)
    answer = session.collect(outputs[0])
    # Both answer= and reply= are the generation method; `calls` below names the part.
    assert answer.attrs['response']['method'] == path
    if source is not None:
        assert session.collect('behavior:probabilities').attrs['response']['method'] == source
    called = {'likelihood': 'likelihood', 'readout': 'extract', 'generation': generator}
    assert called[path] in calls
    for part in {'answer', 'reply', 'likelihood'} - {called[path], called.get(source)}:
        assert part not in calls, (part, calls)


# --- the registrations state the owners the base order chose ------------------

ARGUMENTS_WITH_OUTPUTS = ('answer', 'reply', 'likelihood', 'behavioral_readout_layer')
# Constructor calls whose keywords come partly from ``**`` expansion: the scan cannot
# see those keywords. Each is listed with what the expansion carries.
EXPANDED = {
    'models/qwen25_vl_3b/model.py': 'variant_kwargs: intervention only',
    'tools/auto_register.py': 'brainscore_model_kwargs: the caller\'s keywords, checked by the constructor',
}


def _registrations():
    """Every BrainScoreModel call in the library: its answer arguments and stated owners."""
    root = Path(brainscore.__file__).parent
    found, expanded = [], set()
    for path in sorted(root.rglob('*.py')):
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
            if isinstance(node, ast.Call) and 'BrainScoreModel' in (
                    getattr(node.func, 'id', None), getattr(node.func, 'attr', None)):
                if any(keyword.arg is None for keyword in node.keywords):
                    expanded.add(path.relative_to(root).as_posix())
                keywords = {k.arg: k.value for k in node.keywords if k.arg}
                wired = [name for name in ARGUMENTS_WITH_OUTPUTS if name in keywords
                         and not (isinstance(keywords[name], ast.Constant) and keywords[name].value is None)]
                config = (ast.literal_eval(keywords['capability_config'])
                          if 'capability_config' in keywords else {})
                found.append((path.relative_to(root).as_posix(), wired, config.get('behavior-sessions', {})))
    return found, expanded


def test_every_registration_keeps_the_owner_the_base_order_chose():
    """Covers every ``BrainScoreModel(...)`` and ``<module>.BrainScoreModel(...)`` call in
    the library, helpers included (``model_helpers/causal_lm.py``). A statement may be
    left out where the one native part is the owner the base order chose. Not covered:
    keywords that arrive through ``**`` expansion (``EXPANDED``) and models built outside
    the library; the constructor checks those."""
    registrations, expanded = _registrations()
    assert expanded == set(EXPANDED)
    assert {'models/qwen25_vl_3b/model.py', 'models/qwen36_27b/model.py', 'models/blip2_opt_2_7b/model.py',
            'model_helpers/causal_lm.py'} <= {r[0] for r in registrations}
    for where, wired, stated in registrations:
        if tuple(wired) in REGISTERED_STATEMENTS:
            assert stated == REGISTERED_STATEMENTS[tuple(wired)], where
        owners = _model(*wired, owners=stated)._behavioral.owners   # construction accepts it
        for output, statement in fixed_order_owners(*wired).items():
            part = statement if isinstance(statement, str) else statement[0]
            assert owners[output] == part, (where, output)
