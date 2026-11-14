"""Rule 4.11: the registration states which part owns an answer output several parts can give.

One eligible part answers without a statement; several need an owner, checked
at construction. The parity tests pin that the owners the registrations state
choose exactly the part the removed fixed order (generation, likelihood,
readout) chose, for every part combination and task form.
"""
import ast
import itertools
from pathlib import Path

import pytest

import brainscore
from brainscore import BrainScoreModel
from brainscore_core.compatibility import ExecutionNotFeasibleError
from brainscore_core.contract import TaskContext
from brainscore_core.responses import Choice, Scale, Text
from brainscore_core.streaming_helpers import stimulus_session
from tests.brainscore_model.test_behavior_paths import FITTING, STIMULI, _features
from tests.fixed_order import ORDER, fixed_order_owners, owned

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
    return BrainScoreModel('owned', extractor={'text': _features}, in_channels={'text'},
                           **{name: parts[name] for name in arguments}, **config)


def _run(model, outputs, task):
    session = stimulus_session(STIMULI, outputs, task)
    model.participate(session)
    return session


CHOICE = TaskContext(instruction='Which?', response=Choice(['cat', 'dog'], probabilities=True),
                     fitting_stimuli=FITTING)


def test_one_part_is_selected_without_a_statement():
    calls = []
    choice = _run(_model('likelihood', calls=calls), ['behavior:choice'], CHOICE).collect('behavior:choice')
    assert choice.attrs['measure']['source'] == 'likelihood' and set(calls) == {'likelihood'}


def test_two_parts_without_an_owner_are_refused_at_construction():
    with pytest.raises(ValueError, match=r"\['generation', 'likelihood'\] can all give 'behavior:choice'"
                                         r".*capability_config"):
        _model('reply', 'likelihood')


def test_two_parts_with_a_stated_owner_answer_through_the_owner():
    calls = []
    model = _model('reply', 'likelihood', owners={'behavior:choice': 'likelihood'}, calls=calls)
    choice = _run(model, ['behavior:choice'], CHOICE).collect('behavior:choice')
    assert choice.attrs['measure']['source'] == 'likelihood' and 'reply' not in calls


@pytest.mark.parametrize('probabilities', ['likelihood', 'readout'])
def test_each_output_has_its_own_owner(probabilities):
    """Text by generation, choice by generation, probabilities by the stated part."""
    calls = []
    model = _model('reply', 'likelihood', 'behavioral_readout_layer', calls=calls, owners={
        'behavior:choice': 'generation', 'behavior:rating': 'generation',
        'behavior:probabilities': probabilities})
    session = _run(model, ['behavior:choice', 'behavior:probabilities'], CHOICE)
    assert session.collect('behavior:choice').attrs['measure']['source'] == 'reply'
    assert session.collect('behavior:probabilities').attrs['measure']['source'] == probabilities
    text = _run(model, ['behavior:text'], TaskContext(instruction='Say it.', response=Text()))
    assert text.collect('behavior:text').values.tolist() == ['cat'] * len(STIMULI)


@pytest.mark.parametrize('owners, message', [
    ({'behavior:choice': 'readout'}, r"'readout' stated for 'behavior:choice' cannot give it"),
    ({'behavior:text': 'likelihood'}, r"'likelihood' stated for 'behavior:text' cannot give it"),
    ({'behavior:rt': 'likelihood'}, r"not an answer output.*follows 'behavior:probabilities'"),
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


# --- parity with the removed fixed order -------------------------------------

TASKS = {
    'legacy instruction': dict(task_type='probabilities', instruction='Which?', label_set=['cat', 'dog']),
    'legacy fitting': dict(task_type='probabilities', label_set=['cat', 'dog'], fitting_stimuli=FITTING),
    'legacy both': dict(task_type='probabilities', instruction='Which?', label_set=['cat', 'dog'],
                        fitting_stimuli=FITTING),
    'choice': dict(response=Choice(['cat', 'dog'])),
    'choice fitting': dict(response=Choice(['cat', 'dog']), fitting_stimuli=FITTING),
    'probabilities': dict(response=Choice(['cat', 'dog'], probabilities=True), fitting_stimuli=FITTING),
    'probabilities no fitting': dict(response=Choice(['cat', 'dog'], probabilities=True)),
    'rt': dict(response=Choice(['cat', 'dog'], reaction_time='surprisal'), fitting_stimuli=FITTING),
    'raw scores': dict(response=Choice(['cat', 'dog'], raw_scores=True), fitting_stimuli=FITTING),
    'raw text': dict(response=Choice(['cat', 'dog'], raw_text=True, probabilities=True),
                     fitting_stimuli=FITTING),
    'confidence': dict(response=Choice(['cat', 'dog'], confidence=Scale(1, 5)), fitting_stimuli=FITTING),
    'per-row labels': dict(response=Choice('options', probabilities=True), fitting_stimuli=FITTING),
    'scale': dict(response=Scale(1, 7), fitting_stimuli=FITTING),
    'text': dict(response=Text()),
}


def _fixed_order(wired, task):
    """The (path, probability source) the fixed order chose; None when it refused."""
    response, readout = task.response, 'behavioral_readout_layer' in wired
    if response is None:
        can = {'generation': 'answer' in wired and bool(task.instruction),
               'likelihood': False, 'readout': readout and task.fitting_stimuli is not None}
        path = next((p for p in ORDER if can[p]), None)
        return None if path is None else (path, None)
    choice = isinstance(response, Choice)
    confidence, raw_scores = choice and response.confidence is not None, choice and response.raw_scores
    per_row = choice and isinstance(response.labels, str)
    readout = readout and task.fitting_stimuli is not None
    if raw_scores and 'likelihood' not in wired:
        return None
    can = {'generation': not raw_scores and ('reply' in wired or (
               'answer' in wired and choice and not confidence and not response.raw_text)),
           'likelihood': 'likelihood' in wired and choice and not confidence,
           'readout': readout and isinstance(response, (Choice, Scale)) and not confidence
                      and not raw_scores and not per_row}
    path = next((p for p in ORDER if can[p]), None)
    if path is None:
        return None
    source = None
    if choice and (response.probabilities or response.reaction_time or raw_scores):
        source = ('likelihood' if path != 'readout' and 'likelihood' in wired
                  else 'readout' if readout and not per_row else None)
        if source is None:
            return None
    return path, source


COMBINATIONS = [wired for size in range(1, 5) for wired in itertools.combinations(ARGUMENTS, size)]


@pytest.mark.parametrize('wired', COMBINATIONS, ids=['+'.join(w) for w in COMBINATIONS])
def test_the_fixed_order_owners_choose_what_the_fixed_order_chose(monkeypatch, wired):
    model = _model(*wired, owners=fixed_order_owners(*wired))
    monkeypatch.setattr(model._behavioral, 'fit_behavioral_readout', lambda stimuli: None)
    for name, settings in TASKS.items():
        task = TaskContext(**settings)
        expected = _fixed_order(wired, task)
        if expected is None:
            with pytest.raises(ExecutionNotFeasibleError):
                model.start_task(task)
            continue
        model.start_task(task)
        chosen = (model._behavioral.response_path, model._behavioral.probability_source)
        assert chosen == expected, name


def _registrations():
    """Each BrainScoreModel call of a registration: its answer arguments and stated owners."""
    root = Path(brainscore.__file__).parent
    files = sorted((root / 'models').glob('*/model.py')) + [root / 'model_helpers' / 'causal_lm.py']
    found = []
    for path in files:
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
            if isinstance(node, ast.Call) and getattr(node.func, 'id', None) == 'BrainScoreModel':
                keywords = {k.arg: k.value for k in node.keywords if k.arg}
                wired = [name for name in ARGUMENTS if name in keywords
                         and not (isinstance(keywords[name], ast.Constant) and keywords[name].value is None)]
                config = (ast.literal_eval(keywords['capability_config'])
                          if 'capability_config' in keywords else {})
                found.append((path.parent.name, wired, config.get('behavior-sessions', {})))
    return found


def test_every_registration_states_the_owner_the_fixed_order_chose():
    registrations = _registrations()
    assert {'qwen25_vl_3b', 'qwen36_27b', 'blip2_opt_2_7b', 'model_helpers'} <= {r[0] for r in registrations}
    for folder, wired, stated in registrations:
        assert stated == fixed_order_owners(*wired), folder
        _model(*wired, owners=stated)   # construction accepts it
