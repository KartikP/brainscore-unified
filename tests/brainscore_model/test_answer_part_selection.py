"""Which part answers each task form, for every set of answer parts (rule 4.11).

The parts are ``answer`` (A), ``reply`` (P), ``likelihood`` (L) and a readout
layer with an extractor (R). Each of the 15 part sets carries the statement a
registrant writes for it (``STATEMENTS`` of ``test_owner_answers.py``). For each
task form, ``EXPECTED`` states literally what happens:

- ``'<method>:<parts>'``: the session succeeds, the response record names the
  method (for a reaction time, ``link/<method of its probabilities>``), and exactly
  these parts are called (``readout`` is the extractor of
  the readout layer);
- ``'!<refusal>'``: the session is refused with ``ExecutionNotFeasibleError``,
  the message names the part it lacks (``REFUSALS``), and no part is called;
- ``'-'``: the model does not declare the requested channel, so pre-flight
  makes the pair N/A before any session.

These are the selection facts the fixed-order table pinned before it was
deleted: the legacy forms (label set absent, empty or given; instruction and
fitting stimuli absent or present), Scale and Text with and without fitting
stimuli, and the Choice forms that remain (shared or per-row labels, plain or
verbatim, choice or probabilities requested, with and without fitting stimuli).
"""
import re

import pytest

from brainscore import BrainScoreModel
from brainscore_core.compatibility import ExecutionNotFeasibleError
from brainscore_core.contract import TaskContext
from brainscore_core.responses import Choice, Scale, Text
from brainscore_core.streaming_helpers import stimulus_session
from tests.brainscore_model.stage2_answers_baseline import RATINGS
from tests.brainscore_model.test_answer_ownership import OPTIONS, _model
from tests.brainscore_model.test_behavior_paths import FITTING, STIMULI, _features
from tests.brainscore_model.test_owner_answers import STATEMENTS

pytestmark = pytest.mark.unit

PART = {'A': 'answer', 'P': 'reply', 'L': 'likelihood', 'R': 'behavioral_readout_layer'}
CALLED = {'answer': 'answer', 'reply': 'reply', 'likelihood': 'likelihood', 'extract': 'readout'}


# --- an answer part, a reply part and a likelihood part, no readout: a Scale -----------

@pytest.mark.parametrize('fitting', [None, RATINGS], ids=['without fitting stimuli', 'with fitting stimuli'])
def test_answer_reply_and_likelihood_without_a_readout_answer_a_scale_through_the_reply(fitting):
    """The reply gives the rating (generation); the answer and likelihood parts are not called."""
    calls = []
    model = BrainScoreModel(
        'scale-parts', extractor={'text': _features}, in_channels={'text'},
        answer=lambda stimulus_row, instruction, label_set: calls.append('answer') or label_set[0],
        reply=lambda prompt: calls.append('reply') or '3',
        likelihood=lambda prompt, labels: calls.append('likelihood') or {label: 0.0 for label in labels},
        capability_config={'behavior-sessions': {'behavior:probabilities': ('likelihood', 'softmax')}})
    assert 'behavior:rating' in model.out_channels
    session = stimulus_session(STIMULI, ['behavior:rating'],
                               TaskContext(instruction='How much?', response=Scale(1, 7), fitting_stimuli=fitting))
    model.participate(session)
    rating = session.collect('behavior:rating')
    assert rating.attrs['response']['method'] == 'generation'
    assert calls == ['reply', 'reply', 'reply']
    assert [float(value) for value in rating.values] == [3.0, 3.0, 3.0]


# --- every part set on every task form ----------------------------------------------------

REFUSALS = {
    'legacy': r"cannot run the behavioral task task_type='probabilities'\. Paths tried: "
              r"generation needs answer=, TaskContext\.instruction and TaskContext\.label_set; .*"
              r"readout needs TaskContext\.fitting_stimuli \(and behavioral_readout_layer=\)",
    'scale': r"cannot give a Scale answer\. Paths tried: generation needs reply= .*"
             r"readout needs behavioral_readout_layer=, an extractor and fitting_stimuli",
    'choice': r"cannot give a Choice answer\. Paths tried: generation needs reply= .*"
              r"readout needs behavioral_readout_layer=, an extractor and fitting_stimuli",
    'verbatim': r"cannot give a Choice answer with the task settings verbatim(, labels per row)?\. "
                r"Paths tried: generation needs reply=",
    'owner': r"'behavior:choice' is owned by 'generation', which cannot serve this task "
             r"\(task settings: verbatim(, labels per row)?\); no other part answers it",
    'implied': r"cannot give implied probabilities: it needs likelihood= or a readout "
               r"\(behavioral_readout_layer=, fitting_stimuli\)",
}


def _forms():
    """(stimuli, requested channel, task) of each task form."""
    forms = {}
    for instruction in (None, 'Which?'):
        for labels in (None, [], ['cat', 'dog']):
            for fitting in (None, FITTING):
                forms[f'legacy instr={instruction is not None} labels={labels} fit={fitting is not None}'] = (
                    STIMULI, 'behavior', TaskContext(task_type='probabilities', label_set=labels,
                                                     instruction=instruction, fitting_stimuli=fitting))
    for fitting in (None, RATINGS):
        forms[f'scale fit={fitting is not None}'] = (
            STIMULI, 'behavior:rating',
            TaskContext(instruction='How much?', response=Scale(1, 7), fitting_stimuli=fitting))
    for fitting in (None, FITTING):
        fit = fitting is not None
        forms[f'text fit={fit}'] = (STIMULI, 'behavior:text', TaskContext(
            instruction='Say it.', response=Text(), fitting_stimuli=fitting))
        forms[f'plain choice fit={fit}'] = (STIMULI, 'behavior:choice', TaskContext(
            instruction='Which?', response=Choice(['cat', 'dog']), fitting_stimuli=fitting))
        forms[f'plain probabilities fit={fit}'] = (STIMULI, 'behavior:probabilities', TaskContext(
            instruction='Which?', response=Choice(['cat', 'dog']), fitting_stimuli=fitting))
        forms[f'per-row choice fit={fit}'] = (OPTIONS, 'behavior:choice', TaskContext(
            instruction='Which?', response=Choice('options'), fitting_stimuli=fitting))
        forms[f'per-row verbatim choice fit={fit}'] = (OPTIONS, 'behavior:choice', TaskContext(
            response=Choice('options', verbatim=True), fitting_stimuli=fitting))
        forms[f'verbatim choice fit={fit}'] = (STIMULI, 'behavior:choice', TaskContext(
            response=Choice(['cat', 'dog'], verbatim=True), fitting_stimuli=fitting))
        for link in ('evidence_margin', 'surprisal'):
            forms[f'reaction time {link} fit={fit}'] = (STIMULI, 'behavior:rt', TaskContext(
                instruction='Which?', response=Choice(['cat', 'dog'], reaction_time=link),
                fitting_stimuli=fitting))
    return forms


FORMS = _forms()

EXPECTED = {
    'legacy instr=False labels=None fit=False': {
        'A': '!legacy', 'P': '-', 'L': '-', 'R': '!legacy',
        'AP': '!legacy', 'AL': '!legacy', 'AR': '!legacy', 'PL': '-',
        'PR': '!legacy', 'LR': '!legacy', 'APL': '!legacy', 'APR': '!legacy',
        'ALR': '!legacy', 'PLR': '!legacy', 'APLR': '!legacy',
    },
    'legacy instr=False labels=None fit=True': {
        'A': '!legacy', 'P': '-', 'L': '-', 'R': 'readout:readout',
        'AP': '!legacy', 'AL': '!legacy', 'AR': 'readout:readout', 'PL': '-',
        'PR': 'readout:readout', 'LR': 'readout:readout', 'APL': '!legacy', 'APR': 'readout:readout',
        'ALR': 'readout:readout', 'PLR': 'readout:readout', 'APLR': 'readout:readout',
    },
    'legacy instr=False labels=[] fit=False': {
        'A': '!legacy', 'P': '-', 'L': '-', 'R': '!legacy',
        'AP': '!legacy', 'AL': '!legacy', 'AR': '!legacy', 'PL': '-',
        'PR': '!legacy', 'LR': '!legacy', 'APL': '!legacy', 'APR': '!legacy',
        'ALR': '!legacy', 'PLR': '!legacy', 'APLR': '!legacy',
    },
    'legacy instr=False labels=[] fit=True': {
        'A': '!legacy', 'P': '-', 'L': '-', 'R': 'readout:readout',
        'AP': '!legacy', 'AL': '!legacy', 'AR': 'readout:readout', 'PL': '-',
        'PR': 'readout:readout', 'LR': 'readout:readout', 'APL': '!legacy', 'APR': 'readout:readout',
        'ALR': 'readout:readout', 'PLR': 'readout:readout', 'APLR': 'readout:readout',
    },
    "legacy instr=False labels=['cat', 'dog'] fit=False": {
        'A': '!legacy', 'P': '-', 'L': '-', 'R': '!legacy',
        'AP': '!legacy', 'AL': '!legacy', 'AR': '!legacy', 'PL': '-',
        'PR': '!legacy', 'LR': '!legacy', 'APL': '!legacy', 'APR': '!legacy',
        'ALR': '!legacy', 'PLR': '!legacy', 'APLR': '!legacy',
    },
    "legacy instr=False labels=['cat', 'dog'] fit=True": {
        'A': '!legacy', 'P': '-', 'L': '-', 'R': 'readout:readout',
        'AP': '!legacy', 'AL': '!legacy', 'AR': 'readout:readout', 'PL': '-',
        'PR': 'readout:readout', 'LR': 'readout:readout', 'APL': '!legacy', 'APR': 'readout:readout',
        'ALR': 'readout:readout', 'PLR': 'readout:readout', 'APLR': 'readout:readout',
    },
    'legacy instr=True labels=None fit=False': {
        'A': '!legacy', 'P': '-', 'L': '-', 'R': '!legacy',
        'AP': '!legacy', 'AL': '!legacy', 'AR': '!legacy', 'PL': '-',
        'PR': '!legacy', 'LR': '!legacy', 'APL': '!legacy', 'APR': '!legacy',
        'ALR': '!legacy', 'PLR': '!legacy', 'APLR': '!legacy',
    },
    'legacy instr=True labels=None fit=True': {
        'A': '!legacy', 'P': '-', 'L': '-', 'R': 'readout:readout',
        'AP': '!legacy', 'AL': '!legacy', 'AR': 'readout:readout', 'PL': '-',
        'PR': 'readout:readout', 'LR': 'readout:readout', 'APL': '!legacy', 'APR': 'readout:readout',
        'ALR': 'readout:readout', 'PLR': 'readout:readout', 'APLR': 'readout:readout',
    },
    'legacy instr=True labels=[] fit=False': {
        'A': '!legacy', 'P': '-', 'L': '-', 'R': '!legacy',
        'AP': '!legacy', 'AL': '!legacy', 'AR': '!legacy', 'PL': '-',
        'PR': '!legacy', 'LR': '!legacy', 'APL': '!legacy', 'APR': '!legacy',
        'ALR': '!legacy', 'PLR': '!legacy', 'APLR': '!legacy',
    },
    'legacy instr=True labels=[] fit=True': {
        'A': '!legacy', 'P': '-', 'L': '-', 'R': 'readout:readout',
        'AP': '!legacy', 'AL': '!legacy', 'AR': 'readout:readout', 'PL': '-',
        'PR': 'readout:readout', 'LR': 'readout:readout', 'APL': '!legacy', 'APR': 'readout:readout',
        'ALR': 'readout:readout', 'PLR': 'readout:readout', 'APLR': 'readout:readout',
    },
    "legacy instr=True labels=['cat', 'dog'] fit=False": {
        'A': 'generation:answer', 'P': '-', 'L': '-', 'R': '!legacy',
        'AP': 'generation:answer', 'AL': 'generation:answer', 'AR': 'generation:answer', 'PL': '-',
        'PR': '!legacy', 'LR': '!legacy', 'APL': 'generation:answer', 'APR': 'generation:answer',
        'ALR': 'generation:answer', 'PLR': '!legacy', 'APLR': 'generation:answer',
    },
    "legacy instr=True labels=['cat', 'dog'] fit=True": {
        'A': 'generation:answer', 'P': '-', 'L': '-', 'R': 'readout:readout',
        'AP': 'generation:answer', 'AL': 'generation:answer', 'AR': 'generation:answer', 'PL': '-',
        'PR': 'readout:readout', 'LR': 'readout:readout', 'APL': 'generation:answer', 'APR': 'generation:answer',
        'ALR': 'generation:answer', 'PLR': 'readout:readout', 'APLR': 'generation:answer',
    },
    'scale fit=False': {
        'A': '-', 'P': 'generation:reply', 'L': '-', 'R': '!scale',
        'AP': 'generation:reply', 'AL': '-', 'AR': '!scale', 'PL': 'generation:reply',
        'PR': 'generation:reply', 'LR': '!scale', 'APL': 'generation:reply', 'APR': 'generation:reply',
        'ALR': '!scale', 'PLR': 'generation:reply', 'APLR': 'generation:reply',
    },
    'scale fit=True': {
        'A': '-', 'P': 'generation:reply', 'L': '-', 'R': 'readout:readout',
        'AP': 'generation:reply', 'AL': '-', 'AR': 'readout:readout', 'PL': 'generation:reply',
        'PR': 'generation:reply', 'LR': 'readout:readout', 'APL': 'generation:reply', 'APR': 'generation:reply',
        'ALR': 'readout:readout', 'PLR': 'generation:reply', 'APLR': 'generation:reply',
    },
    'text fit=False': {
        'A': '-', 'P': 'generation:reply', 'L': '-', 'R': '-',
        'AP': 'generation:reply', 'AL': '-', 'AR': '-', 'PL': 'generation:reply',
        'PR': 'generation:reply', 'LR': '-', 'APL': 'generation:reply', 'APR': 'generation:reply',
        'ALR': '-', 'PLR': 'generation:reply', 'APLR': 'generation:reply',
    },
    'plain choice fit=False': {
        'A': 'generation:answer', 'P': 'generation:reply', 'L': '-', 'R': '-',
        'AP': 'generation:answer', 'AL': 'generation:answer', 'AR': 'generation:answer', 'PL': 'generation:reply',
        'PR': 'generation:reply', 'LR': 'likelihood:likelihood', 'APL': 'generation:answer', 'APR': 'generation:answer',
        'ALR': 'generation:answer', 'PLR': 'generation:reply', 'APLR': 'generation:answer',
    },
    'plain probabilities fit=False': {
        'A': '-', 'P': '-', 'L': '-', 'R': '!choice',
        'AP': '-', 'AL': 'likelihood:answer+likelihood', 'AR': '!implied', 'PL': 'likelihood:likelihood+reply',
        'PR': '!implied', 'LR': 'likelihood:likelihood', 'APL': 'likelihood:answer+likelihood', 'APR': '!implied',
        'ALR': 'likelihood:answer+likelihood', 'PLR': 'likelihood:likelihood+reply', 'APLR': 'likelihood:answer+likelihood',
    },
    'per-row choice fit=False': {
        'A': 'generation:answer', 'P': 'generation:reply', 'L': '-', 'R': '-',
        'AP': 'generation:reply', 'AL': 'generation:answer', 'AR': 'generation:answer', 'PL': 'generation:reply',
        'PR': 'generation:reply', 'LR': 'likelihood:likelihood', 'APL': 'generation:reply', 'APR': 'generation:reply',
        'ALR': 'generation:answer', 'PLR': 'generation:reply', 'APLR': 'generation:reply',
    },
    'per-row verbatim choice fit=False': {
        'A': '!verbatim', 'P': 'generation:reply', 'L': '-', 'R': '-',
        'AP': 'generation:reply', 'AL': '!owner', 'AR': '!verbatim', 'PL': 'generation:reply',
        'PR': 'generation:reply', 'LR': 'likelihood:likelihood', 'APL': 'generation:reply', 'APR': 'generation:reply',
        'ALR': '!owner', 'PLR': 'generation:reply', 'APLR': 'generation:reply',
    },
    'verbatim choice fit=False': {
        'A': '!verbatim', 'P': 'generation:reply', 'L': '-', 'R': '-',
        'AP': 'generation:reply', 'AL': '!owner', 'AR': '!verbatim', 'PL': 'generation:reply',
        'PR': 'generation:reply', 'LR': 'likelihood:likelihood', 'APL': 'generation:reply', 'APR': 'generation:reply',
        'ALR': '!owner', 'PLR': 'generation:reply', 'APLR': 'generation:reply',
    },
    'text fit=True': {
        'A': '-', 'P': 'generation:reply', 'L': '-', 'R': '-',
        'AP': 'generation:reply', 'AL': '-', 'AR': '-', 'PL': 'generation:reply',
        'PR': 'generation:reply', 'LR': '-', 'APL': 'generation:reply', 'APR': 'generation:reply',
        'ALR': '-', 'PLR': 'generation:reply', 'APLR': 'generation:reply',
    },
    'plain choice fit=True': {
        'A': 'generation:answer', 'P': 'generation:reply', 'L': '-', 'R': '-',
        'AP': 'generation:answer', 'AL': 'generation:answer', 'AR': 'generation:answer', 'PL': 'generation:reply',
        'PR': 'generation:reply', 'LR': 'likelihood:likelihood', 'APL': 'generation:answer', 'APR': 'generation:answer',
        'ALR': 'generation:answer', 'PLR': 'generation:reply', 'APLR': 'generation:answer',
    },
    'plain probabilities fit=True': {
        'A': '-', 'P': '-', 'L': '-', 'R': 'readout:readout',
        'AP': '-', 'AL': 'likelihood:answer+likelihood', 'AR': 'readout:answer+readout', 'PL': 'likelihood:likelihood+reply',
        'PR': 'readout:readout+reply', 'LR': 'likelihood:likelihood', 'APL': 'likelihood:answer+likelihood', 'APR': 'readout:answer+readout',
        'ALR': 'likelihood:answer+likelihood', 'PLR': 'likelihood:likelihood+reply', 'APLR': 'likelihood:answer+likelihood',
    },
    'per-row choice fit=True': {
        'A': 'generation:answer', 'P': 'generation:reply', 'L': '-', 'R': '-',
        'AP': 'generation:reply', 'AL': 'generation:answer', 'AR': 'generation:answer', 'PL': 'generation:reply',
        'PR': 'generation:reply', 'LR': 'likelihood:likelihood', 'APL': 'generation:reply', 'APR': 'generation:reply',
        'ALR': 'generation:answer', 'PLR': 'generation:reply', 'APLR': 'generation:reply',
    },
    'per-row verbatim choice fit=True': {
        'A': '!verbatim', 'P': 'generation:reply', 'L': '-', 'R': '-',
        'AP': 'generation:reply', 'AL': '!owner', 'AR': '!verbatim', 'PL': 'generation:reply',
        'PR': 'generation:reply', 'LR': 'likelihood:likelihood', 'APL': 'generation:reply', 'APR': 'generation:reply',
        'ALR': '!owner', 'PLR': 'generation:reply', 'APLR': 'generation:reply',
    },
    'verbatim choice fit=True': {
        'A': '!verbatim', 'P': 'generation:reply', 'L': '-', 'R': '-',
        'AP': 'generation:reply', 'AL': '!owner', 'AR': '!owner', 'PL': 'generation:reply',
        'PR': 'generation:reply', 'LR': 'likelihood:likelihood', 'APL': 'generation:reply', 'APR': 'generation:reply',
        'ALR': '!owner', 'PLR': 'generation:reply', 'APLR': 'generation:reply',
    },
    'reaction time evidence_margin fit=False': {
        'A': '-', 'P': '-', 'L': '-', 'R': '!choice',
        'AP': '-', 'AL': 'link/likelihood:answer+likelihood', 'AR': '!implied', 'PL': 'link/likelihood:likelihood+reply',
        'PR': '!implied', 'LR': 'link/likelihood:likelihood', 'APL': 'link/likelihood:answer+likelihood', 'APR': '!implied',
        'ALR': 'link/likelihood:answer+likelihood', 'PLR': 'link/likelihood:likelihood+reply', 'APLR': 'link/likelihood:answer+likelihood',
    },
    'reaction time evidence_margin fit=True': {
        'A': '-', 'P': '-', 'L': '-', 'R': 'link/readout:readout',
        'AP': '-', 'AL': 'link/likelihood:answer+likelihood', 'AR': 'link/readout:answer+readout', 'PL': 'link/likelihood:likelihood+reply',
        'PR': 'link/readout:readout+reply', 'LR': 'link/likelihood:likelihood', 'APL': 'link/likelihood:answer+likelihood', 'APR': 'link/readout:answer+readout',
        'ALR': 'link/likelihood:answer+likelihood', 'PLR': 'link/likelihood:likelihood+reply', 'APLR': 'link/likelihood:answer+likelihood',
    },
    'reaction time surprisal fit=False': {
        'A': '-', 'P': '-', 'L': '-', 'R': '!choice',
        'AP': '-', 'AL': 'link/likelihood:answer+likelihood', 'AR': '!implied', 'PL': 'link/likelihood:likelihood+reply',
        'PR': '!implied', 'LR': 'link/likelihood:likelihood', 'APL': 'link/likelihood:answer+likelihood', 'APR': '!implied',
        'ALR': 'link/likelihood:answer+likelihood', 'PLR': 'link/likelihood:likelihood+reply', 'APLR': 'link/likelihood:answer+likelihood',
    },
    'reaction time surprisal fit=True': {
        'A': '-', 'P': '-', 'L': '-', 'R': 'link/readout:readout',
        'AP': '-', 'AL': 'link/likelihood:answer+likelihood', 'AR': 'link/readout:answer+readout', 'PL': 'link/likelihood:likelihood+reply',
        'PR': 'link/readout:readout+reply', 'LR': 'link/likelihood:likelihood', 'APL': 'link/likelihood:answer+likelihood', 'APR': 'link/readout:answer+readout',
        'ALR': 'link/likelihood:answer+likelihood', 'PLR': 'link/likelihood:likelihood+reply', 'APLR': 'link/likelihood:answer+likelihood',
    },
}


def test_every_task_form_has_an_expectation_for_every_part_set():
    assert set(EXPECTED) == set(FORMS)
    for expectations in EXPECTED.values():
        assert {tuple(PART[letter] for letter in key) for key in expectations} == set(STATEMENTS)


@pytest.mark.parametrize('form', sorted(FORMS))
@pytest.mark.parametrize('key', ['A', 'P', 'L', 'R', 'AP', 'AL', 'AR', 'PL', 'PR', 'LR',
                                 'APL', 'APR', 'ALR', 'PLR', 'APLR'])
def test_the_stated_part_answers_or_the_stated_refusal_is_raised(form, key):
    parts = tuple(PART[letter] for letter in key)
    stimuli, channel, task = FORMS[form]
    expected = EXPECTED[form][key]
    calls = []
    model = _model(*parts, owners=STATEMENTS[parts], calls=calls)
    if expected == '-':
        assert channel not in model.out_channels
        return
    assert channel in model.out_channels
    session = stimulus_session(stimuli, [channel], task)
    if expected.startswith('!'):
        with pytest.raises(ExecutionNotFeasibleError) as refusal:
            model.participate(session)
        assert re.search(REFUSALS[expected[1:]], str(refusal.value)), str(refusal.value)
        assert calls == []
        return
    method, called = expected.split(':')
    model.participate(session)
    record = session.collect(channel).attrs['response']
    if channel == 'behavior:rt':
        assert ('link/' + record['parameters']['source_method']) == method
    else:
        assert record['method'] == method
    assert sorted({CALLED[call] for call in calls}) == called.split('+')
