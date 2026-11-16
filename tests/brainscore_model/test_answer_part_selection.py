"""Which part answers each task form, for every set of answer parts (rule 4.11).

The parts are ``answer`` (A), ``reply`` (P), ``likelihood`` (L) and a readout
layer with an extractor (R). Each part set carries the statement a registrant writes
for it (``STATEMENTS`` of ``test_owner_answers.py``). For each task form, ``EXPECTED``
states literally what happens for the five part sets that registrations use;
``OTHER_PART_SETS`` keeps one cell for each outcome the other ten part sets show:

- ``'<method>:<parts>'``: the session succeeds, the response record names the
  method (for a reaction time, ``link/<method of its probabilities>``), and exactly
  these parts are called (``readout`` is the extractor of
  the readout layer);
- ``'!<refusal>'``: the session is refused with ``ExecutionNotFeasibleError``,
  the message names the part it lacks (``REFUSALS``), and no part is called;
- ``'-'``: the model does not declare the requested channel, so pre-flight
  makes the pair N/A before any session.

A registered part set is built with the statement its registrations carry
(``REGISTERED_STATEMENTS``): the readout-only models state that the readout answers
``behavior:choice`` by argmax.

These are the selection facts the fixed-order table pinned before it was
deleted: the legacy forms (label set absent, empty or given; instruction and
fitting stimuli absent or present), Scale and Text with and without fitting
stimuli, and the Choice forms that remain (shared or per-row labels, plain or
verbatim, choice or probabilities requested, with and without fitting stimuli),
the reaction time of each link, and the log probabilities that replace the old raw
scores (verbatim per-row options; a request that is not verbatim is refused). The
answer and likelihood parts without a reply refuse a verbatim session even when it
requests only the log probabilities: the choice format of the task keeps an owner
that can serve the task (``_choose`` in ``brainscore/_model/behavioral.py``).
"""
import re

import pytest

from brainscore import BrainScoreModel
from brainscore_core.compatibility import ExecutionNotFeasibleError
from brainscore_core.contract import TaskContext
from brainscore_core.responses import Choice, Scale, Text
from brainscore_core.streaming_helpers import stimulus_session
from tests.brainscore_model.stage2_answers_baseline import RATINGS
from tests.brainscore_model.test_answer_ownership import OPTIONS, REGISTERED_PART_SETS, REGISTERED_STATEMENTS, _model
from tests.brainscore_model.test_behavior_paths import FITTING, STIMULI, _features
from tests.brainscore_model.test_owner_answers import STATEMENTS

pytestmark = pytest.mark.unit

PART = {'A': 'answer', 'P': 'reply', 'L': 'likelihood', 'R': 'behavioral_readout_layer'}
CALLED = {'answer': 'answer', 'reply': 'reply', 'likelihood': 'likelihood', 'extract': 'readout'}
# The part sets that registrations use: readout only (CLIP, random ViT, GPT-2), answer and
# readout (BLIP-2), all four (the Qwen family), reply and likelihood (causal LM), answer and
# reply (the API models).
REGISTERED = ('R', 'AR', 'APLR', 'PL', 'AP')


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
    'per-row': r"cannot give a Choice answer with the task settings labels per row\. "
               r"Paths tried: generation needs reply=.*readout needs behavioral_readout_layer=",
    'tokenization': r"gives behavior:log_probabilities only for a verbatim Choice "
                    r"\(Choice\(\.\.\., verbatim=True\)\)",
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
        forms[f'log probabilities fit={fit}'] = (OPTIONS, 'behavior:log_probabilities', TaskContext(
            response=Choice('options', verbatim=True), fitting_stimuli=fitting))
        forms[f'log probabilities not verbatim fit={fit}'] = (
            OPTIONS, 'behavior:log_probabilities', TaskContext(
                instruction='Which?', response=Choice('options'), fitting_stimuli=fitting))
        for link in ('evidence_margin', 'surprisal'):
            forms[f'reaction time {link} fit={fit}'] = (STIMULI, 'behavior:rt', TaskContext(
                instruction='Which?', response=Choice(['cat', 'dog'], reaction_time=link),
                fitting_stimuli=fitting))
    return forms


FORMS = _forms()

EXPECTED = {
    'legacy instr=False labels=None fit=False': {
        'R': '!legacy', 'AR': '!legacy', 'APLR': '!legacy',
        'PL': '-', 'AP': '!legacy',
    },
    'legacy instr=False labels=None fit=True': {
        'R': 'readout:readout', 'AR': 'readout:readout', 'APLR': 'readout:readout',
        'PL': '-', 'AP': '!legacy',
    },
    'legacy instr=False labels=[] fit=False': {
        'R': '!legacy', 'AR': '!legacy', 'APLR': '!legacy',
        'PL': '-', 'AP': '!legacy',
    },
    'legacy instr=False labels=[] fit=True': {
        'R': 'readout:readout', 'AR': 'readout:readout', 'APLR': 'readout:readout',
        'PL': '-', 'AP': '!legacy',
    },
    "legacy instr=False labels=['cat', 'dog'] fit=False": {
        'R': '!legacy', 'AR': '!legacy', 'APLR': '!legacy',
        'PL': '-', 'AP': '!legacy',
    },
    "legacy instr=False labels=['cat', 'dog'] fit=True": {
        'R': 'readout:readout', 'AR': 'readout:readout', 'APLR': 'readout:readout',
        'PL': '-', 'AP': '!legacy',
    },
    'legacy instr=True labels=None fit=False': {
        'R': '!legacy', 'AR': '!legacy', 'APLR': '!legacy',
        'PL': '-', 'AP': '!legacy',
    },
    'legacy instr=True labels=None fit=True': {
        'R': 'readout:readout', 'AR': 'readout:readout', 'APLR': 'readout:readout',
        'PL': '-', 'AP': '!legacy',
    },
    'legacy instr=True labels=[] fit=False': {
        'R': '!legacy', 'AR': '!legacy', 'APLR': '!legacy',
        'PL': '-', 'AP': '!legacy',
    },
    'legacy instr=True labels=[] fit=True': {
        'R': 'readout:readout', 'AR': 'readout:readout', 'APLR': 'readout:readout',
        'PL': '-', 'AP': '!legacy',
    },
    "legacy instr=True labels=['cat', 'dog'] fit=False": {
        'R': '!legacy', 'AR': 'generation:answer', 'APLR': 'generation:answer',
        'PL': '-', 'AP': 'generation:answer',
    },
    "legacy instr=True labels=['cat', 'dog'] fit=True": {
        'R': 'readout:readout', 'AR': 'generation:answer', 'APLR': 'generation:answer',
        'PL': '-', 'AP': 'generation:answer',
    },
    'scale fit=False': {
        'R': '!scale', 'AR': '!scale', 'APLR': 'generation:reply',
        'PL': 'generation:reply', 'AP': 'generation:reply',
    },
    'scale fit=True': {
        'R': 'readout:readout', 'AR': 'readout:readout', 'APLR': 'generation:reply',
        'PL': 'generation:reply', 'AP': 'generation:reply',
    },
    'text fit=False': {
        'R': '-', 'AR': '-', 'APLR': 'generation:reply',
        'PL': 'generation:reply', 'AP': 'generation:reply',
    },
    'plain choice fit=False': {
        'R': '!choice', 'AR': 'generation:answer', 'APLR': 'generation:answer',
        'PL': 'generation:reply', 'AP': 'generation:answer',
    },
    'plain probabilities fit=False': {
        'R': '!choice', 'AR': '!implied', 'APLR': 'likelihood:likelihood',
        'PL': 'likelihood:likelihood', 'AP': '-',
    },
    'per-row choice fit=False': {
        'R': '!per-row', 'AR': 'generation:answer', 'APLR': 'generation:reply',
        'PL': 'generation:reply', 'AP': 'generation:reply',
    },
    'per-row verbatim choice fit=False': {
        'R': '!verbatim', 'AR': '!verbatim', 'APLR': 'generation:reply',
        'PL': 'generation:reply', 'AP': 'generation:reply',
    },
    'verbatim choice fit=False': {
        'R': '!verbatim', 'AR': '!verbatim', 'APLR': 'generation:reply',
        'PL': 'generation:reply', 'AP': 'generation:reply',
    },
    'text fit=True': {
        'R': '-', 'AR': '-', 'APLR': 'generation:reply',
        'PL': 'generation:reply', 'AP': 'generation:reply',
    },
    'plain choice fit=True': {
        'R': 'readout:readout', 'AR': 'generation:answer', 'APLR': 'generation:answer',
        'PL': 'generation:reply', 'AP': 'generation:answer',
    },
    'plain probabilities fit=True': {
        'R': 'readout:readout', 'AR': 'readout:readout', 'APLR': 'likelihood:likelihood',
        'PL': 'likelihood:likelihood', 'AP': '-',
    },
    'per-row choice fit=True': {
        'R': '!per-row', 'AR': 'generation:answer', 'APLR': 'generation:reply',
        'PL': 'generation:reply', 'AP': 'generation:reply',
    },
    'per-row verbatim choice fit=True': {
        'R': '!verbatim', 'AR': '!verbatim', 'APLR': 'generation:reply',
        'PL': 'generation:reply', 'AP': 'generation:reply',
    },
    'verbatim choice fit=True': {
        'R': 'readout:readout', 'AR': '!owner', 'APLR': 'generation:reply',
        'PL': 'generation:reply', 'AP': 'generation:reply',
    },
    'reaction time evidence_margin fit=False': {
        'R': '!choice', 'AR': '!implied', 'APLR': 'link/likelihood:likelihood',
        'PL': 'link/likelihood:likelihood', 'AP': '-',
    },
    'reaction time evidence_margin fit=True': {
        'R': 'link/readout:readout', 'AR': 'link/readout:readout', 'APLR': 'link/likelihood:likelihood',
        'PL': 'link/likelihood:likelihood', 'AP': '-',
    },
    'reaction time surprisal fit=False': {
        'R': '!choice', 'AR': '!implied', 'APLR': 'link/likelihood:likelihood',
        'PL': 'link/likelihood:likelihood', 'AP': '-',
    },
    'reaction time surprisal fit=True': {
        'R': 'link/readout:readout', 'AR': 'link/readout:readout', 'APLR': 'link/likelihood:likelihood',
        'PL': 'link/likelihood:likelihood', 'AP': '-',
    },
    'log probabilities fit=False': {
        'R': '-', 'AR': '-', 'APLR': 'likelihood:likelihood',
        'PL': 'likelihood:likelihood', 'AP': '-',
    },
    'log probabilities fit=True': {
        'R': '-', 'AR': '-', 'APLR': 'likelihood:likelihood',
        'PL': 'likelihood:likelihood', 'AP': '-',
    },
    'log probabilities not verbatim fit=False': {
        'R': '-', 'AR': '-', 'APLR': '!tokenization',
        'PL': '!tokenization', 'AP': '-',
    },
    'log probabilities not verbatim fit=True': {
        'R': '-', 'AR': '-', 'APLR': '!tokenization',
        'PL': '!tokenization', 'AP': '-',
    },
}

# One cell for each outcome that the part sets no registration uses show: they keep each
# outcome of the selection (a refusal, a method and the parts it calls) under test.
OTHER_PART_SETS = {
    ('plain probabilities fit=False', 'PR'): '!implied',
    ('legacy instr=False labels=None fit=False', 'A'): '!legacy',
    ('per-row verbatim choice fit=False', 'AL'): '!owner',
    ('scale fit=False', 'LR'): '!scale',
    ('log probabilities not verbatim fit=False', 'L'): '!tokenization',
    ('per-row verbatim choice fit=False', 'A'): '!verbatim',
    ('legacy instr=False labels=None fit=False', 'P'): '-',
    ("legacy instr=True labels=['cat', 'dog'] fit=False", 'A'): 'generation:answer',
    ('scale fit=False', 'P'): 'generation:reply',
    ('plain probabilities fit=False', 'AL'): 'likelihood:likelihood',
    ('plain choice fit=False', 'LR'): 'likelihood:likelihood',
    ('plain probabilities fit=False', 'PLR'): 'likelihood:likelihood',
    ('reaction time evidence_margin fit=False', 'AL'): 'link/likelihood:likelihood',
    ('reaction time evidence_margin fit=False', 'LR'): 'link/likelihood:likelihood',
    ('reaction time evidence_margin fit=False', 'PLR'): 'link/likelihood:likelihood',
    ('reaction time evidence_margin fit=True', 'APR'): 'link/readout:readout',
    ('reaction time evidence_margin fit=True', 'PR'): 'link/readout:readout',
    ('plain probabilities fit=True', 'APR'): 'readout:readout',
    ('legacy instr=False labels=None fit=True', 'PR'): 'readout:readout',
    ('plain probabilities fit=True', 'PR'): 'readout:readout',
}


def test_every_task_form_has_a_cell_for_every_registered_part_set():
    """A new task form adds one cell for each of the five part sets that registrations use."""
    assert set(EXPECTED) == set(FORMS)
    for form, expectations in EXPECTED.items():
        assert {tuple(PART[letter] for letter in key) for key in expectations} == REGISTERED_PART_SETS, form
    assert {form for form, _ in OTHER_PART_SETS} <= set(FORMS)
    assert not {key for _, key in OTHER_PART_SETS} & set(REGISTERED)


CELLS = {(form, key): outcome for form, cells in EXPECTED.items() for key, outcome in cells.items()}
CELLS.update(OTHER_PART_SETS)


@pytest.mark.parametrize('form, key', sorted(CELLS))
def test_the_stated_part_answers_or_the_stated_refusal_is_raised(form, key):
    parts = tuple(PART[letter] for letter in key)
    stimuli, channel, task = FORMS[form]
    expected = CELLS[form, key]
    calls = []
    # A registered part set states what its registrations state; another, the statement of STATEMENTS.
    owners = REGISTERED_STATEMENTS[parts] if key in REGISTERED else STATEMENTS[parts]
    model = _model(*parts, owners=owners, calls=calls)
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
