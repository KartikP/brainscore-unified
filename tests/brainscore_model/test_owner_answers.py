"""The owner of the requested answer format gives the answer (rule 4.11).

For a task with a response, the part that answers is the owner of the answer
format: the log probabilities (``behavior:log_probabilities``) when the session
requests them, else the response kind's own format. A task form its owner cannot serve is
refused before any compute; only bare ``behavior`` (a legacy task) keeps the
fallback to the one other part that can serve it.
"""
from contextlib import contextmanager

import numpy as np
import pytest

from brainscore import BrainScoreModel
from brainscore.selection import LayerSelector
from brainscore_core.compatibility import ExecutionNotFeasibleError
from brainscore_core.contract import TaskContext
from brainscore_core.responses import Choice
from brainscore_core.streaming_helpers import stimulus_session
from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
from tests.brainscore_model.test_answer_ownership import (
    BLIP2, CAUSAL_LM, QWEN, REGISTERED_STATEMENTS, _model, _run)
from tests.brainscore_model.stage2_answers_baseline import RATINGS, SESSIONS, task_from
from tests.brainscore_model.test_behavior_paths import FITTING, STIMULI

pytestmark = pytest.mark.unit

VERBATIM = TaskContext(response=Choice(['cat', 'dog'], verbatim=True))


@pytest.mark.parametrize('outputs', [['behavior:log_probabilities'],
                                     ['behavior:log_probabilities', 'behavior:probabilities'],
                                     ['behavior:log_probabilities', 'behavior:probabilities',
                                      'behavior:choice']],
                         ids=['log probabilities', 'with probabilities', 'with choice'])
@pytest.mark.parametrize('parts', [QWEN, CAUSAL_LM], ids=['qwen', 'causal lm'])
def test_log_probabilities_come_from_one_likelihood_pass(parts, outputs):
    """The likelihood answers a session that requests log probabilities; the
    probabilities and the choice come from that same pass (one call per trial)."""
    calls = []
    model = _model(*parts, owners=REGISTERED_STATEMENTS[parts], calls=calls)
    session = _run(model, outputs, VERBATIM)
    record = session.collect('behavior:log_probabilities').attrs['response']
    assert (record['method'], record['category']) == ('likelihood', 'model-derived')
    assert calls == ['likelihood'] * len(STIMULI)          # neither reply nor answer, one pass
    for channel in outputs[1:]:
        assert session.collect(channel).attrs['response']['source_format'] == 'behavior:log_probabilities'


def test_blip2_verbatim_choice_is_refused_before_compute():
    calls = []
    model = _model(*BLIP2, owners=REGISTERED_STATEMENTS[BLIP2], calls=calls)
    task = TaskContext(response=Choice(['cat', 'dog'], verbatim=True), fitting_stimuli=FITTING)
    with pytest.raises(ExecutionNotFeasibleError,
                       match=r"'behavior:choice' is owned by 'generation'.*task settings: verbatim"):
        _run(model, ['behavior:choice'], task)
    assert calls.count('extract') == 0 and calls == []


def test_a_readout_alone_gives_probabilities_natively():
    model = _model('behavioral_readout_layer',
                   owners=REGISTERED_STATEMENTS[('behavioral_readout_layer',)])
    task = TaskContext(instruction='Which?', response=Choice(['cat', 'dog']),
                       fitting_stimuli=FITTING)
    probabilities = _run(model, ['behavior:probabilities'], task).collect('behavior:probabilities')
    assert probabilities.values.shape == (len(STIMULI), 2)
    record = probabilities.attrs['response']
    assert (record['method'], record['origin']) == ('readout', 'native')


def test_two_owners_for_choice_and_probabilities_are_refused_before_compute():
    calls = []
    owners = {'behavior:rating': 'generation',
              'behavior:choice': ('likelihood', 'softmax_argmax'), 'behavior:probabilities': 'readout'}
    model = _model(*QWEN, owners=owners, calls=calls)
    task = TaskContext(instruction='Which?', response=Choice(['cat', 'dog']),
                       fitting_stimuli=FITTING)
    with pytest.raises(ExecutionNotFeasibleError,
                       match=r"choices from its likelihood part and probabilities from its readout part"):
        _run(model, ['behavior:choice', 'behavior:probabilities'], task)
    assert calls == []


# --- no part answers a format channel it does not own ---------------------------------

G, L = 'generation', ('likelihood', 'softmax')
# Every part set with the statement a registrant writes for it.
STATEMENTS = {
    ('answer',): {}, ('reply',): {}, ('likelihood',): {}, ('behavioral_readout_layer',): {},
    ('answer', 'reply'): {}, ('answer', 'likelihood'): {'behavior:probabilities': L},
    ('reply', 'likelihood'): {'behavior:probabilities': L},
    ('answer', 'reply', 'likelihood'): {'behavior:probabilities': L},
    ('answer', 'behavioral_readout_layer'): {},
    ('reply', 'behavioral_readout_layer'): {'behavior:rating': G},
    ('likelihood', 'behavioral_readout_layer'): {'behavior:choice': ('likelihood', 'softmax_argmax'),
                                                 'behavior:probabilities': L},
    ('answer', 'reply', 'behavioral_readout_layer'): {'behavior:rating': G},
    ('answer', 'likelihood', 'behavioral_readout_layer'): {'behavior:probabilities': L},
    ('reply', 'likelihood', 'behavioral_readout_layer'): {'behavior:rating': G, 'behavior:probabilities': L},
    ('answer', 'reply', 'likelihood', 'behavioral_readout_layer'): {
        'behavior:rating': G, 'behavior:probabilities': L},
}


# The sessions of SESSIONS each part set refuses (ExecutionNotFeasibleError); every
# other session with a declared output must succeed. A verbatim Choice needs a reply, or
# a likelihood that owns the choice when log probabilities are requested.
REFUSED_SESSIONS = {
    ('answer',): {'log probabilities'},
    ('answer', 'likelihood'): {'log probabilities'},
    ('answer', 'behavioral_readout_layer'): {'log probabilities'},
    ('answer', 'likelihood', 'behavioral_readout_layer'): {'log probabilities'},
}


@pytest.mark.parametrize('parts', sorted(STATEMENTS), ids='+'.join)
def test_each_answered_format_comes_from_its_owner(parts):
    """Every session of SESSIONS on every part set: each answered format channel was given
    by the part that owns it (the probabilities' owner also gives the choice that comes
    with log probabilities; a reaction time names the probabilities' method). A session
    is refused exactly when ``REFUSED_SESSIONS`` says so."""
    model = _model(*parts, owners=STATEMENTS[parts])
    owners = model._behavioral.owners
    for name, (description, outputs) in SESSIONS.items():
        task = task_from(description, RATINGS if description['kind'] == 'scale' else FITTING)
        requested = [c for c in outputs if c in model.out_channels]
        if not requested:
            continue
        if name in REFUSED_SESSIONS.get(parts, set()):
            with pytest.raises(ExecutionNotFeasibleError):
                _run(model, requested, task)
            continue
        session = _run(model, requested, task)
        for channel in requested:
            if channel == 'behavior':
                continue                  # bare behavior: the copy of the first format
            record = session.collect(channel).attrs['response']
            owner = (owners['behavior:log_probabilities'] if 'behavior:log_probabilities' in requested
                     else owners[channel])
            assert record['method'] == owner, (parts, name, channel)


# --- one format of a crossed likelihood/readout pair ----------------------------------

CROSSED = {
    'likelihood choice': {'behavior:choice': ('likelihood', 'softmax_argmax'),
                          'behavior:probabilities': 'readout'},
    'readout choice': {'behavior:choice': ('readout', 'argmax'),
                       'behavior:probabilities': ('likelihood', 'softmax')},
}
CHOICE_AND_PROBABILITIES = TaskContext(instruction='Which?',
                                       response=Choice(['cat', 'dog']),
                                       fitting_stimuli=FITTING)


@pytest.mark.parametrize('arrangement', sorted(CROSSED))
@pytest.mark.parametrize('channel', ['behavior:choice', 'behavior:probabilities'])
def test_one_format_of_a_crossed_pair_is_served_by_its_owner(arrangement, channel):
    owners = CROSSED[arrangement]
    calls = []
    model = _model('likelihood', 'behavioral_readout_layer', owners=owners, calls=calls)
    answer = _run(model, [channel], CHOICE_AND_PROBABILITIES).collect(channel)
    owner = owners[channel] if isinstance(owners[channel], str) else owners[channel][0]
    assert answer.attrs['response']['method'] == owner
    assert answer.sizes['presentation'] == len(STIMULI)
    if owner == 'likelihood':
        assert calls.count('likelihood') == len(STIMULI)
    else:
        assert 'likelihood' not in calls


@pytest.mark.parametrize('arrangement', sorted(CROSSED))
def test_both_formats_of_a_crossed_pair_are_refused_before_compute(arrangement):
    calls = []
    model = _model('likelihood', 'behavioral_readout_layer', owners=CROSSED[arrangement], calls=calls)
    with pytest.raises(ExecutionNotFeasibleError,
                       match=r"from its (likelihood|readout) part and probabilities from its "
                             r"(readout|likelihood) part in one session; request one format in the session"):
        _run(model, ['behavior:choice', 'behavior:probabilities'], CHOICE_AND_PROBABILITIES)
    assert calls == []


# --- the formats the session requests choose the parts --------------------------------

def _count_fits(model):
    """Wrap the readout fit so each call is counted; the fit itself still runs."""
    fits, fit = [], model._behavioral.fit_behavioral_readout
    model._behavioral.fit_behavioral_readout = lambda stimuli: fits.append(1) or fit(stimuli)
    return fits


def test_a_readout_alone_answers_probabilities_with_one_fit():
    model = _model('behavioral_readout_layer',
                   owners=REGISTERED_STATEMENTS[('behavioral_readout_layer',)])
    fits = _count_fits(model)
    answer = _run(model, ['behavior:probabilities'], CHOICE_AND_PROBABILITIES).collect(
        'behavior:probabilities')
    assert answer.attrs['response']['method'] == 'readout'
    assert fits == [1]


@pytest.mark.parametrize('owners', [CROSSED['likelihood choice'], CROSSED['readout choice'],
                                    {'behavior:choice': ('readout', 'argmax')}],
                         ids=['likelihood choice', 'readout choice', 'readout owns both'])
def test_one_configured_task_serves_two_sessions_of_different_formats(owners):
    """Choice only, then probabilities only, on the same task: each format's owner answers,
    and the readout is fitted once in total (the setup is keyed by the task alone)."""
    calls = []
    model = _model('likelihood', 'behavioral_readout_layer', owners=owners, calls=calls)
    fits = _count_fits(model)
    for channel in ('behavior:choice', 'behavior:probabilities'):
        del calls[:]
        answer = _run(model, [channel], CHOICE_AND_PROBABILITIES).collect(channel)
        owner = owners.get(channel, 'readout')
        owner = owner if isinstance(owner, str) else owner[0]
        assert answer.attrs['response']['method'] == owner
        assert calls.count('likelihood') == (len(STIMULI) if owner == 'likelihood' else 0)
    assert fits == [1]


# --- the passes of a task stay as before the requested formats chose the parts ----------

class _PassNet:
    """Each part's pass gives its own activity (likelihood 100, reply 110, answer 120,
    readout extraction 130, plus the stimulus index), so the recording names its pass."""
    BASE = {'likelihood': 100, 'reply': 110, 'answer': 120, 'extract': 130}

    def __init__(self):
        self.calls, self.hooks = [], []

    def forward(self, stimulus_id, kind):
        self.calls.append(kind)
        out = {'early': self.BASE[kind] + 'abc'.index(stimulus_id) + np.arange(3.)}
        for hook in self.hooks:
            hook(out)
        return out


def _rows(rows, ids):
    return NeuroidAssembly(np.array(rows), dims=['presentation', 'neuroid'], coords={
        'stimulus_id': ('presentation', list(ids)),
        'neuroid_id': ('neuroid', [f'early.{unit}' for unit in range(3)]),
        'layer': ('neuroid', ['early'] * 3)})


class _PassExtractor:
    """Records the first pass of each trial, as TextWrapper.record does."""
    identifier = 'pass-net'

    def __init__(self, net):
        self.net = net

    def __call__(self, stimuli, layers, **kwargs):
        ids = list(stimuli['stimulus_id'])
        return _rows([self.net.forward(i, 'extract')['early'] for i in ids], ids)

    @contextmanager
    def record(self, layers, stimuli=None):
        captured, recording = [], type('Recording', (), {})()

        def take(_row):
            first = captured[0]
            captured.clear()
            return {layer: _rows([first['early']], ['?']) for layer in layers}
        recording.take = take
        self.net.hooks.append(captured.append)
        try:
            yield recording
        finally:
            self.net.hooks.remove(captured.append)


def _pass_model(net, parts, owners):
    every = {
        'reply': lambda prompt: net.forward(prompt.stimulus_id, 'reply') and 'cat',
        'likelihood': lambda prompt, labels: net.forward(prompt.stimulus_id, 'likelihood') and {
            label: -float(i) for i, label in enumerate(labels)},
        'answer': lambda stimulus_row, instruction, label_set: net.forward(
            stimulus_row['stimulus_id'], 'answer') and label_set[0],
        'behavioral_readout_layer': 'early'}
    return BrainScoreModel('pass-net', extractor={'text': _PassExtractor(net)}, in_channels={'text'},
                           region_layer_map={'A': LayerSelector('early', modality='text')},
                           **{part: every[part] for part in parts},
                           answer_owners=owners)


PASS_STIMULI = StimulusSet({'stimulus_id': ['a', 'b', 'c'], 'sentence': ['x', 'y', 'z']})
PASS_STIMULI.identifier = 'pass-net'
CHOICE_RAW_TEXT = TaskContext(response=Choice(['cat', 'dog'], verbatim=True))
LIKELIHOOD_ROWS = [[100.0, 101.0, 102.0], [101.0, 102.0, 103.0], [102.0, 103.0, 104.0]]
REPLY_ROWS = [[110.0, 111.0, 112.0], [111.0, 112.0, 113.0], [112.0, 113.0, 114.0]]


@pytest.mark.parametrize('parts', [CAUSAL_LM, QWEN], ids=['causal lm', 'qwen'])
@pytest.mark.parametrize('outputs, calls, rows', [
    (['behavior:choice', 'behavior:probabilities', 'neural:A'], ['likelihood', 'reply'] * 3, LIKELIHOOD_ROWS),
    (['behavior:choice', 'neural:A'], ['reply'] * 3, REPLY_ROWS),
    # Probabilities with a region: the recording still comes from the pass of the task's answer.
    (['behavior:probabilities', 'neural:A'], ['likelihood', 'reply'] * 3, LIKELIHOOD_ROWS),
    (['neural:A'], ['reply'] * 3, REPLY_ROWS),
    # Probabilities alone (no region, no trial history): only their owner runs; nothing reads a reply.
    (['behavior:probabilities'], ['likelihood'] * 3, None),
], ids=['choice, probabilities and neural', 'choice and neural', 'probabilities and neural', 'neural only',
        'probabilities only'])
def test_the_requested_formats_decide_the_passes_and_the_recording(parts, outputs, calls, rows):
    """The likelihood (the owner of the probabilities) runs only when the session requests
    them, and it scores the prompt first, so a recording comes from its pass. Otherwise
    only the pass of the choice runs, and the recording comes from it (rule 4.5)."""
    net = _PassNet()
    session = stimulus_session(PASS_STIMULI, outputs, task=CHOICE_RAW_TEXT)
    _pass_model(net, parts, REGISTERED_STATEMENTS[parts]).participate(session)
    assert net.calls == calls
    if rows is None:
        return
    recorded = session.collect('neural:A')
    assert recorded.values.tolist() == rows
    assert (recorded.attrs['recording']['path'], recorded.attrs['recording']['forward_pass']) == (
        'generation', 'prompt')


# --- within generation, a plain Choice goes to answer= --------------------------------

@pytest.mark.parametrize('verbatim, called', [(False, 'answer'), (True, 'reply')],
                         ids=['plain choice', 'verbatim choice'])
def test_a_plain_choice_goes_to_answer_and_a_task_setting_to_reply(verbatim, called):
    calls = []
    model = _model(*QWEN, owners=REGISTERED_STATEMENTS[QWEN], calls=calls)
    task = TaskContext(instruction=None if verbatim else 'Which?',
                       response=Choice(['cat', 'dog'], verbatim=verbatim))
    choice = _run(model, ['behavior:choice'], task).collect('behavior:choice')
    assert calls == [called] * len(STIMULI)
    assert choice.attrs['response']['method'] == 'generation'


# --- review S7-02: a task that sets 'present' is not a plain Choice ------------------

def _screens():
    from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
    stimuli = StimulusSet({'stimulus_id': ['t0', 't1'], 'rule': ['animal', 'colour'],
                           'word_1': ['/s/t0_1.png', '/s/t1_1.png'], 'word_2': ['/s/t0_2.png', '/s/t1_2.png']})
    stimuli.identifier = 'two-screens'
    return stimuli


SCREENS = TaskContext(present=('image:word_1', 'image:word_2'), response=Choice(['yes', 'no']))


def test_a_choice_that_sets_present_goes_to_the_reply_not_the_registered_answer():
    """Notebook 14's task shape: two image screens, a plain Choice, no instruction. The
    registered Qwen answer (it reads image_file_name) is never called."""
    import torch
    from brainscore.models.qwen25_vl_3b.model import _make_answer
    registered = _make_answer(torch.nn.Linear(1, 1), object())
    answered, shown = [], []

    def answer(**kwargs):
        answered.append(kwargs)
        return registered(**kwargs)

    def reply(prompt):
        shown.append(prompt.parts)
        return 'yes'
    model = BrainScoreModel('screens', in_channels={'vision:image'}, answer=answer, reply=reply)
    session = stimulus_session(_screens(), ['behavior:choice'], SCREENS)
    model.participate(session)
    assert answered == []
    assert [[kind for kind, _ in parts] for parts in shown] == [['image', 'image', 'text']] * 2
    assert shown[0][:2] == (('image', '/s/t0_1.png'), ('image', '/s/t0_2.png'))
    assert session.collect('behavior:choice').values.tolist() == [[1.0, 0.0]] * 2


def test_an_answer_part_alone_refuses_a_choice_that_sets_present_before_any_call():
    calls = []
    model = BrainScoreModel('answer-only', in_channels={'vision:image'},
                            answer=lambda **kwargs: calls.append(kwargs) or 'yes')
    with pytest.raises(ExecutionNotFeasibleError, match='present'):
        model.participate(stimulus_session(_screens(), ['behavior:choice'], SCREENS))
    assert calls == []
