"""The refusal table of the session plan: every row, its class, the start of its message, and when it is raised.

Each row builds a subject whose parts log every call: the extractor (``extract``),
the ``answer``, ``reply`` and ``likelihood`` parts, the policy (``policy``) and the
readout fit (``fit``, then the fit's own ``extract``). A row states the calls made
before the refusal: none for every refusal the session plan makes; the calls of
the earlier trials or of the readout fit for a refusal that is raised per batch,
for a source with no table, or after a pass. The class of each row is one literal
(``CLASS``), so a decision to change a class changes one line.

Each row is named by a number (R1 to R30, with the rows added after) and a short
description.
"""
import re

import numpy as np
import pytest

from brainscore import BrainScoreModel
from brainscore_core.compatibility import ExecutionNotFeasibleError
from brainscore_core.contract import TaskContext
from brainscore_core.events import Perturbation, PerturbationApplied, Selection, StateChange
from brainscore_core.responses import Choice
from brainscore_core.streaming import Session, StreamEvent
from brainscore_core.streaming_helpers import apply_state_change, stimulus_session
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
from tests.brainscore_model.test_behavior_paths import FITTING, STIMULI, _features

pytestmark = pytest.mark.unit

ENFE = ExecutionNotFeasibleError

# One literal class per row (the classes R19 to R24 keep for now are judged by the review).
CLASS = {
    'R1': RuntimeError, 'R2': ValueError, 'R3': ValueError, 'R4': ENFE, 'R5': ValueError,
    'R6': ENFE, 'R7': ENFE, 'R8': ENFE, 'R9 table': ENFE, 'R9 stream': ENFE,
    'R10 owner cannot serve': ENFE, 'R10 settings': ENFE, 'R10 no part': ENFE, 'R10 log probabilities need verbatim': ENFE, 'R10 crossed parts': ENFE,
    'R11 readout_fit': ValueError, 'R11 task_type': ValueError,
    'R12': ENFE, 'R13': ValueError,
    'R14 plan': ENFE, 'R14 direct call': ValueError,
    'R15 table': ValueError,
    'R16 table': ValueError, 'R16 stream with a table': ValueError, 'R16 no table, one at a time': ValueError,
    'R17 table': ENFE, 'R17 stream with a table': ENFE, 'R17 no table': ENFE,
    'R18 table': ENFE, 'R18 no table': ENFE, 'R18 policy step': ENFE,
    'R19': ValueError, 'R20': ENFE, 'R20 direct call': ValueError, 'R21': ValueError, 'R22 process': ValueError,
    'R23 process': ValueError, 'R24 layers': ENFE, 'R24 no extractor': ENFE,
    'R24 layers direct call': ValueError, 'R24 no extractor direct call': NotImplementedError,
    'R24 readout': ENFE, 'R24 readout direct call': ValueError,
    'R25': ValueError, 'R26': ValueError, 'R27': ValueError,
    'R28 parts': ENFE, 'R28 history': ENFE,
    'R29 reset without handle': ValueError, 'R29 unknown handle': KeyError,
    'R29 no part': NotImplementedError, 'R29 bad return': TypeError,
    'R30 facade': ENFE, 'R30 Lahner': ENFE,
}

CHOICE = TaskContext(instruction='Which?', response=Choice(['cat', 'dog']), fitting_stimuli=FITTING)
LEGACY = TaskContext(task_type='probabilities', label_set=['cat', 'dog'], fitting_stimuli=FITTING)


@pytest.fixture
def log(monkeypatch):
    """The calls of every part; the readout fit is logged as ``fit`` before its extraction."""
    from brainscore._model.behavioral import BehavioralReadout
    calls = []
    fit = BehavioralReadout.fit_behavioral_readout
    monkeypatch.setattr(BehavioralReadout, 'fit_behavioral_readout',
                        lambda self, *a, **k: calls.append('fit') or fit(self, *a, **k))
    return calls


def _subject(log, *, parts=(), regions=None, owners=None, memory=None, extractors=('text',),
             in_channels=('text',), **kwargs):
    def extract(stimuli, layers, **_):
        log.append('extract')
        return _features(stimuli, layers)
    available = {
        'answer': lambda stimulus_row, instruction, label_set: log.append('answer') or label_set[0],
        'reply': lambda prompt: log.append('reply') or 'cat',
        'likelihood': lambda prompt, labels: log.append('likelihood') or {
            label: -float(i) for i, label in enumerate(labels)},
        'behavioral_readout_layer': 'late'}
    config = {'capability_config': {'behavior-sessions': owners}} if owners else {}
    return BrainScoreModel('logged', extractor={key: extract for key in extractors}, in_channels=set(in_channels),
                           region_layer_map=regions or {}, memory=memory,
                           **{part: available[part] for part in parts}, **config, **kwargs)


def _events(*rows):
    """A source with no table: one event per row (channel, payload, stimulus id)."""
    return [StreamEvent(channel, payload, t_ms=float(i), meta={'stimulus_id': sid})
            for i, (channel, payload, sid) in enumerate(rows)]


# Each trigger raises the row's refusal; EXPECTED gives the start of its message and the calls before it.
TRIGGERS, EXPECTED = {}, {}


def row(name, message, calls=()):
    def register(trigger):
        TRIGGERS[name], EXPECTED[name] = trigger, (message, list(calls))
        return trigger
    return register


@row('R1', r"Subject 'logged' is already taking part in a session", ['reply'])
def _r1(log, monkeypatch, tmp_path):
    holder = {}

    def reply(prompt):
        log.append('reply')
        holder['model'].participate(stimulus_session(STIMULI, ['behavior:choice'], CHOICE))
    holder['model'] = model = BrainScoreModel('logged', in_channels={'text'}, reply=reply)
    model.participate(stimulus_session(STIMULI, ['behavior:choice'], CHOICE))


@row('R2', r"BRAINSCORE_REGIONS_PER_PASS must be 'all'")
def _r2(log, monkeypatch, tmp_path):
    monkeypatch.setenv('BRAINSCORE_REGIONS_PER_PASS', 'two')
    _subject(log, regions={'IT': 'late'}).participate(stimulus_session(STIMULI, ['neural:IT']))


@row('R3', r'A session must request at least one output channel\.')
def _r3(log, monkeypatch, tmp_path):
    _subject(log, regions={'IT': 'late'}).participate(stimulus_session(STIMULI, []))


@row('R4', r"Session refused \(task type 'none'\): Subject 'logged' cannot answer \['motor', 'neural:IT'\]")
def _r4(log, monkeypatch, tmp_path):
    _subject(log, regions={'IT': 'late'}).participate(stimulus_session(STIMULI, ['neural:IT', 'motor']))


@row('R5', r'Ambiguous capabilities.*give one capability a lower order')
def _r5(log, monkeypatch, tmp_path):
    import brainscore.capabilities as capabilities
    from brainscore.capabilities.base import Capability

    class _Claims(Capability):
        order = 5

        def supports_session(self, subject, channels):
            return True
    own = {}  # own registry: a new plugin with a low order cannot win in place of these two
    for name in ('claims-a', 'claims-b'):
        claim = _Claims()
        claim.identifier = name
        own[name] = claim
    monkeypatch.setattr(capabilities, 'capability_registry', own)
    _subject(log, regions={'IT': 'late'}).participate(stimulus_session(STIMULI, ['neural:IT']))


@row('R6', r'A behavior session needs a task: build it with one')
def _r6(log, monkeypatch, tmp_path):
    _subject(log, parts=('reply',)).participate(stimulus_session(STIMULI, ['behavior:choice']))


@row('R7', r'Recording during a task returns one time bin; request at most one\.')
def _r7(log, monkeypatch, tmp_path):
    model = _subject(log, parts=('behavioral_readout_layer',), regions={'IT': 'late'},
                     owners={'behavior:choice': ('readout', 'argmax')})
    model.participate(stimulus_session(STIMULI, ['neural:IT'], CHOICE, time_bins=[(0, 50), (50, 100)]))


@row('R8', r"\['behavior:probabilities'\] need TaskContext\(response=\.\.\.\)")
def _r8(log, monkeypatch, tmp_path):
    _subject(log, parts=('behavioral_readout_layer',)).participate(
        stimulus_session(STIMULI, ['behavior:probabilities'], LEGACY))


@row('R9 table', r"Subject 'logged' cannot produce trial-to-trial variance")
def _r9_table(log, monkeypatch, tmp_path):
    _subject(log, regions={'IT': 'late'}).participate(
        stimulus_session(STIMULI, ['neural:IT'], settings={'require_variance': True}))


@row('R9 stream', r"Subject 'logged' cannot produce trial-to-trial variance")
def _r9_stream(log, monkeypatch, tmp_path):
    _subject(log, regions={'IT': 'late'}).participate(
        stimulus_session(STIMULI, ['neural:IT'], settings={'require_variance': True}, streaming=True))


@row('R10 owner cannot serve', r"Subject 'logged': 'behavior:choice' is owned by 'generation', which cannot "
                                r"serve this task \(task settings: verbatim\); no other part answers it\.")
def _r10_owner(log, monkeypatch, tmp_path):
    task = TaskContext(response=Choice(['cat', 'dog'], verbatim=True))
    _subject(log, parts=('answer', 'likelihood')).participate(stimulus_session(STIMULI, ['behavior:choice'], task))


@row('R10 settings', r"Subject 'logged' cannot give a Choice answer with the task settings verbatim\. Paths tried")
def _r10_settings(log, monkeypatch, tmp_path):
    task = TaskContext(response=Choice(['cat', 'dog'], verbatim=True))
    _subject(log, parts=('answer',)).participate(stimulus_session(STIMULI, ['behavior:choice'], task))


@row('R10 no part', r"Subject 'logged' cannot give a Choice answer\. Paths tried")
def _r10_no_part(log, monkeypatch, tmp_path):
    task = TaskContext(instruction='Which?', response=Choice(['cat', 'dog']))       # no fitting stimuli
    _subject(log, parts=('behavioral_readout_layer',), owners={'behavior:choice': ('readout', 'argmax')}
             ).participate(stimulus_session(STIMULI, ['behavior:choice'], task))


@row('R10 log probabilities need verbatim', r"Subject 'logged' gives behavior:log_probabilities only for a verbatim "
                                            r"Choice")
def _r10_verbatim(log, monkeypatch, tmp_path):
    task = TaskContext(instruction='Which?', response=Choice(['cat', 'dog']))
    _subject(log, parts=('likelihood',)).participate(
        stimulus_session(STIMULI, ['behavior:log_probabilities'], task))


@row('R10 crossed parts', r"Subject 'logged' cannot give choices from its likelihood part and probabilities "
                          r"from its readout part in one session")
def _r10_crossed(log, monkeypatch, tmp_path):
    model = _subject(log, parts=('likelihood', 'behavioral_readout_layer'), owners={
        'behavior:choice': ('likelihood', 'softmax_argmax'), 'behavior:probabilities': 'readout'})
    model.participate(stimulus_session(STIMULI, ['behavior:choice', 'behavior:probabilities'], CHOICE))


@row('R11 readout_fit', r"readout_fit must be 'current' or 'intact', not 'bad'")
def _r11_fit(log, monkeypatch, tmp_path):
    task = TaskContext(instruction='Which?', response=Choice(['cat', 'dog']), fitting_stimuli=FITTING,
                       readout_fit='bad')
    _subject(log, parts=('reply',)).participate(stimulus_session(STIMULI, ['behavior:choice'], task))


@row('R11 task_type', r"task_type 'foo' states no response")
def _r11_type(log, monkeypatch, tmp_path):
    task = TaskContext(task_type='foo', label_set=['cat', 'dog'], fitting_stimuli=FITTING)
    _subject(log, parts=('behavioral_readout_layer',)).participate(stimulus_session(STIMULI, ['behavior'], task))


@row('R12', r"Subject 'logged' cannot fit on the intact network", ['fit'])
def _r12(log, monkeypatch, tmp_path):
    def intervention(change):
        return PerturbationApplied(handle_id='h', target=change.target, perturbation=change.perturbation,
                                   applied_at=0), lambda: None
    model = _subject(log, parts=('behavioral_readout_layer',), owners={'behavior:choice': ('readout', 'argmax')},
                     intervention=intervention)
    apply_state_change(model, StateChange(kind='ablation', target=Selection(layer='late'),
                                          perturbation=Perturbation(kind='zero')))
    task = TaskContext(instruction='Which?', response=Choice(['cat', 'dog']), fitting_stimuli=FITTING,
                       readout_fit='intact')
    model.participate(stimulus_session(STIMULI, ['behavior:choice'], task))


@row('R13', r'Cannot find labels in fitting_stimuli', ['fit'])
def _r13(log, monkeypatch, tmp_path):
    fitting = StimulusSet({'stimulus_id': ['f0', 'f1'], 'sentence': ['real 0', 'fake 1']})
    fitting.identifier = 'refusals-no-labels'
    task = TaskContext(instruction='Which?', response=Choice(['cat', 'dog']), fitting_stimuli=fitting)
    _subject(log, parts=('behavioral_readout_layer',), owners={'behavior:choice': ('readout', 'argmax')}
             ).participate(stimulus_session(STIMULI, ['behavior:choice'], task))


def _answer_and_readout(log, **kwargs):
    return _subject(log, parts=('answer', 'behavioral_readout_layer'), owners={},
                    **kwargs)


@row('R14 plan', r"memory='trial_history' needs reply= or likelihood=; answer= cannot see earlier trials\.")
def _r14_plan(log, monkeypatch, tmp_path):
    _answer_and_readout(log, memory='trial_history').participate(
        stimulus_session(STIMULI, ['behavior:choice', 'behavior:probabilities'], CHOICE))


@row('R14 direct call', r"memory='trial_history' needs reply= or likelihood=; answer= cannot see earlier")
def _r14_direct(log, monkeypatch, tmp_path):
    model = _subject(log, parts=('answer',), memory='trial_history')
    model.start_task(TaskContext(instruction='Which?', response=Choice(['cat', 'dog'])))
    model._behavioral.respond(STIMULI, ['behavior:choice'])


def _options(counts):
    ids = [f'o{i}' for i in range(len(counts))]
    table = StimulusSet({'stimulus_id': ids, 'sentence': ['real a', 'fake b', 'real c'][:len(counts)],
                         'options': [[f'x{j}' for j in range(n)] for n in counts]})
    table.identifier = f'refusals-options-{counts}'
    return table


@row('R15 table', r"Every row must give the same number of options in column 'options'; got \[2, 3\]")
def _r15(log, monkeypatch, tmp_path):
    task = TaskContext(instruction='Which?', response=Choice('options'))
    _subject(log, parts=('reply',)).participate(stimulus_session(_options([2, 3, 2]), ['behavior:choice'], task))


_MISSING = TaskContext(instruction='Which?', response=Choice(['cat', 'dog']), fitting_stimuli=FITTING,
                       present=('text:no_such_column',))


@row('R16 table', r"TaskContext\.present names column 'no_such_column', which this stimulus lacks\.")
def _r16_table(log, monkeypatch, tmp_path):
    _subject(log, parts=('reply',)).participate(stimulus_session(STIMULI, ['behavior:choice'], _MISSING))


@row('R16 stream with a table', r"TaskContext\.present names column 'no_such_column'")
def _r16_stream(log, monkeypatch, tmp_path):
    _subject(log, parts=('reply',)).participate(
        stimulus_session(STIMULI, ['behavior:choice'], _MISSING, streaming=True))


@row('R16 no table, one at a time', r"TaskContext\.present names 'text', but the stimulus has only \[",
     ['reply'])
def _r16_no_table(log, monkeypatch, tmp_path):
    task = TaskContext(instruction='Which?', response=Choice(['cat', 'dog']), present=('text',))
    source = _events(('text', 'real a', 's0'), ('vision:image', 'b.png', 's1'))
    model = _subject(log, parts=('reply',), memory='trial_history', in_channels=('text', 'vision:image'))
    model.participate(Session(source, ['behavior:choice'], task))


def _two_inputs(log):
    from brainscore.selection import LayerSelector
    return _subject(log, parts=('behavioral_readout_layer',), extractors=('text', 'audio'),
                    regions={'A1': LayerSelector('late', modality='audio')},
                    owners={'behavior:choice': ('readout', 'argmax')})


_A1 = r"Region 'A1' reads 'audio' input, but the readout presents 'text' input; one presentation cannot record it\."


@row('R17 table', _A1)
def _r17_table(log, monkeypatch, tmp_path):
    _two_inputs(log).participate(stimulus_session(STIMULI, ['behavior:choice', 'neural:A1'], CHOICE))


@row('R17 stream with a table', _A1)
def _r17_stream(log, monkeypatch, tmp_path):
    _two_inputs(log).participate(stimulus_session(STIMULI, ['behavior:choice', 'neural:A1'], CHOICE,
                                                  streaming=True))


@row('R17 no table', _A1, ['fit', 'extract'])
def _r17_no_table(log, monkeypatch, tmp_path):
    source = _events(('text', 'real a', 's0'), ('text', 'fake b', 's1'))
    _two_inputs(log).participate(Session(source, ['behavior:choice', 'neural:A1'], CHOICE))


_IT = r"Subject 'logged' cannot record \['IT'\] while it answers or acts"


@row('R18 table', _IT)
def _r18_table(log, monkeypatch, tmp_path):
    _answer_and_readout(log, regions={'IT': 'late'}).participate(
        stimulus_session(STIMULI, ['behavior:choice', 'behavior:probabilities', 'neural:IT'], CHOICE))


@row('R18 no table', _IT, ['fit', 'extract'])
def _r18_no_table(log, monkeypatch, tmp_path):
    source = _events(('text', 'real a', 's0'), ('text', 'fake b', 's1'))
    _answer_and_readout(log, regions={'IT': 'late'}).participate(
        Session(source, ['behavior:choice', 'behavior:probabilities', 'neural:IT'], CHOICE))


# The claim of the policy capability already asks records_while_answering, so a
# region that cannot be recorded during a policy step is refused as R4 (no part answers the outputs
# together), never as R18. The row states what happens.
@row('R18 policy step', r"Session refused \(task type 'none'\): Subject 'logged' cannot answer "
                        r"\['motor', 'neural:IT'\] in one session: no one part of it answers these outputs together")
def _r18_policy(log, monkeypatch, tmp_path):
    from brainscore_core.streaming_helpers import EnvironmentSource

    class _Rows:
        def reset(self):
            return {'observation': 0}

        def step(self, outputs):
            return {'observation': 0}, 0.0, True, {}
    model = _subject(log, regions={'IT': 'late'}, policy=lambda row: log.append('policy') or {'motor': 0})
    model.participate(Session(EnvironmentSource(_Rows()), ('motor', 'neural:IT')))


@row('R19', r"Duplicate output channels requested: \['neural:IT'\]")
def _r19(log, monkeypatch, tmp_path):
    _subject(log, regions={'IT': 'late'}).participate(stimulus_session(STIMULI, ['neural:IT', 'neural:IT']))


@row('R20', r"Recording targets require missing input modalities: \['audio'\]")
def _r20(log, monkeypatch, tmp_path):
    from brainscore.selection import LayerSelector
    model = _subject(log, extractors=('text', 'audio'), regions={'A1': LayerSelector('late', modality='audio')})
    model.participate(stimulus_session(STIMULI, ['neural:A1']))


@row('R20 direct call', r"Recording targets require missing input modalities: \['audio'\]")
def _r20_direct(log, monkeypatch, tmp_path):
    from brainscore.selection import LayerSelector
    model = _subject(log, extractors=('text', 'audio'), regions={'A1': LayerSelector('late', modality='audio')})
    model.start_recording('A1')
    model.process(STIMULI)


@row('R21', r'No column in this stimulus set says what kind of input it holds')
def _r21(log, monkeypatch, tmp_path):
    table = StimulusSet({'stimulus_id': ['s0', 's1'], 'words': ['real a', 'fake b']})
    table.identifier = 'refusals-no-column'
    _subject(log, regions={'IT': 'late'}).participate(stimulus_session(table, ['neural:IT']))


@row('R22 process', r'Recording targets span modalities; pass multi_modality=True\.')
def _r22(log, monkeypatch, tmp_path):
    from brainscore.selection import LayerSelector
    model = _subject(log, extractors=('text', 'audio'), regions={
        'IT': LayerSelector('late', modality='text'), 'A1': LayerSelector('late', modality='audio')})
    table = StimulusSet({'stimulus_id': ['s0'], 'sentence': ['real a'], 'audio_path': ['a.wav']})
    table.identifier = 'refusals-two-inputs'
    model.start_recording(['IT', 'A1'])
    model.process(table)


@row('R23 process', r"Subject 'logged' records no region")
def _r23(log, monkeypatch, tmp_path):
    _subject(log, regions={'IT': 'late'}).process(STIMULI)


class _Layered:
    def __init__(self, layers, log):
        self.layers, self.log = list(layers), log

    def layer_names(self):
        return list(self.layers)

    def __call__(self, stimuli, layers, **_):
        self.log.append('extract')
        return _features(stimuli, layers)


@row('R24 layers', r"Subject 'two-forms' has no 'vision' extractor for the inputs in this stimulus set that "
                   r"has all of the layers \['early'\]")
def _r24_layers(log, monkeypatch, tmp_path):
    model = BrainScoreModel('two-forms', extractor={'vision': _Layered(['late'], log),
                                                    'video': _Layered(['early'], log)},
                            region_layer_map={'IT': 'late', 'V4': 'early'})
    table = StimulusSet({'stimulus_id': ['p0', 'p1'], 'image_path': ['p0.png', 'p1.png'],
                         'sentence': ['real a', 'fake b']})
    table.identifier = 'refusals-pictures'
    model.participate(stimulus_session(table, ['neural:IT', 'neural:V4']))


@row('R24 no extractor', r"Subject 'logged' declares 'text' input but has no activation extractor for it")
def _r24_none(log, monkeypatch, tmp_path):
    _subject(log, extractors=('vision',), regions={'IT': 'late'}).participate(
        stimulus_session(STIMULI, ['neural:IT']))


@row('R24 layers direct call', r"Subject 'two-forms' has no 'vision' extractor for the inputs in this stimulus "
                              r"set that has all of the layers \['early'\]")
def _r24_layers_direct(log, monkeypatch, tmp_path):
    model = BrainScoreModel('two-forms', extractor={'vision': _Layered(['late'], log),
                                                    'video': _Layered(['early'], log)},
                            region_layer_map={'IT': 'late', 'V4': 'early'})
    table = StimulusSet({'stimulus_id': ['p0'], 'image_path': ['p0.png']})
    table.identifier = 'refusals-pictures-direct'
    model.start_recording('V4')
    model.process(table)


@row('R24 no extractor direct call', r"Subject 'logged' declares 'text' input but has no activation extractor for it")
def _r24_none_direct(log, monkeypatch, tmp_path):
    model = _subject(log, extractors=('vision',), regions={'IT': 'late'})
    model.start_recording('IT')
    model.process(STIMULI)


def _readout_two_forms(log):
    """SP-21: an image extractor owns the readout layer 'late'; a video extractor owns the region's 'early'."""
    class _Pictures(_Layered):
        def __call__(self, stimuli, layers, **_):          # features from the row position (no text column)
            from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
            self.log.append('extract')
            ids = list(stimuli['stimulus_id'])
            values = np.array([[2.0, 0.0] if i % 2 == 0 else [0.0, 2.0] for i in range(len(ids))])
            return NeuroidAssembly(values, dims=['presentation', 'neuroid'], coords={
                'stimulus_id': ('presentation', ids), 'stimulus_path': ('presentation', ids),
                'neuroid_id': ('neuroid', ['n0', 'n1']), 'layer': ('neuroid', [layers[0]] * 2)})
    model = BrainScoreModel('readout-missing', extractor={'vision': _Pictures(['late'], log),
                                                          'video': _Pictures(['early'], log)},
                            region_layer_map={'V4': 'early'}, behavioral_readout_layer='late',
                            capability_config={'behavior-sessions': {'behavior:choice': ('readout', 'argmax')}})

    def pictures(ids, labels, identifier):
        table = StimulusSet({'stimulus_id': ids, 'image_path': [f'{i}.png' for i in ids],
                             'sentence': ['real ' + i if label == 'cat' else 'fake ' + i
                                          for i, label in zip(ids, labels)], 'label': labels})
        table.identifier = identifier
        return table
    fitting = pictures(['f0', 'f1', 'f2', 'f3'], ['cat', 'dog', 'cat', 'dog'], 'readout-two-forms-fit')
    task = TaskContext(instruction='Which?', response=Choice(['cat', 'dog']), fitting_stimuli=fitting)
    return model, task, pictures(['s0', 's1'], ['cat', 'dog'], 'readout-two-forms')


_READOUT_R24 = (r"Subject 'readout-missing' has no 'vision' extractor for the inputs in this stimulus set that "
                r"has all of the layers \['late', 'early'\]")


@row('R24 readout', _READOUT_R24)
def _r24_readout(log, monkeypatch, tmp_path):
    """Before SP-21: one fitting extraction, then the direct-call class."""
    model, task, stimuli = _readout_two_forms(log)
    model.participate(stimulus_session(stimuli, ['behavior:choice', 'neural:V4'], task))


@row('R24 readout direct call', _READOUT_R24, ['fit', 'extract'])
def _r24_readout_direct(log, monkeypatch, tmp_path):
    model, task, stimuli = _readout_two_forms(log)
    model.start_task(task)                       # fits the readout: one extraction
    model._behavioral._read_out_and_record(stimuli, {'V4': ['early']})


@row('R25', r"Neural output for region 'IT' has no 'neuroid' dimension\.", ['extract'])
def _r25(log, monkeypatch, tmp_path):
    def extract(stimuli, layers, **_):
        log.append('extract')
        return _features(stimuli, layers).rename(neuroid='unit')
    BrainScoreModel('logged', extractor={'text': extract}, region_layer_map={'IT': 'late'}).participate(
        stimulus_session(STIMULI, ['neural:IT']))


@row('R26', r'.*different lengths')
def _r26(log, monkeypatch, tmp_path):
    from tests.brainscore_model.test_task_recording_grid import TASK, _sound_subject
    waves = {'a': np.array([1, 1], np.float32), 'b': np.array([3, 3, 3, 3], np.float32)}
    model, stimuli, passes = _sound_subject(waves)
    try:
        model.participate(stimulus_session(stimuli, ['behavior:choice', 'neural:low'], TASK))
    finally:
        log.extend('pass' for _ in passes)


@row('R27', r'.*other input', ['reply', 'pass'])
def _r27(log, monkeypatch, tmp_path):
    import torch
    from tests.brainscore_model import tiny_wrappers
    wrapper, stimuli, regions, network = tiny_wrappers.FACTORIES['audio'](tmp_path)
    layer = regions['V4'].name
    network.register_forward_hook(lambda *_: log.append('pass'))
    shown, original = {}, network.forward

    def keep(*args, **kwargs):
        shown.setdefault('args', (args, kwargs))
        return original(*args, **kwargs)
    network.forward = keep

    def reply(prompt):
        log.append('reply')
        tiny_wrappers.show(wrapper, stimuli, prompt.stimulus_id, layer)
        args, kwargs = shown.pop('args')
        tensor = next(v for v in (*args, *kwargs.values()) if torch.is_tensor(v) and v.is_floating_point())
        with torch.no_grad():
            tensor.add_(1.0)                     # a later pass of the same trial with other input
            network(*args, **kwargs)
        return 'yes'
    model = BrainScoreModel('tiny-audio', extractor={'audio': wrapper}, region_layer_map={'V4': regions['V4']},
                            reply=reply)
    task = TaskContext(instruction='Answer.', response=Choice(['yes', 'no']))
    model.participate(stimulus_session(stimuli, ['behavior:choice', 'neural:V4'], task))


@row('R28 parts', r"Plain-text scoring takes exactly one text part; got \['text', 'image'\]")
def _r28_parts(log, monkeypatch, tmp_path):
    from brainscore.model_helpers.replies import _plain_text_scores
    from brainscore_core.responses import Prompt
    _plain_text_scores(None, None, Prompt(parts=(('text', 'a'), ('image', 'b.png'))), ['x'])


@row('R28 history', r'Plain-text scoring scores the text alone; it cannot show earlier trials')
def _r28_history(log, monkeypatch, tmp_path):
    from brainscore.model_helpers.replies import _plain_text_scores
    from brainscore_core.responses import Prompt
    _plain_text_scores(None, None, Prompt(parts=(('text', 'a'),), history=(('earlier', 'cat'),)), ['x'])


def _intervened(log, result=None):
    def intervention(change):
        log.append('intervention')
        return result if result is not None else (PerturbationApplied(
            handle_id='h', target=change.target, perturbation=change.perturbation, applied_at=0), lambda: None)
    return _subject(log, intervention=intervention)


@row('R29 reset without handle', r"StateChange\(kind='reset'\) requires handle_id")
def _r29_reset(log, monkeypatch, tmp_path):
    apply_state_change(_intervened(log), StateChange(kind='reset'))


@row('R29 unknown handle', r"\"No active intervention with handle_id='nope'")
def _r29_unknown(log, monkeypatch, tmp_path):
    apply_state_change(_intervened(log), StateChange(kind='reset', handle_id='nope'))


@row('R29 no part', r"Subject 'logged' has no intervention part")
def _r29_no_part(log, monkeypatch, tmp_path):
    _subject(log).process(StateChange(kind='ablation', target=Selection(layer='late'),
                                      perturbation=Perturbation(kind='zero')))


@row('R29 bad return', r"Subject 'logged' intervention must return a \(PerturbationApplied, cleanup_callable\) "
                       r"tuple", ['intervention'])
def _r29_bad_return(log, monkeypatch, tmp_path):
    apply_state_change(_intervened(log, result='not a tuple'), StateChange(
        kind='ablation', target=Selection(layer='late'), perturbation=Perturbation(kind='zero')))


@row('R30 facade', r"Subject 'logged' refused the legacy language behavioral task")
def _r30_facade(log, monkeypatch, tmp_path):
    from brainscore_core.legacy_facade import LegacyBenchmarkFacade
    LegacyBenchmarkFacade(_subject(log, regions={'IT': 'late'})).start_behavioral_task('reading_times')


@row('R30 Lahner', r"'logged' answers neural:A1 only from \['audio'\], and takes none of the streams shown")
def _r30_lahner(log, monkeypatch, tmp_path):
    from brainscore.benchmarks.lahner2024._util import stream_regions
    from brainscore.selection import LayerSelector
    model = _subject(log, extractors=('vision', 'audio'), regions={'A1': LayerSelector('late', modality='audio')})
    stream_regions(model, {'vision:image'}, ['A1'])


def test_every_row_of_the_table_has_a_class_and_a_trigger():
    assert sorted(CLASS) == sorted(TRIGGERS)


@pytest.mark.parametrize('name', sorted(TRIGGERS))
def test_a_refusal_has_its_class_its_message_and_its_time(name, log, monkeypatch, tmp_path):
    message, calls = EXPECTED[name]
    with pytest.raises(Exception) as raised:
        TRIGGERS[name](log, monkeypatch, tmp_path)
    assert type(raised.value) is CLASS[name], (name, type(raised.value), str(raised.value))
    assert re.match(message, str(raised.value)), (name, str(raised.value))
    assert log == calls, (name, log)


def test_options_per_row_are_not_compared_across_trials_one_at_a_time(log):
    """R15: a subject that keeps its history answers one trial at a time; nothing is compared."""
    task = TaskContext(instruction='Which?', response=Choice('options'))
    model = _subject(log, parts=('reply',), memory='trial_history')
    model.participate(stimulus_session(_options([2, 3, 2]), ['behavior:choice'], task))
    assert log == ['reply'] * 3


@pytest.mark.parametrize('change', [StateChange(kind='reset'), StateChange(kind='reset', handle_id='nope')],
                         ids=['reset without handle', 'unknown handle'])
def test_a_refused_intervention_event_keeps_the_fitted_readout(change, log):
    """R29: the readout fits again only when the model changes, not when an event is refused (O5)."""
    model = _subject(log, parts=('behavioral_readout_layer',), owners={'behavior:choice': ('readout', 'argmax')},
                     intervention=lambda c: (PerturbationApplied(handle_id='h', target=c.target,
                                                                 perturbation=c.perturbation, applied_at=0),
                                             lambda: None))
    model.start_task(CHOICE)
    assert model._behavioral.is_configured_for(CHOICE)
    with pytest.raises((ValueError, KeyError)):
        apply_state_change(model, change)
    assert model._behavioral.is_configured_for(CHOICE)            # before O5: the refused event invalidated it
    apply_state_change(model, StateChange(kind='ablation', target=Selection(layer='late'),
                                          perturbation=Perturbation(kind='zero')))
    assert not model._behavioral.is_configured_for(CHOICE)        # an applied change still invalidates
