"""One rule for every subject class and every driver (Contract Rules 2.4, 4.4, 4.5, 6.2).

The same small cases run over every subject class the packages ship and every core driver:

- a second session gives what it gives on a fresh subject, and keeps the source it was given;
- a request for two declared outputs gives both (a refusal comes before compute, from the
  declarations);
- each ``begin`` clears the trial history;
- a policy's ``None`` (an unreadable reply) is an invalid step in every discrete environment,
  never a crash.

Every subject runs with no weights and no download. A new subject class or a new core driver
is one more entry below; until it has one, ``test_every_subject_class_and_driver_has_an_entry``
fails."""
import re
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest

from brainscore import BrainScoreModel, LayerSelector, load_model
from brainscore.harnesses.human import HumanSubject
from brainscore.harnesses.multi_agent import EchoAgent
from brainscore.model_helpers.provenance import ProvenanceSubject
from brainscore.models.umi_dev_models.model import _StudySubject
from brainscore_core import streaming_helpers
from brainscore_core.contract import Subject, TaskContext
from brainscore_core.environment import DiscreteSpec
from brainscore_core.responses import Choice, Text
from brainscore_core.streaming import Session, StreamEvent, begin_event, end_event, respond_event
from brainscore_core.streaming_helpers import (
    drive_behavior_session, drive_behavior_session_streaming, drive_environment_session,
    drive_neural_session, drive_neural_session_streaming, run_environment, stimulus_session)
from brainscore_core.supported_data_standards.brainio.assemblies import (
    BehavioralAssembly, NeuroidAssembly)
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet

pytestmark = pytest.mark.unit

LABELS = ['cat', 'dog']


def _stimuli(column, ids=('s0', 's1')):
    """A two-row table; both rows show the same input, so a subject with no history answers
    them alike."""
    stimuli = StimulusSet({'stimulus_id': list(ids), column: ['the cat'] * len(ids)})
    stimuli.identifier = f'one-rule-{column}-{len(ids)}'
    return stimuli


def _neuroid(table, value=0.0):
    ids = list(table['stimulus_id'])
    return NeuroidAssembly(np.full((len(ids), 1), float(value)), dims=['presentation', 'neuroid'],
                           coords={'stimulus_id': ('presentation', ids), 'neuroid_id': ('neuroid', ['n'])})


# ---------------------------------------------------------------- the subject classes

LOOKS = []      # one entry for each look_at call of an old vision model


class _OldVisionModel:
    """Stands in for an old vision plugin."""
    identifier = 'old-vision'

    class layer_model:
        region_layer_map = {'IT': 'layer4'}

    def visual_degrees(self):
        return 8

    task = None

    def start_recording(self, region, time_bins=None):
        self.task = None          # as a legacy model: a recording ends the task

    def start_task(self, task, fitting_stimuli=None):
        self.task = None if task == 'passive' else task

    def look_at(self, stimuli, number_of_trials=1):
        LOOKS.append(1)
        ids = list(stimuli['stimulus_id'])
        if self.task is None:
            return _neuroid(stimuli)
        if self.task == 'label':
            return BehavioralAssembly(np.array([LABELS[0]] * len(ids)), dims=['presentation'],
                                      coords={'stimulus_id': ('presentation', ids)})
        return BehavioralAssembly(np.full((len(ids), len(LABELS)), 0.5), dims=['presentation', 'choice'],
                                  coords={'stimulus_id': ('presentation', ids), 'choice': LABELS})


class _OldLanguageModel:
    """Stands in for an old language plugin. Keeps its task between calls; with a task set it
    reads each text as a prompt, so its activations differ (1 instead of 0)."""

    def __init__(self):
        self.region_layer_mapping = {'language_system': 'transformer.h.5'}
        self.neural_recordings, self.behavioral_task = [], None

    def identifier(self):
        return 'old-language'

    def start_neural_recording(self, target, recording_type):
        self.neural_recordings.append((target, recording_type))

    def start_behavioral_task(self, task):
        self.behavioral_task = task

    def digest_text(self, text):
        text = [text] if isinstance(text, str) else list(text)
        coords = {'stimulus': ('presentation', text),
                  'part_number': ('presentation', list(range(len(text))))}
        out = {}
        if self.behavioral_task is not None:
            out['behavior'] = BehavioralAssembly(np.array(['w'] * len(text)), coords=coords,
                                                 dims=['presentation'])
        if self.neural_recordings:
            value = float(self.behavioral_task is not None)
            out['neural'] = NeuroidAssembly(np.full((len(text), 2), value),
                                            dims=['presentation', 'neuroid'],
                                            coords={**coords, 'neuroid_id': ('neuroid', ['a', 'b'])})
        return out


class _Counter:
    """A stateful extractor: its output is how many stimuli it saw since its last reset."""
    identifier = 'counter'

    def __init__(self):
        self.seen = 0

    def reset(self):
        self.seen = 0

    def __call__(self, stimuli, layers, **kwargs):
        out = np.array([[float(self.seen + i)] for i in range(len(stimuli))])
        self.seen += len(stimuli)
        return NeuroidAssembly(out, dims=['presentation', 'neuroid'], coords={
            'stimulus_id': ('presentation', list(stimuli['stimulus_id'])),
            'neuroid_id': ('neuroid', ['late.0']), 'layer': ('neuroid', ['late'])})


def _rig():
    def respond(task, table):
        ids = list(table['stimulus_id'])
        if task.response is None:
            return BehavioralAssembly(np.array(['cat'] * len(ids)), dims=['presentation'],
                                      coords={'stimulus_id': ('presentation', ids)})
        from brainscore_core.response_formats import response_record
        return BehavioralAssembly(np.tile(np.eye(len(LABELS), dtype=np.float32)[0], (len(ids), 1)),
                                  dims=['presentation', 'choice'],
                                  coords={'stimulus_id': ('presentation', ids), 'choice': LABELS},
                                  attrs={'response': response_record('behavior:choice', 'button box')})
    return HumanSubject('rig', in_channels={'text'}, region_layer_map={'IT': 'array'},
                        present_fn=lambda table: None, record_fn=lambda region, table: _neuroid(table),
                        respond_fn=respond, out_channels={'behavior:choice'})


def _adapter(kind):
    if kind == 'vision':
        from brainscore_vision.compat.unified_adapter import VisionModelAdapter
        return VisionModelAdapter(_OldVisionModel())
    from brainscore_language.compat.unified_adapter import LanguageModelAdapter
    return LanguageModelAdapter(_OldLanguageModel())


# name: (a fresh subject, the input column it reads, the region it records)
SUBJECTS = {
    'brainscore-model': (lambda: BrainScoreModel(
        'one-rule', extractor={'text': _Counter()}, memory='trial_history',
        region_layer_map={'IT': LayerSelector('late', modality='text')}), 'sentence', 'IT'),
    'brainscore-model-answer': (lambda: BrainScoreModel(
        'one-rule-answer', in_channels={'text'},
        answer=lambda stimulus_row, instruction, label_set: label_set[0]), 'sentence', 'IT'),
    'brainscore-model-likelihood': (lambda: BrainScoreModel(
        'one-rule-likelihood', in_channels={'text'},
        likelihood=lambda prompt, candidates: {c: float(i) for i, c in enumerate(candidates)},
        answer_owners={'behavior:choice': ('likelihood', 'softmax_argmax'),
                                                 'behavior:probabilities': ('likelihood', 'softmax')}),
        'sentence', 'IT'),
    'vision-adapter': (lambda: _adapter('vision'), 'image_path', 'IT'),
    'language-adapter': (lambda: _adapter('language'), 'sentence', 'language_system'),
    'chance-baseline': (lambda: load_model('chance-baseline'), 'sentence', 'IT'),
    'provenance': (lambda: ProvenanceSubject(load_model('chance-baseline'), {'study': 'one-rule'}),
                   'sentence', 'IT'),
    'study': (lambda: _StudySubject(load_model('chance-baseline'), {'study': 'one-rule'}),
              'image_path', 'IT'),
    'human-harness': (_rig, 'sentence', 'IT'),
    'echo-agent': (lambda: EchoAgent('echo'), 'sentence', 'IT'),
}

# A subject class with no entry, and why.
STATED = {}


# A subject whose joint request is refused at the start of the session, before it reads an input.
STATED_JOINT = {
    'vision-adapter': 'a legacy vision model answers one output per look_at (the refusal names this; '
                      'the declarations do not state it, an open item)',
}


def _probes(region):
    """(outputs, task) of the probe sessions on one table."""
    return (
        (('behavior',), TaskContext(task_type='probabilities', label_set=LABELS)),
        (('behavior:choice',), TaskContext(response=Choice(LABELS))),
        (('behavior:probabilities',), TaskContext(response=Choice(LABELS))),
        ((f'neural:{region}',), TaskContext(task_type='passive')),
        (('behavior:text',), TaskContext(response=Text())),
    )


def _session(column, outputs, task, streaming=False):
    """A table session; a text answer comes in a conversation: two turns of one text, each
    answered on its ``respond`` signal."""
    if 'behavior:text' in outputs:
        turn = [begin_event(), StreamEvent('text', 'the cat', None), respond_event(), end_event('terminal')]
        return Session(turn * 2, list(outputs), task)
    return stimulus_session(_stimuli(column), list(outputs), task, streaming=streaming)


def _values(payload):
    """The values of an answer: of an assembly, or of a text answer's record."""
    return payload['values'] if isinstance(payload, dict) else getattr(payload, 'values', payload)


def _run(subject, column, outputs, task, streaming=False):
    """What one session gives: its answers by channel, or the kind of its refusal. The session
    must keep the source it was built with."""
    session = _session(column, outputs, task, streaming)
    source = session.source
    try:
        subject.participate(session)
        outcome = [(event.channel, str(_values(event.payload)))
                   for event in session.emitted]
    except Exception as refusal:
        outcome = type(refusal).__name__
    assert session.source is source, f'{subject.identifier} replaced the source of the session'
    return outcome


@pytest.mark.parametrize('name', sorted(SUBJECTS))
def test_a_second_session_gives_what_a_fresh_subject_gives(name):
    make, column, region = SUBJECTS[name]
    for a in _probes(region):
        for b in _probes(region):
            used = make()
            _run(used, column, *a)
            after_a, fresh = _run(used, column, *b), _run(make(), column, *b)
            assert after_a == fresh, f'{name}: {b[0]} after {a[0]} gives {after_a}; a fresh subject gives {fresh}'


@pytest.mark.parametrize('name', sorted(SUBJECTS))
def test_a_request_for_two_declared_outputs_gives_both(name):
    """Pre-flight passes a request when the subject declares every output; the run must then
    give each of them (rule 2.4). Bare ``behavior`` is the old-task path and is asked alone."""
    make, column, region = SUBJECTS[name]
    candidates = {'behavior:choice', 'behavior:probabilities', f'neural:{region}'} & set(make().out_channels)
    for pair in combinations(sorted(candidates), 2):
        LOOKS.clear()
        outcome = _run(make(), column, pair, TaskContext(response=Choice(LABELS)))
        if name in STATED_JOINT and outcome == 'ExecutionNotFeasibleError':
            assert LOOKS == [], f'{name}: {list(pair)} is refused after {len(LOOKS)} look_at calls'
            continue
        given = outcome if isinstance(outcome, str) else sorted({channel for channel, _ in outcome})
        assert given == sorted(pair), f'{name}: a request for {list(pair)} gives {given}'


@pytest.mark.parametrize('name', sorted(SUBJECTS))
def test_each_begin_clears_the_trial_history(name):
    """Two trials of one input, each in its own begin ... end, are answered alike."""
    make, column, region = SUBJECTS[name]
    for outputs, task in _probes(region):
        if not set(outputs) <= set(make().out_channels) or isinstance(_run(make(), column, outputs, task), str):
            continue          # not declared, or refused as a table session too: not this case
        session = _session(column, outputs, task, streaming=True)
        make().participate(session)
        rows = [np.asarray(_values(event.payload)) for event in session.emitted]
        if len(rows) == 1:         # one answer for both trials: its rows along the first axis
            rows = list(rows[0])
        assert len(rows) == 2, f'{name}: {list(outputs)} gives {len(rows)} answers for two trials'
        assert all(np.array_equal(row, rows[0]) for row in rows), f'{name}: {list(outputs)} gives {rows}'


# ---------------------------------------------------------------- the core drivers

class _Typed(Subject):
    """A typed subject with trial history: each answer is the number of ``process`` calls it
    made since its history was cleared. It clears its history only through ``on_begin``."""
    identifier = 'typed-history'
    in_channels = {'text', 'control', 'observation'}
    out_channels = {'neural:IT', 'behavior', 'motor'}
    traits = {'memory:trial_history'}

    def __init__(self, drive):
        self._drive, self.calls = drive, 0

    def clear(self):
        self.calls = 0

    def participate(self, session):
        self._drive(self, session)

    def start_task(self, task):
        pass

    def start_recording(self, target, time_bins=None):
        pass

    def process(self, stimuli):
        value, self.calls = self.calls, self.calls + 1
        if isinstance(stimuli, dict):
            return {'motor': value}
        if self._drive is DRIVERS['drive_neural_session_streaming'][0]:
            return _neuroid(stimuli, value)
        return BehavioralAssembly(np.full(len(stimuli), float(value)), dims=['presentation'],
                                  coords={'stimulus_id': ('presentation', list(stimuli['stimulus_id']))})


class _TwoSteps:
    """A discrete environment of two steps."""

    def reset(self, *, seed=None, options=None):
        self.steps = 0
        return {'observation': 0}

    def action_spec(self):
        return DiscreteSpec(10)

    def step(self, outputs):
        self.steps += 1
        return {'observation': self.steps}, 'terminal' if self.steps == 2 else None


def _table_answers(subject, outputs, task, streaming):
    session = stimulus_session(_stimuli('sentence'), outputs, task, streaming=streaming)
    subject.participate(session)
    return [float(v) for event in session.emitted for v in np.ravel(event.payload.values)]


def _first_actions(subject):
    """The first action of two episodes on one subject."""
    return [run_environment(subject, _TwoSteps())[0] for _ in range(2)]


_LABEL_TASK = TaskContext(task_type='probabilities', label_set=LABELS)
_PASSIVE = TaskContext(task_type='passive')

# name: (the participate body that runs the driver, what the subject answers in its sessions)
DRIVERS = {
    'drive_neural_session_streaming': (
        lambda s, session: drive_neural_session_streaming(s, session, on_begin=s.clear),
        lambda s: _table_answers(s, ['neural:IT'], _PASSIVE, True)),
    'drive_behavior_session_streaming': (
        lambda s, session: drive_behavior_session_streaming(s, session, on_begin=s.clear),
        lambda s: _table_answers(s, ['behavior'], _LABEL_TASK, True)),
    'drive_behavior_session': (
        lambda s, session: drive_behavior_session(s, session, on_begin=s.clear),
        lambda s: _table_answers(s, ['behavior'], _LABEL_TASK, False)
        + _table_answers(s, ['behavior'], _LABEL_TASK, False)),     # two sessions on one subject
    'drive_environment_session': (
        lambda s, session: drive_environment_session(s, session, on_begin=s.clear),
        _first_actions),
}

# A core driver with no begin to act on, and why.
STATED_DRIVERS = {
    'drive_neural_session': 'one process call for the whole table: no trial boundary reaches the subject',
    'drive_state_change_session': 'one change, no trials',
    'drive_typed_session': 'routes a session to the drivers above',
}


@pytest.mark.parametrize('name', sorted(DRIVERS))
def test_each_begin_clears_the_history_in_every_driver(name):
    drive, answers = DRIVERS[name]
    values = answers(_Typed(drive))
    assert values == [0] * len(values), f'{name}: the answers after each begin are {values}'


# ---------------------------------------------------------------- a policy's None

def _gymnasium():
    import gymnasium as gym
    import minigrid  # noqa: F401 -- registers MiniGrid environments.
    from brainscore.harnesses.gymnasium_harness import GymnasiumEnvironment, MINIGRID_ACTIONS
    return GymnasiumEnvironment(gym.make('MiniGrid-Empty-5x5-v0', render_mode='rgb_array'),
                                max_steps=2, action_descriptions=MINIGRID_ACTIONS)


def _grid_game():
    from brainscore.harnesses.grid_game import GridGameEnv, GridGameEnvironment
    return GridGameEnvironment(GridGameEnv(size=5, seed=0), max_steps=2)


def _replay():
    from brainscore.harnesses.human_replay import ReplayEnvironment
    return ReplayEnvironment([{'observation': 0}, {'observation': 1}], n_actions=3)


ENVIRONMENTS = {'gymnasium': _gymnasium, 'grid-game': _grid_game, 'replay': _replay}


@pytest.mark.parametrize('name', sorted(ENVIRONMENTS))
def test_a_policy_none_is_an_invalid_step(name):
    """An unreadable reply reaches the environment as None (rule 6.2): the step is invalid and
    the episode goes on."""
    policy = BrainScoreModel('none-policy', policy=lambda row: {'motor': None})
    actions = run_environment(policy, ENVIRONMENTS[name]())
    assert actions and all(action is None for action in actions)


def test_the_harness_entry_points_keep_a_none_step():
    from brainscore.harnesses.gymnasium_harness import play_gym_episode
    from brainscore.harnesses.human_replay import human_agreement, replay_actions
    policy = BrainScoreModel('none-policy', policy=lambda row: {'motor': None})
    assert play_gym_episode(policy, 'MiniGrid-Empty-5x5-v0', max_steps=2)['actions'] == [None, None]
    actions = replay_actions(policy, [{'observation': 0}], n_actions=2)
    assert actions == [None]
    assert human_agreement(actions, [[3, 1]])['raw'] == 0.0


# ---------------------------------------------------------------- the lists are complete

def _subject_classes():
    """The names of the Subject classes defined in the four packages, outside the model plugin
    folders (a plugin may define its own class)."""
    roots = [Path(module.__file__).parent for module in _packages()]
    found, sources = {'Subject'}, []
    for root in roots:
        for path in root.rglob('*.py'):
            if 'test' not in path.name and path.relative_to(root).parts[0] != 'models':
                sources += re.findall(r'^class (\w+)\(([^)]*)\)', path.read_text(), flags=re.M)
    while True:
        new = {name for name, bases in sources if name not in found
               and found & {base.strip().split('.')[-1] for base in bases.split(',')}}
        if not new:
            return found - {'Subject'}
        found |= new


def _packages():
    import brainscore
    import brainscore_core
    import brainscore_language
    import brainscore_vision
    return brainscore, brainscore_core, brainscore_vision, brainscore_language


def test_every_subject_class_and_driver_has_an_entry():
    here = 'tests/test_models/test_one_rule_for_every_subject.py'
    covered = {type(make()).__name__ for make, _, _ in SUBJECTS.values()}
    missing = _subject_classes() - covered - set(STATED)
    assert not missing, (f'Subject classes {sorted(missing)} have no entry: add one to SUBJECTS in {here} '
                         f'(or, with the reason, to STATED).')
    drivers = {name for name in dir(streaming_helpers) if name.startswith('drive_')}
    missing = drivers - set(DRIVERS) - set(STATED_DRIVERS)
    assert not missing, (f'Core drivers {sorted(missing)} have no entry: add one to DRIVERS in {here} '
                         f'(or, with the reason, to STATED_DRIVERS).')


def test_a_new_model_plugin_class_needs_no_entry(tmp_path, monkeypatch):
    """A model plugin may define its own subject class; the entry list names library classes only."""
    import sys
    import types
    for path, text in {'models/plug/model.py': 'class Plug(Subject): pass\n',
                       'harnesses/rig.py': 'class Rig(Subject): pass\n'}.items():
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_text(text)
    package = types.SimpleNamespace(__file__=str(tmp_path / '__init__.py'))
    monkeypatch.setattr(sys.modules[__name__], '_packages', lambda: (package,))
    assert _subject_classes() == {'Rig'}


# ---------------------------------------------------------------- the adapters' time

@pytest.mark.parametrize('kind,column,region', [('vision', 'image_path', 'IT'),
                                                ('language', 'sentence', 'language_system')])
def test_a_whole_table_answer_has_the_time_the_core_driver_gives(kind, column, region):
    """None, as the core driver gives it (rule 4.2)."""
    def times(run):
        stimuli = _stimuli(column)
        stimuli['t_ms'] = [10.0, 20.0]
        session = stimulus_session(stimuli, [f'neural:{region}'], _PASSIVE)
        run(_adapter(kind), session)
        return [event.t_ms for event in session.emitted]

    assert times(lambda subject, session: subject.participate(session)) == times(drive_neural_session)


def test_an_answering_part_in_an_environment_is_refused_before_compute():
    """An answering part answers a table; a closed loop on a behavior action needs a policy.
    The refusal comes from the plan, before the first step."""
    from brainscore_core.compatibility import ExecutionNotFeasibleError
    calls = []
    model = BrainScoreModel('answers', in_channels={'observation'},
                            answer=lambda stimulus_row, instruction, label_set: calls.append(1) or label_set[0])
    with pytest.raises(ExecutionNotFeasibleError, match='policy'):
        run_environment(model, _TwoSteps(), action_channel='behavior:choice',
                        task=TaskContext(response=Choice(LABELS)))
    assert calls == []


def test_a_recording_of_several_pictures_in_one_trial_is_refused_before_compute():
    """Only the first column of each kind feeds a channel (a stated limit): a recording of a
    trial that presents two pictures is refused before any pass."""
    from brainscore_core.compatibility import ExecutionNotFeasibleError
    stimuli = StimulusSet({'stimulus_id': ['s0'], 'image_path': ['a.png'], 'image_path_2': ['b.png']})
    stimuli.identifier = 'one-rule-two-pictures'
    counter = _Counter()
    model = BrainScoreModel('pictures', extractor={'vision': counter},
                            region_layer_map={'IT': LayerSelector('late', modality='vision')})
    task = TaskContext(task_type='passive', present=('image:image_path', 'image:image_path_2'))
    with pytest.raises(ExecutionNotFeasibleError, match='first column'):
        model.participate(stimulus_session(stimuli, ['neural:IT'], task))
    assert counter.seen == 0
