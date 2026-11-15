"""The session parts of BrainScoreModel, one per part that answers.

Each part of a BrainScoreModel (region map, answer or readout, policy,
intervention) answers some channels. These capabilities declare those
channels and run the matching sessions; they are registered the same way as
any added capability, so ``participate``, ``check_session_support`` and the
declared channels all read one registry. They sort after added capabilities
(``order`` 10000), so an added capability that claims the same outputs wins.
Each one calls its part directly: ``record`` for activity, ``respond`` for a
task, ``act`` for a policy step, ``Interventions.process`` for a change.
"""

from contextlib import ExitStack

import pandas as pd

from .base import Capability
from .._model.behavioral import _declared_formats
from .._model.dispatch import act
from ..perturbation import INTERVENTION_CHANNELS
from .._model.recording import (open_recordings, planned_recordings, records_while_answering,
                                region_selector, trial_regions)
from .registry import register_capability
from brainscore_core import streaming_helpers as _drivers
from brainscore_core.streaming import StreamEvent, parse_channel

# After every added capability (default order 1000), so an added one that
# claims the same outputs runs the session.
_SESSION_ORDER = 10000


class NeuralSessions(Capability):
    order = _SESSION_ORDER
    identifier = 'neural-sessions'

    def enabled_for(self, subject) -> bool:
        # A region map names where to record; only a wired extractor can
        # record there, so a map without one declares nothing (rule 2.4).
        return bool(subject.region_layer_map) and bool(subject._extractors)

    def declared_output_channels(self, subject):
        return {f'neural:{region}' for region in subject.region_layer_map}

    def supports_session(self, subject, channels) -> bool:
        return bool(channels) and set(channels) <= self.declared_output_channels(subject)

    def participate(self, subject, session) -> None:
        plan = subject._session_plan
        if plan.task_path:
            # The session's task decides what the subject does (rules 4.1, 4.5):
            # record during the task; the answers are not sent. A subject with
            # no part for this task was refused by the session's plan, before
            # any compute.
            drive_behavior_and_neural_session(subject, session)
            return
        # A passive session records activity even after a task was configured.
        subject._begin_episode()
        # A subject that keeps history is shown one trial at a time (same scores,
        # different arrival), so each independent trial can begin empty (the plan's delivery).
        from .._model.channels import run_neural_channels
        passes = plan.regions_per_pass
        with subject._behavioral.suspended():
            def runner(stimuli, channel_regions, time_bins, multi_modality):
                return run_neural_channels(subject, stimuli, channel_regions,
                                           time_bins, multi_modality, passes)
            if plan.delivery != 'table':
                _drivers.drive_neural_session_streaming(
                    subject, session, channel_runner=runner, on_begin=subject._begin_episode)
            else:
                _drivers.drive_neural_session(subject, session, allow_multi_modality=True,
                                              channel_runner=runner)


class BehaviorSessions(Capability):
    order = _SESSION_ORDER
    identifier = 'behavior-sessions'

    def enabled_for(self, subject) -> bool:
        return (subject._behavioral_readout_layer is not None
                or subject._answer is not None
                or subject._reply is not None or subject._likelihood is not None)

    def declared_input_channels(self, subject):
        replies = (subject._answer, subject._reply, subject._likelihood)
        return {'instruction'} if any(part is not None for part in replies) else set()

    def declared_output_channels(self, subject):
        # The formats with an owner (rules 2.4, 4.11); none by convertibility alone.
        return _declared_formats(subject)

    def supports_session(self, subject, channels) -> bool:
        # Behavior alone, or behavior plus regions recorded during the same answers.
        # A readout records any region from its own extraction; an answering
        # part records only where the wiring can record during its pass.
        behavior = {c for c in channels if c == 'behavior' or c.startswith('behavior:')}
        recorded = set(channels) - behavior
        readout = subject._behavioral_readout_layer is not None and bool(subject._extractors)
        return bool(behavior) and behavior <= self.declared_output_channels(subject) and recorded <= {
            f'neural:{region}' for region in subject.region_layer_map
            if readout or records_while_answering(subject, region)}

    def participate(self, subject, session) -> None:
        if subject._session_plan.task_path:
            drive_behavior_and_neural_session(subject, session)
            return
        # A legacy task answered on 'behavior' alone is packaged by core's typed
        # behavior driver, as for every legacy adapter: one event for the table
        # (one per trial when streamed), one begin. The answer comes from the
        # same respond() as every other behavior session.
        typed = _LegacyTaskAnswers(subject)
        if subject._session_plan.delivery == 'stream':
            _drivers.drive_behavior_session_streaming(typed, session, on_begin=subject._begin_episode)
        else:
            _drivers.drive_behavior_session(typed, session, on_begin=subject._begin_episode)


class _LegacyTaskAnswers:
    """What core's typed behavior driver calls: ``start_task`` and ``process``.

    ``process`` answers through ``BehavioralReadout.respond``; it is not the
    model's ``process``, which no longer answers behavior.
    """

    def __init__(self, subject) -> None:
        self.subject = subject

    @property
    def identifier(self) -> str:
        return self.subject.identifier

    def start_task(self, task) -> None:
        # The session's plan chose the part of bare behavior (formats: ['behavior']).
        self.subject.start_task(task, formats=['behavior'])

    def process(self, stimuli):
        behavior, _ = self.subject._behavioral.respond(stimuli, ['behavior'])
        return behavior['behavior']


class MotorSessions(Capability):
    """Closed loop: read a row per step, call the policy, answer on its output channels.

    ``neural:<region>`` may be requested with the policy's outputs: the regions
    are recorded during the same forward pass that produced the action (the
    extractor's ``record(layers, None)`` hooks, or hooks on ``model=``), never a second
    pass. A policy with no extractor wired declares no ``neural:*`` channel.
    """

    identifier = 'motor-sessions'
    order = _SESSION_ORDER

    def enabled_for(self, subject) -> bool:
        return subject._policy is not None

    def declared_input_channels(self, subject):
        return set(subject._policy_in_channels)

    def declared_output_channels(self, subject):
        return set(subject._policy_out_channels)

    def supports_session(self, subject, channels) -> bool:
        channels = set(channels)
        recorded = ({f'neural:{region}' for region in subject.region_layer_map
                     if records_while_answering(subject, region)}
                    if _NEURAL.enabled_for(subject) else set())
        outputs = set(subject._policy_out_channels)
        return bool(channels & outputs) and channels <= outputs | recorded

    def participate(self, subject, session) -> None:
        requested = _drivers.session_outputs(session)
        regions = [parse_channel(c)[1] for c in requested if parse_channel(c)[0] == 'neural']
        with _PolicySteps(subject, session, regions) as steps:
            _drivers.drive_environment_session(steps, session, on_begin=steps.begin)


class _PolicySteps:
    """What the closed-loop driver calls: ``process(row)`` asks the policy (``act``).

    With regions requested, each step also records them from the forward pass
    the policy ran.
    """

    def __init__(self, subject, session, regions) -> None:
        self.subject, self.session = subject, session
        self.layers = {region: list(region_selector(subject, region).layer_paths)
                       for region in regions}
        self.step = 0
        self._stack = ExitStack()

    def __enter__(self) -> '_PolicySteps':
        recordings = open_recordings(self.subject, None, self.layers,
                                     planned_recordings(self.subject, self.layers))
        try:
            self.live = {regions: self._stack.enter_context(recording)
                         for regions, recording in recordings.items()}
        except BaseException:
            self._stack.close()
            raise
        return self

    def __exit__(self, *exc) -> bool:
        return self._stack.__exit__(*exc)

    def begin(self) -> None:
        self.step = 0
        self.subject._begin_episode()

    def process(self, row):
        outputs = act(self.subject, row)
        if not self.layers:
            return outputs
        trial = {regions: recording.take(None) for regions, recording in self.live.items()}
        stimuli = pd.DataFrame({'stimulus_id': [f'step_{self.step}']})
        for region, assembly in trial_regions(self.subject, [trial], stimuli, self.layers).items():
            self.session.respond(StreamEvent(
                channel=f'neural:{region}', payload=assembly, t_ms=float(self.step),
                meta={'driver': 'interact', 'record': region, 'during': 'policy'}))
        self.step += 1
        return outputs


class InterventionSessions(Capability):
    """Take an intervention (a ``StateChange``) on each intervention channel; answer on ``perturbation``.

    Enabled by ``BrainScoreModel(intervention=...)``. ``apply_state_change`` and
    the ``Intervene`` tool send the change through this session; the driver
    calls the subject's ``Interventions`` directly.
    """

    identifier = 'intervention-sessions'
    order = _SESSION_ORDER

    def enabled_for(self, subject) -> bool:
        return subject._intervention is not None

    def declared_input_channels(self, subject):
        return set(INTERVENTION_CHANNELS)

    def declared_output_channels(self, subject):
        return {'perturbation'}

    def supports_session(self, subject, channels) -> bool:
        return set(channels) == {'perturbation'}

    def participate(self, subject, session) -> None:
        _drivers.drive_state_change_session(subject._interventions, session)


_NEURAL, _BEHAVIOR, _MOTOR = NeuralSessions(), BehaviorSessions(), MotorSessions()
for _capability in (_NEURAL, _BEHAVIOR, _MOTOR, InterventionSessions()):
    register_capability(_capability)


def drive_behavior_and_neural_session(subject, session, driver: str = "interact") -> None:
    """Record ``neural:<region>`` while the subject does the session's task.

    ``behavior`` is sent too when requested.

    Each stimulus is shown once: the regions are recorded during the same
    forward pass that produced the answer (see
    ``BehavioralReadout.respond``). Only ``BrainScoreModel`` runs
    this; it is reached through its session capability. The session's plan
    refused, before any setup or compute, what cannot run (no task, several time
    bins, a format the task does not give, trial-to-trial variance).
    """
    plan = subject._session_plan
    task = session.task
    regions, answers = list(plan.regions), list(plan.answers)
    # The plan chose the answering part for the requested formats. The requested outputs decide whether the readout is fitted (as they decide the parts).
    subject.start_task(task, formats=answers)
    # A streaming session reads one trial, answers it and sends its outputs, then
    # reads the next. Otherwise the inputs are read first. A subject without the
    # trait keeps no history, so it can answer the whole table at once; one that
    # keeps history is answered trial by trial so a begin can fall between trials.
    if plan.delivery == 'stream':
        batches = enumerate(_drivers.stream_trials(session, task))
    else:
        stimuli = _drivers.behavior_stimuli_from_session(session, task)
        if stimuli is None:
            raise ValueError("Recording during a task needs stimuli to present.")
        batches = ([(index, _drivers.preserve_stimulus_set_type(stimuli, stimuli.iloc[[index]]))
                    for index in range(len(stimuli))]
                   if plan.delivery == 'per_trial' else [(0, stimuli)])
    for index, batch in batches:
        if index == 0 or task.trial_memory == 'none':
            subject._begin_episode()
        behavior, neural = subject._behavioral.respond(batch, answers, regions)
        # A value that is a StreamEvent keeps its meta: a reply's extra data (send_outputs).
        _drivers.send_outputs(session, {channel: behavior[channel] for channel in answers},
                              float(index), {"driver": driver, "task_type": task.task_type})
        for region in regions:
            session.respond(StreamEvent(channel=f"neural:{region}", payload=neural[region], t_ms=float(index),
                                     meta={"driver": driver, "record": region, "during": "behavior"}))

