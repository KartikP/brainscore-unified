"""Built-in session handlers for BrainScoreModel's parts.

Each part of a BrainScoreModel (region map, answer or readout, policy,
intervention) handles some channels. These capabilities declare those
channels and run the matching sessions, registered the same way as any added
capability, so ``participate``, ``check_session_support`` and the declared
channels all read one registry. They sort after added capabilities
(``order`` 10000), so an added capability that claims the same outputs wins.
"""

from contextlib import ExitStack

import pandas as pd

from .base import Capability
from .registry import register_capability
from brainscore_core import streaming_helpers as _drivers
from brainscore_core.streaming import StreamEvent, parse_channel
from brainscore_core.responses import response_channels


class _SessionCapability(Capability):
    """A capability that only runs sessions; it handles no ``process`` events."""

    order = 10000

    def handles(self, model, event, **kwargs) -> bool:
        del model, event, kwargs
        return False

    def process(self, model, event, **kwargs):  # pragma: no cover - never handled
        raise NotImplementedError(self.identifier)


class NeuralSessions(_SessionCapability):
    identifier = 'neural-sessions'

    def enabled_for(self, model) -> bool:
        # A region map names where to record; only a wired extractor can
        # record there, so a map without one declares nothing (rule 2.4).
        return bool(model.region_layer_map) and bool(model._extractors)

    def declared_output_channels(self, model):
        return {f'neural:{region}' for region in model.region_layer_map}

    def supports_session(self, model, channels) -> bool:
        return bool(channels) and all(c.startswith('neural:') for c in channels)

    def participate(self, model, session) -> None:
        task = getattr(session, 'task_context', None)
        if task is not None and task.task_type != 'passive':
            # The session's task decides what the subject does (rule 4.1):
            # record during the task; the answers are not sent.
            drive_behavior_and_neural_session(model, session)
            return
        # A passive session records activity even after a task was configured.
        model._begin_episode()
        # A subject that keeps history is shown one trial at a time (same scores,
        # different arrival), so each independent trial can begin empty.
        per_trial = (_keeps_history(model)
                     and (task is None or task.trial_protocol == 'independent'))
        with model._behavioral.suspended():
            from .._model.channels import run_neural_channels

            def runner(stimuli, channel_regions, time_bins, multi_modality):
                return run_neural_channels(model, stimuli, channel_regions,
                                           time_bins, multi_modality)
            if getattr(session, 'streaming', False) or per_trial:
                _drivers.drive_neural_session_streaming(
                    model, session, channel_runner=runner, on_begin=model._begin_episode)
            else:
                _drivers.drive_neural_session(model, session, allow_multi_modality=True,
                                              channel_runner=runner)


class BehaviorSessions(_SessionCapability):
    identifier = 'behavior-sessions'

    def enabled_for(self, model) -> bool:
        return (model._behavioral_readout_layer is not None
                or model._generation_fn is not None
                or model._reply_fn is not None or model._likelihood_fn is not None)

    def declared_input_channels(self, model):
        replies = (model._generation_fn, model._reply_fn, model._likelihood_fn)
        return {'instruction'} if any(part is not None for part in replies) else set()

    def declared_output_channels(self, model):
        # Each part declares the kinds of answer it can give (rule 2.4).
        channels = {'behavior'}
        if model._behavioral_readout_layer is not None:
            channels |= {'behavior:choice', 'behavior:probabilities', 'behavior:rating', 'behavior:rt'}
        if model._generation_fn is not None:
            channels |= {'behavior:choice'}
        if model._reply_fn is not None:
            channels |= {'behavior:choice', 'behavior:rating', 'behavior:confidence',
                         'behavior:text'}
        if model._likelihood_fn is not None:
            channels |= {'behavior:choice', 'behavior:probabilities', 'behavior:rt'}
        return channels

    def supports_session(self, model, channels) -> bool:
        # Behavior alone, or behavior plus regions recorded during the same answers.
        behavior = {c for c in channels if c == 'behavior' or c.startswith('behavior:')}
        recorded = set(channels) - behavior
        return bool(behavior) and behavior <= self.declared_output_channels(model) and recorded <= {
            f'neural:{region}' for region in model.region_layer_map}

    def participate(self, model, session) -> None:
        requested = set(_drivers.requested_output_channels(session))
        task = getattr(session, 'task_context', None)
        if requested != {'behavior'} or (task is not None and task.response is not None):
            drive_behavior_and_neural_session(model, session)
        elif getattr(session, 'streaming', False):
            _drivers.drive_behavior_session_streaming(model, session,
                                                      on_begin=model._begin_episode)
        else:
            _drivers.drive_behavior_session(model, session, on_begin=model._begin_episode)


class MotorSessions(_SessionCapability):
    """Closed loop: read a row per step, call the policy, answer on its output channels.

    ``neural:<region>`` may be requested with the policy's outputs: the regions
    are recorded during the same forward pass that produced the action (the
    extractor's ``record(layers)`` hooks, or hooks on ``model=``), never a second
    pass. A policy with no extractor wired declares no ``neural:*`` channel.
    """

    identifier = 'motor-sessions'

    def enabled_for(self, model) -> bool:
        return model._action_fn is not None

    def declared_input_channels(self, model):
        return set(model._policy_inputs)

    def declared_output_channels(self, model):
        return set(model._policy_outputs)

    def supports_session(self, model, channels) -> bool:
        channels = set(channels)
        recorded = (_NEURAL.declared_output_channels(model)
                    if _NEURAL.enabled_for(model) else set())
        outputs = set(model._policy_outputs)
        return bool(channels & outputs) and channels <= outputs | recorded

    def participate(self, model, session) -> None:
        requested = _drivers.requested_output_channels(session)
        regions = [parse_channel(c)[1] for c in requested if parse_channel(c)[0] == 'neural']
        if not regions:
            _drivers.drive_environment_session(model, session, on_begin=model._begin_episode)
            return
        recorded = _RecordedPolicy(model, session, regions)
        with recorded:
            _drivers.drive_environment_session(recorded, session, on_begin=recorded.begin)


class _RecordedPolicy:
    """``process(row)`` that also records regions from the forward pass the policy ran."""

    def __init__(self, model, session, regions) -> None:
        self.model, self.session, self.regions = model, session, regions
        self.layers = {region: model._behavioral._region_layers(region) for region in regions}
        self.step = 0
        self._stack = ExitStack()

    def __enter__(self) -> '_RecordedPolicy':
        recordings = self.model._behavioral._open_recordings(None, self.layers)
        self.live = {modality: self._stack.enter_context(recording)
                     for modality, recording in recordings.items()}
        return self

    def __exit__(self, *exc) -> bool:
        return self._stack.__exit__(*exc)

    def begin(self) -> None:
        self.step = 0
        self.model._begin_episode()

    def process(self, row):
        outputs = self.model.process(row)
        trial = {modality: recording.take() for modality, recording in self.live.items()}
        stimuli = pd.DataFrame({'stimulus_id': [f'step_{self.step}']})
        for region, assembly in self.model._behavioral._trial_regions(
                [trial], stimuli, self.layers).items():
            self.session.respond(StreamEvent(
                channel=f'neural:{region}', payload=assembly, t_ms=float(self.step),
                meta={'driver': 'interact', 'record': region, 'during': 'policy'}))
        self.step += 1
        return outputs


class PerturbationSessions(_SessionCapability):
    identifier = 'perturbation-sessions'

    def enabled_for(self, model) -> bool:
        return model._state_change_fn is not None

    def declared_input_channels(self, model):
        return {'lesion', 'stimulation'}

    def declared_output_channels(self, model):
        return {'perturbation'}

    def supports_session(self, model, channels) -> bool:
        return set(channels) == {'perturbation'}

    def participate(self, model, session) -> None:
        _drivers.drive_state_change_session(model, session)


_NEURAL = NeuralSessions()
for _capability in (_NEURAL, BehaviorSessions(), MotorSessions(), PerturbationSessions()):
    register_capability(_capability)


def _keeps_history(model) -> bool:
    """True when the model declares ``memory:trial_history``: only then can a trial see an earlier one."""
    return 'memory:trial_history' in model.traits


def drive_behavior_and_neural_session(model, session, driver: str = "interact") -> None:
    """Record ``neural:<region>`` while the subject does the session's task.

    ``behavior`` is sent too when requested.

    Each stimulus is shown once: the regions are recorded during the same
    forward pass that produced the answer (see
    ``BehavioralReadout.respond_and_record``). Only ``BrainScoreModel`` runs
    this; it is reached through its session capability.
    """
    task_context = getattr(session, "task_context", None)
    if task_context is None:
        raise ValueError("A behavior session needs session.task_context.")
    time_bins = getattr(session, "time_bins", None)
    if time_bins is not None and len(time_bins) > 1:
        raise NotImplementedError(
            "Recording during a task returns one time bin; request at most one.")
    requested = _drivers.requested_output_channels(session)
    regions = [parse_channel(c)[1] for c in requested if parse_channel(c)[0] == "neural"]
    answers = [c for c in requested if parse_channel(c)[0] == "behavior"]
    # Refuse answers the task cannot give before any setup or compute.
    if task_context.response is None and set(answers) - {"behavior"}:
        raise ValueError(f"{answers} need TaskContext(response=...); task_type "
                         f"{task_context.task_type!r} answers on 'behavior'.")
    if task_context.response is not None:
        given = set(response_channels(task_context.response))
        unanswered = sorted(set(answers) - given - {"behavior"})
        if unanswered:
            raise ValueError(
                f"{unanswered} requested, but the task's response "
                f"{type(task_context.response).__name__} answers only on "
                f"{sorted(given)}; ask for those or change the response.")
    _drivers.refuse_variance(model, session)
    model.start_task(task_context)
    stimuli = _drivers.behavior_stimuli_from_session(session, task_context)
    if stimuli is None:
        raise ValueError("Recording during a task needs stimuli to present.")
    # A streaming session gets each trial's outputs as soon as that trial ends;
    # otherwise the whole set is answered and sent at once.
    # A model without the trait keeps no history, so it can answer the whole table at
    # once; one that keeps history is answered trial by trial so a begin can fall
    # between trials.
    streaming = getattr(session, "streaming", False)
    batches = ([(index, _drivers.preserve_stimulus_set_type(stimuli, stimuli.iloc[[index]]))
                for index in range(len(stimuli))]
               if streaming or _keeps_history(model) else [(0, stimuli)])
    for index, batch in batches:
        if index == 0 or task_context.trial_protocol == 'independent':
            model._begin_episode()
        if task_context.response is not None:
            behavior, neural = model._behavioral.respond(batch, answers, regions)
        else:
            plain, neural = model._behavioral.respond_and_record(batch, regions)
            behavior = {"behavior": plain} if answers else {}
        for channel in answers:
            session.respond(StreamEvent(channel=channel, payload=behavior[channel], t_ms=float(index),
                                     meta={"driver": driver, "task_type": task_context.task_type}))
        for region in regions:
            session.respond(StreamEvent(channel=f"neural:{region}", payload=neural[region], t_ms=float(index),
                                     meta={"driver": driver, "record": region, "during": "behavior"}))

