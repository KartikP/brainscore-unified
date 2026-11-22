"""Human subject scaffold for the v1.5 subject-agnostic interface.

The specification describes a *subject*, not specifically a model. A human
subject realizes the same contract on a biological subject: it presents inputs
(a screen for vision, speakers for audio, a real coil for stimulation) and
records outputs (electrodes or fMRI for neural, a button box or eye tracker for
behavior). The same benchmark code that scores a model scores a human.

This module is a scaffold. It runs sessions through its typed methods
(``drive_typed_session``) and routes each typed method to pluggable hardware I/O callables that a real deployment fills in.
With no hardware attached, the callables default to raising, making the
"not yet wired to hardware" boundary explicit; tests inject in-memory mocks.
Nothing here drives real hardware. The point is that the specification already
accommodates a human with no change to the contract: ``start_recording`` is a
recording electrode, ``start_task`` is a response collection, ``process`` is a
stimulus presentation.
"""
from typing import Any, Callable, Dict, List, Optional, Set, Union

from brainscore_core.contract import Subject, TaskContext
from brainscore_core.io_catalog import form_problem
from brainscore_core.streaming import StreamEvent


def _no_hardware(*_args, **_kwargs):
    raise NotImplementedError(
        "HumanSubject has no hardware attached. Inject present_fn / record_fn / "
        "respond_fn to drive a real subject, or use mocks in tests."
    )


class HumanSubject(Subject):
    """A ``Subject`` backed by a biological participant via pluggable I/O.

    :param identifier: a label for the participant or session.
    :param in_channels: input channels the rig can present (default images).
    :param region_layer_map: maps a brain region to a recording channel (an
        electrode-array id, an fMRI sequence). Mirrors a model's
        ``region_layer_map`` so benchmark code is byte-identical.
    :param present_fn: ``callable(stimulus)`` presenting one input to the subject.
    :param record_fn: ``callable(region, stimulus) -> measurement`` reading neural
        activity from the configured region.
    :param respond_fn: ``callable(task, stimulus) -> response`` collecting
        a behavioral response.
    :param out_channels: the behavior channels the rig gives with ``respond_fn``,
        for example ``{'behavior:choice'}``. For a stated format the session's task
        and its stimuli table go to ``respond_fn``, and its return value is the answer
        on that channel (an answer carries its record, ``attrs['response']``). A
        benchmark that requests a format the rig did not state is not applicable
        before any stimulus is presented. A rig that states none declares only its
        ``neural:<region>`` outputs: a session built directly on bare ``behavior``
        still gets the return value of ``respond_fn`` on ``behavior``.
    """

    def __init__(self, identifier: str,
                 in_channels: Optional[Set[str]] = None,
                 region_layer_map: Optional[Dict[str, str]] = None,
                 present_fn: Optional[Callable] = None,
                 record_fn: Optional[Callable] = None,
                 respond_fn: Optional[Callable] = None,
                 out_channels: Optional[Set[str]] = None):
        self._identifier = identifier
        self._in_channels = set(in_channels or {"vision:image"})
        self._region_layer_map = dict(region_layer_map or {})
        self._present_fn = present_fn or _no_hardware
        self._record_fn = record_fn or _no_hardware
        self._respond_fn = respond_fn or _no_hardware
        self._behavior_channels = set(out_channels or ())
        for channel in self._behavior_channels:
            if channel.split(':', 1)[0] != 'behavior' or form_problem(channel):
                raise ValueError(
                    f"HumanSubject states {channel!r} in out_channels: state only the behavior "
                    "channels the rig gives, for example 'behavior:choice'; a neural output "
                    "comes from region_layer_map.")
        self._recording_region: Optional[Union[str, List[str]]] = None
        self._task: Optional[TaskContext] = None

    @property
    def identifier(self) -> str:
        return self._identifier

    @property
    def region_layer_map(self) -> Dict[str, str]:
        return self._region_layer_map

    @property
    def in_channels(self) -> Set[str]:
        return self._in_channels

    @property
    def out_channels(self) -> Set[str]:
        """``neural:<region>`` for every mapped region, and the stated behavior channels."""
        return {f"neural:{region}" for region in self._region_layer_map} | self._behavior_channels

    def participate(self, session) -> None:
        """Run ``session`` through the typed methods below. The recording region and the task
        it sets hold for this call only: the next session starts from none.

        Limit: one stated format per session (``respond_fn`` is not told the channel); a
        request for two is refused before any stimulus is presented."""
        from brainscore_core.compatibility import ExecutionNotFeasibleError
        from brainscore_core.streaming_helpers import (
            drain_session_inputs, drive_typed_session, refuse_variance, session_stimuli)
        outputs = list(session.outputs or ())
        stated = [channel for channel in outputs if channel in self._behavior_channels - {'behavior'}]
        if stated:
            # The stated format, and each requested region recorded during the same presentation
            # (rule 4.5): the table is presented once.
            if len(stated) > 1:
                raise ExecutionNotFeasibleError(
                    f"HumanSubject {self.identifier!r} gives one stated format per session from "
                    f"respond_fn; this session asks for {stated}. Ask for each in its own session.",
                    channels=stated)
            refuse_variance(self, session)
            stimuli = session_stimuli(session, drain_session_inputs(session))
            self._present_fn(stimuli)
            for channel in outputs:
                answer = (self._respond_fn(session.task, stimuli) if channel in stated
                          else self._record_fn(channel.split(':', 1)[1], stimuli))
                session.respond(StreamEvent(channel, answer, None, {}))
            return
        try:
            drive_typed_session(self, session)
        finally:
            self.reset()

    def start_recording(self, recording_target, time_bins=None, recording_type=None) -> None:
        self._recording_region = recording_target

    def start_task(self, task: TaskContext) -> None:
        self._task = task

    def reset(self) -> None:
        self._recording_region = None
        self._task = None

    def process(self, input_event) -> Any:
        """Present the input to the subject, then read the configured output.

        A real harness iterates the stimulus set and aggregates per-stimulus
        measurements; the scaffold presents the input event and routes once to
        the recording or response channel, demonstrating that the benchmark-side
        call sequence (``start_recording`` / ``start_task`` then ``process``) is
        identical to the model path.
        """
        self._present_fn(input_event)
        if self._recording_region is not None:
            return self._record_fn(self._recording_region, input_event)
        if self._task is not None:
            return self._respond_fn(self._task, input_event)
        raise RuntimeError(
            "A HumanSubject answers in a session: request 'neural:<region>' to "
            "record activity, or give the session a TaskContext to respond."
        )
