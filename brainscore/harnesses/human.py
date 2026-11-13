"""Human harness scaffold for the v1.5 subject-agnostic interface.

The specification describes a *subject*, not specifically a model. A human
harness realizes the same contract on a biological subject: it presents inputs
(a screen for vision, speakers for audio, a real coil for stimulation) and
records outputs (electrodes or fMRI for neural, a button box or eye tracker for
behavior). The same benchmark code that scores a model scores a human.

This module is a scaffold. It uses the ``UnifiedModel`` compatibility base and routes
each typed method to pluggable hardware I/O callables that a real deployment fills in.
With no hardware attached, the callables default to raising, making the
"not yet wired to hardware" boundary explicit; tests inject in-memory mocks.
Nothing here drives real hardware. The point is that the specification already
accommodates a human with no change to the contract: ``start_recording`` is a
recording electrode, ``start_task`` is a response collection, ``process`` is a
stimulus presentation.
"""
from typing import Any, Callable, Dict, List, Optional, Set, Union

from brainscore_core.contract import UnifiedModel, TaskContext


def _no_hardware(*_args, **_kwargs):
    raise NotImplementedError(
        "HumanHarness has no hardware attached. Inject present_fn / record_fn / "
        "respond_fn to drive a real subject, or use mocks in tests."
    )


class HumanHarness(UnifiedModel):
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
    """

    def __init__(self, identifier: str,
                 in_channels: Optional[Set[str]] = None,
                 region_layer_map: Optional[Dict[str, str]] = None,
                 present_fn: Optional[Callable] = None,
                 record_fn: Optional[Callable] = None,
                 respond_fn: Optional[Callable] = None):
        self._identifier = identifier
        self._in_channels = set(in_channels or {"vision:image"})
        self._region_layer_map = dict(region_layer_map or {})
        self._present_fn = present_fn or _no_hardware
        self._record_fn = record_fn or _no_hardware
        self._respond_fn = respond_fn or _no_hardware
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
