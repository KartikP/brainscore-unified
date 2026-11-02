"""Direct attachment uses the experiment runner's validation and cleanup."""
from contextlib import contextmanager
from os import PathLike
from typing import Any, Iterator, Sequence

from brainscore.experiments.runner import Experiment, RunContext, Tool


class _DirectProtocol:
    identifier = 'tool-scope'
    conditions = ('default',)
    trials = ('direct',)

    def validate(self, subject: Any) -> None:
        pass

    def describe(self) -> dict[str, str]:
        return {'execution': 'caller-owned', 'reset_policy': 'caller-owned',
                'recording': 'tool events; use SessionProtocol for session inputs and outputs'}


@contextmanager
def attach_tools(subject: Any, *, tools: Sequence[Tool], output_dir: str | PathLike,
                 instrumentation: Any = None, metadata: dict | None = None) -> Iterator[RunContext]:
    """Attach several named tools around caller-owned model/session execution.

    Order and validate tools exactly as in Experiment. Tools detach in reverse
    order on success, failure or interruption. This does not reset the subject,
    serialize session inputs/outputs, or fit tools on evaluation inputs.
    Add RecordInputsOutputs to persist emitted measurements.
    """
    experiment = Experiment(subject=subject, protocol=_DirectProtocol(), tools=tools,
                            output_dir=output_dir, instrumentation=instrumentation,
                            metadata=metadata)
    with experiment.open() as context:
        with context.trial('direct'):
            yield context
