"""What every adapter for an external experiment library shares.

An adapter maps the calls of an external library (CogGym, EWoK, NodeKit) onto
the common channels (Contract Rules 3.8, 3.9). Only two things differ from one
library to another, and they stay in that library's adapter: how the library
states a trial (it becomes a stimulus row and a ``TaskContext``) and how it
parses and scores the answers. Everything else is here:

- :func:`run_request` runs one library request as one recorded session;
- :func:`adapter_score` builds the score: the library's own number, never
  ceiling-normalized, with the reason it can differ from the library's own
  evaluator (``attrs['input_adaptation']``);
- :func:`run_recorded` runs the benchmark's protocol in an ``Experiment`` that
  records every request and answer, and keeps or deletes the records.

An adapter benchmark checks no compatibility itself: ``score()`` runs the
pre-flight once (Contract Rule 5.7). An adapter declares
``required_input_channels``, ``requested_output_channels`` and
``trial_memory`` like any benchmark, and its protocol is a
``CallableProtocol`` whose ``evaluate(subject, context)`` calls
:func:`run_request` for each request.
"""
from pathlib import Path
import tempfile
from typing import Optional

from brainscore_core.metrics import Score
from brainscore_core.streaming import Session
from brainscore_core.streaming_helpers import stimulus_session

from brainscore.benchmarks._score import benchmark_score
from brainscore.experiments import Experiment, RecordInputsOutputs
from brainscore.experiments.runner import participate_observed


def run_request(subject, context, repetition, stimuli, outputs, task) -> Session:
    """Run one library request as one session in record repetition ``repetition``.

    The session presents ``stimuli`` under ``task`` and requests ``outputs``.
    It is one repetition of the record, so ``replay_sessions`` rebuilds exactly
    this session. The subject is not reset here: a reset would remove
    interventions the caller applied around the run. Returns the finished
    session; read the answers with ``session.collect(channel)`` or
    ``session.emitted``.
    """
    session = stimulus_session(stimuli, list(outputs), task)
    with context.repetition(repetition):
        participate_observed(subject, session, context)
    return session


def adapter_score(benchmark, value: float, *, metric: str, input_adaptation: dict,
                  context, **attrs) -> Score:
    """The score of an adapter benchmark: the library's own number, unchanged.

    No human ceiling is applied (``normalized`` is ``False``).
    ``input_adaptation`` says why the number can differ from the library's own
    evaluator (Contract Rule 3.9; Communication Rule 2.3).
    """
    return benchmark_score(
        benchmark, value, raw=value, metric=metric, normalized=False,
        benchmark_identifier=benchmark.identifier,
        leaderboard_eligible=benchmark.leaderboard_eligible,
        run_directory=str(context.directory),
        input_adaptation=dict(input_adaptation), **attrs)


def run_recorded(subject, protocol, output_dir: Optional[Path]) -> Score:
    """Run ``protocol`` with every request and answer recorded; return its score.

    With ``output_dir`` the records stay there (``records_retained`` is
    ``True``). Without it they go to a temporary folder that is deleted when the
    run ends, also when it fails; the score then has ``run_directory`` ``None``.
    """
    if output_dir is not None:
        score = _run(subject, protocol, Path(output_dir))
        score.attrs['records_retained'] = True
        return score
    with tempfile.TemporaryDirectory(prefix='brainscore-adapter-') as temporary:
        score = _run(subject, protocol, Path(temporary) / 'run')
        score.attrs['run_directory'] = None
        score.attrs['records_retained'] = False
        return score


def _run(subject, protocol, directory: Path) -> Score:
    result = Experiment(subject=subject, protocol=protocol,
                        tools=[RecordInputsOutputs()], output_dir=directory).run()
    # Experiment also returns the run directory; the benchmark returns the score.
    return result.value
