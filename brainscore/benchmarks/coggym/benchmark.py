"""Registered experiments delegate prompts, trials and scoring to CogGym."""
from collections.abc import Callable
import math
from pathlib import Path

import pandas as pd

from brainscore_core.benchmarks import BenchmarkBase
from brainscore_core.metrics import Score
from brainscore_core.contract import Subject
from brainscore.benchmarks._score import not_estimated
from brainscore.experiments import CallableProtocol
from brainscore.harnesses.adapter import adapter_score, run_recorded
from brainscore.harnesses.coggym import CogGymRunner

from . import EXPERIMENTS

# What the score does with a slider answer (Contract Rule 6.2). CogGym parses the text
# (rule 3.9); the check coggym-slider-v1 gives each slider answer a state.
ANSWER_POLICY = ("the number is CogGym's own fit; a slider answer that coggym-slider-v1 rejects "
                 'but CogGym read (a number taken from prose, a clamped value, a duplicate key) is '
                 'invalid and stays in the fit as CogGym read it; an answer CogGym could not read '
                 '(also a missing written answer, given to CogGym as empty text) is missing and '
                 'CogGym leaves it out of its fit; above 10 percent invalid and missing answers '
                 'the result is marked invalid')
# The share of invalid and missing slider answers above which the result is marked invalid.
INVALID_LIMIT = 0.1


# Why a score can differ from one made with CogGym's own client (Contract Rule 3.9).
_PROMPT_NOTE = {
    'kind': 'subject_prompt',
    'reason': ('The subject builds its own prompt from the instruction and the parts. '
               'A chat model can place the instruction in the user turn, where '
               "CogGym's own client sends a separate system text."),
}


def _require_reset(reset):
    """The adapter's reset(repetition) callback, or an error before any compute.

    The benchmark never reads it from the subject (Contract Rule 2.2): the
    caller supplies it, because only the caller knows its adapter.
    """
    if reset is None:
        raise ValueError(
            'The CogGym benchmark needs a reset callback. Load it with '
            "load_benchmark('CogGym...', reset=adapter.reset). "
            'An adapter with no state can pass reset=lambda repetition: None.'
        )
    return reset


class CogGymBenchmark(BenchmarkBase):
    """One public experiment, returning upstream raw R² without a human ceiling.

    CogGym's evaluator runs in its own environment, in a folder Brain-Score makes at the first
    load (``folder`` names another one; ``BRAINSCORE_ENVIRONMENTS_HOME`` moves the default place;
    ``BS_INSTALL_DEPENDENCIES=no`` forbids making it). ``reset`` is the
    adapter's reset(repetition) callback; it must preserve instrumentation.
    Scoring needs it: give it here or to ``protocol``.
    One repetition is a preliminary evaluation, not a leaderboard replication.

    Each CogGym trial reaches the subject on the common channels: the system
    text as the instruction, the message parts (``text``, ``vision:image``,
    ``vision:video``) in CogGym's order, and the temperature and length as
    settings on a ``Text`` response. The subject answers on ``behavior:text``
    and CogGym parses the answer. A subject that takes in only some of these
    inputs runs with the others dropped (a partial match); ``score()`` checks
    the pair once before any compute and records ``attrs['compatibility']``
    with the match level and the dropped inputs. Only an incompatible pair is N/A.
    A ``BrainScoreModel`` with a ``reply`` part and the matching inputs runs it.

    Scores can differ a little from scores made with CogGym's own client. The
    subject builds its own prompt from the instruction and the parts, so a chat
    model can place the instruction in the user turn where CogGym's client
    sends a separate system text. ``score.attrs['input_adaptation']`` names this
    cause.

    The score is CogGym's own fit. The check ``coggym-slider-v1`` gives each slider
    answer a state and changes no value: ``valid``; ``invalid`` (CogGym read a value the
    check rejects; it stays in the fit as CogGym read it); ``missing`` (CogGym read no
    value and left the item out). The score carries the counts and ``answer_policy``.
    Above 10 percent invalid and missing answers the result is returned with
    ``result_validity == 'invalid'``, without an error (Contract Rule 6.2).
    """

    # Declare the messages the subject must accept and return before a run starts.
    # An instance replaces the inputs with the parts its experiment shows.
    required_input_channels = frozenset({'instruction', 'text'})
    requested_output_channels = frozenset({'behavior:text'})
    # Each call is one trial; the subject keeps no history between trials.
    trial_memory = 'none'
    leaderboard_eligible = False

    def __init__(
        self,
        experiment: str,
        *,
        folder: str | Path | None = None,
        repetitions: int = 5,
        temperature: float = 1.0,
        max_tokens: int = 512,
        reset: Callable[[int], None] | None = None,
        output_dir: str | Path | None = None,
    ) -> None:
        if experiment not in EXPERIMENTS:
            raise ValueError(f'Unknown public CogGym experiment: {experiment}')
        if reset is not None and not callable(reset):
            raise TypeError("reset must be the adapter's reset(repetition) callback")
        self.experiment = experiment
        self.modality = EXPERIMENTS[experiment]
        self._reset = reset
        self._output_dir = Path(output_dir) if output_dir is not None else None
        self._configuration = {
            'folder': None if folder is None else Path(folder),
            'experiment': experiment,
            # Video experiments can also contain still-image instruction screens.
            'modalities': {
                'text': ('text',),
                'image': ('text', 'image'),
                'video': ('text', 'image', 'video'),
            }[self.modality],
            'repetitions': repetitions,
            'temperature': temperature,
            'max_tokens': max_tokens,
        }
        # Validate the reference and all prompts before the caller loads a model.
        # The trials decide which kinds of part a subject must take.
        preflight = CogGymRunner(model='preflight', **self._configuration)
        # The folder the pre-flight found or made: the scoring run uses the same one.
        self._configuration['folder'] = preflight.folder
        self.required_input_channels = preflight.required_input_channels
        identifier = f'CogGym.{experiment.replace("/", ".")}'
        self._reference_profile = (repetitions, temperature, max_tokens) == (5, 1.0, 512)
        if not self._reference_profile:
            identifier += f'.repeats-{repetitions}.tokens-{max_tokens}.temperature-{float(temperature)!r}'
        super().__init__(
            identifier=identifier,
            version=1,
            parent='behavioral',
            ceiling=not_estimated(),
            bibtex='@misc{coggym2026, title={CogGym: Towards Large-Scale Comparative '
                   'Evaluation of Human and Machine Cognition}, year={2026}, '
                   'url={https://arxiv.org/abs/2609.21259}}',
        )

    def protocol(
        self,
        *,
        model: str,
        reset: Callable[[int], None] | None = None,
    ) -> CallableProtocol:
        """Build the evaluation steps for an Experiment; do not run them yet.

        CallableProtocol wraps an evaluation function so Experiment can run it
        with recording and intervention tools. The return annotation names that
        wrapper; the Score is produced only when the protocol runs.
        """
        reset = reset if reset is not None else self._reset
        if reset is not None and not callable(reset):
            raise TypeError('reset must be a reset(repetition) callback')
        runner = CogGymRunner(model=model, **self._configuration)

        # Experiment supplies the subject (model interface) and context (run
        # directory and artifact registration). CogGym still owns the trial loop.
        def evaluate(subject, context):
            native = runner.protocol(reset=_require_reset(reset))
            # Call evaluate(), not run(): the outer protocol already owns the run.
            # The runner records each request and answer as one session, in its
            # own record repetition.
            result = native.evaluate(subject, context)
            validation = result['response_validation']
            rows = result['experiments']
            if len(rows) != 1 or rows[0]['experiment'] != self.experiment or rows[0]['model'] != model:
                raise RuntimeError('CogGym must return exactly the requested model and experiment')
            value = rows[0]['r2']
            undefined = value is None or not math.isfinite(value)
            # One row for each checked slider answer of each repetition, with its state.
            answers = [pd.DataFrame({'response_state': [issue['response_state'] for issue in report['issues']]
                                     + ['valid'] * (report['checked_queries'] - len(report['issues']))})
                       for report in validation['repetitions']]
            # Keep CogGym's R² unchanged. Attributes explain how it was obtained;
            # no human-ceiling normalization is applied.
            score = adapter_score(
                self, float('nan') if undefined else value, metric='pearson_r_squared',
                context=context, input_adaptation=_PROMPT_NOTE,
                modality=self.modality,
                answers=answers, answer_policy=ANSWER_POLICY, invalid_limit=INVALID_LIMIT,
                configuration=result['configuration'],
                coverage=result['repetitions'],
                response_validation=validation,
                native_analysis=result,
                evaluation_scope='public_subset',
                reference_profile=self._reference_profile,
                repetition_note='Five repeats match the reference README example, not a guarantee of leaderboard replication.',
            )
            # A result marked invalid is returned even without a fit (rule 6.2); otherwise an
            # undefined fit is no score.
            if undefined and score.attrs['result_validity'] != 'invalid':
                raise ValueError(
                    'CogGym R² is undefined; inspect the saved responses and coverage. '
                    'An unscorable experiment is not a zero score.'
                )
            return score

        return CallableProtocol(
            self.identifier,
            evaluate,
            # The runner records its sessions itself; no subject method is observed.
            methods=(),
            metadata=runner.describe(),
        )

    def __call__(self, subject: Subject) -> Score:
        """Run benchmark(subject) with input/output recording and return its Score.

        Default scoring deletes the call records on success and failure; an
        explicit ``output_dir`` keeps them. ``score()`` runs the pre-flight.
        """
        protocol = self.protocol(model=subject.identifier, reset=_require_reset(self._reset))
        return run_recorded(subject, protocol, self._output_dir)
