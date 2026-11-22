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

# What the score does with each answer (Contract Rule 6.2). CogGym parses the text (rule
# 3.9); the check coggym-slider-v1 gives each slider answer a state.
ANSWER_POLICY = ("the number is CogGym's own fit; a value CogGym parsed enters the fit only when "
                 'CogGym has human data for it; every requested answer of each repetition has a '
                 'state; a slider answer that coggym-slider-v1 rejects but CogGym read (a number '
                 'taken from prose, a clamped value, a duplicate key) is invalid and stays in the '
                 'fit as CogGym read it; an answer of another type is valid when CogGym parsed a '
                 'value for it; an answer CogGym could not read is missing and stays out of the '
                 'fit; a trial with no written answer reaches CogGym as a failed call (its own '
                 'record of a call with no reply), so its answers are missing and in no fit; a '
                 'written answer that core stores as invalid reaches CogGym as its text and is '
                 'counted invalid; above 10 percent invalid and missing answers, or when CogGym '
                 'makes no fit, the result is marked invalid')
# The share of invalid and missing answers above which the result is marked invalid.
INVALID_LIMIT = 0.1


# Why a score can differ from one made with CogGym's own client (Contract Rule 3.9).
_PROMPT_NOTE = {
    'kind': 'subject_prompt',
    'reason': ('The subject builds its own prompt from the instruction and the parts. '
               'A chat model can place the instruction in the user turn, where '
               "CogGym's own client sends a separate system text."),
}


class CogGymBenchmark(BenchmarkBase):
    """One public experiment, returning upstream raw R² without a human ceiling.

    CogGym's evaluator runs in its own environment, in a folder Brain-Score makes at the first
    load (``folder`` names another one; ``BRAINSCORE_ENVIRONMENTS_HOME`` moves the default place;
    ``BS_INSTALL_DEPENDENCIES=no`` forbids making it). ``reset`` is the
    adapter's reset(repetition) callback; it must preserve instrumentation.
    Without it, nothing is reset between repetitions: each trial is already its
    own session with no trial memory, and interventions applied around the run stay.
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

    The score is CogGym's own fit. Every requested answer of each repetition has a state,
    and no value changes: a slider answer goes through the check ``coggym-slider-v1``
    (``invalid`` when CogGym read a value the check rejects; it stays in the fit as CogGym
    read it); any other answer is ``valid`` when CogGym parsed a value; ``missing`` when
    CogGym has none (also a trial with no written answer, which reaches CogGym as a failed
    call). The score carries the counts and ``answer_policy``; ``response_validation``
    carries the detector's findings counted by reason. Above 10 percent invalid and missing
    answers, or when CogGym makes no fit (no answer it could fit, fewer than 5 scorable
    items, ratings with no variance), the result is returned with ``result_validity ==
    'invalid'``, its reason and NaN for no fit, without an error (Contract Rule 6.2). An
    error stays for a fault of the run: a result for another model or experiment, no
    result although answers are valid, a failed call, an incomplete run.
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
        runner = CogGymRunner(model=model, reset_given=reset is not None, **self._configuration)

        # Experiment supplies the subject (model interface) and context (run
        # directory and artifact registration). CogGym still owns the trial loop.
        def evaluate(subject, context):
            # No reset given: no reset between repetitions. Each trial is its own session
            # with no trial memory; subject.reset() would remove a caller's intervention.
            native = runner.protocol(reset=reset if reset is not None
                                     else lambda repetition: None)
            # Call evaluate(), not run(): the outer protocol already owns the run.
            # The runner records each request and answer as one session, in its
            # own record repetition.
            result = native.evaluate(subject, context)
            validation = result['response_validation']
            rows = result['experiments']
            if len(rows) > 1 or any(row['experiment'] != self.experiment or row['model'] != model
                                    for row in rows):
                raise RuntimeError('CogGym must return exactly the requested model and experiment')
            # One state for each requested answer of each repetition.
            states = [[issue['response_state'] for issue in report['issues']]
                      + ['valid'] * (report['checked_queries'] - len(report['issues']))
                      for report in validation['repetitions']]
            n_answers = sum(len(part) for part in states)
            n_valid = sum(part.count('valid') for part in states)
            # CogGym makes a row exactly when its fit has a scorable query (its own count).
            scorable = result['n_scorable_queries']
            if rows:
                value, n_items = rows[0]['r2'], rows[0].get('n_items')
            elif not scorable:
                value, n_items = None, 0      # no scorable query: no row is CogGym's result
            else:
                raise RuntimeError(f'CogGym returned no result for the experiment, although {scorable} '
                                   'of its queries are scorable: a fault of its analysis')
            undefined = value is None or not math.isfinite(value)
            # Keep CogGym's R² unchanged. Attributes explain how it was obtained;
            # no human-ceiling normalization is applied.
            score = adapter_score(
                self, float('nan') if undefined else value, metric='pearson_r_squared',
                context=context, input_adaptation=_PROMPT_NOTE,
                modality=self.modality,
                answers=[pd.DataFrame({'response_state': part}) for part in states],
                answer_policy=ANSWER_POLICY, invalid_limit=INVALID_LIMIT,
                configuration=result['configuration'],
                coverage=result['repetitions'],
                response_validation=validation,
                native_analysis=result,
                evaluation_scope='public_subset',
                reference_profile=self._reference_profile,
                repetition_note='Five repeats match the reference README example, not a guarantee of leaderboard replication.',
            )
            if undefined:
                # No fit is a result of the answers, marked invalid with its reason (rule 6.2).
                if not rows:
                    why = (f'CogGym made no fit: none of its queries is scorable '
                           f'({n_valid} of {n_answers} answers valid)')
                elif n_items is not None and n_items < 5:
                    why = f'CogGym made no fit: {n_items} scorable items, fewer than its minimum of 5'
                elif n_items is not None:
                    why = 'CogGym made no fit: the model or the human ratings have no variance'
                else:
                    why = 'CogGym made no fit: fewer than 5 scorable items, or ratings with no variance'
                shared = score.attrs.get('result_validity_reason')
                score.attrs['result_validity'] = 'invalid'
                score.attrs['result_validity_reason'] = f'{shared}; {why}' if shared else why
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
        protocol = self.protocol(model=subject.identifier)
        return run_recorded(subject, protocol, self._output_dir)
