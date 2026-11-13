"""Registered experiments delegate prompts, trials and scoring to CogGym."""
from collections.abc import Callable
import math
from pathlib import Path
import tempfile

from brainscore_core.benchmarks import BenchmarkBase
from brainscore_core.compatibility import check_channel_compatibility
from brainscore_core.metrics import Score
from brainscore_core.model_interface import Subject
from brainscore.experiments import CallableProtocol, Experiment, RecordInputsOutputs
from brainscore.harnesses.coggym import CogGymRunner, _PROMPT_NOTE

from . import EXPERIMENTS


def _require_reset(reset):
    """The provider's reset(repetition) callback, or an error before any compute.

    The benchmark never reads it from the subject (Contract Rule 2.2): the
    caller supplies it, because only the caller knows its provider.
    """
    if reset is None:
        raise ValueError(
            'The CogGym benchmark needs a reset callback. Load it with '
            "load_benchmark('CogGym...', reset=provider.reset). "
            'A provider with no state can pass reset=lambda repetition: None.'
        )
    return reset


class CogGymBenchmark(BenchmarkBase):
    """One public experiment, returning upstream raw R² without a human ceiling.

    Supply a pinned checkout or set BRAINSCORE_COGGYM_CHECKOUT. ``reset`` is the
    provider's reset(repetition) callback; it must preserve instrumentation.
    Scoring needs it: give it here or to ``protocol``.
    One repetition is a preliminary evaluation, not a leaderboard replication.

    Each CogGym trial reaches the subject on the common channels: the system
    text as the instruction, the message parts (``text``, ``vision:image``,
    ``vision:video``) in CogGym's order, and the temperature and length as
    settings on a ``Text`` response. The subject answers on ``behavior:text``
    and CogGym parses the answer. A subject that takes in only some of these
    inputs runs with the others dropped (a partial match); the score records
    ``attrs['compatibility']`` with the match level and the dropped inputs.
    The benchmark is N/A only for an incompatible pair, before any compute.
    A ``BrainScoreModel`` with a ``reply`` part and the matching inputs runs it.

    Scores can differ a little from scores made with CogGym's own client. The
    subject builds its own prompt from the instruction and the parts, so a chat
    model can place the instruction in the user turn where CogGym's client
    sends a separate system text. ``score.attrs['input_adaptation']`` names this
    cause.
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
        checkout: str | Path | None = None,
        repetitions: int = 5,
        temperature: float = 1.0,
        max_tokens: int = 512,
        reset: Callable[[int], None] | None = None,
        output_dir: str | Path | None = None,
    ) -> None:
        if experiment not in EXPERIMENTS:
            raise ValueError(f'Unknown public CogGym experiment: {experiment}')
        if checkout is None:
            from brainscore.data.local import COGGYM_CHECKOUT, path
            checkout = path(COGGYM_CHECKOUT.name)      # the registered data location
        if reset is not None and not callable(reset):
            raise TypeError('reset must be a provider reset(repetition) callback')
        self.experiment = experiment
        self.modality = EXPERIMENTS[experiment]
        self._reset = reset
        self._output_dir = Path(output_dir) if output_dir is not None else None
        self._configuration = {
            'checkout': Path(checkout),
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
        self.required_input_channels = preflight.required_input_channels
        ceiling = Score(float('nan'))
        ceiling.attrs['status'] = 'not_estimated'
        identifier = f'CogGym.{experiment.replace("/", ".")}'
        self._reference_profile = (repetitions, temperature, max_tokens) == (5, 1.0, 512)
        if not self._reference_profile:
            identifier += f'.repeats-{repetitions}.tokens-{max_tokens}.temperature-{float(temperature)!r}'
        super().__init__(
            identifier=identifier,
            version=1,
            parent='behavioral',
            ceiling=ceiling,
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
            # A partial match runs with the dropped inputs left out; the score says so.
            match = check_channel_compatibility(subject, self)
            native = runner.protocol(reset=_require_reset(reset))
            # Call evaluate(), not run(): the outer protocol already owns the run.
            # The runner records each request and answer as one session, in its
            # own record repetition.
            result = native.evaluate(subject, context)
            validation = result['response_validation']
            if validation['status'] == 'invalid':
                raise ValueError(
                    f'CogGym slider validation rejected {validation["invalid_queries"]} answers; '
                    'no benchmark score is returned. Inspect response_validation in the saved '
                    'analysis and run artifacts. Native results are unchanged.'
                )
            rows = result['experiments']
            if len(rows) != 1 or rows[0]['experiment'] != self.experiment or rows[0]['model'] != model:
                raise RuntimeError('CogGym must return exactly the requested model and experiment')
            value = rows[0]['r2']
            if value is None or not math.isfinite(value):
                raise ValueError(
                    'CogGym R² is undefined; inspect the saved responses and coverage. '
                    'An unscorable experiment is not a zero score.'
                )
            # Keep CogGym's R² unchanged. Attributes explain how it was obtained;
            # no human-ceiling normalization is applied.
            score = Score(value)
            score.attrs.update(
                benchmark_identifier=self.identifier,
                raw=Score(value),
                metric='pearson_r_squared',
                ceiling_status='not_estimated',
                normalized=False,
                leaderboard_eligible=False,
                modality=self.modality,
                configuration=result['configuration'],
                coverage=result['repetitions'],
                response_validation=validation,
                native_analysis=result,
                run_directory=str(context.directory),
                evaluation_scope='public_subset',
                reference_profile=self._reference_profile,
                repetition_note='Five repeats match the reference README example, not a guarantee of leaderboard replication.',
                input_adaptation=dict(_PROMPT_NOTE),
            )
            score.attrs['compatibility'] = match.as_attrs()
            return score

        return CallableProtocol(
            self.identifier,
            evaluate,
            # The runner records its sessions itself; no subject method is observed.
            methods=(),
            metadata=runner.describe(),
        )

    def __call__(self, subject: Subject) -> Score:
        """Run benchmark(subject) with input/output recording and return its Score."""
        check_channel_compatibility(subject, self)
        protocol = self.protocol(model=subject.identifier, reset=_require_reset(self._reset))
        directory = self._output_dir
        if directory is None:
            # Keep implicit recordings only for this call, even if scoring fails.
            with tempfile.TemporaryDirectory(prefix='brainscore-coggym-') as temporary:
                score = self._run(subject, protocol, Path(temporary) / 'run')
                score.attrs['run_directory'] = None
                score.attrs['records_retained'] = False
                return score
        score = self._run(subject, protocol, directory)
        score.attrs['records_retained'] = True
        return score

    def _run(self, subject: Subject, protocol: CallableProtocol, directory: Path) -> Score:
        result = Experiment(
            subject=subject,
            protocol=protocol,
            tools=[RecordInputsOutputs()],
            output_dir=directory,
        ).run()
        # Experiment also returns the run directory; the benchmark returns the score.
        return result.value
