"""Registered experiments delegate prompts, trials and scoring to CogGym."""
from collections.abc import Callable
import math
import os
from pathlib import Path
import tempfile

from brainscore_core.benchmarks import BenchmarkBase
from brainscore_core.compatibility import check_channel_compatibility
from brainscore_core.metrics import Score
from brainscore_core.model_interface import Subject
from brainscore.experiments import CallableProtocol, Experiment, RecordInputsOutputs
from brainscore.harnesses.coggym import CogGymRunner

from . import EXPERIMENTS


class CogGymBenchmark(BenchmarkBase):
    """One public experiment, returning upstream raw R² without a human ceiling.

    Supply a pinned checkout or set BRAINSCORE_COGGYM_CHECKOUT. ``reset`` is the
    provider's reset(repetition) callback; it must preserve instrumentation.
    One repetition is a preliminary evaluation, not a leaderboard replication.
    """

    # Declare the messages the subject must accept and return before a run starts.
    required_input_channels = frozenset({'generation_request'})
    requested_output_channels = frozenset({'response_trace'})

    def __init__(
        self,
        experiment: str,
        *,
        checkout: str | Path | None = None,
        repetitions: int = 1,
        temperature: float = 1.0,
        max_tokens: int = 8192,
        reset: Callable[[int], None] | None = None,
        output_dir: str | Path | None = None,
    ) -> None:
        if experiment not in EXPERIMENTS:
            raise ValueError(f'Unknown public CogGym experiment: {experiment}')
        checkout = checkout or os.environ.get('BRAINSCORE_COGGYM_CHECKOUT')
        if not checkout:
            raise ValueError(
                'Supply checkout=... or set BRAINSCORE_COGGYM_CHECKOUT to the '
                'pinned CogGym checkout; see docs/coggym.md.'
            )
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
        CogGymRunner(model='preflight', **self._configuration)
        ceiling = Score(float('nan'))
        ceiling.attrs['status'] = 'not_estimated'
        super().__init__(
            identifier=f'CogGym.{experiment.replace("/", ".")}',
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
        if not callable(reset):
            raise TypeError(
                'Supply reset=provider.reset when loading the benchmark or '
                'creating its protocol. Stateless providers can explicitly use '
                'lambda repetition: None.'
            )
        runner = CogGymRunner(model=model, **self._configuration)
        native = runner.protocol(reset=reset)

        # Experiment supplies the subject (model interface) and context (run
        # directory and artifact registration). CogGym still owns the trial loop.
        def evaluate(subject, context):
            check_channel_compatibility(subject, self)
            # Call evaluate(), not run(): the outer protocol already observes
            # subject.process() calls, so nesting observers would record them twice.
            result = native.evaluate(subject, context)
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
                raw=Score(value),
                metric='pearson_r_squared',
                ceiling_status='not_estimated',
                normalized=False,
                modality=self.modality,
                configuration=result['configuration'],
                coverage=result['repetitions'],
                native_analysis=result,
                run_directory=str(context.directory),
                evaluation_scope='public_subset',
                repetition_note='One repeat is preliminary; serving settings must be matched before claiming replication.',
            )
            return score

        return CallableProtocol(
            self.identifier,
            evaluate,
            methods=['process'],  # Expose model calls to the attached tools.
            metadata=runner.describe(),
        )

    def __call__(self, candidate: Subject) -> Score:
        """Run benchmark(subject) with input/output recording and return its Score."""
        check_channel_compatibility(candidate, self)
        protocol = self.protocol(model=candidate.identifier)
        directory = self._output_dir
        if directory is None:
            directory = Path(tempfile.mkdtemp(prefix='brainscore-coggym-')) / 'run'
        result = Experiment(
            subject=candidate,
            protocol=protocol,
            tools=[RecordInputsOutputs()],
            output_dir=directory,
        ).run()
        # Experiment also returns the run directory; the benchmark returns the score.
        return result.value
