"""EWoK accuracy through process(), with ordinary Experiment tools."""
from collections import defaultdict
import json
import math
from numbers import Real
from pathlib import Path
import tempfile

from brainscore_core.benchmarks import BenchmarkBase
from brainscore_core.compatibility import check_channel_compatibility
from brainscore_core.metrics import Score
from brainscore_core.model_interface import Subject
from brainscore_core.streaming import StreamEvent
from brainscore.data.ewok.data import load_prepared
from brainscore.experiments import CallableProtocol, Experiment, RecordInputsOutputs


def logprob_accuracy(values: list[float]) -> float:
    """Compare each target under the two contexts; exact ties earn half credit.

    A target is a possible continuation; a context is the text preceding it.
    Order: T1|C1, T1|C2, T2|C1, T2|C2 (target given context).
    Target 1 should be more likely under context 1, and target 2 under context 2.
    Their average matches the paper's R analysis; exact ties earn half credit.
    """
    if len(values) != 4 or any(
        isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v)
        for v in values
    ):
        raise ValueError('Expected four finite conditional log probabilities')
    def compare(a, b):
        return 1.0 if a > b else 0.5 if a == b else 0.0
    return (compare(values[0], values[1]) + compare(values[3], values[2])) / 2


def _mean_by_version(rows: list[dict]) -> tuple[float, dict]:
    """Average items within each dataset version, then weight versions equally."""
    groups = defaultdict(list)
    for row in rows:
        groups[row['version']].append(row['accuracy'])
    means = {key: sum(values) / len(values) for key, values in sorted(groups.items())}
    return sum(means.values()) / len(means), means


class EWoKBenchmark(BenchmarkBase):
    """Score a prepared local EWoK dataset; no automatic model/data downloads."""

    # Declare the messages the subject must accept and return before a run starts.
    required_input_channels = frozenset({'generation_request'})
    requested_output_channels = frozenset({'response_trace'})

    def __init__(
        self,
        *,
        mode: str = 'logprobs',
        root: str | Path | None = None,
        domains: list[str] | None = None,
        batch_size: int = 8,
        output_dir: str | Path | None = None,
    ) -> None:
        if mode not in ('logprobs', 'choice'):
            raise ValueError('Choose logprobs or choice')
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError('batch_size must be a positive integer')
        # Check local data and its checksum before spending time on model calls.
        rows, manifest = load_prepared(root)
        if domains is not None:
            if not domains or set(domains) - set(manifest['domains']):
                raise ValueError('Select one or more domains present in this dataset')
            rows = [row for row in rows if row['Domain'] in domains]
        self._rows = rows
        self._manifest = manifest
        self._batch_size = batch_size
        self._output_dir = Path(output_dir) if output_dir is not None else None
        self._mode = mode
        self._subset = domains is not None
        super().__init__(
            identifier=f'EWoK-core-1.0-{mode}', version=1, parent='behavioral',
            ceiling=Score(float('nan')),
            bibtex='@article{ivanova2025elements, title={Elements of World Knowledge}, '
                   'author={Ivanova, Anna and others}, year={2025}, '
                   'journal={Transactions of the Association for Computational Linguistics}, '
                   'url={https://arxiv.org/abs/2405.09605}}',
        )

    def protocol(self) -> CallableProtocol:
        """Build the evaluation steps for an Experiment; do not run them yet.

        CallableProtocol wraps an evaluation function so Experiment can run it
        with recording and intervention tools. The return annotation names that
        wrapper; the Score is produced only when the protocol runs.
        """
        # Experiment supplies the subject (model interface) and context (run
        # directory and artifact registration).
        def evaluate(subject, context):
            check_channel_compatibility(subject, self)
            scored = []
            for offset in range(0, len(self._rows), self._batch_size):
                batch = self._rows[offset:offset + self._batch_size]
                returned = {}
                # Log probabilities compare each target under each context.
                # Choice requests show both contexts; 0 means neither is selected.
                pairs = ((1, 1), (1, 2), (2, 1), (2, 2)) if self._mode == 'logprobs' else ((1, 0), (2, 0))
                for target, context_number in pairs:
                    request = {
                        'operation': f'ewok.{self._mode}',
                        'targets': [r[f'Target{target}'] for r in batch],
                    }
                    if self._mode == 'logprobs':
                        request['contexts'] = [r[f'Context{context_number}'] for r in batch]
                    else:
                        request.update(
                            contexts1=[r['Context1'] for r in batch],
                            contexts2=[r['Context2'] for r in batch],
                            gen_type='constrained', prompt_type='optimized',
                        )
                    # Route through the subject so tools can observe or intervene.
                    # Expected answers and target-number labels stay in the benchmark.
                    response = subject.process(StreamEvent(
                        'generation_request', request, None,
                        {'trial_ids': [r['item_id'] for r in batch]},
                    ))
                    if not isinstance(response, StreamEvent) or response.channel != 'response_trace':
                        raise TypeError('EWoK requires a response_trace event')
                    if response.payload.get('valid') is not True:
                        raise ValueError('EWoK provider returned an invalid response')
                    # The provider returns one value per item in the same batch order.
                    values = response.payload['answer']
                    if not isinstance(values, list) or len(values) != len(batch):
                        raise ValueError('EWoK provider must return one value per requested target')
                    returned[target, context_number] = values
                for index, row in enumerate(batch):
                    values = [returned[pair][index] for pair in pairs]
                    if self._mode == 'logprobs':
                        accuracy = logprob_accuracy(values)
                        invalid = 0
                    else:
                        # Never parse a number out of reasoning or silently drop a bad answer.
                        choices = [str(v).strip() if isinstance(v, (str, int)) and not isinstance(v, bool) else '' for v in values]
                        accuracy = ((choices[0] == '1') + (choices[1] == '2')) / 2
                        invalid = sum(v not in ('1', '2') for v in choices)
                    scored.append({
                        'item_id': row['item_id'], 'domain': row['Domain'],
                        'family': row['family'], 'version': row['Version'],
                        'accuracy': accuracy, 'invalid_answers': invalid,
                    })
                # Save each completed batch. Replace the previous file only after
                # writing finishes, so an interrupted write keeps the last checkpoint.
                path = context.directory / 'item-scores.json'
                temporary = path.with_suffix('.partial')
                temporary.write_text(json.dumps(scored, indent=2, allow_nan=False) + '\n')
                temporary.replace(path)
            context.artifact(path, producer='EWoK benchmark', description='Item scores without benchmark text')
            # Report overall and per-domain accuracy with the same weighting rule.
            value, versions = _mean_by_version(scored)
            domains = {}
            for domain in sorted({r['domain'] for r in scored}):
                domains[domain] = _mean_by_version([r for r in scored if r['domain'] == domain])[0]
            score = Score(value)
            score.attrs.update(
                raw=Score(value), normalized=False, metric='accuracy',
                ceiling_status='not_estimated', protocol=self._mode,
                aggregation='mean of within-version mean accuracies',
                version_scores=versions, domain_scores=domains,
                items=len(scored), questions=2 * len(scored),
                invalid_answers=sum(r['invalid_answers'] for r in scored),
                dataset_sha256=self._manifest['items_sha256'],
                dataset_scope=self._manifest['scope'], domain_subset=self._subset,
                run_directory=str(context.directory),
                paper_replication=False,
            )
            return score
        return CallableProtocol(
            self.identifier,
            evaluate,
            methods=['process'],  # Expose model calls to the attached tools.
            metadata={
                'protocol': self._mode, 'dataset': self._manifest,
                'items': len(self._rows), 'batch_size': self._batch_size,
                'domains': sorted({r['Domain'] for r in self._rows}),
            },
        )

    def __call__(self, candidate: Subject) -> Score:
        """Run benchmark(subject) with input/output recording and return its Score."""
        check_channel_compatibility(candidate, self)
        directory = self._output_dir
        if directory is None:
            directory = Path(tempfile.mkdtemp(prefix='brainscore-ewok-')) / 'run'
        result = Experiment(
            subject=candidate, protocol=self.protocol(),
            tools=[RecordInputsOutputs()], output_dir=directory,
        ).run()
        # Experiment also returns the run directory; the benchmark returns the score.
        return result.value
