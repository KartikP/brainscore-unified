"""EWoK accuracy through participate(), with ordinary Experiment tools.

Choice mode runs on the common channels. Each question is one stimulus row
whose ``text`` is EWoK's own choice prompt (instructions, two worked examples,
the two contexts numbered 1 and 2, the scenario and a response cue). The task's
response is ``Choice(labels=('1', '2'))`` and the subject answers on
``behavior:choice``. Any subject with a ``reply``, ``likelihood`` or ``answer``
part and the ``text`` input runs it.

Choice scores can differ from EWoK's own evaluator. The subject builds its own
prompt (a chat model adds its template, and the choice adds its directive
"Answer with exactly one of: 1, 2.") and decodes by its own rule, where EWoK
sends the raw prompt and limits greedy decoding to 1 or 2.
``score.attrs['input_adaptation']`` names this cause.
"""
from collections import defaultdict
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from brainscore_core.benchmarks import BenchmarkBase
from brainscore_core.compatibility import check_channel_compatibility
from brainscore_core.contract import TaskContext
from brainscore_core.metrics import Score
from brainscore_core.model_interface import Subject
from brainscore_core.responses import Choice
from brainscore_core.streaming import Session, StreamEvent
from brainscore_core.streaming_helpers import stimulus_session
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
from brainscore.experiments import CallableProtocol, Experiment, RecordInputsOutputs
from brainscore.instrumentation import observed
from brainscore.model_helpers.response_trace import register_trace_channels
from brainscore.presentation import response_stimulus_ids


# Why a choice score can differ from EWoK's own evaluator (Contract Rule 3.9).
_CHOICE_NOTE = {
    'kind': 'subject_prompt',
    'reason': ("The subject builds its own prompt from EWoK's prompt text and the choice "
               'directive, and decodes by its own rule. EWoK sends the raw prompt and limits '
               'greedy decoding to 1 or 2.'),
}


def _choice_item(heading, context1, context2, target, response):
    return (f'\n\n# {heading}\n\n## Contexts\n1. "{context1}"\n2. "{context2}"\n\n'
            f'## Scenario\n"{target}"\n\n## Task\nWhich context makes more sense given the '
            'scenario? Please answer using either "1" or "2".\n\n'
            f'## Response\n{response}')


def _choice_prompt(target: str, context1: str, context2: str) -> str:
    """EWoK's choice prompt with two worked examples, character for character.

    Copied from ``format_choice_prompt(..., 'optimized')`` in
    ``ewok/evaluate/util.py`` of the ewok-paper repository at revision
    9e40d30e242925866ee50448f80a13bcdf971318.
    """
    prompt = ('# INSTRUCTIONS\n\nIn this study, you will see multiple examples. In each '
              'example, you will be given two contexts and a scenario. Your task is to read '
              'the two contexts and the subsequent scenario, and pick the context that makes '
              'more sense considering the scenario that follows. The contexts will be numbered '
              '"1" or "2". You must answer using "1" or "2" in your response.\n')
    prompt += _choice_item('TRIAL EXAMPLE', 'The bag is full of blocks.',
                           'The bag is full of balls.', 'I drew a ball from the bag.', '2\n')
    prompt += _choice_item('TRIAL EXAMPLE', 'The boy likes cookies.',
                           'The boy does not like cookies.', 'The boy chose to eat a cookie.', '1\n')
    return prompt + _choice_item('TEST EXAMPLE', context1, context2, target, '')


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


def _ask(subject: Subject, context, request: StreamEvent, repetition: int) -> StreamEvent:
    """Send one request through ``participate`` and return the subject's answer.

    The session holds the one request and asks for ``response_trace``. It is
    one repetition of the record, so ``replay_sessions`` rebuilds exactly this
    session. The subject is not reset here, as on the old ``process`` path:
    a reset would remove interventions the caller applied around the run.

    Known limit: ``replay_sessions`` resets the subject before and after
    each repetition and this run does not, so the replay of a subject that
    keeps state between requests, or of a run inside a caller's
    intervention, can differ from the original.

    A streaming subject sends fragments, and the last one is the answer. The
    old ``process`` path returned only the final fragment, so a response
    whose ``final`` flag is present and not True is refused.
    """
    session = Session([request], ('response_trace',))
    with context.repetition(repetition):
        context.publish('session_context',
                        {'outputs': session.outputs})
        with observed(session, context):
            subject.participate(session)
    responses = [event for event in session.emitted if event.channel == 'response_trace']
    if not responses:
        raise TypeError('EWoK requires a response_trace event')
    if responses[-1].payload.get('final', True) is not True:
        raise ValueError('EWoK provider ended without a final response')
    return responses[-1]


def _choose(subject: Subject, context, stimuli, repetition: int) -> list:
    """Show each prompt in ``stimuli`` and return the chosen label of each row.

    One ``stimulus_session`` asks for ``behavior:choice`` with the task
    ``Choice(labels=('1', '2'))``; each row is its own trial. The session is
    one repetition of the record, as in ``_ask``. A row the subject could not
    answer (no valid label) gives ``None``.
    """
    task = TaskContext(response=Choice(labels=('1', '2')), trial_memory='none')
    session = stimulus_session(stimuli, ['behavior:choice'], task)
    with context.repetition(repetition):
        context.publish('session_context', {
            name: getattr(session, name) for name in Session.context_fields
            if getattr(session, name, None) is not None})
        with observed(session, context):
            subject.participate(session)
    if not any(event.channel == 'behavior:choice' for event in session.emitted):
        raise TypeError('EWoK choice requires a behavior:choice answer')
    answers = session.collect('behavior:choice')
    if 'choice' not in getattr(answers, 'dims', ()):
        raise TypeError('EWoK choice requires one row of label scores per question')
    requested = list(stimuli['stimulus_id'])
    order = {stimulus_id: row for row, stimulus_id in
             enumerate(response_stimulus_ids(answers, requested))}
    labels = [str(label) for label in answers['choice'].values]
    values = np.asarray(answers.transpose('presentation', 'choice').values, dtype=float)
    chosen = []
    for stimulus_id in requested:
        row = values[order[stimulus_id]]
        chosen.append(labels[int(row.argmax())] if np.isfinite(row).all() else None)
    return chosen


def _mean_by_version(rows: list[dict]) -> tuple[float, dict]:
    """Average items within each dataset version, then weight versions equally."""
    groups = defaultdict(list)
    for row in rows:
        groups[row['version']].append(row['accuracy'])
    means = {key: sum(values) / len(values) for key, values in sorted(groups.items())}
    return sum(means.values()) / len(means), means


class EWoKBenchmark(BenchmarkBase):
    """Score a prepared local EWoK dataset; no automatic model/data downloads.

    ``mode='choice'`` shows EWoK's choice prompt as ``text`` and asks for
    ``behavior:choice`` between the labels 1 and 2 (see the module docstring
    for why its numbers can differ from EWoK's own evaluator).
    ``mode='logprobs'`` still sends ``conditional_logprob_request`` and reads
    ``response_trace``.
    """

    # Each request is a self-contained batch of items; the subject keeps no history.
    trial_memory = 'none'
    leaderboard_eligible = False

    def __init__(
        self,
        *,
        mode: str = 'logprobs',
        root: str | Path | None = None,
        domains: list[str] | None = None,
        batch_size: int = 8,
        output_dir: str | Path | None = None,
    ) -> None:
        register_trace_channels()
        if mode not in ('logprobs', 'choice'):
            raise ValueError('Choose logprobs or choice')
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError('batch_size must be a positive integer')
        # Check local data and its checksum before spending time on model calls.
        from brainscore import load_stimulus_set
        stimuli = load_stimulus_set('EWoK-core-1.0', root=root)
        manifest = stimuli.attrs['manifest']
        rows = stimuli.drop(columns=['stimulus_id']).astype(object).to_dict('records')
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
        # Declare the messages the subject must accept and return before a run starts.
        self.required_input_channels = frozenset(
            {'conditional_logprob_request'} if mode == 'logprobs' else {'text'})
        self.requested_output_channels = frozenset(
            {'response_trace'} if mode == 'logprobs' else {'behavior:choice'})
        # Local exports and domain subsets must never share a score identity.
        identifier = f'EWoK-core-1.0-{mode}-data-{manifest["items_sha256"][:16]}'
        if domains is not None:
            selection = json.dumps(sorted(set(domains))).encode()
            identifier += f'-domains-{hashlib.sha256(selection).hexdigest()[:12]}'
        ceiling = Score(float('nan'))
        ceiling.attrs['status'] = 'not_estimated'
        super().__init__(
            identifier=identifier, version=1, parent='behavioral',
            ceiling=ceiling,
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
            # Recorded requests contain restricted item text; keep runs private.
            context.directory.chmod(0o700)
            scored = []
            requests_sent = 0   # one record repetition per request
            for offset in range(0, len(self._rows), self._batch_size):
                batch = self._rows[offset:offset + self._batch_size]
                returned = {}
                # Log probabilities compare each target under each context.
                # Choice requests show both contexts; 0 means neither is selected.
                pairs = ((1, 1), (1, 2), (2, 1), (2, 2)) if self._mode == 'logprobs' else ((1, 0), (2, 0))
                for target, context_number in pairs:
                    if self._mode == 'choice':
                        # One row per item: EWoK's prompt for this target. Expected
                        # answers stay in the benchmark.
                        stimuli = StimulusSet(pd.DataFrame({
                            'stimulus_id': [f'{r["item_id"]}.target{target}' for r in batch],
                            'text': [_choice_prompt(r[f'Target{target}'], r['Context1'], r['Context2'])
                                     for r in batch],
                        }))
                        returned[target, context_number] = _choose(
                            subject, context, stimuli, repetition=requests_sent)
                        requests_sent += 1
                        continue
                    request = {
                        'operation': 'ewok.logprobs',
                        'targets': [r[f'Target{target}'] for r in batch],
                        'contexts': [r[f'Context{context_number}'] for r in batch],
                    }
                    # Route through participate so tools can observe or intervene.
                    # Expected answers and target-number labels stay in the benchmark.
                    response = _ask(subject, context, StreamEvent(
                        'conditional_logprob_request', request, None,
                        {'trial_ids': [r['item_id'] for r in batch]},
                    ), repetition=requests_sent)
                    requests_sent += 1
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
                        # An unparsed answer is wrong and counted, never dropped.
                        accuracy = ((values[0] == '1') + (values[1] == '2')) / 2
                        invalid = sum(value is None for value in values)
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
                benchmark_identifier=self.identifier,
                raw=Score(value), normalized=False, metric='accuracy',
                ceiling_status='not_estimated', protocol=self._mode,
                leaderboard_eligible=False,
                aggregation='mean of within-version mean accuracies',
                version_scores=versions, domain_scores=domains,
                items=len(scored), questions=2 * len(scored),
                invalid_answers=sum(r['invalid_answers'] for r in scored),
                dataset_sha256=self._manifest['items_sha256'],
                dataset_scope=self._manifest['scope'], domain_subset=self._subset,
                run_directory=str(context.directory),
                paper_replication=False,
            )
            if self._mode == 'choice':
                score.attrs['input_adaptation'] = dict(_CHOICE_NOTE)
            return score
        return CallableProtocol(
            self.identifier,
            evaluate,
            # No method is observed: each session is recorded through
            # observed in _ask, one input and one output per request.
            methods=(),
            # The record repetition ids _ask uses: one per request, counted
            # from 0. Tools select repetitions by these ids.
            repetitions=range(math.ceil(len(self._rows) / self._batch_size)
                              * (4 if self._mode == 'logprobs' else 2)),
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
            # Default scoring returns only scores; delete private call records
            # on both success and failure. Explicit output_dir opts into retention.
            with tempfile.TemporaryDirectory(prefix='brainscore-ewok-') as temporary:
                score = self._run(candidate, Path(temporary) / 'run')
                score.attrs['run_directory'] = None
                score.attrs['records_retained'] = False
                return score
        score = self._run(candidate, directory)
        score.attrs['records_retained'] = True
        return score

    def _run(self, candidate: Subject, directory: Path) -> Score:
        result = Experiment(
            subject=candidate, protocol=self.protocol(),
            tools=[RecordInputsOutputs()], output_dir=directory,
        ).run()
        # Experiment also returns the run directory; the benchmark returns the score.
        return result.value
