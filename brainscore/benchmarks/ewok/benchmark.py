"""EWoK accuracy through participate(), with ordinary Experiment tools.

Both modes run on the common channels; each question is one stimulus row.

Log-probability mode shows each context as ``text``, exactly as given. The two
targets are the options of that row (``Choice(labels='options',
raw_scores=True, raw_text=True)``), and the subject returns the raw score of
each target on ``behavior:probabilities``: the summed log probability of the
target's tokens in ``f"{context} {target}"``, raw text, no template, as EWoK's
``score`` does. A subject needs a ``likelihood`` part; without one the
benchmark is N/A. The scoring compares log p(T1 | C1) with log p(T1 | C2) and
log p(T2 | C2) with log p(T2 | C1).
Log-probability scores can differ slightly from EWoK's own evaluator, for two
reasons. EWoK sets the logit of the padding or end-of-sequence token to minus
infinity before it normalizes; the subject here does not. EWoK tokenizes with
``add_special_tokens=True``, so some tokenizers add a start token before the
context; the subject here adds none. ``score.attrs['input_adaptation']`` names
these two differences.

Choice mode shows EWoK's own choice prompt (instructions, two worked examples,
the two contexts numbered 1 and 2, the scenario and a response cue) as
``text``, with no added sentence (``Choice(labels=('1', '2'),
raw_text=True)``). The subject answers on ``behavior:choice``; any subject with
a ``reply``, ``likelihood`` or ``answer`` part and the ``text`` input runs it.
Choice scores can differ from EWoK's own evaluator: a chat model answers
through its own chat template and the subject decodes by its own rule, where
EWoK sends the raw prompt and limits greedy decoding to 1 or 2.
``score.attrs['input_adaptation']`` names this cause.
"""
from collections import defaultdict
import hashlib
import json
import math
from numbers import Real
from pathlib import Path

import numpy as np
import pandas as pd

from brainscore_core.benchmarks import BenchmarkBase
from brainscore_core.contract import TaskContext
from brainscore_core.metrics import Score
from brainscore_core.contract import Subject
from brainscore_core.responses import Choice
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
from brainscore.benchmarks._score import not_estimated
from brainscore.experiments import CallableProtocol
from brainscore.harnesses.adapter import adapter_score, run_recorded, run_request
from brainscore.presentation import response_stimulus_ids


# Why a choice score can differ from EWoK's own evaluator (Contract Rule 3.9).
_CHOICE_NOTE = {
    'kind': 'subject_prompt',
    'reason': ("No sentence is added to EWoK's prompt, but a chat model answers through its "
               'own chat template, and the subject decodes by its own rule. EWoK sends the raw '
               'prompt and limits greedy decoding to 1 or 2.'),
}
# Why a log-probability score can differ slightly from EWoK's own evaluator.
_LOGPROBS_NOTE = {
    'kind': 'subject_scoring',
    'reason': ("EWoK sets the logit of the padding or end-of-sequence token to minus infinity "
               'before it normalizes, and it tokenizes with add_special_tokens=True, so some '
               'tokenizers add a start token before the context. The subject does neither, so '
               "the numbers can differ slightly from EWoK's own."),
}
# What each mode asks for. Log probabilities: the raw score of each target after a context.
_TASKS = {
    'logprobs': (Choice(labels='options', raw_scores=True, raw_text=True),
                 'behavior:probabilities'),
    'choice': (Choice(labels=('1', '2'), raw_text=True), 'behavior:choice'),
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


def _answers(subject: Subject, context, stimuli, mode: str, repetition: int) -> tuple:
    """Present ``stimuli`` in one request and return one row of values per stimulus.

    The session asks for the mode's output channel with the mode's task;
    each row is its own trial. Returns the values, one row per stimulus in the
    order of ``stimuli`` and one column per option, and the labels of the
    option axis.
    """
    response, channel = _TASKS[mode]
    session = run_request(subject, context, repetition, stimuli, [channel],
                          TaskContext(response=response, trial_memory='none'))
    if not any(event.channel == channel for event in session.emitted):
        raise TypeError(f'EWoK {mode} requires an answer on {channel}')
    answers = session.collect(channel)
    if 'choice' not in getattr(answers, 'dims', ()):
        raise TypeError(f'EWoK {mode} requires one row of option values per question')
    if mode == 'logprobs' and answers.attrs.get('measure', {}).get('scale') != 'log':
        raise ValueError('EWoK log-probability mode requires raw log scores (measure scale "log")')
    requested = list(stimuli['stimulus_id'])
    order = {stimulus_id: row for row, stimulus_id in
             enumerate(response_stimulus_ids(answers, requested))}
    values = np.asarray(answers.transpose('presentation', 'choice').values, dtype=float)
    return np.array([values[order[stimulus_id]] for stimulus_id in requested]), list(answers['choice'].values)


def _mean_by_version(rows: list[dict]) -> tuple[float, dict]:
    """Average items within each dataset version, then weight versions equally."""
    groups = defaultdict(list)
    for row in rows:
        groups[row['version']].append(row['accuracy'])
    means = {key: sum(values) / len(values) for key, values in sorted(groups.items())}
    return sum(means.values()) / len(means), means


class EWoKBenchmark(BenchmarkBase):
    """Score a prepared local EWoK dataset; no automatic model/data downloads.

    ``mode='logprobs'`` shows each context as ``text`` (raw text, no template)
    and asks for the raw score of each target on ``behavior:probabilities``.
    ``mode='choice'`` shows EWoK's choice prompt as ``text`` and asks for
    ``behavior:choice`` between the labels 1 and 2. The module docstring says
    what each needs and why the numbers can differ from EWoK's own evaluator.
    """

    # Each question is its own trial; the subject keeps no history.
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
        self.required_input_channels = frozenset({'text'})
        self.requested_output_channels = frozenset({_TASKS[mode][1]})
        # Local exports and domain subsets must never share a score identity.
        identifier = f'EWoK-core-1.0-{mode}-data-{manifest["items_sha256"][:16]}'
        if domains is not None:
            selection = json.dumps(sorted(set(domains))).encode()
            identifier += f'-domains-{hashlib.sha256(selection).hexdigest()[:12]}'
        super().__init__(
            identifier=identifier, version=1, parent='behavioral',
            ceiling=not_estimated(),
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
            # Recorded requests contain restricted item text; keep runs private.
            context.directory.chmod(0o700)
            scored = []
            requests_sent = 0   # one record repetition per request
            for offset in range(0, len(self._rows), self._batch_size):
                batch = self._rows[offset:offset + self._batch_size]
                returned = {}
                # Two sessions per batch, one row per item in each. Log probabilities:
                # context n, with both targets as its options. Choice: EWoK's prompt
                # for target n. Expected answers stay in the benchmark.
                for number in (1, 2):
                    if self._mode == 'logprobs':
                        stimuli = StimulusSet(pd.DataFrame({
                            'stimulus_id': [f'{r["item_id"]}.context{number}' for r in batch],
                            'text': [r[f'Context{number}'] for r in batch],
                            'options': [(r['Target1'], r['Target2']) for r in batch],
                        }))
                    else:
                        stimuli = StimulusSet(pd.DataFrame({
                            'stimulus_id': [f'{r["item_id"]}.target{number}' for r in batch],
                            'text': [_choice_prompt(r[f'Target{number}'], r['Context1'], r['Context2'])
                                     for r in batch],
                        }))
                    returned[number] = _answers(subject, context, stimuli, self._mode,
                                                repetition=requests_sent)
                    requests_sent += 1
                for index, row in enumerate(batch):
                    if self._mode == 'logprobs':
                        # Columns are the options in order: Target1, Target2.
                        first, second = returned[1][0][index], returned[2][0][index]
                        # T1|C1, T1|C2, T2|C1, T2|C2 (target given context).
                        accuracy = logprob_accuracy([float(first[0]), float(second[0]),
                                                     float(first[1]), float(second[1])])
                        invalid = 0
                    else:
                        choices = []
                        for number in (1, 2):
                            values, labels = returned[number]
                            row_values = values[index]
                            choices.append(str(labels[int(row_values.argmax())])
                                           if np.isfinite(row_values).all() else None)
                        # An unparsed answer is wrong and counted, never dropped.
                        accuracy = ((choices[0] == '1') + (choices[1] == '2')) / 2
                        invalid = sum(choice is None for choice in choices)
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
            return adapter_score(
                self, value, metric='accuracy', context=context,
                input_adaptation=_CHOICE_NOTE if self._mode == 'choice' else _LOGPROBS_NOTE,
                protocol=self._mode,
                aggregation='mean of within-version mean accuracies',
                version_scores=versions, domain_scores=domains,
                items=len(scored), questions=2 * len(scored),
                invalid_answers=sum(r['invalid_answers'] for r in scored),
                dataset_sha256=self._manifest['items_sha256'],
                dataset_scope=self._manifest['scope'], domain_subset=self._subset,
                paper_replication=False,
            )
        return CallableProtocol(
            self.identifier,
            evaluate,
            # No method is observed: each request is recorded by run_request.
            methods=(),
            # The record repetition ids _answers uses: two per batch, counted
            # from 0. Tools select repetitions by these ids.
            repetitions=range(math.ceil(len(self._rows) / self._batch_size) * 2),
            metadata={
                'protocol': self._mode, 'dataset': self._manifest,
                'items': len(self._rows), 'batch_size': self._batch_size,
                'domains': sorted({r['Domain'] for r in self._rows}),
            },
        )

    def __call__(self, subject: Subject) -> Score:
        """Run benchmark(subject) with input/output recording and return its Score.

        Default scoring deletes the private call records on success and failure;
        an explicit ``output_dir`` keeps them. ``score()`` runs the pre-flight.
        """
        return run_recorded(subject, self.protocol(), self._output_dir)
