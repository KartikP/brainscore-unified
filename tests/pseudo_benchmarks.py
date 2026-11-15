"""Pseudo benchmarks: one for each response format a benchmark can request today.

Response Format Standard, conformance test T4 ("same format, different
implementations"). Each pseudo benchmark has made-up stimuli and made-up human
data with a known relation to the stimuli, so a stand-in producer scores a known
value. A pseudo benchmark proves that a path runs; it proves nothing about
science. It is never registered (Contract Rule 5.13: a registered benchmark has
brain or behavior data).

Each benchmark has the surface of a unified benchmark: an identifier, its inputs,
the one output it requests, its trial memory, and ``__call__(subject)`` that
builds its task and session with the public builder, runs ``participate``, reads
the stored answer with ``collect``, applies a small metric and builds its score
with ``benchmark_score``. A metric reads the values and the trial states, never
the response record.

The producers are stand-ins (no weights): a ``BrainScoreModel`` with a reply
part, one with a likelihood part, one with a readout, a custom subject that
answers one cell per trial, a custom subject that answers one assembly, and a
subject that declares only text.
"""
import re

import numpy as np

from brainscore import BrainScoreModel
from brainscore.benchmarks._score import benchmark_score, not_estimated
from brainscore_core import Subject
from brainscore_core.benchmarks import BenchmarkBase
from brainscore_core.contract import TaskContext
from brainscore_core.response_formats import argmax, response_record
from brainscore_core.responses import Choice, Scale, Text
from brainscore_core.streaming import StreamEvent
from brainscore_core.streaming_helpers import stimulus_session
from brainscore_core.supported_data_standards.brainio.assemblies import (
    BehavioralAssembly, NeuroidAssembly)
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet

# --- the made-up experiment ------------------------------------------------------------

IDS = [f's{i}' for i in range(8)]
TRUTH = ['real' if i % 2 == 0 else 'pseudo' for i in range(8)]   # what each item is
HARD = {'s2', 's5'}                    # items the stand-ins are less sure of
OPTIONS = ['real', 'pseudo']
# Made-up human data. People err on the two hard items; they rate and time them as hard.
HUMAN_CHOICE = dict(zip(IDS, ['real', 'pseudo', 'pseudo', 'pseudo', 'real', 'real', 'real', 'pseudo']))
HUMAN_RATING = dict(zip(IDS, [7.0, 1.0, 4.0, 2.0, 6.0, 3.0, 6.0, 2.0]))     # on Scale(1, 7, 1)
HUMAN_RT_MS = dict(zip(IDS, [500.0, 520.0, 800.0, 510.0, 490.0, 780.0, 505.0, 515.0]))


def _table(column, values, identifier):
    stimuli = StimulusSet({'stimulus_id': IDS, column: values})
    stimuli.identifier = identifier
    if column == 'image_path':
        stimuli.stimulus_paths = dict(zip(IDS, values))
    return stimuli


def text_stimuli():
    return _table('sentence', [f'{"real" if t == "real" else "fake"} word {i}'
                               for i, t in enumerate(TRUTH)], 'pseudo-words')


def image_stimuli(folder):
    """Eight 4x4 images: white for a real item, black for a pseudo item."""
    from PIL import Image
    paths = []
    for i, truth in enumerate(TRUTH):
        path = folder / f'pseudo_{i}.png'
        Image.new('RGB', (4, 4), (255, 255, 255) if truth == 'real' else (0, 0, 0)).save(path)
        paths.append(str(path))
    return _table('image_path', paths, 'pseudo-word-images')


def _fitting(labels, identifier):
    stimuli = StimulusSet({'stimulus_id': ['f0', 'f1', 'f2', 'f3'],
                           'sentence': ['real 0', 'fake 1', 'real 2', 'fake 3'], 'label': labels})
    stimuli.identifier = identifier
    return stimuli


FITTING = _fitting(['real', 'pseudo', 'real', 'pseudo'], 'pseudo-fit')
FITTING_RATINGS = _fitting([6.0, 2.0, 6.0, 2.0], 'pseudo-fit-ratings')


def _states(answer):
    return dict(zip(answer['stimulus_id'].values, answer['response_state'].values))


def _counts(answer):
    states = list(answer['response_state'].values)
    return {'n_trials': len(states), 'n_invalid': states.count('invalid'),
            'n_missing': states.count('missing')}


def _pearson(pairs):
    model, human = (np.array(column, dtype=float) for column in zip(*pairs)) if pairs else ([], [])
    return float(np.corrcoef(model, human)[0, 1]) if len(pairs) > 2 else float('nan')


# --- the pseudo benchmarks -------------------------------------------------------------

class PseudoChoice(BenchmarkBase):
    """Accuracy of the chosen option against made-up human choices.
    Policy: an invalid or missing choice counts as wrong."""
    requested_output_channels = frozenset({'behavior:choice'})
    trial_memory = 'none'

    def __init__(self, channel='text', folder=None):
        self.required_input_channels = frozenset({channel})
        self.stimuli = text_stimuli() if channel == 'text' else image_stimuli(folder)
        super().__init__(identifier=f'pseudo-choice-{channel}', version=1, parent='behavioral',
                         ceiling=not_estimated())

    def __call__(self, subject):
        task = TaskContext(instruction='Is this a real word?', response=Choice(OPTIONS),
                           fitting_stimuli=FITTING)
        session = stimulus_session(self.stimuli, ['behavior:choice'], task)
        subject.participate(session)
        answer = session.collect('behavior:choice')
        options = list(answer['choice'].values)
        correct = [state == 'valid' and options[int(np.argmax(row))] == HUMAN_CHOICE[sid]
                   for sid, state, row in zip(answer['stimulus_id'].values,
                                              answer['response_state'].values, answer.values)]
        correct += [False] * (len(IDS) - len(correct))
        accuracy = float(np.mean(correct))
        return benchmark_score(self, accuracy, raw=accuracy, metric='accuracy', normalized=False,
                               **_counts(answer))


class PseudoProbabilities(BenchmarkBase):
    """Mean probability given to the option people chose.
    Policy: an invalid or missing row is left out and counted."""
    requested_output_channels = frozenset({'behavior:probabilities'})
    required_input_channels = frozenset({'text'})
    trial_memory = 'none'

    def __init__(self, fitting=True):
        self.stimuli, self.fitting = text_stimuli(), FITTING if fitting else None
        super().__init__(identifier='pseudo-probabilities', version=1, parent='behavioral',
                         ceiling=not_estimated())

    def __call__(self, subject):
        task = TaskContext(instruction='Is this a real word?', response=Choice(OPTIONS),
                           fitting_stimuli=self.fitting)
        session = stimulus_session(self.stimuli, ['behavior:probabilities'], task)
        subject.participate(session)
        answer = session.collect('behavior:probabilities')
        options = list(answer['choice'].values)
        given = [row[options.index(HUMAN_CHOICE[sid])]
                 for sid, state, row in zip(answer['stimulus_id'].values,
                                            answer['response_state'].values, answer.values)
                 if state == 'valid']
        value = float(np.mean(given)) if given else float('nan')
        return benchmark_score(self, value, raw=value, metric='mean probability of the human choice',
                               normalized=False, **_counts(answer))


class PseudoRating(BenchmarkBase):
    """Pearson correlation with made-up human ratings on a 1 to 7 scale, step 1.
    Policy: an invalid or missing rating is left out and counted."""
    requested_output_channels = frozenset({'behavior:rating'})
    required_input_channels = frozenset({'text'})
    trial_memory = 'none'
    scale = Scale(1, 7, 1)

    def __init__(self):
        self.stimuli = text_stimuli()
        super().__init__(identifier='pseudo-rating', version=1, parent='behavioral',
                         ceiling=not_estimated())

    def __call__(self, subject):
        task = TaskContext(instruction='How sure are you that this is a real word?',
                           response=self.scale, fitting_stimuli=FITTING_RATINGS)
        session = stimulus_session(self.stimuli, ['behavior:rating'], task)
        subject.participate(session)
        answer = session.collect('behavior:rating')
        pairs = [(value, HUMAN_RATING[sid]) for sid, state, value in zip(
            answer['stimulus_id'].values, answer['response_state'].values, answer.values)
            if state == 'valid']
        value = _pearson(pairs)
        return benchmark_score(self, value, raw=value, metric='pearson r with human ratings',
                               normalized=False, **_counts(answer))


class PseudoResponseTime(BenchmarkBase):
    """Pearson correlation with made-up human response times (ms).
    A model that derives a time from probabilities uses the link the task names
    (surprisal); a subject that measures a time answers it with its units.
    Policy: an invalid or missing time is left out and counted."""
    requested_output_channels = frozenset({'behavior:rt'})
    required_input_channels = frozenset({'text'})
    trial_memory = 'none'

    def __init__(self):
        self.stimuli = text_stimuli()
        super().__init__(identifier='pseudo-response-time', version=1, parent='behavioral',
                         ceiling=not_estimated())

    def __call__(self, subject):
        task = TaskContext(instruction='Is this a real word?',
                           response=Choice(OPTIONS, reaction_time='surprisal'),
                           fitting_stimuli=FITTING)
        session = stimulus_session(self.stimuli, ['behavior:rt'], task)
        subject.participate(session)
        answer = session.collect('behavior:rt')
        pairs = [(value, HUMAN_RT_MS[sid]) for sid, state, value in zip(
            answer['stimulus_id'].values, answer['response_state'].values, answer.values)
            if state == 'valid']
        value = _pearson(pairs)
        return benchmark_score(self, value, raw=value, metric='pearson r with human response times',
                               normalized=False, **_counts(answer))


class PseudoText(BenchmarkBase):
    """Exact match of the parsed answer with made-up human answers. The benchmark parses
    the free text itself ('Answer: <word>'), as an external library does.
    Policy: a text the parse cannot read, an invalid or a missing trial counts as wrong."""
    requested_output_channels = frozenset({'behavior:text'})
    required_input_channels = frozenset({'text'})
    trial_memory = 'none'

    def __init__(self):
        self.stimuli = text_stimuli()
        super().__init__(identifier='pseudo-text', version=1, parent='behavioral',
                         ceiling=not_estimated())

    @staticmethod
    def parse(text):
        found = re.search(r'answer:\s*(\w+)', text or '', re.IGNORECASE)
        return found.group(1).lower() if found else None

    def __call__(self, subject):
        task = TaskContext(instruction='Is this a real word? Write "Answer: real" or "Answer: pseudo".',
                           response=Text(max_tokens=8, temperature=0))
        session = stimulus_session(self.stimuli, ['behavior:text'], task)
        subject.participate(session)
        answer = session.collect('behavior:text')
        parsed = [self.parse(text) if state == 'valid' else None for state, text in zip(
            answer['response_state'].values, answer.values)]
        correct = [p == HUMAN_CHOICE[sid] for sid, p in zip(answer['stimulus_id'].values, parsed)]
        accuracy = float(np.mean(correct))
        return benchmark_score(self, accuracy, raw=accuracy, metric='exact match after parse',
                               normalized=False, n_unparsed=parsed.count(None), **_counts(answer))


BENCHMARKS = {'behavior:choice': PseudoChoice, 'behavior:probabilities': PseudoProbabilities,
              'behavior:rating': PseudoRating, 'behavior:rt': PseudoResponseTime,
              'behavior:text': PseudoText}

# --- the producers (stand-ins) ---------------------------------------------------------


def _sure(stimulus_id):
    """How sure every stand-in is of the item's truth: 0.6 on a hard item, else 0.9."""
    return 0.6 if stimulus_id in HARD else 0.9


def _truth(stimulus_id):
    return TRUTH[IDS.index(stimulus_id)]


def _rating(stimulus_id):
    return 4.0 if stimulus_id in HARD else 6.0 if _truth(stimulus_id) == 'real' else 2.0


def _features(stimuli, layers, **kwargs):
    """The readout's activity: two units that tell a real item from a pseudo item, less
    clearly for a hard item."""
    ids = list(stimuli['stimulus_id'])
    values = np.array([([1.5, 0.5] if sid in HARD else [2.0, 0.0])[::1 if 'real' in text else -1]
                       for sid, text in zip(ids, stimuli['sentence'])])
    return NeuroidAssembly(values, dims=['presentation', 'neuroid'], coords={
        'stimulus_id': ('presentation', ids), 'sentence': ('presentation', list(stimuli['sentence'])),
        'neuroid_id': ('neuroid', ['n0', 'n1']), 'layer': ('neuroid', [layers[0]] * 2)})


def reply_model(calls=None, replies=None, in_channels=('text',)):
    """A BrainScoreModel with a reply part. ``replies`` overrides the reply of a stimulus."""
    calls = [] if calls is None else calls
    replies = replies or {}

    def reply(prompt):
        calls.append('reply')
        sid = prompt.stimulus_id
        if sid in replies:
            return replies[sid]
        if isinstance(prompt.response, Scale):
            return f'{_rating(sid):g}'
        if isinstance(prompt.response, Text):
            return f'Answer: {_truth(sid)}'
        return _truth(sid)
    return BrainScoreModel('pseudo-reply', in_channels=set(in_channels), reply=reply)


def likelihood_model(calls=None):
    """A BrainScoreModel with a likelihood part and its statements."""
    calls = [] if calls is None else calls

    def likelihood(prompt, labels):
        calls.append('likelihood')
        sure = _sure(prompt.stimulus_id)
        return {label: float(np.log(sure if label == _truth(prompt.stimulus_id) else 1 - sure))
                for label in labels}
    return BrainScoreModel('pseudo-likelihood', in_channels={'text'}, likelihood=likelihood,
                           capability_config={'behavior-sessions': {
                               'behavior:choice': ('likelihood', 'softmax_argmax'),
                               'behavior:probabilities': ('likelihood', 'softmax')}})


def readout_model(calls=None):
    """A BrainScoreModel with a readout and its statement; the benchmark gives fitting stimuli."""
    calls = [] if calls is None else calls

    def extract(stimuli, layers, **kwargs):
        calls.append('extract')
        return _features(stimuli, layers, **kwargs)
    return BrainScoreModel('pseudo-readout', extractor={'text': extract}, in_channels={'text'},
                           behavioral_readout_layer='late',
                           capability_config={'behavior-sessions': {
                               'behavior:choice': ('readout', 'argmax')}})


ALL_FORMATS = frozenset(BENCHMARKS)


def _probabilities(stimulus_id, options):
    sure = _sure(stimulus_id)
    return np.array([sure if option == _truth(stimulus_id) else 1 - sure for option in options])


class CellSubject(Subject):
    """A custom subject that answers one cell for each trial: a string (choice, text), a
    small array (probabilities) or a number (rating; a time in ms with its units).

    ``cells`` overrides the cell of ``(channel, stimulus_id)``; ``skip`` names stimuli it
    does not answer. ``calls`` counts its trials."""
    identifier = 'pseudo-cells'
    required_channels = frozenset()

    def __init__(self, out_channels=ALL_FORMATS, in_channels=('text', 'vision:image'),
                 cells=None, skip=()):
        self._out, self._in = frozenset(out_channels), frozenset(in_channels)
        self.cells, self.skip, self.calls = cells or {}, set(skip), []

    # The contract's Subject makes these abstract properties: an instance attribute set in
    # __init__ does not satisfy them, so a subject that varies them writes two properties.
    @property
    def in_channels(self):
        return self._in

    @property
    def out_channels(self):
        return self._out

    def reset(self):
        pass

    def cell(self, channel, sid, options):
        if (channel, sid) in self.cells:
            return self.cells[channel, sid]
        return {'behavior:choice': lambda: _truth(sid),
                'behavior:probabilities': lambda: _probabilities(sid, options),
                'behavior:rating': lambda: _rating(sid),
                'behavior:rt': lambda: 750.0 if sid in HARD else 450.0,
                'behavior:text': lambda: f'Answer: {_truth(sid)}'}[channel]()

    def participate(self, session):
        response = session.task.response
        options = list(response.labels) if isinstance(response, Choice) else None
        while (event := session.next_input()) is not None:
            sid = event.meta.get('stimulus_id')
            if sid is None or sid in self.skip:
                continue
            self.calls.append(sid)
            for channel in session.outputs:
                meta = {'stimulus_id': sid, **({'units': 'ms'} if channel == 'behavior:rt' else {})}
                session.respond(StreamEvent(channel, self.cell(channel, sid, options), event.t_ms, meta))


class AssemblySubject(Subject):
    """A custom subject that answers one assembly for each requested format, with its
    response record. Its choice is derived from its own probabilities with core's argmax,
    so its record names the conversion and keeps the source rows."""
    identifier = 'pseudo-assembly'
    in_channels = frozenset({'text'})
    out_channels = ALL_FORMATS
    required_channels = frozenset()
    METHOD = 'lookup'                 # a subject that is not a BrainScoreModel names its own

    def __init__(self):
        self.calls = []

    def reset(self):
        pass

    def answer(self, channel, ids, options):
        coords = {'stimulus_id': ('presentation', ids)}
        if channel in ('behavior:choice', 'behavior:probabilities'):
            rows = np.array([_probabilities(sid, options) for sid in ids])
            if channel == 'behavior:probabilities':
                return BehavioralAssembly(rows.astype(np.float32), dims=['presentation', 'choice'],
                                          coords={**coords, 'choice': options},
                                          attrs={'response': response_record(channel, self.METHOD)})
            indices, valid = argmax(rows)
            one_hot = np.where(valid[:, None], np.eye(len(options))[indices], np.nan).astype(np.float32)
            coords['raw_answer'] = ('presentation', [str(list(row)) for row in rows])
            record = response_record(channel, self.METHOD, conversion='argmax',
                                     source_format='behavior:probabilities')
            return BehavioralAssembly(one_hot, dims=['presentation', 'choice'],
                                      coords={**coords, 'choice': options}, attrs={'response': record})
        if channel == 'behavior:text':
            values = np.array([f'Answer: {_truth(sid)}' for sid in ids], dtype=object)
            return BehavioralAssembly(values, dims=['presentation'], coords=coords,
                                      attrs={'response': response_record(channel, self.METHOD)})
        values = np.array([_rating(sid) if channel == 'behavior:rating'
                           else 750.0 if sid in HARD else 450.0 for sid in ids])
        parameters = {'units': 'ms'} if channel == 'behavior:rt' else {}
        return BehavioralAssembly(values, dims=['presentation'], coords=coords, attrs={
            'response': response_record(channel, self.METHOD, parameters=parameters)})

    def participate(self, session):
        response = session.task.response
        options = list(response.labels) if isinstance(response, Choice) else None
        ids = []
        while (event := session.next_input()) is not None:
            if event.meta.get('stimulus_id') is not None:
                ids.append(event.meta['stimulus_id'])
        self.calls.extend(ids)
        for channel in session.outputs:
            session.respond(StreamEvent(channel, self.answer(channel, ids, options), None, {}))


def text_only_subject():
    """A subject that declares only free text: it does not declare the other formats."""
    return CellSubject(out_channels={'behavior:text'}, in_channels={'text'})
