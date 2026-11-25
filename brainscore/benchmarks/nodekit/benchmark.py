"""NodeKit tasks as behavior benchmarks: the subject takes part in the task as the participants did (the
same program and trials, ``brainscore.harnesses.nodekit``), and its trials are compared with theirs.

A run's result is the program's list of trial records, as for a participant (``data.rows``). Practice
trials are out of every score: only ``phase == 'test'`` rows are compared. Rows are matched by what was
shown (the condition, the triplet, the pair of faces), never by trial number. Each score states its form
(``score_form``), its chance or floor, and its ceiling.

One episode is one whole run of the task, with its instructions, practice and feedback, as the participants
did it (``trial_memory = 'none'``: each run starts empty; within the run a subject keeps what it keeps, as a
participant does; nkl's trace marks no trial starts, so there is no begin or end for each trial). The run
may last twice the participants' longest run in simulated time (``max_ms``, read from their traces) and
20000 steps; then the harness leaves it. nkl gives the trial records only when the program completes, so a
run that does not complete has none: every test trial is missing and the result is invalid. The subject's
response times are simulated times (``subject_response_ms``): recorded, not scored. Each run's trace is
kept under ``record`` (``attrs['trace']``) and replays in the task repository's viewer
(``python -m brainscore.harnesses.nodekit replay <record>``).

- ``NodeKit.DriftingGabor`` (after Tadin et al. 2003): the subject's proportion correct, as the mean
  over the 10 conditions (patch size x duration; one contrast, 0.95), divided by the participants' mean
  proportion correct over the same conditions. Chance is 0.5 (raw). Per condition, the subject's and the
  participants' proportions are in ``attrs['by_condition']``.
- ``NodeKit.TripletSimilarity`` (NIGHTS triplets): for each test trial of the subject, the share of the
  participants' answers to the same triplet that chose the same candidate; the mean over trials, divided
  by the ceiling. Chance is 0.5 (raw) for any participants' answers. Ceiling: each participant against the
  other participants in the same way (leave one participant out), the mean over participants.
- ``NodeKit.FaceArrangement`` (after Kriegeskorte and Mur 2012): the distances of the subject's final
  arrangements, one per pair of faces shown together (100 pairs), against the mean distances of the
  participants: Spearman correlation, divided by the ceiling. Ceiling: each participant against the mean
  of the others (leave one participant out), the mean over participants. Floor: the mean Spearman
  correlation of 1000 random permutations of the participants' mean distances with themselves (seed 0),
  with its 2.5 and 97.5 percentiles.
"""
from __future__ import annotations

import json
from functools import cached_property

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from brainscore_core.benchmarks import BenchmarkBase
from brainscore_core.metrics import Score
from brainscore.benchmarks._score import benchmark_score
from brainscore.benchmarks.nodekit import data

ANSWER_POLICY = ('A trial the subject did not reach is missing; a run that does not complete has no trial '
                 'record, so all its test trials are missing and the result is invalid.')


def test_rows(table: pd.DataFrame) -> pd.DataFrame:
    """The test trials only: practice trials never reach a score."""
    return table[table['phase'] == 'test'] if 'phase' in table else table


# --- Drifting Gabor -----------------------------------------------------------------------------------------
GABOR_CONDITION = ['size_degrees', 'duration_milliseconds']


def gabor_by_condition(table: pd.DataFrame) -> pd.Series:
    return test_rows(table).astype({'correct': float}).groupby(GABOR_CONDITION)['correct'].mean()


def _gabor_people(participants: pd.DataFrame) -> pd.Series:
    return participants.groupby('participant').apply(gabor_by_condition).mean()      # by condition


def gabor_ceiling(participants: pd.DataFrame) -> float:
    return float(_gabor_people(participants).mean())


def gabor(subject: pd.DataFrame, participants: pd.DataFrame) -> dict:
    people = _gabor_people(participants)
    own = gabor_by_condition(subject).reindex(people.index)
    return {'raw': float(own.mean()), 'chance': 0.5,
            'score_form': 'proportion correct (mean over size x duration conditions) / participants\' proportion',
            'by_condition': [{'size_degrees': size, 'duration_milliseconds': duration, 'subject': float(own[size, duration]),
                              'participants': float(people[size, duration])} for size, duration in people.index]}


# --- Triplet similarity -------------------------------------------------------------------------------------
TRIPLET = ['stimulus_id', 'left_id', 'right_id']


def _agreement(own: pd.DataFrame, others: pd.DataFrame) -> float:
    """The mean, over ``own`` test trials, of the share of ``others``' answers to the same triplet that agree."""
    shares = test_rows(others).groupby(TRIPLET)['response'].apply(lambda answers: answers.tolist())
    agree = [np.mean([answer == row.response for answer in shares[tuple(row[TRIPLET])]])
             for _, row in test_rows(own).iterrows()]
    return float(np.mean(agree))


def triplets_ceiling(participants: pd.DataFrame) -> float:
    return float(np.mean([_agreement(participants[participants.participant == p],
                                     participants[participants.participant != p])
                          for p in participants.participant.unique()]))


def triplets(subject: pd.DataFrame, participants: pd.DataFrame) -> dict:
    return {'raw': _agreement(subject, participants), 'chance': 0.5,
            'score_form': 'agreement with the participants\' choice per triplet / leave-one-participant-out agreement'}


# --- Face arrangement ---------------------------------------------------------------------------------------
def pair_distances(table: pd.DataFrame) -> pd.Series:
    """The final distance of each pair of faces shown together, by the pair's two face ids (sorted)."""
    pairs = {}
    for _, row in test_rows(table).iterrows():
        ids = row['face_ids'] if isinstance(row['face_ids'], list) else json.loads(row['face_ids'])
        distances = row['distances'] if isinstance(row['distances'], list) else json.loads(row['distances'])
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                pairs[tuple(sorted((ids[i], ids[j])))] = distances[i][j]
    return pd.Series(pairs).sort_index()


def _each(participants: pd.DataFrame) -> dict:
    return {p: pair_distances(participants[participants.participant == p]) for p in participants.participant.unique()}


def faces_ceiling(participants: pd.DataFrame) -> float:
    each = _each(participants)
    return float(np.mean([spearmanr(each[p], pd.concat([each[q] for q in each if q != p], axis=1).mean(axis=1)
                                    .reindex(each[p].index))[0] for p in each]))


def faces(subject: pd.DataFrame, participants: pd.DataFrame) -> dict:
    mean = pd.concat(_each(participants), axis=1).mean(axis=1)
    rng = np.random.default_rng(0)
    floor = [spearmanr(rng.permutation(mean.values), mean.values)[0] for _ in range(1000)]
    own = pair_distances(subject).reindex(mean.index)
    return {'raw': float(spearmanr(own.values, mean.values)[0]),
            'floor': float(np.mean(floor)), 'floor_interval': [float(np.percentile(floor, 2.5)),
                                                              float(np.percentile(floor, 97.5))],
            'score_form': 'Spearman of pair distances with the participants\' mean / leave-one-participant-out'}


# The act form each task's participants used (Contract Rules 5.15): the Gabor answer is a key press; the
# triplet and face answers are pointer acts.
ACT_FORMS = {'drifting-gabor': 'motor:key', 'triplet-similarity': 'motor:pointer', 'face-arrangement': 'motor:pointer'}
# Each task's metric and ceiling: metric(subject, participants) -> {'raw', 'score_form', ...}; ceiling(participants).
METRICS = {'drifting-gabor': (gabor, gabor_ceiling), 'triplet-similarity': (triplets, triplets_ceiling),
           'face-arrangement': (faces, faces_ceiling)}


class NodeKitBenchmark(BenchmarkBase):
    """One NodeKit task (a bundle of the task repository) as a behavior benchmark. The subject sees the
    screen on ``vision:image`` (and the acts on ``instruction``) and acts on the form its participants used
    (``ACT_FORMS``): a subject that does not declare that form is N/A before any compute."""

    trial_memory = 'none'       # one run of the task is one episode that starts empty

    def __init__(self, identifier: str, task: str, max_steps: int = 20000, record=None):
        self.task, self.max_steps, self.record = task, max_steps, record
        # The screen is the stimulus: a subject that cannot take it never sees the task (N/A, Rule 5.1).
        # The frames since the last step go on vision:video as well; a subject without it sees the screen only.
        self.essential_input_channels = frozenset({'vision:image'})
        self.required_input_channels = self.essential_input_channels | {'vision:video', 'instruction'}
        self.requested_output_channels = frozenset({ACT_FORMS[task]})
        super().__init__(identifier=identifier, version=1, parent='behavior', ceiling=None, bibtex='')

    @cached_property
    def _trials(self):
        return data.trials(self.task)

    @cached_property
    def participants(self) -> pd.DataFrame:
        return data.table(self._trials)

    @property
    def max_ms(self) -> float:
        """Twice the participants' longest run, in simulated ms."""
        return 2 * self._trials.attrs['longest_run_ms']

    @cached_property
    def ceiling(self) -> Score:
        return Score(METRICS[self.task][1](self.participants))

    def __call__(self, subject) -> Score:
        from brainscore.data.local import data_home
        from brainscore.harnesses.nodekit import bundle, play
        record = self.record or data_home() / 'records' / 'nodekit'
        result, environment, session = play(subject, bundle(self.task), form=ACT_FORMS[self.task], max_steps=self.max_steps,
                                            max_ms=self.max_ms, record=record)
        return self.score_result(result, steps=environment.steps, invalid_steps=environment.invalid_steps,
                                 run_ms=environment.t_ms, end_reason=session.source.end_reason,
                                 trace=str(environment.trace))

    def score_result(self, result, **run) -> Score:
        """The score of one run's result (None for a run that did not complete)."""
        tests = len(test_rows(self.participants[self.participants.participant == 'p1']))
        if result is None:
            measured = {'raw': float('nan'), 'score_form': 'no trial record'}
            states = ['missing'] * tests
        else:
            own = test_rows(data.rows(result))
            measured = METRICS[self.task][0](own, self.participants)
            states = ['valid'] * len(own) + ['missing'] * max(0, tests - len(own))
            if 'response_milliseconds' in own:
                measured['subject_response_ms'] = [float(t) for t in own['response_milliseconds']]
        attrs = {key: value for key, value in measured.items() if key not in ('raw', 'score_form')}
        return benchmark_score(self, raw=measured['raw'], metric=measured['score_form'],
                               ceiling=self.ceiling, answers=pd.DataFrame({'response_state': states}),
                               answer_policy=ANSWER_POLICY, invalid_limit=0.1, max_ms=self.max_ms,
                               max_steps=self.max_steps, **run, **attrs)
