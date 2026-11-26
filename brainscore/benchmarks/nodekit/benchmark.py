"""NodeKit tasks as behavior benchmarks: the subject takes part in the task as the participants did (the
same program and trials, ``brainscore.harnesses.nodekit``), and its trials are compared with theirs.

A run's result is the program's list of trial records, as for a participant (``data.rows``). Practice
trials are out of every score: only ``phase == 'test'`` rows are compared. Rows are matched by what was
shown (the condition, the triplet, the pair of faces), never by trial number. Each score states its form
(``score_form``), its chance or floor, and its ceiling.

One episode is one run of the task, with its instructions, practice and feedback, as the participants
did it (``trial_memory = 'none'``: each run starts empty; within the run a subject keeps what it keeps, as a
participant does; nkl's trace marks no trial starts, so there is no begin or end for each trial). The defaults
are the NodeKit author's evaluation settings (owner, 2026-11-25; ``AUTHOR_*`` below): nkl's agent view (one
picture a turn and the look act), realtime pacing (the subject's answering time passes on the task's clock, less
nkl's network allowance), a 640 x 480 picture, 100 acts a reply, 4 looks, his hand, and his slice of each task
(Gabor and triplets: the practice block and the first 10 test trials; faces: its 10 trials) with his turns and
task clock. The score states each of them, and names the author's settings when all hold. ``trials='all'`` runs
the whole task (20000 steps; twice the participants' longest run in simulated time). The participant view (the
frames since the last step and the screen), the frozen and dense pacings (with ``dense_step_ms``) and the other
settings are stated settings. The ceiling and the chance or floor are computed on the participants' same trials
(their first test trials for a slice). nkl gives the trial records only when the program completes, so a
run that does not complete has none: every test trial is missing and the result is invalid. The subject's
response times are simulated times (``subject_response_ms``): recorded, not scored. Each run's trace is
kept under ``record`` (``attrs['trace']``) and replays in the task repository's viewer
(``python -m brainscore.harnesses.nodekit replay <record>``).

- ``NodeKit.DriftingGabor`` (after Tadin et al. 2003; owner, 2026-11-25): the Pearson correlation of the
  subject's proportion correct in each of the 10 conditions (patch size x duration; one contrast, 0.95) with
  the participants' mean proportions, divided by the ceiling: each participant's profile against the mean of
  the others (leave one participant out), the mean over participants. Chance is 0 (a responder at chance has
  no profile). One run has 5 test trials a condition, so one run's proportions move in steps of 0.2 (noise);
  repeated runs of a subject are scored together (``score_result`` with a list of runs). A profile that is the
  same in every condition, or a condition with no trial, has no correlation: the score states it and is not a
  number. Extra values: the overall proportion correct (``proportion_correct``, the participants'
  ``proportion_correct_participants``) and the per-condition table (``by_condition``).
- ``NodeKit.TripletSimilarity`` (NIGHTS triplets): for each test trial of the subject, the share of the
  participants' answers to the same triplet that chose the same candidate; the mean over trials, divided
  by the ceiling. Chance is 0.5 (raw) for any participants' answers. Ceiling: each participant against the
  other participants in the same way (leave one participant out), the mean over participants.
- ``NodeKit.FaceArrangement`` (after Kriegeskorte and Mur 2012; each score states the arrangements' spread, the
  largest distance between two faces in arena radii, beside the participants'): the distances of the subject's final
  arrangements, one per pair of faces shown together (100 pairs), against the mean distances of the
  participants: Spearman correlation, divided by the ceiling. Ceiling: each participant against the mean
  of the others (leave one participant out), the mean over participants. Floor: the mean Spearman
  correlation of 1000 random permutations of the participants' mean distances with themselves (seed 0),
  with its 2.5 and 97.5 percentiles.
"""
from __future__ import annotations

import functools
from typing import Optional
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


def _profile_correlation(own: pd.Series, reference: pd.Series) -> float:
    """Pearson correlation of two per-condition profiles; NaN when one is constant or has a missing condition."""
    if own.isna().any() or own.nunique() < 2 or reference.nunique() < 2:
        return float('nan')
    return float(np.corrcoef(own.values, reference.values)[0, 1])


def gabor_ceiling(participants: pd.DataFrame) -> float:
    each = participants.groupby('participant').apply(gabor_by_condition)
    return float(np.nanmean([_profile_correlation(each.loc[p], each.drop(p).mean()) for p in each.index]))


def gabor(subject: pd.DataFrame, participants: pd.DataFrame) -> dict:
    people = _gabor_people(participants)
    own = gabor_by_condition(subject).reindex(people.index)
    raw = _profile_correlation(own, people)
    measured = {'raw': raw, 'chance': 0.0,
                'score_form': 'Pearson correlation of the per-condition proportion correct (size x duration) with '
                              'the participants\' mean / leave-one-participant-out',
                'proportion_correct': float(test_rows(subject)['correct'].astype(float).mean()),
                'proportion_correct_participants': float(people.mean()),
                'by_condition': [{'size_degrees': size, 'duration_milliseconds': duration,
                                  'subject': float(own[size, duration]), 'participants': float(people[size, duration])}
                                 for size, duration in people.index]}
    if not np.isfinite(raw):
        measured['score_note'] = ('no correlation: the subject\'s proportion is the same in every condition'
                                  if not own.isna().any() else 'no correlation: a condition has no trial')
    return measured


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
    """The final distance of each pair of faces shown together, by the pair's two face ids (sorted); a pair
    shown in more than one trial has the mean of its distances."""
    pairs = {}
    for _, row in test_rows(table).iterrows():
        ids = row['face_ids'] if isinstance(row['face_ids'], list) else json.loads(row['face_ids'])
        distances = row['distances'] if isinstance(row['distances'], list) else json.loads(row['distances'])
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                pairs.setdefault(tuple(sorted((ids[i], ids[j]))), []).append(distances[i][j])
    return pd.Series({pair: float(np.mean(values)) for pair, values in pairs.items()}).sort_index()


def _each(participants: pd.DataFrame) -> dict:
    return {p: pair_distances(participants[participants.participant == p]) for p in participants.participant.unique()}


def faces_ceiling(participants: pd.DataFrame) -> float:
    each = _each(participants)
    return float(np.mean([spearmanr(each[p], pd.concat([each[q] for q in each if q != p], axis=1).mean(axis=1)
                                    .reindex(each[p].index))[0] for p in each]))


@functools.lru_cache(maxsize=4)
def _faces_floor(mean: tuple) -> list:
    """The Spearman correlations of 1000 random permutations (seed 0) of the participants' mean distances
    with themselves; computed once for each set of means."""
    rng = np.random.default_rng(0)
    return [spearmanr(rng.permutation(mean), mean)[0] for _ in range(1000)]


def spread(table: pd.DataFrame) -> list:
    """Each test trial's largest distance between two faces, in arena radii (2.0: opposite edges of the arena):
    how far the arrangement spreads (a pile of faces at the centre is near 0)."""
    return [value for value in _spreads(table) if value is not None]


def _spreads(table: pd.DataFrame) -> list:
    """Each test trial's spread (``spread``), None for a trial that records no positions."""
    out = []
    for _, row in test_rows(table).iterrows():
        positions = row.get('positions')
        if positions is None or (isinstance(positions, float) and np.isnan(positions)):
            out.append(None)
            continue
        positions = positions if isinstance(positions, list) else json.loads(positions)
        points = np.array([[p['x'], p['y']] for p in positions], float)
        out.append(float(np.linalg.norm(points[:, None] - points[None], axis=-1).max() / float(row['arena_radius'])))
    return out


def faces(subject: pd.DataFrame, participants: pd.DataFrame) -> dict:
    mean = pd.concat(_each(participants), axis=1).mean(axis=1)
    floor = _faces_floor(tuple(mean.values))
    own = pair_distances(subject).reindex(mean.index)
    return {'raw': float(spearmanr(own.values, mean.values)[0]),
            'floor': float(np.mean(floor)), 'floor_interval': [float(np.percentile(floor, 2.5)),
                                                              float(np.percentile(floor, 97.5))],
            'score_form': 'Spearman of pair distances with the participants\' mean / leave-one-participant-out',
            **_spread_attrs(subject, participants)}


def _spread_attrs(subject: pd.DataFrame, participants: pd.DataFrame) -> dict:
    """The arrangements' spread (``spread``): the subject's per trial, median and range, beside the participants'
    range of medians."""
    own = spread(subject)
    people = [float(np.median(spread(participants[participants.participant == p])))
              for p in participants.participant.unique() if spread(participants[participants.participant == p])]
    return {'spread_by_trial': own, 'spread_median': float(np.median(own)) if own else float('nan'),
            'spread_range': [min(own), max(own)] if own else [], 'participants_spread_medians': people}


# The act form each task's participants used (Contract Rules 5.15): the Gabor answer is a key press; the
# triplet and face answers are pointer acts.
ACT_FORMS = {'drifting-gabor': 'motor:key', 'triplet-similarity': 'motor:pointer', 'face-arrangement': 'motor:pointer'}
# The benchmark's instructions at the end of nkl's briefing: what the participants could do that the act form
# does not show. Gabor's pages have a Next button for the pointer; in the key form, Enter or the right arrow
# key does what it does (checked on the first page).
INSTRUCTIONS = {'drifting-gabor': 'Enter or the right arrow key turns an instruction page, as the Next button does.'}
# Each task's metric and ceiling: metric(subject, participants) -> {'raw', 'score_form', ...}; ceiling(participants).
METRICS = {'drifting-gabor': (gabor, gabor_ceiling), 'triplet-similarity': (triplets, triplets_ceiling),
           'face-arrangement': (faces, faces_ceiling)}
# The defaults are the NodeKit author's evaluation settings (the task repository's src/umi_nkl/evaluations/full.toml at
# AUTHOR_COMMIT; owner, 2026-11-25): nkl's agent view under realtime pacing (nkl's network allowance, 500 ms), a
# 640 x 480 picture of the 1024 x 768 screen, 100 acts a reply, 4 looks, his hand (``AUTHOR_MOTOR``), and for each
# task his slice, turns and task clock (``AUTHOR_TASKS``: Gabor and triplets, the practice block and the first 10 test
# trials; faces, its 10 trials). ``trials='all'`` runs the whole task (20000 steps, twice the participants' longest
# run). The model's request settings (reasoning, max_tokens 8000, no temperature) belong to its registration.
AUTHOR_COMMIT = '0ef0518'
AUTHOR_SETTINGS = f"the NodeKit author's settings, umi-nkl {AUTHOR_COMMIT}"
AUTHOR_PICTURE = (640, 480)
AUTHOR_ACTS = 100
AUTHOR_MOTOR = {'top_speed': 5.0, 'top_acceleration': 0.08}
AUTHOR_TASKS = {'drifting-gabor': {'shown': 10, 'max_steps': 160, 'max_ms': 900_000},
                'triplet-similarity': {'shown': 10, 'max_steps': 260, 'max_ms': 1_200_000},
                'face-arrangement': {'shown': None, 'max_steps': 60, 'max_ms': 1_500_000}}


class NodeKitBenchmark(BenchmarkBase):
    """One NodeKit task (a bundle of the task repository) as a behavior benchmark. The subject sees the
    screen on ``vision:image`` (and the acts on ``instruction``) and acts on the form its participants used
    (``ACT_FORMS``): a subject that does not declare that form is N/A before any compute."""

    trial_memory = 'none'       # one run of the task is one episode that starts empty

    def __init__(self, identifier: str, task: str, max_steps: Optional[int] = None, pacing: str = 'realtime',
                 dense_step_ms: Optional[float] = None, view: str = 'agent', acts_per_step: Optional[int] = None,
                 looks_kept: Optional[int] = 4, picture=AUTHOR_PICTURE, motor: Optional[dict] = AUTHOR_MOTOR,
                 trials: Optional[str] = None, max_ms: Optional[float] = None, record=None):
        """The defaults are the NodeKit author's settings (``AUTHOR_*``). ``trials``: None (his slice,
        ``AUTHOR_TASKS``) or ``'all'`` (the whole task). ``acts_per_step``: None, 100 in the agent view (his grid) and
        the harness's 20 in the participant view. ``picture``, ``motor`` (None: nkl's own) and ``looks_kept`` are
        the harness's (``NodeKitEnvironment``); ``max_steps`` and ``max_ms`` (the task clock's limit): None, his for
        the slice, 20000 and twice the participants' longest run for the whole task."""
        from brainscore.harnesses.nodekit import PACINGS
        if trials not in (None, 'all'):
            raise ValueError(f"trials is None (the NodeKit author's slice) or 'all' (the whole task), not {trials!r}.")
        own = AUTHOR_TASKS[task] if trials is None else {'shown': None, 'max_steps': 20000, 'max_ms': None}
        max_steps = own['max_steps'] if max_steps is None else max_steps
        if acts_per_step is None and view == 'agent':
            acts_per_step = AUTHOR_ACTS
        self.trials, self.picture, self.motor, self.shown = trials or 'slice', picture, motor, own['shown']
        self._max_ms = own['max_ms'] if max_ms is None else max_ms
        # The score names the author's settings when every setting is his.
        self._author = (trials is None and max_steps == own['max_steps'] and max_ms is None and pacing == 'realtime'
                        and view == 'agent' and acts_per_step == AUTHOR_ACTS and looks_kept == 4
                        and picture == AUTHOR_PICTURE and motor == AUTHOR_MOTOR and dense_step_ms is None)
        if pacing not in PACINGS:
            raise ValueError(f'pacing is one of {PACINGS} (nkl\'s Frozen, Realtime and Dense), not {pacing!r}.')
        if dense_step_ms is not None and pacing != 'dense':
            raise ValueError(f'dense_step_ms is the step of dense pacing; this benchmark runs {pacing!r} pacing.')
        self.task, self.max_steps, self.pacing, self.record = task, max_steps, pacing, record
        from brainscore.harnesses.nodekit import VIEWS
        if view not in VIEWS:
            raise ValueError(f'view is one of {VIEWS}, not {view!r}.')
        self.dense_step_ms, self.view, self.acts_per_step = dense_step_ms, view, acts_per_step
        self.looks_kept = looks_kept       # None: nkl's own (4); stamped when stated
        # Faces: the participants' ceiling (.174) is inside the random floor's interval, so the benchmark cannot
        # separate a subject from chance (owner, 2026-11-25: in the code, off the leaderboard).
        self.leaderboard_eligible = task != 'face-arrangement'
        # The screen is the stimulus: a subject that cannot take it never sees the task (N/A, Rule 5.1).
        # The frames since the last turn go on vision:video (a subject without it sees the last screen only),
        # nkl's briefing on instruction and the facts of each turn (time, hands, turns left) on observation.
        self.essential_input_channels = frozenset({'vision:image'})
        self.required_input_channels = self.essential_input_channels | {'vision:video', 'instruction', 'observation'}
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
        """The task clock's limit in simulated ms: the stated one, else twice the participants' longest run."""
        return self._max_ms if self._max_ms is not None else 2 * self._trials.attrs['longest_run_ms']

    @cached_property
    def compared(self) -> pd.DataFrame:
        """The participants' trials the subject's are compared with: with a slice, each participant's practice
        trials and first ``shown`` test trials (the same trials, in the same order, as the subject's run)."""
        if self.shown is None:
            return self.participants
        table = self.participants
        tests = table[table['phase'] == 'test'].sort_values('trial').groupby('participant').head(self.shown)
        return pd.concat([table[table['phase'] != 'test'], tests]).sort_index()

    @cached_property
    def ceiling(self) -> Score:
        return Score(METRICS[self.task][1](self.compared))

    def __call__(self, subject) -> Score:
        from brainscore.data.local import data_home
        from brainscore.harnesses.nodekit import bundle, play
        record = self.record or data_home() / 'records' / 'nodekit'
        result, environment, session = play(subject, bundle(self.task), form=ACT_FORMS[self.task], pacing=self.pacing,
                                            **({} if self.dense_step_ms is None else {'dense_step_ms': self.dense_step_ms}),
                                            view=self.view, acts_per_step=self.acts_per_step,
                                            looks_kept=self.looks_kept, picture=self.picture, motor=self.motor,
                                            shown=self.shown,
                                            instructions=INSTRUCTIONS.get(self.task, ''),
                                            max_steps=self.max_steps, max_ms=self.max_ms, record=record)
        return self.score_result(result, steps=environment.steps, invalid_steps=environment.invalid_steps,
                                 acts_per_step=environment.acts_per_step,
                                 run_ms=environment.t_ms, end_reason=session.source.end_reason,
                                 trace=str(environment.trace))

    def score_result(self, result, **run) -> Score:
        """The score of one run's result (None for a run that did not complete), or of repeated runs of one
        subject scored together: a list of results (their test trials together; the test-trial count is that of
        all the runs)."""
        if result is not None and result and isinstance(result[0], (list, type(None))):
            runs = result
            done = [records for records in runs if records is not None]
            result = [record for records in done for record in records] if done else None
            return self._scored(result, self._tests * len(runs), runs=len(runs), **run)
        return self._scored(result, self._tests, **run)

    @property
    def _tests(self) -> int:
        """The test trials of one run: the participants' count, or the ``shown`` slice of it."""
        count = int(test_rows(self.participants).groupby('participant').size().max())   # all of one run
        return count if self.shown is None else min(count, self.shown)

    def _scored(self, result, tests: int, **run) -> Score:
        if result is None:
            measured = {'raw': float('nan'), 'score_form': 'no trial record'}
            states = ['missing'] * tests
        else:
            own = test_rows(data.rows(result))
            measured = METRICS[self.task][0](own, self.compared)
            # A raw that is not a number (a constant arrangement has no rank order) is no measurement.
            state = 'valid' if np.isfinite(measured['raw']) else 'invalid'
            states = [state] * len(own)
            states += ['missing'] * max(0, tests - len(own))
            if 'response_milliseconds' in own:
                measured['subject_response_ms'] = [float(t) for t in own['response_milliseconds']]
        attrs = {key: value for key, value in measured.items() if key not in ('raw', 'score_form')}
        return benchmark_score(self, raw=measured['raw'], metric=measured['score_form'],
                               ceiling=self.ceiling, answers=pd.DataFrame({'response_state': states}),
                               answer_policy=ANSWER_POLICY, invalid_limit=0.1, max_ms=self.max_ms,
                               max_steps=self.max_steps, pacing=self.pacing, view=self.view,
                               **({} if self.looks_kept is None else {'looks_kept': self.looks_kept}),
                               trials=self.trials, test_trials=self._tests, picture=self.picture,
                               motor=self.motor if self.motor is not None else "nkl's own",
                               acts_per_step_limit=self.acts_per_step,
                               **({'settings': AUTHOR_SETTINGS} if self._author else {}),
                               **({} if self.dense_step_ms is None else {'dense_step_ms': self.dense_step_ms}),
                               **run, **attrs,
                               **({'invalid_step_share': run['invalid_steps'] / run['steps']}
                                  if run.get('steps') else {}))
