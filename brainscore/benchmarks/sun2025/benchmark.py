"""Sun 2025 multi-order motion: model reports of direction and speed against three participants.

Each trial shows one reconstructed 16-frame clip on ``vision:video`` with an
instruction; the subject answers on ``behavior:text`` with one JSON object
(``direction_deg``, ``speed_px_per_frame``) that this benchmark parses. The four
spatial flips of a scene and modulation are unflipped and averaged, then compared
with each participant's mean report by the endpoint error (displayed pixels per
frame; lower is closer). No ceiling is estimated; the reference is the three
participants, and the no-motion answer gives the floor.

This is an adaptation of the study, and the score says so in ``adaptation``.
"""
import hashlib
import json

import numpy as np
import pandas as pd

from brainscore_core.benchmarks import BenchmarkBase
from brainscore_core.contract import Subject, TaskContext
from brainscore_core.metrics import Score
from brainscore_core.responses import Text
from brainscore_core.streaming_helpers import stimulus_session
from brainscore_core.supported_data_standards.brainio.assemblies import BehavioralAssembly
from brainscore.benchmarks._score import benchmark_score, not_estimated

from .metric import endpoint_errors, parse_report, unflip_vector

IDENTIFIER = 'Sun2025-motion-vector-error'
STUDY_DOI = '10.1038/s42256-025-01068-w'
INSTRUCTION = (
    'Watch the short video and judge the motion of the texture or distorted region '
    'at the center, around the time the central dot flashes. Judge the moving region, '
    'not the flashing dot. Report the perceived direction and speed. '
    'Direction is in degrees counterclockwise from right: right=0, up=90, left=180, down=270. '
    'Speed is in pixels per video frame, measured in the ORIGINAL 600 by 600 pixel clip. '
    'It has 16 frames at 30 frames per second. A motion traversing the clip width in '
    'one second would be 20 pixels per frame. Report zero speed only for no perceived motion.'
    '\n\nReply with only a JSON object containing two numeric fields: '
    '"direction_deg" and "speed_px_per_frame". No explanation or Markdown.'
)
# What differs from the study (the owner decides whether the adaptation is accepted).
ADAPTATION = (
    'response: a numeric report of direction and speed replaces the interactive motion matching of the study',
    'stimuli: the clips are reconstructed from the published frames (16 frames at 30 Hz, 600 px circular '
    'aperture, OpenCV raster), not the PsychoPy display',
    'display: no monitor calibration, visual angle, feedback, blanks or matching loop',
    'reference: each participant\'s mean of the four unflipped reports; no human reliability ceiling',
)
ANSWER_POLICY = ('invalid (the reply is not one JSON object, plain or as the whole reply inside one '
                 'json code fence, with exactly the two numbers in range: direction in '
                 '[0, 360), speed in [0, 1e6]) or missing: '
                 'the trial is kept with its raw text and counted; any such trial makes the score NaN, '
                 'so a hard group is never dropped from the mean')
_FULL_DESIGN = {(s, m, f) for s in range(1, 41) for m in range(1, 8) for f in range(1, 5)}


def _check(stimuli, reference) -> None:
    """The design and the reference agree before any subject runs (as in pull request #63)."""
    if stimuli.empty or stimuli.stimulus_id.duplicated().any():
        raise ValueError('Use nonempty stimuli with distinct stimulus ids')
    for base_id, rows in stimuli.groupby('base_id'):
        if sorted(rows.flip) != [1, 2, 3, 4] or rows.scene.nunique() != 1 or rows.modulation.nunique() != 1:
            raise ValueError(f'{base_id}: each scene and modulation group needs one scene, one modulation '
                             'and the four flips')
    if stimuli.drop_duplicates('base_id').duplicated(['scene', 'modulation']).any():
        raise ValueError('Each scene and modulation pair must be exactly one group')
    if set(reference.dims) != {'presentation', 'participant', 'axis'}:
        raise ValueError('The reference has the dimensions presentation, participant and axis')
    ids = [str(i) for i in reference['stimulus_id'].values]
    if len(ids) != len(set(ids)) or set(ids) != set(stimuli.base_id):
        raise ValueError('The reference ids must match the scene and modulation groups of the stimuli')
    if (list(reference['axis'].values) != ['x', 'y'] or list(reference['participant'].values) != [1, 2, 3]
            or not np.isfinite(reference.values).all()):
        raise ValueError('Use finite x/y reference vectors of the three participants 1, 2 and 3')
    groups = stimuli.drop_duplicates('base_id').set_index('base_id')
    for coordinate in ('scene', 'modulation'):
        if list(reference[coordinate].values) != list(groups.loc[ids, coordinate]):
            raise ValueError(f'The reference {coordinate} coordinates do not match the stimuli')


def _full_design(stimuli) -> bool:
    return (len(stimuli) == len(_FULL_DESIGN)
            and set(zip(stimuli.scene, stimuli.modulation, stimuli.flip)) == _FULL_DESIGN)


def _subset_identifier(stimuli, reference) -> str:
    """A hash of the content: the same subset in any row or dimension order has one identifier."""
    human = reference.transpose('presentation', 'participant', 'axis')
    ids = [str(i) for i in human['stimulus_id'].values]
    order = np.argsort(ids)
    content = {
        'trials': sorted((str(r.stimulus_id), str(r.base_id), int(r.scene), int(r.modulation), int(r.flip))
                         for r in stimuli.itertuples(index=False)),
        'reference_ids': [ids[i] for i in order],
        'reference': human.values[order].tolist(),
        'stimulus_set': getattr(stimuli, 'identifier', None),
        'reference_build': reference.attrs.get('identifier'),
    }
    return f'{IDENTIFIER}-subset-{hashlib.sha256(json.dumps(content).encode()).hexdigest()[:16]}'


class Sun2025MotionBenchmark(BenchmarkBase):
    """Score direction and speed reports of a subject on the Sun 2025 clips.

    The complete design (40 scenes x 7 modulations x 4 flips, a complete reference) has
    the identifier ``Sun2025-motion-vector-error``. Any other selection is a subset: its
    identifier ends in a hash of its stimulus ids and reference values, and the score
    says ``full_study=False``.
    """
    trial_memory = 'none'
    required_input_channels = frozenset({'vision:video', 'instruction'})
    requested_output_channels = frozenset({'behavior:text'})
    leaderboard_eligible = False

    def __init__(self, stimuli, reference: BehavioralAssembly) -> None:
        _check(stimuli, reference)
        self.full_study = _full_design(stimuli) and reference.attrs.get('complete') is True
        identifier = IDENTIFIER if self.full_study else _subset_identifier(stimuli, reference)
        # A fixed, seeded trial order; each clip is its own trial.
        self.stimulus_set = stimuli.sample(frac=1, random_state=0).reset_index(drop=True)
        self.stimulus_set.identifier = f'{getattr(stimuli, "identifier", "sun2025")}-order-0'
        self.stimulus_set.stimulus_paths = dict(zip(stimuli.stimulus_id, stimuli.video_path))
        self.task = TaskContext(instruction=INSTRUCTION, response=Text(max_tokens=64, temperature=0),
                                trial_memory='none')
        self.reference = reference
        super().__init__(identifier=identifier, version=1, parent='behavioral', ceiling=not_estimated(),
                         bibtex='@article{sun2025motion, title={Machine learning modelling for '
                         'multi-order human visual motion processing}, author={Sun and others}, '
                         'journal={Nature Machine Intelligence}, year={2025}, '
                         f'doi={{{STUDY_DOI}}}}}')

    def __call__(self, subject: Subject) -> Score:
        session = stimulus_session(self.stimulus_set, sorted(self.requested_output_channels), self.task)
        subject.participate(session)
        answers = session.collect('behavior:text')
        reports = self._reports(answers)
        valid = reports['response_state'].values == 'valid'
        human = self.reference.transpose('presentation', 'participant', 'axis').values
        floor = float(np.linalg.norm(human, axis=-1).mean())
        raw, by_participant = float('nan'), None
        if valid.all():
            vectors = {}
            for sid, text, flip in zip(reports['stimulus_id'].values, reports.values, reports['flip'].values):
                report = parse_report(text)[0]
                vectors[str(sid)] = unflip_vector(report['direction_deg'], report['speed_px_per_frame'], int(flip))
            groups = self.stimulus_set.groupby('base_id').stimulus_id.apply(list).to_dict()
            errors = pd.DataFrame(endpoint_errors(vectors, groups, self.reference),
                                  columns=['group', 'participant', 'error'])
            # The mean over groups and participants, in the order of the original implementation.
            raw = float(errors.error.mean())
            by_participant = {int(p): float(e) for p, e in errors.groupby('participant').error.mean().items()}
        return benchmark_score(
            self, raw=raw, metric='mean_vector_endpoint_error', answers=reports,
            answer_policy=ANSWER_POLICY, invalid_limit=0.0,
            units='displayed pixels per frame', higher_is_better=False,
            reference='three study participants (no ceiling estimated)',
            floor=floor, floor_method='no-motion answer (speed 0) for every trial',
            errors_by_participant=by_participant, reports=reports,
            full_study=self.full_study,
            subset=None if self.full_study else f'{len(self.stimulus_set)} of {len(_FULL_DESIGN)} trials',
            adaptation=ADAPTATION, study_doi=STUDY_DOI,
            reference_identifier=self.reference.attrs.get('identifier'),
            stimulus_set_identifier=self.stimulus_set.identifier)

    def _reports(self, answers) -> BehavioralAssembly:
        """Every trial's raw text with its state: core's missing stays; an unparsable text is invalid."""
        rows = dict(zip(map(str, answers['stimulus_id'].values),
                        zip(answers['raw_answer'].values, answers['response_state'].values,
                            answers['response_reason'].values)))
        texts, states, reasons = [], [], []
        for sid in self.stimulus_set.stimulus_id:
            text, state, reason = rows.get(sid, (None, 'missing', 'no answer'))
            if state == 'valid':
                error = parse_report(text)[1]
                state, reason = ('invalid', error) if error else ('valid', '')
            texts.append(text)
            states.append(state)
            reasons.append(reason)
        table = self.stimulus_set
        return BehavioralAssembly(
            np.array(texts, dtype=object), dims=['presentation'],
            coords={'stimulus_id': ('presentation', list(table.stimulus_id)),
                    'base_id': ('presentation', list(table.base_id)),
                    'scene': ('presentation', list(table.scene)),
                    'modulation': ('presentation', list(table.modulation)),
                    'flip': ('presentation', list(table.flip)),
                    'response_state': ('presentation', states),
                    'response_reason': ('presentation', reasons)})


def load_benchmark(*, root=None, scenes=None) -> Sun2025MotionBenchmark:
    """The registered benchmark from the prepared data; ``scenes`` selects a named subset."""
    from brainscore import load_dataset, load_stimulus_set
    from brainscore.data.sun2025 import IDENTIFIER as DATA
    stimuli = load_stimulus_set(DATA, root=root)
    reference = load_dataset(DATA, root=root)
    if scenes is None and not (_full_design(stimuli) and reference.attrs.get('complete')):
        raise ValueError('This build holds only some scenes; prepare all 40 scenes for the registered '
                         'benchmark, or pass scenes=[...] for a named subset')
    if scenes is not None:
        stimuli = stimuli[stimuli.scene.isin(scenes)].reset_index(drop=True)
        if stimuli.empty:
            raise ValueError(f'No stimuli for scenes {scenes} in this build')
        reference = reference.isel(presentation=np.isin(reference['scene'].values, list(scenes)))
        reference.attrs['complete'] = False
    return Sun2025MotionBenchmark(stimuli, reference)
