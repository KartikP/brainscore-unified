"""CNeuroMod-THINGS fMRI benchmarks (St-Laurent et al. 2026, Scientific Data 13:141).

Each participant saw 4320 THINGS images (720 categories, 6 exemplars each), each image three
times, in 36 sessions of a continuous recognition task; the images spanned 10 degrees of visual
angle. The data are the GLMsingle trial responses (FITHRF_GLMDENOISE_RR, z-scored per voxel),
packaged locally with ``package_data('cneuromod-things')``.

One benchmark per region, the participants pooled: ``StLaurent2026-fMRI-things-<region>`` asks
``neural:<region>`` on ``vision:image``. The region's voxels come from the participant's own masks
(pRF: V1, V2, V3, hV4 as ``V4``; fLoc localizer: FFA, OFA, pSTS, PPA, OPA, MPA, EBA), measured
in other scans than THINGS. Participants sub-01, 02 and 03 (sub-04 is restricted; sub-06 has no
own masks).

For each participant:

1. Images: those with three answered trials (``things.answered_viewing``), as the paper's noise
   ceiling counts them. Split by category, one fixed list (``test_categories.txt``: 144 of the
   720 categories, 20 percent, drawn once): no test category is in the fit.
2. Target: per image and voxel, the mean of the three answered viewings; each viewing z-scored
   per voxel with the mean and standard deviation over the training images.
3. Voxels: inside the mask, those with a noise ceiling (``noise_ceiling``) of at least 10 percent
   on the training images. A participant with fewer than 50 such voxels in the region does not
   count there; the score states it.
4. Fit: the model's features for the region (one ``participate`` session over every image),
   ridge with an intercept in the dual form (``NormalEquations``), the penalty chosen on held-out
   training categories (5 folds), one fit on the training images; Pearson r per voxel on the test
   images.
5. Score: the median over the selected voxels of r / c, c = sqrt(noise ceiling / 100) of the
   voxel on the test images (``voxel_score``).

The benchmark value is the mean over the participants that count; each participant's value,
raw median r, median c and counts are on the score.
"""
from functools import partial
from pathlib import Path

import numpy as np

from brainscore_core.benchmarks import BenchmarkBase
from brainscore_core.execution_plan import ExecutionPlan
from brainscore_core.metrics import Score
from brainscore import load_dataset, load_stimulus_set
from brainscore.benchmarks._score import benchmark_score, noise_ceiling, not_estimated, voxel_score
from brainscore.data.cneuromod.data_packaging.things import answered_viewing
from brainscore.metrics.encoding import NormalEquations, cross_validated_r, grid_search, group_kfold, ridge

BIBTEX = """@article{stlaurent2026things,
  title={CNeuroMod-THINGS, a densely-sampled fMRI dataset for visual neuroscience},
  author={St-Laurent, Marie and Pinsard, Basile and Contier, Oliver and DuPre, Elizabeth and
          Seeliger, Katja and Borghesani, Valentina and Boyle, Julie A. and Bellec, Lune and
          Hebart, Martin N.},
  journal={Scientific Data},
  volume={13},
  pages={141},
  year={2026},
}"""

DATA = 'cneuromod-things'
REGION_PARTICIPANTS = ('sub-01', 'sub-02', 'sub-03')
TEST_CATEGORIES = frozenset((Path(__file__).parent / 'test_categories.txt').read_text().split())
PACKAGED_REGION = {'V4': 'hV4'}     # the request names the macaque area; the mask is its human homolog
VIEWINGS = 3
NC_THRESHOLD = 10.0                 # percent, on the training images (ceiling r 0.32)
MIN_VOXELS = 50
ALPHA_GRID = tuple(float(10.0 ** e) for e in range(-2, 9))
STATED = {
    'responses': 'GLMsingle FITHRF_GLMDENOISE_RR trial responses, z-scored per voxel (things.glmsingle)',
    'target': 'mean of the three answered viewings of each image; each viewing z-scored per voxel with the mean '
              'and standard deviation over the training images',
    'voxel_selection': f'noise ceiling on the training images at least {NC_THRESHOLD:g} percent; a participant '
                       f'with fewer than {MIN_VOXELS} such voxels does not count',
    'split': f'by category: {len(TEST_CATEGORIES)} test categories (test_categories.txt), the other categories '
             'for the fit; penalty on held-out training categories (5 folds)',
    'ceiling_method': 'per voxel c = sqrt(NC / 100), NC of the three viewings on the test images '
                      '(St-Laurent et al. 2026, p.5-6); score = median over voxels of r / c',
    'visual_degrees': 10,
}


def category(stimulus_ids) -> np.ndarray:
    """The THINGS category of each image: the file stem without its exemplar (``aardvark_01b`` -> ``aardvark``)."""
    return np.array([str(stimulus_id).rsplit('_', 1)[0] for stimulus_id in stimulus_ids])


class StLaurent2026Things(BenchmarkBase):
    """Predict the trial responses of one region to THINGS images (module docstring).

    Args:
        region: the requested region (``neural:<region>``).
        participants: the participants scored.
        identifier: the registry key.
        root: the packaged folder (default: the place ``package_data`` writes).
    """

    trial_memory = 'none'
    required_input_channels = frozenset({'vision:image'})

    def __init__(self, region: str, participants=REGION_PARTICIPANTS, *, identifier: str, root=None):
        self._region = region
        self.requested_output_channels = frozenset({f'neural:{region}'})
        self._participants = tuple(participants)
        self._root = root
        self._stimulus_set = None
        super().__init__(identifier=identifier, version=1, parent='neural', ceiling=not_estimated(),
                         bibtex=BIBTEX)

    @property
    def stimulus_set(self):
        if self._stimulus_set is None:
            self._stimulus_set = load_stimulus_set(DATA, root=self._root)
        return self._stimulus_set

    @property
    def execution_plan(self) -> ExecutionPlan:
        stimuli = self.stimulus_set
        return ExecutionPlan(n_extraction_presentations=len(stimuli), metric_category='ridge',
                             runs_ceiling_metric=False, probe_stimuli=stimuli.iloc[:1])

    def _target(self, participant: str) -> dict:
        """The images, target, selected voxels and ceilings of one participant (steps 1-3)."""
        packaged = PACKAGED_REGION.get(self._region, self._region)
        trials = load_dataset(DATA, participant, packaged, root=self._root)
        image = trials['stimulus_id'].values.astype(str)
        viewing = answered_viewing(image, trials['response'].values)
        labels, counts = np.unique(image[viewing > 0], return_counts=True)
        images = labels[counts == VIEWINGS]                                   # sorted
        test = np.isin(category(images), list(TEST_CATEGORIES))
        responses = trials.values
        # An image outside a set is given viewing 0 (as a trial without response): noise_ceiling leaves it out.
        on_train = viewing * np.isin(image, images[~test])
        on_test = viewing * np.isin(image, images[test])
        selected = noise_ceiling(responses, image, on_train, VIEWINGS)[0] >= NC_THRESHOLD
        facts = {'atlas': str(trials['atlas'].values[0]), 'n_voxels_in_mask': int(responses.shape[1]),
                 'n_voxels_selected': int(selected.sum()), 'n_images': int(len(images)),
                 'n_images_fitted': int((~test).sum()), 'n_images_scored': int(test.sum())}
        if selected.sum() < MIN_VOXELS:
            return {'facts': facts, 'counts': False}
        ceiling = np.sqrt(noise_ceiling(responses[:, selected], image, on_test, VIEWINGS)[0] / 100)
        cube = np.stack([responses[(viewing == place) & np.isin(image, images)][:, selected][
            np.argsort(image[(viewing == place) & np.isin(image, images)], kind='stable')]
            for place in range(1, VIEWINGS + 1)], axis=1)                    # images x viewings x voxels
        cube = (cube - cube[~test].mean(axis=0)) / cube[~test].std(axis=0)
        return {'facts': facts, 'counts': True, 'images': images, 'test': test,
                'target': cube.mean(axis=1), 'ceiling': ceiling}

    def _features(self, subject) -> tuple:
        """``(features, stimulus ids)``: the model's response for the region, one session over every image."""
        from brainscore_core.streaming_helpers import neural_response
        response = neural_response(subject, self.stimulus_set, [f'neural:{self._region}'])
        if 'time_bin' in response.dims:
            response = response.mean('time_bin')
        response = response.transpose('presentation', 'neuroid')
        return np.asarray(response.values), np.asarray(response['stimulus_id'].values).astype(str)

    def _fit(self, features, part) -> np.ndarray:
        """Held-out r per selected voxel: one ridge on the training images, r on the test images (step 4)."""
        groups = category(part['images'])
        fit = ridge(grid_search([(alpha,) for alpha in ALPHA_GRID], partial(group_kfold, n_splits=5)),
                    NormalEquations(shape_aware=True))
        r, _ = cross_validated_r(features, part['target'], [(~part['test'], part['test'])], fit, groups=groups)
        return r

    def __call__(self, subject) -> Score:
        parts = {participant: self._target(participant) for participant in self._participants}  # data before the run
        features, ids = self._features(subject)
        rows = {stimulus_id: row for row, stimulus_id in enumerate(ids)}
        scores, by_participant = {}, {}
        for participant, part in parts.items():
            by_participant[participant] = dict(part['facts'])
            if not part['counts']:
                by_participant[participant]['excluded'] = (f'fewer than {MIN_VOXELS} voxels with a noise ceiling '
                                                           f'of at least {NC_THRESHOLD:g} percent')
                continue
            missing = [image for image in part['images'] if image not in rows]
            if missing:
                raise ValueError(f'the response has no row for {len(missing)} images of {participant} '
                                 f'(for example {missing[0]}); a response answers every image shown.')
            score = voxel_score(self, self._fit(features[[rows[image] for image in part['images']]], part),
                                part['ceiling'])
            scores[participant] = score
            by_participant[participant].update(
                value=float(score), raw=float(score.attrs['raw']), median_ceiling=float(score.attrs['ceiling']),
                n_voxels_scored=score.attrs['n_voxels_scored'])
        if not scores:
            raise ValueError(f'no participant has {MIN_VOXELS} voxels in {self._region} with a noise ceiling of at '
                             f'least {NC_THRESHOLD:g} percent: nothing to score ({by_participant}).')
        result = benchmark_score(
            self, value=float(np.mean([float(s) for s in scores.values()])),
            raw=float(np.mean([float(s.attrs['raw']) for s in scores.values()])),
            ceiling=Score(float(np.mean([float(s.attrs['ceiling']) for s in scores.values()]))),
            metric='median_per_voxel_pearson_r', ceiling_aggregation='per_voxel',
            participants=sorted(scores), per_participant=by_participant, region=self._region,
            **STATED)
        if self._region in PACKAGED_REGION:
            result.attrs['region_note'] = f'{self._region} = {PACKAGED_REGION[self._region]}, the human homolog (pRF mask)'
        return result
