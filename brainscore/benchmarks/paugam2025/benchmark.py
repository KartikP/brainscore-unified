"""CNeuroMod mario fMRI benchmark (data: Paugam et al. 2025, bioRxiv 10.1101/2025.11.28.691119).

Five participants played Super Mario Bros. (NES; 22 levels) in the scanner and heard its sound, about
15 hours each (TR 1.49 s). The data are the ``cneuromod2026`` time series (1134 parcels of the
participant's own atlas, MNI space), made locally from the public fMRIPrep outputs
(``package_data('cneuromod-mario-timeseries')``), and the media of each TR, made locally from the
participant's own play videos (``python -m brainscore.data.cneuromod.data_packaging.mario``).

``Paugam2025-fMRI-mario-cortex`` asks ``neural:cortex``. Passive form: the subject sees the
participant's own play, it does not play. For each participant:

1. Input: one row per TR that shows a repetition (a play of a level): the clip of the TR's frames on
   ``vision:video`` (60 frames per second, up to 1.49 s), its middle frame on ``vision:image`` and its
   sound on ``audio`` (16 kHz, mono). A subject gets the channels it declares (``audio`` is optional). A
   TR without a repetition (between plays, the questionnaire) gets no row; its features are 0.
2. Features: the response to ``neural:cortex``, one band per layer (at most 12); each layer is first
   reduced to at most 1000 units by one fixed Gaussian projection (seed 0, no fit, no data), then to at
   most 100 components on the training TRs (``prepare_bands``); each band is convolved per run with the
   canonical SPM double-gamma HRF (``brainscore.temporal.hrf_convolve``, at the TR rate).
3. Split by run, one fixed list (``test_runs.txt``: 88 of 483 runs, about 20 percent of each
   participant's runs, drawn once with seed 2026 within groups of the run's main level). The paper
   used an 80/10/10 split of the plays, balanced by level.
4. Target: each parcel z-scored within each run.
5. Fit: banded ridge on the training runs (one penalty per band, one penalty tuple for all parcels,
   chosen on one held-out fold of training runs); Pearson r per parcel on the TRs of the test runs.
6. Score: the median r over parcels, raw (no ceiling: no play is repeated frame for frame); the
   benchmark value is the mean over the participants. Each participant's median r and per-parcel r
   are on the score.

``Paugam2025-behavior-mario-buttons`` asks ``behavior:choice``: which buttons the participant held in
each TR, on the same input rows. The answer is one label per TR: the buttons held for at least half
of the TR, joined by ``+`` in the order A, B, LEFT, RIGHT, UP, DOWN (``none``: no button); a label
seen in fewer than 1 percent of the training TRs of all participants is ``other``. The choices are in the
order of their training frequency, the most frequent first. Per participant,
one session over the TRs of the test runs (the same ``test_runs.txt``), with every 8th training TR as
the fitting stimuli. Score: the accuracy on the test TRs, raw, mean over the participants; chance is
stated (the rate on the test TRs of the most frequent training label, which chance-baseline gets; the rate of
the most frequent label of the test TRs; the expected accuracy of a guess drawn from the training label rates;
and of a uniform guess).
"""
from functools import lru_cache, partial
from pathlib import Path

import numpy as np

from brainscore_core.benchmarks import BenchmarkBase
from brainscore_core.contract import TaskContext
from brainscore_core.responses import Choice
from brainscore_core.execution_plan import ExecutionPlan
from brainscore_core.metrics import Score
from brainscore import load_dataset, load_stimulus_set
from brainscore.benchmarks._score import benchmark_score, not_estimated
from brainscore.metrics.encoding import (
    NormalEquations, cross_validated_r, grid_search, group_kfold, penalty_grid, prepare_bands, ridge, summarize)
from brainscore.temporal import hrf_convolve

BIBTEX = """@article{paugam2025brittle,
  title={Training neural networks from scratch in a videogame leads to brittle brain encoding},
  author={Paugam, Fran{\\c{c}}ois and Pinsard, Basile and St-Laurent, Marie and Lajoie, Guillaume and Bellec, Lune},
  journal={bioRxiv},
  doi={10.1101/2025.11.28.691119},
  year={2025},
}"""

DATA = 'cneuromod-mario'
PARTICIPANTS = ('sub-01', 'sub-02', 'sub-03', 'sub-05', 'sub-06')
TEST_RUNS = frozenset((Path(__file__).parent / 'test_runs.txt').read_text().split())
TR_SECONDS = 1.49
ALPHA_GRID = tuple(float(10.0 ** e) for e in range(-2, 9))
MAX_BANDS = 12
LAYER_UNITS = 1000          # the fixed projection of each layer before the fit
BAND_COMPONENTS = 100
BANDED_CANDIDATES = 60
CHUNK_ROWS = 2000           # TRs per session: a layer of a chunk is reduced before the next chunk runs
STATED = {
    'responses': 'cneuromod2026 time series (1134 parcels, own atlas, MNI space), made from the public '
                 'fMRIPrep outputs with cneuromod_extract_tseries',
    'input': 'per TR with a play: its clip (vision:video, 60 fps), its middle frame (vision:image), its sound '
             '(audio, 16 kHz mono); the participant\'s own play; a TR without a play has features 0',
    'features': f'one band per layer (at most {MAX_BANDS}); each layer projected to at most {LAYER_UNITS} units by '
                f'a fixed Gaussian matrix (seed 0), then at most {BAND_COMPONENTS} components on the training '
                'TRs; SPM double-gamma HRF per run',
    'target': 'each parcel z-scored within each run',
    'split': f'by run: {len(TEST_RUNS)} test runs (test_runs.txt, about 20 percent, by main level, seed 2026); '
             'penalty on one held-out fold of training runs. The paper used 80/10/10 of the plays by level',
    'ceiling_method': 'none: the score is the raw median per-parcel Pearson r',
}


BUTTONS = ('A', 'B', 'LEFT', 'RIGHT', 'UP', 'DOWN')
HELD_SECONDS = TR_SECONDS / 2
RARE_LABEL = 0.01
FIT_EVERY = 8
ANSWER_POLICY = 'invalid: wrong, counted; missing: wrong, counted'
INSTRUCTION = ('This is one moment (1.49 s) of a person playing Super Mario Bros. Which buttons does the player '
               'hold for most of this moment?')


def run_label(participant: str, session: int, run: int) -> str:
    return f'{participant}_ses-{int(session):03d}_run-{int(run)}'


def project(values: np.ndarray) -> np.ndarray:
    """``values`` (rows x units of one layer) times a fixed Gaussian matrix to ``LAYER_UNITS`` columns; a layer
    with ``LAYER_UNITS`` units or fewer is kept. The matrix depends only on the width and the seed."""
    if values.shape[1] <= LAYER_UNITS:
        return values.astype(np.float32)
    return values.astype(np.float32) @ _matrix(values.shape[1])


@lru_cache(maxsize=MAX_BANDS)
def _matrix(width: int) -> np.ndarray:
    matrix = np.random.default_rng(0).standard_normal((width, LAYER_UNITS), dtype=np.float32)
    return matrix / np.float32(np.sqrt(LAYER_UNITS))


class Paugam2025Mario(BenchmarkBase):
    """Predict the parcel time series from the subject's response to the participant's play (module docstring)."""

    trial_memory = 'none'
    required_input_channels = frozenset({'vision:video', 'vision:image', 'audio'})
    requested_output_channels = frozenset({'neural:cortex'})

    def __init__(self, participants=PARTICIPANTS, *, identifier: str, root=None):
        self._participants = tuple(participants)
        self._root = root
        super().__init__(identifier=identifier, version=1, parent='neural', ceiling=not_estimated(),
                         bibtex=BIBTEX)

    def _media(self, participant):
        return load_stimulus_set(DATA, participant, root=self._root)

    @property
    def execution_plan(self) -> ExecutionPlan:
        media = self._media(self._participants[0])
        return ExecutionPlan(n_extraction_presentations=CHUNK_ROWS, metric_category='ridge',
                             runs_ceiling_metric=False, probe_stimuli=media.iloc[:1])

    def _bands(self, subject, media, n_trs: int, rows: np.ndarray) -> dict:
        """``{layer: (TRs x units)}``: the response to ``neural:cortex`` of each media row, put at its TR row
        ``rows``, each layer projected (``project``); 0 at a TR without a row. One session per chunk."""
        from brainscore_core.streaming_helpers import neural_response
        bands = {}
        for start in range(0, len(media), CHUNK_ROWS):
            chunk = media.iloc[start:start + CHUNK_ROWS]
            chunk.identifier = f'{media.identifier}-{start}'
            response = neural_response(subject, chunk, ['neural:cortex'])
            if 'time_bin' in response.dims:
                response = response.mean('time_bin')
            response = response.transpose('presentation', 'neuroid')
            if list(np.asarray(response['stimulus_id'].values).astype(str)) != list(chunk['stimulus_id']):
                raise ValueError('the response to neural:cortex does not answer the rows in the order given; '
                                 'a response answers each row once, in order.')
            try:
                layers = np.asarray(response['layer'].values).astype(str)
            except KeyError:
                layers = np.full(response.sizes['neuroid'], 'response')
            if len(set(layers)) > MAX_BANDS:
                raise ValueError(f'the response for neural:cortex comes from {len(set(layers))} layers; the '
                                 f'benchmark fits at most {MAX_BANDS} bands, one per layer. Map cortex to at most '
                                 f'{MAX_BANDS} layers (for example every k-th block).')
            values = np.asarray(response.values)
            for layer in dict.fromkeys(layers):
                reduced = project(values[:, layers == layer])
                band = bands.setdefault(layer, np.zeros((n_trs, reduced.shape[1]), np.float32))
                band[rows[start:start + len(chunk)]] = reduced
        return bands

    def _participant(self, subject, participant) -> dict:
        series = load_dataset(DATA, participant, root=self._root)
        media = self._media(participant)
        session, run = series['session'].values.astype(int), series['run'].values.astype(int)
        labels = np.array([run_label(participant, s, r) for s, r in zip(session, run)])
        tr_row = {(s, r, t): i for i, (s, r, t) in enumerate(zip(session, run, series['tr'].values.astype(int)))}
        rows = np.array([tr_row[(s, r, t)] for s, r, t in zip(media['session'], media['run'], media['tr'])])
        test = np.isin(labels, list(TEST_RUNS))
        train = ~test
        if not test.any() or not train.any():
            raise ValueError(f'{participant}: {int(test.sum())} test TRs and {int(train.sum())} training TRs; '
                             'the packaged runs do not match test_runs.txt.')
        Y = np.asarray(series.values, dtype=np.float64)
        for label in np.unique(labels):                       # z-score each parcel within each run
            inside = labels == label
            Y[inside] = (Y[inside] - Y[inside].mean(0)) / (Y[inside].std(0) + 1e-8)
        raw = self._bands(subject, media, len(labels), rows)
        shown = np.zeros(len(labels), bool)
        shown[rows] = True
        prepared, _ = prepare_bands({layer: (band, None) for layer, band in raw.items()}, train & shown,
                                    BAND_COMPONENTS)
        bands = []
        for band in prepared.values():
            band = np.where(shown[:, None], band, 0.0)            # no play, no input
            for label in np.unique(labels):
                inside = labels == label
                band[inside] = hrf_convolve(band[inside], 1 / TR_SECONDS)
            bands.append(band)
        fit = ridge(grid_search(penalty_grid(ALPHA_GRID, len(bands), BANDED_CANDIDATES),
                                partial(group_kfold, n_splits=5, n_used=1)), NormalEquations())
        r, _ = cross_validated_r(bands, Y, [(train, test)], fit, groups=labels)
        finite, median_r, mean_r = summarize(r)
        return {'median_r': median_r, 'mean_r': mean_r, 'per_parcel_r': r.tolist(),
                'parcel_names': series['parcel_name'].values.astype(str).tolist(),
                'n_trs_fitted': int(train.sum()), 'n_trs_scored': int(test.sum()), 'n_trs_with_play': int(shown.sum()),
                'n_test_runs': int(len(np.unique(labels[test]))), 'n_runs': int(len(np.unique(labels))),
                'components_per_band': {layer: int(band.shape[1]) for layer, band in prepared.items()}}

    def __call__(self, subject) -> Score:
        by_participant = {participant: self._participant(subject, participant) for participant in self._participants}
        medians = [part['median_r'] for part in by_participant.values()]
        return benchmark_score(self, raw=float(np.mean(medians)), metric='median_per_parcel_pearson_r',
                               participants=list(by_participant), per_participant=by_participant,
                               band_cap=MAX_BANDS, **STATED)


def button_labels(series) -> np.ndarray:
    """Per TR: the buttons held for at least half of the TR, joined by ``+`` (``none``: no button)."""
    held = np.stack([series[f'button_{button}'].values >= HELD_SECONDS for button in BUTTONS], axis=1)
    return np.array(['+'.join(b for b, on in zip(BUTTONS, row) if on) or 'none' for row in held])


class Paugam2025MarioButtons(BenchmarkBase):
    """Name the buttons the participant held in each TR of the play (module docstring)."""

    trial_memory = 'none'
    required_input_channels = frozenset({'vision:video', 'vision:image', 'audio'})
    requested_output_channels = frozenset({'behavior:choice'})

    def __init__(self, participants=PARTICIPANTS, *, identifier: str, root=None):
        self._participants = tuple(participants)
        self._root = root
        super().__init__(identifier=identifier, version=1, parent='behavioral', ceiling=not_estimated(),
                         bibtex=BIBTEX)

    @property
    def execution_plan(self) -> ExecutionPlan:
        media = load_stimulus_set(DATA, self._participants[0], root=self._root)
        return ExecutionPlan(n_extraction_presentations=len(media), metric_category='behavioral',
                             runs_ceiling_metric=False, probe_stimuli=media.iloc[:1])

    def _rows(self, participant):
        """``(media rows with label and test columns)``: the label of each media row's TR."""
        series = load_dataset(DATA, participant, root=self._root)
        media = load_stimulus_set(DATA, participant, root=self._root)
        tr_row = {(s, r, t): i for i, (s, r, t) in enumerate(zip(
            series['session'].values.astype(int), series['run'].values.astype(int), series['tr'].values.astype(int)))}
        rows = [tr_row[(s, r, t)] for s, r, t in zip(media['session'], media['run'], media['tr'])]
        media['label'] = button_labels(series)[rows]
        media['test'] = [run_label(participant, s, r) in TEST_RUNS for s, r in zip(media['session'], media['run'])]
        return media

    def __call__(self, subject) -> Score:
        tables = {participant: self._rows(participant) for participant in self._participants}
        training = np.concatenate([table.loc[~table['test'], 'label'].values for table in tables.values()])
        labels, counts = np.unique(training, return_counts=True)
        # most frequent first: an answer that ties every label (chance-baseline, core's argmax) is the most
        # frequent label, the stated floor chance_most_frequent; chance_uniform is a uniform guess
        order = np.argsort(-counts, kind='stable')
        kept = [str(label) for label, count in zip(labels[order], counts[order]) if count >= RARE_LABEL * len(training)]
        choices = kept + ['other']
        rates = {label: float(np.mean(training == label)) for label in kept}
        rates['other'] = 1.0 - sum(rates.values())
        by_participant, accuracies, answers = {}, [], {}
        for participant, table in tables.items():
            table['label'] = np.where(np.isin(table['label'], kept), table['label'], 'other')
            fitting = table[~table['test']].iloc[::FIT_EVERY].reset_index(drop=True)
            test = table[table['test']].reset_index(drop=True)
            fitting = type(table)(fitting)
            fitting.identifier = f'{table.identifier}-fit{FIT_EVERY}'
            fitting.stimulus_paths = dict(zip(fitting['stimulus_id'], fitting['image_path']))
            stimuli = type(table)(test.drop(columns=['label']))
            stimuli.identifier = f'{table.identifier}-test'
            stimuli.stimulus_paths = dict(zip(stimuli['stimulus_id'], stimuli['image_path']))
            from brainscore_core.streaming_helpers import stimulus_session
            session = stimulus_session(stimuli, ['behavior:choice'], TaskContext(
                fitting_stimuli=fitting, instruction=INSTRUCTION, response=Choice(choices)))
            subject.participate(session)
            predictions = answers[participant] = session.collect('behavior:choice')
            options = list(predictions['choice'].values)
            valid = predictions['response_state'].values == 'valid'
            picked = np.nan_to_num(predictions.values, nan=-1).argmax(axis=1)
            truth = dict(zip(test['stimulus_id'], test['label']))
            answered = [options[i] if ok else None for i, ok in zip(picked, valid)]
            correct = np.array([a == truth[s] for a, s in zip(answered, predictions['stimulus_id'].values)])
            test_labels = test['label'].values
            accuracies.append(float(correct.mean()))
            by_participant[participant] = {
                'accuracy': accuracies[-1], 'n_test_trs': int(len(test)), 'n_fitting_trs': int(len(fitting)),
                'chance_most_frequent': float(max(np.mean(test_labels == label) for label in choices)),
                'chance_training_majority': float(np.mean(test_labels == choices[0])),
                'chance_by_rates': float(sum(rates[label] * np.mean(test_labels == label) for label in choices)),
                'n_invalid': int((~valid).sum())}
        return benchmark_score(
            self, raw=float(np.mean(accuracies)), metric='accuracy', answers=answers, answer_policy=ANSWER_POLICY,
            participants=list(by_participant),
            per_participant=by_participant, choices=choices, label_rates_training=rates,
            chance_most_frequent=float(np.mean([p['chance_most_frequent'] for p in by_participant.values()])),
            chance_training_majority=float(np.mean([p['chance_training_majority'] for p in by_participant.values()])),
            chance_by_rates=float(np.mean([p['chance_by_rates'] for p in by_participant.values()])),
            chance_uniform=1 / len(choices),
            label=f'buttons held for at least {HELD_SECONDS:g} s of the TR, in the order {"+".join(BUTTONS)}; '
                  f'a label under {RARE_LABEL:.0%} of the training TRs is other',
            fitting=f'every {FIT_EVERY}th TR of the training runs', split=STATED['split'])
