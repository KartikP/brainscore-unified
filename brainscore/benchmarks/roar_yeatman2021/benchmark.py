"""
ROAR (Rapid Online Assessment of Reading) — Yeatman 2021 lexical decision.

Protocol replicates Honarmand et al. (2026 ICLR) "Inducing Dyslexia in Vision
Language Models" (https://arxiv.org/abs/2509.24597). Section 3:

    "We present 200 real words and 200 pseudo words drawn from the full
    ROAR-Word corpus as the train set for finding the minimal VWFA mask,
    with the remaining 50 real words and 50 pseudo words serving as the
    test set for lexical evaluation. [...] we evaluate performance solely
    based on the accuracy of lexical decisions. [...] We define 65% ROAR
    performance as the threshold at which subjects are considered as
    dyslexic. This number is one standard deviation below the mean
    ROAR-score of the human population."

Our implementation:
1. Deterministic 400/100 train/test split (200 real + 200 pseudo train,
   50 real + 50 pseudo test) seeded on stimulus_id.
2. Fit ProbabilitiesClassifier on the 400 train stimuli.
3. Predict on the 100 test stimuli; compute accuracy.
4. Return model accuracy. Score is ceiling-normalized by human mean
   accuracy on the same 100 test stimuli.
5. `attrs['dyslexic']` flag: True if raw accuracy < 0.65 (paper's threshold).

This is unified-interface-native: it asks for the choice
(``behavior:choice``) of a ``Choice(['real', 'pseudo'])`` task with an
instruction and fitting stimuli, in one session over the test stimuli. The
subject's parts decide how it answers: an answer part, a reply, or a readout
whose registration states ``('readout', 'argmax')``.

Data are resolved through the unified data plugin:
`brainscore.load_stimulus_set('Yeatman2021')` and
`brainscore.load_dataset('Yeatman2021')`.

Human assembly schema (verified April 2026):
- `data`: subject response (1 = "real", 0 = "pseudo")
- `correct`: ground-truth label (1 = real word, 0 = pseudo-word) — the
  name is misleading; this is the stimulus label, NOT per-trial accuracy.
  Accuracy is computed as (data == correct).
"""

from typing import List, Sequence, Tuple

import numpy as np

from brainscore import load_dataset, load_stimulus_set
from brainscore_core.benchmarks import BenchmarkBase
from brainscore_core.metrics import Score
from brainscore_core.contract import TaskContext
from brainscore_core.responses import Choice
from brainscore_core.streaming_helpers import stimulus_session
from brainscore.benchmarks._score import benchmark_score
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet


BIBTEX = """@article{yeatman2021rapid,
  title={Rapid online assessment of reading ability},
  author={Yeatman, Jason D and Tang, Kenny An and Donnelly, Patrick M and
          Yablonski, Maya and Ramamurthy, Mahalakshmi and Karipidis, Iliana I and
          Caffarra, Sendy and Takada, Megumi E and Kanopka, Klint and
          Ben-Shachar, Michal and Domingue, Benjamin W},
  journal={Scientific Reports},
  volume={11},
  number={1},
  pages={6396},
  year={2021},
  doi={10.1038/s41598-021-85907-x}
}

@inproceedings{honarmand2026inducing,
  title={Inducing Dyslexia in Vision Language Models},
  author={Honarmand, Melika and Sharma, Ayati and AlKhamissi, Badr and
          Mehrer, Johannes and Schrimpf, Martin},
  booktitle={International Conference on Learning Representations},
  year={2026},
  url={https://arxiv.org/abs/2509.24597}
}"""


# Paper protocol: 200/50 real, 200/50 pseudo
TRAIN_PER_CLASS = 200
TEST_PER_CLASS = 50
DYSLEXIA_THRESHOLD = 0.65  # 1 SD below human mean (Honarmand et al. 2026)
# What the accuracy does with a trial that has no usable reading (Contract Rule 6.2).
ANSWER_POLICY = 'invalid: wrong, counted; missing: wrong, counted'
SPLIT_SEED = 0


def _split_train_test(stimulus_set: StimulusSet,
                      train_per_class: int = TRAIN_PER_CLASS,
                      test_per_class: int = TEST_PER_CLASS,
                      seed: int = SPLIT_SEED) -> Tuple[StimulusSet, StimulusSet]:
    """Deterministic stratified split.

    Returns (train_stimuli, test_stimuli). The split is seeded on
    stimulus_id ordering for reproducibility.
    """
    rng = np.random.default_rng(seed)
    train_ids: List[str] = []
    test_ids: List[str] = []

    for label in ('real', 'pseudo'):
        mask = stimulus_set['image_label'] == label
        ids = sorted(stimulus_set.loc[mask, 'stimulus_id'].tolist())
        if len(ids) < train_per_class + test_per_class:
            raise ValueError(
                f"Not enough {label} stimuli: need "
                f"{train_per_class + test_per_class}, got {len(ids)}.")
        shuffled = rng.permutation(ids)
        train_ids.extend(shuffled[:train_per_class].tolist())
        test_ids.extend(shuffled[train_per_class:train_per_class + test_per_class].tolist())

    train = _slice_stimulus_set(stimulus_set, train_ids, suffix='train')
    test = _slice_stimulus_set(stimulus_set, test_ids, suffix='test')
    return train, test


def _slice_stimulus_set(stimulus_set: StimulusSet, ids: List[str],
                        suffix: str) -> StimulusSet:
    subset_df = stimulus_set[stimulus_set['stimulus_id'].isin(ids)].reset_index(drop=True)
    new_set = StimulusSet(subset_df)
    new_set.identifier = f'{stimulus_set.identifier}-{suffix}'
    new_set.stimulus_paths = {
        sid: stimulus_set.stimulus_paths[sid] for sid in subset_df['stimulus_id']
    }
    return new_set


def _human_accuracy_on_stimuli(ds, stimulus_ids: List[str]) -> float:
    """Mean human accuracy across subjects and the given stimuli."""
    accuracy = (ds['data'].values == ds['correct'].values).astype(float)
    stim_ids = ds['stimulus_id'].values
    mask = np.isin(stim_ids, stimulus_ids)
    if mask.sum() == 0:
        raise ValueError("No human trials matched the given stimulus_ids")
    return float(accuracy[mask].mean())


# Columns to retain per channel. Anything not in the retained set is dropped from
# the projected stimulus set, so the subject sees one input format and no stimulus
# row carries a column the benchmark isn't presenting.
_CHANNEL_COLUMNS = {
    'vision:image': {'image_file_name'},
    'text': {'sentence'},
}


def _project_to_channel(stimulus_set: StimulusSet, channel: str) -> StimulusSet:
    """Return a view of the stimulus set restricted to one input channel.

    Drops the columns belonging to the other channel so that the session sends
    one input format. Preserves all metadata columns (label, word, etc.) and
    the stimulus_paths mapping.
    """
    drop = set()
    for other, cols in _CHANNEL_COLUMNS.items():
        if other != channel:
            drop |= cols
    keep_cols = [c for c in stimulus_set.columns if c not in drop]
    df = stimulus_set[keep_cols].copy()

    projected = StimulusSet(df)
    projected.identifier = f"{stimulus_set.identifier}-{channel.split(':')[-1]}"   # -image, -text
    if channel == 'vision:image':
        # Images resolve stimulus paths via stimulus_paths.
        projected.stimulus_paths = dict(stimulus_set.stimulus_paths)
    return projected


class Yeatman2021LexicalDecision(BenchmarkBase):
    """ROAR lexical decision — paper-replication (Honarmand et al. 2026).

    Test on 100 stimuli (50 real + 50 pseudo). Readout fits on 400 separate
    stimuli; generation answers the instruction without fitting. A subject
    with an answering part answers through it; a subject built without one
    answers through its readout (the path is a property of the subject).
    Metric: model accuracy on test set, ceiling-normalized by human
    mean accuracy on the same test stimuli.

    Per the April 30, 2026 design decision, each leaderboard benchmark
    declares exactly one input format. ROAR is parameterized over the
    `channels` argument and registered as two pinned variants:
    `Yeatman2021-lexical_decision-image` and
    `Yeatman2021-lexical_decision-text`. Both share the same train/test
    split, ceiling, and metric — only the presentation differs.

    Toolbox use: construct directly with `channels=('vision:image',)` or
    `channels=('text',)` to compare across input formats. ``identifier`` is
    required; a registry line passes its key.
    """

    # Asks for the choice of each word; every word is an independent trial.
    # Pre-flight reads the inputs from stimulus_set and task: the word and the question.
    requested_output_channels = frozenset({'behavior:choice'})
    trial_memory = 'none'

    def __init__(self, channels: Sequence[str] = ('vision:image',), *, identifier: str):
        channels = tuple(channels)
        if len(channels) != 1 or channels[0] not in _CHANNEL_COLUMNS:
            raise ValueError(
                f"channels must be one of {[(c,) for c in _CHANNEL_COLUMNS]}; "
                f"got {channels!r}")
        channel = channels[0]

        full_stimulus_set = load_stimulus_set('Yeatman2021')
        # Split first on the full set so the train/test partition is the
        # same across the variants. Then project each side to the
        # chosen channel.
        train_full, test_full = _split_train_test(full_stimulus_set)
        self._train_stimuli = _project_to_channel(train_full, channel)
        self.stimulus_set = _project_to_channel(test_full, channel)
        self._human_ds = load_dataset('Yeatman2021')

        test_ids = list(self.stimulus_set['stimulus_id'].values)
        self._human_accuracy = _human_accuracy_on_stimuli(self._human_ds, test_ids)

        super().__init__(
            identifier=identifier,
            version=1,
            parent='behavior',
            ceiling=Score(self._human_accuracy),
            bibtex=BIBTEX,
        )

    @property
    def task(self) -> TaskContext:
        """The question and its two answers; a readout fits on the 400 training words."""
        return TaskContext(
            fitting_stimuli=self._train_stimuli,
            instruction='Is this a real word or a pseudo word?',
            response=Choice(['real', 'pseudo']),
        )

    def __call__(self, subject) -> Score:
        # The subject's parts decide how it answers: an answer part, a reply, or a readout.
        # The 100-item test set is what gets answered.
        session = stimulus_session(self.stimulus_set, ['behavior:choice'], self.task)
        subject.participate(session)
        predictions = session.collect('behavior:choice')
        choice_values = list(predictions['choice'].values)

        # The chosen label of a valid trial is the column holding the 1. An invalid or
        # a missing trial (its state, set by core) counts as wrong: it is no reading.
        prob_matrix = predictions.values  # (n_test, n_choices): one-hot rows
        valid = predictions['response_state'].values == 'valid'
        pred_idx = np.nan_to_num(prob_matrix, nan=-1).argmax(axis=1)
        predicted_labels = [choice_values[i] if ok else None
                            for i, ok in zip(pred_idx, valid)]

        # Ground truth
        stim_ids = list(predictions['stimulus_id'].values)
        truth_map = dict(zip(
            self.stimulus_set['stimulus_id'].values,
            self.stimulus_set['image_label'].values,
        ))
        truth = [truth_map[sid] for sid in stim_ids]

        correct = [p == t for p, t in zip(predicted_labels, truth)]
        raw_accuracy = float(np.mean(correct))

        score = benchmark_score(self, raw=raw_accuracy, metric='accuracy', answers=predictions,
                                answer_policy=ANSWER_POLICY)
        score.attrs['dyslexia_threshold'] = DYSLEXIA_THRESHOLD
        score.attrs['dyslexic'] = bool(raw_accuracy < DYSLEXIA_THRESHOLD)
        # Breakdown by stimulus class
        truth_arr = np.array(truth)
        correct_arr = np.array(correct)
        for label in ('real', 'pseudo'):
            mask = truth_arr == label
            if mask.any():
                score.attrs[f'accuracy_{label}'] = float(correct_arr[mask].mean())
        return score
