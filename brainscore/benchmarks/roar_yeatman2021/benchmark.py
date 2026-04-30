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

This is unified-interface-native: it uses BrainScoreModel's behavioral
readout (start_task(TaskContext(task_type='probabilities', ...)) +
process(test_stimuli) -> BehavioralAssembly).

Data sources (s3://brainscore-storage/brainscore-vision/data/user_764/):
- assy_Yeatman2021.nc       — 60,000 trials (500 stimuli × 120 subjects)
- stimulus_Yeatman2021.csv  — stimulus metadata (label, word, real/pseudo)
- stimulus_Yeatman2021.zip  — 500 PNG images, 500×300 RGB

Human assembly schema (verified April 2026):
- `data`: subject response (1 = "real", 0 = "pseudo")
- `correct`: ground-truth label (1 = real word, 0 = pseudo-word) — the
  name is misleading; this is the stimulus label, NOT per-trial accuracy.
  Accuracy is computed as (data == correct).
"""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr

from brainscore_core.benchmarks import BenchmarkBase
from brainscore_core.metrics import Score
from brainscore_core.model_interface import TaskContext
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet


BIBTEX = """@article{yeatman2021rapid,
  title={Rapid online assessment of reading ability},
  author={Yeatman, Jason D and Tang, Kenny A and Donnelly, Patrick M and
          Mezer, Aviv A and Wandell, Brian A and White, Alex L},
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


DATA_DIR = Path('/Users/kartik/Brain-Score Unified/data/roar_yeatman2021')

# Paper protocol: 200/50 real, 200/50 pseudo
TRAIN_PER_CLASS = 200
TEST_PER_CLASS = 50
DYSLEXIA_THRESHOLD = 0.65  # 1 SD below human mean (Honarmand et al. 2026)
SPLIT_SEED = 0


def _load_stimulus_set(data_dir: Path = DATA_DIR) -> StimulusSet:
    """Load the ROAR stimulus set with labels.

    Columns: stimulus_id, image_file_name, image_label ('real'/'pseudo'),
    numeric_label (0/1), word, realpseudo.
    """
    csv_path = data_dir / 'stimulus_Yeatman2021.csv'
    stimuli_dir = data_dir / 'stimuli'
    df = pd.read_csv(csv_path)
    df['image_label'] = df['label'].map({1: 'real', 0: 'pseudo'})
    df['numeric_label'] = df['label']
    df['image_file_name'] = df['filename'].apply(
        lambda fn: str(stimuli_dir / fn))
    # `sentence` column lets text-only models (e.g., GPT-2) process the
    # word string directly. Vision models still route via image_file_name
    # (MODALITY_PRIORITY in BrainScoreModel picks vision when both are
    # present).
    df['sentence'] = df['word']
    df = df[['stimulus_id', 'image_file_name', 'sentence', 'image_label',
             'numeric_label', 'word', 'realpseudo']]

    stimulus_set = StimulusSet(df)
    stimulus_set.identifier = 'Yeatman2021'
    stimulus_set.stimulus_paths = dict(
        zip(df['stimulus_id'].values, df['image_file_name'].values))
    return stimulus_set


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


def _load_human_assembly(data_dir: Path = DATA_DIR) -> xr.Dataset:
    """Load the 60k-trial human behavioral dataset."""
    return xr.open_dataset(data_dir / 'assy_Yeatman2021.nc')


def _human_accuracy_on_stimuli(ds: xr.Dataset, stimulus_ids: List[str]) -> float:
    """Mean human accuracy across subjects and the given stimuli."""
    accuracy = (ds['data'].values == ds['correct'].values).astype(float)
    stim_ids = ds['stimulus_id'].values
    mask = np.isin(stim_ids, stimulus_ids)
    if mask.sum() == 0:
        raise ValueError("No human trials matched the given stimulus_ids")
    return float(accuracy[mask].mean())


SUPPORTED_MODALITIES = {'vision', 'text'}

# Maps the model-side modality name to the user-facing leaderboard suffix.
# 'vision' (model speak) → 'image' (benchmark identifier suffix). 'text'
# stays 'text'. Keep this mapping local to ROAR so other benchmarks pick
# their own conventions.
_MODALITY_TO_SUFFIX = {'vision': 'image', 'text': 'text'}

# Columns to retain per modality. Anything not in the retained set is
# dropped from the projected stimulus set so model-side modality detection
# picks unambiguously and no stimulus row carries a modality the benchmark
# isn't presenting.
_MODALITY_COLUMNS = {
    'vision': {'image_file_name'},
    'text': {'sentence'},
}


def _project_to_modality(stimulus_set: StimulusSet, modality: str) -> StimulusSet:
    """Return a view of the stimulus set restricted to one input format.

    Drops the columns belonging to other modalities so that model-side
    `_detect_modalities` picks unambiguously. Preserves all metadata
    columns (label, word, etc.) and the stimulus_paths mapping.
    """
    keep_modality = _MODALITY_COLUMNS[modality]
    drop = set()
    for mod, cols in _MODALITY_COLUMNS.items():
        if mod != modality:
            drop |= cols
    keep_cols = [c for c in stimulus_set.columns if c not in drop]
    df = stimulus_set[keep_cols].copy()

    projected = StimulusSet(df)
    projected.identifier = f'{stimulus_set.identifier}-{_MODALITY_TO_SUFFIX[modality]}'
    if modality == 'vision':
        # Vision still resolves stimulus paths via stimulus_paths.
        projected.stimulus_paths = dict(stimulus_set.stimulus_paths)
    return projected


class Yeatman2021LexicalDecision(BenchmarkBase):
    """ROAR lexical decision — paper-replication (Honarmand et al. 2026).

    Train on 400 stimuli (200 real + 200 pseudo), test on 100 (50/50).
    Metric: model accuracy on test set, ceiling-normalized by human
    mean accuracy on the same test stimuli.

    Per the April 30, 2026 design decision, each leaderboard benchmark
    declares exactly one input format. ROAR is parameterized over the
    `modality` argument and registered as two pinned variants:
    `Yeatman2021-lexical_decision-image` and
    `Yeatman2021-lexical_decision-text`. Both share the same train/test
    split, ceiling, and metric — only the presentation differs.

    Toolbox use: construct directly with `modality='vision'` or
    `modality='text'` to compare across input formats.
    """

    SUPPORTED_MODALITIES = SUPPORTED_MODALITIES

    def __init__(self, modality: str = 'vision'):
        assert modality in self.SUPPORTED_MODALITIES, (
            f"modality must be one of {self.SUPPORTED_MODALITIES}; "
            f"got {modality!r}")
        self.modality = modality
        self.required_modalities = {modality}

        full_stimulus_set = _load_stimulus_set()
        # Split first on the full set so the train/test partition is the
        # same across modality variants. Then project each side to the
        # chosen modality.
        train_full, test_full = _split_train_test(full_stimulus_set)
        self._train_stimuli = _project_to_modality(train_full, modality)
        self._test_stimuli = _project_to_modality(test_full, modality)
        self._human_ds = _load_human_assembly()

        test_ids = list(self._test_stimuli['stimulus_id'].values)
        self._human_accuracy = _human_accuracy_on_stimuli(self._human_ds, test_ids)

        suffix = _MODALITY_TO_SUFFIX[modality]
        super().__init__(
            identifier=f'Yeatman2021-lexical_decision-{suffix}',
            version=2,
            parent='behavioral',
            ceiling=Score(self._human_accuracy),
            bibtex=BIBTEX,
        )

    def __call__(self, candidate) -> Score:
        # Fit behavioral readout on the 400-item train set
        task_context = TaskContext(
            task_type='probabilities',
            fitting_stimuli=self._train_stimuli,
            label_set=['real', 'pseudo'],
            instruction='Is this a real word or a pseudo word?',
        )
        candidate.start_task(task_context)

        # Predict on the 100-item test set
        predictions = candidate.process(self._test_stimuli)
        choice_values = list(predictions['choice'].values)

        # argmax over choices to get predicted label per stimulus
        prob_matrix = predictions.values  # (n_test, n_choices)
        pred_idx = prob_matrix.argmax(axis=1)
        predicted_labels = [choice_values[i] for i in pred_idx]

        # Ground truth
        stim_ids = list(predictions['stimulus_id'].values)
        truth_map = dict(zip(
            self._test_stimuli['stimulus_id'].values,
            self._test_stimuli['image_label'].values,
        ))
        truth = [truth_map[sid] for sid in stim_ids]

        correct = [p == t for p, t in zip(predicted_labels, truth)]
        raw_accuracy = float(np.mean(correct))

        ceiling_value = float(self.ceiling)
        ceiled = raw_accuracy / ceiling_value

        score = Score(ceiled)
        score.attrs['raw'] = Score(raw_accuracy)
        score.attrs['ceiling'] = ceiling_value
        score.attrs['n_test_stimuli'] = len(stim_ids)
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
