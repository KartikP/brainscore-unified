"""
ROAR (Rapid Online Assessment of Reading) — Yeatman 2021 benchmark.

Lexical decision task: 500 visually-presented words and pseudo-words. Humans
judge "real" vs "pseudo". Behavioral ground truth is per-stimulus mean
accuracy across 120 subjects.

Model must implement behavioral readout (see BrainScoreModel.behavioral_readout_layer):
the model's predicted P(real) for each stimulus is correlated against human
per-stimulus accuracy via Pearson r.

Data sources (s3://brainscore-storage/brainscore-vision/data/user_764/):
- assy_Yeatman2021.nc       — 60,000 trials (500 stimuli × 120 subjects)
- stimulus_Yeatman2021.csv  — stimulus metadata (label, word, real/pseudo)
- stimulus_Yeatman2021.zip  — 500 PNG images, 500×300 RGB

This is the first Brain-Score behavioral benchmark that uses the unified
interface: it calls model.process() directly, relying on BrainScoreModel's
behavioral readout (ProbabilitiesClassifier) to turn features into
per-label probabilities.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import pearsonr

from brainscore_core.benchmarks import BenchmarkBase
from brainscore_core.metrics import Score
from brainscore_core.model_interface import TaskContext
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
from brainscore_core.supported_data_standards.brainio.assemblies import BehavioralAssembly


BIBTEX = """@article{yeatman2021rapid,
  title={Rapid online assessment of reading ability},
  author={Yeatman, Jason D and Tang, Kenny A and Donnelly, Patrick M and
          Mezer, Aviv A and Wandell, Brian A and White, Alex L},
  journal={Scientific Reports},
  volume={11},
  number={1},
  pages={6396},
  year={2021},
  publisher={Nature Publishing Group UK London},
  doi={10.1038/s41598-021-85907-x}
}"""


DATA_DIR = Path('/Users/kartik/Brain-Score Unified/data/roar_yeatman2021')


def _load_stimulus_set(data_dir: Path = DATA_DIR) -> StimulusSet:
    """Load the ROAR stimulus set.

    Returns a StimulusSet with columns:
    - stimulus_id: 'roar_0000' through 'roar_0499'
    - image_file_name: absolute path to the PNG
    - image_label: 'real' or 'pseudo' (string labels for ProbabilitiesClassifier)
    - word: the displayed word string
    - numeric_label: 1 (real) or 0 (pseudo)
    """
    csv_path = data_dir / 'stimulus_Yeatman2021.csv'
    stimuli_dir = data_dir / 'stimuli'
    df = pd.read_csv(csv_path)

    # Map numeric labels to canonical string labels used by ProbabilitiesClassifier
    label_map = {1: 'real', 0: 'pseudo'}
    df['image_label'] = df['label'].map(label_map)
    df['numeric_label'] = df['label']
    df['image_file_name'] = df['filename'].apply(
        lambda fn: str(stimuli_dir / fn))

    # Keep only the columns we need in a clean order
    df = df[['stimulus_id', 'image_file_name', 'image_label',
             'numeric_label', 'word', 'realpseudo']]

    stimulus_set = StimulusSet(df)
    stimulus_set.identifier = 'Yeatman2021'
    # stimulus_paths dict is used by some downstream code (e.g., VLMVisionWrapper)
    stimulus_set.stimulus_paths = dict(
        zip(df['stimulus_id'].values, df['image_file_name'].values))
    return stimulus_set


def _load_human_assembly(data_dir: Path = DATA_DIR) -> xr.Dataset:
    """Load the 60,000-trial human behavioral dataset.

    Schema (verified against CSV labels, April 2026):
    - `data`: subject's response (1 = "real", 0 = "pseudo")
    - `correct`: ground-truth label (1 = real word, 0 = pseudo-word)
      — NOTE: the name is misleading; this is the stimulus label, not
      whether the subject was correct. Per-trial accuracy is (data == correct).
    - `stimulus_id`, `subject`: coordinates on the presentation dim.
    """
    return xr.open_dataset(data_dir / 'assy_Yeatman2021.nc')


def _aggregate_per_stimulus_accuracy(ds: xr.Dataset) -> pd.Series:
    """Mean human accuracy per stimulus across all subjects.

    Returns a pandas Series indexed by stimulus_id with values in [0, 1].
    Accuracy is computed as (response == ground_truth_label).
    """
    accuracy = (ds['data'].values == ds['correct'].values).astype(float)
    df = pd.DataFrame({
        'stimulus_id': ds['stimulus_id'].values,
        'accuracy': accuracy,
    })
    return df.groupby('stimulus_id')['accuracy'].mean()


def _split_half_ceiling(ds: xr.Dataset, n_splits: int = 100,
                        random_state: int = 0) -> float:
    """Split-half subject reliability as the noise ceiling.

    Split subjects into two random halves, compute per-stimulus mean accuracy
    in each half, correlate, and apply Spearman-Brown correction. Repeat
    `n_splits` times and return the mean.
    """
    rng = np.random.default_rng(random_state)
    subjects = np.unique(ds['subject'].values)
    n_sub = len(subjects)

    accuracy = (ds['data'].values == ds['correct'].values).astype(float)
    df = pd.DataFrame({
        'stimulus_id': ds['stimulus_id'].values,
        'subject': ds['subject'].values,
        'accuracy': accuracy,
    })

    correlations = []
    for _ in range(n_splits):
        shuffled = rng.permutation(subjects)
        half_a = set(shuffled[:n_sub // 2])
        half_b = set(shuffled[n_sub // 2:])
        acc_a = df[df['subject'].isin(half_a)].groupby('stimulus_id')['accuracy'].mean()
        acc_b = df[df['subject'].isin(half_b)].groupby('stimulus_id')['accuracy'].mean()
        common = acc_a.index.intersection(acc_b.index)
        r, _ = pearsonr(acc_a.loc[common].values, acc_b.loc[common].values)
        correlations.append(r)
    r_half = np.mean(correlations)
    # Spearman-Brown: double the correlation because full data has 2× subjects
    r_full = 2 * r_half / (1 + r_half)
    return float(r_full)


class Yeatman2021LexicalDecision(BenchmarkBase):
    """ROAR lexical decision benchmark (Yeatman 2021).

    Workflow:
    1. Start task: tell the model to produce probabilities with ROAR stimuli
       as fitting data (labels: 'real' / 'pseudo').
    2. Process stimuli: get a BehavioralAssembly (500, 2) of per-label probs.
    3. Score: Pearson r between model P(real) and human per-stimulus accuracy.

    The fitting and evaluation stimuli are the same 500 items — this benchmark
    tests how well the model's wordness judgment tracks human accuracy
    patterns, not held-out classification accuracy.
    """

    def __init__(self, ceiling: Optional[float] = None):
        self._stimulus_set = _load_stimulus_set()
        self._human_assembly = _load_human_assembly()
        self._human_accuracy = _aggregate_per_stimulus_accuracy(self._human_assembly)

        if ceiling is None:
            # Compute ceiling once (it's deterministic given random_state)
            ceiling = _split_half_ceiling(self._human_assembly, n_splits=100)

        super().__init__(
            identifier='Yeatman2021-lexical_decision',
            version=1,
            parent='behavioral',
            ceiling=Score(ceiling),
            bibtex=BIBTEX,
        )

    def __call__(self, candidate) -> Score:
        # Fit behavioral readout on the stimulus set with its real/pseudo labels
        task_context = TaskContext(
            task_type='probabilities',
            fitting_stimuli=self._stimulus_set,
            label_set=['real', 'pseudo'],
            instruction="Is the displayed string a real English word or a pseudo-word?",
        )
        candidate.start_task(task_context)

        # Get per-stimulus probabilities
        predictions = candidate.process(self._stimulus_set)
        # predictions has dims (presentation, choice). 'choice' contains
        # 'real' and 'pseudo' in the order the classifier first saw them.
        model_p_real = predictions.sel(choice='real').values

        # Align to the same stimulus order as human_accuracy
        stim_ids = list(predictions['stimulus_id'].values)
        human_acc = self._human_accuracy.reindex(stim_ids).values

        mask = ~np.isnan(human_acc)
        if not mask.all():
            # Shouldn't normally happen — every ROAR stimulus has subject data
            n_missing = int((~mask).sum())
            raise ValueError(
                f"{n_missing} stimuli have no human accuracy data; "
                f"check stimulus_id alignment.")

        raw_r, p_value = pearsonr(model_p_real, human_acc)
        ceiling_value = float(self.ceiling)
        ceiled = raw_r / ceiling_value
        score = Score(ceiled)
        score.attrs['raw'] = Score(raw_r)
        score.attrs['ceiling'] = ceiling_value
        score.attrs['p_value'] = float(p_value)
        score.attrs['n_stimuli'] = int(mask.sum())
        return score
