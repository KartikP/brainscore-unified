"""Test word relationships using a vision subject's recorded activity.

Implements the paired word directions and forward partner retrieval in
Gjølbye et al., arXiv:2610.13252, sections 3.3–3.4. This is a raw research
score, without a human ceiling. Construct it explicitly with local data.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
from brainscore_core.benchmarks import BenchmarkBase
from brainscore_core.extraction_cache import weight_fingerprint_scope
from brainscore_core.metrics import Score
from brainscore_core.model_interface import Subject


def file_sha256(path: Path) -> str:
    """Hash file contents without loading a whole file into memory."""
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def retrieval_ranks(directions: np.ndarray, words: list[str],
                    pairs: list[tuple[str, str]]) -> list[int]:
    """Return zero-based ranks, counting only candidates strictly closer.

    All words remain candidates, including pseudowords. The query itself is
    excluded. Equal similarities favor the partner, as in the authors' code.
    """
    values = np.asarray(directions, dtype=np.float32)
    if values.ndim != 2 or len(values) != len(words):
        raise ValueError('Directions must have one row per word')
    if not np.isfinite(values).all() or np.any(np.linalg.norm(values, axis=1) == 0):
        raise ValueError('Word directions must be finite and nonzero')
    index = {word: position for position, word in enumerate(words)}
    normalized = values / (np.linalg.norm(values, axis=1, keepdims=True) + 1e-9)
    similarities = normalized @ normalized.T
    np.fill_diagonal(similarities, -np.inf)
    return [int(np.sum(similarities[index[a]] > similarities[index[a], index[b]]))
            for a, b in pairs]


class RenderedWordRetrieval(BenchmarkBase):
    """Present paired images, record features, and rank each query's partner.

    ``manifest`` contains word, slot, condition (word/scramble), and an
    image_path for every rendering. Each word must have the same slots.
    ``pairs`` contains query/partner strings, not training examples. Nothing
    is fitted to these labels. ``batch_size`` counts word/control pairs.
    """

    required_modalities = {'vision'}
    leaderboard_eligible = False
    recording_target = 'word_representation'

    def __init__(self, manifest: pd.DataFrame, pairs: list[tuple[str, str]], *,
                 top_k: int = 10, batch_size: int = 16) -> None:
        required = {'word', 'slot', 'condition', 'image_path'}
        if not required.issubset(manifest.columns):
            raise ValueError(f'Manifest needs columns {sorted(required)}')
        rows = manifest[list(sorted(required))].copy()
        if rows.empty or rows.isna().any().any():
            raise ValueError('Manifest must contain complete, nonempty rows')
        if not rows['condition'].isin(['word', 'scramble']).all():
            raise ValueError('Only word and scramble conditions are accepted')
        if rows.duplicated(['word', 'slot', 'condition']).any():
            raise ValueError('Duplicate word/slot/condition')
        if not rows['word'].map(lambda value: isinstance(value, str) and bool(value)).all():
            raise ValueError('Words must be nonempty strings')
        if not rows['slot'].map(
            lambda value: isinstance(value, (int, np.integer)) and value >= 0
        ).all():
            raise ValueError('Slots must be nonnegative integers')
        # Sort each condition by the same keys, then pair explicitly.
        word = rows[rows['condition'] == 'word'].sort_values(['word', 'slot'])
        control = rows[rows['condition'] == 'scramble'].sort_values(['word', 'slot'])
        keys = ['word', 'slot']
        if not word[keys].reset_index(drop=True).equals(control[keys].reset_index(drop=True)):
            raise ValueError('Every word/slot needs exactly one scrambled control')
        self.words = sorted(word['word'].unique())
        self.slots = [int(slot) for slot in sorted(word['slot'].unique())]
        if len(word) != len(self.words) * len(self.slots):
            raise ValueError('Every word must have the same slots')
        self.pairs = [tuple(pair) for pair in pairs]
        if not self.pairs or len(set(self.pairs)) != len(self.pairs):
            raise ValueError('Query pairs must be nonempty and unique')
        for a, b in self.pairs:
            if a == b or a not in self.words or b not in self.words:
                raise ValueError('Each query needs a distinct partner in the vocabulary')
        if not isinstance(top_k, int) or not 1 <= top_k < len(self.words):
            raise ValueError('top_k must be positive and smaller than the vocabulary')
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError('batch_size must be positive')
        self.top_k = top_k
        self.batch_size = batch_size
        # Interleave each word with its own control. Row order in the source
        # manifest cannot accidentally pair unrelated images.
        self.rows = pd.concat([word, control]).sort_values(
            ['word', 'slot', 'condition'], ascending=[True, True, False],
        ).reset_index(drop=True)
        digest = hashlib.sha256()
        self.image_hashes = []
        for row in self.rows.itertuples():
            path = Path(row.image_path)
            checksum = file_sha256(path)
            self.image_hashes.append(checksum)
            digest.update(json.dumps(
                [row.word, int(row.slot), row.condition, checksum],
                ensure_ascii=False,
            ).encode())
        digest.update(json.dumps(self.pairs).encode())
        self.data_sha256 = digest.hexdigest()
        super().__init__(
            identifier=f'RenderedWords-partner-top{top_k}-data-{self.data_sha256}',
            version=1,
            ceiling=None,
            parent='engineering',
            bibtex='@misc{gjolbye2026vision, title={Vision Transformers Learn Word Meaning '
                   'from Images Alone}, author={Gjølbye, Anders and Mason, Sebastian Ray '
                   'and Frank, Stella and Hansen, Lars Kai}, year={2026}, '
                   'url={https://arxiv.org/abs/2610.13252}}',
        )

    @weight_fingerprint_scope()
    def __call__(self, candidate: Subject) -> Score:
        """Return raw retrieval accuracy and the directions used to calculate it."""
        if 'vision' not in candidate.supported_modalities:
            raise ValueError('Rendered words require a subject with vision input')
        # Refuse edited images before making any model calls.
        for path, checksum in zip(self.rows['image_path'], self.image_hashes):
            if file_sha256(Path(path)) != checksum:
                raise ValueError('Image contents changed after preflight')
        candidate.start_recording(self.recording_target)
        index = {word: position for position, word in enumerate(self.words)}
        sums = None
        feature_ids = None
        for start in range(0, len(self.rows), 2 * self.batch_size):
            batch = self.rows.iloc[start:start + 2 * self.batch_size]
            ids = [str(position) for position in batch.index]
            stimuli = StimulusSet({
                'stimulus_id': ids,
                'image_file_name': batch['image_path'].tolist(),
            })
            stimuli.stimulus_paths = dict(zip(ids, batch['image_path'].map(Path)))
            stimuli.identifier = f'{self.identifier}-rows-{start}-{len(batch)}'
            activity = candidate.process(stimuli)
            if set(activity.dims) != {'presentation', 'neuroid'}:
                raise ValueError('Expected activity with presentation and neuroid dimensions')
            returned_ids = list(map(str, activity['stimulus_id'].values))
            if len(returned_ids) != len(ids) or set(returned_ids) != set(ids):
                raise ValueError('Model returned missing or duplicate stimulus IDs')
            current_features = list(map(str, activity['neuroid_id'].values))
            if feature_ids is not None and current_features != feature_ids:
                raise ValueError('Recorded units changed between batches')
            feature_ids = current_features
            order = {identifier: position for position, identifier in enumerate(returned_ids)}
            values = activity.transpose('presentation', 'neuroid').values[
                [order[identifier] for identifier in ids]
            ]
            # The authors store each image's features in float16 before
            # subtracting controls and accumulating in float32.
            with np.errstate(over='ignore', invalid='ignore'):
                values = values.astype(np.float16).astype(np.float32)
            if not np.isfinite(values).all():
                raise ValueError('Recorded activity is nonfinite or exceeds float16 range')
            differences = values[0::2] - values[1::2]
            if sums is None:
                sums = np.zeros((len(self.words), values.shape[1]), dtype=np.float32)
            positions = [index[word] for word in batch['word'].iloc[0::2]]
            np.add.at(sums, positions, differences)
        directions = sums / len(self.slots)
        directions -= directions.mean(axis=0, keepdims=True)
        ranks = retrieval_ranks(directions, self.words, self.pairs)
        score = Score(float(np.mean(np.asarray(ranks) < self.top_k)))
        score.attrs.update(
            raw=Score(float(score)),
            leaderboard_eligible=False,
            benchmark_identifier=self.identifier,
            data_sha256=self.data_sha256,
            words=self.words,
            slots=self.slots,
            directions=directions,
            pairs=self.pairs,
            ranks=ranks,
            chance=self.top_k / (len(self.words) - 1),
            feature_storage_dtype='float16',
        )
        return score
