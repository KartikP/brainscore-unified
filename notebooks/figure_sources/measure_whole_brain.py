"""Rerun the full evaluation and save the values used by notebook 16.

Set BRAINSCORE_LEBEL_PICKLE before running. GPT-2 weights are downloaded on first
use. This writes scores and provenance; render_whole_brain_figures.py draws them.
"""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time

import numpy as np
import torch
import brainscore
from brainscore.data.lebel2023.data import _pickle_path
from brainscore.model_helpers._device import select_device

REPO = Path(__file__).resolve().parents[2]
ASSETS = REPO / 'notebooks' / 'assets' / 'whole_brain'
ASSETS.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(4)
started = time.time()

# Use the registered model and benchmark without changing their settings.
model = brainscore.load_model('gpt2')
benchmark = brainscore.load_benchmark('LeBel2023-UTS03-encoding')
score = benchmark(model)
per_vertex = np.asarray(score.attrs['raw'])
# The full benchmark uses None to mean every target in assembly order.
selected = score.attrs['target_index']
indices = np.arange(len(per_vertex)) if selected is None else np.asarray(selected)
assert len(per_vertex) == 20484
assert np.array_equal(np.sort(indices), np.arange(20484))
np.savez_compressed(
    ASSETS / 'gpt2_lebel_scores.npz',
    per_vertex=per_vertex,
    target_index=indices,
)

# These private data fields record provenance, not an extension API example.
words, assembly, story_times = benchmark._word_data()
with _pickle_path().open('rb') as handle:
    data_hash = hashlib.file_digest(handle, 'sha256').hexdigest()
metadata = {
    'measured': True,
    'model': 'gpt2',
    'layer': 'h.11',
    'benchmark': 'LeBel2023-UTS03-encoding',
    'subject': 'UTS03',
    'median_r': float(score),
    'mean_r': float(score.attrs['mean_r']),
    'n_targets': len(per_vertex),
    'n_samples': len(assembly),
    'n_words': len(words),
    'n_stories': len(story_times),
    'finite_vertices': int(np.isfinite(per_vertex).sum()),
    'positive_fraction': float(np.mean(per_vertex > 0)),
    'alphas': score.attrs['alphas'],
    'elapsed_seconds': time.time() - started,
    'timestamp_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    'dataset_url': 'https://drive.google.com/file/d/1q-XLPjvhd8doGFhYBmeOkcenS9Y59x64/view',
    'dataset_source': 'https://github.com/GT-LIT-Lab/litcoder_core',
    'dataset_sha256': data_hash,
    'python': platform.python_version(),
    'device': str(select_device()),
    'package_versions': {
        name: importlib.metadata.version(name)
        for name in ('torch', 'transformers', 'numpy', 'scipy', 'scikit-learn')
    },
    'notes': (
        'Raw held-out Pearson correlations. Five whole-story folds; registered '
        'defaults. No noise-ceiling normalization. No anatomical mask.'
    ),
}
try:
    metadata['unified_git_revision'] = subprocess.check_output(
        ['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True,
    ).strip()
except (OSError, subprocess.CalledProcessError):
    metadata['unified_git_revision'] = None
(ASSETS / 'measurements.json').write_text(json.dumps(metadata, indent=2) + '\n')
print(f"Saved {len(per_vertex):,} measured scores; median r = {float(score):.4f}")
