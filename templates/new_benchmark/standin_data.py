"""Replace this file with your data package.

Stand-in stimuli and measurements, so the template runs before you have data. The
images are real files on disk, so a real model (``templates/new_model/``) can be
scored against them.
"""
import io
import os
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet

BIBTEX = """TODO: @article{...} citation for the data."""

# 32 stimuli in 4 categories: the metric's default test fold is 10 % of the stimuli,
# stratified by `object_name`, so it needs one stimulus per category (4 of 32). A
# correlation on fewer stimuli means nothing and the metric does not warn; with fewer
# stimuli, set a larger test fold (see the metric in benchmark.py).
N_STIMULI, N_NEUROIDS = 32, 16
CATEGORIES = ['alpha', 'beta', 'gamma', 'delta']


def _write_once(path, data: bytes) -> None:
    """Write ``data`` to ``path`` unless the same bytes are already there.

    The bytes go to an exclusive staging file with a unique name in the same folder, which
    then replaces the final name in one step, so a reader never sees half a file and two
    writers do not collide. A link at the final name is refused: the write would land in
    another file.
    """
    if os.path.islink(path):
        raise ValueError(f'{path} is a link; remove it and load again')
    if os.path.isfile(path):
        with open(path, 'rb') as existing:
            if existing.read() == data:
                return
    descriptor, staging = tempfile.mkstemp(dir=os.path.dirname(path), suffix='.partial')
    try:
        with os.fdopen(descriptor, 'wb') as file:
            file.write(data)
        os.replace(staging, path)
    except BaseException:
        os.unlink(staging)
        raise


def load_assembly():
    """TODO: load your real target measurements as a DataAssembly.

    Replace this whole function. For naturalistic/temporal data, build the stimulus_set
    + per-(subject, run) assembly and reuse `brainscore/temporal.py`
    (temporal_bin, hrf_convolve, contiguous_block_cv).

    Note what the stand-in establishes, because your real assembly needs all of it:
      - dims (presentation, neuroid)
      - a `stimulus_id` coord on presentation, matching the stimulus set
      - an `object_name` coord — the cross-validated metrics stratify their splits on it
      - at least TWO coords per axis (a single coord is not promoted to a MultiIndex,
        and metrics then fail with "no stimulus_id on the presentation axis")
      - a `stimulus_set` whose `stimulus_paths` point at files that REALLY EXIST; a real
        subject opens them, unlike the dummy in the tests
    """
    rng = np.random.RandomState(0)
    stimulus_ids = [f'stim{i:03d}' for i in range(N_STIMULI)]
    categories = [CATEGORIES[i % len(CATEGORIES)] for i in range(N_STIMULI)]

    # The images are the same on every load, so they are written once, to one folder in the
    # Brain-Score home, and reused.
    directory = Path(os.getenv('BRAINSCORE_HOME', '~/.brain-score')).expanduser() / 'your_benchmark_stimuli'
    directory.mkdir(parents=True, exist_ok=True)
    paths = {}
    for i, sid in enumerate(stimulus_ids):
        path = directory / f'{sid}.png'
        image = io.BytesIO()
        Image.fromarray(
            np.random.RandomState(i).randint(0, 255, (64, 64, 3), dtype=np.uint8)).save(image, format='PNG')
        _write_once(str(path), image.getvalue())
        paths[sid] = str(path)

    stimulus_set = StimulusSet([
        {'stimulus_id': sid, 'image_path': paths[sid], 'object_name': obj}
        for sid, obj in zip(stimulus_ids, categories)])
    stimulus_set.stimulus_paths = paths
    stimulus_set.identifier = 'your-benchmark-stimuli'

    assembly = NeuroidAssembly(
        rng.randn(N_STIMULI, N_NEUROIDS),
        coords={'stimulus_id': ('presentation', stimulus_ids),
                'object_name': ('presentation', categories),
                'neuroid_id': ('neuroid', list(range(N_NEUROIDS))),
                'region': ('neuroid', ['IT'] * N_NEUROIDS)},
        dims=['presentation', 'neuroid'])
    assembly.attrs['stimulus_set'] = stimulus_set
    return assembly
