"""TODO: one-line description — whose measurements, what stimuli, which paper.

A data plugin supplies the two halves a benchmark needs:

  * a **StimulusSet** — what was shown, one row per stimulus
  * a **DataAssembly** — what was measured, `(presentation, neuroid)` for neural data
    or `(presentation, ...)` for behavioral

Nothing here scores anything. A benchmark loads these and compares a model against them.

**This file runs as-is.** It builds both from files it writes to a temp directory, so you
can see the shapes before you have your own data wired up. Replace `_build_example_files`
with something that reads *your* files, keep the rest, and the tests tell you when the
result is still benchmark-usable.
"""
import os
import tempfile

import numpy as np
import pandas as pd
from PIL import Image

from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet

STIMULUS_SET_ID = 'your-stimuli'
ASSEMBLY_ID = 'your-measurements'

N_STIMULI, N_NEUROIDS, N_SUBJECTS = 32, 16, 4
CATEGORIES = ['alpha', 'beta', 'gamma', 'delta']


def _build_example_files():
    """TODO: replace entirely. Returns (directory, rows) for the example stimuli.

    Yours will instead point at images already on disk and read a CSV of metadata.
    The only thing that must survive is the SHAPE of what this returns.
    """
    directory = tempfile.mkdtemp(prefix='your_data_')
    rows = []
    for i in range(N_STIMULI):
        stimulus_id = f'stim{i:03d}'
        path = os.path.join(directory, f'{stimulus_id}.png')
        rng = np.random.RandomState(i)
        Image.fromarray(rng.randint(0, 255, (64, 64, 3), dtype=np.uint8)).save(path)
        rows.append({'stimulus_id': stimulus_id,
                     # 'image_file_name' is load-bearing: dispatch decides a stimulus
                     # set's MODALITY by looking for a recognized column name. Use
                     # 'sentence' for text, 'audio_path' for audio, 'video_path' for
                     # video. Without one of these, process() raises
                     # "No recognized modality columns".
                     'image_file_name': path,
                     # Any extra columns ride along as presentation metadata. Keep
                     # 'object_name' (or whatever your categories are called) — the
                     # cross-validated metrics stratify their splits on a presentation
                     # coord, and fail without it.
                     'object_name': CATEGORIES[i % len(CATEGORIES)]})
    return directory, rows


def load_stimulus_set() -> StimulusSet:
    """What was shown. One row per stimulus."""
    _, rows = _build_example_files()
    stimulus_set = StimulusSet(pd.DataFrame(rows))
    # stimulus_paths maps each id to a file that must REALLY EXIST — a model opens them.
    stimulus_set.stimulus_paths = {r['stimulus_id']: r['image_file_name'] for r in rows}
    stimulus_set.identifier = STIMULUS_SET_ID
    return stimulus_set


def load_assembly() -> NeuroidAssembly:
    """What was measured, aligned to the stimulus set by `stimulus_id`.

    TODO: replace the random values with your recordings. Everything else about the
    shape below is what makes an assembly usable by a benchmark:

      * dims `(presentation, neuroid)`
      * `stimulus_id` on presentation, matching the stimulus set exactly
      * at least TWO coords per axis — with only one, brainio does not build the
        MultiIndex, and metrics then fail with "no stimulus_id on the presentation axis"
      * whatever your metric stratifies on (here `object_name`)
      * per-neuroid provenance: which subject, which region, which unit
    """
    stimulus_set = load_stimulus_set()
    stimulus_ids = list(stimulus_set['stimulus_id'].values)
    categories = list(stimulus_set['object_name'].values)

    rng = np.random.RandomState(0)
    values = rng.randn(len(stimulus_ids), N_NEUROIDS)

    # Neuroids belong to subjects; keep that, because leave-one-subject-out CV needs it.
    subjects = [f'sub-{(i % N_SUBJECTS) + 1:02d}' for i in range(N_NEUROIDS)]

    assembly = NeuroidAssembly(
        values,
        coords={'stimulus_id': ('presentation', stimulus_ids),
                'object_name': ('presentation', categories),
                'neuroid_id': ('neuroid', [f'n{i:03d}' for i in range(N_NEUROIDS)]),
                'subject': ('neuroid', subjects),
                'region': ('neuroid', ['IT'] * N_NEUROIDS)},
        dims=['presentation', 'neuroid'])
    assembly.attrs['stimulus_set'] = stimulus_set
    assembly.attrs['identifier'] = ASSEMBLY_ID
    return assembly


# ---------------------------------------------------------------------------
# Publishing, when you are ready to share
# ---------------------------------------------------------------------------
# The loaders above read local files, which is right while you are developing. Registered
# datasets in this repository instead pull from S3 with pinned checksums and version ids,
# so everyone scores against byte-identical data:
#
#     from brainscore_core.supported_data_standards.brainio.s3 import (
#         load_stimulus_set_from_s3, load_assembly_from_s3)
#
#     stimulus_set = load_stimulus_set_from_s3(
#         identifier=STIMULUS_SET_ID, bucket=..., csv_sha1=..., zip_sha1=...,
#         csv_version_id=..., zip_version_id=...)
#
# See `brainscore/data/roar_yeatman2021/__init__.py` for a complete worked example.
# Pin the checksums: an unpinned dataset can change under a published score.
