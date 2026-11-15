"""TODO: one-line description — whose measurements, what stimuli, which paper.

A data plugin supplies the two halves a benchmark needs:

  * a **StimulusSet** — what was shown, one row per stimulus
  * a **DataAssembly** — what was measured, `(presentation, neuroid)` for neural data
    or `(presentation, ...)` for behavioral

Nothing here scores anything. A benchmark loads these and compares a model against them.

**This file runs as-is.** It builds both from files it writes once to a folder in the Brain-Score home, so you
can see the shapes before you have your own data wired up. Replace `_build_example_files`
with something that reads *your* files, keep the rest, and the tests tell you when the
result is still benchmark-usable.
"""
import io
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from brainscore_core.supported_data_standards.brainio.assemblies import (
    BehavioralAssembly, NeuroidAssembly)
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet

STIMULUS_SET_ID = 'your-stimuli'
ASSEMBLY_ID = 'your-measurements'
CHOICES_ID = 'your-choices'

N_STIMULI, N_NEUROIDS, N_SUBJECTS = 32, 16, 4
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


def _build_example_files():
    """TODO: replace entirely. Returns (directory, rows) for the example stimuli.

    Yours will instead point at images already on disk and read a CSV of metadata.
    The only thing that must survive is the SHAPE of what this returns.
    """
    # The images are the same on every call, so they are written once, to one folder in the
    # Brain-Score home, and reused.
    directory = Path(os.getenv('BRAINSCORE_HOME', '~/.brain-score')).expanduser() / 'your_data_stimuli'
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(N_STIMULI):
        stimulus_id = f'stim{i:03d}'
        path = str(directory / f'{stimulus_id}.png')
        image = io.BytesIO()
        Image.fromarray(np.random.RandomState(i).randint(0, 255, (64, 64, 3), dtype=np.uint8)) \
            .save(image, format='PNG')
        _write_once(path, image.getvalue())
        rows.append({'stimulus_id': stimulus_id,
                     # 'image_path' is load-bearing: dispatch decides a stimulus
                     # set's MODALITY by looking for a recognized column name. Use
                     # 'sentence' for text, 'audio_path' for audio, 'video_path' for
                     # video. Without one of these, a session raises "No column in
                     # this stimulus set says what kind of input it holds".
                     'image_path': path,
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
    stimulus_set.stimulus_paths = {r['stimulus_id']: r['image_path'] for r in rows}
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


def load_choices() -> BehavioralAssembly:
    """What participants answered: one row per (participant, trial), the label as the value.

    TODO: replace the made-up answers with yours. A behavioral benchmark compares a model's
    answers with these, matched by `stimulus_id`. Here each participant names the category
    of each stimulus and is right four times in five.
    """
    stimulus_set = load_stimulus_set()
    rng = np.random.RandomState(1)
    stimulus_ids, participants, answers = [], [], []
    for participant in range(N_SUBJECTS):
        for stimulus_id, category in zip(stimulus_set['stimulus_id'], stimulus_set['object_name']):
            stimulus_ids.append(stimulus_id)
            participants.append(f'sub-{participant + 1:02d}')
            answers.append(category if rng.rand() < 0.8 else rng.choice(CATEGORIES))
    choices = BehavioralAssembly(
        answers,
        coords={'stimulus_id': ('presentation', stimulus_ids),
                'participant': ('presentation', participants)},
        dims=['presentation'])
    choices.attrs['stimulus_set'] = stimulus_set
    choices.attrs['identifier'] = CHOICES_ID
    return choices


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
