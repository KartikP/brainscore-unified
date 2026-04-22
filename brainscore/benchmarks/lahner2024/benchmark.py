"""
Lahner 2024 BOLDMoments fMRI naturalistic benchmark (unified interface).

Subjects watched short (~3s) video clips while being scanned with fMRI. The
dataset contains fMRI responses to 1026 videos across multiple subjects.
This is the first naturalistic Brain-Score benchmark authored against the
unified model interface.

Data source (same S3 location as brain-score/vision PR #1249, not a
duplicate fetch):

    s3://brainscore-storage/brainscore-vision/benchmarks/Lahner2024-fMRI/
        ├── stimulus_BOLDMoments.csv   (1026 video stimuli)
        ├── stimulus_BOLDMoments.zip   (~250 MB, MP4 files)
        └── assy_Lahner2024-fMRI.nc    (~1 GB, NeuroidAssembly)

Architecture — what's taken from PR #1249 and what's reframed:

- **Taken:** dataset versioning metadata (S3 version_ids, SHA1s), identifier
  conventions ('Lahner2024-fMRI', 'BOLDMoments'), citation, the core pattern
  that a naturalistic benchmark loads a StimulusSet + time-resolved
  NeuroidAssembly from S3.
- **Reframed:** instead of depending on the vision-specific `Video`/`Stimulus`
  class hierarchy and `TemporalInferencer` (which work only for
  video-native models like V-JEPA), this benchmark uses ``temporal_bin``
  from ``brainscore_core.temporal``. Models that process independent
  frames (CLIP, Qwen, BLIP-2 — anything with a vision preprocessor) can
  score via frame expansion → per-frame extraction → post-hoc temporal
  binning. Video-native models can still use the vision-side
  TemporalInferencer if PR #1249 lands; both paths produce the same
  ``(presentation, time_bin, neuroid)`` assembly shape.

Scoring status on laptop: data load alone is ~1.2 GB (zip + assembly).
Actual scoring is intended for EC2. This file ships as infrastructure and
is exercised by a small smoke test (metadata-only) locally.
"""

from typing import Optional

import numpy as np
import xarray as xr

from brainscore_core.benchmarks import BenchmarkBase
from brainscore_core.metrics import Score
from brainscore_core.supported_data_standards.brainio.assemblies import (
    NeuroidAssembly, NeuronRecordingAssembly,
)
from brainscore_core.supported_data_standards.brainio.s3 import (
    load_assembly_from_s3, load_stimulus_set_from_s3,
)


BIBTEX = """@article{lahner2024modeling,
  title={Modeling short visual events through the BOLD moments video fMRI dataset and metadata},
  author={Lahner, Benjamin and Dwivedi, Kshitij and Iamshchinina, Polina and
          Graumann, Monika and Lascelles, Alex and Roig, Gemma and
          Gifford, Alessandro T and Pan, Bowen and Jin, Sunny-Yu and
          Ratan Murty, N Apurva and others},
  journal={Nature Communications},
  volume={15},
  pages={6241},
  year={2024},
}"""


# S3 versioning (same as brain-score/vision PR #1249)
STIMULUS_ID = 'BOLDMoments'
STIMULUS_BUCKET = 'brainscore-storage/brainscore-vision/benchmarks/Lahner2024-fMRI'
STIMULUS_CSV_SHA1 = '0b27388f5898c908f58cd1f21f8f5cb3eda8536e'
STIMULUS_ZIP_SHA1 = 'dc9c3bf631632cd433d02f2f1847fd33c01ae0b3'
STIMULUS_CSV_VERSION_ID = 'WaGkWh59b1drhy1MmAVVSxh7_VT_eTay'
STIMULUS_ZIP_VERSION_ID = 'OxpOYy_3bveay9NFFFxNCVyghyAbqyIt'

ASSEMBLY_ID = 'Lahner2024-fMRI'
ASSEMBLY_VERSION_ID = 'zr_i3T9Saww44rPNJwLaxo0hgp8rYjPO'
ASSEMBLY_SHA1 = '2c7f1d2e5724b8cc3c5cf47986e956c4f13001e4'


def load_stimulus_set():
    """Load the BOLDMoments video stimulus set (1026 short clips)."""
    return load_stimulus_set_from_s3(
        identifier=STIMULUS_ID,
        bucket=STIMULUS_BUCKET,
        csv_sha1=STIMULUS_CSV_SHA1,
        zip_sha1=STIMULUS_ZIP_SHA1,
        csv_version_id=STIMULUS_CSV_VERSION_ID,
        zip_version_id=STIMULUS_ZIP_VERSION_ID,
    )


def load_assembly(merge_stimulus_set_meta: bool = True):
    """Load the Lahner2024 fMRI assembly."""
    return load_assembly_from_s3(
        identifier=ASSEMBLY_ID,
        version_id=ASSEMBLY_VERSION_ID,
        sha1=ASSEMBLY_SHA1,
        bucket=STIMULUS_BUCKET,
        cls=NeuronRecordingAssembly,
        stimulus_set_loader=load_stimulus_set,
        merge_stimulus_set_meta=merge_stimulus_set_meta,
    )


class Lahner2024BOLDMoments(BenchmarkBase):
    """Naturalistic fMRI benchmark: predict time-resolved BOLD responses to
    short video clips.

    The benchmark is deliberately model-architecture-agnostic:
    - Models that expand videos into timestamped frames (CLIP, Qwen, BLIP-2)
      extract per-frame activations via their existing activations_model,
      and the benchmark applies ``temporal_bin`` after extraction.
    - Models with native video handling can return an already-temporal
      assembly; the benchmark only cares about the output shape.

    Both cases produce a ``(presentation, time_bin, neuroid)`` model
    assembly that gets correlated against the fMRI data.

    Scoring is NOT implemented end-to-end in this first cut — it requires:

    1. A ``frame_extractor`` callable to turn videos into timestamped images
       (OpenCV/ffmpeg — kept out of brainscore_core to avoid the dependency).
    2. A decision about the regression metric (per-voxel ridge vs. PLS) and
       cross-validation scheme for time-resolved neural data.
    3. A stable way to match the benchmark's ``time_bin`` coordinate with
       the fMRI TR grid, including hemodynamic delay correction.

    This class currently loads the data and exposes it. A full ``__call__``
    implementation is the next engineering step (best done on EC2 given the
    1.2 GB total dataset size).
    """

    def __init__(self, time_bin_width_ms: float = 1000.0):
        self._time_bin_width_ms = time_bin_width_ms
        # Lazy: don't hit S3 in __init__; load on first access.
        self._assembly: Optional[NeuronRecordingAssembly] = None
        self._stimulus_set = None

        super().__init__(
            identifier='Lahner2024-fMRI-naturalistic',
            version=1,
            parent='naturalistic',
            ceiling=Score(np.nan),  # TODO: split-half across subjects
            bibtex=BIBTEX,
        )

    @property
    def assembly(self) -> NeuronRecordingAssembly:
        if self._assembly is None:
            self._assembly = load_assembly()
        return self._assembly

    @property
    def stimulus_set(self):
        if self._stimulus_set is None:
            self._stimulus_set = load_stimulus_set()
        return self._stimulus_set

    def __call__(self, candidate) -> Score:
        raise NotImplementedError(
            "Lahner2024 scoring end-to-end is not yet implemented. "
            "This benchmark currently ships as data loaders + scaffolding. "
            "To complete: (1) supply a frame_extractor for video->frame "
            "expansion, (2) decide regression metric + CV scheme for "
            "time-resolved fMRI, (3) compare temporally-binned model "
            "assembly against self.assembly via that metric. Best run on "
            "EC2 — the dataset is ~1.2 GB."
        )
