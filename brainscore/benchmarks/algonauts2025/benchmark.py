"""Algonauts 2025 challenge benchmark classes.

SCAFFOLD ONLY. Data download + assembly preparation happen on EC2 first
(see README.md). __call__ raises until the assembly is loadable.

Three subclasses share most logic:
- Algonauts2025Friends: train/test on Friends S1-S6 + Movie10 (in-dist)
- Algonauts2025FriendsS7: held-out S7 — predictions only, no fMRI ground truth
- Algonauts2025OOD: held-out 2 h of OOD movies — predictions only

For S7 and OOD, the benchmark's __call__ doesn't compute Pearson r —
it returns predicted-parcel arrays formatted for Codabench submission.

Pipeline (per subject):
1. For each TR in the assembly, gather the candidate model's three
   modality feature vectors (video, audio, transcript) for that TR.
2. Stack across the stimulus_window TRs (default 5) to capture local
   temporal context.
3. HRF-shift by hrf_delay TRs.
4. Banded ridge with per-modality α tuned via inner CV.
5. Per-parcel Pearson r on held-out movies. Median across parcels for
   the headline score; also report per-network-mean.

Inherits from BenchmarkBase; uses the same wrappers and banded-ridge
implementation as the Lahner multimodal benchmarks.
"""
from pathlib import Path
from typing import Optional

import numpy as np

from brainscore_core.benchmarks import BenchmarkBase
from brainscore_core.metrics import Score


BIBTEX = """@article{gifford2025algonauts,
  title={The Algonauts Project 2025 Challenge: How the Human Brain Makes Sense of Multimodal Movies},
  author={Gifford, Alessandro T and Bersch, Domenic and St-Laurent, Marie and Pinsard, Basile
          and Boyle, Julie and Bellec, Pierre and Oliva, Aude and Roig, Gemma and Cichy, Radoslaw M},
  journal={arXiv preprint arXiv:2501.00504},
  year={2025},
}"""

# Schaefer 1000 parcellation, TR for Courtois NeuroMod
SCHAEFER_N_PARCELS = 1000
TR_SEC = 1.49

# Default fMRI assembly cache root (set by prepare_assembly.py).
# Override via the ALGONAUTS_DATA_ROOT environment variable.
DEFAULT_ASSEMBLY_ROOT = Path('~/.brainio/algonauts2025').expanduser()


class _Algonauts2025Base(BenchmarkBase):
    """Shared logic across the three splits."""

    VALID_MODES = ('concat', 'per_modality', 'banded',
                   'video_only', 'audio_only', 'language_only')
    BANDED_ALPHA_GRID = (1.0, 10.0, 100.0, 1000.0, 10000.0)

    def __init__(
        self,
        subject: int,
        split: str,
        identifier_suffix: str,
        stimulus_window: int = 5,
        hrf_delay: int = 3,
        excluded_samples_start: int = 5,
        excluded_samples_end: int = 5,
        mode: str = 'banded',
        assembly_root: Optional[Path] = None,
    ):
        if subject not in (1, 2, 3, 5):
            raise ValueError(
                f"Algonauts subjects are {{1, 2, 3, 5}}; got {subject}.")
        if mode not in self.VALID_MODES:
            raise ValueError(
                f"mode must be one of {self.VALID_MODES}; got {mode!r}.")

        self._subject = subject
        self._split = split
        self._stimulus_window = stimulus_window
        self._hrf_delay = hrf_delay
        self._excluded_samples_start = excluded_samples_start
        self._excluded_samples_end = excluded_samples_end
        self._mode = mode
        self._assembly_root = (Path(assembly_root)
                               if assembly_root else DEFAULT_ASSEMBLY_ROOT)
        self._assembly = None
        self._stimulus_set = None

        super().__init__(
            identifier=(f'Algonauts2025-{split}-sub{subject:02d}'
                        f'{identifier_suffix}'),
            version=1,
            parent='naturalistic',
            ceiling=Score(1.0),
            bibtex=BIBTEX,
        )

    @property
    def assembly(self):
        if self._assembly is None:
            self._assembly = self._load_assembly()
        return self._assembly

    @property
    def stimulus_set(self):
        if self._stimulus_set is None:
            self._stimulus_set = self._load_stimulus_set()
        return self._stimulus_set

    def _load_assembly(self):
        """Load this subject's per-TR Schaefer parcels for this split.

        Expects prepare_assembly.py to have produced a netCDF at
        ``{assembly_root}/algonauts2025_{split}_sub{subject:02d}.nc``.
        Raises FileNotFoundError if the data hasn't been prepared yet.
        """
        path = (self._assembly_root /
                f'algonauts2025_{self._split}_sub{self._subject:02d}.nc')
        if not path.exists():
            raise FileNotFoundError(
                f"Algonauts assembly missing at {path}. Run "
                f"`python -m brainscore.benchmarks.algonauts2025.prepare_assembly` "
                f"on EC2 first; see README.md."
            )
        import xarray as xr
        from brainscore_core.supported_data_standards.brainio.assemblies import (
            NeuronRecordingAssembly)
        data = xr.open_dataarray(str(path))
        return NeuronRecordingAssembly(data)

    def _load_stimulus_set(self):
        """Load the StimulusSet for this split. CSV with movie/episode/split
        rows, plus paths to .mkv stimuli + per-TR transcripts."""
        import pandas as pd
        from brainscore_core.supported_data_standards.brainio.stimuli import (
            StimulusSet)
        path = (self._assembly_root /
                f'algonauts2025_stim_{self._split}.csv')
        if not path.exists():
            raise FileNotFoundError(
                f"Algonauts stim_set missing at {path}. Run "
                f"prepare_assembly.py on EC2 first.")
        df = pd.read_csv(path)
        out = StimulusSet(df)
        out.identifier = f'algonauts2025-{self._split}'
        out.stimulus_paths = dict(
            zip(df['stimulus_id'], df['video_path']))
        return out

    def __call__(self, candidate) -> Score:
        raise NotImplementedError(
            "Algonauts2025 benchmark scoring not yet implemented. "
            "Phase plan in README.md: download data → prepare assembly → "
            "score on EC2 → submit to Codabench."
        )


class Algonauts2025Friends(_Algonauts2025Base):
    """Train + score per-parcel encoding model on Friends S1-S6 +
    Movie10 (in-distribution). Cross-validated within each subject.

    Score is the median per-parcel Pearson r on held-out folds.
    """

    def __init__(self, subject: int, **kwargs):
        super().__init__(
            subject=subject, split='friends',
            identifier_suffix='', **kwargs)


class Algonauts2025FriendsS7(_Algonauts2025Base):
    """Held-out Friends Season 7 — Codabench leaderboard target.

    fMRI ground truth is withheld. __call__ returns predicted
    per-parcel time series in the format Codabench expects rather than
    a Pearson-r score.
    """

    def __init__(self, subject: int, **kwargs):
        super().__init__(
            subject=subject, split='friends_s7',
            identifier_suffix='-test', **kwargs)


class Algonauts2025OOD(_Algonauts2025Base):
    """Held-out OOD movies — Codabench OOD leaderboard target.

    Same shape as FriendsS7: returns predicted per-parcel time series
    rather than a Pearson-r score.
    """

    def __init__(self, subject: int, **kwargs):
        super().__init__(
            subject=subject, split='ood',
            identifier_suffix='-ood', **kwargs)
