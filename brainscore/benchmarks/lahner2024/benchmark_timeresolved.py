"""
Lahner 2024 BOLDMoments — TR-resolved fMRI variant (continuous-time encoding).

Same stimuli as the existing GLM-beta variant (`Lahner2024BOLDMoments`), but
predicts the per-TR BOLD time-series of each scanning run rather than a
per-clip GLM beta. M12-lite — the first benchmark that actually exercises
the temporal kit (`temporal_bin`, `hrf_convolve`, `contiguous_block_cv`,
absolute timestamps) end-to-end.

## Design: continuous-time encoding

For each (subject, run), the brain produces a continuous BOLD time-series at
TR=1.75 s across the whole run (~263 TRs covering ~113 stimulus events at
4 s SOA). The model produces per-stimulus features. We HRF-convolve and
place the per-stimulus features at the right TR offsets within run-time,
producing a continuous "expected response" time-series at TR resolution.
Per-voxel ridge regression maps that to the actual BOLD signal, with
leave-one-run-out cross-validation.

This avoids the rapid-event-related design's overlap problem (per-trial
windowing with K>2 TRs has neighboring-trial contamination at SOA=4 s
relative to TR=1.75 s). Continuous-time encoding is the standard
naturalistic-fMRI approach (Huth lab, Friends/Sherlock benchmarks).

## Pipeline (per model)

    1. Discover unique stimuli from events (1102 unique videos).
    2. Extract per-stimulus features ONCE via model.process(unique_stim_set):
       - video-native models: per-clip features at the model's native time bins
         (mean-pooled if per-frame, kept temporal otherwise)
       - frame-aggregation models: per-clip features (mean of N still frames)
    3. For each (subject, run):
       a. Build a feature time-series at TR resolution by placing each clip's
          features at floor(onset_sec / TR) within the run.
       b. HRF-convolve along the time axis using the canonical SPM double-gamma.
       c. Crop / pad to the run's actual TR length.
    4. Concatenate (subject, run) feature time-series into the design matrix X.
       Concatenate (subject, run) BOLD time-series into Y.
    5. Per-voxel ridge regression with leave-one-run-out CV via
       contiguous_block_cv at the run-block level.
    6. Per-voxel Pearson on held-out predictions, median across voxels.

## Comparison story

Same 1026+ videos, same models, but model has to predict TR-resolved
trajectories rather than per-clip betas. Score delta between this variant
and `Lahner2024-fMRI-naturalistic` quantifies how much temporal information
the model adds (or how much the GLM beta already captured).
"""

from typing import List, Optional

import numpy as np
import xarray as xr

from brainscore_core.benchmarks import BenchmarkBase
from brainscore_core.metrics import Score
from brainscore_core.supported_data_standards.brainio.assemblies import (
    NeuronRecordingAssembly,
)
from brainscore_core.supported_data_standards.brainio.s3 import (
    load_assembly_from_s3,
)
from brainscore_core.temporal import (
    contiguous_block_cv,
    double_gamma_hrf,
    hrf_convolve,
)

# Reuse stimulus loader + bibtex + S3 paths from the existing variant
from .benchmark import (
    BIBTEX,
    STIMULUS_BUCKET,
    Lahner2024BOLDMoments,        # parent class — provides _videos_stimulus_set
    load_stimulus_set,
)


# ── S3 metadata for the TR-resolved artifacts ─────────────────────────
# Filled in once `prepare_timeresolved_assembly.py` uploads to S3.
# Until then, loading raises with an explicit handoff message.

TIMERESOLVED_ASSEMBLY_ID = 'Lahner2024-fMRI-timeresolved'
TIMERESOLVED_ASSEMBLY_VERSION_ID: Optional[str] = None    # TODO: paste from prep script
TIMERESOLVED_ASSEMBLY_SHA1: Optional[str]       = None    # TODO: paste from prep script
TIMERESOLVED_EVENTS_VERSION_ID: Optional[str]   = None    # TODO: paste from prep script
TIMERESOLVED_EVENTS_SHA1: Optional[str]         = None    # TODO: paste from prep script

# Confirmed scanner / paradigm parameters (from EC2 recon, ds005165 v1.0.4)
TR_SEC = 1.75
SOA_SEC = 4.0
CLIP_DURATION_SEC = 3.0


# ── Loader ────────────────────────────────────────────────────────────

def load_timeresolved_assembly(merge_stimulus_set_meta: bool = True) -> NeuronRecordingAssembly:
    if TIMERESOLVED_ASSEMBLY_VERSION_ID is None or TIMERESOLVED_ASSEMBLY_SHA1 is None:
        raise RuntimeError(
            "Lahner2024 TR-resolved assembly is not yet hosted on S3. "
            "Run `prepare_timeresolved_assembly.py` on EC2 to download from "
            "OpenNeuro ds005165 v1.0.4, downsample fsaverage→fsaverage5, build "
            "the per-(subject, run) assembly, and upload — then paste the "
            "resulting (version_id, sha1) tuples into benchmark_timeresolved.py."
        )
    return load_assembly_from_s3(
        identifier=TIMERESOLVED_ASSEMBLY_ID,
        version_id=TIMERESOLVED_ASSEMBLY_VERSION_ID,
        sha1=TIMERESOLVED_ASSEMBLY_SHA1,
        bucket=STIMULUS_BUCKET,
        cls=NeuronRecordingAssembly,
        stimulus_set_loader=load_stimulus_set,
        merge_stimulus_set_meta=merge_stimulus_set_meta,
    )


def load_timeresolved_events():
    """Load the per-trial events sidecar CSV (long format).

    Columns: subject, session, run, task, trial_idx, stimulus_id,
             onset_sec, duration_sec, trial_type.
    """
    if TIMERESOLVED_EVENTS_VERSION_ID is None or TIMERESOLVED_EVENTS_SHA1 is None:
        raise RuntimeError(
            "Lahner2024 TR-resolved events sidecar is not yet hosted on S3."
        )
    # TODO on EC2-side prep: implement loader. For now, the same load_assembly_from_s3
    # plumbing won't quite work since this is a CSV not an xarray .nc. Use boto3
    # directly OR add a load_csv_from_s3 helper to brainio.
    raise NotImplementedError(
        "TODO: implement S3 CSV loader. Either use boto3.s3.get_object directly, "
        "or add load_csv_from_s3 to brainscore_core.supported_data_standards.brainio.s3."
    )


# ── Benchmark class ───────────────────────────────────────────────────

class Lahner2024BOLDMoments_timeresolved(BenchmarkBase):
    """Predict per-TR BOLD time-series for each BoldMoments scanning run.

    Inherits stimulus-handling helpers from `Lahner2024BOLDMoments` (the
    GLM-beta variant) — stimuli are identical between the two variants.
    Only the neural target and the regression machinery differ.
    """

    def __init__(
        self,
        ceiling: Optional[float] = None,
        cv_n_held_out_runs: int = 1,
        identifier_suffix: str = '-timeresolved',
    ):
        self._cv_n_held_out_runs = cv_n_held_out_runs
        self._assembly: Optional[NeuronRecordingAssembly] = None
        self._events = None
        self._stimulus_set = None

        # Delegate stimulus prep to the GLM-beta variant — identical stimuli.
        self._stim_helper = Lahner2024BOLDMoments(reliability_threshold=None)

        super().__init__(
            identifier=f'Lahner2024-fMRI-naturalistic{identifier_suffix}',
            version=1,
            parent='naturalistic',
            ceiling=Score(1.0 if ceiling is None else float(ceiling)),
            bibtex=BIBTEX,
        )

    @property
    def assembly(self) -> NeuronRecordingAssembly:
        if self._assembly is None:
            self._assembly = load_timeresolved_assembly()
            self._sanity_check_assembly(self._assembly)
        return self._assembly

    @property
    def events(self):
        if self._events is None:
            self._events = load_timeresolved_events()
        return self._events

    @property
    def stimulus_set(self):
        return self._stim_helper.stimulus_set

    def _sanity_check_assembly(self, assembly):
        assert 'time_bin_start_ms' in assembly.coords
        assert 'subject' in assembly.coords
        assert 'run' in assembly.coords
        assert 'n_valid_TR' in assembly.coords
        assert assembly.sizes['neuroid'] == 20484
        assert assembly.sizes['time_bin'] > 1, \
            "TR-resolved assembly must have time_bin > 1; got the GLM-beta variant by mistake?"

    def __call__(self, candidate) -> Score:
        # 1. Get per-stimulus model features (one feature vector per unique stimulus).
        per_stim_features, stimulus_ids = self._extract_per_stimulus_features(candidate)
        # per_stim_features shape: (n_stimuli, n_features)
        # stimulus_ids: list of length n_stimuli

        stim_idx = {sid: i for i, sid in enumerate(stimulus_ids)}

        # 2. For each (subject, run) presentation: build feature time-series at TR
        #    resolution by placing per-stim features at onset TRs, then HRF-convolve.
        assembly = self.assembly
        n_TR_max = assembly.sizes['time_bin']
        n_features = per_stim_features.shape[1]
        n_runs = assembly.sizes['presentation']

        feature_ts = np.zeros((n_runs, n_TR_max, n_features), dtype=np.float32)
        for run_idx in range(n_runs):
            run_id = (str(assembly['subject'].values[run_idx]),
                      str(assembly['session'].values[run_idx]),
                      str(assembly['run'].values[run_idx]))
            run_events = self._events_for_run(*run_id)
            for _, ev in run_events.iterrows():
                if ev['stimulus_id'] not in stim_idx:
                    continue   # oddball or stimulus the model couldn't process
                onset_TR = int(np.floor(ev['onset_sec'] / TR_SEC))
                if 0 <= onset_TR < n_TR_max:
                    feature_ts[run_idx, onset_TR, :] += per_stim_features[stim_idx[ev['stimulus_id']]]

        # HRF-convolve along time axis. hrf_convolve expects (n_time, n_features).
        hrf_kernel = double_gamma_hrf(duration_sec=32.0, sampling_rate_hz=1.0/TR_SEC)
        feature_ts_convolved = np.zeros_like(feature_ts)
        for run_idx in range(n_runs):
            feature_ts_convolved[run_idx] = hrf_convolve(
                feature_ts[run_idx], sampling_rate_hz=1.0/TR_SEC, hrf=hrf_kernel)

        # 3. Concatenate (run, TR) → flat (n_obs, n_features); same for BOLD.
        #    Mask padded entries via n_valid_TR.
        X_flat, Y_flat, run_idx_per_obs = self._concatenate_with_mask(
            feature_ts_convolved, assembly)

        # 4. Per-voxel ridge regression with leave-one-run-out CV.
        per_voxel_r = self._cross_validated_ridge(
            X_flat, Y_flat, run_idx_per_obs,
            n_held_out=self._cv_n_held_out_runs,
        )

        # 5. Aggregate.
        per_voxel_r_finite = per_voxel_r[~np.isnan(per_voxel_r)]
        median_r = float(np.median(per_voxel_r_finite))
        mean_r = float(np.mean(per_voxel_r_finite))

        score = Score(median_r / float(self.ceiling))
        score.attrs['raw'] = Score(median_r)
        score.attrs['mean_r'] = mean_r
        score.attrs['n_voxels_scored'] = int(len(per_voxel_r_finite))
        score.attrs['n_runs'] = n_runs
        score.attrs['n_observations'] = int(len(Y_flat))
        score.attrs['cv_n_held_out_runs'] = self._cv_n_held_out_runs
        score.attrs['pipeline'] = 'continuous_time_encoding'
        return score

    # ── Helpers (each TODO has detailed instructions) ──────────────

    def _extract_per_stimulus_features(self, candidate):
        """One forward pass per unique stimulus → (n_stimuli, n_features).

        TODO:
            - Build a unique-stimulus set from events (1102 unique videos).
            - Dispatch on candidate.supported_modalities:
                * 'video' in supported  → use _videos_stimulus_set, call process,
                  mean-pool over time_bin axis to get per-stimulus features
                * else                  → use _expand_videos + per-frame extraction +
                  temporal_bin to single bin = mean over frames
            - Return (features (n_stimuli, n_features), stimulus_ids list)
        """
        raise NotImplementedError(
            "TODO: extract per-stimulus features once. See docstring."
        )

    def _events_for_run(self, subject, session, run):
        """Filter the events DataFrame to one (subject, session, run)."""
        ev = self.events
        mask = ((ev['subject'] == subject) &
                (ev['session'] == session) &
                (ev['run'] == run))
        return ev[mask]

    def _concatenate_with_mask(self, feature_ts_convolved, assembly):
        """Flatten (run, TR) → (n_obs,) for valid TRs only.

        TODO:
            - For each presentation idx, take feature_ts_convolved[idx, :n_valid_TR, :]
              and assembly[:n_valid_TR, :, idx].T  → (n_valid_TR, n_voxels)
            - Stack across runs into X_flat (n_obs, n_features) and Y_flat (n_obs, n_voxels)
            - Track run_idx_per_obs for CV.
        """
        raise NotImplementedError("TODO: flatten + mask. See docstring.")

    def _cross_validated_ridge(self, X_flat, Y_flat, run_idx_per_obs,
                               n_held_out: int):
        """Leave-K-runs-out ridge per voxel; return (n_voxels,) Pearson array.

        TODO:
            - Use contiguous_block_cv at the RUN level (not the obs level).
              Iterate held-out RUN indices; build train/test masks via run_idx_per_obs.
            - For each fold:
                X_train, X_test, Y_train, Y_test
                Ridge(alpha=1.0).fit(X_train, Y_train).predict(X_test) → fold_preds
                Accumulate per-voxel held-out predictions
            - After all folds: per-voxel Pearson(Y_flat, accumulated_preds).
        """
        raise NotImplementedError("TODO: leave-K-runs-out ridge per voxel. See docstring.")
