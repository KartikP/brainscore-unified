"""Compose a passive UMI experiment and analyze its saved neural assembly."""

from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from brainscore.experiments import Experiment, RecordNeural, TaskProtocol
from brainscore_core import TaskContext

from .data import KEYS, PreparedStudy, write_json
from .geometry import curvature


def make_experiment(subject, study: PreparedStudy, output: str | Path, *,
                    region: str = 'V1', seed: int = 0, repetitions: int = 1,
                    provenance: dict | None = None) -> Experiment:
    """One session per independent repetition, with 400 shuffled image trials.

    Each repetition uses the same seeded order. The protocol owns both resets.
    Natural/artificial and scale are within-session stimulus factors. Reusing
    shared endpoint responses assumes a static image representation.
    """
    if not isinstance(repetitions, int) or repetitions < 1:
        raise ValueError('repetitions must be a positive integer')
    protocol = TaskProtocol(
        study.stimuli(seed), TaskContext(task_type='passive'),
        identifier='Henaff2021-static-frames-v1', repetitions=range(repetitions),
        metadata={'study': study.provenance, 'shuffle_seed': seed,
                  'representation': 'raw recorded features; Euclidean geometry',
                  'scope': 'static-frame adaptation; no timed or stateful acquisition replay'})
    return Experiment(subject=subject, protocol=protocol, tools=[RecordNeural([region])],
                      output_dir=output, metadata={'execution_provenance': dict(provenance or {})})


def model_curvatures(recording, study: PreparedStudy) -> pd.DataFrame:
    """Join by stimulus identity and reconstruct movie order after viewing."""
    for dim in set(recording.dims) - {'presentation', 'neuroid'}:
        if recording.sizes[dim] != 1:
            raise ValueError(f'Choose a scientific aggregation for non-singleton {dim!r}')
        recording = recording.squeeze(dim, drop=True)
    recording = recording.transpose('presentation', 'neuroid')
    ids = list(recording['stimulus_id'].values.astype(str))
    expected = set(study.frames.stimulus_id)
    if len(ids) != len(set(ids)) or set(ids) != expected:
        raise ValueError('Recording must contain every prepared stimulus exactly once')
    positions = {name: i for i, name in enumerate(ids)}
    rows = []
    for (category, scale, movie), frames in study.frames.groupby(['category', 'scale', 'movie']):
        frames = frames.sort_values('frame')
        if list(frames.frame) != list(range(1, 12)):
            raise ValueError('Each trajectory must contain frames 1..11 exactly once')
        values = recording.values[[positions[name] for name in frames.stimulus_id]]
        for stride in (1, 2):
            try:
                value, status, reason = curvature(values, stride), 'valid', ''
            except ValueError as error:
                value, status, reason = np.nan, 'invalid', str(error)
            rows.append(dict(category=category, scale=scale, movie=movie, stride=stride,
                             model_curvature_deg=value, status=status, reason=reason))
    result = pd.DataFrame(rows).merge(study.pixels, on=KEYS, validate='one_to_one')
    result['model_relative_deg'] = result.model_curvature_deg - result.presented_png_pixel_deg
    return result


def _correlation(x, y) -> float | None:
    if len(x) < 3 or not np.isfinite(x).all() or not np.isfinite(y).all():
        return None
    if np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def compare_to_macaque(model: pd.DataFrame, reference: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Descriptive agreement across 20 movie/scale pairs per category.

    Macaque relative curvature uses the archived noise-matched null correction.
    Model relative curvature uses its PNG pixel baseline and raw feature metric.
    These are explicitly different estimators; this is not a ceiling-normalized
    Brain-Score benchmark or an exact reproduction of the fitted LN-LN model.
    """
    if model.duplicated(KEYS).any() or reference.duplicated(['recording', *KEYS]).any():
        raise ValueError('Duplicate trajectory keys')
    retained = reference[reference.included]
    targets = retained.groupby(KEYS).agg(
        macaque_relative_deg=('macaque_relative_deg', 'mean'),
        macaque_population_sem_deg=('macaque_relative_deg', 'sem'),
        n_populations=('recording', 'size'),
        archived_lnln_relative_deg=('archived_lnln_relative_deg', 'mean'),
        archived_inference_pixel_deg=('archived_pixel_deg', 'mean'),
        archived_lnln_pixel_deg=('archived_lnln_pixel_deg', 'mean')).reset_index()
    paired = model.merge(targets, on=KEYS, how='outer', validate='one_to_one', indicator=True)
    if not (paired['_merge'] == 'both').all():
        raise ValueError('Model and macaque reference must cover the same trajectory keys')
    paired = paired.drop(columns='_merge')
    summaries = []
    for (category, stride), group in paired.groupby(['category', 'stride']):
        valid = (group.status == 'valid') & np.isfinite(group.model_relative_deg)
        # Missing/invalid trajectories make the planned whole-category comparison N/A.
        complete = bool(valid.all())
        x, y = group.model_relative_deg.to_numpy(), group.macaque_relative_deg.to_numpy()
        source = retained[(retained.category == category) & (retained.stride == stride)]
        summaries.append(dict(
            category=category, stride=int(stride), n_movie_scale_pairs=len(group),
            n_valid_model_pairs=int(valid.sum()), n_macaque_trajectories=len(source),
            macaque_trajectory_median_deg=float(source.macaque_relative_deg.median()),
            macaque_movie_scale_median_deg=float(np.median(y)),
            model_movie_scale_median_deg=float(np.median(x)) if complete else None,
            pearson_r=_correlation(x, y) if complete else None,
            mean_absolute_difference_deg=float(np.mean(np.abs(x - y))) if complete else None,
            archived_lnln_pearson_r=_correlation(group.archived_lnln_relative_deg.to_numpy(), y),
            status='complete' if complete else 'N/A: invalid model trajectories',
        ))
    return paired, dict(
        analysis='descriptive comparison, not a registered benchmark score',
        primary_stride=2, sign='negative relative curvature means straightening',
        model_estimator='Euclidean raw-feature curvature minus exported-PNG pixel curvature',
        macaque_estimator='archived Delta Curvature: neural minus Delta Curvature: control',
        weighting='equal movies/scales after averaging retained populations per movie/scale',
        uncertainty='population SEM is descriptive; no independence across shared monkeys assumed',
        controls={'pixel_identity_relative_deg': 0.0,
                  'constant_representation': 'undefined curvature; must be N/A',
                  'archived_lnln': 'provided fitted model, not refitted here'},
        comparisons=summaries)


def analyze(recording, study: PreparedStudy, output: str | Path, *,
            provenance: dict | None = None) -> dict:
    """Save trajectory-level results and separate natural/artificial comparisons."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    model = model_curvatures(recording, study)
    paired, summary = compare_to_macaque(model, study.reference)
    paired.to_csv(output / 'trajectories.csv', index=False)
    study.reference.to_csv(output / 'macaque_inclusion.csv', index=False)
    summary['prepared_id'] = study.provenance['prepared_id']
    summary['prepared_source_sha256'] = study.provenance['source_sha256']
    summary['versions'] = {name: version(name) for name in
                           ['brainscore', 'brainscore_core', 'numpy', 'pandas', 'scipy']}
    summary['recording_dtype'] = str(recording.dtype)
    summary['execution_provenance'] = dict(provenance or {})
    summary['pixel_diagnostics'] = {
        'max_png_vs_archive_float_deg': float(np.max(np.abs(
            paired.presented_png_pixel_deg - paired.archive_float_pixel_deg))),
        'max_archive_float_vs_lnln_baseline_deg': float(np.max(np.abs(
            paired.archive_float_pixel_deg - paired.archived_lnln_pixel_deg))),
        'max_archive_float_vs_inference_csv_baseline_deg': float(np.max(np.abs(
            paired.archive_float_pixel_deg - paired.archived_inference_pixel_deg))),
        'interpretation': 'The archive contains different pixel baselines. Their cause is unresolved. '
                          'Keep them separate; do not call this an exact acquisition replication.'}
    write_json(output / 'summary.json', summary)
    return summary
