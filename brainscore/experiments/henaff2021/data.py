"""Read the VF2XK archive without MATLAB or changes to its source files."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.io import loadmat

from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet

from .geometry import curvature


SOURCE = 'https://doi.org/10.17605/OSF.IO/VF2XK'
CATEGORIES = {1: 'natural', 2: 'artificial'}
CSV_PATH = 'curvatureEstimations/Acute data - combined_235x.csv'
MODEL_PATH = 'modelResults/aggregated_LNLN_nat_sigG2_sqrt.mat'
# plot_summary.m deletes position 2, then position 7 of the shortened vector:
# these are original recording IDs 2 and 8, not IDs 2 and 7.
EXCLUDED_RECORDINGS = ('ay2_u007_image_sequences', 'ay5_u002_image_sequences')
KEYS = ['category', 'scale', 'movie', 'stride']


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


@dataclass(frozen=True)
class PreparedStudy:
    directory: Path
    frames: pd.DataFrame
    pixels: pd.DataFrame
    reference: pd.DataFrame
    provenance: dict

    def stimuli(self, seed: int = 0) -> StimulusSet:
        """One shuffled presentation per distinct source frame (400 images)."""
        rows = self.frames.drop_duplicates('stimulus_id').copy()
        rows = rows.iloc[np.random.default_rng(seed).permutation(len(rows))].reset_index(drop=True)
        rows['image_path'] = rows['image_file'].map(lambda p: str(self.directory / p))
        rows['presentation_id'] = np.arange(len(rows))
        stimuli = StimulusSet(rows)
        stimuli.identifier = f'henaff2021-{self.provenance["prepared_id"]}-seed{seed}'
        stimuli.stimulus_paths = dict(zip(stimuli.stimulus_id, stimuli.image_path))
        return stimuli


def load_prepared(directory: str | Path) -> PreparedStudy:
    directory = Path(directory).expanduser().resolve()
    provenance = json.loads((directory / 'manifest.json').read_text())
    for name, expected in provenance['prepared_files'].items():
        if sha256(directory / name) != expected:
            raise ValueError(f'Prepared artifact changed: {name}')
    frames = pd.read_csv(directory / 'frames.csv')
    for row in frames.drop_duplicates('stimulus_id').itertuples():
        if sha256(directory / row.image_file) != row.image_sha256:
            raise ValueError(f'Prepared image changed: {row.image_file}')
    return PreparedStudy(directory, frames, pd.read_csv(directory / 'pixels.csv'),
                         pd.read_csv(directory / 'macaque.csv'), provenance)


def read_reference(archive: str | Path) -> pd.DataFrame:
    """Archived noise-corrected estimates; this does not refit spike trains.

    The source CSV omits movie IDs. summary.m assigns them by original row
    order within each recording/category/dither group. Preserve that mapping
    through the CSV's explicit Index column, even if physical rows are shuffled.
    """
    archive = Path(archive)
    raw = pd.read_csv(archive / CSV_PATH).sort_values(
        ['sequence type', 'dither', 'Index']).reset_index(drop=True)
    blocks = raw.groupby(['sequence type', 'dither'])
    if len(blocks) != 4 or any(not np.array_equal(block['Index'], np.arange(1, 181))
                               for _, block in blocks):
        raise ValueError('Expected source Index values 1..180 once per category/dither')
    model = loadmat(archive / MODEL_PATH, simplify_cells=True)
    names = [f'{m}_{p}_image_sequences' for m, p in
             zip(model['monkey_id_list'], model['penetration_id_list'])]
    groups = raw.groupby(['Dataset', 'sequence type', 'dither'], sort=False)
    if len(groups) != 36 or not (groups.size() == 20).all():
        raise ValueError('Expected 9 recordings x 2 categories x 2 strides x 20 sequences')
    rows = []
    for (recording, category, stride), group in groups:
        r = names.index(recording)
        if category not in CATEGORIES or stride not in (1, 2):
            raise ValueError('Unexpected source category or dither')
        prefix = 'natural' if category == 1 else 'artificial'
        labels = model[f'{prefix}_movie_labels']
        for position, (_, row) in enumerate(group.iterrows()):
            scale, movie = divmod(position, 10)
            reasons = []
            if recording in EXCLUDED_RECORDINGS:
                reasons.append('hardware_failure')
            for name, value, threshold in [
                ('distance', row['Distances'], .25),
                ('mean_r2', row['Rates, correlation empirical vs predicted'] ** 2, .75),
                ('variance_r2', row['Variance, correlation empirical vs predicted'] ** 2, .5),
                ('covariance_r2', row['Covariance, correlation empirical vs predicted'] ** 2, .25),
            ]:
                if not np.isfinite(value) or value <= threshold:
                    reasons.append(name)
            neural = row['Delta Curvature: neural']
            control = row['Delta Curvature: control']
            if not np.isfinite([neural, control]).all():
                reasons.append('missing_curvature')
            index = (r, stride - 1, scale, category - 1, movie)
            rows.append(dict(
                source_index=int(row['Index']), recording=recording,
                category=CATEGORIES[category], scale=scale + 1, movie=movie + 1,
                sequence=str(labels[position]), stride=stride,
                macaque_relative_deg=neural - control,
                macaque_delta_pixel_deg=neural, macaque_null_delta_deg=control,
                archived_pixel_deg=row['Pixel curvature'],
                archived_lnln_relative_deg=float(np.degrees(
                    model['c_V1_model'][index] - model['c_V1_pixel'][index])),
                archived_lnln_pixel_deg=float(np.degrees(model['c_V1_pixel'][index])),
                included=not reasons, exclusion_reason=';'.join(reasons),
                distance=row['Distances'],
                mean_r=row['Rates, correlation empirical vs predicted'],
                variance_r=row['Variance, correlation empirical vs predicted'],
                covariance_r=row['Covariance, correlation empirical vs predicted']))
    return pd.DataFrame(rows)


def prepare(archive: str | Path, output: str | Path) -> PreparedStudy:
    """Export the archived unblurred matrix as explicit 8-bit grayscale PNGs.

    Slight interpolation overshoot is clipped to [0,1], then rounded to 255
    levels. Both original-float and exported-PNG pixel curvatures are saved.
    No per-frame normalization, resizing, or artificial-sequence regeneration.
    Run remotely: MATLAB decompression needs several GB of working memory.
    """
    archive, output = Path(archive).expanduser(), Path(output).expanduser()
    if output.exists():
        raise FileExistsError(f'Use a new prepared directory: {output}')
    reference = read_reference(archive)
    data = loadmat(archive / 'stimulus/stim_matrix.mat',
                   variable_names=['stim_matrix', 'natural_movie_labels', 'artificial_movie_labels'],
                   simplify_cells=True)
    matrix = data['stim_matrix']
    if matrix.shape != (2, 3, 10, 512, 512, 11):
        raise ValueError(f'Unexpected stimulus shape {matrix.shape}')
    relevant = matrix[:, :2]
    if not np.isfinite(relevant).all() or relevant.min() < -.01 or relevant.max() > 1.01:
        raise ValueError('Unexpected pixel range; do not infer a new intensity scale')
    for frame in (0, 10):
        if not np.array_equal(matrix[:, 0, :, :, :, frame], matrix[:, 1, :, :, :, frame]):
            raise ValueError('Natural and artificial endpoints must match')
    output.mkdir(parents=True)
    (output / 'images').mkdir()
    frames, pixels, saved = [], [], {}
    for category_id, category in CATEGORIES.items():
        for scale in (1, 2):
            for movie in range(1, 11):
                sequence = str(data[f'{category}_movie_labels'][(scale - 1) * 10 + movie - 1])
                x = np.moveaxis(matrix[scale - 1, category_id - 1, movie - 1], -1, 0)
                png = np.rint(np.clip(x, 0, 1) * 255).astype('uint8')
                for frame in range(11):
                    owner = 'natural' if frame in (0, 10) else category
                    stimulus_id = f'{owner}-s{scale}-m{movie:02d}-f{frame + 1:02d}'
                    filename = f'images/{stimulus_id}.png'
                    if stimulus_id not in saved:
                        Image.fromarray(png[frame]).save(output / filename)
                        saved[stimulus_id] = sha256(output / filename)
                    frames.append(dict(stimulus_id=stimulus_id, sequence=sequence, category=category,
                                       scale=scale, movie=movie, frame=frame + 1,
                                       image_file=filename, image_sha256=saved[stimulus_id]))
                for stride in (1, 2):
                    pixels.append(dict(category=category, scale=scale, movie=movie, stride=stride,
                                       archive_float_pixel_deg=curvature(x, stride),
                                       presented_png_pixel_deg=curvature(png, stride)))
    pd.DataFrame(frames).to_csv(output / 'frames.csv', index=False)
    pd.DataFrame(pixels).to_csv(output / 'pixels.csv', index=False)
    reference.to_csv(output / 'macaque.csv', index=False)
    source_names = ['stimulus/stim_matrix.mat', CSV_PATH, MODEL_PATH, 'summary.m', 'plot_summary.m']
    sources = {name: sha256(archive / name) for name in source_names}
    provenance = dict(version=1, source=SOURCE, source_sha256=sources,
                      prepared_id=sources['stimulus/stim_matrix.mat'][:16],
                      image_conversion='round(255 * clip(stim_matrix, 0, 1)), uint8 grayscale PNG',
                      matrix_key='stim_matrix', source_min=float(relevant.min()),
                      source_max=float(relevant.max()),
                      source_out_of_range_pixels=int(np.sum((relevant < 0) | (relevant > 1))),
                      unique_images=len(saved), sequence_frame_rows=len(frames),
                      source_on_ms=200, source_blank_ms=100, source_acquisition_repeats=50,
                      presentation='static frames; no physical clock or blank-screen dynamics',
                      excluded_recordings=list(EXCLUDED_RECORDINGS),
                      prepared_files={name: sha256(output / name) for name in
                                      ['frames.csv', 'pixels.csv', 'macaque.csv']})
    write_json(output / 'manifest.json', provenance)
    return load_prepared(output)


def load_spike_counts(archive: str | Path, recording: str) -> NeuroidAssembly:
    """Raw counts: 440 sequence frames x units x 50 acquisition repeats.

    Shared endpoints appear twice because this assembly describes trajectories.
    Acquisition repeats are not independent UMI sessions. NaNs remain missing.
    """
    import h5py
    archive = Path(archive)
    allowed = {p.stem for p in (archive / 'V1_responses').glob('*_image_sequences.mat')}
    if recording not in allowed:
        raise ValueError(f'Unknown recording {recording!r}')
    path = archive / 'V1_responses' / f'{recording}.mat'
    with h5py.File(path) as f:
        g = f['naturalStruct']
        # HDF5 reverses the MATLAB order: repeat, unit, frame, movie, category, scale.
        raw = g['sortedResponses'][()]
        if raw.shape[0] != 50 or raw.shape[2:] != (11, 10, 3, 2):
            raise ValueError(f'Unexpected spike-count shape {raw.shape}')
        values = raw[:, :, :, :, :2, :].transpose(5, 4, 3, 2, 1, 0).reshape(440, raw.shape[1], 50)
        labels = [(s, c, m, frame) for s in (1, 2) for c in (1, 2)
                  for m in range(1, 11) for frame in range(1, 12)]
        ids = [f'{"natural" if frame in (1, 11) else CATEGORIES[c]}-s{s}-m{m:02d}-f{frame:02d}'
               for s, c, m, frame in labels]
        return NeuroidAssembly(values, dims=['presentation', 'neuroid', 'acquisition_repeat'], coords={
            'stimulus_id': ('presentation', ids), 'presentation_id': ('presentation', np.arange(440)),
            'scale': ('presentation', [t[0] for t in labels]),
            'category': ('presentation', [CATEGORIES[t[1]] for t in labels]),
            'movie': ('presentation', [t[2] for t in labels]),
            'frame': ('presentation', [t[3] for t in labels]),
            'neuroid_id': ('neuroid', [f'{recording}.{i}' for i in g['cluster_list'][()].ravel()]),
            'region': ('neuroid', ['V1'] * raw.shape[1]),
            'acquisition_repeat': np.arange(50),
        }, attrs={'source': SOURCE, 'recording': recording, 'units': 'spike counts',
                  'source_sha256': sha256(path), 'integration_period_sec': float(g['integrationPeriod'][0, 0]),
                  'analysis': 'raw counts; not the paper noise-corrected curvature'})
