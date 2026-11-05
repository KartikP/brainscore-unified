"""Build lossless, probe-centered clips and human vectors from pinned sources."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .fetch import GITHUB_REVISION, HF_REVISION

DISPLAY = {'fps': 30, 'frames': 16, 'width': 600, 'height': 600,
           'source_scale': 1.5, 'aperture_radius': 300, 'probe_frames': list(range(5, 11)),
           'probe_diameter': 15, 'marker_diameter': 5, 'marker_offset': 59,
           'codec': 'FFV1', 'container': 'avi', 'renderer': 'OpenCV linear interpolation',
           'scope': 'raster reconstruction of target interval; no monitor calibration or matching loop'}


def checksum(path: Path) -> str:
    value = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024**2), b''):
            value.update(block)
    return value.hexdigest()


def reference_rows(table: pd.DataFrame, participants: list[pd.DataFrame], scenes: list[int]) -> list[dict]:
    """Validate scene identities independently before attaching the published row order."""
    base = table.groupby('MovI').first().sort_index()
    if list(base.index) != list(range(1, 41)):
        raise ValueError('Expected trial metadata for scenes 1..40')
    expected = base[['flowU', 'flowV']].to_numpy() * 1.5 / np.sqrt(2)
    rows = []
    if len(participants) != 3:
        raise ValueError('Expected all three participant tables')
    for participant, data in enumerate(participants, 1):
        if len(data) != 40 or not np.allclose(data[['Avg_gt_X', 'Avg_gt_Y']], expected, atol=2e-6, rtol=0):
            raise ValueError(f'Participant {participant}: scene ordering or archived GT does not match')
        for scene in scenes:
            for modulation in range(1, 8):
                row = data.iloc[scene - 1]
                x, y = (float(row[f'Mod{modulation}_avg_res_{axis}']) for axis in ('X', 'Y'))
                if not np.isfinite([x, y]).all():
                    raise ValueError('Nonfinite human response')
                rows.append(dict(base_id=f'sun2025-s{scene:02d}-m{modulation}-v1', scene=scene,
                                 modulation=modulation, participant=participant, x=x, y=y,
                                 archived_gt_x=float(row.Avg_gt_X), archived_gt_y=float(row.Avg_gt_Y),
                                 source_flow_x=float(base.loc[scene, 'flowU']),
                                 source_flow_y=float(base.loc[scene, 'flowV'])))
    return rows


def render_clip(frames: list[np.ndarray], row, output: Path) -> None:
    """Approximate the published target display in an explicitly versioned raster."""
    import cv2
    if len(frames) != 16 or any(frame.shape != (1024, 1024, 3) for frame in frames):
        raise ValueError('Expected sixteen 1024 by 1024 RGB source frames')
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*'FFV1'), 30, (600, 600))
    if not writer.isOpened():
        raise RuntimeError('OpenCV cannot write lossless FFV1 video')
    yy, xx = np.mgrid[:600, :600]
    aperture = (xx - 299.5)**2 + (yy - 299.5)**2 <= 300**2
    # PsychoPy's default signed RGB colors: (R,G,B) is mapped from [-1,1] to bytes.
    probe_color = tuple(int(round((float(getattr(row, key)) + 1) * 127.5)) for key in ('R', 'G', 'B'))
    opacities = np.sin(np.linspace(.02, np.pi - .02, 16))
    try:
        for index, frame in enumerate(frames):
            enlarged = cv2.resize(frame, (1536, 1536), interpolation=cv2.INTER_LINEAR)
            # Some source probes are near an image edge. The human display shows
            # its gray background outside the image, rather than repeated edge pixels.
            padded = cv2.copyMakeBorder(enlarged, 300, 300, 300, 300,
                                        cv2.BORDER_CONSTANT, value=(128, 128, 128))
            crop = cv2.getRectSubPix(padded, (600, 600),
                                     (float(row.picX) * 1.5 + 300, float(row.picY) * 1.5 + 300))
            if int(row.fliptype) in (2, 4):
                crop = crop[:, ::-1]
            if int(row.fliptype) in (3, 4):
                crop = crop[::-1]
            shown = np.clip(np.rint(127.5 + opacities[index] * (crop.astype(float) - 127.5)), 0, 255).astype('uint8')
            shown[~aperture] = 128
            for x, y in ((359, 300), (300, 359), (241, 300), (300, 241)):
                cv2.circle(shown, (x, y), 2, probe_color, -1)
            if index in DISPLAY['probe_frames']:
                cv2.circle(shown, (300, 300), 7, probe_color, -1)
            writer.write(cv2.cvtColor(shown, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def build(source: Path, destination: Path) -> dict:
    """Registered builder: source files stay intact; prepared data is written separately."""
    import cv2
    source, destination = Path(source), Path(destination)
    manifest_path = source / 'source-manifest.json'
    source_manifest = json.loads(manifest_path.read_text())
    if (source_manifest['github_revision'], source_manifest['hf_revision']) != (GITHUB_REVISION, HF_REVISION):
        raise ValueError('Unsupported source revision')
    for name, info in source_manifest['files'].items():
        if checksum(source / name) != info['sha256']:
            raise ValueError(f'Source checksum mismatch: {name}')
    scenes = source_manifest['scenes']
    table = pd.read_excel(source / 'second-order-exp/Static_AllTrails_Final_shuffled.xlsx')
    if table.duplicated(['MovI', 'modulation', 'fliptype']).any() or len(table) != 1120:
        raise ValueError('Expected 1,120 unique scene/modulation/flip combinations')
    participants = [pd.read_csv(source / f'second-order-exp/data/participant{i}/data.csv') for i in (1, 2, 3)]
    references = reference_rows(table, participants, scenes)
    (destination / 'clips').mkdir(parents=True)
    stimuli, outputs = [], {}
    for scene in scenes:
        for modulation in range(1, 8):
            prefix = source / f'human_static/image/Scene_{scene}/type_{modulation}'
            frames = [cv2.cvtColor(cv2.imread(str(prefix / f'frame_{frame:04d}.png')), cv2.COLOR_BGR2RGB)
                      for frame in range(16)]
            rows = table[(table.MovI == scene) & (table.modulation == modulation)].sort_values('fliptype')
            if rows.fliptype.tolist() != [1, 2, 3, 4]:
                raise ValueError('Missing flip condition')
            for row in rows.itertuples(index=False):
                base_id = f'sun2025-s{scene:02d}-m{modulation}-v1'
                stimulus_id = f'{base_id}-f{row.fliptype}'
                name = f'clips/{stimulus_id}.avi'
                render_clip(frames, row, destination / name)
                outputs[name] = checksum(destination / name)
                stimuli.append(dict(stimulus_id=stimulus_id, base_id=base_id, scene=scene,
                                    modulation=modulation, flip=int(row.fliptype), video_path=name))
        print(f'Packaged scene {scene}', flush=True)
    for name, rows in (('stimuli.json', stimuli), ('human-responses.json', references)):
        (destination / name).write_text(json.dumps(rows, indent=2, allow_nan=False) + '\n')
        outputs[name] = checksum(destination / name)
    (destination / 'source-manifest.json').write_bytes(manifest_path.read_bytes())
    outputs['source-manifest.json'] = checksum(manifest_path)
    manifest = {'dataset': 'Sun2025-second-order-motion', 'schema_version': 1,
                'scenes': scenes, 'complete': scenes == list(range(1, 41)),
                'display': DISPLAY, 'files': outputs, 'trials': len(stimuli),
                'participants': 3, 'human_aggregation': 'unflip then average four vectors per scene/modulation',
                'speed_units': 'displayed pixels per frame', 'coordinates': 'x right, y down',
                'source_manifest_sha256': checksum(manifest_path)}
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    return manifest
