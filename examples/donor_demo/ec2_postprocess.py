"""EC2 post-processing after donor_lahner_modality.py finished.

- copies the demo clips' MP4 + WAV next to the demo outputs
- writes stimuli_manifest.csv for all 1026 clips (sha1, duration, fps, frames, audio flag)
- writes vertex coordinates of fsaverage5 (pial + inflated) for the viz package
"""
import csv
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

OUT = Path('/opt/dlami/nvme/donor_demo/out')
STIM = Path('/home/ubuntu/.brainio/dc9c3bf631632cd433d02f2f1847fd33c01ae0b3/stimulus_BOLDMoments')
WAV = Path('/home/ubuntu/.brainio/lahner2024_audio_16k')


def sha1(p):
    h = hashlib.sha1()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


sel = json.load(open(OUT / 'demo' / 'selection.json'))
ids = sel['top'] + [sel['median']] + sel['worst']
(OUT / 'demo' / 'clips').mkdir(exist_ok=True)
for cid in ids:
    shutil.copy(STIM / f'{cid}.mp4', OUT / 'demo' / 'clips' / f'{cid}.mp4')
    if (WAV / f'{cid}.wav').exists():
        shutil.copy(WAV / f'{cid}.wav', OUT / 'demo' / 'clips' / f'{cid}.wav')

man = json.load(open(WAV / 'manifest.json'))
silent = {e['stimulus_id']: e['action'] for e in man['entries']}
rows = []
for p in sorted(STIM.glob('*.mp4')):
    cap = cv2.VideoCapture(str(p))
    fps = cap.get(cv2.CAP_PROP_FPS); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); cap.release()
    rows.append([p.stem, p.name, sha1(p), p.stat().st_size, round(fps, 3), n, round(n / fps, 3) if fps else '', w, h, silent.get(p.stem, '')])
with open(OUT / 'stimuli_manifest.csv', 'w', newline='') as f:
    wr = csv.writer(f)
    wr.writerow(['stimulus_id', 'file', 'sha1', 'bytes', 'fps', 'n_frames', 'duration_s', 'width', 'height', 'audio_sidecar_action'])
    wr.writerows(rows)

from nilearn import datasets, surface
fs = datasets.fetch_surf_fsaverage('fsaverage5')
out = {}
for hemi in ('left', 'right'):
    for kind in ('pial', 'infl'):
        c, fcs = surface.load_surf_mesh(fs[f'{kind}_{hemi}'])
        out[f'{kind}_{hemi}_coords'] = c.astype(np.float32); out[f'{kind}_{hemi}_faces'] = fcs.astype(np.int32)
    out[f'sulc_{hemi}'] = surface.load_surf_data(fs[f'sulc_{hemi}']).astype(np.float32)
np.savez_compressed(OUT / 'neural' / 'fsaverage5_surfaces.npz', **out)
print('postprocess done', len(rows), 'clips in manifest')
