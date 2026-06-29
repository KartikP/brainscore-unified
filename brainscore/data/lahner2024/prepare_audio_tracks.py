"""Demux audio tracks from BOLDMoments MP4 clips → 16 kHz mono WAV files.

Pre-processing step for the multimodal Lahner2024 variant. Reads the
existing BOLDMoments stimulus_set, locates each MP4 in the local brainio
cache, and writes a parallel WAV per clip into ``audio_dir``.

Silent clips (a small number of source MP4s ship without an audio stream)
get a zero waveform of matching duration, so every clip ends up with a
file at the expected path. The AudioWrapper sees a uniform schema.

Usage:
    python -m brainscore.data.lahner2024.prepare_audio_tracks \
        --audio-dir ~/.brainio/lahner2024_audio_16k \
        --target-rate 16000

Idempotent — skips clips whose WAV already exists. Re-run safely.
"""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from tqdm import tqdm

# Local import — placed inside main so importing this data-plugin script stays
# side-effect free.


VIDEO_DURATION_SEC = 3.0


def _has_audio_stream(video_path: Path) -> bool:
    """True if ffprobe reports an audio stream in the file."""
    out = subprocess.run(
        ['ffprobe', '-v', 'error',
         '-select_streams', 'a',
         '-show_entries', 'stream=codec_type',
         '-of', 'csv=p=0',
         str(video_path)],
        capture_output=True, text=True, check=False,
    )
    return out.returncode == 0 and out.stdout.strip().startswith('audio')


def _extract_one(video_path: Path, audio_path: Path,
                 target_rate: int) -> str:
    """Extract a single clip's audio. Returns the action taken
    ('extracted', 'cached', 'silent_placeholder')."""
    if audio_path.exists():
        return 'cached'
    audio_path.parent.mkdir(parents=True, exist_ok=True)

    if _has_audio_stream(video_path):
        # ffmpeg: -vn drops video, -ac 1 mono, -ar target rate, -y overwrite
        result = subprocess.run(
            ['ffmpeg', '-y', '-loglevel', 'error',
             '-i', str(video_path),
             '-vn', '-ac', '1', '-ar', str(target_rate),
             str(audio_path)],
            capture_output=True, text=True, check=False,
        )
        if result.returncode == 0 and audio_path.exists():
            return 'extracted'
        # Fall through to placeholder if ffmpeg produced nothing
    # Silent placeholder
    n_samples = int(VIDEO_DURATION_SEC * target_rate)
    wavfile.write(audio_path, target_rate,
                  np.zeros(n_samples, dtype=np.int16))
    return 'silent_placeholder'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio-dir', type=Path, required=True,
                        help='Directory to write WAV files into')
    parser.add_argument('--target-rate', type=int, default=16000)
    args = parser.parse_args()

    if shutil.which('ffmpeg') is None:
        raise SystemExit("ffmpeg not found in PATH; install ffmpeg first.")

    from brainscore import load_stimulus_set
    stim = load_stimulus_set('BOLDMoments')
    print(f"Loaded stim_set with {len(stim)} clips")

    args.audio_dir.mkdir(parents=True, exist_ok=True)

    counts = {'extracted': 0, 'cached': 0, 'silent_placeholder': 0}
    manifest = []
    for _, row in tqdm(stim.iterrows(), total=len(stim),
                       desc="extracting audio"):
        sid = row['stimulus_id']
        video_path = Path(stim.get_stimulus(sid))
        audio_path = args.audio_dir / f'{video_path.stem}.wav'
        action = _extract_one(video_path, audio_path, args.target_rate)
        counts[action] += 1
        manifest.append({
            'stimulus_id': sid,
            'video_path': str(video_path),
            'audio_path': str(audio_path),
            'action': action,
        })

    print(f"Counts: {counts}")
    manifest_path = args.audio_dir / 'manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump({'counts': counts,
                   'target_rate': args.target_rate,
                   'entries': manifest}, f, indent=2)
    print(f"Wrote manifest to {manifest_path}")


if __name__ == '__main__':
    main()
