"""Cut the frames a stills-only subject sees from each BOLDMoments clip.

Writes one PNG per (clip, sample time) into the local asset ``lahner2024-frames``
(``$BRAINSCORE_DATA_HOME/lahner2024-frames``, default ``~/brainscore-data/lahner2024-frames``).
The Lahner2024 benchmarks read these files and extract nothing themselves.

Usage:
    python -m brainscore.data.lahner2024.data_packaging.prepare_frames \
        [--frames-dir <folder>] [--sample-times-ms 500 1500 2500]

Idempotent: an existing frame is kept. Re-run safely.
"""
import argparse
from pathlib import Path

from brainscore.data.lahner2024 import DEFAULT_SAMPLE_TIMES_MS, LAHNER_FRAMES, frame_file


def extract_frame(video_path, time_ms: float, frames_dir: Path) -> str:
    """Extract one frame of ``video_path`` at ``time_ms`` as a PNG under ``frames_dir``.

    Returns the path. Idempotent: an existing frame is returned without
    extracting it again.
    """
    import cv2  # imported here to avoid making it a brainscore_core dep

    video_path = Path(video_path)
    frames_dir.mkdir(parents=True, exist_ok=True)
    out_path = frame_file(frames_dir, video_path, time_ms)
    if out_path.exists():
        return str(out_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"cv2 could not open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    target_frame_idx = int(round(time_ms / 1000.0 * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise IOError(
            f"cv2 failed to read frame at {time_ms}ms "
            f"(target idx {target_frame_idx}, fps {fps}) from {video_path}")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   # cv2 reads BGR
    from PIL import Image
    Image.fromarray(frame).save(out_path)
    return str(out_path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--frames-dir', type=Path, default=LAHNER_FRAMES.resolved(),
                        help=f'where to write the frames (default: {LAHNER_FRAMES.resolved()})')
    parser.add_argument('--sample-times-ms', type=float, nargs='+',
                        default=list(DEFAULT_SAMPLE_TIMES_MS),
                        help='times of the frames in each clip, in milliseconds')
    args = parser.parse_args(argv)

    from tqdm import tqdm
    from brainscore import load_stimulus_set
    stimuli = load_stimulus_set('BOLDMoments')
    for stimulus_id in tqdm(stimuli['stimulus_id'], desc='extracting frames'):
        video_path = stimuli.get_stimulus(stimulus_id)
        for time_ms in args.sample_times_ms:
            extract_frame(video_path, time_ms, args.frames_dir)
    print(f'{len(stimuli) * len(args.sample_times_ms)} frames in {args.frames_dir}')


if __name__ == '__main__':
    main()
