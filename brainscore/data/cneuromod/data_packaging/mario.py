"""CNeuroMod mario: what differs from the other CNeuroMod data sets (Paugam et al. 2025).

The participants play Super Mario Bros. (NES) in the scanner and hear its sound. A run holds several
repetitions (one play of one level, one ``.bk2`` replay). The events are the annotated event files
of ``mario.annotations`` (``*_desc-annotated_events.tsv``): one ``gym-retro_game`` row per repetition
(onset, duration, level, frames) and one row per button press, kill, coin, hit, ... in run time.

The stimulus of a TR is the repetition that holds its start; its identifier is the replay name
(``sub-01_ses-001_task-mario_level-w1l1_rep-006``). The stimulus is the participant's own play
video of that repetition, shipped by ``mario.replays`` (``beh/videos/<identifier>.mp4``, 60 frames
per second, with the game sound): each video starts at its repetition's onset in the run.
Brain-Score stores no frame; the stimulus table names the video. After the packaging, the media of
each TR are made on this machine from those videos (``write_media``; run this module):

    python -m brainscore.data.cneuromod.data_packaging.mario --packaged <packaged folder> [--part sub-01]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from brainscore.data.cneuromod.data_packaging import shared

NAME = 'mario'
REPOSITORY = 'mario.annotations'     # the annotated event files, same folders as the BIDS data set
PARTICIPANTS = ('sub-01', 'sub-02', 'sub-03', 'sub-05', 'sub-06')
TR_SECONDS = 1.49
FRAMES_PER_SECOND = 60
BUTTONS = ('A', 'B', 'UP', 'DOWN', 'LEFT', 'RIGHT')
STIMULUS_SOURCE = ('the play videos of CNeuroMod mario.replays (https://github.com/courtois-neuromod/mario.replays, '
                   'beh/videos/<stimulus_id>.mp4); Super Mario Bros. is (c) Nintendo. Brain-Score does not host them.')
STIMULUS_COLUMNS = ('level', 'replay_video')     # not ``video``: a path relative to mario.replays, not a file here
REPLAYS = 'mario.replays'
SOUND_RATE_HZ = 16000           # the sound of a TR: mono, as the other sound benchmarks give it
CLIP_FRAMES = round(TR_SECONDS * FRAMES_PER_SECOND)     # 89: every clip lasts one TR
_GAME = 'gym-retro_game'
# Limit: mario.timeseries names cneuromod_extract_tseries at this commit, which is not in its public
# repository; the time series are made with extraction.CODE_COMMIT (the THINGS and movie10 commit, which
# has the mario configuration). The atlas and mask files are compared with the release by their MD5.
UNPUBLISHED_RELEASE_CODE = '4492c476e8d94d4c302046fc12df5ec9a67da03d'


def event_patterns(participant: str, session: str) -> list:
    return [f'{participant}/{session}/func/{participant}_{session}_task-mario_run-*_desc-annotated_events.tsv']


def _annotated(clone: Path, participant: str, run: dict) -> pd.DataFrame:
    files = [path for path in clone.glob(f'{participant}/ses-*/func/{participant}_ses-*_task-mario_run-*'
                                         '_desc-annotated_events.tsv')
             if shared.number(shared.entities(path.name)['ses']) == run['session']
             and shared.number(shared.entities(path.name)['run']) == run['run']]
    if len(files) != 1:
        raise FileNotFoundError(f'{participant} session {run["session"]} run {run["run"]}: '
                                f'{len(files)} annotated event files found, 1 expected')
    return shared.read_tsv(files[0])


def run_events(clone: Path, participant: str, run: dict) -> pd.DataFrame:
    """The repetitions of one run: onset and duration in run time, level, the replay name and its video."""
    events = _annotated(clone, participant, run)
    games = events[events['trial_type'] == _GAME].reset_index(drop=True)
    stems = [Path(str(path)).stem for path in games['stim_file']]
    return pd.DataFrame({
        'onset': games['onset'], 'duration': games['duration'], 'stimulus_id': stems,
        'repetition': [shared.number(shared.entities(stem)['rep']) for stem in stems],
        'level': games['level'], 'phase': games['phase'],
        'frames': games['frame_stop'].astype(int),
        'replay_video': [str(Path(str(path)).parent.parent / 'beh' / 'videos' / f'{stem}.mp4')
                  for path, stem in zip(games['stim_file'], stems)]})


def tr_columns(clone: Path, participant: str, run: dict) -> dict:
    """Per TR of ``run`` (``[start, start + TR)`` in run time, ``start`` = volume x TR): the level and the frame range of its repetition's
    video (``frame_start`` included, ``frame_stop`` excluded; both -1 without a repetition), and per button
    the seconds it was held in the TR (``button_A``, ...). A TR may end after its repetition: its frames
    stop at the repetition's last frame."""
    events = _annotated(clone, participant, run)
    games = run_events(clone, participant, run)
    starts = run['volumes'] * TR_SECONDS
    rows = shared.tr_events(run['volumes'], TR_SECONDS, games)
    columns = {'level': [games['level'].iloc[r] if r >= 0 else '' for r in rows]}
    first, last = np.full(len(starts), -1), np.full(len(starts), -1)
    for i, (start, row) in enumerate(zip(starts, rows)):
        if row < 0:
            continue
        onset, frames = games['onset'].iloc[row], games['frames'].iloc[row]
        first[i] = min(int(np.floor((start - onset) * FRAMES_PER_SECOND)), frames)
        last[i] = min(int(np.floor((start + TR_SECONDS - onset) * FRAMES_PER_SECOND)), frames)
    columns['frame_start'], columns['frame_stop'] = first, last
    for button in BUTTONS:
        presses = events[events['trial_type'] == button]
        ends = presses['onset'] + presses['duration']
        columns[f'button_{button}'] = np.array([
            np.clip(np.minimum(ends, start + TR_SECONDS) - np.maximum(presses['onset'], start), 0, None).sum()
            for start in starts])
    return columns



def tr_identifier(participant: str, session: int, run: int, tr: int) -> str:
    return f'{participant}_ses-{session:03d}_run-{run}_tr-{tr:04d}'


def write_media(packaged: Path, participant: str, replays: Path) -> Path:
    """The media of each TR of ``participant`` that shows a repetition (``frame_start`` < ``frame_stop``), from
    the play videos in the clone ``replays`` (``mario.replays``): the clip of its frames (``.mp4``, 60 frames per
    second, 240 x 224, ``CLIP_FRAMES`` frames: a TR that ends after its play repeats the play's last frame and
    its sound is silent there; ``frames`` counts the frames of the play), its middle frame (``.png``) and the
    sound of the same frames (``.wav``, 16 kHz, mono),
    in ``<participant>/media/``, with ``media.csv`` (one row per TR, paths relative to ``packaged``). The
    videos are fetched with DataLad, which checks each against the MD5 of its annex key."""
    import cv2
    import xarray as xr
    from scipy.io import wavfile
    sessions = sorted((packaged / participant).glob('ses-*/timeseries_cneuromod2026.nc'))
    if not sessions:
        raise FileNotFoundError(f'no ses-*/timeseries_cneuromod2026.nc under {packaged / participant}: package '
                                f"it first (`package_data('cneuromod-mario-timeseries', part='{participant}')`).")
    events = pd.concat([pd.read_csv(path.parent / 'events.csv') for path in sessions], ignore_index=True)
    videos = {stimulus: replays / video for stimulus, video in zip(events['stimulus_id'], events['replay_video'])}
    rows = []
    for path in sessions:
        with xr.open_dataarray(path) as series:
            trs = pd.DataFrame({name: series[name].values for name in
                                ('session', 'run', 'tr', 'stimulus_id', 'frame_start', 'frame_stop')})
        trs = trs[trs['frame_start'] < trs['frame_stop']]
        shared.get(sorted({videos[stimulus] for stimulus in trs['stimulus_id']}))
        for stimulus, group in trs.groupby('stimulus_id', sort=False):
            rows += _cut(videos[stimulus], participant, group.sort_values('frame_start'), packaged, cv2, wavfile)
    table = pd.DataFrame(rows)
    shared.write_csv(table, packaged / participant / 'media' / 'media.csv')
    return packaged / participant / 'media' / 'media.csv'


def _cut(video: Path, participant: str, trs: pd.DataFrame, packaged: Path, cv2, wavfile) -> list:
    """The media of the TRs ``trs`` of one repetition video, read once in frame order."""
    import subprocess
    sound = np.frombuffer(subprocess.run(
        ['ffmpeg', '-v', 'error', '-i', str(video), '-ac', '1', '-ar', str(SOUND_RATE_HZ), '-f', 's16le', '-'],
        capture_output=True, check=True).stdout, dtype=np.int16)
    capture = cv2.VideoCapture(str(video))
    frame, rows = 0, []
    for tr in trs.itertuples():
        name = tr_identifier(participant, int(tr.session), int(tr.run), int(tr.tr))
        folder = packaged / participant / 'media' / f'ses-{int(tr.session):03d}'
        folder.mkdir(parents=True, exist_ok=True)
        while frame < tr.frame_start:
            capture.grab()
            frame += 1
        frames = []
        while frame < tr.frame_stop:
            ok, image = capture.read()
            if not ok:
                break           # a video may hold a frame or two fewer than its replay
            frames.append(image)
            frame += 1
        if not frames:
            continue
        played = len(frames)
        frames = (frames + [frames[-1]] * CLIP_FRAMES)[:CLIP_FRAMES]
        writer = cv2.VideoWriter(str(folder / f'{name}.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), FRAMES_PER_SECOND,
                                 (frames[0].shape[1], frames[0].shape[0]))
        for image in frames:
            writer.write(image)
        writer.release()
        cv2.imwrite(str(folder / f'{name}.png'), frames[len(frames) // 2])
        start = int(tr.frame_start) * SOUND_RATE_HZ // FRAMES_PER_SECOND
        samples = CLIP_FRAMES * SOUND_RATE_HZ // FRAMES_PER_SECOND
        heard = sound[start:(int(tr.frame_start) + min(played, CLIP_FRAMES)) * SOUND_RATE_HZ // FRAMES_PER_SECOND][:samples]
        wavfile.write(folder / f'{name}.wav', SOUND_RATE_HZ, np.pad(heard, (0, samples - len(heard))))
        relative = folder.relative_to(packaged)
        rows.append({'stimulus_id': name, 'session': int(tr.session), 'run': int(tr.run), 'tr': int(tr.tr),
                     'repetition_id': tr.stimulus_id, 'frames': played,
                     'video_path': str(relative / f'{name}.mp4'), 'image_path': str(relative / f'{name}.png'),
                     'audio_path': str(relative / f'{name}.wav')})
    capture.release()
    return rows


def main(argv=None) -> None:
    import argparse
    from brainscore.data.local import data_home
    parser = argparse.ArgumentParser(description='Make the media of each TR from the mario play videos.')
    parser.add_argument('--packaged', required=True, help='the folder package_data wrote')
    parser.add_argument('--replays', help='a clone of mario.replays (default: cloned in the download folder)')
    parser.add_argument('--part', default='', help='one participant, sub-XX (default: every packaged one)')
    args = parser.parse_args(argv)
    packaged = Path(args.packaged)
    if args.replays:
        replays = Path(args.replays)
    else:
        shared.check_prerequisites()
        replays = shared.clone(data_home() / 'sources' / 'cneuromod', REPLAYS)
    participants, _ = shared.scope(sys.modules[__name__], args.part)
    for participant in participants:
        done = packaged / participant / 'media' / 'media.csv'     # written last: its media are complete
        if done.is_file():
            print(f'{participant}: media made already ({done}); delete that file to make them again')
        elif (packaged / participant).is_dir():
            print(write_media(packaged, participant, replays))


if __name__ == '__main__':
    main()
