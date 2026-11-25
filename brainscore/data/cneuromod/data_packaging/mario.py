"""CNeuroMod mario: what differs from the other CNeuroMod data sets (Paugam et al. 2025).

The participants play Super Mario Bros. (NES) in the scanner and hear its sound. A run holds several
repetitions (one play of one level, one ``.bk2`` replay). The events are the annotated event files
of ``mario.annotations`` (``*_desc-annotated_events.tsv``): one ``gym-retro_game`` row per repetition
(onset, duration, level, frames) and one row per button press, kill, coin, hit, ... in run time.

The stimulus of a TR is the repetition that holds its start; its identifier is the replay name
(``sub-01_ses-001_task-mario_level-w1l1_rep-006``). The stimulus is the participant's own play
video of that repetition, shipped by ``mario.replays`` (``beh/videos/<identifier>.mp4``, 60 frames
per second, with the game sound): each video starts at its repetition's onset in the run.
Brain-Score stores no frame; the stimulus table names the video.
"""
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
STIMULUS_COLUMNS = ('level', 'video')
_GAME = 'gym-retro_game'
# Limit: mario.timeseries names cneuromod_extract_tseries at this commit, which is not in its public
# repository; the time series are made with extraction.CODE_COMMIT (the THINGS and movie10 commit, which
# has the mario configuration). The atlas and mask files are compared with the release by their MD5.
UNPUBLISHED_RELEASE_CODE = '4492c476d8d94d4c302046fc12df5ec9a67da03d'


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
        'video': [str(Path(str(path)).parent.parent / 'beh' / 'videos' / f'{stem}.mp4')
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

