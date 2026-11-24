"""CNeuroMod movie10: what differs from the other CNeuroMod data sets.

Each run is one film segment; its event table is one file per segment at the repository root
(``task-bourne01_events.tsv``: onset, duration, stim_file), not one per run. The stimulus
identifier is the segment (``bourne01``). Two films were shown twice: the run number is the
viewing (``repetition``).
"""
from pathlib import Path

import pandas as pd

NAME = 'movie10'
REPOSITORY = 'movie10'
PARTICIPANTS = ('sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06')    # sub-04 is refused by the shared part
TR_SECONDS = 1.49
STIMULUS_SOURCE = ('the film files of CNeuroMod movie10 (The Bourne Supremacy, Hidden Figures, Life, The Wolf '
                   'of Wall Street); copyright kept by their owners. Brain-Score does not host them.')
STIMULUS_COLUMNS = ('stim_file',)


def event_patterns(participant: str, session: str) -> list:
    return ['task-*_events.tsv']        # in git: one table per segment, shared by all participants


def run_events(clone: Path, participant: str, run: dict) -> pd.DataFrame:
    events = pd.read_csv(clone / f'task-{run["task"]}_events.tsv', sep='\t')
    events['stimulus_id'] = run['task']
    events['repetition'] = run['run']
    return events
