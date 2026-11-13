"""Small seeded Lahner2024 data at the loader seam, and subjects that answer by participate.

``install(monkeypatch, tmp_path)`` puts a synthetic stimulus set, GLM-beta
assembly, TR assembly, events table, motion sidecar, frame files, audio files and
Destrieux atlas behind the registered loaders, so a benchmark loaded by name runs
end to end without network or model weights. No private name of a benchmark is
touched.
"""
import hashlib

import numpy as np
import pandas as pd

import brainscore
from brainscore_core.contract import Subject
from brainscore_core.streaming import StreamEvent
from brainscore_core.streaming_helpers import drain_session_inputs, session_stimuli
from brainscore_core.supported_data_standards.brainio.assemblies import (
    NeuroidAssembly, NeuronRecordingAssembly)
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet

N_VOXELS = 20484          # the TR benchmark checks the fsaverage5 size
N_TR = 12
SAMPLE_TIMES = (500, 1500, 2500)
CLIPS = [f'stimulus_{i:04d}' for i in (3, 1, 6, 2, 5, 4)]   # not sorted on purpose
MOTION_COLUMNS = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z',
                  'framewise_displacement', 'csf', 'white_matter']


def stimulus_set():
    stimuli = StimulusSet(pd.DataFrame({'stimulus_id': CLIPS}))
    stimuli.identifier = 'BOLDMoments'
    stimuli.stimulus_paths = {clip: f'/synthetic/{clip}.mp4' for clip in CLIPS}
    return stimuli


GLM_REPS = 4


def glm_values(reps=GLM_REPS, seed=1):
    """``(presentation, voxel)`` betas; presentation ``i`` shows ``CLIPS[i % len(CLIPS)]``."""
    rng = np.random.default_rng(seed)
    signal = rng.standard_normal((len(CLIPS), N_VOXELS)) * np.linspace(0.0, 2.0, N_VOXELS)
    rows = [(clip, rep) for rep in range(reps) for clip in CLIPS]
    return np.stack([signal[CLIPS.index(clip)] + rng.standard_normal(N_VOXELS)
                     for clip, _ in rows]).astype(np.float32)


def glm_assembly(values=None, reps=GLM_REPS):
    values = glm_values(reps) if values is None else np.asarray(values, dtype=np.float32)
    rows = [(clip, rep) for rep in range(reps) for clip in CLIPS]
    return NeuronRecordingAssembly(values.T[None], dims=['time_bin', 'neuroid', 'presentation'], coords={
        'time_bin_start': ('time_bin', [0]), 'time_bin_end': ('time_bin', [3000]),
        'neuroid_id': ('neuroid', np.arange(N_VOXELS)),
        'hemisphere': ('neuroid', ['L'] * (N_VOXELS // 2) + ['R'] * (N_VOXELS // 2)),
        'stimulus_id': ('presentation', [clip for clip, _ in rows]),
        'repetition': ('presentation', [rep for _, rep in rows])})


def tr_runs(participants=('01', '01', '02', '02')):
    """One run per entry; runs of one participant share (session, run) across the
    two tasks, as in BOLDMoments."""
    runs, seen = [], {}
    for participant in participants:
        k = seen.get(participant, 0)
        seen[participant] = k + 1
        runs.append(dict(subject=participant, session='01', run='01',
                         task='train' if k % 2 == 0 else 'test', n_valid_TR=N_TR))
    return runs


def tr_assembly(runs, bold=None, seed=2):
    rng = np.random.default_rng(seed)
    if bold is None:
        bold = 1000.0 + 50.0 * rng.standard_normal((len(runs), N_TR, N_VOXELS))
    coords = {key: ('presentation', [run[key] for run in runs])
              for key in ('subject', 'session', 'task', 'run', 'n_valid_TR')}
    coords.update({
        'time_bin_start_ms': ('time_bin', [int(t * 1750) for t in range(N_TR)]),
        'time_bin_end_ms': ('time_bin', [int((t + 1) * 1750) for t in range(N_TR)]),
        'neuroid_id': ('neuroid', np.arange(N_VOXELS)),
        'hemisphere': ('neuroid', ['L'] * (N_VOXELS // 2) + ['R'] * (N_VOXELS // 2))})
    return NeuronRecordingAssembly(np.asarray(bold, dtype=np.float32),
                                   dims=['presentation', 'time_bin', 'neuroid'], coords=coords)


def tr_events(runs, onsets=(0.0, 4.0, 8.0, 12.0)):
    """Each run shows the clips in turn; a train run and a test run differ."""
    rows = []
    for index, run in enumerate(runs):
        for k, onset in enumerate(onsets):
            clip = CLIPS[(k + (0 if run['task'] == 'train' else 3)) % len(CLIPS)]
            rows.append(dict(subject=run['subject'], session=run['session'], task=run['task'],
                             run=run['run'], stimulus_id=clip, onset_sec=onset, duration_sec=3.0))
    return pd.DataFrame(rows)


def tr_motion(runs, seed=4):
    rng = np.random.default_rng(seed)
    return pd.concat([pd.DataFrame(dict(
        subject=run['subject'], session=run['session'], task=run['task'], run=run['run'],
        **{column: rng.standard_normal(N_TR) for column in MOTION_COLUMNS})) for run in runs],
        ignore_index=True)


def install(monkeypatch, tmp_path, runs=None, bold=None, events=None, glm=None):
    """Put the synthetic data behind the loaders; returns the TR runs. ``glm`` replaces
    the GLM betas (``glm_values`` layout), ``bold`` the TR series."""
    runs = tr_runs() if runs is None else runs
    cache = {}

    def once(name, build):
        def load(*args, **kwargs):
            if name not in cache:
                cache[name] = build()
            return cache[name]
        return load

    monkeypatch.setitem(brainscore.stimulus_set_registry, 'BOLDMoments', once('s', stimulus_set))
    for identifier, build in (
            ('Lahner2024-fMRI', lambda: glm_assembly(glm)),
            ('Lahner2024-fMRI-timeresolved', lambda: tr_assembly(runs, bold)),
            ('Lahner2024-fMRI-timeresolved-events',
             lambda: tr_events(runs) if events is None else events),
            ('Lahner2024-fMRI-timeresolved-motion', lambda: tr_motion(runs))):
        monkeypatch.setitem(brainscore.data_registry, identifier, once(identifier, build))
    import nilearn.datasets
    atlas = np.random.default_rng(5)
    monkeypatch.setattr(nilearn.datasets, 'fetch_atlas_surf_destrieux', lambda: {
        'map_left': atlas.integers(0, 76, 10242), 'map_right': atlas.integers(0, 76, 10242)})
    frames = tmp_path / 'frames'
    frames.mkdir()
    for clip in CLIPS:
        for t in SAMPLE_TIMES:
            (frames / f'{clip}_t{t}.png').write_bytes(b'png')
    monkeypatch.setenv('BRAINSCORE_LAHNER_FRAMES_DIR', str(frames))
    audio = tmp_path / '.brainio' / 'lahner2024_audio_16k'
    audio.mkdir(parents=True)
    for clip in CLIPS:
        (audio / f'{clip}.wav').write_bytes(b'wav')
    monkeypatch.setenv('HOME', str(tmp_path))
    return runs


def unit_values(stimulus_id, channel, width=4, n_bins=1, salt=''):
    seed = int.from_bytes(hashlib.sha256(f'{salt}|{stimulus_id}|{channel}'.encode()).digest()[:8],
                          'little')
    return np.random.default_rng(seed).standard_normal((n_bins, width)).astype(np.float32)


class Recorder(Subject):
    """Answers every requested ``neural:<region>`` for whatever it is shown; keeps
    ``(input channels, outputs, stimulus ids, stimuli identifier)`` per session."""

    def __init__(self, in_channels, regions=('IT',), width=4, time_bins=None, values=None):
        self._in = frozenset(in_channels)
        self._regions = tuple(regions)
        self._width = width
        self._time_bins = time_bins
        self._values = values or (lambda stimulus_id, channel: unit_values(
            stimulus_id, channel, self._width, self._time_bins or 1))
        self.seen = []

    @property
    def identifier(self):
        return 'synthetic-recorder'

    @property
    def in_channels(self):
        return self._in

    @property
    def out_channels(self):
        return frozenset(f'neural:{region}' for region in self._regions)

    def reset(self):
        pass

    def participate(self, session):
        session.declare_inputs(self._in)
        events = drain_session_inputs(session)
        stimuli = session_stimuli(session, events)
        ids = [str(s) for s in stimuli['stimulus_id']]
        self.seen.append((sorted({event.channel for event in events}), list(session.outputs),
                          ids, getattr(stimuli, 'identifier', None)))
        for channel in session.outputs:
            block = np.stack([self._values(sid, channel) for sid in ids])
            width = block.shape[-1]
            neuroid = {'neuroid_id': ('neuroid', [f'{channel}.{u}' for u in range(width)]),
                       'layer': ('neuroid', ['layer'] * width)}
            if self._time_bins:
                assembly = NeuroidAssembly(block, dims=['presentation', 'time_bin', 'neuroid'], coords={
                    'stimulus_id': ('presentation', ids),
                    'time_bin_start_ms': ('time_bin', [i * 100 for i in range(self._time_bins)]),
                    'time_bin_end_ms': ('time_bin', [(i + 1) * 100 for i in range(self._time_bins)]),
                    **neuroid})
            else:
                assembly = NeuroidAssembly(block[:, 0, :], dims=['presentation', 'neuroid'], coords={
                    'stimulus_id': ('presentation', ids), **neuroid})
            session.respond(StreamEvent(channel, assembly, t_ms=None))
