"""MIRAGE BOLD predictions through the existing UMI session channels."""

from __future__ import annotations

from dataclasses import dataclass
import math
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Protocol, TypedDict

import numpy as np

from brainscore_core.contract import Subject
from brainscore_core.streaming import Session, StreamEvent
from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly


SUBJECTS = ('sub-01', 'sub-02', 'sub-03', 'sub-05')
OUTPUT_CHANNEL = 'neural:Schaefer1000'
TR_MS = 1490.0


class TranscriptWord(TypedDict):
    word: str
    onset: float
    duration: float


@dataclass(frozen=True)
class Clip:
    identifier: str
    video: Path
    transcript: tuple[TranscriptWord, ...]
    t_ms: float | None


@dataclass(frozen=True)
class Prediction:
    values: np.ndarray
    provenance: dict[str, object]


class Predictor(Protocol):
    def __call__(self, clip: Clip) -> Prediction: ...


def _clip(events: dict[str, StreamEvent]) -> Clip:
    vision, audio, text = (events[name] for name in ('vision', 'audio', 'text'))
    identifiers = [event.meta.get('stimulus_id') for event in events.values()]
    if any(not isinstance(value, str) or not value for value in identifiers):
        raise ValueError('Every stream needs a nonempty stimulus_id')
    if len(set(identifiers)) != 1:
        raise ValueError('All streams must refer to the same clip')
    if any(event.t_ms != vision.t_ms for event in events.values()):
        raise ValueError('All streams must share the clip onset')
    if vision.t_ms is not None and not math.isfinite(vision.t_ms):
        raise ValueError('Clip onset must be finite or None')
    if not isinstance(vision.payload, Path) or not isinstance(audio.payload, Path):
        raise TypeError('Vision and audio must be Path references to the audiovisual container')
    if not vision.payload.is_file() or not audio.payload.is_file():
        raise ValueError('The audiovisual container must exist')
    if vision.payload.resolve() != audio.payload.resolve():
        raise ValueError('Native MIRAGE inference requires video and audio from the same container')
    if not isinstance(text.payload, str):
        raise TypeError('Text must contain the actual transcript string')
    words = text.meta.get('transcript')
    if not isinstance(words, (list, tuple)):
        raise TypeError('Text metadata must include timed transcript word records')
    previous_onset = -1.0
    transcript = []
    for item in words:
        if not isinstance(item, dict) or set(item) != {'word', 'onset', 'duration'}:
            raise ValueError('Transcript records require word, onset and duration')
        if not isinstance(item['word'], str) or not item['word']:
            raise ValueError('Transcript words must be nonempty strings')
        for key in ('onset', 'duration'):
            value = item[key]
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0):
                raise ValueError('Transcript times must be finite nonnegative seconds')
        if item['onset'] < previous_onset:
            raise ValueError('Transcript onsets must be ordered')
        previous_onset = item['onset']
        transcript.append(dict(item))
    if ' '.join(item['word'] for item in transcript) != text.payload:
        raise ValueError('Transcript text and timed words must match exactly')
    return Clip(identifiers[0], vision.payload, tuple(transcript), vision.t_ms)


class MirageSubject(Subject):
    """Predict parcel-wise BOLD for complete audiovisual clips, without refitting."""

    in_channels = {'vision', 'audio', 'text'}
    required_channels = {'vision', 'audio', 'text'}
    out_channels = {OUTPUT_CHANNEL}

    def __init__(self, *, predictor: Predictor, subject: str = 'sub-01') -> None:
        if subject not in SUBJECTS:
            raise ValueError(f'subject must be one of {SUBJECTS}')
        self._predictor = predictor
        self._subject = subject

    @property
    def subject(self) -> str:
        return self._subject

    @property
    def identifier(self) -> str:
        return f'mirage-{self.subject}'

    def interact(self, session: Session) -> None:
        requested = session.requested_output_channels
        if requested is not None and set(requested) != self.out_channels:
            raise ValueError(f'Request exactly {OUTPUT_CHANNEL}')
        if session.time_bins is not None:
            raise ValueError('MIRAGE returns its native BOLD grid; custom time bins are unsupported')
        pending: dict[str, StreamEvent] = {}
        while (event := session.next_input()) is not None:
            if event.channel not in self.in_channels:
                continue
            if event.channel in pending:
                raise ValueError('Duplicate stream before completing a clip')
            pending[event.channel] = event
            if len(pending) != len(self.required_channels):
                continue
            clip = _clip(pending)
            prediction = self._predictor(clip)
            values = prediction.values
            if (values.ndim != 2 or values.shape[0] < 1 or values.shape[1] != 1000
                    or values.dtype != np.float32 or not np.isfinite(values).all()):
                raise ValueError('Native predictions must be finite float32 (TR, 1000) arrays')
            if prediction.provenance.get('subject') != self.subject:
                raise ValueError('Predictor returned a different trained subject head')
            provenance = {
                **prediction.provenance,
                'umi_host': {'python': sys.version.split()[0], 'numpy': np.__version__,
                             'xarray': version('xarray')},
            }
            n_trs = values.shape[0]
            assembly = NeuroidAssembly(
                values,
                dims=('presentation', 'neuroid'),
                coords={
                    'stimulus_id': ('presentation', [f'{clip.identifier}:tr-{i}' for i in range(n_trs)]),
                    'clip_id': ('presentation', [clip.identifier] * n_trs),
                    'tr_index': ('presentation', np.arange(n_trs)),
                    'relative_time_ms': ('presentation', np.arange(n_trs) * TR_MS),
                    'neuroid_id': ('neuroid', [f'{self.subject}:parcel-{i}' for i in range(1000)]),
                    'parcel_index': ('neuroid', np.arange(1000)),
                    'subject_id': ('neuroid', [self.subject] * 1000),
                },
                attrs={'signal_type': 'predicted_BOLD', 'provenance': provenance,
                       'time_reference': 'native output row grid relative to clip onset'},
            )
            session.respond(StreamEvent(
                channel=OUTPUT_CHANNEL,
                payload=assembly,
                t_ms=clip.t_ms,
                meta={'stimulus_id': clip.identifier, 'signal_type': 'predicted_BOLD',
                      'provenance': provenance},
            ))
            pending.clear()
        if pending:
            raise ValueError('Incomplete clip: vision, audio and text streams are required')

    def reset(self) -> None:
        """No model state or worker process survives a prediction call."""
