"""A stateless UMI subject that delegates RSA inference to memo."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import sys
from typing import Set

import numpy as np

from brainscore_core.contract import Subject
from brainscore_core.streaming import Session, StreamEvent

from .worker import MemoWorker


UTTERANCES = ('green', 'pink', 'square', 'round')
CHOICES = ('green square', 'green circle', 'pink circle')
SCENE = ('Choose among three objects: a green square, a green circle, and a pink circle. '
         'A speaker chose one object and uttered one of green, pink, square, or round. '
         'Infer the intended referent.')


@dataclass(frozen=True)
class MemoConfig:
    depth: int = 1
    beta: float = 1.0

    def __post_init__(self) -> None:
        if type(self.depth) is not int or self.depth < 0:
            raise ValueError('depth must be a nonnegative integer')
        if isinstance(self.beta, bool) or not isinstance(self.beta, (int, float)):
            raise TypeError('beta must be numeric')
        if not math.isfinite(self.beta):
            raise ValueError('beta must be finite')


class MemoSubject(Subject):
    """Memo RSA predictions presented through instruction and text channels."""

    def __init__(self, *, config: MemoConfig, python_executable: str,
                 timeout_s: float = 30.0) -> None:
        self._config = config
        self.python_executable = python_executable
        self.timeout_s = timeout_s
        self._worker = MemoWorker(python_executable, timeout_s=timeout_s)

    @property
    def config(self) -> MemoConfig:
        return self._config

    @property
    def identifier(self) -> str:
        return f'memo-rsa-depth-{self.config.depth}-beta-{self.config.beta:g}'

    @property
    def in_channels(self) -> Set[str]:
        return {'instruction', 'text'}

    @property
    def out_channels(self) -> Set[str]:
        return {'behavior'}

    @property
    def required_channels(self) -> Set[str]:
        return {'instruction', 'text'}

    def interact(self, session: Session) -> None:
        requested = session.requested_output_channels
        if requested is not None and set(requested) - self.out_channels:
            raise ValueError(f'Unsupported requested outputs: {set(requested) - self.out_channels}')
        if requested is not None and 'behavior' not in requested:
            raise ValueError("MemoSubject requires the 'behavior' output channel")
        instruction_seen = False
        utterance_seen = False
        try:
            while (event := session.next_input()) is not None:
                if not isinstance(event, StreamEvent):
                    raise TypeError('MemoSubject requires StreamEvent inputs')
                if event.channel == 'instruction':
                    if instruction_seen or event.payload != SCENE:
                        raise ValueError('Expected exactly one canonical green/pink/square/round task instruction')
                    instruction_seen = True
                    continue
                if event.channel != 'text':
                    continue
                if not instruction_seen:
                    raise ValueError('Expected the canonical task instruction before utterances')
                if not isinstance(event.payload, str) or event.payload not in UTTERANCES:
                    raise ValueError(f'Utterances must be labels in {UTTERANCES!r}')
                utterance_seen = True
                rows = self._worker.predict(
                    depth=self.config.depth,
                    beta=self.config.beta,
                    utterances=[event.payload],
                )
                if len(rows) != 1:
                    raise RuntimeError('memo worker returned the wrong number of probability rows')
                row = np.asarray(rows[0], dtype=np.float64)
                if row.shape != (len(CHOICES),):
                    raise RuntimeError(f'memo worker returned probability shape {row.shape}')
                if (not np.all(np.isfinite(row)) or np.any(row < 0)
                        or math.fsum(map(float, row)) <= 0.0):
                    raise RuntimeError('memo worker returned an invalid probability distribution')
                session.respond(StreamEvent(
                    channel='behavior', payload=row, t_ms=event.t_ms,
                    meta={
                        'stimulus_id': event.meta.get('stimulus_id', event.payload),
                        'label_set': CHOICES,
                        'utterance': event.payload,
                        'provenance': self.provenance,
                    },
                ))
            if not instruction_seen:
                raise ValueError('Expected exactly one canonical task instruction')
            if not utterance_seen:
                raise ValueError('At least one utterance is required')
        except BaseException:
            self._worker.close()
            raise

    @property
    def provenance(self) -> dict[str, object]:
        return {
            'implementation': 'memo Rational Speech Acts example',
            'memo_source_reviewed_revision': 'd8aac635ee53483e8e0311b85f6898ef37710a26',
            'memo_distribution': 'memo-lang',
            'implementation_sha256': {
                name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                for name in ('model.py', 'worker.py', 'memo_worker.py', 'memo_model.py')
            },
            'host_numpy': np.__version__,
            'output_dtype': 'float64',
            'reasoning_depth': self.config.depth,
            'beta': float(self.config.beta),
            'worker_python': self.python_executable,
            'host_python': sys.version.split()[0],
            'umi_base_revision': 'b8880276242296c6857363777b4bd12a02ab5213',
            'pinned_revisions': {
                'core': '15dcc3c6cbfea16565052abd2663b8de4bf1b203',
                'vision': 'eca24202bb7b188c1115d3b140b96abc0fa848c7',
                'language': '62e2dcfe17daa81ecdf939f6e075d9edd33389ca',
            },
            'runtime': self._worker.runtime,
        }

    def reset(self) -> None:
        self._worker.close()


__all__ = ['MemoConfig', 'MemoSubject', 'UTTERANCES', 'CHOICES', 'SCENE']
