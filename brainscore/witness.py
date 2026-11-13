"""Witness: see what a subject received and sent in every session of a run.

``Witness`` watches the sessions of one subject during a ``with`` block. It
uses the one way to watch a run, ``observed(session, sink)``: every session the
subject takes part in is wrapped in ``observed``, so the Witness sees the
session's input events and the subject's responses. It reads no attribute of
the subject except ``identifier`` and works for every subject, because every
subject takes part through ``participate``.

One :class:`WitnessEvent` is kept per response: the inputs delivered since the
previous response (what the subject saw) and the response (what it did). A
batch session that answers all stimuli at once is one event; a closed-loop
episode is one event per action.

    from brainscore.witness import Witness
    with Witness(subject, label='CLIP · MajajHong.IT') as w:
        score = benchmark(subject)
    w.save('/tmp/run')          # trace.json + one panel PNG per recorded event

Rendering (``w.save``) imports matplotlib and PIL only when it draws a panel.
"""
import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from brainscore_core.streaming import CONTROL_CHANNEL, parse_channel

# Event-type tags
STIMULUS, INTERVENTION, ENV_STEP = 'stimuli', 'intervention', 'environment_step'

_INTERVENTION_FAMILIES = ('lesion', 'stimulation', 'pharmacological')
_MEDIA_FAMILIES = {'vision': 'vision', 'audio': 'audio'}


@dataclass
class WitnessEvent:
    """One response of the subject, with the inputs it answered."""
    index: int
    event_type: str
    mode: str                                   # 'neural' | 'behavior' | 'action' | 'intervention'
    modalities: List[str] = field(default_factory=list)
    instruction: Optional[str] = None
    recording_target: Optional[List[str]] = None
    saw: List[Dict[str, Any]] = field(default_factory=list)   # per-input "what the subject saw"
    did: Dict[str, Any] = field(default_factory=dict)          # output summary
    n_inputs: int = 0
    duration_s: float = 0.0
    error: Optional[str] = None

    def to_dict(self):
        return asdict(self)


def _mode(channel: str) -> str:
    family, _ = parse_channel(channel)
    if family == 'neural':
        return 'neural'
    if family == 'behavior':
        return 'behavior'
    if family == 'perturbation':
        return 'intervention'
    return 'action'


class _SessionWatch:
    """The ``observed`` sink for one session: groups inputs by the response that follows."""

    def __init__(self, witness: 'Witness', session) -> None:
        self.witness, self.session = witness, session
        self.inputs: List[Any] = []
        self.started = time.perf_counter()
        task = getattr(session, 'task', None)
        self.instruction = None if task is None else task.instruction
        self.regions = [parse_channel(c)[1] for c in session.outputs
                        if parse_channel(c)[0] == 'neural'] or None

    def record(self, direction: str, event) -> None:
        if event.meta.get('action_status', 'proposed') != 'proposed':
            return              # an execution receipt of the action already recorded
        if direction == 'input':
            if event.channel != CONTROL_CHANNEL:
                if not self.inputs:
                    self.started = time.perf_counter()
                self.inputs.append(event)
            return
        self.witness._add(self._event(_mode(event.channel)), event)
        self.inputs = []
        self.started = time.perf_counter()

    def on_error(self, call, error) -> None:
        self.fail(error)

    def fail(self, error) -> None:
        ev = self._event('action' if self.inputs and self.inputs[-1].channel == 'observation'
                         else 'behavior')
        ev.error = f'{type(error).__name__}: {error}'
        self.witness.events.append(ev)
        self.inputs = []

    def _event(self, mode: str) -> WitnessEvent:
        families = [parse_channel(e.channel)[0] for e in self.inputs]
        if any(f in _INTERVENTION_FAMILIES for f in families):
            event_type = INTERVENTION
        elif 'observation' in families:
            event_type = ENV_STEP
        else:
            event_type = STIMULUS
        ev = WitnessEvent(index=len(self.witness.events), event_type=event_type, mode=mode,
                          instruction=self.instruction, recording_target=self.regions,
                          duration_s=time.perf_counter() - self.started)
        ev.saw, ev.modalities, ev.n_inputs = self.witness._views(self.inputs)
        table = self.session.stimuli
        if not ev.saw and table is not None:
            # The stimuli were read from the session's table, not sent as events.
            ev.n_inputs = len(table)
            ev.saw = [{'modalities': [], 'stimulus_id': str(sid)}
                      for sid in list(table['stimulus_id'])[:self.witness.max_saw]]
        instructions = [e.payload for e in self.inputs if e.channel == 'instruction']
        if instructions:
            ev.instruction = str(instructions[-1])
        return ev


class Witness:
    def __init__(self, subject, label: Optional[str] = None, max_saw: int = 8):
        """:param subject: any subject (it takes part through ``participate``).
        :param label: free-form label for the run (subject · benchmark).
        :param max_saw: the most inputs to keep "saw" detail for, per event."""
        self.subject = subject
        self.label = label or subject.identifier
        self.max_saw = max_saw
        self.events: List[WitnessEvent] = []
        self._saved = None

    # -- watch every session the subject takes part in ----------------------
    def __enter__(self):
        if self._saved is not None:
            raise RuntimeError('A Witness cannot be entered twice concurrently')
        state = vars(self.subject)
        self._saved = ('participate' in state, state.get('participate'))
        original = self.subject.participate

        def participate(session):
            watch = _SessionWatch(self, session)
            from .instrumentation import observed
            try:
                with observed(session, watch):
                    return original(session)
            except BaseException as error:
                watch.fail(error)
                raise

        self.subject.participate = participate
        return self

    def __exit__(self, *exc):
        present, previous = self._saved
        if present:
            self.subject.participate = previous
        else:
            del self.subject.participate
        self._saved = None
        return False

    # -- views of inputs and outputs -----------------------------------------
    def _add(self, ev: WitnessEvent, output) -> None:
        self._record_output(ev, output.payload)
        self.events.append(ev)

    def _views(self, inputs):
        views, rows, modalities = [], {}, set()
        for event in inputs:
            family, address = parse_channel(event.channel)
            if family in _INTERVENTION_FAMILIES:
                change = event.payload
                views.append({'modalities': ['intervention'], 'kind': change.kind,
                              'target': None if change.target is None else str(change.target),
                              'perturbation': (None if change.perturbation is None
                                               else str(change.perturbation))})
                modalities.add('intervention')
                continue
            if family in ('instruction', 'reward'):
                continue                                    # shown in the caption, not as an input
            if family == 'observation':
                views.append(self._observation_view(event.payload))
                modalities.add('embodied')
                continue
            modality = address if family == 'vision' and address == 'video' else _MEDIA_FAMILIES.get(family, family)
            modalities.add(modality)
            key = event.meta.get('table_row', len(rows))
            view = rows.get(key)
            if view is None:
                view = rows[key] = {'modalities': []}
                if 'stimulus_id' in event.meta:
                    view['stimulus_id'] = str(event.meta['stimulus_id'])
                views.append(view)
            view['modalities'].append(modality)
            if modality in ('vision', 'video', 'audio') and isinstance(event.payload, (str, os.PathLike)):
                view[modality] = {'path': str(event.payload)}
            elif modality in ('vision', 'video'):
                view['_frame'] = event.payload                 # kept for rendering; stripped from json
                view['frame_shape'] = list(getattr(event.payload, 'shape', []))
            else:
                view[modality] = {'text': str(event.payload)}
        return views[:self.max_saw], sorted(modalities), len(views)

    @staticmethod
    def _observation_view(observation):
        view = {'modalities': ['embodied']}
        frame = observation.get('frame') if isinstance(observation, dict) else None
        if frame is not None:
            view['frame_shape'] = list(getattr(frame, 'shape', []))
            view['_frame'] = frame                      # kept for rendering; stripped from json
        return view

    def _record_output(self, ev, out):
        dims = getattr(out, 'dims', None)
        if ev.mode == 'action':
            try:
                import numpy as np
                out = np.asarray(out).reshape(-1).tolist()
            except Exception:
                pass
            ev.did = {'kind': 'action', 'action': out}
            return
        if dims is not None:
            ev.did = {'kind': 'neural' if 'neuroid' in dims else 'behavior',
                      'dims': list(dims), 'shape': list(getattr(out, 'shape', []))}
            self._summarize_assembly(ev, out)
            return
        ev.did = {'kind': 'value', 'repr': str(out)[:200]}

    def _summarize_assembly(self, ev, out):
        try:
            import numpy as np
            if ev.did['kind'] == 'neural':
                for c in ('region', 'layer', 'neuroid_id'):
                    if c in out.coords:
                        ev.did[c] = list(map(str, np.unique(out[c].values)))[:6]
                ev.did['value_sample'] = [round(float(x), 4) for x in np.asarray(out.values).ravel()[:5]]
                return
            if 'choice' in out.coords:
                ev.did['labels'] = list(map(str, np.asarray(out['choice'].values).ravel()[:12]))
            vals = np.asarray(out.values)
            if vals.dtype.kind in 'fc':
                ev.did['value_sample'] = [round(float(x), 4) for x in vals.ravel()[:12]]
            else:
                ev.did['prediction_sample'] = list(map(str, vals.ravel()[:8]))
        except Exception as e:
            ev.did['summary_error'] = str(e)

    # -- serialization & rendering ----------------------------------------
    def trace(self) -> List[Dict[str, Any]]:
        out = []
        for ev in self.events:
            d = ev.to_dict()
            for s in d.get('saw', []):
                s.pop('_frame', None)               # numpy frames aren't json-serializable
            out.append(d)
        return out

    def summary(self) -> Dict[str, Any]:
        modes = {}
        for ev in self.events:
            modes[ev.mode] = modes.get(ev.mode, 0) + 1
        return {'label': self.label, 'n_calls': len(self.events),
                'modes': modes, 'total_seconds': round(sum(e.duration_s for e in self.events), 3)}

    def save(self, out_dir: str, render_panels: bool = True, max_panels: int = 12):
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, 'trace.json'), 'w') as f:
            json.dump({'summary': self.summary(), 'events': self.trace()}, f, indent=2, default=str)
        panels = []
        if render_panels:
            from .witness_render import render_event
            step = max(1, len(self.events) // max_panels)
            for ev in self.events[::step][:max_panels]:
                p = os.path.join(out_dir, f'event_{ev.index:04d}.png')
                try:
                    render_event(ev, p, label=self.label)
                    panels.append(p)
                except Exception as e:
                    print(f'witness: failed to render event {ev.index}: {e}')
        return {'trace': os.path.join(out_dir, 'trace.json'), 'panels': panels,
                'summary': self.summary()}
