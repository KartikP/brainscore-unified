"""Witness — watch how Brain-Score runs *any* benchmark on a model.

Every capability in the unified interface funnels through one method,
``BrainScoreModel.process(input_event) -> DataAssembly``. That single chokepoint
is all you need to instrument to see, for any benchmark:

  * **what the model saw**  — the input event, rendered per modality (a stimulus
    image, a text+instruction card, a 2-AFC montage, an embodied frame, or a
    state-change/lesion description);
  * **what the model did**  — the raw output (neuroid activations, label
    probabilities, a generated answer, or a chosen action);
  * **in what mode**        — neural recording vs. behavioral generation vs.
    readout vs. embodied action, inferred from model state at call time.

``Witness`` is non-invasive: it wraps the model's ``process`` (and, when present,
the behavioral generation/readout helpers) for the duration of a ``with`` block,
records one :class:`WitnessEvent` per call, and restores the originals on exit.
Nothing about the benchmark or the model needs to change — point it at any
``benchmark(model)`` call and watch.

    from brainscore.witness import Witness
    with Witness(model, label='CLIP · MajajHong.IT') as w:
        score = benchmark(model)
    w.save('/tmp/run')          # trace.json + one panel PNG per recorded call

Rendering (``w.render`` / ``w.save``) lazily imports matplotlib/PIL so the core
recorder stays dependency-light and importable anywhere.
"""
import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

# Event-type tags
STIMULUS, STATE_CHANGE, ENV_STEP = 'stimulus_set', 'state_change', 'environment_step'

# Fallback column->modality map for models that don't define COLUMN_TO_MODALITY
# (mirrors BrainScoreModel.COLUMN_TO_MODALITY so witnessing works on any model).
_DEFAULT_COLUMN_TO_MODALITY = {
    'image_file_name': 'vision', 'image_path': 'vision', 'filename': 'vision',
    'montage_path': 'vision',
    'sentence': 'text', 'text': 'text',
    'video_path': 'video', 'audio_path': 'audio', 'audio_file_name': 'audio',
    'audio_file': 'audio',
}


@dataclass
class WitnessEvent:
    """One recorded ``process()`` call."""
    index: int
    event_type: str
    mode: str                                   # 'neural' | 'readout' | 'generation' | 'action' | 'state_change'
    modalities: List[str] = field(default_factory=list)
    instruction: Optional[str] = None
    recording_target: Optional[List[str]] = None
    saw: List[Dict[str, Any]] = field(default_factory=list)   # per-stimulus "what the model saw"
    did: Dict[str, Any] = field(default_factory=dict)          # output summary
    n_inputs: int = 0
    duration_s: float = 0.0
    error: Optional[str] = None

    def to_dict(self):
        return asdict(self)


class Witness:
    def __init__(self, model, label: Optional[str] = None, max_saw: int = 8):
        """:param model: a ``BrainScoreModel`` (or anything with ``process``).
        :param label: free-form label for the run (model · benchmark).
        :param max_saw: max stimuli to record "saw" detail for, per call (cap clutter)."""
        self.model = model
        self.label = label or getattr(model, 'identifier', 'model')
        self.max_saw = max_saw
        self.events: List[WitnessEvent] = []
        self._orig_process = None

    # -- context management: patch the instance's process() ----------------
    def __enter__(self):
        self._orig_process = self.model.process

        def wrapped(input_event, *args, **kwargs):
            idx = len(self.events)
            mode = self._infer_mode(input_event)
            ev = WitnessEvent(index=idx, event_type=self._event_type(input_event), mode=mode)
            self._record_input(ev, input_event)
            t0 = time.perf_counter()
            try:
                out = self._orig_process(input_event, *args, **kwargs)
            except Exception as e:                      # record the failure, then re-raise
                ev.error = f"{type(e).__name__}: {e}"
                ev.duration_s = time.perf_counter() - t0
                self.events.append(ev)
                raise
            ev.duration_s = time.perf_counter() - t0
            self._record_output(ev, out)
            self.events.append(ev)
            return out

        self.model.process = wrapped
        return self

    def __exit__(self, *exc):
        if self._orig_process is not None:
            try:
                del self.model.process          # fall back to the class method
            except AttributeError:
                self.model.process = self._orig_process
        return False

    # -- introspection helpers --------------------------------------------
    def _event_type(self, input_event):
        cls = type(input_event).__name__
        if cls == 'StateChange':
            return STATE_CHANGE
        if cls == 'EnvironmentStep':
            return ENV_STEP
        return STIMULUS

    def _infer_mode(self, input_event):
        et = self._event_type(input_event)
        if et == STATE_CHANGE:
            return 'state_change'
        if et == ENV_STEP:
            return 'action'
        m = self.model
        # a model may declare its behavioral path explicitly (e.g. a bespoke
        # 2-AFC driver model that isn't a BrainScoreModel)
        hinted = getattr(m, '_witness_mode', None)
        if hinted:
            return hinted
        tc = getattr(m, '_task_context', None)
        if getattr(m, '_use_generation_for_task', False) and tc is not None:
            return 'generation'
        if getattr(m, '_readout_classifier', None) is not None and tc is not None:
            return 'readout'
        return 'neural'

    def _record_input(self, ev, input_event):
        et = ev.event_type
        m = self.model
        tc = getattr(m, '_task_context', None)
        if tc is not None:
            ev.instruction = getattr(tc, 'instruction', None)
        regions = getattr(m, '_recording_regions', None)
        if regions:
            ev.recording_target = list(regions)

        if et == ENV_STEP:
            ev.n_inputs = 1
            ev.saw.append(self._env_step_view(input_event))
            ev.instruction = getattr(input_event, 'instruction', ev.instruction)
            return
        if et == STATE_CHANGE:
            ev.n_inputs = 1
            ev.saw.append(self._state_change_view(input_event))
            return

        # StimulusSet
        stimuli = input_event
        try:
            ev.modalities = sorted(m._detect_modalities(stimuli))
        except Exception:
            ev.modalities = []
        try:
            ev.n_inputs = len(stimuli)
        except Exception:
            ev.n_inputs = 0
        ev.saw = self._stimulus_views(stimuli, ev.modalities)

    def _stimulus_views(self, stimuli, modalities):
        views = []
        try:
            cols = list(stimuli.columns)
        except Exception:
            cols = []
        # which column carries each modality (defensive: not every UnifiedModel
        # defines COLUMN_TO_MODALITY — fall back to a sensible default mapping)
        col_map = getattr(self.model, 'COLUMN_TO_MODALITY', None) or _DEFAULT_COLUMN_TO_MODALITY
        col_for = {}
        for col in cols:
            mod = col_map.get(col)
            if mod:
                col_for.setdefault(mod, col)
        n = min(len(stimuli), self.max_saw)
        get_stimulus = getattr(stimuli, 'get_stimulus', None)
        sid_col = 'stimulus_id' if 'stimulus_id' in cols else ('image_id' if 'image_id' in cols else None)
        for i in range(n):
            row = stimuli.iloc[i]
            view = {'modalities': modalities}
            for mod, col in col_for.items():
                val = row[col]
                if mod in ('vision', 'video', 'audio'):
                    path = None
                    if get_stimulus is not None and sid_col is not None:
                        try:
                            path = str(get_stimulus(row[sid_col]))
                        except Exception:
                            path = None
                    if path is None:
                        path = str(val)
                    view[mod] = {'path': path}
                else:                                   # text
                    view[mod] = {'text': str(val)}
            if sid_col is not None:
                view['stimulus_id'] = str(row[sid_col])
            views.append(view)
        return views

    def _env_step_view(self, step):
        view = {'modalities': ['embodied']}
        # frame may live in observation['frame'] or cameras
        frame = None
        obs = getattr(step, 'observation', None)
        if isinstance(obs, dict) and 'frame' in obs:
            frame = obs['frame']
        cams = getattr(step, 'cameras', None)
        if frame is None and isinstance(cams, dict) and cams:
            cf = next(iter(cams.values()))
            frame = getattr(cf, 'rgb', cf)
        if frame is not None:
            view['frame_shape'] = list(getattr(frame, 'shape', []))
            view['_frame'] = frame                      # kept for rendering; stripped from json
        view['step_num'] = getattr(step, 'step_num', None)
        return view

    def _state_change_view(self, sc):
        target = getattr(sc, 'target', None)
        perturbation = getattr(sc, 'perturbation', None)
        return {'modalities': ['perturbation'],
                'kind': getattr(sc, 'kind', None),
                'target': str(target) if target is not None else None,
                'perturbation': str(perturbation) if perturbation is not None else None}

    def _record_output(self, ev, out):
        cls = type(out).__name__
        if cls == 'EnvironmentResponse':
            action = getattr(out, 'action', None)
            try:
                import numpy as np
                action = np.asarray(action).reshape(-1).tolist()
            except Exception:
                pass
            ev.did = {'kind': 'action', 'action': action}
            return
        # DataAssembly-like (xarray)
        dims = getattr(out, 'dims', None)
        if dims is not None:
            ev.did = {'kind': self._assembly_kind(out, ev.mode),
                      'dims': list(dims), 'shape': list(getattr(out, 'shape', []))}
            self._summarize_assembly(ev, out)
            return
        ev.did = {'kind': 'value', 'repr': str(out)[:200]}

    def _assembly_kind(self, out, mode):
        dims = set(getattr(out, 'dims', []))
        coords = set(getattr(out, 'coords', {}).keys()) if hasattr(out, 'coords') else set()
        if 'neuroid' in dims:
            return 'neural'
        if 'choice' in dims or 'choice' in coords or mode in ('readout', 'generation'):
            return 'behavior'
        return 'assembly'

    def _summarize_assembly(self, ev, out):
        try:
            import numpy as np
            kind = ev.did['kind']
            if kind == 'neural':
                for c in ('region', 'layer', 'neuroid_id'):
                    if c in getattr(out, 'coords', {}):
                        vals = list(map(str, np.unique(out[c].values)))[:6]
                        ev.did[c] = vals
                flat = np.asarray(out.values).ravel()
                ev.did['value_sample'] = [round(float(x), 4) for x in flat[:5]]
            elif kind == 'behavior':
                # behavioral readout (probabilities) or generation (labels)
                if 'choice' in getattr(out, 'coords', {}):
                    labels = list(map(str, np.asarray(out['choice'].values).ravel()[:12]))
                    ev.did['labels'] = labels
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
