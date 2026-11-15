"""Model provenance: a registration's facts on its responses (``ProvenanceSubject``), and the
note a ``BrainScoreModel`` sends once per session about the parts it used (``session_entry``)."""
from copy import deepcopy
from dataclasses import replace
from typing import Mapping, Any

from brainscore_core.contract import Subject
from brainscore_core.streaming import Session


def _attach(provenance, event):
    from xarray import DataArray

    payload = event.payload
    if isinstance(payload, DataArray):
        payload = payload.assign_attrs(provenance=deepcopy(provenance))
    return replace(event, payload=payload,
                   meta={**event.meta, 'provenance': deepcopy(provenance)})


class ProvenanceSubject(Subject):
    """A response-metadata decorator; all declarations and lifecycle delegate.

    This adds no model ability or evaluation path. Assemblies share their data
    with the original response and receive a separate attrs dictionary. Each
    response gets its own metadata copy so consumers cannot rewrite later runs.
    """

    def __init__(self, subject: Subject, provenance: Mapping[str, Any]) -> None:
        self._subject = subject
        self._provenance = deepcopy(dict(provenance))

    @property
    def identifier(self) -> str:
        return self._subject.identifier

    @property
    def in_channels(self) -> set[str]:
        return self._subject.in_channels

    @property
    def out_channels(self) -> set[str]:
        return self._subject.out_channels

    @property
    def required_channels(self) -> set[str]:
        return self._subject.required_channels

    @property
    def traits(self) -> set[str]:
        return self._subject.traits

    def participate(self, session: Session) -> None:
        from brainscore_core.score_metadata import note_session
        # The registration's facts are their own provenance entry; the model notes its parts.
        note_session(session, provenance={'subject': getattr(self._subject, 'identifier', None),
                                          'registration': self._provenance})
        with session.rewriting_responses(lambda event: _attach(self._provenance, event)):
            self._subject.participate(session)

    def reset(self) -> None:
        self._subject.reset()


# The provenance note of one session of a BrainScoreModel (rule 6.3). The model sends it
# once, at the end of the session, with core's note_session; no response is changed.

NOT_STATED = 'not stated'


def _kind(part) -> str:
    """``<module>.<qualname>`` of a function or method, or of the class of any other part."""
    import inspect
    if part is None:
        return NOT_STATED
    named = part if inspect.isfunction(part) or inspect.ismethod(part) else type(part)
    module, qualname = getattr(named, '__module__', None), getattr(named, '__qualname__', None)
    return f'{module}.{qualname}' if module and qualname else NOT_STATED


def _one(values):
    """One value, a sorted list when the session saw several, or not stated when none."""
    values = sorted({str(value) for value in values})
    return NOT_STATED if not values else values[0] if len(values) == 1 else values


def _parameter_devices(parameters):
    """Where the part's parameters are: one device, or a sorted list.

    ``'not stated'`` when the parameters cannot tell where the part runs: none, a
    parameter on the meta device, or parameters on more than one device type (an
    automatic device map with offload keeps some on the CPU or on meta and moves
    them in for each call). Several devices of one type (two GPUs) are listed.
    """
    devices = {p.device for p in parameters}
    types = {device.type for device in devices}
    if not devices or 'meta' in types or len(types) > 1:
        return NOT_STATED
    return _one(devices)


def _extractor_facts(extractor, facts) -> dict:
    """The facts of an extractor: a caching wrapper states them, any other part does not.

    ``facts`` is the set of settings fingerprints its calls of this session gave
    (``CachingExtractor._session_facts``), or None for a part that keeps none.
    ``parameter_devices`` and ``parameter_dtypes`` are read from the part's
    parameters at the end of the session; the device the inputs are sent to is
    not a fact of the part and is not recorded.
    """
    from brainscore.model_helpers._extractors import CachingExtractor
    if not isinstance(extractor, CachingExtractor) or facts is None:
        return {'kind': _kind(extractor), 'settings': NOT_STATED, 'parameter_devices': NOT_STATED,
                'parameter_dtypes': NOT_STATED, 'batch_size': NOT_STATED}
    model = getattr(extractor, '_model', None)
    parameters = list(model.parameters()) if hasattr(model, 'parameters') else []
    batch_size = getattr(extractor, '_batch_size', None)
    return {'kind': _kind(extractor), 'settings': _one(facts),
            'parameter_devices': _parameter_devices(parameters),
            'parameter_dtypes': _one(str(p.dtype) for p in parameters),
            'batch_size': batch_size if batch_size is not None else NOT_STATED}


def noted_extractors(model) -> list:
    """The distinct extractors of ``model`` (single-type and fused), in registration order."""
    seen, extractors = set(), []
    for extractor in list(model._extractors.values()) + [e for _, e in model._fused_extractors]:
        if id(extractor) not in seen:
            seen.add(id(extractor))
            extractors.append(extractor)
    return extractors


def _serving_extractor(model, stimuli, region_or_layer: str, modality=None):
    """The extractor that records a region (or a layer) from ``stimuli``: the run's resolver.

    ponytail: this reads the dispatch resolver (``extractor_for``, ``fused_for``); the
    session plan, when it lands, names the parts it chose and replaces this lookup.
    """
    from brainscore._model import dispatch
    if stimuli is not None:
        fused = dispatch.fused_for(model, model.detect_modalities(stimuli))
        if fused is not None:
            return fused
    layer = model._region_layer_map_dict.get(region_or_layer, region_or_layer)
    if modality is None:
        modality = model._region_modality_map.get(region_or_layer)
    try:
        return dispatch.extractor_for(model, modality, stimuli, [layer])
    except Exception:
        return None


def _generation_part(model, task) -> str:
    """The generation part that answers ``task``: ``answer`` for a plain Choice when wired, else ``reply``.

    ponytail: the rule of ``BehavioralReadout`` (``by_answer``); the session plan will name it.
    """
    from brainscore_core.responses import Choice
    response = getattr(task, 'response', None)
    by_answer = (model._answer is not None and isinstance(response, Choice)
                 and not response.verbatim and getattr(task, 'present', None) is None
                 and not isinstance(response.labels, str) and model._memory != 'trial_history')
    if by_answer or model._reply is None:
        return 'answer'
    return 'reply'


_PROBABILITY_FORMATS = frozenset({'behavior:probabilities', 'behavior:rt'})


def session_entry(model, session, facts_by_extractor) -> dict:
    """The provenance entry of ``session``: each part it used, with the facts it can state.

    One part per requested output: the extractor that records a ``neural:<region>``;
    for a behavior format, the part that answered it (``readout`` with its extractor,
    ``likelihood``, ``answer`` or ``reply``); the policy for an action. Every part
    has the same keys; a fact a part cannot give is ``'not stated'``.
    """
    stimuli = getattr(session, 'stimuli', None)
    behavioral = model._behavioral
    parts = []
    for channel in sorted(session.outputs or ()):
        if channel in model._policy_out_channels:      # the policy answers the channels it declares
            parts.append({'serves': channel, 'part': 'policy', **_extractor_facts(model._policy, None)})
        elif channel.startswith('neural:'):
            extractor = _serving_extractor(model, stimuli, channel.split(':', 1)[1])
            parts.append({'serves': channel, 'part': 'extractor',
                          **_extractor_facts(extractor, facts_by_extractor.get(id(extractor)))})
        elif channel == 'behavior' or channel.startswith('behavior:'):
            path = behavioral.response_path
            if channel in _PROBABILITY_FORMATS and behavioral.probability_source:
                path = behavioral.probability_source
            if channel == 'behavior:log_probabilities':
                path = 'likelihood'
            if path == 'readout':
                extractor = _serving_extractor(model, stimuli, model._behavioral_readout_layer)
                parts.append({'serves': channel, 'part': 'readout', 'layer': model._behavioral_readout_layer,
                              **_extractor_facts(extractor, facts_by_extractor.get(id(extractor)))})
                continue
            part = (_generation_part(model, getattr(session, 'task', None)) if path == 'generation'
                    else path if path in ('likelihood',) else None)
            callable_part = {'answer': model._answer, 'reply': model._reply,
                             'likelihood': model._likelihood}.get(part)
            parts.append({'serves': channel, 'part': part or NOT_STATED, **_extractor_facts(callable_part, None)})
        else:
            parts.append({'serves': channel, 'part': NOT_STATED, **_extractor_facts(None, None)})
    return {'subject': model.identifier, 'parts': parts}
