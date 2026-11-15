"""Model provenance: a registration's facts on its responses (``ProvenanceSubject``), and the
note a ``BrainScoreModel`` sends once per session about the parts it used (``session_entry``)."""
import sys
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


def _facts(obj, settings) -> dict:
    """What a part states about itself, read once per session.

    A caching wrapper states the settings fingerprints of its calls (``settings``), where
    its network's parameters are, their dtypes and its batch size. A network recorded by
    forward hooks states where its parameters are and their dtypes. Any other part (a
    plain function, the legacy ``PytorchWrapper``, an answer, reply, likelihood or policy
    callable, a readout classifier) states none of these. The device the inputs are sent
    to is not a fact of the part and is not recorded.
    """
    from brainscore.model_helpers._extractors import CachingExtractor
    facts = {'settings': NOT_STATED, 'parameter_devices': NOT_STATED, 'parameter_dtypes': NOT_STATED,
             'batch_size': NOT_STATED}
    network = getattr(obj, '_model', None) if isinstance(obj, CachingExtractor) else obj
    if isinstance(obj, CachingExtractor) or _is_network(obj):
        parameters = list(network.parameters()) if hasattr(network, 'parameters') else []
        facts.update(parameter_devices=_parameter_devices(parameters),
                     parameter_dtypes=_one(str(p.dtype) for p in parameters))
    if isinstance(obj, CachingExtractor):
        batch_size = getattr(obj, '_batch_size', None)
        facts.update(settings=_one(settings or ()),
                     batch_size=batch_size if batch_size is not None else NOT_STATED)
    return facts


def _is_network(obj) -> bool:
    torch = sys.modules.get('torch')
    return torch is not None and isinstance(obj, torch.nn.Module)


def session_entry(model, session, log) -> dict:
    """The provenance entry of ``session``: every part that ran, as each call site noted it.

    One item per part and the requested outputs its calls served (``serves``); an output
    served by several parts is in each. ``layers`` are the layers the part's calls read;
    ``conversion`` maps an output to the conversion applied to that part's answer. A
    part that ran for no requested output (the readout fit) has ``serves: []``. A requested
    output that no noted call served has ``part: 'not stated'``. Nothing is looked up again.
    """
    items = []
    served = set()
    facts = {}      # the facts of each part object, read once in the session
    for record in log.records.values():
        obj = log.objects[record['object']]
        served |= set(record['serves'])
        if record['object'] not in facts:
            facts[record['object']] = _facts(obj, log.settings.get(record['object']))
        items.append({'part': record['part'], 'kind': _kind(obj), 'serves': list(record['serves']),
                      'layers': sorted(record['layers']), 'conversion': dict(record['conversion']),
                      **facts[record['object']]})
    for channel in sorted(set(session.outputs or ()) - served):
        items.append({'part': NOT_STATED, 'kind': NOT_STATED, 'serves': [channel], 'layers': [],
                      'conversion': {}, **_facts(None, None)})
    items.sort(key=lambda item: (item['serves'], item['part'], item['kind'], item['layers']))
    return {'subject': model.identifier, 'parts': items}
