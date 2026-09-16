"""An opt-in generation capability preserving provider-visible responses.

No claim is made that returned text exposes a model's hidden reasoning.
"""

from brainscore_core.capabilities.base import Capability
from brainscore_core.capabilities.registry import register_capability
from brainscore_core.extensions import CatalogEntry, StreamEvent, register_channel
from brainscore_core.model_interface import BrainScoreModel


class ResponseTraceCapability(Capability):
    identifier = 'response-trace'
    input_channels = frozenset({'generation_request'})
    output_channels = frozenset({'response_trace'})

    def enabled_for(self, model):
        return self.identifier in model.capability_config

    def handles(self, model, event, **kwargs):
        return isinstance(event, StreamEvent) and event.channel == 'generation_request'

    def process(self, model, event, **kwargs):
        config = model.capability_config[self.identifier]
        raw = config['provider'](event.payload)
        if not isinstance(raw, dict) or not isinstance(raw.get('text'), str):
            raise TypeError('Provider must return a dict containing text: str')
        try:
            answer = config['parse'](raw['text'])
            valid, error = True, None
        except (ValueError, TypeError) as exc:
            answer, valid, error = None, False, str(exc)
        return StreamEvent('response_trace',
            {'raw': raw, 'answer': answer, 'valid': valid, 'parse_error': error,
             'provenance': config['provenance']}, event.t_ms, dict(event.meta))

    def supports_session(self, model, channels):
        return set(channels) == {'response_trace'}

    def interact(self, model, session):
        while (event := session.next_input()) is not None:
            if not self.handles(model, event):
                raise ValueError('Response trace session requires generation_request events')
            session.emit(model.process(event))


_CAPABILITY = ResponseTraceCapability()


def build_trace_subject(identifier, *, provider, parse, provenance):
    """Build a subject from provider(request)->dict and parse(text)->answer.

    Provider/model/configuration identity belongs in provenance; never credentials.
    A parse error is preserved as an invalid response, never a fabricated label.
    Provider errors propagate to observers and the caller.
    """
    from brainscore_core import io_catalog
    for name, direction, validator in (
            ('generation_request', 'input', lambda value: isinstance(value, dict)),
            ('response_trace', 'output', lambda value: isinstance(value, dict))):
        if not io_catalog.has(name):
            register_channel(CatalogEntry(name, direction, 'StreamEvent',
                'dict; see response_trace module', 'response-trace capability',
                owner='brainscore.response_trace', shape_validator=validator))
        elif io_catalog.get(name).owner != 'brainscore.response_trace':
            raise ValueError(f'Channel {name} already belongs to another extension')
    register_capability(_CAPABILITY)
    return BrainScoreModel(identifier, capability_config={'response-trace': {
        'provider': provider, 'parse': parse, 'provenance': dict(provenance)}})
