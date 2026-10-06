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

    def _response(self, model, event, raw, *, final=True):
        config = model.capability_config[self.identifier]
        if not isinstance(raw, dict) or not isinstance(raw.get('text'), str):
            raise TypeError('Provider must return a dict containing text: str')
        answer, valid, error = None, None, None
        if final:
            try:
                answer = config['parse'](raw['text'])
                valid = True
            except (ValueError, TypeError) as exc:
                answer, valid, error = None, False, str(exc)
        payload = {'raw': raw, 'answer': answer, 'valid': valid,
                   'parse_error': error, 'provenance': config['provenance']}
        if config.get('streaming', False):
            payload['final'] = final
        return StreamEvent('response_trace', payload, event.t_ms, dict(event.meta))

    def _stream(self, model, event):
        from copy import deepcopy
        config = model.capability_config[self.identifier]
        source = config['provider'](event.payload)
        final_seen = False
        try:
            for index, raw in enumerate(source):
                if final_seen:
                    raise ValueError('Provider emitted a fragment after the final response')
                if not isinstance(raw, dict) or type(raw.get('final')) is not bool:
                    raise TypeError('Stream fragments require a boolean final field')
                # Providers may reuse a buffer; each emitted response is a snapshot.
                raw = deepcopy(raw)
                final_seen = raw['final']
                response = self._response(model, event, raw, final=final_seen)
                response.meta['fragment_index'] = index
                yield response
            if not final_seen:
                raise ValueError('Provider stream ended without a final response')
        finally:
            close = getattr(source, 'close', None)
            if callable(close):
                close()

    def process(self, model, event, **kwargs):
        config = model.capability_config[self.identifier]
        if not config.get('streaming', False):
            return self._response(model, event, config['provider'](event.payload))
        # A direct call returns one result, retaining every raw stream fragment.
        responses = list(self._stream(model, event))
        final = responses[-1]
        final.payload['stream'] = [response.payload['raw'] for response in responses]
        return final

    def supports_session(self, model, channels):
        return set(channels) == {'response_trace'}

    def interact(self, model, session):
        while (event := session.next_input()) is not None:
            if not self.handles(model, event):
                raise ValueError('Response trace session requires generation_request events')
            config = model.capability_config[self.identifier]
            if config.get('streaming', False):
                responses = self._stream(model, event)
                try:
                    for response in responses:
                        session.emit(response)
                finally:
                    responses.close()
            else:
                session.emit(model.process(event))


_CAPABILITY = ResponseTraceCapability()


def build_trace_subject(identifier, *, provider, parse, provenance, streaming=False):
    """Build a subject from provider(request)->dict and parse(text)->answer.

    With streaming=True, provider yields dicts containing text and final: bool.
    Intermediate text may be empty; the final fragment contains the complete
    answer text. Optional reasoning blocks declare complete/delta/snapshot format.
    Sessions emit fragments immediately; process() retains them in payload['stream'].

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
        'provider': provider, 'parse': parse, 'provenance': dict(provenance),
        'streaming': streaming}})
