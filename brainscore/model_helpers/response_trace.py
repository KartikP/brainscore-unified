"""An opt-in generation capability preserving provider-visible responses.

No claim is made that returned text exposes a model's hidden reasoning.
"""
from functools import lru_cache
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from brainscore_core.capabilities.base import Capability
from brainscore_core.capabilities.registry import register_capability
from brainscore_core.extensions import CatalogEntry, StreamEvent, register_channel
from brainscore_core.model_interface import BrainScoreModel

REQUEST_CHANNELS = {
    'generation_request': 'Text generation request',
    'image_generation_request': 'Generation request containing text and images',
    'video_generation_request': 'Generation request containing text, images and video',
    'conditional_logprob_request': 'Conditional log probabilities for targets given contexts',
    'context_choice_request': 'Constrained choice between two contexts for each target',
}

# Request keys a provider may rely on. Providers ignore keys they do not use.
PAYLOAD_CONTRACTS = {
    'generation_request': (
        "dict: messages (list of {role, content}) for one conversation, or conversations "
        "(list of message lists) for a batch; optional system, temperature (0 = greedy), "
        "top_p, max_tokens, seed. Raw response: text (a batch also returns texts), "
        "reasoning, token_usage."),
    'conditional_logprob_request': (
        "dict: targets and contexts (equal-length lists of str), scoring log p(target | context); "
        "optional reduction ('sum', default, or 'mean' over target tokens) and context_format "
        "('raw', default, or 'chat': each context is a user turn). Answer: one float per target."),
}


class ResponseTraceCapability(Capability):
    identifier = 'response-trace'
    input_channels = frozenset({'generation_request'})
    output_channels = frozenset({'response_trace'})

    def __init__(self, input_channels=('generation_request',)):
        self.input_channels = frozenset(input_channels)
        if self.input_channels != {'generation_request'}:
            self.identifier = 'response-trace:' + ','.join(sorted(self.input_channels))

    def enabled_for(self, model):
        return self.identifier in model.capability_config

    def handles(self, model, event, **kwargs):
        return isinstance(event, StreamEvent) and event.channel in self.input_channels

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
                raise ValueError(f'Response trace session requires one of {sorted(self.input_channels)}')
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


@lru_cache(maxsize=None)
def _capability(input_channels):
    return ResponseTraceCapability(input_channels)


def build_trace_subject(
    identifier: str,
    *,
    provider: Callable[[dict], Any],
    parse: Callable[[str], Any],
    provenance: Mapping[str, Any],
    streaming: bool = False,
    input_channels: Sequence[str] = ('generation_request',),
    reset: Callable[[int], None] | None = None,
) -> BrainScoreModel:
    """Build a subject from provider(request)->dict and parse(text)->answer.

    With streaming=True, provider yields dicts containing text and final: bool.
    Intermediate text may be empty; the final fragment contains the complete
    answer text. Optional reasoning blocks declare complete/delta/snapshot format.
    Sessions emit fragments immediately; process() retains them in payload['stream'].

    Provider/model/configuration identity belongs in provenance; never credentials.
    A parse error is preserved as an invalid response, never a fabricated label.
    Provider errors propagate to observers and the caller.

    Declare only request channels the provider actually implements. Text
    generation alone does not imply media support or access to probabilities.
    Optional reset(repetition) prepares the provider for each CogGym repetition.
    """
    if isinstance(input_channels, str) or not input_channels or set(input_channels) - REQUEST_CHANNELS.keys():
        raise ValueError(f'Declare supported request channels from {sorted(REQUEST_CHANNELS)}')
    if reset is not None and not callable(reset):
        raise TypeError('reset must be a reset(repetition) callback')
    channels = tuple(sorted(set(input_channels)))
    register_trace_channels()
    capability = _capability(channels)
    register_capability(capability)
    return BrainScoreModel(identifier, capability_config={capability.identifier: {
        'provider': provider, 'parse': parse, 'provenance': dict(provenance),
        'streaming': streaming, 'reset': reset}})


def register_trace_channels() -> None:
    """Make request schemas known without claiming any model supports them."""
    from brainscore_core import io_catalog
    for name in (*REQUEST_CHANNELS, 'response_trace'):
        direction = 'output' if name == 'response_trace' else 'input'
        if not io_catalog.has(name):
            register_channel(CatalogEntry(name, direction, 'StreamEvent',
                PAYLOAD_CONTRACTS.get(name, 'dict; see response_trace module'),
                REQUEST_CHANNELS.get(name, 'Provider response with raw text and parsed answer'),
                owner='brainscore.response_trace',
                shape_validator=lambda value: isinstance(value, dict)))
        elif io_catalog.get(name).owner != 'brainscore.response_trace':
            raise ValueError(f'Channel {name} already belongs to another extension')


def provider_reset(subject: Any) -> Callable[[int], None]:
    """Find the reset explicitly supplied with a response-trace subject."""
    from brainscore_core.compatibility import CompatibilityError
    if isinstance(subject, BrainScoreModel):
        for name, config in subject.capability_config.items():
            if name == 'response-trace' or name.startswith('response-trace:'):
                if callable(config.get('reset')):
                    return config['reset']
    raise CompatibilityError(
        'Supply reset=provider.reset to build_trace_subject or the CogGym benchmark. '
        'Stateless providers can explicitly use lambda repetition: None.'
    )
