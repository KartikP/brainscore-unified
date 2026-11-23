"""Template: a new capability -- a new kind of request a model can answer.

SYNTHETIC EXAMPLE. The word counter below only demonstrates the wiring; it
measures nothing about brains or models. Replace the counting with your behavior.

A capability adds input and output channels to the models that opt in. It needs
three pieces, none of which edit Brain-Score's core:

1. Typed payloads: small dataclasses for what goes in and what comes out.
2. Catalog entries: register each channel name with its payload type, so a
   wrong payload is rejected before any model sees it.
3. A ``Capability`` subclass: which sessions it runs (``supports_session``)
   and how it answers each request (``participate``).

Call ``register()`` once from your package's setup code. A second call raises
``ValueError`` instead of silently replacing a registered name. Only a model
built with the capability (``BrainScoreModel(capabilities=[...])``) gains the
channels; every other model is unchanged.
"""
from dataclasses import dataclass

from brainscore.capabilities import Capability
from brainscore_core.io_catalog import CatalogEntry, register_channel
from brainscore_core.streaming import StreamEvent
from brainscore import BrainScoreModel

IDENTIFIER = 'example-word-count'   # TODO: your capability's name
REQUEST = 'example_word_request'    # TODO: input channel (lowercase, digits, underscores)
RESULT = 'example_word_count'       # TODO: output channel


@dataclass(frozen=True)
class WordRequest:
    """What the experiment sends: the text to count."""
    text: str


@dataclass(frozen=True)
class WordCount:
    """What the model sends back: words in this request and since the last reset."""
    words: int
    words_so_far: int


def _valid_request(payload) -> bool:
    # The catalog calls this before any model sees a request; False rejects it.
    return isinstance(payload, WordRequest) and isinstance(payload.text, str)


def _valid_count(payload) -> bool:
    # Counts are whole numbers (not True/False), never negative, and the
    # running total includes this request's words.
    return (isinstance(payload, WordCount)
            and type(payload.words) is int and type(payload.words_so_far) is int
            and 0 <= payload.words <= payload.words_so_far)


def _check_min_length(value) -> int:
    # Words shorter than this are not counted. It must be a whole number >= 1.
    if type(value) is not int or value < 1:
        raise ValueError(f"{IDENTIFIER}: min_length must be a whole number >= 1; got {value!r}")
    return value


class WordCountCapability(Capability):
    identifier = IDENTIFIER
    input_channels = frozenset({REQUEST})
    output_channels = frozenset({RESULT})

    def __init__(self, min_length: int = 1) -> None:
        # Settings are checked here, so bad settings fail before anything is counted.
        # The running total is this instance's state: give each model its own instance.
        self.min_length = _check_min_length(min_length)
        self.words_so_far = 0

    def reset(self, subject) -> None:
        # subject.reset() starts the count again; the settings stay.
        self.words_so_far = 0

    def supports_session(self, subject, channels) -> bool:
        # Claim sessions that ask for exactly our output channel.
        return set(channels) == {RESULT}

    def participate(self, subject, session) -> None:
        # Read each request the session delivers and send one answer back.
        # The session checks both payloads against the catalog entries.
        event = session.next_input()
        while event is not None:
            session.respond(self._count(event))
            event = session.next_input()

    def _count(self, event) -> StreamEvent:
        # Answer one request. TODO: replace the counting with your behavior.
        words = [word for word in event.payload.text.split()
                 if len(word) >= self.min_length]
        self.words_so_far += len(words)
        return StreamEvent(
            channel=RESULT,
            payload=WordCount(words=len(words), words_so_far=self.words_so_far),
            t_ms=event.t_ms,
            meta=dict(event.meta),
        )


def register() -> None:
    """Add the two channels. Call once."""
    register_channel(CatalogEntry(
        name=REQUEST,
        direction='input',
        carried_by='StreamEvent',
        payload_contract='WordRequest(text: str)',
        handled_by=IDENTIFIER,
        owner='your-package',   # TODO: your package name
        payload_type=WordRequest,
        shape_validator=_valid_request,
    ))
    register_channel(CatalogEntry(
        name=RESULT,
        direction='output',
        carried_by='StreamEvent',
        payload_contract='WordCount(words: int, words_so_far: int)',
        handled_by=IDENTIFIER,
        owner='your-package',
        payload_type=WordCount,
        shape_validator=_valid_count,
    ))


def build_model(identifier: str = 'example-word-counter',
                min_length: int = 1) -> BrainScoreModel:
    """A model that answers word-count requests; it needs no feature extractor."""
    return BrainScoreModel(
        identifier=identifier,
        # This model gets its own capability, with its settings.
        capabilities=[WordCountCapability(min_length)],
    )
