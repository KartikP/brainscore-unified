"""Use the authors' EWoK model evaluator as a ``likelihood`` part."""
from collections.abc import Sequence
from typing import Any


class EWoKProvider:
    """Score options with EWoK's own ``score``: a ``likelihood`` part.

    ``native_model`` is an instance of ``ewok.evaluate.model.Model`` from the
    pinned paper environment, or an implementation of its ``score`` method.
    ``BrainScoreModel(likelihood=EWoKProvider(native), in_channels={'text'})``
    runs EWoK's log-probability mode with EWoK's own tokenization and scoring:
    for a trial whose text is a context, each option's score is
    ``native_model.score([option], [context])``, called once per trial for all
    options. Put source revision and model settings in the model's provenance.
    The native model object remains available for ordinary UMI instrumentation.
    """

    def __init__(self, native_model: Any) -> None:
        self.native_model = native_model

    def __call__(self, prompt, candidates: Sequence[str]) -> dict:
        """The EWoK score of each candidate after the trial's text."""
        texts = [content for kind, content in prompt.parts if kind == 'text']
        if len(texts) != 1 or len(texts) != len(prompt.parts):
            raise ValueError('EWoK scores one text part; got '
                             f'{[kind for kind, _ in prompt.parts]}')
        candidates = list(candidates)
        values = self.native_model.score(candidates, [texts[0]] * len(candidates))
        return {candidate: float(value) for candidate, value in zip(candidates, values)}
