"""Adapt the authors' EWoK model evaluator to a UMI response-trace provider."""
import json
from typing import Any


class EWoKProvider:
    """Delegate tokenization and log probabilities to EWoK.

    ``native_model`` is an instance of ``ewok.evaluate.model.Model`` from the
    pinned paper environment, or an implementation of its ``score`` method.
    Choice mode does not use this provider: it runs on the common channels.
    Put source revision and model settings in subject provenance. The model
    object remains available for ordinary UMI instrumentation.
    """

    def __init__(self, native_model: Any) -> None:
        self.native_model = native_model

    def __call__(self, request: dict) -> dict:
        """Translate a subject's request into a call to the EWoK model evaluator."""
        if request['operation'] != 'ewok.logprobs':
            raise ValueError('Unsupported EWoK provider operation')
        values = [float(value) for value in
                  self.native_model.score(request['targets'], request['contexts'])]
        # build_trace_subject(parse=json.loads) turns this JSON text into the
        # response_trace answer list used by the benchmark. This is the shared
        # provider interface, even when the answers are numbers rather than prose.
        return {'text': json.dumps(values, allow_nan=False)}
