"""Record provider-exposed reasoning using the shared run-record format."""
from contextlib import contextmanager

from brainscore.run_record import RunRecorder
from brainscore_core.streaming import StreamEvent
from .runner import Tool
from .tools import RecordInputsOutputs


def _trace_reasoning(output):
    """Read the explicit reasoning field; never infer it from answer text."""
    if isinstance(output, StreamEvent) and output.channel == 'response_trace':
        if 'stream' in output.payload:
            values = [raw['reasoning'] for raw in output.payload['stream']
                      if raw.get('reasoning') is not None]
            return [block for value in values for block in _blocks(value)] if values else None
        return output.payload['raw'].get('reasoning')
    return None


def _blocks(value):
    """Normalize exposed text while preserving delta/snapshot distinctions."""
    if value is None:
        return []
    values = value if isinstance(value, list) else [value]
    result = []
    for item in values:
        if isinstance(item, str):
            block = {'text': item}
        elif isinstance(item, dict):
            block = dict(item)
        else:
            block = None
        if block is None or not isinstance(block.get('text'), str):
            raise TypeError('Reasoning must be text, a text block, or a list of text blocks')
        block.setdefault('kind', 'unspecified')
        block.setdefault('format', 'complete')
        if not isinstance(block['kind'], str) or not block['kind']:
            raise ValueError('Reasoning kind must be a nonempty string')
        if block['format'] not in ('complete', 'delta', 'snapshot'):
            raise ValueError('Reasoning format must be complete, delta, or snapshot')
        result.append(block)
    return result


class RecordReasoning(Tool):
    """Save explicitly exposed reasoning with its original response and context.

    Requires RecordInputsOutputs for the linked input and complete experiment log.
    By default reads response_trace.payload['raw']['reasoning']. An optional
    extract(output) callback adapts another response format. It returns text,
    a block, a list of blocks, or None when no reasoning is exposed.

    Blocks contain text, optional kind (e.g. chain_of_thought or summary), and
    format (complete, delta, snapshot). Stream fragments are saved separately;
    this tool neither initiates token streaming nor reconstructs hidden reasoning.
    Read the dedicated reasoning directory through the existing RunRecord API.
    """
    name = 'reasoning'

    def __init__(self, extract=None):
        if extract is not None and not callable(extract):
            raise TypeError('extract must be callable')
        self.extract = _trace_reasoning if extract is None else extract

    def describe(self):
        name = getattr(self.extract, '__qualname__', type(self.extract).__qualname__)
        module = getattr(self.extract, '__module__', type(self.extract).__module__)
        return {
            'schema_version': 1,
            'extractor': f'{module}.{name}',
            'source': 'model/provider-exposed text only',
            'formats': ['complete', 'delta', 'snapshot'],
            'missing_reasoning': 'recorded explicitly as unavailable',
        }

    def validate(self, experiment):
        if not any(isinstance(tool, RecordInputsOutputs) for tool in experiment.tools):
            raise ValueError('RecordReasoning requires RecordInputsOutputs for linked inputs')

    @contextmanager
    def attach(self, context):
        path = context.directory / self.name
        try:
            with RunRecorder(path, metadata={
                'experiment': context.manifest['run_id'],
                'plan': context.manifest['plan'],
                'reasoning_schema_version': 1,
            }) as recorder:
                def capture(event):
                    if event['kind'] != 'output':
                        return
                    value = self.extract(event['payload'])
                    # Serialize immediately: later model mutations cannot change the record.
                    saved = dict(event)
                    saved['reasoning_available'] = value is not None
                    saved['reasoning'] = _blocks(value)
                    recorder.record('output', saved)

                try:
                    with context.subscribe(capture):
                        yield
                finally:
                    if context.failed:
                        recorder.close(failed=True)
        finally:
            for filename in ('manifest.json', 'events.jsonl'):
                artifact = path / filename
                if artifact.is_file():
                    context.artifact(
                        artifact,
                        producer='brainscore.RecordReasoning',
                        description='Exposed reasoning and linked original responses',
                    )
