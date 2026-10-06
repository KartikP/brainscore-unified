"""Record an exposed reasoning stream from a synthetic provider; no model download."""
import argparse
from contextlib import contextmanager
from pathlib import Path

from brainscore.experiments import (
    Experiment, RecordInputsOutputs, RecordReasoning, SessionProtocol,
)
from brainscore.model_helpers.response_trace import build_trace_subject
from brainscore.run_record import RunRecord
from brainscore_core.streaming import InMemorySession, StreamEvent


def provider(request):
    # A real provider adapter would yield fragments from its model's stream.
    for text in ('Six groups of ', 'seven give forty-two.'):
        yield {
            'text': '',
            'reasoning': {'text': text, 'kind': 'chain_of_thought', 'format': 'delta'},
            'final': False,
        }
    yield {'text': '42', 'final': True}


@contextmanager
def make_session(trial):
    # Declare the response type requested from the subject.
    session = InMemorySession([
        StreamEvent('generation_request', {'prompt': 'What is six times seven?'}, 0),
    ])
    session.requested_output_channels = ('response_trace',)
    yield session


def run(output_dir):
    # Adapt the provider to UMI and parse only its final answer.
    subject = build_trace_subject(
        'reasoning-fixture', provider=provider, parse=int, streaming=True,
        provenance={'model': 'synthetic fixture', 'trained': False},
    )
    protocol = SessionProtocol(
        'one-question', make_session,
        input_channels=['generation_request'], output_channels=['response_trace'],
    )
    # Preserve the complete exchange and a focused reasoning record.
    result = Experiment(
        subject=subject, protocol=protocol,
        tools=[RecordInputsOutputs(), RecordReasoning()], output_dir=output_dir,
    ).run()
    for response in RunRecord(result.directory / 'reasoning').outputs():
        for block in response['reasoning']:
            print(block['format'], block['text'])
        trace = response['payload'].payload
        if trace['final']:
            print('Answer:', trace['answer'])
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    run(parser.parse_args().out)
