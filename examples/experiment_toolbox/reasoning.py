"""Record exposed reasoning from a synthetic subject; no model download."""
import argparse
from contextlib import contextmanager
from pathlib import Path

from brainscore.experiments import (
    Experiment, RecordInputsOutputs, RecordReasoning, SessionProtocol,
)
from brainscore.run_record import RunRecord
from brainscore_core.contract import Subject
from brainscore_core.streaming import Session, StreamEvent


class ReasoningSubject(Subject):
    """Answers in three events on behavior:text: two reasoning fragments, then the answer.

    The reasoning goes in the extra data of each event (``meta['reasoning']``),
    where the common channels keep it; it never affects a score. A real
    subject would send the fragments its model streams.
    """

    identifier = 'reasoning-fixture'
    in_channels = frozenset({'text'})
    out_channels = frozenset({'behavior:text'})

    def participate(self, session):
        while (question := session.next_input()) is not None:
            for text in ('Six groups of ', 'seven give forty-two.'):
                session.respond(StreamEvent('behavior:text', '', question.t_ms, {
                    'reasoning': {'text': text, 'kind': 'chain_of_thought', 'format': 'delta'}}))
            session.respond(StreamEvent('behavior:text', '42', question.t_ms, {}))

    def reset(self):
        pass


@contextmanager
def make_session(trial):
    # Ask for a written answer to one question.
    yield Session([StreamEvent('text', 'What is six times seven?', 0)], ['behavior:text'])


def run(output_dir):
    protocol = SessionProtocol(
        'one-question', make_session,
        input_channels=['text'], output_channels=['behavior:text'],
    )
    # Preserve the complete exchange and a focused reasoning record.
    result = Experiment(
        subject=ReasoningSubject(), protocol=protocol,
        tools=[RecordInputsOutputs(), RecordReasoning()], output_dir=output_dir,
    ).run()
    for response in RunRecord(result.directory / 'reasoning').outputs():
        for block in response['reasoning']:
            print(block['format'], block['text'])
        if response['payload'].payload:
            print('Answer:', response['payload'].payload)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    run(parser.parse_args().out)
