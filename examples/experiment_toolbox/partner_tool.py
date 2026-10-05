"""Example user-owned tool: public imports only, no UMI registration or edits."""
from collections import Counter
from contextlib import contextmanager
import json
from brainscore.experiments import Tool


class CountEvents(Tool):
    name = 'event_counts'

    @contextmanager
    def attach(self, context):
        counts = Counter()
        def receive(event):
            counts[event['kind']] += 1
        try:
            with context.subscribe(receive):
                yield
        finally:
            path = context.directory / 'event_counts.json'
            path.write_text(json.dumps(dict(counts), indent=2) + '\n')
            context.artifact(path, producer='partner_tool.CountEvents',
                             description='Counts of delivered experiment events')
