# Your own package

You can extend Brain-Score from a package of your own, outside its repositories, with no edit to Brain-Score. This page shows the whole path once: register a channel, a subject and a tool, then run an experiment.

## What you need, and where it goes

| What you need | Where it goes |
| --- | --- |
| Another experiment platform (a game engine, a task library) | An adapter or a harness: [Experiments](experiments.md#connect-an-external-experiment-library-an-adapter), [External libraries](external_libraries.md) |
| A new data type that a subject reads or sends | A catalog entry with its payload and a validator: [Extend UMI](extend.md#a-new-channel), [A new kind of output](new_output.md) |
| A model family | Model parts on `BrainScoreModel`, or a subject adapter that implements `participate`: [Wrap your model](wrap_your_model.md) |
| A new way for `BrainScoreModel` to run a session | A capability: [Extend UMI](extend.md#a-new-capability) |
| Measure or change an experiment | A tool: [Extend UMI](extend.md#a-new-tool) |
| A scientific comparison with brain or behavior data | A benchmark and a metric: [Write a benchmark](write_a_benchmark.md) |

## The package

The example is made up: a reader that looks further to the right for a longer word, with no human data. The package registers everything at module level, so the registrations run once, when the package is imported. `register_channel` refuses a second registration of a new entry with the same name, so a function that builds the entry and registers it fails when it is called a second time. `register_tool` refuses a second registration of a tool name, and a registry of `brainscore` refuses a name that another source registered.

```python
# mylab_ext/__init__.py
import numpy as np
import brainscore
from brainscore_core import Subject
from brainscore_core.extensions import CatalogEntry, register_channel
from brainscore_core.streaming import StreamEvent
from brainscore.experiments import Tool
from brainscore.tools import register_tool

register_channel(CatalogEntry(
    'eye_position', 'output', 'StreamEvent', 'gaze in degrees of visual angle, shape (2,)',
    'mylab_ext', owner='my-lab', shape_validator=lambda x: getattr(x, 'shape', None) == (2,)))


class Reader(Subject):
    identifier = 'mylab-reader'
    in_channels = {'text'}
    out_channels = {'eye_position'}

    def participate(self, session):
        while (event := session.next_input()) is not None:
            gaze = np.array([0.5 * len(event.payload), 0.0])
            session.respond(StreamEvent('eye_position', gaze, event.t_ms, dict(event.meta)))


class CountOutputs(Tool):
    name = 'output_count'

    def __init__(self):
        self.count = 0

    def on_event(self, event):
        if event['kind'] == 'output':
            self.count += 1


brainscore.model_registry['mylab-reader'] = Reader
register_tool('mylab-count-outputs', CountOutputs)
```

## Use it by name

Brain-Score has no entry point: it finds your package only after you import it. Your script imports the package first (`import mylab_ext`), then loads by name. On this page, the block above stands for that import.

```python
from contextlib import contextmanager
import json
import pandas as pd
from brainscore.experiments import Experiment, SessionProtocol
from brainscore.tools import load_tool
from brainscore_core.streaming_helpers import stimulus_session
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet

words = StimulusSet(pd.DataFrame({'stimulus_id': ['w0', 'w1'], 'sentence': ['cat', 'elephant']}))
words.identifier = 'two-words'


@contextmanager
def make_session(repetition):
    yield stimulus_session(words, ['eye_position'])


protocol = SessionProtocol('gaze-on-words', make_session, required_input_channels=['text'],
                           requested_output_channels=['eye_position'])
counter = load_tool('mylab-count-outputs')
experiment = Experiment(subject=brainscore.load_model('mylab-reader'), protocol=protocol,
                        tools=[counter])
experiment.validate()          # checks channels and tools; runs nothing
result = experiment.run()
status = json.loads((result.directory / 'experiment.json').read_text())['status']
print(status, counter.count)   # complete 2
```

The session checks each answer against the catalog entry: a gaze of the wrong shape is refused (`payload failed shape_validator`). The experiment keeps the outputs only when a recorder saves them, for example `RecordInputsOutputs` ([Experiments](experiments.md#tools)).

## Test it

Keep the tests in your package. A test imports the package and checks a fact a user depends on through the public names, as the block above does. [examples/partner_tool](../../examples/partner_tool/README.md) is a complete package with its own `pyproject.toml` and tests.
