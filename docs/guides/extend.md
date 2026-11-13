# Extend UMI

You can add channels, tools and capabilities from your own package, without editing Brain-Score. Import your registration module before building models. Parts of this are still evolving; see [Feature status](../reference/features.md).

## A new channel

```python
from brainscore_core.extensions import CatalogEntry, register_channel

register_channel(CatalogEntry(
    'eye_position', 'output', 'StreamEvent',                # name, direction, carried by
    'Gaze position in degrees of visual angle, shape (2,)',  # payload description
    'my_package',                                            # who interprets it
    owner='my-lab',
    shape_validator=lambda x: getattr(x, 'shape', None) == (2,),
))
```

Experiments and benchmark scoring both refuse a channel that is not registered.

An input channel can also claim stimulus-table columns. Each column then feeds the channel: a table session sends it on that channel, pre-flight accepts the channel, and a subject that does not declare the channel does not get the column.

```python
register_channel(CatalogEntry('odor', 'input', 'StimulusSet column', 'an odorant name', 'my_package'),
                 columns=('odor_name',))
```

One table says which channel each column feeds: `brainscore_core.io_catalog.stimulus_columns()`. Core's columns are there too, for example `image_path` feeds `vision:image`, `video_path` feeds `vision:video`, `sentence` feeds `text` and `audio_path` feeds `audio`. A column that already feeds a channel needs `replace=True`.

## A new tool

A tool observes or changes an experiment. Subclass `Tool` and override what you need:

```python
from brainscore.experiments import Tool

class CountOutputs(Tool):
    name = 'output_count'

    def __init__(self):
        self.count = 0

    def on_event(self, event):
        if event['kind'] == 'output':
            self.count += 1
```

For resources or saved files, override `attach(context)` as a context manager; add `validate` and `describe` for requirements and configuration. Copy payloads before keeping them; never modify them. `register_tool(identifier, factory)` makes a tool loadable by name. A complete external package is in [examples/partner_tool](../../examples/partner_tool/README.md).

## A new instrumentation backend

To let `RecordActivity` and `Ablate` reach a framework other than PyTorch, implement an object with `describe`, `validate`, `record` and `ablate`. The last two return context managers that clean up even after a partial failure.

## A new capability

A `Capability` adds a new kind of input handling to `BrainScoreModel` itself, with its own channels. Subclass `Capability` and register it with `register_capability`, both from `brainscore.capabilities`. Use this only when the behavior belongs inside the subject; for observing or changing runs, write a tool.

A capability has one role. A session capability claims the outputs a session requests (`supports_session(subject, channels)`) and runs it (`participate(subject, session)`); `subject.participate(session)` reaches it. An added capability normally takes this role. An event capability claims one event (`handles(subject, event)`) and answers it (`process(subject, event)`); the built-in ones are the steps the session drivers reach through the model's `process`. Implement one pair; the other declines by default.

Every method takes the model first, named `subject`. Per-model state comes from `setup(subject)` and is read back with `subject.capability_state(identifier)`; settings come from `subject.capability_config`, which holds what the registration passes as `capability_config={identifier: settings}`.

One registry holds every capability and tries them in `order`, lower first, ties by `identifier`. Added capabilities default to 1000. The built-in event capabilities are 50 to 1000 and the built-in session capabilities 10000, so an added capability runs a session that a built-in one also claims. See `templates/new_capability/`.
