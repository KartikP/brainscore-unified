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

Experiments accept unregistered channel names with a warning; benchmark scoring requires registration. An input channel can also claim stimulus-table columns with `columns=('gaze_target',)`.

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

A `Capability` adds a new kind of input handling to `BrainScoreModel` itself, with its own channels. Subclass `Capability` and register it with `register_capability`, both from `brainscore_core.capabilities`. Use this only when the behavior belongs inside the subject; for observing or changing runs, write a tool.
