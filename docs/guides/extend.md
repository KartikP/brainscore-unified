# Extend UMI

You can add channels, tools and capabilities from your own package, without editing Brain-Score. Import your registration module before building models. [Your own package](your_own_package.md) shows the whole path, with a table of what goes where. Parts of this are still evolving; see [Feature status](../reference/features.md).

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

A tool observes or changes an experiment. Subclass `Tool` and override what you need.
`templates/new_tool/` is a complete tool with its test.

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

The experiment calls `on_event` with one dict per event. A tool keeps its result on
itself, as `count` here. Copy payloads before keeping them; never modify them.

| Key | Value |
| --- | --- |
| `kind` | what happened: `repetition_start`, `session_context`, `input`, `output`, `error`, `repetition_end`; tools add `intervention`, `intervention_start`, `intervention_end`, `activity`, `patch_reference` |
| `payload` | for `input` and `output` of a session, the `StreamEvent` (`channel`, `payload`, `t_ms`, `meta`); for the other kinds, a dict |
| `source` | who published it: `'protocol'`, or the `name` of a tool |
| `condition`, `repetition_id` | the condition and the repetition the event belongs to |
| `event_id` | one identifier per input; its outputs carry the same one |
| `sequence`, `elapsed_s` | the order of the event in the repetition, and seconds since the run started |

A tool that needs outputs from the subject names them in `requested_output_channels`
(for example `frozenset({'neural:IT'})`); `TaskProtocol` requests them. For resources or
saved files, override `attach(context)` as a context manager; add `validate` and
`describe` for requirements and configuration.

**Register it.** A tool that ships in Brain-Score is one module, `brainscore/tools/<name>.py`,
and one line in the table `_TOOLS` in `brainscore/tools/_registry.py`: the module, the
class, and whether `artifact=` loads a saved tool. The module is imported only when
`load_tool` asks for it, so importing the package imports no tool:

```text
'linear-probe': ('brainscore.tools.linear_probe', 'LinearProbe', True),
```

A tool in your own package is registered with `register_tool(identifier, factory)` (from
`brainscore.tools`); `load_tool(identifier, **configuration)` then calls the factory:

```python
from brainscore.tools import load_tool, register_tool
register_tool('count-outputs', CountOutputs)          # a class or any function that returns a tool
print(load_tool('count-outputs').count)               # 0
```

A complete external package is in [examples/partner_tool](../../examples/partner_tool/README.md).

## A new instrumentation backend

To let `RecordActivity` and `Ablate` reach a framework other than PyTorch, implement an object with `describe`, `validate`, `record` and `ablate`. The last two return context managers that clean up even after a partial failure.

## A new capability

A `Capability` adds a new kind of input handling to `BrainScoreModel` itself, with its own channels. Subclass `Capability` and register it with `register_capability`, both from `brainscore.capabilities`. Use this only when the behavior belongs inside the subject; for observing or changing runs, write a tool. A new output often needs no capability: see [A new kind of output](new_output.md#a-channel-or-a-capability).

A capability runs sessions. It claims the outputs a session requests (`supports_session(subject, channels)`) and runs it (`participate(subject, session)`); `subject.participate(session)` selects it.

Every method takes the model first, named `subject`. Per-model state comes from `setup(subject)` and is read back with `subject.capability_state(identifier)`; settings come from `subject.capability_config`, which holds what the registration passes as `capability_config={identifier: settings}`.

One registry holds every capability. For each session the model selects the claimant with the lowest `order`; two claimants at that order are an error. Added capabilities default to 1000 and the built-in ones are 10000, so an added capability runs a session that a built-in one also claims. See `templates/new_capability/`.
