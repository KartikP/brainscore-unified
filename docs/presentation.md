# Optional stimulus presentation prototype

This proposal is separate from the required UMI interface. It adds no methods to
subjects, sessions, protocols, or environments.

A **screen** presents an image; a **speaker** presents a sound. Here they are
software transformations of decoded inputs, not physical devices. An environment
still owns its world and action loop. A model's preprocessor still prepares the
presented input for its architecture.

```text
Stimulus or environment observation
    -> Presentation: position, contrast, waveform amplitude
    -> Session: deliver the resulting event
    -> Subject: prepare model inputs, compute, respond
```

## Interface

```python
from brainscore.presentation import Presentation, Screen, Speaker

present = Presentation(
    components={
        'vision': [Screen(height=224, width=224, top=32, left=32, contrast=0.5)],
        'audio': [Speaker(gain=0.25)],
    },
)

# Each event keeps its channel, timestamp, and stimulus metadata.
# The payload changes; the original event remains untouched.
delivered_event = present(source_event)
```

| Component | Input | Actual effect |
| --- | --- | --- |
| `Screen` | Float image in [0, 1], shaped HxW or HxWxC | Places it on a canvas; contrast scales differences from the background. Oversized images raise rather than crop. |
| `Speaker` | `{'samples': waveform, 'sample_rate': Hz}`; float samples in [-1, 1] | Scales amplitude by `gain` in [0, 1]. Preserves sample rate and sample count. |
| `Presentation` | `StreamEvent` | Applies components in list order to the exact configured channel; saves their settings in `meta['presentation']`. |

Channels without configured components pass through as independent copies. An
addressed channel such as `vision:left` must be configured under that exact name.
There is no implicit camera selection or channel conversion.

## Use the existing session and protocol

```python
from contextlib import contextmanager
import numpy as np
from brainscore_core.streaming import BufferedSession, StreamEvent
from brainscore.experiments import SessionProtocol

@contextmanager
def make_session(trial):
    source = StreamEvent(
        channel='vision',
        payload=np.ones((160, 160, 3), dtype=np.float32),
        t_ms=0,
        meta={'stimulus_id': 'white-square'},
    )
    yield BufferedSession(inputs=[present(source)])

protocol = SessionProtocol(
    identifier='present-a-square',
    session_factory=make_session,
    input_channels=['vision'],
    output_channels=['behavior'],
)
```

Run this protocol with a compatible subject using the existing `Experiment`.
`RecordInputsOutputs` records the delivered pixels and presentation settings.
Existing replay uses these saved inputs directly, so contrast or gain is not
applied a second time. Existing activity and intervention tools attach to the
subject as usual; this module does not grant new model instrumentation access.

For a simulator, its observation adapter can call the same presentation function
before delivering the sensory event. The simulator retains action, termination,
and clock ownership. This prototype does not automatically modify nested camera
fields in `EnvironmentStep` or the built-in `EnvironmentSession`.

## Extend without changing UMI

A component needs only `__call__(payload)` and `describe()`. The latter returns
JSON-serializable settings. No registry or base class is required.

```python
class InvertImage:
    def __call__(self, image):
        return 1 - image

    def describe(self):
        return {'component': 'InvertImage', 'version': 1}
```

Use deterministic components with descriptions that include all settings needed
to understand the transformation. Implement model-specific tokenization and
normalization in the extractor, not in a presentation component.

## Boundaries and review questions

- Timing is preserved, not scheduled. Equal timestamps do not make two model calls simultaneous.
- Pixel coordinates are not visual degrees. Existing `place_on_screen` calibration is not replaced.
- Waveform amplitude is not physical loudness or sound-pressure level.
- Paths and StimulusSet rows must be decoded upstream. This is not a new dataset loader.
- Transforming an event copies its payload to protect the source. Large videos or
  live streams need separate memory/performance qualification.
- No simulator, physical display, hardware audio, camera projection, or acoustics is implemented.

Review whether these concrete transformations justify a shared public module,
whether the screen/speaker names clarify their role, and whether event metadata
is the right place to retain presentation settings. The prototype can remain
unmerged without changing ordinary UMI use.

The example payloads target native subjects that accept decoded image/audio
arrays. Existing file-based extraction wrappers are not automatically converted
to accept them; their integration must preserve the declared input format.
