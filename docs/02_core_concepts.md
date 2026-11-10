# Core concepts

These are the eight ideas the rest of UMI is built from, in the order they depend on each other. Each section gives a definition, the reason it exists, and a few lines of code. The [lessons](../notebooks/README.md#lessons) run the same code end to end.

## 1. Subject

A **subject** is the participant in an experiment. It is any Python object that subclasses `Subject` and provides:

| Member | Meaning |
| --- | --- |
| `identifier` | A unique name. |
| `in_channels` | The input channels it can read. |
| `out_channels` | The output channels it can produce. |
| `participate(session)` | The one evaluation method: read inputs, send responses, return when inputs run out. |
| `reset()` | Optional. Clears run state (history, active perturbations) between independent runs. |

```python
from brainscore_core import Subject
from brainscore_core.streaming import StreamEvent

class WordLengthReader(Subject):
    identifier = 'word-length-reader'
    in_channels = {'text'}
    out_channels = {'behavior'}

    def participate(self, session):
        while (event := session.next_input()) is not None:
            session.respond(StreamEvent('behavior', 200 + 30 * len(event.payload), t_ms=event.t_ms))
```

The experiment only sees this interface, so a CNN, a language model, a robot policy and a remote API are all handled the same way.

## 2. Channel

A **channel** names one kind of information flowing into or out of a subject. Names are written `family` or `family:address`.

| Inputs | Outputs |
| --- | --- |
| `vision` (images and video), `text`, `audio`, `observation` (environment state), `instruction`, `lesion`, `stimulation` | `behavior` (choices, ratings), `neural:<region>` (activity standing in for a brain region, such as `neural:IT`), `motor` (actions), `perturbation` |

Channel names come from a shared catalog, so a benchmark and a subject mean the same thing by `vision`. You can register new families (see [Extend UMI](guides/extend.md)).

## 3. Event

A **`StreamEvent`** is one item on a channel: `channel`, `payload` (the data), `t_ms` (time in milliseconds, or `None`) and optional `meta`.

```python
StreamEvent(channel='text', payload='elephant', t_ms=0)
```

## 4. Session

A **session** is one run of an experiment with one subject. It has two methods: `next_input()` returns the next input event or `None` when there are no more, and `respond(event)` accepts a response. `collect()` returns the responses afterwards.

```python
from brainscore_core.streaming import BufferedSession

session = BufferedSession(inputs=[StreamEvent('text', w, t_ms=i * 500) for i, w in enumerate(['cat', 'elephant'])])
WordLengthReader().participate(session)
session.collect()
```

**Sessions do not have to stream.** One event can carry a whole batch, and `BufferedSession(stimulus_set=...)` delivers a full stimulus table. Streaming one input at a time, time windows and real-time clocks are options, not requirements. Session types:

| Session | Use it when |
| --- | --- |
| `BufferedSession` | Inputs are fixed in advance (most benchmarks). |
| `EnvironmentSession` | Each action changes the next input (games, simulators, robots). |
| `StreamingStimulusSetSession`, `WindowedStreamSession`, `RealTimeStreamSession` | Delivery timing matters. Advanced. |

## 5. Benchmark, metric and score

A **benchmark** is a fixed test: it presents stimuli, requests outputs, and compares them with measured data using a **metric**, returning a **`Score`**. A benchmark is called with the subject.

```python
from brainscore_core import Benchmark, Score

class ToyReadingTimes(Benchmark):
    identifier = 'toy-reading-times'
    uses_session = True
    required_input_channels = {'text'}
    requested_output_channels = {'behavior'}

    def __call__(self, subject):
        session = BufferedSession(inputs=[...])   # present the stimuli
        subject.participate(session)
        return Score(my_metric(session.collect(), measured_data))
```

A **raw** score is the metric value itself. A **ceiled** score divides by the benchmark's estimate of how well the data can be predicted at all, and can exceed 1. Report which one you use.

## 6. Compatibility

Before a benchmark runs, Brain-Score compares declarations only:

- every requested output channel must be in the subject's `out_channels`;
- at least one provided input channel must be in the subject's `in_channels`;
- every input the subject requires must be provided.

A pair that fails is **not applicable** (N/A). It is not run and never scored as zero. A subject that reads only some of the provided inputs still runs; the other inputs never reach it.

Each role names its channel fields differently; use the one for the object you are writing:

| Role | Inputs | Outputs |
| --- | --- | --- |
| Subject (model) | `in_channels` (plus `required_channels`) | `out_channels` |
| Benchmark | `required_input_channels` | `requested_output_channels` |
| `SessionProtocol` | `input_channels` | `output_channels` |

**Stills and clips.** `vision` is the one visual channel, and a subject also says which forms it takes natively: `vision:image` (pictures) and `vision:video` (video). A benchmark shows each subject the stimulus in the form people saw it when the subject has that form, and adapts it only when it doesn't (for example, frames from each clip for a picture-only model), recording the adaptation on the score as `attrs['input_adaptation']`.

```python
import brainscore
brainscore.score(WordLengthReader(), ToyReadingTimes())   # checks compatibility, then runs
```

## 7. BrainScoreModel and its parts

Writing `participate` by hand works for anything. For neural networks, `BrainScoreModel` is a ready-made subject assembled from parts. Each part you supply adds channels.

| Part | What it does | Adds |
| --- | --- | --- |
| `extractor={'vision': ..., 'text': ...}` | Prepares inputs, runs the network, returns the activity of requested layers. One per input type. | input channels |
| `region_layer_map={'IT': 'layer4'}` | Says which layer stands in for which brain region. | `neural:<region>` outputs |
| `behavioral_readout_layer` | Fits a classifier on one layer to produce choices. | `behavior` |
| `answer` | Your function that returns an answer for a stimulus, such as a generated reply. | `behavior` |
| `policy` | Your function that maps an environment observation to an action. | `observation` in, `motor` out |
| `intervention` | Your function that applies a perturbation and returns how to undo it. | `lesion`, `stimulation` in |

```python
from brainscore import BrainScoreModel
from brainscore.model_helpers.vision_wrapper import VisionWrapper

model = BrainScoreModel(
    identifier='my-cnn',
    model=net,
    extractor={'vision': VisionWrapper(identifier='my-cnn', model=net, preprocessing=preprocess)},
    region_layer_map={'V4': 'layer3', 'IT': 'layer4'},
)
model.out_channels   # {'neural:V4', 'neural:IT'}
```

A `region_layer_map` is a hypothesis about correspondence. Only a benchmark's comparison with measured data can support it.

## 8. Session, protocol, experiment, benchmark

These four are easy to confuse:

| Term | Scope | Produces |
| --- | --- | --- |
| **Session** | One exchange of inputs and responses with one subject. | Responses. |
| **Protocol** | The procedure: which sessions to run, under which **conditions** (such as baseline and silenced) and how many **repetitions** (independent sessions, each between two resets). A session holds the **trials**: stimuli and responses. | A set of sessions. |
| **Experiment** | A protocol plus a subject plus **tools** (recorders, interventions) plus an output folder. | A saved record. |
| **Benchmark** | A fixed experiment with a metric and measured data. | A score. |

See [Experiments](guides/experiments.md) and [Lesson 10](../notebooks/lessons/10_run_an_experiment.ipynb).

## Next

The [class map](03_class_map.md) shows how these concepts correspond to classes and which method to call.
