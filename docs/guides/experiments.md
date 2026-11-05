# From a session to an experiment to a benchmark

This walkthrough builds one study step by step. We ask a pretrained SqueezeNet whether
an image shows an animal or a vehicle, record its pooled visual features as an
illustrative `IT` region, and check its answers against held-out CIFAR-10 labels. The
`IT` mapping is a teaching hypothesis, not a validated model of IT.

First the study runs once, by hand: a subject, a stimulus set, a task and one session.
Then the same study is packaged twice. An **experiment** adds named conditions,
repetitions, a check of the whole setup before anything runs, and a saved record. A
**benchmark** adds reference data and a score.

Everything is bundled: no downloads, CPU only, under two minutes. In a notebook, run
from the lesson folder; for this guide, start Python from the `unified/` checkout.

| Step | What is introduced | What it does |
| --- | --- | --- |
| 1. Choose a model | the subject | Declares what it takes in, what it can produce, and its traits. |
| 2. Choose the stimuli | `StimulusSet` | One row per trial, with IDs and input columns. |
| 3. Describe the task | `TaskContext` | What to do, the form of the answer, how a readout is fitted. |
| 4. Run it once | `BufferedSession` | Shows the stimuli once and collects what was asked for. |
| 5. Read the results back | assemblies | The answers and the activity behind them. |
| 6. Make it an experiment | `TaskProtocol`, recorders, `Experiment` | Conditions, repetitions, checks and a saved record. |
| 7. Make it a benchmark | `Benchmark`, reference data, metric | The same procedure, returning a `Score`. |

```
benchmark  = protocol + reference data + score
protocol   : for each condition, for each repetition:
                 reset -> one session -> reset
session    : trial, trial, trial, ...        (or one continuous stream)
trial      : stimulus parts -> response       (one StimulusSet row)
```

## Setup

Imports, quiet output, the bundled files and image preprocessing.

```python
# Keep the output short: no progress bars, no library warnings.
import warnings
from functools import partialmethod
from tqdm.auto import tqdm
import tqdm as _tqdm
for _bar in (tqdm, _tqdm.tqdm):
    _bar.__init__ = partialmethod(_bar.__init__, disable=True)
warnings.filterwarnings('ignore')
import os
os.environ['BS_INSTALL_DEPENDENCIES'] = 'no'      # every requirement is already installed

from pathlib import Path
from unittest.mock import patch
import time
import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image

started = time.perf_counter()
torch.set_num_threads(1)                   # one CPU thread, so runs repeat exactly
# The bundled files, found from the lesson folder (notebook) or the unified/ checkout (guide).
ASSETS = next(path for path in (Path('../assets/lessons'), Path('notebooks/assets/lessons'))
              if (path / 'squeezenet1_1.pth').is_file())

# The network expects 224 x 224 images, normalized the way it was trained.
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(224), torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def preprocess(paths):
    """Image files -> one (batch, 3, 224, 224) array."""
    images = []
    for path in paths:
        with Image.open(path) as image:
            images.append(transform(image.convert('RGB')).numpy())
    return np.stack(images)
```

## 1. Choose a model

For a registered model, start with `model = brainscore.load_model('your-registered-id')`
and look at what it declares; its registration already supplies its extractors, readout
and region map.

Here we build a SqueezeNet subject locally so the example runs offline. Pooling gives a
small 512-unit recording, and the readout answers from the same layer.

```python
from brainscore_core import BrainScoreModel
from brainscore.model_helpers.vision_wrapper import VisionWrapper

def make_subject(backbone, identifier):
    # Pool each of the 512 feature maps to one number, so a recording has 512 units.
    network = torch.nn.Sequential(
        backbone.features, torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten()).eval()
    # Build the extractor on the CPU even when a GPU is present, so every machine gets the same numbers.
    with patch('torch.cuda.is_available', return_value=False), \
            patch('torch.backends.mps.is_available', return_value=False):
        extractor = VisionWrapper(model=network, preprocessing=preprocess,
                                  identifier=identifier, kind='frame', batch_size=16)
    # Layer '2' is the Flatten step: it stands for IT and the readout answers from it.
    return BrainScoreModel(identifier=identifier, model=network,
                           extractor={'vision': extractor}, region_layer_map={'IT': '2'},
                           behavioral_readout_layer='2')

backbone = torchvision.models.squeezenet1_1(weights=None)       # weights=None: no download
backbone.load_state_dict(torch.load(ASSETS / 'squeezenet1_1.pth',
                                    map_location='cpu', weights_only=True))
model = make_subject(backbone, 'lesson-squeezenet1_1-pooled')
print('inputs:', sorted(model.in_channels))
print('outputs:', sorted(model.out_channels))
print('traits:', sorted(model.traits))
```

## 2. Choose the stimuli

48 images from four categories of the bundled collection: the first eight of each
category teach the readout, and four other images of each are the test trials. The test
table carries no labels; they become reference data only at step 7.

```python
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet

metadata = pd.read_csv(ASSETS / 'images.csv')
metadata = metadata[metadata['category'].isin(['cat', 'dog', 'airplane', 'ship'])].copy()
metadata['stimulus_id'] = metadata['file'].str.removesuffix('.png')
metadata['image_path'] = metadata['file'].map(lambda name: str(ASSETS / 'images' / name))
is_fitting = metadata.groupby('category').cumcount() < 8      # the first 8 of each category

def stimulus_table(rows, identifier, fitting=False):
    # Only the fitting table carries labels; the test table gets IDs and paths only.
    columns = ['stimulus_id', 'image_path'] + (['kind'] if fitting else [])
    table = StimulusSet(rows[columns].rename(columns={'kind': 'label'}).reset_index(drop=True))
    table.identifier = identifier
    table.stimulus_paths = dict(zip(table['stimulus_id'], table['image_path']))
    return table

fitting_stimuli = stimulus_table(metadata[is_fitting], 'lesson-fit-32', fitting=True)
stimuli = stimulus_table(metadata[~is_fitting], 'lesson-test-16')
assert set(fitting_stimuli['stimulus_id']).isdisjoint(stimuli['stimulus_id'])
print(f'{len(fitting_stimuli)} fitting images; {len(stimuli)} test images')
```

## 3. Describe the task

`Choice` asks for an animal or vehicle answer and for the implied probability of each.
`fitting_stimuli` are the labelled images the readout learns from, and
`prefer_path='readout'` says to answer through that readout: an instruction does not make
an image network understand language.

```python
from brainscore_core import TaskContext
from brainscore_core.responses import Choice

task = TaskContext(response=Choice(['animal', 'vehicle'], probabilities=True),
                   fitting_stimuli=fitting_stimuli, prefer_path='readout')
```

## 4. Run it once

A session is one visit by the subject. This one shows the test images once and asks for
three things: the answers, their probabilities, and the `IT` activity. All three come
from the same presentation of each image. The readout is fitted first, on the fitting
images, in a separate pass.

```python
from brainscore_core.streaming import BufferedSession

# One session: show the test images once; ask for the answers and the IT activity behind them.
session = BufferedSession(stimulus_set=stimuli, task_context=task,
                          requested_output_channels=['behavior:choice', 'behavior:probabilities',
                                                     'neural:IT'])
model.participate(session)
model.reset()                              # forget the fitted readout before anything else runs
```

## 5. Read the results back

Each measurement is an assembly: one row per test image, matched to its stimulus by
`stimulus_id`. Each recording says how it was made (`attrs['recording']`).

```python
choices = session.collect('behavior:choice')
probabilities = session.collect('behavior:probabilities')
it = session.collect('neural:IT')
assert it.sizes['presentation'] == choices.sizes['presentation'] == len(stimuli)
assert it.sizes['neuroid'] == 512
print('neural:', it.shape, 'choices:', choices.shape)
print('recording:', it.attrs['recording'])
# The chosen label is the column holding the 1.
predicted_labels = choices['choice'].values[choices.values.argmax(axis=1)]
print(pd.DataFrame({'stimulus_id': choices['stimulus_id'].values,
                    'answer': predicted_labels}).head())
```

### Measure the answers

The reference data are the held-out labels, stored in a `BehavioralAssembly`. They are
CIFAR-10 annotations, not human or neural recordings, so accuracy against them is an
engineering demonstration, not a brain-alignment result. The metric matches answers to
labels by stimulus ID and counts invalid answers as wrong.

```python
from brainscore_core import Score
from brainscore_core.supported_data_standards.brainio.assemblies import BehavioralAssembly

# The held-out labels, one per test image, kept apart from the stimuli the subject saw.
reference = BehavioralAssembly(
    metadata.loc[~is_fitting, 'kind'].to_numpy(), dims=['presentation'],
    coords={'stimulus_id': ('presentation', metadata.loc[~is_fitting, 'stimulus_id'].to_numpy()),
            'category': ('presentation', metadata.loc[~is_fitting, 'category'].to_numpy())})

def accuracy(prediction, reference):
    # Look up each answered image's true label by its stimulus ID, not by row position.
    by_id = dict(zip(reference['stimulus_id'].values, reference.values))
    target = np.array([by_id[identifier] for identifier in prediction['stimulus_id'].values])
    labels = prediction['choice'].values[prediction.values.argmax(axis=1)]
    valid = (prediction['answer_valid'].values if 'answer_valid' in prediction.coords
             else np.ones(len(labels), dtype=bool))
    return Score(float(np.mean((labels == target) & valid)))
```

```python
hand_score = accuracy(choices, reference)
print('accuracy:', float(hand_score))
```

## 6. Make it an experiment

Running a session by hand is enough to explore. A study needs more, and an `Experiment`
provides it from three pieces:

- **`TaskProtocol`** combines the stimuli and the task under named conditions and
  repetitions. It builds each session itself and resets the subject before and after.
- **Recorders** say what to request and save. `RecordBehavior()` saves every answer the
  task asks for; `RecordNeural(['IT'])` saves the activity of a region the subject declares.
- **`Experiment`** checks the subject, protocol and recorders against each other before
  running (`validate()`), runs the protocol, and saves every recording. The result reads
  them back. A run is kept in a temporary folder until the result is no longer used:
  call `result.save('runs/my-study')`, or pass `output_dir='runs/my-study'`, to keep it
  for inspection or replay. A folder you name must be new, so a run never overwrites an
  earlier one.

The answers it saves are the same as the ones collected by hand.

```python
from brainscore.experiments import Experiment, RecordBehavior, RecordNeural, TaskProtocol

# The same stimuli and task, under one named condition and one repetition.
protocol = TaskProtocol(stimuli, task, identifier='animal-or-vehicle',
                        conditions=['baseline'], repetitions=[0])

def recorders():
    """What to request and save: the IT activity, and every answer the task asks for."""
    return [RecordNeural(['IT']), RecordBehavior()]

experiment = Experiment(subject=model, protocol=protocol, tools=recorders())
plan = experiment.validate()               # checks the whole setup without running the model
print(plan['protocol']['config']['compatibility'])
result = experiment.run()
assert result.manifest['status'] == 'complete'

# Read back from the saved files: the same answers and activity as the session run by hand.
saved_choices = result.behavior(condition='baseline', repetition=0, measure='choice')
saved_it = result.neural('IT', condition='baseline', repetition=0)
assert np.array_equal(saved_choices.values, choices.values)
assert np.allclose(saved_it.values, it.values)
print('saved:', sorted(path.name for path in result.directory.iterdir()))   # what the run wrote
# This run is temporary; result.save('runs/animal-or-vehicle-001') would keep a copy.
```

Four deliberate mistakes, each caught before the model runs. Each prints its reason.

```python
text_only = StimulusSet({'stimulus_id': ['word'], 'sentence': ['cat']})
text_only.identifier = 'wrong-input-example'
mistakes = [                                      # (name, protocol, tools), each wrong in one way
    ('missing-region', TaskProtocol(stimuli, task), [RecordNeural(['not-a-region'])]),
    ('passive-behavior', TaskProtocol(stimuli, TaskContext(task_type='passive')),
     [RecordBehavior()]),
    ('missing-memory', TaskProtocol(stimuli, task, required_traits={'memory:session'}),
     [RecordBehavior()]),
    ('wrong-input', TaskProtocol(text_only, task), [RecordBehavior()]),
]
for name, wrong_protocol, wrong_tools in mistakes:
    try:
        Experiment(subject=model, protocol=wrong_protocol, tools=wrong_tools).validate()
    except ValueError as error:
        print(f'{name}: {error}')
    else:
        raise AssertionError(f'{name} should have failed validation')
```

## 7. Make it a benchmark

A **benchmark is a protocol plus reference data and a score**. A small `Benchmark`
subclass runs the same experiment in `__call__` and applies the metric. Core has no
constructor that turns a `TaskProtocol` into a benchmark; this is the supported path.

A random-weight network is scored first, as the control.

```python
from brainscore_core import Benchmark

class AnimalVehicleBenchmark(Benchmark):
    identifier = 'lesson-animal-vehicle-accuracy'
    version = 1
    uses_session = True

    def __init__(self, protocol, reference, metric):
        self.protocol, self.reference, self.metric = protocol, reference, metric
        # Resolve the same recorders before copying benchmark declarations.
        protocol.accept_requests(set().union(*(tool.output_channels for tool in recorders())))
        self.required_input_channels = protocol.input_channels
        self.requested_output_channels = protocol.output_channels
        self.required_traits = protocol.required_traits

    def __call__(self, subject):
        # Each call runs the same experiment, then applies the metric to its answers.
        run = Experiment(subject=subject, protocol=self.protocol, tools=recorders()).run()
        prediction = run.behavior(condition='baseline', repetition=0, measure='choice')
        score = self.metric(prediction, self.reference)
        score.attrs.update(reference='CIFAR-10 labels', metric='accuracy', normalization='raw')
        return score

benchmark = AnimalVehicleBenchmark(protocol, reference, accuracy)
assert benchmark.protocol is protocol
```

```python
import brainscore

torch.manual_seed(7)                       # fixes the random weights of the control
control = make_subject(torchvision.models.squeezenet1_1(weights=None), 'lesson-random-squeezenet')
control_score = brainscore.score(control, benchmark, check_mem=False)
model_score = brainscore.score(model, benchmark, check_mem=False)
# Repackaging the procedure must not change its score.
assert float(model_score) == float(hand_score)
assert float(model_score) == float(accuracy(choices, reference.isel(presentation=slice(None, None, -1))))
print({'random-weight accuracy': float(control_score), 'pretrained accuracy': float(model_score)})
print(model_score.attrs['compatibility'])
print(f'Elapsed: {time.perf_counter() - started:.1f} seconds')
```

`brainscore.score` checks compatibility before running the benchmark and records the match
on the score. A `CompatibilityError` means the pair is not applicable, not a score of zero.
`check_mem=False` skips the memory probe, which this small example does not need.

## When to use manual sessions

`TaskProtocol` is the standard path for a fixed stimulus table. Use `SessionProtocol`
with a session factory only for custom event ordering, continuous streams or environments
that need their own session construction. The factory receives
`Repetition(condition, identifier)`, and the protocol still owns resets.
`CallableProtocol` instead wraps an existing evaluator that owns its loop and resets.

For interventions, attach `Intervene(change, conditions=[...])` after the recorders.
It applies the subject's undoable state change per matching repetition and cleans it up
at its end. A readout is fitted on the intact network by default;
`TaskProtocol(..., refit_readout=True)` asks a different question by fitting after the change.

For models that generate answers, `RecordNeural(..., during='prompt')` keeps the prompt
forward pass. The extractor must support recording during answers. Declared channels
and traits are pre-flight checks; they do not guarantee every task setting, data file
or extractor implementation is valid. Such errors may be raised when a session starts.

`RecordInputsOutputs` can additionally save input events and lifecycle events for
`result.record` and replay. Neural and behavioral recorders save their own folders;
they do not by themselves create a complete input replay log.

See [Lesson 10](../../notebooks/lessons/10_run_an_experiment.ipynb) for the executable
version and [Write a benchmark](write_a_benchmark.md) for benchmark authoring details.
The bundled [asset notes](../../notebooks/assets/lessons/README.md) record data and weight sources.

# Reference

## Lower-level tools

When the recorders do not cover what you need, build sessions yourself and attach tools that reach into the network:

```python
from contextlib import contextmanager
from brainscore_core.streaming import BufferedSession
from brainscore.experiments import (Experiment, SessionProtocol, RecordInputsOutputs,
                                    RecordActivity, Ablate, TorchInstrumentation)

@contextmanager
def make_session(repetition):                # repetition.condition, repetition.identifier
    yield BufferedSession(stimulus_set=stimuli, requested_output_channels=['behavior'])

protocol = SessionProtocol(identifier='silence-layer3', session_factory=make_session,
                           conditions=['baseline', 'silenced'], repetitions=[0, 1, 2],
                           input_channels=['vision'], output_channels=['behavior'])

experiment = Experiment(
    subject=subject,
    protocol=protocol,
    tools=[RecordInputsOutputs(),                        # save inputs, outputs, events
           Ablate(['layer3'], conditions=['silenced']),   # zero layer3 in one condition
           RecordActivity(['layer3'])],                   # every forward pass, unaggregated
    instrumentation=TorchInstrumentation(net),
    output_dir='runs/silence-layer3',
    metadata={'checkpoint': 'v1.2', 'seed': 7},
)
experiment.validate()          # checks channels and tools without running the model
result = experiment.run()
```

List tools in this order: recorders, activity recorded *before* an intervention, interventions, activity recorded *after*.

## Tools

| Tool | Does |
| --- | --- |
| `RecordNeural(regions)` | Requests `neural:<region>` from the subject and saves it. Read with `result.neural(region, condition=...)`. |
| `RecordBehavior()` | Requests every kind of answer the task's response asks for and saves them. Read with `result.behavior(condition=..., measure=...)`. |
| `Intervene(change, conditions=...)` | Applies a `StateChange` through the subject's `intervention`, undone after each repetition. |
| `RecordInputsOutputs` | Saves every input, output and lifecycle event with its condition and repetition. Needs no access to model internals. |
| `RecordActivity(targets)` | Records layer outputs through instrumentation. |
| `Ablate(targets, conditions=...)` | Zeros layer outputs or a `Selection` of units, in chosen conditions. |
| `ScaleActivity(targets, factor=0.5)` | Multiplies outputs by a factor. |
| `PatchActivity` | Replaces activity with activity recorded on another input. |
| `ObserveCalls` | Attaches an existing observer to method calls. |

Targets are layer paths or `Selection` objects, the same ones used in [Perturb and select units](perturb_and_select_units.md).

## Read and replay

```python
for event in result.record.events():        # or RunRecord('runs/silence-layer3')
    print(event['condition'], event['repetition_id'], event['kind'])
outputs = list(result.record.outputs())

from brainscore.experiments import replay_sessions, compare_outputs
replay = Experiment(subject=subject, protocol=replay_sessions(result.directory),
                    tools=[RecordInputsOutputs()], output_dir='runs/replay').run()
compare_outputs(result.directory, replay.directory)
```

Replay sends the saved inputs again. It does not re-run a simulator; in closed-loop tasks the new actions do not change the saved observations. Reattach intervention tools explicitly when replaying.

The output folder holds `experiment.json` (setup, status, provenance), `result.json`, and `inputs_outputs/` (events and arrays). Without `output_dir`, that folder is temporary and is deleted with the result; replay and inspection need a kept run, so pass `output_dir` or call `result.save(path)` first. A failed run stays readable for diagnosis in a folder you named; `RunRecord.inspect(path)` reports what survived an interrupted run.

## Wrap an existing evaluator

```python
from brainscore.experiments import CallableProtocol

protocol = CallableProtocol(identifier='classic-benchmark',
                            evaluate=lambda subject, context: benchmark(subject),
                            methods=['look_at'])     # calls to observe
```

The evaluator keeps its own loop, resets and scoring; the tools still attach.

