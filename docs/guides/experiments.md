# From a model to an experiment to a benchmark

This walkthrough develops one procedure through all six steps. We ask a pretrained
SqueezeNet whether an image contains an animal or a vehicle, record its pooled visual
features as an illustrative `IT` mapping, and then score its choices against held-out
CIFAR-10 labels. The mapping is a teaching hypothesis, not a validated model of IT.

Run the cells in order. Everything is bundled with the lessons: no downloads, CPU
only, and a target runtime below two minutes. In a notebook, use the lesson directory;
for the guide, start Python from the `unified/` checkout. Neither needs an earlier lesson.

| Step | Object introduced | Responsibility |
| --- | --- | --- |
| 1. Choose a model | `model` | Declares accepted inputs, available outputs and traits. |
| 2. Design the task | `StimulusSet` | Holds trial content: one row per trial, with IDs and input columns. |
| | `TaskContext` | Says what to do, what form an answer takes, and how a readout is fitted. |
| | `TaskProtocol` | Combines stimuli and task; runs conditions x repetitions and owns resets. |
| 3. Attach recorders | `RecordNeural`, `RecordBehavior` | Request and save measurements. |
| 4. Run | `Experiment` | Validates the setup, runs the protocol and saves the run. |
| 5. Inspect | `ExperimentResult` | Returns saved assemblies by condition, repetition and measure. |
| 6. Make a benchmark | `Benchmark`, reference data, metric | Reuses the procedure and returns a `Score`. |

A **repetition** is one session between two resets. A session holds trials (the rows of
our `StimulusSet`), or one continuous stream in other protocols. Within a session,
`memory:session` means earlier trials can be remembered; across repetitions the subject
is reset. A condition name is a label: naming one `silenced` does not itself apply a lesion.

```
benchmark  = protocol + reference data + score
protocol   : for each condition, for each repetition:
                 reset -> one session -> reset
session    : trial, trial, trial, ...        (or one continuous stream)
trial      : stimulus parts -> response       (one StimulusSet row)
```

## 1. Choose a model

For a registered model, start with `model = brainscore.load_model('your-registered-id')`
and inspect its declarations; its registration already supplies extractors, readouts
and region mappings. Do not redeclare its channels in the experiment.

Here we assemble a local SqueezeNet subject once so the example can run offline.
The bundled checkpoint is loaded explicitly with `weights=None` to prevent a download.
Pooling gives a small 512-unit recording. The temporary device overrides only force the
wrapper to choose CPU; they are restored immediately after construction.

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
from tempfile import TemporaryDirectory
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
workspace = TemporaryDirectory(prefix='umi-beginner-')
run_root = Path(workspace.name)            # every run of this walkthrough is saved under here

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

Then build the subject.

```python
from brainscore_core import BrainScoreModel
from brainscore.model_helpers.vision_wrapper import VisionWrapper

def make_subject(backbone, identifier):
    # Pool each of the 512 feature maps to one number, so a recording has 512 units.
    network = torch.nn.Sequential(
        backbone.features, torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten()).eval()
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

## 2. Design the task

First introduce the **StimulusSet**: 48 images from four categories in the bundled
120-image collection. Eight images per category fit the readout; four different images
per category are test trials. IDs, paths and split membership stay explicit.

`fitting_stimuli` teaches the readout. The table passed to `TaskProtocol` is what the
subject answers during evaluation. Test labels are withheld from that table and become
reference data only at step 6.

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
print(f'{len(fitting_stimuli)} fitting images; {len(stimuli)} test trials per session')
```

Now introduce the **TaskContext**. `Choice` specifies an animal/vehicle answer and asks
for implied probabilities as well. `prefer_path='readout'` fixes how the answer is
produced: a classifier fitted to this subject's pooled activity. An instruction does
not make a CNN understand language; this task uses its labeled fitting examples.

Finally introduce the **TaskProtocol**. It combines the test stimuli with that task,
under one `baseline` condition and one repetition. It creates a fresh `BufferedSession`
automatically for each condition/repetition, and resets the subject before and after.
You do not construct sessions or call `reset()` yourself on this path.

```python
from brainscore_core import TaskContext
from brainscore_core.responses import Choice
from brainscore.experiments import TaskProtocol

task = TaskContext(response=Choice(['animal', 'vehicle'], probabilities=True),
                   fitting_stimuli=fitting_stimuli, prefer_path='readout')
protocol = TaskProtocol(stimuli, task, identifier='animal-or-vehicle',
                        conditions=['baseline'], repetitions=[0])
```

## 3. Attach neural and behavioral recorders

The task decides what the subject does. The recorders decide what is requested and
saved. `RecordBehavior()` resolves to the answer channels in `task.response`, here
`behavior:choice` and `behavior:probabilities`. `RecordNeural(['IT'])` requests the
region the subject already declares. Both measurements use the same test presentation;
readout fitting is a separate pass over the fitting images.

| Task | Recorders | Measurement |
| --- | --- | --- |
| `TaskContext(task_type='passive')` | Neural | Activity during plain viewing. |
| A task with answers | Neural | Activity during answering; answers are not saved. |
| A task with answers | Behavior | Answers. |
| A task with answers | Both | Activity and answers from the same presentation. |

Passive viewing has no answer, so attaching `RecordBehavior` to it is an error.

```python
from brainscore.experiments import RecordNeural, RecordBehavior

def recorders():
    return [RecordNeural(['IT']), RecordBehavior()]

equipment = recorders()
```

## 4. Validate and run

`validate()` checks declarations and tools without running or resetting the model.
Its plan records consumed and dropped input channels as a full or partial match.
At least one input must match, every requested output must be declared, and all
hard-required inputs and required traits must be present. Unsupported extra input
streams are ignored by the subject. A fully unsupported stimulus kind is rejected.

`run()` validates again before execution. The output directory must be new; using a
fresh child of a temporary directory makes this walkthrough rerunnable. Use a new
persistent path such as `runs/animal-or-vehicle-001` to retain your own study.

```python
from brainscore.experiments import Experiment

experiment = Experiment(subject=model, protocol=protocol, tools=equipment,
                        output_dir=run_root / 'exploration')
plan = experiment.validate()
print(plan['protocol']['config']['compatibility'])
result = experiment.run()
assert result.manifest['status'] == 'complete'
```

These are four deliberate mistakes, checked before inference. Each prints the reason
and leaves no run directory. A missing trait is a declaration failure; setting
`required_traits` does not add memory to the model.

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
    destination = run_root / name
    try:
        Experiment(subject=model, protocol=wrong_protocol, tools=wrong_tools,
                   output_dir=destination).validate()
    except ValueError as error:
        print(f'{name}: {error}')
    else:
        raise AssertionError(f'{name} should have failed validation')
    assert not destination.exists()
```

## 5. Inspect the saved results

These accessors read saved assemblies; they do not run the model again. `condition`
selects a named condition. `repetition` selects one session when there are several;
it is not a trial index. To select trials, use the assembly's `stimulus_id` coordinate.
`measure` selects a kind of behavioral output that the task requested.

```python
it = result.neural('IT', condition='baseline', repetition=0)
choices = result.behavior(condition='baseline', repetition=0, measure='choice')
probabilities = result.behavior(condition='baseline', repetition=0, measure='probabilities')
assert it.sizes['presentation'] == choices.sizes['presentation'] == len(stimuli)
assert it.sizes['neuroid'] == 512
assert list(choices['stimulus_id'].values) == list(stimuli['stimulus_id'])
print('neural:', it.shape, 'choices:', choices.shape)
print('recording:', it.attrs['recording'])
# The chosen label is the column holding the 1.
predicted_labels = choices['choice'].values[choices.values.argmax(axis=1)]
print(pd.DataFrame({'stimulus_id': choices['stimulus_id'].values,
                    'answer': predicted_labels}).head())
```

## 6. Reuse the protocol as a benchmark

A **benchmark = protocol + reference data + score**. Core currently provides `Benchmark`
and `BenchmarkBase`; it has no constructor that turns a `TaskProtocol` into a benchmark.
The supported path is a small `Benchmark` subclass whose `__call__` runs the same
`Experiment` and applies a metric. No Subject or Session changes are needed.

Introduce reference data now: the held-out dataset labels, stored in a
`BehavioralAssembly`. These are CIFAR-10 category annotations, not human behavioral or
neural recordings. Accuracy against them is an engineering demonstration, not a
brain-alignment result. A scientific benchmark would supply its measured reference
data, metric, splits and, where appropriate, a ceiling.

The metric aligns by stimulus ID and uses the assembly's choice labels, so it does not
assume that label columns happen to be in alphabetical order. Invalid answers count as
incorrect. The controlled example requests one baseline session and scores its choices;
neural activity and probabilities remain available in the saved run.

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

Applied to the saved run, the metric gives the pretrained subject's accuracy.

```python
hand_score = accuracy(choices, reference)
print('accuracy of the saved run:', float(hand_score))
```

The benchmark runs the same experiment and applies the same metric.

```python
from brainscore_core import Benchmark

class AnimalVehicleBenchmark(Benchmark):
    identifier = 'lesson-animal-vehicle-accuracy'
    version = 1
    uses_session = True

    def __init__(self, protocol, reference, metric, output_root):
        self.protocol, self.reference, self.metric = protocol, reference, metric
        self.output_root = output_root
        # Resolve the same recorders before copying benchmark declarations.
        protocol.accept_requests(set().union(*(tool.output_channels for tool in recorders())))
        self.required_input_channels = protocol.input_channels
        self.requested_output_channels = protocol.output_channels
        self.required_traits = protocol.required_traits

    def __call__(self, subject):
        from uuid import uuid4
        # Each call runs the same experiment into a new folder, then applies the metric.
        run = Experiment(subject=subject, protocol=self.protocol, tools=recorders(),
                         output_dir=self.output_root / uuid4().hex).run()
        prediction = run.behavior(condition='baseline', repetition=0, measure='choice')
        score = self.metric(prediction, self.reference)
        score.attrs.update(reference='CIFAR-10 labels', metric='accuracy', normalization='raw',
                           experiment_directory=str(run.directory))
        return score

benchmark = AnimalVehicleBenchmark(protocol, reference, accuracy, run_root / 'benchmarks')
assert benchmark.protocol is protocol
```

Score a random-weight control before the pretrained subject. Both use the same protocol,
splits, fitting rule, recorders and metric. The control is local and needs no download.
`brainscore.score` checks compatibility before the benchmark and stamps the match on the
score. A `CompatibilityError` means N/A; it is not a score of zero. Here `check_mem=False`
skips the memory probe for this small CPU example; keep the default for larger studies.

Registration is optional when passing objects. Publishing a benchmark additionally
needs a stable identifier/version and reviewed scientific choices. The explicit mapping
from protocol fields to benchmark declarations above is current authoring boilerplate.

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

The sample is small and the region mapping illustrative. The result demonstrates the
workflow; it does not establish a scientific advantage or a validated neural score.
The protocol, task, fitting split and measurements were reused without rebuilding a
session or changing the subject contract.

The temporary files remain inspectable until the final cleanup cell. Omit cleanup and
choose a persistent `output_dir` in a real study.

```python
workspace.cleanup()
```

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

The output folder holds `experiment.json` (setup, status, provenance), `result.json`, and `inputs_outputs/` (events and arrays). A failed run stays readable for diagnosis; `RunRecord.inspect(path)` reports what survived an interrupted run.

## Wrap an existing evaluator

```python
from brainscore.experiments import CallableProtocol

protocol = CallableProtocol(identifier='classic-benchmark',
                            evaluate=lambda subject, context: benchmark(subject),
                            methods=['look_at'])     # calls to observe
```

The evaluator keeps its own loop, resets and scoring; the tools still attach.

