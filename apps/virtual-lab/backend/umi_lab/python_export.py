"""Readable Python equivalents of saved lab configurations; no model imports."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from pprint import pformat
from textwrap import dedent, indent
from zipfile import ZipFile, ZIP_DEFLATED

MAX_PLAN_BYTES = 4 * 1024 * 1024

ARCHIVE_README = '''UMI experiment export

experiment.py is readable, editable Python that recreates this experiment.
With Python 3.11 and a matching UMI v2 installation:
    python experiment.py --output ./my-rerun
    python experiment.py --validate-only
Validation loads the model but performs no inference. Weights are not bundled.
Registered datasets may require their original data source. App-local profiles
also require their factory and umi_lab; see the script's --help.

inputs/ contains your uploaded media. Keep it beside experiment.py, or use
--inputs /path/to/inputs. Each execution creates a new run with its own outputs.

plan.json stores the exact machine-readable configuration, useful for automated
replay or comparing settings. With the lab source available, replay it using:
    python lab.py --run /path/to/plan.json --output ./direct-runs

summary.json describes the completed original run. experiment/ contains its
recorded outputs, arrays and manifests. Running experiment.py does not overwrite
those original measurements. Python preserves the saved model identifier and
settings; it does not silently replace an old model with a newer variant.
'''


def literal(value: object) -> str:
    return pformat(value, width=88, sort_dicts=False)


def block_list(items: list[str]) -> str:
    """A list literal with one item per line."""
    if not items:
        return '[]'
    return '[\n' + ''.join(indent(item, '    ') + ',\n' for item in items) + ']'


def rows_literal(rows: list) -> str:
    """One stimulus per block, so each trial is easy to read and edit."""
    return block_list([literal(row) for row in rows])


def call(name: str, args: list[str]) -> str:
    """name(a, b) on one line when short, else one argument per line."""
    one_line = f"{name}({', '.join(args)})"
    if len(one_line) <= 80 and '\n' not in one_line:
        return one_line
    return f"{name}(\n" + ''.join(indent(arg, '    ') + ',\n' for arg in args) + ')'


def repetitions_literal(count: int) -> str:
    return literal(tuple(range(count))) if count <= 10 else f'tuple(range({count}))'


STIMULI_HELPER = '''
def make_stimuli(rows, inputs):
    """Turn rows into a StimulusSet, with media paths resolved inside `inputs`."""
    import hashlib
    from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet

    if not rows:
        raise ValueError("Add at least one stimulus")
    copied, paths = [], {}
    for index, original in enumerate(rows):
        row = dict(original)
        row.setdefault("stimulus_id", f"stimulus-{index}")
        for column in ("image_path", "audio_path", "video_path"):
            if column not in row:
                continue
            path = (inputs / str(row[column])).resolve()
            if not path.is_file():
                raise FileNotFoundError(f"Missing {path}. Extract All artifacts, or supply --inputs.")
            row[column] = str(path)
            if column == "image_path":
                paths[row["stimulus_id"]] = path
        copied.append(row)
    table = StimulusSet(copied)
    if table["stimulus_id"].duplicated().any():
        raise ValueError("Stimulus IDs must be unique")
    table.identifier = "lab-" + hashlib.sha256(json.dumps(copied, sort_keys=True).encode()).hexdigest()[:24]
    table.stimulus_paths = paths
    return table
'''


FITTING_CHECK = '''
def check_fitting(fitting, labels=None):
    """A fitted readout needs labeled examples; a choice needs two of its labels."""
    if fitting is None or "label" not in fitting.columns:
        raise ValueError("A fitted readout needs a separate fitting table with a label column")
    if any(value is None or not str(value).strip() for value in fitting["label"]):
        raise ValueError("Fill the label for every fitting example")
    fitted_labels = set(map(str, fitting["label"]))
    if labels is not None and (len(fitted_labels) < 2 or fitted_labels - set(labels)):
        raise ValueError("Choice fitting examples need at least two classes drawn from the response labels")
'''


class Parts:
    """Pieces of one exported script: editable constants, imports and the build body."""

    def __init__(self) -> None:
        self.constants = ''
        self.imports: dict[str, list[str]] = {}
        self.body = ''
        self.uses_stimuli = False
        self.uses_fitting_check = False
        self.read_back: list[str] = []

    def need(self, module: str, *names: str) -> None:
        listed = self.imports.setdefault(module, [])
        listed.extend(name for name in names if name not in listed)

    def import_lines(self) -> str:
        return ''.join(f"from {module} import {', '.join(names)}\n" for module, names in self.imports.items())


def add_tables(parts: Parts, request) -> None:
    """Stimuli become editable module constants; the build body only loads them."""
    fitting = [{**row, 'stimulus_id': row.get('stimulus_id', f'fitting-{index}')}
               for index, row in enumerate(request.fitting_rows)] if request.mode == 'protocol' else request.fitting_rows
    if request.stimulus_set:
        parts.need('brainscore', 'load_stimulus_set')
        parts.body += f"stimuli = load_stimulus_set({literal(request.stimulus_set)})\n"
    else:
        parts.uses_stimuli = True
        parts.constants += ('\n# 2. Stimuli: one row per trial. Media paths are relative to the inputs/ folder.\n'
                            f'STIMULI = {rows_literal(request.rows)}\n')
        parts.body += 'stimuli = make_stimuli(STIMULI, inputs)\n'
    if fitting:
        parts.uses_stimuli = True
        parts.constants += ('\n# Labeled examples that fit a readout. They are not test trials.\n'
                            f'FITTING = {rows_literal(fitting)}\n')
        parts.body += 'fitting = make_stimuli(FITTING, inputs)\n'
    else:
        parts.body += 'fitting = None\n'


def change_expression(change: dict, *, protocol: bool) -> str:
    operation = change.get('operation', 'zero')
    value = change.get('value', 0)
    return call('StateChange', [
        f"kind={literal('ablation' if operation == 'zero' else 'stimulation')}",
        call('target=Selection', [f"layer={literal(change['target'])}", f"indices={literal(change.get('indices'))}",
                                  f"axis={literal(change.get('axis', -1) if protocol else None)}"]),
        call('perturbation=Perturbation', [f"kind={literal(operation)}", f"scale={literal(value)}",
                                           f"amount={literal(value)}"]),
    ])


TASK_COMMENTS = {
    'passive': 'The subject views each stimulus and gives no answer.',
    'choice': 'The subject chooses one of the response labels.',
    'rating': 'The subject gives a rating on the scale.',
    'text': 'The subject writes an answer.',
    'legacy': 'The subject gives label probabilities (legacy behavior).',
}


def protocol_parts(request) -> Parts:
    from .protocol import ProtocolSpec

    spec = ProtocolSpec.parse(request.protocol)
    task = spec.task
    kind = task['kind']
    parts = Parts()
    parts.need('brainscore.experiments', 'Experiment', 'TaskProtocol')
    parts.need('brainscore_core', 'TaskContext')
    add_tables(parts, request)
    readout = kind != 'passive' and task['path'] == 'readout'
    if not request.fitting_rows and not readout:
        parts.body = parts.body.replace('fitting = None\n', '')  # nothing below reads it
    if readout:
        parts.uses_fitting_check = True
        labels = literal(task['labels']) if kind in ('choice', 'legacy') else 'None'
        parts.body += f'check_fitting(fitting, labels={labels})\n'

    response = None
    if kind == 'choice':
        parts.need('brainscore_core.responses', 'Choice')
        args = [literal(task['labels'])]
        if task['probabilities']:
            args.append('probabilities=True')
        if task.get('confidence'):
            parts.need('brainscore_core.responses', 'Scale')
            args.append(f"confidence=Scale(**{literal(task['confidence'])})")
        response = call('Choice', args)
    elif kind == 'rating':
        parts.need('brainscore_core.responses', 'Scale')
        response = f"Scale(**{literal(task['scale'])})"
    elif kind == 'text':
        parts.need('brainscore_core.responses', 'Text')
        response = 'Text()'
    # The GUI passes the same values; arguments equal to TaskContext defaults are left out.
    settings = [
        ('task_type', 'passive' if kind == 'passive' else 'probabilities' if kind == 'legacy' else None, None),
        ('instruction', task['instruction'] or None, None),
        ('label_set', (task['labels'] or None) if kind != 'passive' else None, None),
        ('prefer_path', task['path'], None),
        ('readout_fit', task['readout_fit'], 'current'),
        ('trial_memory', task['trial_memory'], 'none'),
        ('present', tuple(task['present']) or None, None),
    ]
    args = [f'{name}={literal(value)}' for name, value, default in settings if value != default]
    if response is not None:
        args.append(f'response={response}')
    if request.fitting_rows:
        args.append('fitting_stimuli=fitting')
    parts.body += f'\n# 3. Task. {TASK_COMMENTS[kind]}\ntask = {call("TaskContext", args)}\n'
    if kind != 'passive':
        parts.need('brainscore_core.responses', 'trial_prompt')
        parts.body += ('for _, row in stimuli.iterrows():\n'
                       '    trial_prompt(task, row)  # Check the order of trial parts before the model runs.\n')

    tools = []
    regions = spec.measurements['regions']
    if regions:
        parts.need('brainscore.experiments', 'RecordNeural')
        tools.append(f'RecordNeural({literal(regions)})')
    if spec.measurements['behavior']:
        parts.need('brainscore.experiments', 'RecordBehavior')
        tools.append('RecordBehavior()')
    if spec.measurements['replay']:
        parts.need('brainscore.experiments', 'RecordInputsOutputs')
        tools.append('RecordInputsOutputs()')
    for index, condition in enumerate(spec.conditions):
        if condition['intervention']:
            parts.need('brainscore.experiments', 'Intervene')
            parts.need('brainscore_core', 'StateChange', 'Selection', 'Perturbation')
            tools.append(call('Intervene', [change_expression(condition['intervention'], protocol=True),
                                            f"conditions={literal([condition['name']])}",
                                            f"name={literal(f'condition_{index}')}"]))
    parts.body += f'\n# 4. Recordings: what to save from each presentation.\ntools = {block_list(tools)}\n'

    conditions = [condition['name'] for condition in spec.conditions]
    protocol_args = ['stimuli', 'task', f'identifier={literal(spec.identifier)}', f'conditions={literal(conditions)}']
    if spec.repetitions != 1:
        protocol_args.append(f'repetitions={repetitions_literal(spec.repetitions)}')
    if spec.required_traits:
        protocol_args.append(f'required_traits={literal(spec.required_traits)}')
    parts.body += ('\n# 5. Conditions and repetitions. Each repetition starts from a reset subject,\n'
                   '#    and interventions are undone automatically.\n'
                   f'protocol = {call("TaskProtocol", protocol_args)}\n'
                   'return Experiment(subject=subject, protocol=protocol, tools=tools, output_dir=output)\n')
    # repr() never contains a line break, so these comment lines cannot become code.
    if regions:
        parts.read_back.append(f'result.neural({regions[0]!r}, condition={conditions[0]!r}, repetition=0)')
    if spec.measurements['behavior']:
        parts.read_back.append(f'result.behavior(condition={conditions[0]!r}, repetition=0)')
    return parts


def session_parts(request) -> Parts:
    parts = Parts()
    parts.need('contextlib', 'contextmanager')
    parts.need('brainscore.experiments', 'Experiment', 'SessionProtocol')
    parts.need('brainscore_core.streaming', 'Session', 'StreamEvent')
    if request.mode == 'session':
        if not request.events:
            raise ValueError('Add at least one input event')
        for event in request.events:
            if (set(event) - {'channel', 'payload', 't_ms', 'meta'}
                    or not isinstance(event.get('channel'), str) or 'payload' not in event):
                raise ValueError('Each event needs channel and payload; optional t_ms and meta')
        parts.constants += ('\n# 2. Input events, in the order the subject receives them.\n'
                            f'EVENTS = {rows_literal(request.events)}\n')
        parts.body += ("events = [StreamEvent(row['channel'], row['payload'], row.get('t_ms'), row.get('meta', {})) for row in EVENTS]\n"
                       "inputs_declared = sorted({event.channel for event in events})\n")
        session = f"Session(events, requested_output_channels=({literal(request.output)},))"
    else:
        add_tables(parts, request)
        parts.need('brainscore_core.streaming_helpers', 'stimulus_session')
        parts.body += ("probe = stimulus_session(stimuli)\n"
                       "inputs_declared = sorted({event.channel for event in iter(probe.next_input, None)\n"
                       "                          if event.channel != 'control'})\n")
        if request.output == 'behavior':
            parts.need('brainscore_core', 'TaskContext')
            parts.body += (f"task = TaskContext(task_type='probabilities', instruction={literal(request.instruction or None)},\n"
                           f"    label_set={literal(request.labels or None)}, fitting_stimuli=fitting,\n"
                           f"    prefer_path={literal(request.prefer_path)}, metadata={{'stimulus_set': stimuli}})\n")
            session = "stimulus_session(stimuli, ('behavior',), task)"
        elif request.output.startswith('neural:'):
            session = f"stimulus_session(stimuli, ({literal(request.output)},))"
        elif request.output == 'response_trace':
            parts.body += f'''events = [StreamEvent("generation_request", {{
    "system": {literal(request.instruction or None)},
    "messages": [{{"role": "user", "content": str(row["sentence"])}}],
    "temperature": 0, "max_tokens": 64,
}}, None, {{"stimulus_id": str(row["stimulus_id"]), "sentence": str(row["sentence"])}})
    for _, row in stimuli.iterrows()]
inputs_declared = ["generation_request"]
'''
            session = "Session(events, requested_output_channels=('response_trace',))"
        else:
            raise ValueError('Use a declared neural output, behavior or response_trace for stimuli')
    parts.body += f'\n@contextmanager\ndef session_factory(repetition):\n    yield {session}\n\n'
    tools = []
    if request.record:
        parts.need('brainscore.experiments', 'RecordInputsOutputs')
        tools.append('RecordInputsOutputs()')
    if request.intervention:
        parts.need('brainscore.experiments', 'Intervene')
        parts.need('brainscore_core', 'StateChange', 'Selection', 'Perturbation')
        tools.append(call('Intervene', [change_expression(request.intervention, protocol=False),
                                        "conditions=['intervention']", "name='subject_intervention'"]))
    conditions = ['baseline', 'intervention', 'restored'] if request.intervention else ['baseline']
    parts.body += f'''tools = {block_list(tools)}
protocol = SessionProtocol(
    "umi-workbench-session-v1", session_factory,
    input_channels=inputs_declared, output_channels=({literal(request.output)},),
    conditions={literal(conditions)}, repetitions={repetitions_literal(request.repeats)},
)
return Experiment(subject=subject, protocol=protocol, tools=tools, output_dir=output)
'''
    return parts


def benchmark_parts(request) -> Parts:
    parts = Parts()
    parts.need('brainscore', 'load_benchmark', 'score')
    parts.need('brainscore.experiments', 'Experiment')
    parts.constants += ('\n# 2. Benchmark: a registered benchmark with its own stimuli and scoring.\n'
                        f'BENCHMARK = {literal(request.benchmark)}\n')
    parts.body = '''benchmark = load_benchmark(BENCHMARK)

class BenchmarkProtocol:
    identifier = BENCHMARK
    def describe(self):
        return {"benchmark": self.identifier, "reset_policy": "before and after the whole benchmark"}
    def validate(self, candidate):
        pass  # brainscore.score owns compatibility and memory preflight.
    def run(self, candidate, context):
        try:
            candidate.reset()
            with context.repetition("benchmark"):
                return score(candidate, benchmark)
        finally:
            candidate.reset()

return Experiment(subject=subject, protocol=BenchmarkProtocol(), output_dir=output)
'''
    return parts


def calibration_parts(saved: dict) -> Parts:
    from .experiment import Plan

    plan = Plan.parse(saved)
    parts = Parts()
    parts.need('umi_lab.experiment', 'Plan', 'build_experiment as calibration_experiment')
    parts.constants += f'\n# 2. Calibration plan.\nPLAN = {literal(asdict(plan))}\n'
    parts.body = 'return calibration_experiment(profile, Plan.parse(PLAN), output)\n'
    return parts


DOCSTRING = '''\
"""Brain-Score experiment, exported from the UMI lab.

Run it (needs Python 3.11 and the matching UMI v2 installation):

    python experiment.py                    run, and save results in a new folder
    python experiment.py --validate-only    check the setup; the model does not run
    python experiment.py --output my-run    choose the results folder
    python experiment.py --inputs media/    use a media folder that is not beside this file

Edit the numbered steps below to change the experiment. Each run writes a new
folder and never overwrites earlier results. Model weights are not included;
the first run can download them. Importing this file runs nothing.
"""
import argparse
import json
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent
'''

LOAD_PROFILE = '''\
if not args.factory:
    raise SystemExit("Supply the original --factory module:create_profile. "
                     "For lab calibration, use umi_lab.demo:create_profile.")
from umi_lab.worker import load_profile
profile = load_profile(args.factory)
subject = profile.subject
'''

LOAD_MODEL = '''
# Load the subject. This can download weights and needs memory.
from brainscore import load_model
subject = load_model(SUBJECT)
profile = None
'''

RUN = '''experiment = build_experiment(subject, args.output, args.inputs, profile)

# 6. Check the setup, then run. Only experiment.run() runs the model.
summary = experiment.validate()
if args.validate_only:
    print(json.dumps(summary, indent=2, default=str))
    return
experiment.run()
print(f"Saved results to {args.output.resolve()}")

# Read the results back later, for example in a notebook:
#   from brainscore.experiments import ExperimentResult
#   result = ExperimentResult.open("path/to/results")
'''

ARGUMENTS = '''def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=HERE / f"rerun-{time.time_ns()}",
                        help="folder for the new results (default: a new rerun-* folder)")
    parser.add_argument("--inputs", type=Path, default=HERE / "inputs",
                        help="folder with the uploaded media (default: inputs/ beside this file)")
    parser.add_argument("--validate-only", action="store_true",
                        help="check the setup without running the model")
'''

FACTORY_ARGUMENT = '''    parser.add_argument("--factory", default=PROFILE_FACTORY,
                        help="for app-local profiles: importable module:create_profile")
'''


def python_source(saved: dict, *, factory: str | None = None) -> str:
    """Generate data literals and public UMI calls, without loading a subject."""
    # Reject non-JSON values/NaNs instead of emitting undefined Python names.
    if not isinstance(saved, dict):
        raise ValueError('A Python export needs a saved experiment configuration')
    json.dumps(saved, allow_nan=False)
    subject = ''
    use_profile = True
    if saved.get('kind') == 'workbench':
        from .workbench import Request

        subject = saved.get('subject')
        if not isinstance(subject, str) or not 0 < len(subject) <= 300:
            raise ValueError('A Python export needs the saved subject identifier')
        source = saved.get('model_source', 'registered')
        if source not in ('registered', 'profile'):
            raise ValueError('Unknown model source')
        use_profile = source == 'profile'
        request = Request.parse(saved.get('request'))
        parts = (protocol_parts(request) if request.mode == 'protocol' else
                 benchmark_parts(request) if request.mode == 'benchmark' else session_parts(request))
    else:
        parts = calibration_parts(saved)
    if factory is not None and (not isinstance(factory, str) or len(factory) > 1000):
        raise ValueError('Invalid profile factory')

    out = DOCSTRING
    out += f'\n# 1. Subject: the model that takes part in the experiment.\nSUBJECT = {literal(subject)}\n'
    if use_profile:
        out += ('# This subject comes from an app-local profile; its factory and umi_lab must be importable.\n'
                f'PROFILE_FACTORY = {literal(factory)}\n')
    out += parts.constants
    out += ('\n\ndef build_experiment(subject, output, inputs=HERE / "inputs", profile=None):\n'
            '    """Describe the experiment. Building it does not run the model."""\n')
    out += indent(parts.import_lines() + '\n' + parts.body, '    ')
    out += '\n\ndef main():\n    args = parse_arguments()\n'
    out += indent(LOAD_PROFILE if use_profile else LOAD_MODEL, '    ')
    out += indent(RUN, '    ')
    out += ''.join(f'    #   {line}\n' for line in parts.read_back)
    out += '\n\n# Helpers. You do not need to edit below this line.\n\n' + ARGUMENTS
    if use_profile:
        out += FACTORY_ARGUMENT
    out += '    return parser.parse_args()\n'
    if parts.uses_stimuli:
        out += '\n' + STIMULI_HELPER
    if parts.uses_fitting_check:
        out += '\n' + FITTING_CHECK
    out += '\n\nif __name__ == "__main__":\n    main()\n'
    return out


def script_for_run(directory: Path) -> str:
    """Read saved configuration only; existing runs need no regeneration/inference."""
    filename = directory / 'plan.json'
    if filename.stat().st_size > MAX_PLAN_BYTES:
        raise ValueError('The saved configuration exceeds the Python export size limit')
    factory = None
    if (directory / 'profile.json').is_file():
        factory = json.loads((directory / 'profile.json').read_text()).get('factory')
    return python_source(json.loads(filename.read_text()), factory=factory)


def write_archive(directory: Path, destination: Path) -> None:
    source = script_for_run(directory)
    with ZipFile(destination, 'w', compression=ZIP_DEFLATED) as archive:
        for file in directory.rglob('*'):
            if file.is_file() and file.relative_to(directory).as_posix() != 'experiment.py':
                archive.write(file, file.relative_to(directory).as_posix())
        archive.writestr('experiment.py', source)
        if not (directory / 'README.txt').exists():
            archive.writestr('README.txt', ARCHIVE_README)
