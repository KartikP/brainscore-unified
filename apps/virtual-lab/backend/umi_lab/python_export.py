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
    return pformat(value, width=92, sort_dicts=False)


STIMULI_HELPER = '''
def make_stimuli(rows, inputs):
    """Resolve the archive's media files and preserve stimulus IDs and columns."""
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


def tables(request) -> str:
    fitting = [{**row, 'stimulus_id': row.get('stimulus_id', f'fitting-{index}')}
               for index, row in enumerate(request.fitting_rows)] if request.mode == 'protocol' else request.fitting_rows
    source = (f"load_stimulus_set({literal(request.stimulus_set)})" if request.stimulus_set
              else f"make_stimuli({literal(request.rows)}, inputs)")
    return (f"stimuli = {source}\n"
            f"fitting = make_stimuli({literal(fitting)}, inputs)\n" if fitting else f"stimuli = {source}\nfitting = None\n")


def change_expression(change: dict, *, protocol: bool) -> str:
    operation = change.get('operation', 'zero')
    return (f"StateChange(kind={literal('ablation' if operation == 'zero' else 'stimulation')},\n"
            f"    target=Selection(layer={literal(change['target'])}, indices={literal(change.get('indices'))}, "
            f"axis={literal(change.get('axis', -1) if protocol else None)}),\n"
            f"    perturbation=Perturbation(kind={literal(operation)}, "
            f"scale={literal(change.get('value', 0))}, amount={literal(change.get('value', 0))}))")


def protocol_body(request) -> str:
    from .protocol import ProtocolSpec

    spec = ProtocolSpec.parse(request.protocol)
    task = spec.task
    kind = task['kind']
    response = 'None'
    if kind == 'choice':
        confidence = f"Scale(**{literal(task['confidence'])})" if task.get('confidence') else 'None'
        response = f"Choice({literal(task['labels'])}, probabilities={task['probabilities']}, confidence={confidence})"
    elif kind == 'rating':
        response = f"Scale(**{literal(task['scale'])})"
    elif kind == 'text':
        response = 'Text()'
    body = dedent('''\
        from brainscore import load_stimulus_set
        from brainscore.experiments import Experiment, TaskProtocol, RecordBehavior, RecordNeural, RecordInputsOutputs, Intervene
        from brainscore_core import TaskContext, Selection, Perturbation, StateChange
        from brainscore_core.responses import Choice, Scale, Text, trial_prompt

        # 2. Prepare test stimuli and separate fitting examples.
        ''') + tables(request)
    if kind != 'passive' and task['path'] == 'readout':
        body += dedent('''\
            if fitting is None or "label" not in fitting.columns:
                raise ValueError("A fitted readout needs a separate fitting table with a label column")
            if any(value is None or not str(value).strip() for value in fitting["label"]):
                raise ValueError("Fill the label for every fitting example")
            ''')
        if kind in ('choice', 'legacy'):
            body += ("fitted_labels = set(map(str, fitting['label']))\n"
                     f"if len(fitted_labels) < 2 or fitted_labels - set({literal(task['labels'])}):\n"
                     "    raise ValueError('Choice fitting examples need at least two classes drawn from the response labels')\n")
    body += f'''
# 3. Define the task and the form of its answer.
# Text() keeps a written reply; Choice(...) constrains the answer labels.
task = TaskContext(
    task_type={literal('passive' if kind == 'passive' else 'probabilities' if kind == 'legacy' else None)},
    instruction={literal(task['instruction'] or None)},
    response={response},
    label_set={literal(task['labels'] or None)},
    fitting_stimuli=fitting,
    prefer_path={literal(task['path'])},
    readout_fit={literal(task['readout_fit'])},
    present={literal(tuple(task['present']) or None)},
)
'''
    if kind != 'passive':
        body += 'for _, row in stimuli.iterrows():\n    trial_prompt(task, row)  # Check multimodal part order before inference.\n'
    body += '\n# 4. Choose measurements from the same presentation.\ntools = []\n'
    if spec.measurements['regions']:
        body += f"tools.append(RecordNeural({literal(spec.measurements['regions'])}))\n"
    if spec.measurements['behavior']:
        body += 'tools.append(RecordBehavior())\n'
    if spec.measurements['replay']:
        body += 'tools.append(RecordInputsOutputs())\n'
    body += '\n# 5. Set independent conditions and repetitions. Interventions are undone automatically.\n'
    for index, condition in enumerate(spec.conditions):
        if condition['intervention']:
            body += (f"tools.append(Intervene(\n    {change_expression(condition['intervention'], protocol=True)},\n"
                     f"    conditions={literal([condition['name']])}, name={literal(f'condition_{index}')}))\n")
    body += f'''
protocol = TaskProtocol(
    stimuli, task,
    identifier={literal(spec.identifier)},
    conditions={literal([condition['name'] for condition in spec.conditions])},
    repetitions=tuple(range({spec.repetitions})),
    required_traits={literal(spec.required_traits)},
)
return Experiment(subject=subject, protocol=protocol, tools=tools, output_dir=output,
                  metadata={{"interface": "umi-lab-protocol-builder-v1"}})
'''
    return body


def session_body(request) -> str:
    body = dedent('''\
        from contextlib import contextmanager
        from brainscore import load_stimulus_set
        from brainscore.experiments import Experiment, SessionProtocol, RecordInputsOutputs, Intervene
        from brainscore_core import TaskContext, Selection, Perturbation, StateChange
        from brainscore_core.streaming import BufferedSession, StreamEvent
        from brainscore_core.streaming_helpers import BehavioralSession, STIMULUS_COLUMN_TO_MODALITY

        class FiniteSession(BufferedSession):
            complete = False
            def next_input(self):
                event = super().next_input()
                if event is None:
                    self.complete = True
                return event

        class FiniteBehaviorSession(BehavioralSession):
            complete = False
            def next_input(self):
                event = super().next_input()
                if event is None:
                    self.complete = True
                return event

        ''')
    if request.mode == 'session':
        if not request.events:
            raise ValueError('Add at least one input event')
        for event in request.events:
            if (set(event) - {'channel', 'payload', 't_ms', 'meta'}
                    or not isinstance(event.get('channel'), str) or 'payload' not in event):
                raise ValueError('Each event needs channel and payload; optional t_ms and meta')
        body += (f"events = [StreamEvent(row['channel'], row['payload'], row.get('t_ms'), row.get('meta', {{}})) for row in {literal(request.events)}]\n"
                 "inputs_declared = sorted({event.channel for event in events})\n")
        session = f"FiniteSession(events, requested_output_channels=({literal(request.output)},))"
    else:
        body += tables(request)
        body += 'inputs_declared = sorted({modality for column, modality in STIMULUS_COLUMN_TO_MODALITY.items() if column in stimuli.columns})\n'
        if request.output == 'behavior':
            body += (f"task = TaskContext(task_type='probabilities', instruction={literal(request.instruction or None)},\n"
                     f"    label_set={literal(request.labels or None)}, fitting_stimuli=fitting,\n"
                     f"    prefer_path={literal(request.prefer_path)}, metadata={{'stimulus_set': stimuli}})\n")
            session = 'FiniteBehaviorSession(task)'
        elif request.output.startswith('neural:'):
            session = f"FiniteSession(stimulus_set=stimuli, requested_output_channels=({literal(request.output)},))"
        elif request.output == 'response_trace':
            body += f'''events = [StreamEvent("generation_request", {{
    "system": {literal(request.instruction or None)},
    "messages": [{{"role": "user", "content": str(row["sentence"])}}],
    "temperature": 0, "max_tokens": 64,
}}, None, {{"stimulus_id": str(row["stimulus_id"]), "sentence": str(row["sentence"])}})
    for _, row in stimuli.iterrows()]
inputs_declared = ["generation_request"]
'''
            session = "FiniteSession(events, requested_output_channels=('response_trace',))"
        else:
            raise ValueError('Use a declared neural output, behavior or response_trace for stimuli')
    body += f'\n@contextmanager\ndef session_factory(repetition):\n    yield {session}\n\n'
    body += 'tools = [RecordInputsOutputs()]\n' if request.record else 'tools = []\n'
    if request.intervention:
        body += (f"tools.append(Intervene(\n    {change_expression(request.intervention, protocol=False)},\n"
                 "    conditions=['intervention'], name='subject_intervention'))\n")
    conditions = ['baseline', 'intervention', 'restored'] if request.intervention else ['baseline']
    body += f'''protocol = SessionProtocol(
    "umi-workbench-session-v1", session_factory,
    input_channels=inputs_declared, output_channels=({literal(request.output)},),
    conditions={literal(conditions)}, repetitions=tuple(range({request.repeats})),
)
return Experiment(subject=subject, protocol=protocol, tools=tools, output_dir=output)
'''
    return body


def benchmark_body(request) -> str:
    return f'''from brainscore import load_benchmark, score
from brainscore.experiments import Experiment

benchmark = load_benchmark({literal(request.benchmark)})

class BenchmarkProtocol:
    identifier = {literal(request.benchmark)}
    def describe(self):
        return {{"benchmark": self.identifier, "reset_policy": "before and after the whole benchmark"}}
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
        body = (protocol_body(request) if request.mode == 'protocol' else
                benchmark_body(request) if request.mode == 'benchmark' else session_body(request))
    else:
        from .experiment import Plan

        plan = Plan.parse(saved)
        body = ("from umi_lab.experiment import Plan, build_experiment as calibration_experiment\n"
                f"return calibration_experiment(profile, Plan.parse({literal(asdict(plan))}), output)\n")
    if factory is not None and (not isinstance(factory, str) or len(factory) > 1000):
        raise ValueError('Invalid profile factory')
    header = dedent('''\
        """Editable UMI experiment exported by the virtual lab.

        Requires Python 3.11 and a matching UMI v2 installation, model weights and
        any registered dataset dependencies. Nothing runs when this file is imported.

        Extract All artifacts, then: python experiment.py --output ./my-rerun
        Or supply --inputs /path/to/extracted/inputs when downloading this file alone.
        --validate-only loads the subject and checks the plan without model inference.
        This creates a new run; it does not reopen or overwrite the original recordings.
        """
        import argparse
        import json
        from pathlib import Path
        import time

        BASE = Path(__file__).resolve().parent
        ''')
    header += f'SUBJECT = {literal(subject)}\n'
    if use_profile:
        header += f'PROFILE_FACTORY = {literal(factory)}\n'
        header += '# This run uses an app-local profile: its factory and umi_lab must be importable.\n'
    if saved.get('kind') == 'workbench':
        header += '\n' + STIMULI_HELPER
    header += '\n\ndef build_experiment(subject, output, inputs=BASE / "inputs", profile=None):\n'
    header += indent(body, '    ')
    header += dedent('''

        def main():
            parser = argparse.ArgumentParser(description=__doc__)
            parser.add_argument("--output", type=Path, default=BASE / f"rerun-{time.time_ns()}")
            parser.add_argument("--inputs", type=Path, default=BASE / "inputs")
            parser.add_argument("--validate-only", action="store_true")
        ''')
    if use_profile:
        header += indent(dedent('''\
            parser.add_argument("--factory", default=PROFILE_FACTORY,
                                help="For app-local profiles: importable module:create_profile")
            args = parser.parse_args()
            if not args.factory:
                parser.error("Supply the original --factory module:create_profile. For lab calibration, use umi_lab.demo:create_profile.")
            from umi_lab.worker import load_profile
            profile = load_profile(args.factory)
            subject = profile.subject
            '''), '    ')
    else:
        header += indent(dedent('''\
            args = parser.parse_args()
            # 1. Load the subject. This may require weights and substantial memory.
            from brainscore import load_model
            subject = load_model(SUBJECT)
            profile = None
            '''), '    ')
    header += indent(dedent('''\
            experiment = build_experiment(subject, args.output, args.inputs, profile)
            # 6. Validate, then run. The experiment owns resets and intervention cleanup.
            plan = experiment.validate()
            if args.validate_only:
                print(json.dumps(plan, indent=2, default=str))
                return
            result = experiment.run()
            print(f"Saved a new experiment to {args.output.resolve()}")
            # Reopen saved measurements later with ExperimentResult.open(args.output).
            # TaskProtocol recordings support result.neural(region, condition=..., repetition=0)
            # and result.behavior(condition=..., repetition=0, measure="text"), as applicable.
            '''), '    ')
    header += '\n\nif __name__ == "__main__":\n    main()\n'
    return header


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
