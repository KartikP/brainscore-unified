"""CogGym's evaluator in its own process: run as ``<CogGym environment python> _worker.py <checkout>``.

This script runs in the environment Brain-Score makes for CogGym, which holds CogGym's
pinned packages and no Brain-Score package. It imports only the standard library, CogGym's
``evaluation`` package from the checkout, and ``validation.py`` (standard library
only) from its own folder. It answers the parent's commands on the two pipes of
``brainscore/harnesses/_child_process.py``:

- ``{"do": "prepare", "experiment", "trial_ids"}``: the canonical trial ids, and the system
  text and messages of each selected trial (CogGym's ``build_messages``);
- ``{"do": "run", "model", "temperature", "max_tokens"}``: CogGym's
  ``_run_experiment`` for one repetition. Each model call goes to the parent as
  ``{"call": {"trial_id", "system", "messages", "model", "temperature", "max_tokens"}}``, where
  ``trial_id`` is the id of the trial CogGym last built messages for (its own trial object, so
  a retry carries the same id); the reply is
  ``{"reply": {"text", "reasoning", "token_usage"}}`` or ``{"error": [type name, message]}``,
  which this side raises as an exception of that type name and message, so CogGym stores
  the same failure text as in the parent's process (also for a trial with no written
  answer: CogGym's own record of a call with no reply). It returns the results, the
  state of each requested answer and CogGym's summary (or the summary's error);
- ``{"do": "analyze", "paths"}``: CogGym's ``analyze`` of the saved run files.

One process serves every repetition and the analysis of a run, so CogGym's module state
lives across repetitions as in a single process.
"""
import importlib
import importlib.util
import json
import os
import sys
import traceback
from pathlib import Path


class _Pipe:
    def __init__(self):
        read, write = (int(fd) for fd in os.environ['BRAINSCORE_CHILD_FDS'].split(','))
        self.read, self.write = os.fdopen(read, 'rb'), os.fdopen(write, 'wb')

    def send(self, message):
        # NaN and infinity pass as Python's JSON writes them: CogGym's values reach the parent
        # unchanged, and the parent's own checks refuse them as in one process.
        self.write.write((json.dumps(message) + '\n').encode())
        self.write.flush()

    def receive(self):
        line = self.read.readline()
        if not line:
            raise SystemExit(0)          # the parent closed its end: stop
        return json.loads(line)


def _load(name, path, package=False):
    spec = importlib.util.spec_from_file_location(
        name, path, submodule_search_locations=[str(path.parent)] if package else None)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class _Provider:
    """CogGym's model client: every call goes to the parent, which asks the subject."""

    def __init__(self, pipe):
        self.pipe = pipe
        self.trial_id = None        # the trial CogGym's loop last built messages for

    def complete_with_metadata(self, system, messages, model, temperature, max_tokens):
        self.pipe.send({'call': {'trial_id': self.trial_id, 'system': system, 'messages': messages,
                                 'model': model, 'temperature': temperature, 'max_tokens': max_tokens}})
        answer = self.pipe.receive()
        if 'reply' in answer:
            return answer['reply']
        name, message = answer['error']
        raise type(name, (Exception,), {})(message)


def main(checkout):
    pipe = _Pipe()
    try:
        # CogGym's evaluation package from the checkout; its providers module first, as CogGym's
        # own client imports it (it needs the Google SDK even for a local model).
        _load('evaluation', checkout / 'evaluation' / '__init__.py', package=True)
        importlib.import_module('evaluation.providers')
        cli = importlib.import_module('evaluation.cli')
        analysis = importlib.import_module('evaluation.analyze')
        validation = _load('_validation', Path(__file__).with_name('validation.py'))
        startup = None
    except Exception as error:
        # The evaluator cannot start (a missing package, a broken checkout): every command is
        # answered with this error, so the parent reports it in full.
        startup = {'type': type(error).__name__, 'message': str(error),
                   'traceback': traceback.format_exc(), 'startup': True}
    experiment, trials = None, {}
    while True:
        command = pipe.receive()
        if startup is not None:
            pipe.send({'failed': startup})
            continue
        try:
            if command['do'] == 'prepare':
                experiment = cli.load_experiment(checkout / 'EML' / command['experiment'])
                selection = cli._load_selection_map(checkout / 'evaluation' / 'trial_selection_map.json')
                canonical = cli._selected_trials(experiment, selection, False)
                wanted = command['trial_ids']
                trials = [trial for trial in canonical if wanted is None or trial.id in wanted]
                prompts = []
                for trial in trials:
                    system, messages = cli.build_messages(trial, experiment)
                    prompts.append({'id': trial.id, 'system': system, 'messages': messages})
                # A trial whose answers cannot be told apart is refused before any call.
                validation.validate_responses(trials, [])
                done = {'canonical': [trial.id for trial in canonical], 'trials': prompts}
            elif command['do'] == 'run':
                # The trials of this process's prepare: the same objects in every repetition.
                provider, build = _Provider(pipe), cli.build_messages

                def build_messages(trial, experiment):
                    # CogGym's loop builds a trial's messages before its calls (and their
                    # retries): the trial's own id goes with each of them. Prompts are unchanged.
                    # This holds for the pinned commit (cli.py:78); on a change of the pin,
                    # check it again with the real CogGym.
                    provider.trial_id = trial.id
                    return build(trial, experiment)
                cli.build_messages = build_messages
                try:
                    results = cli._run_experiment(experiment, trials, provider, command['model'],
                                                  command['temperature'], command['max_tokens'])
                finally:
                    cli.build_messages = build
                done = {'results': results,
                        'validation': validation.validate_responses(trials, results)}
                try:
                    done['summary'] = cli.aggregate_experiment(results)
                except Exception as error:
                    done['summary_error'] = {'type': type(error).__name__, 'message': str(error)}
            elif command['do'] == 'analyze':
                rows, summary, skipped = analysis.analyze(
                    [Path(path) for path in command['paths']], checkout / 'EML',
                    checkout / 'evaluation' / 'public_manifest.json', min_items=5)
                done = {'rows': rows, 'summary': summary, 'skipped': skipped}
            else:
                raise ValueError(f"unknown command {command.get('do')!r}")
        except Exception as error:
            pipe.send({'failed': {'type': type(error).__name__, 'message': str(error),
                                  'traceback': traceback.format_exc()}})
            continue
        pipe.send({'done': done})


if __name__ == '__main__':
    main(Path(sys.argv[1]))
