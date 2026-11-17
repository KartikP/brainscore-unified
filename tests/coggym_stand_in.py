"""A stand-in for CogGym's evaluation package, written as files into a temporary checkout.

It has CogGym's shape (``evaluation/{__init__,providers,cli,analyze}.py``, the manifest, the
trial selection map, ``EML/<experiment>/``) and CogGym's call boundary: ``load_experiment``,
``_load_selection_map``, ``_selected_trials``, ``build_messages``, ``_run_experiment`` with
a provider, ``aggregate_experiment`` and ``analyze``. It is a provider-call seam, not a
substitute for CogGym's scientific evaluator. A test changes one behavior by passing Python
source that redefines a function of ``cli.py`` or ``analyze.py``.
"""
import json
from pathlib import Path

EXPERIMENT = 'Hu2023Fine/exp1'

CLI = '''
from types import SimpleNamespace

TRIALS = [f'trial-{index}' for index in range(3)]
# Calls of _run_experiment in this process: one process serves every repetition.
RUNS = []


def load_experiment(path):
    return SimpleNamespace(path=str(path), human_data_mean={'secret': 'never pass to provider'})


def _load_selection_map(path):
    return {}


def _selected_trials(experiment, selection, use_all):
    return [SimpleNamespace(id=trial, queries=[]) for trial in TRIALS]


def build_messages(trial, experiment):
    return 'system', [{'role': 'user', 'content': trial.id}]


def _run_experiment(experiment, selected, provider, model, temperature, max_tokens):
    RUNS.append(model)
    results = []
    for trial in selected:
        try:
            system, messages = build_messages(trial, experiment)
            raw = provider.complete_with_metadata(system, messages, model, temperature, max_tokens)
            results.append({'trial_id': trial.id, 'response': raw['text'],
                            'scores': {'answer': {'scorable': True}}})
        except Exception as error:
            results.append({'trial_id': trial.id, 'error': str(error)})
    return results


def aggregate_experiment(rows):
    return {'n_trials': len(rows)}
'''

ANALYZE = '''
def analyze(paths, eml, manifest, min_items=5):
    return [], [], []
'''

# An analysis that scores the run: one row with a fixed R², from the saved run files.
ANALYZE_ROW = '''
import json


def analyze(paths, eml, manifest, min_items=5):
    runs = [json.loads(path.read_text()) for path in paths]
    row = {'model': runs[0]['model'], 'experiment': runs[0]['experiment'],
           'r2': 0.375, 'n_items': 3, 'n_artifacts': len(runs)}
    return [row], [], []
'''


def write_checkout(root: Path, cli: str = '', analyze: str = '', providers: str = '') -> Path:
    """Write the stand-in checkout into ``root``; ``cli``, ``analyze`` and ``providers`` are
    Python source appended to those modules (a later definition replaces a function)."""
    evaluation = Path(root) / 'evaluation'
    evaluation.mkdir(parents=True, exist_ok=True)
    (evaluation / '__init__.py').write_text('')
    (evaluation / 'providers.py').write_text('# Stand-in: CogGym imports its model clients here.\n' + providers)
    (evaluation / 'cli.py').write_text(CLI + cli)
    (evaluation / 'analyze.py').write_text(ANALYZE + analyze)
    (evaluation / 'public_manifest.json').write_text(json.dumps({'experiments': [{'path': EXPERIMENT}]}))
    (evaluation / 'trial_selection_map.json').write_text('{}')
    (Path(root) / 'EML' / EXPERIMENT).mkdir(parents=True, exist_ok=True)
    return Path(root)


def write_ready_folder(folder: Path, **sources) -> Path:
    """A ready CogGym folder as Brain-Score makes it: the stand-in checkout, an environment
    whose Python runs this interpreter, and ``ready.json`` for the pinned commit and
    requirements. ``sources`` go to ``write_checkout``."""
    import hashlib
    import platform
    import sys
    from brainscore.harnesses._environment_folder import environment_python
    from brainscore.harnesses.coggym import COGGYM
    write_checkout(Path(folder) / 'checkout', **sources)
    python = environment_python(folder)
    python.parent.mkdir(parents=True, exist_ok=True)
    python.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    python.chmod(0o755)
    (Path(folder) / 'ready.json').write_text(json.dumps({
        'library': COGGYM.name, 'repository': COGGYM.repository, 'revision': COGGYM.commit,
        'requirements_sha256': hashlib.sha256(COGGYM.requirements.read_bytes()).hexdigest(),
        'python': platform.python_version(),
        'freeze': ['stand-in==0']}))
    return Path(folder)


class StandIn:
    """Change one behavior of the stand-in evaluator: Python source appended to its module."""

    def __init__(self, folder: Path) -> None:
        self.evaluation = Path(folder) / 'checkout' / 'evaluation'

    def cli(self, source: str) -> None:
        with (self.evaluation / 'cli.py').open('a') as module:
            module.write('\n' + source)

    def analyze(self, source: str) -> None:
        with (self.evaluation / 'analyze.py').open('a') as module:
            module.write('\n' + source)


def default_place(root: Path) -> Path:
    """Where Brain-Score puts CogGym's folder when ``BRAINSCORE_ENVIRONMENTS_HOME`` is ``root``."""
    from brainscore.harnesses._environment_folder import default_folder
    from brainscore.harnesses.coggym import COGGYM
    return Path(root) / 'CogGym' / default_folder(COGGYM).name
