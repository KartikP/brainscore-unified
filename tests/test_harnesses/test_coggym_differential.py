"""The evaluator in its own process gives exactly what it gave in the scoring process.

The same stand-in evaluator runs twice with the same provider: through ``CogGymRunner``
(a child process with its own Python) and through the earlier in-process path, which is kept
here only, as the reference: CogGym's package loaded into this process with ``importlib`` and
its functions called directly. Every trial table, summary, validation report and analysis
row must be equal.
"""
import importlib
import importlib.util
import json
import sys

import pytest

from brainscore.harnesses import coggym
from brainscore.harnesses.coggym_validation import validate_slider_responses

from tests.command_stand_ins import pinned_git  # noqa: F401  (a fixture)
from tests.coggym_stand_in import ANALYZE_ROW, EXPERIMENT, write_ready_folder

pytestmark = pytest.mark.unit

# Trials with a slider question, so the validation report has content.
SLIDER_TRIALS = '''
def _selected_trials(experiment, selection, use_all):
    slider = SimpleNamespace(tag='rating', type='single-slider', slider_config={'min': 0, 'max': 100}, option=[])
    return [SimpleNamespace(id=trial, queries=[slider]) for trial in TRIALS]
'''


class _Provider:
    """A deterministic model client with state: each reply counts the calls it has served."""

    def __init__(self, fail_on=None):
        self.calls, self.fail_on = 0, fail_on

    def reset(self, repetition):
        """The runner's reset callback: each repetition starts with a fresh count."""
        self.calls = 0

    def complete_with_metadata(self, system, messages, model, temperature, max_tokens):
        self.calls += 1
        if messages[0]['content'] == self.fail_on:
            raise ValueError(f'no answer for {self.fail_on}')
        return {'text': f'{{"answer": {self.calls}}}', 'reasoning': None,
                'token_usage': {'calls': self.calls}}


def _in_process(folder, provider, repetitions, directory):
    """The earlier path: CogGym's package in this process, its functions called directly, the
    reset before each repetition, and its own run files (in ``directory``) for its analysis."""
    checkout = folder / 'checkout'
    name = '_coggym_reference_in_process'
    spec = importlib.util.spec_from_file_location(
        name, checkout / 'evaluation' / '__init__.py', submodule_search_locations=[str(checkout / 'evaluation')])
    package = importlib.util.module_from_spec(spec)
    sys.modules[name] = package
    try:
        spec.loader.exec_module(package)
        importlib.import_module(f'{name}.providers')
        cli, analysis = importlib.import_module(f'{name}.cli'), importlib.import_module(f'{name}.analyze')
        experiment = cli.load_experiment(checkout / 'EML' / EXPERIMENT)
        trials = cli._selected_trials(experiment, cli._load_selection_map(
            checkout / 'evaluation' / 'trial_selection_map.json'), False)
        runs, paths = [], []
        directory.mkdir(parents=True)
        for repetition in range(1, repetitions + 1):
            provider.reset(repetition)
            results = cli._run_experiment(experiment, trials, provider, 'fixture', 1.0, 512)
            runs.append({'trials': results, 'summary': cli.aggregate_experiment(results),
                         'validation': validate_slider_responses(trials, results)})
            path = directory / f'run-{repetition:03d}.json'
            path.write_text(json.dumps({'model': 'fixture', 'experiment': EXPERIMENT, 'trials': results}))
            paths.append(path)
        rows = analysis.analyze(paths, checkout / 'EML', checkout / 'evaluation' / 'public_manifest.json',
                                min_items=5)
        return runs, rows
    finally:
        for module in [key for key in sys.modules if key == name or key.startswith(name + '.')]:
            del sys.modules[module]


@pytest.fixture
def folder(tmp_path, pinned_git):
    return write_ready_folder(tmp_path / 'coggym', cli=SLIDER_TRIALS, analyze=ANALYZE_ROW)


def test_a_complete_run_is_equal_in_both_paths(folder, tmp_path):
    runner = coggym.CogGymRunner(folder, model='fixture', repetitions=2)
    provider = _Provider()
    child = runner.run(provider, output_dir=tmp_path / 'child', reset=provider.reset)
    saved = [json.loads((tmp_path / f'child/run-00{index}.json').read_text()) for index in (1, 2)]
    reference, rows = _in_process(folder, _Provider(), 2, tmp_path / 'in-process')
    for file, expected in zip(saved, reference):
        assert file['trials'] == expected['trials']
        assert file['summary'] == expected['summary']
        assert file['umi']['response_validation'] == expected['validation']
    assert child['experiments'] == rows[0]
    assert (child['summary'], child['skipped']) == (rows[1], rows[2])
    assert child['response_validation']['repetitions'] == [run['validation'] for run in reference]


def test_a_failed_trial_is_recorded_alike_in_both_paths(folder, tmp_path):
    runner = coggym.CogGymRunner(folder, model='fixture')
    provider = _Provider(fail_on='trial-1')
    with pytest.raises(RuntimeError, match='incomplete or failed'):
        runner.run(provider, output_dir=tmp_path / 'child', reset=provider.reset)
    saved = json.loads((tmp_path / 'child/run-001.json').read_text())
    reference, _ = _in_process(folder, _Provider(fail_on='trial-1'), 1, tmp_path / 'in-process')
    assert saved['trials'] == reference[0]['trials']
    assert saved['trials'][1]['error'] == 'no answer for trial-1'
