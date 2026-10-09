"""Run the pinned CogGym evaluator with existing UMI experiment tools.

CogGym owns prompts, trial selection, parsing, and scientific scoring. This
adapter supplies a model provider and saves its unmodified trial results.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys

from brainscore.experiments import CallableProtocol
from brainscore_core.streaming import StreamEvent


REFERENCE_REVISION = 'a1cd9df1118fec80eba7463de237d1497d77e041'
DEFAULT_EXPERIMENT = 'Hu2023Fine/exp1'


def _prompt_key(system, messages):
    value = json.dumps([system, messages], sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(value.encode()).hexdigest()


def _check_checkout(checkout):
    def git(*arguments):
        return subprocess.check_output(
            ['git', '-C', str(checkout), *arguments], text=True,
            stderr=subprocess.STDOUT,
        ).strip()

    try:
        if git('rev-parse', 'HEAD') != REFERENCE_REVISION:
            raise ValueError(f'CogGym requires revision {REFERENCE_REVISION}')
        if git('status', '--porcelain', '--untracked-files=no', '--',
               'evaluation', 'EML'):
            raise ValueError('CogGym evaluation code or experiment data has tracked edits')
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError('Provide a local Git checkout of the pinned CogGym revision') from error


def _load_reference(checkout):
    _check_checkout(checkout)
    # Load relative imports without claiming the generic package name "evaluation".
    suffix = hashlib.sha256(str(checkout).encode()).hexdigest()[:16]
    name = f'_brainscore_coggym_{suffix}'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, checkout / 'evaluation' / '__init__.py',
            submodule_search_locations=[str(checkout / 'evaluation')],
        )
        package = importlib.util.module_from_spec(spec)
        sys.modules[name] = package
        spec.loader.exec_module(package)
    try:
        # Upstream imports google-genai even when the provider is a local model.
        importlib.import_module(f'{name}.providers')
    except ImportError as error:
        raise ImportError(
            'Install the pinned CogGym checkout\'s evaluation/requirements.txt '
            'in your experiment environment. Local providers need no API key.'
        ) from error
    return importlib.import_module(f'{name}.cli'), importlib.import_module(f'{name}.analyze')


class _SubjectProvider:
    """Translate CogGym provider calls into the existing response-trace contract."""

    def __init__(self, subject, repetition, experiment, prompt_trials):
        self.subject = subject
        self.experiment = experiment
        self.repetition = repetition
        self.call_index = 0
        self.prompt_trials = prompt_trials

    def complete_with_metadata(self, system, messages, model, temperature, max_tokens):
        request = dict(system=system, messages=messages, model=model,
                       temperature=temperature, max_tokens=max_tokens)
        trial_ids = self.prompt_trials[_prompt_key(system, messages)]
        event = StreamEvent(
            'generation_request', request, t_ms=None,
            meta={'experiment': self.experiment, 'repetition': self.repetition,
                  'provider_call': self.call_index, 'matching_trial_ids': trial_ids},
        )
        self.call_index += 1
        response = self.subject.process(event)
        if not isinstance(response, StreamEvent) or response.channel != 'response_trace':
            raise TypeError('CogGym needs a response_trace subject; use build_trace_subject')
        raw = response.payload['raw']
        if not isinstance(raw.get('text'), str):
            raise TypeError('CogGym provider output must contain text: str')
        # These fields are part of CogGym's provider contract. Do not invent them.
        if 'reasoning' not in raw or 'token_usage' not in raw:
            raise ValueError('Provider output needs reasoning and token_usage (None is allowed)')
        return raw


class CogGymRunner:
    """Prepare a public CogGym experiment before loading a model.

    ``protocol(reset=...)`` returns a standard CallableProtocol. ``run`` calls
    the same reference evaluator directly, for parity checks. The required
    reset callback receives a one-based repetition number; it must reset the
    provider's state/RNG without replacing the instrumented model or its hooks.

    ``trial_ids`` selects an explicit smoke subset of the canonical selection.
    It is always labelled as a subset, never a complete CogGym evaluation.
    """

    def __init__(self, checkout, *, model, temperature=1.0, max_tokens=512,
                 repetitions=1, trial_ids=None, experiment=DEFAULT_EXPERIMENT,
                 modalities=('text',)):
        if not isinstance(model, str) or not model.strip():
            raise ValueError('Supply the provider model identifier')
        if not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature < 0:
            raise ValueError('temperature must be finite and nonnegative')
        if type(max_tokens) is not int or max_tokens < 1:
            raise ValueError('max_tokens must be a positive integer')
        if type(repetitions) is not int or repetitions < 1:
            raise ValueError('repetitions must be a positive integer')
        self.checkout = Path(checkout).resolve()
        self.model, self.temperature = model, temperature
        self.max_tokens, self.repetitions = max_tokens, repetitions
        self._cli, self._analysis = _load_reference(self.checkout)
        manifest = json.loads((self.checkout / 'evaluation' / 'public_manifest.json').read_text())
        public = {entry['path'] for entry in manifest['experiments']}
        if experiment not in public:
            raise ValueError('experiment must be in the pinned public CogGym manifest')
        if not modalities or set(modalities) - {'text', 'image', 'video'}:
            raise ValueError('Declare supported modalities: text, image, video')
        self.experiment = experiment
        self.modalities = tuple(sorted(set(modalities)))
        self._experiment = self._cli.load_experiment(self.checkout / 'EML' / experiment)
        selection = self._cli._load_selection_map(
            self.checkout / 'evaluation' / 'trial_selection_map.json',
        )
        canonical = self._cli._selected_trials(self._experiment, selection, False)
        if trial_ids is None:
            self._trials = canonical
        else:
            requested = list(trial_ids)
            if not requested or len(set(requested)) != len(requested):
                raise ValueError('trial_ids must be nonempty and distinct')
            known = {trial.id for trial in canonical}
            if set(requested) - known:
                raise ValueError('trial_ids must belong to the canonical CogGym selection')
            self._trials = [trial for trial in canonical if trial.id in requested]
        if not self._trials:
            raise ValueError('The reference trial selection is empty')
        self.canonical_trial_count = len(canonical)
        self._prompt_trials = {}
        # Preflight the reference prompts, including media loading, before inference.
        for trial in self._trials:
            system, messages = self._cli.build_messages(trial, self._experiment)
            required = {'text'}
            for message in messages:
                content = message['content']
                if isinstance(content, list):
                    for part in content:
                        kind = part.get('type')
                        if kind not in {'text', 'image', 'image_url', 'video'}:
                            raise ValueError(f'Unsupported CogGym content type: {kind}')
                        required.add('image' if kind == 'image_url' else kind)
                elif not isinstance(content, str):
                    raise TypeError('CogGym message content must be text or a list of parts')
            if not isinstance(system, str) or required - set(self.modalities):
                raise ValueError(f'Provider must support these modalities: {sorted(required)}')
            self._prompt_trials.setdefault(_prompt_key(system, messages), []).append(trial.id)

    @property
    def trial_ids(self):
        return tuple(trial.id for trial in self._trials)

    def describe(self):
        return {
            'reference_revision': REFERENCE_REVISION,
            'experiment': self.experiment, 'model': self.model,
            'modalities': list(self.modalities),
            'temperature': self.temperature, 'max_tokens': self.max_tokens,
            'repetitions': self.repetitions, 'trial_ids': list(self.trial_ids),
            'canonical_trial_count': self.canonical_trial_count,
            'scope': ('canonical_selection' if len(self._trials) == self.canonical_trial_count
                      else 'smoke_subset'),
            'clock': 'no physical clock; provider_call is an attempt index',
            'reset': 'caller-supplied callback before each repetition',
        }

    def protocol(self, *, reset):
        """Build evaluation steps for Experiment without running CogGym yet.

        CallableProtocol lets tools observe the subject while CogGym keeps its
        trial loop and scoring. reset(repetition) clears provider history and
        sampling state as needed, while preserving model instrumentation.
        """
        if not callable(reset):
            raise TypeError('Supply a reset(repetition) callback for your provider')

        def evaluate(subject, context):
            def register(path):
                context.artifact(
                    path, producer='CogGym evaluator via UMI adapter',
                    description='Reference trial results, summary, and explicit run coverage',
                )
            # Translate CogGym provider calls into subject.process() calls.
            # This is where the external evaluator connects to UMI tools.
            return self._run(
                lambda repetition: _SubjectProvider(
                    subject, repetition, self.experiment, self._prompt_trials,
                ),
                context.directory / 'coggym', reset, register,
            )

        return CallableProtocol(
            f'coggym:{self.experiment}', evaluate, methods=['process'], metadata=self.describe(),
        )

    def run(self, provider, *, output_dir, reset):
        """Run CogGym directly, without UMI observation or instrumentation."""
        if not callable(getattr(provider, 'complete_with_metadata', None)):
            raise TypeError('Provider must implement complete_with_metadata')
        return self._run(lambda repetition: provider, Path(output_dir), reset, lambda path: None)

    def _run(self, provider_factory, directory, reset, register):
        if not callable(reset):
            raise TypeError('Supply a reset(repetition) callback for your provider')
        _check_checkout(self.checkout)
        directory.mkdir(parents=True, exist_ok=False)
        artifacts = []
        for repetition in range(1, self.repetitions + 1):
            reset(repetition)
            results = self._cli._run_experiment(
                self._experiment, self._trials, provider_factory(repetition),
                self.model, self.temperature, self.max_tokens,
            )
            coverage = {
                'expected_trials': len(self._trials),
                'returned_trials': len(results),
                'error_trials': sum('error' in result for result in results),
                'unscorable_trials': sum(not any(
                    score.get('scorable') for score in result.get('scores', {}).values()
                ) for result in results if 'error' not in result),
            }
            complete = (
                [result.get('trial_id') for result in results] == list(self.trial_ids)
                and coverage['error_trials'] == 0
            )
            artifact = {
                'model': self.model, 'experiment': self.experiment, 'repetition': repetition,
                'temperature': self.temperature, 'max_tokens': self.max_tokens,
                'selection_map': str(self.checkout / 'evaluation' / 'trial_selection_map.json'),
                'summary': None, 'trials': results,
                'umi': {**self.describe(), 'coverage': coverage,
                        'status': 'complete' if complete else 'failed',
                        'summary_status': 'pending'},
            }
            path = directory / f'run-{repetition:03d}.json'
            path.write_text(json.dumps(artifact, indent=2, allow_nan=False) + '\n')
            artifacts.append(path)
            # Keep completed model calls even if the reference scorer overflows.
            try:
                artifact['summary'] = self._cli.aggregate_experiment(results)
                artifact['umi']['summary_status'] = 'complete'
            except BaseException as error:
                artifact['umi']['summary_status'] = 'failed'
                artifact['umi']['summary_error'] = {
                    'type': type(error).__name__, 'message': str(error),
                }
                raise
            finally:
                path.write_text(json.dumps(artifact, indent=2, allow_nan=False) + '\n')
                register(path)
            if not complete:
                raise RuntimeError(f'CogGym returned incomplete or failed trials; inspect {path}')
        try:
            rows, summary, skipped = self._analysis.analyze(
                artifacts, self.checkout / 'EML',
                self.checkout / 'evaluation' / 'public_manifest.json', min_items=5,
            )
        except BaseException as error:
            path = directory / 'analysis-error.json'
            path.write_text(json.dumps({
                'status': 'failed', 'type': type(error).__name__, 'message': str(error),
                'configuration': self.describe(),
            }, indent=2, allow_nan=False) + '\n')
            register(path)
            raise
        result = {'configuration': self.describe(), 'experiments': rows,
                  'summary': summary, 'skipped': skipped,
                  'repetitions': [json.loads(path.read_text())['umi']['coverage'] for path in artifacts]}
        path = directory / 'analysis.json'
        path.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        register(path)
        if skipped:
            raise RuntimeError(f'CogGym analysis skipped artifacts; inspect {path}')
        return result
