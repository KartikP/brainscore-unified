"""Run the pinned CogGym evaluator with existing UMI experiment tools.

CogGym owns prompts, trial selection, parsing, and scientific scoring. This
adapter turns each provider call into a session on the common channels and
saves CogGym's unmodified trial results.

Each call becomes one stimulus row: its text, image and video parts in
CogGym's order (``TaskContext.present``). The system text is the task's
instruction, the temperature and length are settings on the ``Text``
response, and the subject answers on ``behavior:text``. The subject builds its
own prompt from these, so a chat model can place the instruction in the user
turn where CogGym's own client sends a separate system text. Scores can
therefore differ a little from scores made with CogGym's own client.
"""
from __future__ import annotations

import base64
import hashlib
import importlib
import importlib.util
import json
import math
import mimetypes
from pathlib import Path
import subprocess
import sys
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd

from brainscore.experiments import CallableProtocol
from brainscore.harnesses.coggym_validation import SLIDER_VALIDATION_POLICY, validate_slider_responses
from brainscore_core.compatibility import check_channel_compatibility
from brainscore_core.contract import TaskContext
from brainscore_core.responses import Text
from brainscore_core.streaming import Session
from brainscore_core.streaming_helpers import stimulus_session
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
from brainscore.instrumentation import observed


REFERENCE_REVISION = 'a1cd9df1118fec80eba7463de237d1497d77e041'
DEFAULT_EXPERIMENT = 'Hu2023Fine/exp1'
# The input channel and the stimulus column of each kind of message part.
_PART_CHANNELS = {'text': 'text', 'image': 'vision:image', 'video': 'vision:video'}
_PART_COLUMNS = {'text': 'text', 'image': 'image_path', 'video': 'video_path'}
# Why a score can differ from one made with CogGym's own client (Contract Rule 3.9).
_PROMPT_NOTE = {
    'kind': 'subject_prompt',
    'reason': ('The subject builds its own prompt from the instruction and the parts. '
               'A chat model can place the instruction in the user turn, where '
               "CogGym's own client sends a separate system text."),
}


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


def _check_image(part) -> bytes:
    """The bytes of a base64 image part, or an error before any compute."""
    source = part.get('source') or {}
    if source.get('type') != 'base64' or not isinstance(source.get('data'), str):
        raise ValueError('A CogGym image part must carry base64 data')
    return base64.b64decode(source['data'], validate=True)


def _media_file(part, directory: Path) -> str:
    """Write a base64 image part to a file named by its content; return the path."""
    data = _check_image(part)
    media_type = (part.get('source') or {}).get('media_type') or ''
    suffix = '.jpg' if media_type == 'image/jpeg' else mimetypes.guess_extension(media_type) or '.png'
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'{hashlib.sha256(data).hexdigest()}{suffix}'
    if not path.exists():
        path.write_bytes(data)
    return str(path)


def _trial_row(messages, media_directory: Path) -> tuple[dict, tuple[str, ...]]:
    """One stimulus row from CogGym's user message, and the order of its parts.

    The first part of each kind goes in the usual column (``text``,
    ``image_path``, ``video_path``); later parts of the same kind go in
    ``<column>_2``, ``<column>_3`` and so on. The order is CogGym's.
    """
    if len(messages) != 1 or messages[0].get('role') != 'user':
        raise ValueError('A CogGym request must hold exactly one user message')
    content = messages[0]['content']
    parts = [{'type': 'text', 'text': content}] if isinstance(content, str) else content
    row, present, counts = {}, [], {}
    for part in parts:
        kind = part.get('type')
        if kind == 'text':
            value = part['text']
        elif kind == 'image':
            value = _media_file(part, media_directory)
        elif kind == 'video':
            value = str(part['path'])
        else:
            raise ValueError(f'Unsupported CogGym content type: {kind}')
        counts[kind] = counts.get(kind, 0) + 1
        column = _PART_COLUMNS[kind] + ('' if counts[kind] == 1 else f'_{counts[kind]}')
        row[column] = value
        present.append(f'{kind}:{column}')
    return row, tuple(present)


def _written_answer(event) -> str:
    """The one written answer in a ``behavior:text`` event."""
    payload = event.payload
    if not isinstance(payload, str):
        valid = getattr(payload, 'coords', {}).get('answer_valid')
        if valid is not None and not bool(np.all(valid.values)):
            raise TypeError('The subject gave no written answer on behavior:text')
        values = np.asarray(getattr(payload, 'values', payload), dtype=object).reshape(-1)
        if values.size != 1:
            raise ValueError(f'A CogGym call needs one written answer, not {values.size}')
        payload = values[0]
    if not isinstance(payload, str):
        raise TypeError('CogGym needs the answer on behavior:text as text (str)')
    return payload


class _SubjectProvider:
    """Translate CogGym provider calls into sessions on the common channels.

    Each provider call is one ``stimulus_session``: one stimulus row (the
    message parts in order), a ``TaskContext`` (the system text as the
    instruction, a ``Text`` response with the call's temperature and length,
    the order of the parts, ``trial_memory='none'``) and the output
    ``behavior:text``. The subject answers through ``participate`` only
    (Contract Rule 1.1). Each session is one repetition of the run record
    (``<CogGym repetition>.<call>``) and is observed with ``observed``, so
    tools record the request and the answer between that session's own
    boundaries, and ``replay_sessions`` rebuilds one session per call. The
    subject is not reset between calls; the provider's reset(repetition)
    runs only between CogGym repetitions.

    CogGym stores ``reasoning`` and ``token_usage`` with each trial and never
    scores them. They come from the extra data (``meta``) of the answer event
    when the subject gives them; otherwise they are ``None``.
    """

    def __init__(self, subject, context, repetition, prompt_trials):
        self.subject = subject
        self.context = context
        self.repetition = repetition
        self.call_index = 0
        self.prompt_trials = prompt_trials
        self.media_directory = context.directory / 'coggym-media'

    def complete_with_metadata(self, system, messages, model, temperature, max_tokens):
        trial_ids = self.prompt_trials[_prompt_key(system, messages)]
        row, present = _trial_row(messages, self.media_directory)
        stimuli = StimulusSet(pd.DataFrame([{'stimulus_id': trial_ids[0], **row}]))
        task = TaskContext(instruction=system, present=present, trial_memory='none',
                           response=Text(max_tokens=max_tokens, temperature=float(temperature)))
        session = stimulus_session(stimuli, ['behavior:text'], task)
        record_repetition = f'{self.repetition}.{self.call_index}'
        self.call_index += 1
        with self.context.repetition(record_repetition):
            self.context.publish('session_context', {
                name: getattr(session, name) for name in Session.context_fields
                if getattr(session, name, None) is not None})
            with observed(session, self.context):
                self.subject.participate(session)
        answers = [event for event in session.emitted if event.channel == 'behavior:text']
        if len(answers) != 1:
            raise TypeError(f'CogGym needs one behavior:text answer for each call; got {len(answers)}')
        meta = answers[0].meta if isinstance(answers[0].meta, dict) else {}
        return {'text': _written_answer(answers[0]),
                'reasoning': meta.get('reasoning'), 'token_usage': meta.get('token_usage')}


class CogGymRunner:
    """Prepare a public CogGym experiment before loading a model.

    ``protocol(reset=...)`` returns a standard CallableProtocol. ``run`` calls
    the same reference evaluator directly, for parity checks. The required
    reset callback receives a one-based repetition number; it must reset the
    provider's state/RNG without replacing the instrumented model or its hooks.

    ``trial_ids`` selects an explicit smoke subset of the canonical selection.
    It is always labelled as a subset, never a complete CogGym evaluation.

    Defaults match the pinned CLI: one repetition, 512 tokens, temperature 1.
    CogGymBenchmark instead uses the README example's five repetitions.
    Set these values explicitly when comparing direct and registered runs.

    A subject runs the protocol when it takes in at least one of the inputs
    the selected trials send (``instruction``, ``text``, ``vision:image``,
    ``vision:video``: ``required_input_channels``) and answers on
    ``behavior:text``. An input it cannot take is dropped (a partial match). A ``BrainScoreModel`` with a ``reply`` part does both.
    """

    def __init__(
        self,
        checkout: str | Path,
        *,
        model: str,
        temperature: float = 1.0,
        max_tokens: int = 512,
        repetitions: int = 1,
        trial_ids: Sequence[str] | None = None,
        experiment: str = DEFAULT_EXPERIMENT,
        modalities: Sequence[str] = ('text',),
    ) -> None:
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
        shown = {'text'}
        # Preflight the reference prompts, including media loading, before inference.
        for trial in self._trials:
            system, messages = self._cli.build_messages(trial, self._experiment)
            required = {'text'}
            for message in messages:
                content = message['content']
                if isinstance(content, list):
                    for part in content:
                        kind = part.get('type')
                        if kind not in _PART_CHANNELS:
                            raise ValueError(f'Unsupported CogGym content type: {kind}')
                        if kind == 'image':
                            _check_image(part)
                        required.add(kind)
                elif not isinstance(content, str):
                    raise TypeError('CogGym message content must be text or a list of parts')
            if not isinstance(system, str) or required - set(self.modalities):
                raise ValueError(f'Provider must support these modalities: {sorted(required)}')
            shown |= required
            self._prompt_trials.setdefault(_prompt_key(system, messages), []).append(trial.id)
        # The inputs the experiment sends: the system text is the instruction, and
        # every kind of part the selected trials show. A subject that cannot take
        # one runs with it dropped (Contract Rules 5.2, 5.5, 5.6).
        self.required_input_channels = frozenset(
            {'instruction'} | {_PART_CHANNELS[kind] for kind in shown})
        self.requested_output_channels = frozenset({'behavior:text'})

    @property
    def trial_ids(self) -> tuple[str, ...]:
        return tuple(trial.id for trial in self._trials)

    def describe(self) -> dict[str, Any]:
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
            'slider_validation': SLIDER_VALIDATION_POLICY,
        }

    def protocol(self, *, reset: Callable[[int], None]) -> CallableProtocol:
        """Build evaluation steps for Experiment without running CogGym yet.

        CallableProtocol lets tools observe the subject while CogGym keeps its
        trial loop and scoring. reset(repetition) clears provider history and
        sampling state as needed, while preserving model instrumentation.

        Each provider call is one record repetition ``<CogGym repetition>.<call>``
        in condition ``default``. CogGym question IDs are call metadata, not tool
        filters.
        """
        if not callable(reset):
            raise TypeError('Supply a reset(repetition) callback for your provider')

        def evaluate(subject, context):
            # Reject missing media support before the first provider call.
            check_channel_compatibility(subject, self)
            def register(path):
                context.artifact(
                    path, producer='CogGym evaluator via UMI adapter',
                    description='Reference trial results, summary, and explicit run coverage',
                )
            # Translate CogGym provider calls into sessions on the common channels.
            # This is where the external evaluator connects to UMI tools.
            return self._run(
                lambda repetition: _SubjectProvider(
                    subject, context, repetition, self._prompt_trials,
                ),
                context.directory / 'coggym', reset, register,
            )

        # The provider records its own sessions, so no subject method is observed.
        return CallableProtocol(
            f'coggym:{self.experiment}', evaluate, methods=(), metadata=self.describe(),
        )

    def run(
        self,
        provider: Any,
        *,
        output_dir: str | Path,
        reset: Callable[[int], None],
    ) -> dict[str, Any]:
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
        validations = []
        for repetition in range(1, self.repetitions + 1):
            reset(repetition)
            results = self._cli._run_experiment(
                self._experiment, self._trials, provider_factory(repetition),
                self.model, self.temperature, self.max_tokens,
            )
            # Keep reference parsing and scoring intact. Check separately whether
            # slider ratings really occur in the response, rather than in prose.
            validation = validate_slider_responses(self._trials, results)
            validations.append(validation)
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
                        'response_validation': validation,
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
        checked = sum(report['checked_queries'] for report in validations)
        invalid = sum(report['invalid_queries'] for report in validations)
        result['response_validation'] = {
            'policy': SLIDER_VALIDATION_POLICY,
            'status': 'invalid' if invalid else 'valid' if checked else 'not_applicable',
            'checked_queries': checked,
            'invalid_queries': invalid,
            'repetitions': validations,
        }
        path = directory / 'analysis.json'
        path.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        register(path)
        if skipped:
            raise RuntimeError(f'CogGym analysis skipped artifacts; inspect {path}')
        return result
