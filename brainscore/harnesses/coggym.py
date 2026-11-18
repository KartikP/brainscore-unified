"""Run the pinned CogGym evaluator with existing UMI experiment tools.

CogGym owns prompts, trial selection, parsing, and scientific scoring. This
adapter turns each call CogGym makes to its model into a session on the common channels and
saves CogGym's unmodified trial results.

Each call becomes one stimulus row: its text, image and video parts in
CogGym's order (``TaskContext.present``). The system text is the task's
instruction, the temperature and length are settings on the ``Text``
response, and the subject answers on ``behavior:text``. The subject builds its
own prompt from these, so a chat model can place the instruction in the user
turn where CogGym's own client sends a separate system text. Scores can
therefore differ a little from scores made with CogGym's own client.

CogGym's evaluator runs in its own environment, in a child process (``_coggym_worker.py``)
started with the Python of CogGym's folder (``_environment_folder.py``). Its pre-flight gives the
trials and prompts; one process per run serves every repetition and the analysis, and each
model call comes back here as one request and one reply. The subject, the sessions, the
recording and ``reset`` stay in this process; nothing is installed into it.

What crosses the pipe, as JSON (UTF-8): the commands and their settings; the prepared trial
ids, system texts and messages (pictures as CogGym's base64 text, videos as paths); each model
call (system text, messages, model, temperature, length) and its reply (text, reasoning, token
usage); the trial rows, summary, answer states and analysis. Strings, integers of any size,
finite floats, None and string-keyed objects in order cross unchanged. A tuple arrives as a
list and a non-string key as a string, as in the trial file CogGym writes in one process; a
value JSON cannot hold (bytes, a NumPy scalar, a NaN in a reply) stops the run with a clear
error. CogGym scores only the reply's text, so the values a score depends on cross exactly; the
trial rows' NaN and infinity reach this process unchanged, and its strict file writing refuses
them, as in one process. A failure of the subject arrives in the evaluator as an exception of
the same type name and message, not of the same class. The evaluator's module state lives for
one run (all its repetitions and its analysis); the pre-flight and each run start a new process.
"""
from __future__ import annotations

import base64
from collections import Counter
import hashlib
import json
import math
import mimetypes
from pathlib import Path
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd

from brainscore.experiments import CallableProtocol
from brainscore.harnesses._child_process import ChildFailed, ChildProcess
from brainscore.harnesses._environment_folder import (Library, _command_environment, environment_python,
                                                      ready_folder)
from brainscore.harnesses.adapter import run_request
from brainscore.harnesses.coggym_validation import SLIDER_VALIDATION_POLICY
from brainscore_core.compatibility import ExecutionNotFeasibleError
from brainscore_core.contract import TaskContext
from brainscore_core.responses import Text
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet


DEFAULT_EXPERIMENT = 'Hu2023Fine/exp1'
# CogGym at the pinned commit, its pinned packages and the files of the checkout the worker reads.
COGGYM = Library(
    name='CogGym', repository='https://github.com/lance-ying/coggym.git',
    commit='a1cd9df1118fec80eba7463de237d1497d77e041',
    requirements=Path(__file__).with_name('coggym_requirements.txt'),
    required_files=('evaluation/__init__.py', 'evaluation/providers.py', 'evaluation/cli.py',
                    'evaluation/analyze.py', 'evaluation/public_manifest.json',
                    'evaluation/trial_selection_map.json', 'EML'))
# The input channel and the stimulus column of each kind of message part.
_PART_CHANNELS = {'text': 'text', 'image': 'vision:image', 'video': 'vision:video'}
_PART_COLUMNS = {'text': 'text', 'image': 'image_path', 'video': 'video_path'}
# The worker that runs CogGym's evaluator in its own environment.
_WORKER = Path(__file__).with_name('_coggym_worker.py')
# Seconds the evaluator may work without a word while the parent waits for it (between two
# model calls, or in its analysis): a stated bound, not a measured budget.
_EVALUATOR_SILENCE_S = 900.0


def _check_checkout(checkout):
    def git(*arguments):
        # The command environment of CogGym's folder: no GIT_DIR or other setting of this process.
        return subprocess.check_output(
            ['git', '-C', str(checkout), *arguments], text=True,
            stderr=subprocess.STDOUT, env=_command_environment(),
        ).strip()

    try:
        if git('rev-parse', 'HEAD') != COGGYM.commit:
            raise ValueError(f'CogGym requires revision {COGGYM.commit}')
        if git('status', '--porcelain', '--untracked-files=no', '--',
               'evaluation', 'EML'):
            raise ValueError('CogGym evaluation code or experiment data has tracked edits')
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError('Provide a local Git checkout of the pinned CogGym revision') from error


def _evaluator(folder: Path, log_directory: Path) -> ChildProcess:
    """CogGym's evaluator in a child process with the Python of the folder's own environment.

    The child gets no ``PYTHONPATH`` or ``PYTHONHOME`` of the scoring environment."""
    environment = {name: value for name, value in os.environ.items()
                   if name not in ('PYTHONPATH', 'PYTHONHOME')}
    return ChildProcess([environment_python(folder), _WORKER, folder / 'checkout'], env=environment,
                        log_directory=log_directory, idle_timeout_s=_EVALUATOR_SILENCE_S)


def _refuse_start(identifier: str, folder: Path, error: ChildFailed) -> None:
    """An evaluator that could not import a module becomes one load error with the next step."""
    if error.error_type in ('ModuleNotFoundError', 'ImportError'):
        from brainscore import LoadError
        raise LoadError(
            f"{identifier}: CogGym's evaluator could not start in {folder}: {error}. The folder's "
            f'environment lacks a module that CogGym imports. Next step: remove {folder} and load '
            f'the benchmark again with installs allowed (BS_INSTALL_DEPENDENCIES unset), so '
            f'Brain-Score makes it anew; if the error comes back, the pinned list {COGGYM.requirements} '
            f'lacks the package.') from error


def _answer(complete):
    """The parent's side of a model call: the reply of ``complete(call)``, or the failure
    CogGym records."""
    def answer(call):
        try:
            reply = complete(call)
        except Exception as error:
            # CogGym stores a failed trial with this type and message, as in one process.
            return {'error': [type(error).__name__, str(error)]}
        for key, value in (reply or {}).items():
            try:
                json.dumps(value, allow_nan=False)
            except (TypeError, ValueError) as error:
                raise TypeError(
                    f"The answer's {key!r} cannot go to CogGym's evaluator, which runs in its own "
                    f'process and stores it in its trial file as JSON: {error}') from None
        return {'reply': reply}
    return answer


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


def _written_answer(answer) -> tuple[str | None, str, str]:
    """The written answer of one call (the collected ``behavior:text`` table), its state and
    core's reason (Contract Rule 6.2). A missing answer has no text."""
    values = np.asarray(answer.values, dtype=object).reshape(-1)
    if values.size != 1:
        raise ValueError(f'A CogGym call needs one written answer, not {values.size}')
    state = str(np.asarray(answer['response_state'].values).reshape(-1)[0])
    reason = str(np.asarray(answer['response_reason'].values).reshape(-1)[0])
    if state == 'missing':
        return None, state, reason
    if not isinstance(values[0], str):
        raise TypeError('CogGym needs the answer on behavior:text as text (str)')
    return values[0], state, reason


def _core_states(validation, results, answers):
    """The check's report with core's states: ``answers`` holds core's stored row of each trial
    whose written answer core stored as missing or invalid. Each answer of a missing trial
    (CogGym's entry of a call with no reply) keeps core's reason; each answer of an invalid
    trial is counted ``invalid`` for core's reason."""
    if not answers:
        return validation
    issues = []
    for issue in validation['issues']:
        row = answers.get(issue['trial_id'])
        if row is None:
            issues.append(issue)
        elif row['response_state'] == 'missing':
            issues.append({**issue, 'reason': row['response_reason']})
    issues += [{'trial_id': result['trial_id'], 'query': tag, 'reason': row['response_reason'],
                'response_state': 'invalid'}
               for result in results
               if (row := answers.get(result.get('trial_id'))) and row['response_state'] == 'invalid'
               for tag in result.get('parsed') or {}]
    return {**validation, 'issues': issues}


class _SubjectAdapter:
    """Translate CogGym's model calls into sessions on the common channels.

    CogGym's model client in the evaluator (its "provider") sends each call here with the
    id of the trial CogGym built it for; this adapter answers through the subject. Each call
    is one ``run_request``
    session: one stimulus row (the
    message parts in order), a ``TaskContext`` (the system text as the
    instruction, a ``Text`` response with the call's temperature and length,
    the order of the parts, ``trial_memory='none'``) and the output
    ``behavior:text``. The subject answers through ``participate`` only
    (Contract Rule 1.1). Each session is one repetition of the run record
    (``<CogGym repetition>.<call>``), so tools record the request and the
    answer between that session's own boundaries, and ``replay_sessions``
    rebuilds one session per call. The
    subject is not reset between calls; the adapter's reset(repetition)
    runs only between CogGym repetitions.

    CogGym stores ``reasoning`` and ``token_usage`` with each trial and never
    scores them. They come from the extra data (``meta``) of the answer event
    when the subject gives them; otherwise they are ``None``.

    A trial with no written answer (core stores it as ``missing``) reaches CogGym as a
    failed call, CogGym's own record of a call with no reply; the run goes on and the
    trial's answers are counted ``missing`` with core's reason. An answer core stores as
    ``invalid`` reaches CogGym as its text, and its answers are counted ``invalid`` with
    core's reason. The adapter keeps core's stored row of each such trial (``answers``, by
    CogGym's trial id; a retry of the trial replaces it) for the answer states.

    A trial the subject cannot serve (``ExecutionNotFeasibleError``, for example a
    trial with more pictures than the subject takes) is kept in ``refusal``; no
    later trial reaches the subject, and the run ends with that error.
    """

    def __init__(self, subject, context, repetition):
        self.subject = subject
        self.context = context
        self.repetition = repetition
        self.call_index = 0
        self.media_directory = context.directory / 'coggym-media'
        self.refusal = None
        # Core's stored row of each trial whose written answer is missing or invalid.
        self.answers = {}

    def complete(self, call):
        """Answer one model call of CogGym (its system text, messages and settings)."""
        system, messages = call['system'], call['messages']
        temperature, max_tokens = call['temperature'], call['max_tokens']
        if self.refusal is not None:
            raise self.refusal      # an earlier trial was refused: no further trial runs
        # CogGym's own id of the trial (several trials can share one prompt). A retry of the
        # trial replaces the state of its earlier call: CogGym keeps the last call.
        trial_id = call['trial_id']
        if trial_id is None:
            raise ValueError('CogGym made a model call outside a trial')
        self.answers.pop(trial_id, None)
        row, present = _trial_row(messages, self.media_directory)
        stimuli = StimulusSet(pd.DataFrame([{'stimulus_id': trial_id, **row}]))
        task = TaskContext(instruction=system, present=present, trial_memory='none',
                           response=Text(max_tokens=max_tokens, temperature=float(temperature)))
        record_repetition = f'{self.repetition}.{self.call_index}'
        self.call_index += 1
        try:
            session = run_request(self.subject, self.context, record_repetition,
                                  stimuli, ['behavior:text'], task)
        except ExecutionNotFeasibleError as error:
            # CogGym keeps a failed trial and goes on; the run stops with this error instead.
            self.refusal = ExecutionNotFeasibleError(
                f'CogGym trial {trial_id}: the subject cannot serve it: {error}',
                channels=error.channels, task_type=error.task_type)
            raise self.refusal from error
        # Collect first: it stores the missing row of a trial the subject did not answer.
        collected = session.collect('behavior:text')
        answers = [event for event in session.emitted if event.channel == 'behavior:text']
        if len(answers) != 1:
            raise TypeError(f'CogGym needs one behavior:text answer for each call; got {len(answers)}')
        text, state, reason = _written_answer(collected)
        if state != 'valid':
            self.answers[trial_id] = {'stimulus_id': trial_id, 'value': text,
                                      'response_state': state, 'response_reason': reason}
        if state == 'missing':
            # CogGym's own record of a call with no reply: an entry with the error and no
            # scores, which its fit leaves out for every question type. The message is fixed:
            # CogGym calls again on some words (rate limits).
            raise ValueError('no answer: the subject wrote no answer for this trial')
        # An invalid answer: CogGym parses the raw text; the count says invalid.
        meta = answers[0].meta if isinstance(answers[0].meta, dict) else {}
        return {'text': text, 'reasoning': meta.get('reasoning'), 'token_usage': meta.get('token_usage')}


class CogGymRunner:
    """Prepare a public CogGym experiment before loading a model.

    ``protocol(reset=...)`` returns a standard CallableProtocol. ``run`` calls
    the same reference evaluator directly, for parity checks. The required
    reset callback receives a one-based repetition number; it must reset the
    adapter's state/RNG without replacing the instrumented model or its hooks.

    ``trial_ids`` selects an explicit smoke subset of the canonical selection.
    It is always labelled as a subset, never a complete CogGym evaluation.

    Defaults match the pinned CLI: one repetition, 512 tokens, temperature 1.
    CogGymBenchmark instead uses the README example's five repetitions.
    Set these values explicitly when comparing direct and registered runs.

    A subject runs the protocol when it takes in at least one of the inputs
    the selected trials send (``instruction``, ``text``, ``vision:image``,
    ``vision:video``: ``required_input_channels``) and answers on
    ``behavior:text``. An input it cannot take is dropped (a partial match). A
    ``BrainScoreModel`` with a ``reply`` part does both. The protocol checks no
    compatibility itself: ``score()`` does it for the benchmark, and
    ``check_compatibility(subject, runner)`` says before an ``Experiment``
    whether a subject fits.
    """

    def __init__(
        self,
        folder: str | Path | None = None,
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
            raise ValueError('Supply the model identifier')
        if not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature < 0:
            raise ValueError('temperature must be finite and nonnegative')
        if type(max_tokens) is not int or max_tokens < 1:
            raise ValueError('max_tokens must be a positive integer')
        if type(repetitions) is not int or repetitions < 1:
            raise ValueError('repetitions must be a positive integer')
        self.model, self.temperature = model, temperature
        self.max_tokens, self.repetitions = max_tokens, repetitions
        if not modalities or set(modalities) - {'text', 'image', 'video'}:
            raise ValueError('Declare supported modalities: text, image, video')
        self.experiment = experiment
        self.modalities = tuple(sorted(set(modalities)))
        # CogGym's folder (made at the first use when installs are allowed), then its checkout.
        self.folder = ready_folder(COGGYM, f'CogGym.{experiment.replace("/", ".")}', folder).resolve()
        self.checkout = self.folder / 'checkout'
        _check_checkout(self.checkout)
        manifest = json.loads((self.checkout / 'evaluation' / 'public_manifest.json').read_text())
        public = {entry['path'] for entry in manifest['experiments']}
        if experiment not in public:
            raise ValueError('experiment must be in the pinned public CogGym manifest')
        requested = None
        if trial_ids is not None:
            requested = list(trial_ids)
            if not requested or len(set(requested)) != len(requested):
                raise ValueError('trial_ids must be nonempty and distinct')
        # CogGym's trial selection and prompts, from its evaluator in its own environment.
        with tempfile.TemporaryDirectory(prefix='coggym-prepare-') as logs:
            with _evaluator(self.folder, Path(logs)) as child:
                try:
                    prepared = child.ask({'do': 'prepare', 'experiment': experiment, 'trial_ids': requested})
                except ChildFailed as error:
                    _refuse_start(f'CogGym.{experiment.replace("/", ".")}', self.folder, error)
                    raise
        # The evaluator's environment, for the record of every run (the scoring environment's
        # fingerprint does not cover it).
        marker = json.loads((self.folder / 'ready.json').read_text())
        self._environment = {
            'revision': marker['revision'], 'requirements_sha256': marker['requirements_sha256'],
            'freeze_sha256': hashlib.sha256('\n'.join(marker['freeze']).encode()).hexdigest()}
        canonical = prepared['canonical']
        if requested is not None and set(requested) - set(canonical):
            raise ValueError('trial_ids must belong to the canonical CogGym selection')
        self._trials = [trial['id'] for trial in prepared['trials']]
        if not self._trials:
            raise ValueError('The reference trial selection is empty')
        self.canonical_trial_count = len(canonical)
        shown = {'text'}
        # Preflight the reference prompts, including media loading, before inference.
        for trial in prepared['trials']:
            system, messages = trial['system'], trial['messages']
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
                raise ValueError(f'The adapter must support these modalities: {sorted(required)}')
            shown |= required
        # The inputs the experiment sends: the system text is the instruction, and
        # every kind of part the selected trials show. A subject that cannot take
        # one runs with it dropped (Contract Rules 5.2, 5.5, 5.6).
        self.required_input_channels = frozenset(
            {'instruction'} | {_PART_CHANNELS[kind] for kind in shown})
        self.requested_output_channels = frozenset({'behavior:text'})

    @property
    def trial_ids(self) -> tuple[str, ...]:
        return tuple(self._trials)

    def describe(self) -> dict[str, Any]:
        return {
            'reference_revision': COGGYM.commit,
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
            'evaluator_environment': dict(self._environment),
        }

    def protocol(self, *, reset: Callable[[int], None]) -> CallableProtocol:
        """Build evaluation steps for Experiment without running CogGym yet.

        CallableProtocol lets tools observe the subject while CogGym keeps its
        trial loop and scoring. reset(repetition) clears the adapter's history and
        sampling state as needed, while preserving model instrumentation.

        Each model call is one record repetition ``<CogGym repetition>.<call>``
        in condition ``default``. CogGym question IDs are call metadata, not tool
        filters.
        """
        if not callable(reset):
            raise TypeError('Supply a reset(repetition) callback for your adapter')

        def evaluate(subject, context):
            def register(path):
                context.artifact(
                    path, producer='CogGym evaluator via UMI adapter',
                    description='Reference trial results, summary, and explicit run coverage',
                )
            # Translate CogGym's model calls into sessions on the common channels.
            # This is where the external evaluator connects to UMI tools.
            def start(repetition):
                adapter = _SubjectAdapter(subject, context, repetition)
                return adapter.complete, adapter
            return self._run(start, context.directory / 'coggym', reset, register)

        # The adapter records its own sessions, so no subject method is observed.
        return CallableProtocol(
            f'coggym:{self.experiment}', evaluate, methods=(), metadata=self.describe(),
        )

    def run(
        self,
        adapter: Any,
        *,
        output_dir: str | Path,
        reset: Callable[[int], None],
    ) -> dict[str, Any]:
        """Run CogGym directly, without UMI observation or instrumentation.

        ``adapter`` implements CogGym's "provider" interface (``complete_with_metadata``).
        """
        if not callable(getattr(adapter, 'complete_with_metadata', None)):
            raise TypeError('The adapter must implement complete_with_metadata')

        def complete(call):
            return adapter.complete_with_metadata(call['system'], call['messages'], call['model'],
                                                  call['temperature'], call['max_tokens'])
        # Only Brain-Score's session adapter states answers; any failure of this one is a failure.
        return self._run(lambda repetition: (complete, None), Path(output_dir), reset, lambda path: None)

    def _repetitions(self, child, start, directory, reset, register):
        """Each repetition: reset (in this process), CogGym's run (in the evaluator), its file.

        ``start(repetition)`` gives the function that answers CogGym's calls and Brain-Score's
        session adapter (``None`` for a direct run), the only source of answer states."""
        artifacts, validations = [], []
        for repetition in range(1, self.repetitions + 1):
            reset(repetition)
            complete, adapter = start(repetition)
            ran = child.ask({'do': 'run', 'model': self.model, 'temperature': self.temperature,
                             'max_tokens': self.max_tokens}, _answer(complete))
            # A trial the subject could not serve ends the run with the subject's own reason.
            if adapter is not None and adapter.refusal is not None:
                raise adapter.refusal
            results = ran['results']
            # Keep reference parsing and scoring intact. The state of each requested answer
            # comes from the check; an answer core stored as invalid keeps core's state.
            answers = {} if adapter is None else adapter.answers
            validation = _core_states(ran['validation'], results, answers)
            validations.append(validation)
            failed = [result for result in results if 'error' in result
                      and answers.get(result.get('trial_id'), {}).get('response_state') != 'missing']
            coverage = {
                'expected_trials': len(self._trials),
                'returned_trials': len(results),
                'error_trials': len(failed),
                'no_answer_trials': sum('error' in result for result in results) - len(failed),
                'unscorable_trials': sum(not any(
                    score.get('scorable') for score in result.get('scores', {}).values()
                ) for result in results if 'error' not in result),
                # CogGym's own count of the queries its fit can use: no row with none is its result.
                'scorable_queries': sum(bool(score.get('scorable')) for result in results
                                        for score in result.get('scores', {}).values()),
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
            # The completed model calls first (summary pending): they stay saved even if the
            # summary cannot be written (an overflow, a value that is not finite).
            path.write_text(json.dumps(artifact, indent=2, allow_nan=False) + '\n')
            artifacts.append(path)
            try:
                if 'summary_error' in ran:
                    artifact['umi'].update(summary_status='failed', summary_error=ran['summary_error'])
                else:
                    artifact['summary'] = ran['summary']
                    artifact['umi']['summary_status'] = 'complete'
                path.write_text(json.dumps(artifact, indent=2, allow_nan=False) + '\n')
            finally:
                register(path)
            if 'summary_error' in ran:
                failed = ran['summary_error']
                raise RuntimeError(f"CogGym's summary failed: {failed['type']}: {failed['message']}; "
                                   f'inspect {path}')
            if not complete:
                raise RuntimeError(f'CogGym returned incomplete or failed trials; inspect {path}')
        return artifacts, validations

    def _run(self, start, directory, reset, register):
        if not callable(reset):
            raise TypeError('Supply a reset(repetition) callback for your adapter')
        _check_checkout(self.checkout)
        directory.mkdir(parents=True, exist_ok=False)
        # One evaluator process for the whole run: CogGym's module state lives across the
        # repetitions and the analysis, as in one process. It is stopped and reaped on any exit.
        with _evaluator(self.folder, directory / 'evaluator') as child:
            child.ask({'do': 'prepare', 'experiment': self.experiment, 'trial_ids': list(self._trials)})
            artifacts, validations = self._repetitions(child, start, directory, reset, register)
            try:
                analyzed = child.ask({'do': 'analyze', 'paths': [str(path) for path in artifacts]})
            except ChildFailed as error:
                path = directory / 'analysis-error.json'
                path.write_text(json.dumps({
                    'status': 'failed', 'type': error.error_type or type(error).__name__,
                    'message': str(error), 'configuration': self.describe(),
                }, indent=2, allow_nan=False) + '\n')
                register(path)
                raise
        rows, summary, skipped = analyzed['rows'], analyzed['summary'], analyzed['skipped']
        result = {'configuration': self.describe(), 'experiments': rows,
                  'summary': summary, 'skipped': skipped,
                  'repetitions': [json.loads(path.read_text())['umi']['coverage'] for path in artifacts]}
        # The detector's findings, counted by reason; the validity of the score is the
        # benchmark's decision.
        result['response_validation'] = {
            'policy': SLIDER_VALIDATION_POLICY,
            'checked_queries': sum(report['checked_queries'] for report in validations),
            'reason_counts': dict(Counter(issue['reason'] for report in validations
                                          for issue in report['issues'])),
            'repetitions': validations,
        }
        path = directory / 'analysis.json'
        path.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        register(path)
        if skipped:
            raise RuntimeError(f'CogGym analysis skipped artifacts; inspect {path}')
        return result
