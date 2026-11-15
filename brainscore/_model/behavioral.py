"""How a BrainScoreModel answers a behavioral task.

One chooser (``_choose_path``) picks the part that answers: ``generation``
(the ``reply`` part, or the ``answer`` part), ``likelihood`` or ``readout``.
One entry (``respond``) answers the stimuli and records regions during the
same presentation. A task that states a ``response`` (Choice, Scale, Text)
and a legacy task (``task_type`` with ``label_set``) differ only in how the
answer is packaged: a legacy task keeps its one-hot or probability assembly,
which legacy behavioral benchmarks score. The packaging is in ``answers``;
the recording during the answering pass is in ``recording``.
"""

from contextlib import contextmanager, nullcontext
from dataclasses import replace
from typing import Any, Dict, Optional, Tuple

from brainscore_core.contract import TaskContext
from brainscore_core.response_formats import argmax, response_record, softmax, softmax_argmax
from brainscore_core.streaming import StreamEvent

import numpy as np

from . import answers
from .dispatch import detect_modalities, extract_for_modality, pick_modality, record
from .recording import gather_region, per_stimulus, region_modality, region_selector


# A likelihood part gives each option's summed log probability: the continuation
# log-probability format (Response Format Standard, F6; no channel yet).
LOG_PROBABILITIES = 'behavior:log_probabilities'


def _softmax_record():
    return response_record('behavior:probabilities', 'likelihood', conversion='softmax',
                           source_format=LOG_PROBABILITIES)


def _softmax_probabilities(outputs, logp, stimuli, axis):
    """Store the softmax of likelihood scores, with the scores as source evidence;
    return the probabilities a reaction time reads, their validity and the record.

    The link reads the stored float32 probabilities, as it did before the response
    format standard (finite rows stay bit for bit); a readout's link reads its
    float64 rows (``_readout_probabilities``).
    """
    probabilities, valid = softmax(logp)
    record = _softmax_record()
    outputs['behavior:probabilities'] = answers.implied_array(
        probabilities, valid, stimuli, axis, record, raw=answers.source_rows(logp))
    return np.asarray(outputs['behavior:probabilities'].values, dtype=float), valid, record


class BehavioralReadout:
    """Owns behavioral task state for a BrainScoreModel instance."""

    BEHAVIORAL_TASK_TYPES = frozenset({
        'probabilities', 'classification', 'label',
    })

    def __init__(self, owner) -> None:
        self.owner = owner
        self.task: Optional[TaskContext] = None
        self.readout_classifier: Any = None
        # The TaskContext object the current configuration came from, so a
        # repeated send of the same task keeps the fitted readout.
        self.configured_from: Optional[TaskContext] = None
        self._configured_as: Optional[set] = None  # fingerprints of the fitted configuration
        self._stale = False
        # With memory='trial_history': the episode's earlier trials as (prompt, reply),
        # kept from begin until end (BrainScoreModel._begin_episode empties it).
        self.history: list = []
        # The part that answers ('generation', 'likelihood' or 'readout') and, for
        # a Choice that asks for them, where its implied probabilities come from.
        self.response_path: Optional[str] = None
        self.probability_source: Optional[str] = None
        # The last configuration that completed (mark_configured), so the same
        # task can reuse it after a session gave the caller's task back.
        # Dropped when the model changes or is reset.
        self._remembered: Optional[Dict[str, Any]] = None
        # The part that owns each answer output several parts can give (rule 4.11),
        # stated at registration and checked by ``answer_owners``.
        self.owners: Dict[str, str] = {}
        self._changes = 0   # counts invalidations, so a session can see one happen
        self._resets = 0    # counts resets, so a session does not undo one

    _STATE = ('task', 'readout_classifier',
              'configured_from', '_configured_as', '_stale', 'history',
              'response_path', 'probability_source')

    @classmethod
    def requires_behavioral_readout(cls, task_type: str) -> bool:
        return task_type in cls.BEHAVIORAL_TASK_TYPES

    @staticmethod
    def _fitting_content(fitting) -> Any:
        """The fitting table's columns, cell values and stimulus paths.

        An in-place edit of any of them gives a new value, so the readout
        fits again. A table that pandas cannot hash (e.g. a list-valued
        column) falls back to the earlier check, its stimulus ids and
        labels, so it never fits more often than before. Not a table: the
        readout cannot fit on it, so its content does not matter.
        """
        if getattr(fitting, 'columns', None) is None:
            return None
        try:
            import pandas as pd
            paths = getattr(fitting, 'stimulus_paths', None)
            paths = (tuple(sorted((str(key), str(value)) for key, value in paths.items()))
                     if isinstance(paths, dict) else None)
            return (tuple(str(column) for column in fitting.columns),
                    tuple(pd.util.hash_pandas_object(fitting, index=True)), paths)
        except Exception:
            return ('ids-and-labels',) + BehavioralReadout._ids_and_labels(fitting)

    @staticmethod
    def _ids_and_labels(fitting) -> tuple:
        """The fitting rows and labels: the check used before table contents were hashed."""
        rows = labels = None
        try:
            columns = fitting.columns
            if 'stimulus_id' in columns:
                rows = tuple(fitting['stimulus_id'])
            for column in ('image_label', 'label'):
                if column in columns:
                    labels = tuple(fitting[column])
                    break
        except (TypeError, KeyError, AttributeError):
            rows = labels = None
        return rows, labels

    @classmethod
    def _fingerprint(cls, task: TaskContext) -> tuple:
        """Every TaskContext field the configuration depends on, and the fitting table's content.

        ``metadata`` is left out: it carries the scored stimuli and the
        benchmark's settings, which the configuration does not read.
        """
        fitting = task.fitting_stimuli
        label_set = task.label_set
        present = task.present
        return (task.task_type, id(fitting), cls._fitting_content(fitting),
                task.instruction,
                tuple(label_set) if label_set is not None else None,
                task.readout_fit,
                repr(task.response),
                tuple(present) if present is not None else None)

    def is_configured_for(self, task: TaskContext) -> bool:
        """Whether ``task`` is the task already configured, unchanged since.

        True for the object the configuration came from and for the context
        this component holds (the same task after input projection), as long
        as every field in ``_fingerprint`` is the one fitted. TaskContext is
        mutable: a caller that swaps or edits the fitting table, relabels it
        or changes a setting and calls start_task again gets a new fit.
        """
        if self._stale or self.task is None or not self._configured_as:
            return False
        if task is not self.configured_from and task is not self.task:
            return False
        return self._fingerprint(task) in self._configured_as

    def mark_configured(self, task: Optional[TaskContext]) -> None:
        """Remember the configuration just fitted, as the caller's context and as the one held."""
        self.configured_from = task
        self._configured_as = None
        if task is not None:
            self._configured_as = {self._fingerprint(task)}
            if self.task is not None and self.task is not task:
                self._configured_as.add(self._fingerprint(self.task))
        self._stale = False
        # Only a configuration that completed is remembered for recall.
        self._remembered = (dict(self._snapshot(), history=[])   # memory spans one session
                            if task is not None else None)

    def abandon(self) -> None:
        """A start_task failed part way: no task counts as configured, so the next one fits."""
        self.configured_from = None
        self._configured_as = None

    def invalidate(self) -> None:
        """The model changed (an intervention was applied or undone): fit again next time.

        A configuration with ``readout_fit='intact'`` stays: its readout was
        fitted with every intervention switched off, so a change leaves it as it is.
        """
        if not _fitted_intact(self.task):
            self._stale = True
        if self._remembered is not None and not _fitted_intact(self._remembered['task']):
            self._remembered = None
        self._changes += 1

    def _snapshot(self) -> Dict[str, Any]:
        return {name: getattr(self, name) for name in self._STATE}

    def _restore(self, state: Dict[str, Any]) -> None:
        for name, value in state.items():
            setattr(self, name, value)

    @contextmanager
    def session_scope(self):
        """Give back the task state this model had before a session (rules 4.5, 8.3).

        A session may configure its own task; when the session ends, on
        success or on error, the caller's task, readout and history return.
        A reset during the session stands: no task comes back. The memo for
        ``recall`` is set only by ``mark_configured``, never here.
        """
        prior, changes, resets = self._snapshot(), self._changes, self._resets
        try:
            yield
        finally:
            if self._resets != resets:
                self._clear_task()
            else:
                self._restore(prior)
                if self._changes != changes and not _fitted_intact(self.task):
                    # The model changed during the session: the caller's readout
                    # was fitted on the model as it was before.
                    self._stale = True

    def recall(self, task: TaskContext) -> bool:
        """Reinstate the configuration a past session fitted for ``task``, if still valid."""
        if self._remembered is None:
            return False
        current = self._snapshot()
        self._restore(self._remembered)
        if self.is_configured_for(task):
            return True
        self._restore(current)
        return False

    def start_task(self, task_or_type, fitting_stimuli=None,
                   **kwargs) -> None:
        del kwargs
        if isinstance(task_or_type, TaskContext):
            task = task_or_type
        else:
            task = TaskContext(
                task_type=task_or_type,
                fitting_stimuli=fitting_stimuli,
            )
        self.task = task

        if task.readout_fit not in ('current', 'intact'):
            raise ValueError(f"readout_fit must be 'current' or 'intact', not {task.readout_fit!r}.")
        if task.response is None and task.task_type != 'passive' \
                and not self.requires_behavioral_readout(task.task_type):
            raise ValueError(
                f"Unknown task_type {task.task_type!r}. BrainScoreModel accepts "
                f"'passive' or one of {sorted(self.BEHAVIORAL_TASK_TYPES)}; "
                f"answer and readout models both use 'probabilities'."
            )
        if task.response is None and task.task_type == 'passive':
            self.response_path = self.probability_source = None
            self.readout_classifier = None
            return
        # The setup is keyed by the task alone: each format the task can give is
        # selected on its own (a single format never crosses two parts), so every
        # refusal comes before the fit, and the readout fits once when it serves any.
        # A probability format that no part serves is skipped here: a session that
        # requests it is refused by its own selection, before any compute.
        from brainscore_core.compatibility import ExecutionNotFeasibleError
        from brainscore_core.responses import response_channels
        picks = []
        for output in (response_channels(task.response) or ('behavior',)):
            try:
                picks.append(self._choose_path(task, [output]))
            except ExecutionNotFeasibleError:
                if output not in self._PROBABILITY_SIDE:
                    raise
        self.response_path = picks[0][0]
        self.probability_source = next((source for _, source in picks if source), None)
        self.readout_classifier = None
        if any('readout' in pick for pick in picks):
            self.fit_behavioral_readout(task.fitting_stimuli)

    def _clear_task(self) -> None:
        self.history = []
        self.response_path = self.probability_source = None
        self.task = None
        self.readout_classifier = None
        self.configured_from = None
        self._configured_as = None
        self._stale = False

    def reset(self) -> None:
        self._clear_task()
        self._remembered = None
        self._resets += 1

    def fit_behavioral_readout(self, fitting_stimuli) -> None:
        """Extract readout-layer features and fit a probabilities classifier."""
        from brainscore._model.behavior import ProbabilitiesClassifier

        owner = self.owner
        task_type = (
            self.task.task_type
            if self.task is not None else None
        )
        if owner._behavioral_readout_layer is None:
            raise ValueError(
                f"Subject '{owner.identifier}' was asked to perform a behavioral "
                f"task (task_type={task_type!r}) but no "
                f"behavioral_readout_layer is set. Register the model with "
                f"BrainScoreModel(..., behavioral_readout_layer='<layer name>')."
            )

        labels = self.extract_labels(fitting_stimuli)
        intact = (self.task is not None and self.task.readout_fit == 'intact')
        with (owner._interventions.intact() if intact else nullcontext()):
            features = self.extract_behavioral_features(fitting_stimuli)

        from brainscore_core.responses import Scale
        response = self.task.response if self.task is not None else None
        if isinstance(response, Scale):
            from brainscore._model.behavior import RatingRegressor
            self.readout_classifier = RatingRegressor().fit(features, labels)
        else:
            self.readout_classifier = ProbabilitiesClassifier()
            self.readout_classifier.fit(features, labels)

    @staticmethod
    def extract_labels(stimuli):
        for col in ('image_label', 'label'):
            if col in stimuli.columns:
                return list(stimuli[col].values)
        raise ValueError(
            f"Cannot find labels in fitting_stimuli. Expected one of "
            f"'image_label' or 'label' columns; got: {list(stimuli.columns)}."
        )

    def extract_behavioral_features(self, stimuli):
        """Run the model once at the behavioral readout layer."""
        owner, recorder = self.owner, self.owner._recorder
        saved, saved_classifier = recorder.snapshot(), self.readout_classifier
        layer = owner._behavioral_readout_layer
        # Record the readout layer alone; the time bins stay as configured.
        recorder.recording_layer, recorder.recording_layers = layer, [layer] if layer else []
        recorder.recording_regions, recorder.is_multi_region, recorder.composite_recording = [], False, False
        self.readout_classifier = None
        try:
            features = record(owner, stimuli)
        finally:
            recorder.restore(saved)
            self.readout_classifier = saved_classifier

        if 'presentation' in features.dims and 'neuroid' in features.dims:
            features = features.transpose('presentation', 'neuroid')
        return features

    def predict_probabilities(self, stimuli):
        """Return a BehavioralAssembly of per-label probabilities."""
        features = self.extract_behavioral_features(stimuli)
        return answers.with_record(self.readout_classifier.predict_proba(features),
                                   response_record('behavior', 'readout'))

    def _select(self, output, can, task) -> Optional[str]:
        """The part that answers bare ``behavior`` (a legacy task), from the parts that can serve it.

        The owner stated at registration answers when it can serve the task
        (``answer_owners``; with one wired part, that part). When the task form
        excludes the owner, the one other part that can serve it answers: a
        legacy task without an instruction, as the facade sends, goes to the
        readout (design note ANSWER_ownership, section 5). Only bare
        ``behavior`` keeps this fallback; a format channel is answered by its
        owner (``_owner_serving``). When two or more parts that do not own the
        output can serve it, the task is refused before any compute. ``None``:
        no part can.
        """
        from brainscore_core.compatibility import ExecutionNotFeasibleError
        formats = _formats_by_part(self.owner)
        eligible = sorted(p for p, given in formats.items() if can.get(p) and output in given)
        owner = self.owners.get(output)
        if owner in eligible:
            return owner
        if len(eligible) <= 1:
            only_part_for_this_task_form = eligible[0] if eligible else None
            return only_part_for_this_task_form
        raise ExecutionNotFeasibleError(
            f"Subject '{self.owner.identifier}': {output!r} is owned by {owner!r}, which cannot "
            f"serve this task, and the parts {eligible} that can do not own it. State an owner "
            f"that serves the task, or build one model per part.", task_type=task.task_type)

    def _owner_serving(self, output, can, task, settings) -> Optional[str]:
        """The owner of the answer format ``output``, when it can serve ``task`` (rule 4.11).

        The owner is the stated part, or the one wired part that gives the format.
        When the owner cannot serve the task form, the task is refused before any
        compute: no other part answers a format it does not own (only bare
        ``behavior`` keeps that fallback, ``_select``). ``None``: no part gives it.
        """
        from brainscore_core.compatibility import ExecutionNotFeasibleError
        formats = _formats_by_part(self.owner)
        givers = sorted(p for p, given in formats.items() if output in given)
        owner = self.owners.get(output) or (givers[0] if len(givers) == 1 else None)
        if owner is None and len(givers) > 1:
            raise ExecutionNotFeasibleError(
                f"Subject '{self.owner.identifier}': the parts {givers} give {output!r} and no "
                f"owner is stated.", task_type=task.task_type)
        if owner is None or not (can.get(owner) or any(can.get(p) for p in givers)):
            return None          # no part can serve it: the caller names what each needs
        if not can.get(owner):
            raise ExecutionNotFeasibleError(
                f"Subject '{self.owner.identifier}': {output!r} is owned by {owner!r}, which "
                f"cannot serve this task"
                + (f" (task settings: {', '.join(settings)})" if settings else '')
                + '; no other part answers it.', task_type=task.task_type)
        return owner

    # The formats of a response task on each side of a crossed likelihood/readout pair.
    _CHOICE_SIDE = frozenset({'behavior', 'behavior:choice'})
    _PROBABILITY_SIDE = frozenset({'behavior:probabilities', 'behavior:rt'})
    _ANSWER_FORMATS = (('Choice', 'behavior:choice'), ('Scale', 'behavior:rating'),
                       ('Text', 'behavior:text'))

    def _choose_path(self, task, formats=()) -> Tuple[str, Optional[str]]:
        """The part that answers ``task`` on the requested ``formats`` and the probability source.

        A task with a response is answered by the owner of its answer format
        (the raw scores of a raw-scores Choice, else the response kind's own
        format: choice, rating or text) and, when ``formats`` request
        probabilities or a reaction time (or the response asks for raw
        scores), its implied probabilities by the
        owner of ``behavior:probabilities`` (rule 4.11, ``answer_owners``); an
        owner that cannot serve the task's settings (``can`` below) is refused,
        no other part stands in. The choice that accompanies raw scores comes
        from the owner of the probabilities. When the choice and the
        probabilities have two owners, a likelihood and a readout, one pass
        cannot give both: formats on one side are answered by that side's
        owner, formats on both sides are refused (point j). A legacy task
        (``task_type`` with ``label_set``) answers on ``behavior``, through
        ``answer=`` (with an instruction and labels) or the readout (with
        fitting stimuli).

        A task that no part of the model can answer raises
        ``ExecutionNotFeasibleError`` before any compute. Sessions refuse with
        this choice; the refusal is an error of the run.
        """
        from brainscore_core.compatibility import ExecutionNotFeasibleError
        from brainscore_core.responses import Choice, Scale
        owner, response = self.owner, task.response
        readout_wired = (owner._behavioral_readout_layer is not None
                         and bool(owner._extractors))
        if response is None:
            can = {'generation': bool(owner._answer is not None and task.instruction and task.label_set),
                   'likelihood': False,
                   'readout': readout_wired and task.fitting_stimuli is not None}
            needs = {'generation': 'answer=, TaskContext.instruction and TaskContext.label_set',
                     'likelihood': 'a TaskContext(response=Choice(...)); a legacy task has no likelihood path',
                     'readout': 'TaskContext.fitting_stimuli (and behavioral_readout_layer=)'}
            path = self._select('behavior', can, task)
            if path is None:
                raise ExecutionNotFeasibleError(
                    f"Subject '{owner.identifier}' cannot run the behavioral task "
                    f"task_type={task.task_type!r}. Paths tried: "
                    + '; '.join(f'{p} needs {need}' for p, need in needs.items()),
                    task_type=task.task_type)
            return path, None
        readout_wired = readout_wired and task.fitting_stimuli is not None
        reported_confidence = isinstance(response, Choice) and response.confidence is not None
        raw_scores = isinstance(response, Choice) and response.raw_scores
        if raw_scores and owner._likelihood is None:
            # Raw scores come only from a likelihood part: no softmax of a readout, no guess.
            raise ExecutionNotFeasibleError(
                f"Subject '{owner.identifier}' cannot give raw scores of the options: "
                'they come only from a likelihood= part.', task_type=task.task_type)
        per_row = isinstance(response, Choice) and isinstance(response.labels, str)
        raw_text = isinstance(response, Choice) and response.raw_text
        can = {
            'generation': not raw_scores and (owner._reply is not None or (
                owner._answer is not None and isinstance(response, Choice)
                and not reported_confidence and not raw_text)),   # answer= takes no settings
            'likelihood': (owner._likelihood is not None and isinstance(response, Choice)
                           and not reported_confidence),
            'readout': (readout_wired and isinstance(response, (Choice, Scale))
                        and not reported_confidence and not raw_scores and not per_row),
        }
        # The answer format: the raw scores of a raw-scores Choice (they travel on
        # behavior:probabilities), else the response's first format. Its owner answers;
        # a task form the owner cannot serve is refused (no other part stands in).
        settings = [name for name, on in (('raw_text', raw_text), ('raw_scores', raw_scores),
                                          ('confidence', reported_confidence),
                                          ('labels per row', per_row)) if on]
        answer_format = ('behavior:probabilities' if raw_scores   # decision: raw scores travel there
                         else dict(self._ANSWER_FORMATS)[type(response).__name__])
        path = self._owner_serving(answer_format, can, task, settings)
        if path is None:
            needs = {'generation': 'reply= (or answer= for a plain Choice)',
                     'likelihood': 'likelihood= and a Choice without reported confidence',
                     'readout': 'behavioral_readout_layer=, an extractor and fitting_stimuli'}
            raise ExecutionNotFeasibleError(
                f"Subject '{owner.identifier}' cannot give a {type(response).__name__} answer"
                + (' with reported confidence (needs reply=)' if reported_confidence else '')
                + '. Paths tried: ' + '; '.join(f'{p} needs {need}' for p, need in needs.items()),
                task_type=task.task_type)
        source = None
        requested = set(formats)
        wanted = requested & self._PROBABILITY_SIDE
        # The requested formats decide the passes: the probability owner runs (first,
        # for a reply, so a recording comes from its pass) only when the session
        # requests probabilities or a reaction time. Raw scores keep their pass.
        if isinstance(response, Choice) and (wanted or raw_scores):
            source = self._owner_serving('behavior:probabilities', {
                'likelihood': owner._likelihood is not None,
                'readout': readout_wired and not per_row}, task, settings)
            if source is None:
                raise ExecutionNotFeasibleError(
                    f"Subject '{owner.identifier}' cannot give implied probabilities: "
                    'it needs likelihood= or a readout (behavioral_readout_layer=, '
                    'fitting_stimuli).', task_type=task.task_type)
        if {path, source} != {'likelihood', 'readout'}:
            return path, source
        if requested & self._CHOICE_SIDE and wanted:
            raise ExecutionNotFeasibleError(
                f"Subject '{owner.identifier}' cannot give choices from its {path} part "
                f"and probabilities from its {source} part in one session; request one format "
                "in the session, or state one owner for both outputs.",
                task_type=task.task_type)
        return (source, source) if wanted else (path, None)

    def respond(self, stimuli, channels, regions=()):
        """Answer a response task; return ``({channel: assembly}, {region: NeuroidAssembly})``.

        Each stimulus is presented once on the answer's path, and regions are
        recorded during that presentation: the prompt pass of a reply or of
        likelihood scoring, or the readout's own extraction. Implied
        probabilities next to a reply are scored first, from the same prompt,
        so the recording is still the prompt pass.
        """
        from brainscore_core.responses import Choice, Scale, Text, labels_of, trial_prompt
        owner, task = self.owner, self.task
        response, path, probability_source = task.response, self.response_path, self.probability_source
        if response is not None:
            path, probability_source = self._choose_path(task, channels)
        layers = {region: list(region_selector(owner, region).layer_paths) for region in regions}
        if response is None:
            behavior, neural = self._answer_legacy_task(stimuli, layers)
            return ({channel: behavior for channel in channels}, neural)
        labels = labels_of(response)
        # Every trial's prompt is built first, so a bad layout fails before any compute.
        prompts = [trial_prompt(task, row) for _, row in stimuli.iterrows()]
        neural, outputs, extras = {}, {}, []
        rt_source = None   # (float64 probabilities, row validity, record) for a reaction time
        # The options of each row: the task's labels, or the row's own (labels names a column).
        # The choice axis is then the labels, or the position of each option.
        row_labels, axis = None, labels
        if isinstance(response, Choice):
            row_labels = [list(prompt.response.labels) for prompt in prompts]
            if labels is None:
                counts = {len(options) for options in row_labels}
                if len(counts) > 1:
                    raise ValueError(f'Every row must give the same number of options in column '
                                     f'{response.labels!r}; got {sorted(counts)}')
                axis = list(range(counts.pop())) if counts else []

        if path == 'readout':
            values, neural = self._read_out_and_record(stimuli, layers, raw=True)
            if isinstance(response, Scale):
                outputs['behavior:rating'] = answers.numbers(
                    values, stimuli, response_record('behavior:rating', 'readout'))
            else:
                # The readout's own column order, as before: ties go to its first column.
                source = np.asarray(values.values)
                indices, chose = argmax(source)
                options = values['choice'].values
                outputs['behavior:choice'] = answers.one_hot(
                    [str(options[i]) if ok else None for i, ok in zip(indices, chose)],
                    list(chose), stimuli, row_labels, axis, response_record(
                        'behavior:choice', 'readout', conversion='argmax',
                        source_format='behavior:probabilities',
                        parameters={'tie': 'first', 'option_order': 'readout'}),
                    raw=answers.source_rows(source),
                    errors=answers.conversion_errors(chose))
                if probability_source == 'readout':
                    rt_source = self._readout_probabilities(
                        outputs, answers.in_order(values, labels), stimuli)
        else:
            score_first = path == 'generation' and probability_source == 'likelihood'
            remember = owner._memory == 'trial_history'
            if remember and owner._reply is None and owner._likelihood is None:
                raise ValueError("memory='trial_history' needs reply= or likelihood=; answer= cannot see "
                                 'earlier trials.')

            def with_history(prompt):
                return replace(prompt, history=tuple(self.history)) if remember else prompt

            if path == 'likelihood':
                def call(row, prompt):
                    scores = owner._likelihood(with_history(prompt), list(prompt.response.labels))
                    if remember:
                        self.history.append((prompt, max(scores, key=scores.get)))
                    return scores
            elif owner._reply is not None:
                def call(row, prompt):
                    shown = with_history(prompt)
                    # Scoring first keeps the recorded first pass the prompt alone.
                    scores = (owner._likelihood(shown, list(prompt.response.labels))
                              if score_first else None)
                    reply, extra = answers.reply_and_meta(owner._reply(shown))
                    if remember:
                        self.history.append((prompt, str(reply)))
                    return reply, scores, extra
            else:   # a registration's answer: it phrases and parses a Choice itself
                def call(row, prompt):
                    # The likelihood scores the prompt first when it owns the probabilities.
                    scores = (owner._likelihood(prompt, list(prompt.response.labels))
                              if score_first else None)
                    reply = owner._answer(stimulus_row=row, instruction=task.instruction,
                                          label_set=list(prompt.response.labels))
                    return reply, scores, {}
            replies, neural = per_stimulus(owner, stimuli, layers, call, prompts)
            scores = None
            if path == 'generation':
                replies, scores, extras = (list(column) for column in zip(*replies)) if replies else ([], [], [])
            if path == 'likelihood':
                logp = answers.score_rows(replies, row_labels)
                indices, chose = softmax_argmax(logp)
                chosen = [options[i] if ok else None
                          for options, i, ok in zip(row_labels, indices, chose)]
                outputs['behavior:choice'] = answers.one_hot(
                    chosen, list(chose), stimuli, row_labels, axis, response_record(
                        'behavior:choice', 'likelihood', conversion='softmax_argmax',
                        source_format=LOG_PROBABILITIES),
                    raw=answers.source_rows(logp), errors=answers.conversion_errors(chose))
                if response.raw_scores:
                    outputs['behavior:probabilities'] = answers.implied_array(
                        logp, None, stimuli, axis, response_record(
                            'behavior:probabilities', 'likelihood', category='model-derived',
                            parameters={'scale': 'log'}))
                elif probability_source == 'likelihood':
                    rt_source = _softmax_probabilities(outputs, logp, stimuli, axis)
            elif isinstance(response, Text):
                parsed = [response.parse(reply) for reply in replies]
                outputs['behavior:text'] = answers.texts(parsed, replies, stimuli)
            elif isinstance(response, Scale):
                parsed = [response.parse(reply) for reply in replies]
                outputs['behavior:rating'] = answers.numbers(
                    parsed, stimuli, response_record('behavior:rating', 'generation',
                                                     parsed_by='model'), raw=replies)
            else:
                parsed = [prompt.response.parse(str(reply)) for prompt, reply in zip(prompts, replies)]
                outputs['behavior:choice'] = answers.one_hot(
                    [label for label, _ in parsed], [label is not None for label, _ in parsed],
                    stimuli, row_labels, axis,
                    response_record('behavior:choice', 'generation', parsed_by='model'),
                    raw=replies, errors=[error for _, error in parsed])
                if response.confidence is not None:
                    outputs['behavior:confidence'] = answers.numbers(
                        [response.parse_confidence(reply) for reply in replies], stimuli,
                        response_record('behavior:confidence', 'generation', parsed_by='model'),
                        raw=replies)
                if score_first:
                    rt_source = _softmax_probabilities(
                        outputs, answers.score_rows(scores, row_labels), stimuli, axis)
                elif probability_source == 'readout':
                    rt_source = self._readout_probabilities(outputs, answers.in_order(
                        self.readout_classifier.predict_proba(
                            self.extract_behavioral_features(stimuli)), labels), stimuli)
        if isinstance(response, Choice) and response.reaction_time and rt_source is not None:
            # From the probabilities each path read before stage 1, with their validity.
            outputs['behavior:rt'] = answers.reaction_time(*rt_source, response.reaction_time,
                                                           stimuli)
        # The bare channel carries the same answer; its record names that channel.
        first = outputs[response.channels[0]]
        outputs['behavior'] = answers.with_record(first, {**first.attrs['response'],
                                                          'format': 'behavior'})
        meta = answers.trials_meta(extras)
        if meta:
            # The reply's extra data goes into the meta of its answer event, as a
            # policy's does (send_outputs merges the meta of a returned StreamEvent).
            for channel in (response.channels[0], 'behavior'):
                outputs[channel] = StreamEvent(channel, outputs[channel], None, meta)
        if prompts:
            # A multi-image trial records one step per image: name each by its part.
            image_names = [name for (kind, _), name in zip(prompts[0].parts, prompts[0].names)
                           if kind == 'image']
            neural = {region: (assembly.assign_coords(part_name=('part', image_names))
                               if 'part' in assembly.dims and assembly.sizes['part'] == len(image_names)
                               else assembly)
                      for region, assembly in neural.items()}
        stamp = self._stamp(path, 'readout extraction' if path == 'readout' else 'prompt')
        return ({c: outputs[c] for c in channels},
                {r: a.assign_attrs(recording=stamp) for r, a in neural.items()})

    @staticmethod
    def _readout_probabilities(outputs, ordered, stimuli):
        """Store a readout's probabilities (float32); return the float64 rows, their
        validity as probability rows and the record, for a reaction time."""
        values = np.asarray(ordered.values)
        outputs['behavior:probabilities'] = answers.readout_probabilities(ordered, stimuli)
        return values, argmax(values)[1], outputs['behavior:probabilities'].attrs['response']

    @contextmanager
    def suspended(self):
        """Keep the configured task, but let it answer nothing for a while.

        A session that asks only for ``neural:<region>`` must get activity, even
        when an earlier session configured a task on this model (rule 4.1).
        """
        saved = self.task
        self.task = None
        try:
            yield
        finally:
            self.task = saved

    def _answer_legacy_task(self, stimuli, layers):
        """A legacy task's answer and the regions recorded during it.

        Returns ``(BehavioralAssembly, {region: NeuroidAssembly})``. With
        ``answer``, the recording is the first forward pass of each answer (the
        prompt, before any generated token). With a readout and regions, the
        region layers are extracted in the same call as the readout layer;
        without regions, the readout reads the features it was fitted on.
        """
        if self.response_path == 'generation':
            task = self.task
            label_set = list(task.label_set)
            replies, neural = per_stimulus(
                self.owner, stimuli, layers, lambda row, _: self.owner._answer(
                    stimulus_row=row, instruction=task.instruction, label_set=label_set))
            stamp = self._stamp('answer', 'prompt')
            return (answers.legacy_one_hot(replies, label_set, stimuli),
                    {region: assembly.assign_attrs(recording=stamp) for region, assembly in neural.items()})
        if layers:
            return self._read_out_and_record(stimuli, layers)
        return self.predict_probabilities(stimuli), {}

    def _stamp(self, path, forward_pass):
        """How a recording was made, so task and passive recordings stay distinct."""
        return {'task_type': self.task.task_type, 'path': path,
                'response': type(self.task.response).__name__
                if self.task.response is not None else None,
                'forward_pass': forward_pass, 'readout_fit': self.task.readout_fit}

    def _read_out_and_record(self, stimuli, layers, raw: bool = False):
        from brainscore_core.responses import Scale
        owner = self.owner
        readout_layer = owner._behavioral_readout_layer
        modality = pick_modality(owner, detect_modalities(owner, stimuli))
        for region in layers:
            if owner._region_modality_map.get(region) and region_modality(owner, region) != modality:
                raise ValueError(
                    f'Region {region!r} reads {region_modality(owner, region)!r} input, but '
                    f'the readout presents {modality!r} input; one presentation cannot '
                    'record it.')

        def extract(call_layers):
            extracted = owner._recorder.ensure_layer_provenance(
                extract_for_modality(owner, stimuli, modality, call_layers), call_layers,
                description='Readout extraction')
            layer_of = extracted['layer'].values
            return lambda layer: extracted.isel(neuroid=layer_of == layer)

        # BRAINSCORE_REGIONS_PER_PASS: 'all' extracts the readout layer and every
        # region in one call; 'one' gives the readout and each region its own call,
        # as does 'all' when the layers have different dtypes (one call cannot keep them).
        from brainscore.model_helpers._extractors import LayerDtypesDiffer
        joint = None
        if owner._regions_per_pass == 'all':
            all_layers = list(dict.fromkeys([readout_layer, *(l for ls in layers.values() for l in ls)]))
            try:
                joint = extract(all_layers)
            except LayerDtypesDiffer:
                pass
        if joint is not None:
            part_of = {region: joint for region in layers}
            features = joint(readout_layer)
        else:
            features = extract([readout_layer])(readout_layer)
            part_of = {region: extract(list(region_layers)) for region, region_layers in layers.items()}
        if 'presentation' in features.dims and 'neuroid' in features.dims:
            features = features.transpose('presentation', 'neuroid')
        if raw and isinstance(self.task.response, Scale):
            behavior = [(float(v), None) for v in self.readout_classifier.predict(features)]
        else:
            behavior = answers.with_record(self.readout_classifier.predict_proba(features),
                                           response_record('behavior', 'readout'))
        neural = {}
        for region, region_layers in layers.items():
            per_layer = [part_of[region](layer) for layer in region_layers]
            selected = gather_region(owner, region, per_layer)
            neural[region] = selected.assign_coords(
                region=('neuroid', [region] * selected.sizes['neuroid']))
            if not raw:
                neural[region] = neural[region].assign_attrs(
                    recording=self._stamp('readout', 'readout extraction'))
        return behavior, neural


# The answer formats of each part kind (Response Format Standard 4.2; rule
# 4.11): format -> None when the part gives it natively, else the conversion
# that derives it. ``answer`` and ``reply`` are the two kinds of the part
# ``generation``. The likelihood gives log scores natively; until step 5 they
# travel on ``behavior:probabilities`` with ``raw_scores``. ``'link'``:
# ``behavior:rt`` follows the owner of ``behavior:probabilities`` and is
# never stated. ``behavior`` is the answer to a legacy task.
_PART_FORMATS = {
    'answer': {'behavior': None, 'behavior:choice': None},
    'reply': {'behavior:choice': None, 'behavior:rating': None,
              'behavior:confidence': None, 'behavior:text': None},
    'likelihood': {'behavior:probabilities': 'softmax', 'behavior:choice': 'softmax_argmax',
                   'behavior:rt': 'link'},
    'readout': {'behavior': None, 'behavior:probabilities': None, 'behavior:rating': None,
                'behavior:choice': 'argmax', 'behavior:rt': 'link'},
}


def _formats_by_part(subject) -> Dict[str, Dict[str, Optional[str]]]:
    """The answer formats of each wired part, read from ``_PART_FORMATS``.

    Per part, format -> ``None`` (native) or the conversion that derives it.
    """
    wired = {'answer': subject._answer, 'reply': subject._reply,
             'likelihood': subject._likelihood, 'readout': subject._behavioral_readout_layer}
    parts: Dict[str, Dict[str, Optional[str]]] = {}
    for kind, formats in _PART_FORMATS.items():
        if wired[kind] is not None:
            part = 'generation' if kind in ('answer', 'reply') else kind
            parts.setdefault(part, {}).update(formats)
    return parts


def _declared_formats(subject) -> set:
    """The answer formats the subject declares (rules 2.4, 4.11): each format with an owner.

    A format has an owner when a statement names it or exactly one part gives it
    natively (``answer_owners``); a derived format with no statement has none, so
    it is not declared. Bare ``behavior`` therefore needs ``answer=`` or a readout
    with an extractor. A ``'link'`` format (``behavior:rt``) is declared with
    ``behavior:probabilities``, whose owner gives it.
    """
    declared = set(subject._behavioral.owners)
    if 'behavior:probabilities' in declared:
        declared |= {f for formats in _PART_FORMATS.values() for f, how in formats.items()
                     if how == 'link'}
    return declared


_ARGUMENTS = {'generation': 'answer= or reply=', 'likelihood': 'likelihood=',
              'readout': 'behavioral_readout_layer= (with an extractor)'}


def answer_owners(subject, stated) -> Dict[str, str]:
    """Check the owner statements at construction; return the owner part of each format (rule 4.11).

    ``stated`` is ``capability_config['behavior-sessions']``: format ->
    a part (``'generation'``, ``'likelihood'``, ``'readout'``) for a format the
    part gives natively, or ``(part, conversion)`` for a format it derives
    (``_PART_FORMATS``). A format with exactly one native part is owned by that
    part without a statement; a format with two or more native parts needs a
    statement. A wrong statement fails with a message that says what to write.
    """
    who = f"Subject '{subject.identifier}': "
    prefix = who + "capability_config['behavior-sessions'] "
    parts = _formats_by_part(subject)
    if not subject._extractors:
        parts.pop('readout', None)   # a readout layer without an extractor never answers
    followers = {f for formats in _PART_FORMATS.values() for f, how in formats.items()
                 if how == 'link'}
    formats = sorted({f for kinds in _PART_FORMATS.values() for f in kinds} - followers)
    owners: Dict[str, str] = {}
    for key, value in (stated or {}).items():
        if key in followers:
            raise ValueError(f"{prefix}names {key!r}: {key!r} follows the owner of "
                             f"'behavior:probabilities'; state that owner instead.")
        if key not in formats:
            raise ValueError(f"{prefix}names {key!r}, which is not a response format. "
                             f"Formats: {formats}.")
        if not (isinstance(value, str) or (type(value) is tuple and len(value) == 2
                                           and all(isinstance(v, str) for v in value))):
            raise ValueError(f"{prefix}gives {value!r} for {key!r}; a statement is a part "
                             f"('readout') or a tuple of a part and a conversion "
                             f"(('readout', 'argmax')).")
        part, conversion = (value, None) if isinstance(value, str) else value
        if part not in _ARGUMENTS:
            raise ValueError(f"{prefix}names the part {part!r} for {key!r}; parts are "
                             f"'generation', 'likelihood', 'readout'.")
        if conversion is not None and key == 'behavior':
            raise ValueError(f"{prefix}gives {value!r} for 'behavior'; 'behavior' takes a part, "
                             f"not a conversion.")
        if conversion is not None and conversion not in ('argmax', 'softmax', 'softmax_argmax'):
            raise ValueError(f"{prefix}names the conversion {conversion!r} for {key!r}; "
                             f"conversions are 'argmax', 'softmax', 'softmax_argmax'.")
        if part not in parts:
            raise ValueError(f"{prefix}names {part!r} for {key!r}, but no {_ARGUMENTS[part]} "
                             f"part is wired.")
        if key not in parts[part]:
            givers = sorted(p for p, given in parts.items() if key in given)
            raise ValueError(f"{prefix}names {part!r} for {key!r}, but {part!r} does not give "
                             f"{key!r}. Parts of this subject that give it: {givers}.")
        given = parts[part][key]
        if conversion is None and given is not None:
            raise ValueError(f"{prefix}{part!r} gives {key!r} only by a conversion. "
                             f"Write {key!r}: ({part!r}, {given!r}).")
        if conversion is not None and given is None:
            raise ValueError(f"{prefix}{part!r} gives {key!r} natively. Write {key!r}: {part!r}.")
        if conversion != given:
            raise ValueError(f"{prefix}{part!r} gives {key!r} by {given!r}, not by "
                             f"{conversion!r}. Write ({part!r}, {given!r}).")
        owners[key] = part
    for output in formats:
        native = sorted(p for p, given in parts.items()
                        if output in given and given[output] is None)
        if output in owners or not native:
            continue
        if len(native) > 1:
            raise ValueError(
                f"{who}the parts {native} give {output!r} natively. State the owner: "
                f"capability_config={{'behavior-sessions': {{{output!r}: {native[0]!r}}}}} (or "
                + ' or '.join(repr(p) for p in native[1:]) + ').')
        owners[output] = native[0]
    return owners


def _fitted_intact(task) -> bool:
    """Whether a configuration for ``task`` is fitted with every intervention switched off."""
    return task is not None and task.readout_fit == 'intact'
