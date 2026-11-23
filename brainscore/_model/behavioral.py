"""How a BrainScoreModel answers a behavioral task.

One chooser (``_choose_path``) picks the part that answers: ``generation``
(the ``reply`` part, or the ``answer`` part), ``likelihood`` or ``readout``.
One entry (``respond``) answers the stimuli and records regions during the
same presentation. A task states its ``response`` (Choice, Scale, Text); the
bare ``behavior`` channel carries the same answer as the response's first
format. The packaging is in ``answers``; the recording during the answering
pass is in ``recording``.
"""

import math
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from typing import Any, Dict, Optional, Tuple

from brainscore_core.contract import TaskContext
from brainscore_core.response_formats import argmax, response_record, softmax, softmax_argmax
from brainscore_core.streaming import StreamEvent

import numpy as np

from . import answers, parts
from .dispatch import detect_modalities, extract_for_modality, pick_modality, record
from .recording import gather_region, openable_paths, per_stimulus, region_modality, region_selector


# A likelihood part gives each option's summed log probability: the continuation
# log-probability format (Response Format Standard, F6; no channel yet).
LOG_PROBABILITIES = 'behavior:log_probabilities'


def _softmax_record():
    return response_record('behavior:probabilities', 'likelihood', conversion='softmax',
                           source_format=LOG_PROBABILITIES)


def _softmax_probabilities(outputs, logp, stimuli, axis):
    """Store the softmax of likelihood scores, with the scores as source evidence;
    return the probabilities a reaction time reads, their validity and the record.

    The likelihood stores its probabilities as float32 and the link reads that float32
    copy, as it did before the response format standard (finite rows stay bit for bit);
    a readout's link reads its float64 rows (``_readout_probabilities``).
    """
    probabilities, valid = softmax(logp)
    record = _softmax_record()
    outputs['behavior:probabilities'] = answers.implied_array(
        probabilities, valid, stimuli, axis, record, raw=answers.source_rows(logp))
    return np.asarray(outputs['behavior:probabilities'].values, dtype=float), valid, record


class BehavioralReadout:
    """Owns behavioral task state for a BrainScoreModel instance."""

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

    def start_task(self, task_or_type, fitting_stimuli=None, formats=None,
                   **kwargs) -> None:
        """Configure ``task``; fit the readout when it serves the requested ``formats``.

        ``formats`` are the outputs a session requests (``None``: every format
        of the task, as for a direct call). A later session that requests a
        format the readout serves fits it then (``fit_for``).
        """
        del kwargs
        if isinstance(task_or_type, TaskContext):
            task = task_or_type
        else:
            task = TaskContext(
                task_type=task_or_type,
                fitting_stimuli=fitting_stimuli,
            )
        self.task = task
        self.validate_task(task)
        if task.response is None and task.task_type == 'passive':
            self.response_path = self.probability_source = None
            self.readout_classifier = None
            return
        # A session chose the answering part once, in its plan; a direct call chooses here.
        planned = self._planned(task, formats)
        picks = [planned] if planned is not None else self._eligible_picks(task)
        self.response_path = picks[0][0]
        self.probability_source = next((source for _, source in picks if source), None)
        self.readout_classifier = None
        self.fit_for(formats, picks)

    def validate_task(self, task) -> None:
        """Refuse a malformed task: an unknown ``readout_fit``, or a task with no response
        that is not passive (``ValueError``)."""
        if task.readout_fit not in ('current', 'intact'):
            raise ValueError(f"readout_fit must be 'current' or 'intact', not {task.readout_fit!r}.")
        if task.response is None and task.task_type != 'passive':
            raise ValueError(
                f"task_type {task.task_type!r} states no response. BrainScoreModel answers a "
                f"task with a response (Choice, Scale or Text, from brainscore_core.responses), "
                f"for example TaskContext(response=Choice(labels), fitting_stimuli=...), or "
                f"task_type='passive'.")

    def by_answer(self, task) -> bool:
        """Within generation, whether the ``answer`` part answers ``task`` (else the ``reply`` part).

        A plain Choice goes to answer= when it is wired: the call a
        registration makes for a choice. The reply, which receives the full prompt,
        answers task settings (verbatim, present, options per row), a Scale or a
        Text, trial history, or when answer= is absent. A session decides it once,
        in its plan. When the registration states ``similarity`` for the choice, the
        answer part answers every Choice; a setting it cannot serve is refused.
        """
        from brainscore_core.responses import Choice
        owner, response = self.owner, task.response
        if self.owners.get('behavior:choice') == 'similarity':
            # The stated owner is the answer part: no reply stands in (rule 4.11).
            return owner._answer is not None and isinstance(response, Choice)
        return (owner._answer is not None and isinstance(response, Choice)
                and not response.verbatim and task.present is None
                and not isinstance(response.labels, str) and owner._memory != 'trial_history')

    def _planned(self, task, formats):
        """The answering part the running session's plan chose for ``task`` and ``formats``, or None."""
        plan = self.owner._session_plan
        return None if plan is None or formats is None else plan.pick_for(task, formats)

    def _eligible_picks(self, task) -> list:
        """The selection of each format ``task`` can be answered in, one format at a time.

        The setup and its reuse both use it. The setup is keyed by the task alone:
        a single format never crosses two parts, so every refusal comes before the
        fit. A probability or log-probability format that no part serves is skipped: a session that
        requests it is refused by its own selection, before any compute.
        """
        from brainscore_core.compatibility import ExecutionNotFeasibleError
        from brainscore_core.responses import response_channels
        picks = []
        for output in response_channels(task.response):
            try:
                picks.append(self._choose_path(task, [output]))
            except ExecutionNotFeasibleError:
                if output not in self._PROBABILITY_SIDE | {LOG_PROBABILITIES}:
                    raise
        return picks

    def fit_for(self, formats=None, picks=None) -> bool:
        """Fit the readout of the configured task once, when ``formats`` need it.

        The readout is needed when it answers the task or owns a requested
        format (``formats=None``: any format of the task). Returns whether it fitted.
        """
        if self.task is None or self.readout_classifier is not None:
            return False
        if formats is None:
            picks = picks or self._eligible_picks(self.task)
        else:
            picks = [self._planned(self.task, formats) or self._choose_path(self.task, formats)]
        if not any('readout' in pick for pick in picks):
            return False
        self.fit_behavioral_readout(self.task.fitting_stimuli)
        return True

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
        parts.note('readout', self.readout_classifier, [owner._behavioral_readout_layer])   # the fit

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
        owner, request = self.owner, self.owner._recording_request
        saved, saved_classifier = request.snapshot(), self.readout_classifier
        layer = owner._behavioral_readout_layer
        # Record the readout layer alone; the time bins stay as configured.
        request.recording_layer, request.recording_layers = layer, [layer] if layer else []
        request.recording_regions, request.is_multi_region, request.composite_recording = [], False, False
        self.readout_classifier = None
        try:
            features = record(owner, stimuli)
        finally:
            request.restore(saved)
            self.readout_classifier = saved_classifier

        if 'time_bin' in features.dims:
            # Limit: this refusal comes after one pass at the readout layer, not before compute:
            # no extractor states a time axis before it runs. Name a readout layer with no time axis.
            from brainscore_core.compatibility import ExecutionNotFeasibleError
            raise ExecutionNotFeasibleError(
                f"Subject {owner.identifier!r}: the features of the behavioral readout layer "
                f"{layer!r} keep a time axis (time_bin), and the readout fits one feature "
                "vector per stimulus; it does not reduce over time. Name a readout layer whose "
                "features have no time axis, or answer with an answering part (answer or "
                "reply) that states its own reduction over time.",
                task_type=self.task.task_type if self.task is not None else None)
        if 'presentation' in features.dims and 'neuroid' in features.dims:
            features = features.transpose('presentation', 'neuroid')
        return features

    def _owner_serving(self, output, can, task, settings) -> Optional[str]:
        """The owner of the answer format ``output``, when it can serve ``task`` (rule 4.11).

        The owner is the stated part, or the one part that gives the format natively
        (``answer_owners``). A part that only derives the format owns it only by a
        statement. When the owner cannot serve the task form, the task is refused before
        any compute: no other part answers a format it does not own. ``None``: no part gives it.

        Limit: a subject whose only choice-giving part derives the choice (a likelihood-only
        part set) and that states no choice owner is refused, with the statement to write, also
        when it records a region with probabilities; write the statement the error names.
        """
        from brainscore_core.compatibility import ExecutionNotFeasibleError
        formats = _formats_by_part(self.owner)
        givers = sorted(p for p, given in formats.items() if output in given)
        stated = self.owners.get(output)
        owner = 'generation' if stated == 'similarity' else stated
        if owner is None and givers:
            raise ExecutionNotFeasibleError(
                f"Subject '{self.owner.identifier}': the parts {givers} derive {output!r} and no "
                f"owner is stated." + statement_hint(self.owner, [output]), task_type=task.task_type)
        if owner is None or (stated != 'similarity'
                             and not (can.get(owner) or any(can.get(p) for p in givers))):
            return None          # no part can serve it: the caller names what each needs
        # A stated similarity owner is refused by name: the paths of the caller do not apply.
        if not can.get(owner):
            raise ExecutionNotFeasibleError(
                f"Subject '{self.owner.identifier}': {output!r} is owned by {stated or owner!r}, which "
                f"cannot serve this task"
                + (f" (task settings: {', '.join(settings)})" if settings else '')
                + '; no other part answers it. '
                + ('Remove the named settings from the task' if settings else 'Change the task')
                + f', or use a registration whose owner of {output!r} serves it.',
                task_type=task.task_type)
        return owner

    # The formats of a response task on each side of a crossed likelihood/readout pair.
    _CHOICE_SIDE = frozenset({'behavior', 'behavior:choice'})
    _PROBABILITY_SIDE = frozenset({'behavior:probabilities', 'behavior:rt'})
    _ANSWER_FORMATS = (('Choice', 'behavior:choice'), ('Scale', 'behavior:rating'),
                       ('Text', 'behavior:text'))

    def _choose_path(self, task, formats=(), recording: bool = False) -> Tuple[str, Optional[str]]:
        """The part that answers ``task`` on the requested ``formats`` and the probability source.

        The task decides what the subject does; the requested outputs decide what
        is measured (Contract Rule 4.5). Under a choice task the owner of the
        choice gives its answer, except that a session that requests only
        probabilities or a reaction time, records no region (``recording``), has
        trial memory ``'none'`` and a subject that keeps no trial history runs only
        their owner: nothing reads that answer. The
        owner of the probabilities runs when, and only when, the session requests
        probabilities or a reaction time. A format that the session does not request needs no owner:
        a reaction-time link names the rt format; it is no setting of the choice.

        A session that requests ``behavior:log_probabilities`` is answered by the
        likelihood, its only native owner: for a ``Segments`` task by its ``segment``
        layout, else only for a verbatim Choice; the
        choice and the probabilities it also requests come from that same pass
        (``softmax_argmax``, ``softmax``), so the likelihood runs once. The owner
        of the task's choice must also be able to serve the task, except in a
        session that requests only the log probabilities, records no region, has
        trial memory ``'none'`` and a subject that keeps no trial history.

        A task with a response is answered by the owner of its answer format
        (the response kind's own format: choice, rating or text) and, when
        ``formats`` request probabilities or a reaction time, its implied
        probabilities by the owner of ``behavior:probabilities`` (rule 4.11,
        ``answer_owners``); an owner that cannot serve the task's settings
        (``can`` below) is refused, no other part stands in. When the choice and the
        probabilities have two owners, a likelihood and a readout, one pass
        cannot give both: formats on one side are answered by that side's
        owner, formats on both sides are refused.

        A task that no part of the model can answer raises
        ``ExecutionNotFeasibleError`` before any compute. Sessions refuse with
        this choice; the refusal is an error of the run. A session makes it
        once, in its plan (``_model/plan.py``).
        """
        return self._choose(task, formats, recording)

    def _choice_owner_serves(self, task) -> None:
        """Refuse a task whose choice format has no owner that can serve it (part of one choice)."""
        self._choose(task, ['behavior:choice'])

    def _choose(self, task, formats, recording=False):
        from brainscore_core.compatibility import ExecutionNotFeasibleError
        from brainscore_core.responses import Choice, Scale, Segments
        owner, response = self.owner, task.response
        if isinstance(response, Segments):
            # Only a likelihood part that states the segment layout answers it (ADR-0075).
            if owner._segment_likelihood is None:
                raise ExecutionNotFeasibleError(
                    f"Subject '{owner.identifier}' cannot give {LOG_PROBABILITIES} for a "
                    "Segments task: its likelihood= part states no 'segment' layout "
                    "(likelihood={'continuation': ..., 'segment': ...}).", task_type=task.task_type)
            # The segment part scores the row's parts only: no instruction (it could stand in
            # for the row's text) and no earlier trials. The parts each trial shows are checked
            # before any call (_segment_answers).
            shown = [name for name, on in (('an instruction', bool((task.instruction or '').strip())),
                                           ("earlier trials (trial_memory='history')",
                                            task.trial_memory != 'none')) if on]
            if shown:
                raise ExecutionNotFeasibleError(
                    f"Subject '{owner.identifier}' scores the parts of a Segments row only; it "
                    f"cannot show {' or '.join(shown)}.", task_type=task.task_type)
            return 'likelihood', None
        readout_wired = (owner._behavioral_readout_layer is not None
                         and bool(owner._extractors) and task.fitting_stimuli is not None)
        requested = set(formats)
        log_probabilities = isinstance(response, Choice) and LOG_PROBABILITIES in requested
        if log_probabilities and owner._likelihood is None:
            # Log probabilities come only from a likelihood part: no readout, no guess.
            raise ExecutionNotFeasibleError(
                f"Subject '{owner.identifier}' cannot give {LOG_PROBABILITIES}: it comes only "
                'from a likelihood= part.', task_type=task.task_type)
        per_row = isinstance(response, Choice) and isinstance(response.labels, str)
        verbatim = isinstance(response, Choice) and response.verbatim
        if log_probabilities and not verbatim:
            # The record's tokenization is known only for plain-text scoring.
            raise ExecutionNotFeasibleError(
                f"Subject '{owner.identifier}' gives {LOG_PROBABILITIES} only for a verbatim "
                'Choice (Choice(..., verbatim=True)): its tokenization is then the plain text.',
                task_type=task.task_type)
        presented = task.present is not None
        # A stated similarity owner is the answer part alone: the reply does not stand in.
        similarity = isinstance(response, Choice) and self.owners.get('behavior:choice') == 'similarity'
        can = {
            'generation': not log_probabilities and (owner._reply is not None and not similarity or (
                owner._answer is not None and isinstance(response, Choice) and not verbatim
                and not presented and not (similarity and owner._memory == 'trial_history'))),
            'likelihood': owner._likelihood is not None and isinstance(response, Choice),
            'readout': (readout_wired and isinstance(response, (Choice, Scale))
                        and not log_probabilities and not per_row),
        }
        # The answer format: the log probabilities when the session requests them,
        # else the response's first format. Its owner answers; a task form the owner
        # cannot serve is refused (no other part stands in).
        settings = [name for name, on in (('verbatim', verbatim), ('present', presented),
                                          ('labels per row', per_row)) if on]
        answer_format = (LOG_PROBABILITIES if log_probabilities
                         else dict(self._ANSWER_FORMATS)[type(response).__name__])
        source = None
        wanted = requested & self._PROBABILITY_SIDE
        # Nothing reads the answer of the choice owner: no region is recorded and no later trial reads this one.
        unread = (not recording and task.trial_memory == 'none'
                  and owner._memory != 'trial_history')
        # The requested formats decide the passes: the probability owner runs (first,
        # for a reply, so a recording comes from its pass) only when the session
        # requests probabilities or a reaction time.
        if isinstance(response, Choice) and wanted and not log_probabilities:
            source = self._owner_serving('behavior:probabilities', {
                'likelihood': owner._likelihood is not None,
                'readout': readout_wired and not per_row}, task, settings)
            if source is None:
                raise ExecutionNotFeasibleError(
                    f"Subject '{owner.identifier}' cannot give implied probabilities: "
                    'it needs likelihood= or a readout (behavioral_readout_layer=, '
                    'fitting_stimuli).', task_type=task.task_type)
            if not requested - self._PROBABILITY_SIDE and unread:
                # Only the probability side is requested, no region is recorded and no later
                # trial reads this one: nothing reads an answer of the choice owner, so it
                # needs none and only the probability owner runs (ADR-0063 decision 1,
                # ADR-0073 decision 5).
                return source, source
        path = self._owner_serving(answer_format, can, task, settings)
        if path is None:
            needs = {'generation': 'reply= (or answer= for a plain Choice with none of the settings '
                                   'verbatim, present)',
                     'likelihood': 'likelihood= and a Choice',
                     'readout': 'behavioral_readout_layer=, an extractor and fitting_stimuli'}
            raise ExecutionNotFeasibleError(
                f"Subject '{owner.identifier}' cannot give a {type(response).__name__} answer"
                + (f" with the task settings {', '.join(settings)}" if settings else '')
                + '. Paths tried: ' + '; '.join(f'{p} needs {need}' for p, need in needs.items()),
                task_type=task.task_type)
        if log_probabilities and not (requested == {LOG_PROBABILITIES} and unread):
            # The task's choice format keeps an owner that can serve the task, as the setup
            # of the task always checked (``_eligible_picks``); the likelihood answers it here.
            # A session that requests only the log probabilities and that nothing else reads
            # needs no choice owner (as the probability-only pass below).
            self._choice_owner_serves(task)
        if log_probabilities:
            # One likelihood pass gives the log probabilities and what derives from them.
            return path, (path if wanted else None)
        if wanted and not requested - self._PROBABILITY_SIDE and unread:
            # A probability-only session of a response with no implied probabilities.
            return source, source
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
        from brainscore_core.responses import Choice, Scale, Segments, Text, labels_of
        owner, task = self.owner, self.task
        response, path, probability_source = task.response, self.response_path, self.probability_source
        planned = self._planned(task, channels)
        path, probability_source = planned or self._choose_path(task, channels, bool(regions))
        layers = {region: list(region_selector(owner, region).layer_paths) for region in regions}
        labels = labels_of(response)
        # Every trial's prompt is built first, so a bad layout fails before any compute
        # (for a session with its table, the plan built them before the readout fit).
        shown = openable_paths(stimuli, task)   # the parts see paths that open; the stored answer keeps the table
        prompts, row_labels, axis = trial_prompts(task, shown)
        if isinstance(response, Segments):
            return self._segment_answers(stimuli, shown, prompts, channels, layers)
        neural, outputs, extras = {}, {}, []
        rt_source = None   # (float64 probabilities, row validity, record) for a reaction time

        # The outputs each part answers here: what the code below writes from its result.
        probability_side = [c for c in channels if c in self._PROBABILITY_SIDE]
        answer_side = [c for c in channels if c not in self._PROBABILITY_SIDE and c != LOG_PROBABILITIES]
        if path == 'readout':
            with parts.serving(channels):
                values, neural = self._read_out_and_record(stimuli, layers)
            if isinstance(response, Scale):
                # The readout answers on the task's scale: the regression value is put on the
                # nearest step (the raw answer keeps it); a value outside the bounds stays.
                rated, parameters = _on_scale_steps([v for v, _ in values], response)
                outputs['behavior:rating'] = answers.numbers(
                    [(v, None) for v in rated], stimuli,
                    response_record('behavior:rating', 'readout', parameters=parameters),
                    raw=[v for v, _ in values])
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
            # Trial history without a reply or likelihood part: a session's plan refused it
            # before any set-up; a direct call with no matching plan is refused here.
            remember = owner._memory == 'trial_history'
            if planned is None and remember and owner._reply is None and owner._likelihood is None:
                raise ValueError("memory='trial_history' needs reply= or likelihood=; answer= cannot see "
                                 'earlier trials.')

            def with_history(prompt):
                return replace(prompt, history=tuple(self.history)) if remember else prompt

            # Within generation: the answer part or the reply part (the plan's decision).
            by_answer = (self.owner._session_plan.by_answer if planned is not None
                         else self.by_answer(task))
            by_reply = owner._reply is not None and not by_answer
            if path == 'likelihood':
                def call(row, prompt):
                    parts.note('likelihood', owner._likelihood, serves=channels)
                    scores = owner._likelihood(with_history(prompt), list(prompt.response.labels))
                    if remember:
                        self.history.append((prompt, max(scores, key=scores.get)))
                    return scores
            elif by_reply:
                def call(row, prompt):
                    shown = with_history(prompt)
                    # Scoring first keeps the recorded first pass the prompt alone.
                    if score_first:
                        parts.note('likelihood', owner._likelihood, serves=probability_side)
                    scores = (owner._likelihood(shown, list(prompt.response.labels))
                              if score_first else None)
                    parts.note('reply', owner._reply, serves=answer_side)
                    reply, extra = answers.reply_and_meta(owner._reply(shown))
                    if remember:
                        self.history.append((prompt, str(reply)))
                    return reply, scores, extra
            else:   # a registration's answer: it phrases a Choice itself; the reply is parsed below
                def call(row, prompt):
                    # The likelihood scores the prompt first when it owns the probabilities.
                    if score_first:
                        parts.note('likelihood', owner._likelihood, serves=probability_side)
                    scores = (owner._likelihood(prompt, list(prompt.response.labels))
                              if score_first else None)
                    parts.note('answer', owner._answer, serves=answer_side)
                    reply, extra = answers.reply_and_meta(owner._answer(
                        stimulus_row=row, instruction=task.instruction,
                        label_set=list(prompt.response.labels)))
                    return reply, scores, extra
            replies, neural = per_stimulus(owner, stimuli, layers, call, prompts, shown)
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
                if LOG_PROBABILITIES in channels:
                    outputs[LOG_PROBABILITIES] = answers.log_probabilities(
                        logp, stimuli, axis, response_record(
                            LOG_PROBABILITIES, 'likelihood', category='model-derived',
                            parameters=VERBATIM_LOG_PROBABILITIES))
                if probability_source == 'likelihood':
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
                # The declared method of the part that answered (rule 4.11): the answer
                # part states generation or similarity; the reply states generation.
                method = ('similarity' if not by_reply and self.owners.get('behavior:choice') == 'similarity'
                          else 'generation')
                outputs['behavior:choice'] = answers.one_hot(
                    [label for label, _ in parsed], [label is not None for label, _ in parsed],
                    stimuli, row_labels, axis,
                    response_record('behavior:choice', method, parsed_by='model',
                                    parameters=(VERBATIM_IN_TEMPLATE if response.verbatim
                                                and owner._reply is not None else None)),
                    raw=replies, errors=[error for _, error in parsed])
                if score_first:
                    rt_source = _softmax_probabilities(
                        outputs, answers.score_rows(scores, row_labels), stimuli, axis)
                elif probability_source == 'readout':
                    with parts.serving(probability_side):    # a separate probability pass
                        features = self.extract_behavioral_features(stimuli)
                        parts.note('readout', self.readout_classifier, [owner._behavioral_readout_layer])
                    rt_source = self._readout_probabilities(outputs, answers.in_order(
                        self.readout_classifier.predict_proba(features), labels), stimuli)
        if isinstance(response, Choice) and response.reaction_time and rt_source is not None:
            # From the probabilities each path read before the response format standard, with their validity.
            outputs['behavior:rt'] = answers.reaction_time(*rt_source, response.reaction_time,
                                                           stimuli)
        # The bare channel carries the same answer; its record names that channel.
        first = outputs[response.channels[0]]
        outputs['behavior'] = answers.with_record(first, {**first.attrs['response'],
                                                          'format': 'behavior'})
        log = parts.active()
        if log is not None:     # the conversion each answer record names, on the part that gave it
            for channel in channels:
                log.conversion(channel, outputs[channel].attrs['response'].get('conversion'))
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
                {r: a.assign_attrs(recording={**a.attrs.get('recording', {}), **stamp})
                 for r, a in neural.items()})

    def _segment_answers(self, stimuli, shown, prompts, channels, layers):
        """Answer a ``Segments`` task with the likelihood's ``segment`` part: one row of log
        probabilities for each stimulus, positions 1 to R, with the part's parameters.

        A value that is not finite stays as given; its row is invalid with the reason of each
        such position (position 1: no conditioning context, as no token of the first part
        has a prediction).
        """
        import json
        from brainscore_core.compatibility import ExecutionNotFeasibleError
        owner = self.owner
        # Before any compute: one trial that shows one text.
        kinds = {tuple(kind for kind, _ in prompt.parts) for prompt in prompts}
        if len(prompts) > 1 or kinds - {('text',)}:
            # ponytail: one Segments trial per session, so no row is scored before a later row
            # is refused; several passages need one session each
            raise ExecutionNotFeasibleError(
                f"Subject '{owner.identifier}' answers one Segments trial in a session, and it shows "
                f"one text (no picture); got {len(prompts)} trials with parts {sorted(kinds)}",
                task_type=self.task.task_type)

        def call(row, prompt):
            parts.note('likelihood', owner._segment_likelihood, serves=channels)
            return owner._segment_likelihood(prompt)

        results, neural = per_stimulus(owner, stimuli, layers, call, prompts, shown)
        if len({json.dumps(parameters, sort_keys=True) for _, parameters in results}) > 1:
            raise ExecutionNotFeasibleError(
                f"Subject '{owner.identifier}' gave the Segments rows of one session different "
                'records', task_type=self.task.task_type)
        # None: a part with no scored token, as the part states it.
        unscored = [[value is None for value in row] for row, _ in results]
        values = np.array([[np.nan if value is None else value for value in row]
                           for row, _ in results], dtype=np.float64)
        reasons = [' and '.join(f'position {i + 1}: '
                                + ('no conditioning context' if gaps[i] else 'not finite')
                                for i in np.flatnonzero(~np.isfinite(row)))
                   for row, gaps in zip(values, unscored)]
        record = response_record(LOG_PROBABILITIES, 'likelihood', category='model-derived',
                                 parameters=results[0][1])
        axis = [str(i) for i in range(1, values.shape[1] + 1)]
        answer = answers.log_probabilities(values, stimuli, axis, record).assign_coords(
            answer_valid=('presentation', [not reason for reason in reasons]),
            parse_error=('presentation', reasons))
        stamp = self._stamp('likelihood', 'prompt')
        return ({c: answer for c in channels},
                {r: a.assign_attrs(recording={**a.attrs.get('recording', {}), **stamp})
                 for r, a in neural.items()})

    @staticmethod
    def _readout_probabilities(outputs, ordered, stimuli):
        """Store a readout's probabilities (float64, as the readout gives them); return the
        rows, their validity as probability rows and the record, for a reaction time."""
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

    def _stamp(self, path, forward_pass):
        """How a recording was made, so task and passive recordings stay distinct."""
        return {'task_type': self.task.task_type, 'path': path,
                'response': type(self.task.response).__name__
                if self.task.response is not None else None,
                'forward_pass': forward_pass, 'readout_fit': self.task.readout_fit}

    def refuse_readout_regions(self, stimuli, regions) -> str:
        """The input the readout presents for ``stimuli``; refuse a region that reads another one.

        One presentation cannot record a region of another input
        (``ExecutionNotFeasibleError``). The session's plan calls it before the
        readout is fitted; the readout calls it again for a trial of a stream.
        """
        from brainscore_core.compatibility import ExecutionNotFeasibleError
        owner = self.owner
        modality = pick_modality(owner, detect_modalities(owner, stimuli))
        for region in regions:
            if owner._region_modality_map.get(region) and region_modality(owner, region) != modality:
                raise ExecutionNotFeasibleError(
                    f'Region {region!r} reads {region_modality(owner, region)!r} input, but '
                    f'the readout presents {modality!r} input; one presentation cannot '
                    'record it.', channels={f'neural:{region}'},
                    task_type=self.task.task_type if self.task is not None else None)
        return modality

    def _readout_requests(self, layers, passes):
        """The layer lists the readout extracts: one call for the readout layer and every
        region under ``BRAINSCORE_REGIONS_PER_PASS=all``, else one for each."""
        readout_layer = self.owner._behavioral_readout_layer
        if passes == 'all':
            return [list(dict.fromkeys([readout_layer, *(l for ls in layers.values() for l in ls)]))]
        return [[readout_layer], *(list(region_layers) for region_layers in layers.values())]

    def refuse_readout_request(self, stimuli, modality, layers, passes) -> None:
        """Refuse a readout extraction that no extractor of ``modality`` can make.

        The session's plan calls it before the readout is fitted, with the requests
        the readout will make (``_readout_requests``) and the one resolver of the
        recording (``extractor_or_refuse``), in the session's class
        (``ExecutionNotFeasibleError``); a direct call of the readout keeps its class.
        """
        from .dispatch import extractor_or_refuse
        for request in self._readout_requests(layers, passes):
            extractor_or_refuse(self.owner, stimuli, modality, request, in_session=True)

    def _read_out_and_record(self, stimuli, layers):
        from brainscore_core.responses import Scale
        owner = self.owner
        readout_layer = owner._behavioral_readout_layer
        modality = self.refuse_readout_regions(stimuli, layers)

        answering = parts.serving_now()

        def extract(call_layers):
            # This call serves the readout's outputs when it reads the readout layer, and each
            # region whose layers it reads.
            serves = (list(answering) if readout_layer in call_layers else []) + [
                f'neural:{region}' for region, region_layers in layers.items()
                if set(region_layers) & set(call_layers)]
            with parts.serving(serves):
                extracted = owner._recording_request.ensure_layer_provenance(
                    extract_for_modality(owner, stimuli, modality, call_layers), call_layers,
                    description='Readout extraction')
            layer_of = extracted['layer'].values
            return lambda layer: extracted.isel(neuroid=layer_of == layer)

        # BRAINSCORE_REGIONS_PER_PASS: 'all' extracts the readout layer and every
        # region in one call; 'one' gives the readout and each region its own call,
        # as does 'all' when the layers have different dtypes (one call cannot keep them).
        from brainscore.model_helpers.extractors import LayerDtypesDiffer
        joint = None
        plan = owner._session_plan
        if (plan.regions_per_pass if plan is not None else 'all') == 'all':
            all_layers = self._readout_requests(layers, 'all')[0]
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
        parts.note('readout', self.readout_classifier, [readout_layer])
        if isinstance(self.task.response, Scale):
            behavior = [(float(v), None) for v in self.readout_classifier.predict(features)]
        else:
            behavior = self.readout_classifier.predict_proba(features)
        neural = {}
        for region, region_layers in layers.items():
            per_layer = [part_of[region](layer) for layer in region_layers]
            selected = gather_region(owner, region, per_layer)
            neural[region] = selected.assign_coords(
                region=('neuroid', [region] * selected.sizes['neuroid']))
        return behavior, neural


def trial_prompts(task, stimuli, compare_options: bool = True):
    """Every trial's prompt, each row's options and the choice axis, before any compute.

    A bad layout raises ``ValueError`` (core ``trial_prompt``). With
    ``compare_options``, rows of a ``Choice`` whose options come from a column
    must give the same number of options (``ValueError``); the choice axis is
    then the labels, or the position of each option.
    """
    from brainscore_core.responses import Choice, labels_of, trial_prompt
    response = task.response
    prompts = [trial_prompt(task, row) for _, row in stimuli.iterrows()]
    labels = labels_of(response)
    row_labels, axis = None, labels
    if isinstance(response, Choice):
        row_labels = [list(prompt.response.labels) for prompt in prompts]
        if labels is None:
            counts = {len(options) for options in row_labels}
            if compare_options and len(counts) > 1:
                raise ValueError(f'Every row must give the same number of options in column '
                                 f'{response.labels!r}; got {sorted(counts)}')
            axis = list(range(counts.pop())) if counts else []
    return prompts, row_labels, axis


# The answer formats of each part kind (Response Format Standard 4.2; rule
# 4.11): format -> None when the part gives it natively, else the conversion
# that derives it. ``answer`` and ``reply`` are the two kinds of the part
# ``generation``. The likelihood gives the log probabilities natively. ``'link'``:
# ``behavior:rt`` follows the owner of ``behavior:probabilities`` and is
# never stated. Bare ``behavior`` is no part's format: it carries the answer of a response
# task under its first format, given by that format's owner (``_declared_formats``).
_PART_FORMATS = {
    'answer': {'behavior:choice': None},
    'reply': {'behavior:choice': None, 'behavior:rating': None, 'behavior:text': None},
    'likelihood': {LOG_PROBABILITIES: None, 'behavior:probabilities': 'softmax',
                   'behavior:choice': 'softmax_argmax', 'behavior:rt': 'link'},
    'readout': {'behavior:probabilities': None, 'behavior:rating': None,
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
    it is not declared. Bare ``behavior`` is declared when ``answer=`` or a readout
    with an extractor is wired. A ``'link'`` format (``behavior:rt``) is declared with
    ``behavior:probabilities``, whose owner gives it.
    """
    declared = set(subject._behavioral.owners)
    if subject._answer is not None or (subject._behavioral_readout_layer is not None
                                       and subject._extractors):
        declared.add('behavior')
    if 'behavior:probabilities' in declared:
        declared |= {f for formats in _PART_FORMATS.values() for f, how in formats.items()
                     if how == 'link'}
    return declared


def statement_hint(subject, missing) -> str:
    """The statement that would declare each missing format a wired part only derives.

    A refusal names the exact line to write, for example
    ``answer_owners={'behavior:choice': ('readout', 'argmax')}``.
    With two candidate parts it names each line and asks for one. Empty when no wired
    part derives a missing format.
    """
    parts = _formats_by_part(subject)
    if not subject._extractors:
        parts.pop('readout', None)
    lines = []
    for channel in sorted(missing):
        candidates = [(part, formats[channel]) for part, formats in sorted(parts.items())
                      if formats.get(channel) not in (None, 'link')]
        written = [f"answer_owners={{{channel!r}: ({part!r}, {how!r})}}"
                   for part, how in candidates]
        if len(written) == 1:
            lines.append(f' {channel!r} is derived by a wired part; state it: {written[0]}.')
        elif written:
            lines.append(f' {channel!r} is derived by {len(written)} wired parts; state one: '
                         + ' or '.join(written) + '.')
    return ''.join(lines)


# The methods a part states (the record's ``method``) and the argument that wires each.
# ``similarity``: an answer chosen by the similarity of an input to each option (rule
# 4.11); the ``answer=`` part states it in place of ``generation``, and reads no instruction.
_ARGUMENTS = {'generation': 'answer= or reply=', 'likelihood': 'likelihood=',
              'readout': 'behavioral_readout_layer= (with an extractor)',
              'similarity': 'answer='}


def answer_owners(subject, stated) -> Dict[str, str]:
    """Check the owner statements at construction; return the owner part of each format (rule 4.11).

    ``stated`` is the ``answer_owners`` argument of the model: format ->
    a part (``'generation'``, ``'likelihood'``, ``'readout'``, or ``'similarity'`` for an
    ``answer=`` part that chooses by similarity) for a format the part gives natively, or ``(part, conversion)`` for a format it derives
    (``_PART_FORMATS``). A format with exactly one native part is owned by that
    part without a statement; a format with two or more native parts needs a
    statement. A wrong statement fails with a message that says what to write.
    """
    who = f"Subject '{subject.identifier}': "
    prefix = who + "answer_owners "
    parts = _formats_by_part(subject)
    if not subject._extractors:
        parts.pop('readout', None)   # a readout layer without an extractor never answers
    followers = {f for formats in _PART_FORMATS.values() for f, how in formats.items()
                 if how == 'link'}
    formats = sorted({f for kinds in _PART_FORMATS.values() for f in kinds} - followers)
    stated_parts = dict(parts)
    if subject._answer is not None:
        stated_parts['similarity'] = dict(_PART_FORMATS['answer'])
    owners: Dict[str, str] = {}
    for key, value in (stated or {}).items():
        if key == 'behavior':
            raise ValueError(f"{prefix}names 'behavior'. Delete this line: bare 'behavior' "
                             f"carries the answer of a response task under its first format, "
                             f"given by the owner of that format; it takes no statement.")
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
                             + ', '.join(map(repr, _ARGUMENTS)) + '.')
        if conversion is not None and conversion not in ('argmax', 'softmax', 'softmax_argmax'):
            raise ValueError(f"{prefix}names the conversion {conversion!r} for {key!r}; "
                             f"conversions are 'argmax', 'softmax', 'softmax_argmax'.")
        if part not in stated_parts:
            raise ValueError(f"{prefix}names {part!r} for {key!r}, but no {_ARGUMENTS[part]} "
                             f"part is wired.")
        if key not in stated_parts[part]:
            givers = sorted(p for p, given in stated_parts.items() if key in given)
            raise ValueError(f"{prefix}names {part!r} for {key!r}, but {part!r} does not give "
                             f"{key!r}. Parts of this subject that give it: {givers}.")
        given = stated_parts[part][key]
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
                f"answer_owners={{{output!r}: {native[0]!r}}} (or "
                + ' or '.join(repr(p) for p in native[1:]) + ').')
        owners[output] = native[0]
    return owners


RATING_ROUNDING = 'nearest step, ties to the higher step'
# The schema of the log probabilities a likelihood part gives under a verbatim Choice:
# f"{text} {option}" as plain text, no special tokens, no template, summed natural logs.
VERBATIM_LOG_PROBABILITIES = {
    'layout': 'continuation', 'context': 'shared', 'aggregation': 'sum', 'units': 'nats',
    'tokenization': {'join': 'space', 'special_tokens': 'none', 'template': 'none',
                     'truncation': 'none'}}
# A verbatim trial text reaches a reply part as given; the part may wrap it in its own chat
# template. The record says only that: a reply part is any callable.
VERBATIM_IN_TEMPLATE = {'template': "the reply part's own, if it has one"}


def _on_scale_steps(values, scale) -> Tuple[list, dict]:
    """Each value on the nearest step of ``scale``, and the record parameters that say so.

    The grid of a scale is decimal: the points Decimal(repr(low)) + n * Decimal(repr(step)).
    The step index is read from the raw value in decimal; an index within 1e-9 of a half is a
    tie and goes to the higher step. The stored value is the float of that grid point, the
    value core's step rule accepts, kept inside the bounds: the index never passes the last
    point inside them, and a point one float rounding beyond ``high`` (or ``low``) is
    stored as that bound. A raw value outside the bounds, or not finite,
    is not rounded and not clipped (core marks it invalid). A scale with no step leaves every
    value unchanged and records no rounding.
    """
    if scale.step is None:
        return list(values), {}
    return [_grid_point(v, scale) for v in values], {'rounding': RATING_ROUNDING}


def _grid_point(value, scale):
    from decimal import Decimal, ROUND_FLOOR, localcontext
    if not (math.isfinite(value) and scale.low <= value <= scale.high):
        return value
    with localcontext() as context:
        context.prec = 50
        low, step = Decimal(repr(float(scale.low))), Decimal(repr(float(scale.step)))
        last = int(((Decimal(repr(float(scale.high))) - low) / step + Decimal('1e-9'))
                   .to_integral_value(ROUND_FLOOR))
        index = (Decimal(repr(float(value))) - low) / step
        below = int(index.to_integral_value(ROUND_FLOOR))
        n = below + 1 if index - below > Decimal('0.5') - Decimal('1e-9') else below
        point = float(low + min(n, last) * step)
    # A high (or low) one float rounding off a grid point: the stored value stays inside the
    # bounds; it moves by at most 1e-9 of a step, which core's step rule accepts.
    return min(max(point, float(scale.low)), float(scale.high))


def _fitted_intact(task) -> bool:
    """Whether a configuration for ``task`` is fitted with every intervention switched off."""
    return task is not None and task.readout_fit == 'intact'
