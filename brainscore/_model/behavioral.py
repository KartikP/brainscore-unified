"""How a BrainScoreModel answers a behavioral task.

One chooser (``_choose_path``) picks the part that answers: ``generation``
(the ``reply`` part, or the ``answer`` part), ``likelihood`` or ``readout``.
One entry (``respond``) answers the stimuli and records regions during the
same presentation. A task that states a ``response`` (Choice, Scale, Text)
and a legacy task (``task_type`` with ``label_set``) differ only in how the
answer is packaged: a legacy task keeps its one-hot or probability assembly,
which legacy behavioral benchmarks score.
"""

from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import replace
from typing import Any, Dict, List, Optional, Tuple

from brainscore_core.contract import TaskContext
from brainscore_core.streaming import StreamEvent

import numpy as np


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
                task.prefer_path or 'auto', task.readout_fit,
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
        """The model changed (an intervention was applied or undone): fit again next time."""
        self._stale = True
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
                if self._changes != changes:
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
        path, source = self._choose_path(task)
        self.response_path, self.probability_source = path, source
        self.readout_classifier = None
        if 'readout' in (path, source):
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
        owner = self.owner
        saved_layer = owner._recorder.recording_layer
        saved_layers = owner._recorder.recording_layers
        saved_regions = owner._recorder.recording_regions
        saved_multi = owner._recorder.is_multi_region
        saved_composite = owner._recorder.composite_recording
        saved_classifier = self.readout_classifier
        owner._recorder.recording_layer = owner._behavioral_readout_layer
        owner._recorder.recording_layers = (
            [owner._behavioral_readout_layer]
            if owner._behavioral_readout_layer else []
        )
        owner._recorder.recording_regions = []
        owner._recorder.is_multi_region = False
        owner._recorder.composite_recording = False
        self.readout_classifier = None
        try:
            features = owner.process(stimuli)
        finally:
            owner._recorder.recording_layer = saved_layer
            owner._recorder.recording_layers = saved_layers
            owner._recorder.recording_regions = saved_regions
            owner._recorder.is_multi_region = saved_multi
            owner._recorder.composite_recording = saved_composite
            self.readout_classifier = saved_classifier

        if 'presentation' in features.dims and 'neuroid' in features.dims:
            features = features.transpose('presentation', 'neuroid')
        return features

    def predict_probabilities(self, stimuli):
        """Return a BehavioralAssembly of per-label probabilities."""
        features = self.extract_behavioral_features(stimuli)
        return _implied(self.readout_classifier.predict_proba(features), 'readout')

    _PATHS = ('generation', 'likelihood', 'readout')

    def _choose_path(self, task) -> Tuple[str, Optional[str]]:
        """The part that answers ``task`` and the source of implied probabilities, before any compute.

        ``prefer_path='auto'`` tries generation, then likelihood, then the
        readout; another value names the one path to use. A legacy task
        (``task_type`` with ``label_set``) is answered by ``answer=`` (with an
        instruction and labels) or by the readout (with fitting stimuli).
        """
        from brainscore_core.responses import Choice, Scale
        owner, response = self.owner, task.response
        prefer = task.prefer_path or 'auto'
        if prefer not in ('auto',) + self._PATHS:
            raise ValueError(f"TaskContext.prefer_path must be one of {('auto',) + self._PATHS}; "
                             f'got {prefer!r}.')
        readout_wired = (owner._behavioral_readout_layer is not None
                         and bool(owner._extractors))
        order = list(self._PATHS) if prefer == 'auto' else [prefer]
        if response is None:
            can = {'generation': bool(owner._answer is not None and task.instruction and task.label_set),
                   'likelihood': False,
                   'readout': task.fitting_stimuli is not None}
            needs = {'generation': 'answer=, TaskContext.instruction and TaskContext.label_set',
                     'likelihood': 'a TaskContext(response=Choice(...)); a legacy task has no likelihood path',
                     'readout': 'TaskContext.fitting_stimuli (and behavioral_readout_layer=)'}
            path = next((p for p in order if can[p]), None)
            if path is None:
                raise ValueError(
                    f"Subject '{owner.identifier}' cannot run the behavioral task "
                    f"task_type={task.task_type!r}. Paths tried: "
                    + '; '.join(f'{p} needs {needs[p]}' for p in order))
            if prefer == 'auto' and path == 'generation' and can['readout'] and readout_wired:
                import warnings
                warnings.warn(
                    'Both generation and fitted readout are available; '
                    "prefer_path='auto' selects generation and does not fit on "
                    'fitting_stimuli. Set TaskContext.prefer_path to '
                    "'generation' or 'readout' to choose explicitly.",
                    UserWarning, stacklevel=4)
            return path, None
        readout_wired = readout_wired and task.fitting_stimuli is not None
        reported_confidence = isinstance(response, Choice) and response.confidence is not None
        raw_scores = isinstance(response, Choice) and response.raw_scores
        if raw_scores and owner._likelihood is None:
            # Raw scores come only from a likelihood part: no softmax of a readout, no guess.
            from brainscore_core.compatibility import CompatibilityError
            raise CompatibilityError(
                f"Subject '{owner.identifier}' cannot give raw scores of the options: "
                'they come only from a likelihood= part.')
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
        path = next((p for p in order if can[p]), None)
        if path is None:
            needs = {'generation': 'reply= (or answer= for a plain Choice)',
                     'likelihood': 'likelihood= and a Choice without reported confidence',
                     'readout': 'behavioral_readout_layer=, an extractor and fitting_stimuli'}
            raise ValueError(
                f"Subject '{owner.identifier}' cannot give a {type(response).__name__} answer"
                + (' with reported confidence (needs reply=)' if reported_confidence else '')
                + '. Paths tried: ' + '; '.join(f'{p} needs {needs[p]}' for p in order))
        source = None
        if isinstance(response, Choice) and (response.probabilities or response.reaction_time
                                             or response.raw_scores):
            if path != 'readout' and owner._likelihood is not None:
                source = 'likelihood'
            elif readout_wired and not per_row:
                source = 'readout'
            else:
                raise ValueError(f"Subject '{owner.identifier}' cannot give implied probabilities: "
                                 'it needs likelihood= or a readout (behavioral_readout_layer=, '
                                 'fitting_stimuli).')
        return path, source

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
        response, path = task.response, self.response_path
        layers = {region: self._region_layers(region) for region in regions}
        if response is None:
            behavior, neural = self._answer_legacy_task(stimuli, layers)
            return ({channel: behavior for channel in channels}, neural)
        labels = labels_of(response)
        # Every trial's prompt is built first, so a bad layout fails before any compute.
        prompts = [trial_prompt(task, row) for _, row in stimuli.iterrows()]
        neural, outputs, extras = {}, {}, []
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
                outputs['behavior:rating'] = _numbers(values, stimuli, 'readout')
            else:
                outputs['behavior:choice'] = _one_hot(
                    [str(values['choice'].values[i]) for i in np.asarray(values.values).argmax(1)],
                    [True] * len(stimuli), stimuli, row_labels, axis, 'readout', raw=None)
                if self.probability_source == 'readout':
                    outputs['behavior:probabilities'] = _implied(_in_order(values, labels), 'readout')
        else:
            score_first = path == 'generation' and self.probability_source == 'likelihood'
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
                    reply, extra = _reply_and_meta(owner._reply(shown))
                    if remember:
                        self.history.append((prompt, str(reply)))
                    return reply, scores, extra
            else:   # a registration's answer: it phrases and parses a Choice itself
                def call(row, prompt):
                    return owner._answer(stimulus_row=row, instruction=task.instruction,
                                                label_set=list(prompt.response.labels))
            replies, neural = self._per_stimulus(stimuli, layers, call, prompts)
            scores = None
            if path == 'generation' and owner._reply is not None:
                replies, scores, extras = (list(column) for column in zip(*replies)) if replies else ([], [], [])
            if path == 'likelihood':
                logp = _score_rows(replies, row_labels)
                probabilities = _softmax(logp)
                chosen = [options[i] for options, i in zip(row_labels, probabilities.argmax(1))]
                outputs['behavior:choice'] = _one_hot(chosen, [True] * len(stimuli), stimuli,
                                                      row_labels, axis, 'likelihood', raw=None)
                if response.raw_scores:
                    outputs['behavior:probabilities'] = _implied_array(
                        logp, stimuli, axis, 'likelihood', scale='log')
                elif self.probability_source == 'likelihood':
                    outputs['behavior:probabilities'] = _implied_array(
                        probabilities, stimuli, axis, 'likelihood')
            elif isinstance(response, Text):
                parsed = [response.parse(reply) for reply in replies]
                outputs['behavior:text'] = _texts(parsed, replies, stimuli)
            elif isinstance(response, Scale):
                parsed = [response.parse(reply) for reply in replies]
                outputs['behavior:rating'] = _numbers(parsed, stimuli, 'reply', raw=replies)
            else:
                parsed = [prompt.response.parse(str(reply)) for prompt, reply in zip(prompts, replies)]
                source = 'reply' if owner._reply is not None else 'answer'
                outputs['behavior:choice'] = _one_hot(
                    [label for label, _ in parsed], [label is not None for label, _ in parsed],
                    stimuli, row_labels, axis, source, raw=[str(r) for r in replies],
                    errors=[error for _, error in parsed])
                if response.confidence is not None:
                    outputs['behavior:confidence'] = _numbers(
                        [response.parse_confidence(reply) for reply in replies], stimuli,
                        'reply', raw=replies)
                if score_first:
                    outputs['behavior:probabilities'] = _implied_array(
                        _softmax(_score_rows(scores, row_labels)), stimuli, axis, 'likelihood')
                elif self.probability_source == 'readout':
                    outputs['behavior:probabilities'] = _implied(_in_order(
                        self.readout_classifier.predict_proba(self.extract_behavioral_features(stimuli)),
                        labels), 'readout')
        if isinstance(response, Choice) and response.reaction_time:
            outputs['behavior:rt'] = _reaction_time(outputs['behavior:probabilities'],
                                                    response.reaction_time, stimuli)
        outputs['behavior'] = outputs[response.channels[0]]
        meta = _trials_meta(extras)
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

    def _open_recordings(self, stimuli, layers):
        """One live recording per input type, before any compute.

        An extractor's own ``record(layers)`` when it has one; otherwise forward
        hooks on the model passed as ``BrainScoreModel(model=...)``, with layer
        paths read from that model.
        """
        owner = self.owner
        from .dispatch import extractor_for
        by_modality: Dict[str, list] = {}
        for region, region_layers in layers.items():
            by_modality.setdefault(self._region_modality(region), []).extend(region_layers)
        recordings = {}
        for modality, modality_layers in by_modality.items():
            modality_layers = list(dict.fromkeys(modality_layers))
            extractor = extractor_for(owner, modality, stimuli)
            if extractor is None:
                raise NotImplementedError(f'No {modality!r} extractor to record from.')
            try:
                record = extractor.record
            except AttributeError:
                record = None
            if record is not None:
                recordings[modality] = record(modality_layers)
            elif owner._model is not None:
                recordings[modality] = _hook_recording(owner._model, modality_layers, owner.identifier)
            else:
                raise NotImplementedError(
                    f'The {modality!r} extractor cannot record while the model answers: give it '
                    'a record(layers) method, or pass the network as BrainScoreModel(model=...).')
        return recordings

    def _per_stimulus(self, stimuli, layers, call, prompts=None):
        """Call ``call(row, prompt)`` once per stimulus, recording ``layers`` on each first pass."""
        recordings = self._open_recordings(stimuli, layers)
        replies, trials = [], []
        with ExitStack() as stack:
            live = {m: stack.enter_context(r) for m, r in recordings.items()}
            for index, (_, row) in enumerate(stimuli.iterrows()):
                replies.append(call(row, None if prompts is None else prompts[index]))
                trials.append({m: recording.take() for m, recording in live.items()})
        return replies, self._trial_regions(trials, stimuli, layers)

    def _trial_regions(self, trials, stimuli, layers):
        """Per-trial recordings -> one NeuroidAssembly per region (composite regions gathered)."""
        import xarray as xr
        neural = {}
        for region, region_layers in layers.items():
            modality = self._region_modality(region)
            recorded = [trial[modality] for trial in trials]
            needed = self._required_layers(region)
            # Only required layers: a layer whose unit subset is empty contributes no
            # units, so it may be absent from any trial (UnitSelector.required_layers).
            per_layer = [_trials_to_assembly(recorded, stimuli, layer, region)
                         for layer in region_layers if layer in needed]
            neural[region] = self._gather(region, per_layer)
        return neural

    def _required_layers(self, region: str):
        """The layers that must contribute neuroids to ``region`` (the selector's rule)."""
        from .recording import region_selector
        return region_selector(self.owner, region).required_layers

    def _gather(self, region, per_layer):
        """A region's units: one layer as is; a composite region's units gathered from its layers.

        Every layer the region requires must have contributed neuroids, as in
        the neural session path, whichever path (readout or generation) recorded.
        """
        import numpy as np
        import xarray as xr
        from .recording import region_selector
        selector = region_selector(self.owner, region)
        needed = selector.required_layers
        present = {str(layer) for block in per_layer if block.sizes['neuroid']
                   for layer in np.asarray(block['layer'].values)}
        missing = [layer for layer in needed if layer not in present]
        if missing or not present:
            raise ValueError(
                f'Region {region!r} needs neuroids from every selected layer '
                f'{list(needed)}; the extractor returned no neuroids for '
                f'{missing or "any of its layers"}.')
        if not selector.gathers_units:
            return per_layer[0]
        recorder = self.owner._recorder
        from brainscore_core.time_axis import require_one_time_axis
        require_one_time_axis(per_layer, sources=[f'{region}:layer{i}' for i in range(len(per_layer))])
        combined = xr.concat([recorder.ensure_unit_indices(a) for a in per_layer], dim='neuroid')
        selected = selector.gather_units(combined)
        if not selected.sizes['neuroid']:
            raise ValueError(f'Region {region!r} selects no neuroids from its layers.')
        return selected

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
            answers, neural = self._per_stimulus(
                stimuli, layers, lambda row, _: self.owner._answer(
                    stimulus_row=row, instruction=task.instruction, label_set=label_set))
            stamp = self._stamp('answer', 'prompt')
            return (_legacy_one_hot(answers, label_set, stimuli),
                    {region: assembly.assign_attrs(recording=stamp) for region, assembly in neural.items()})
        if layers:
            return self._read_out_and_record(stimuli, layers)
        return self.predict_probabilities(stimuli), {}

    def _region_layers(self, region: str) -> List[str]:
        selector = self.owner._region_layer_selectors.get(region)
        if selector is None:
            raise ValueError(f'Region {region!r} is not in region_layer_map.')
        return list(selector.layer_paths)

    def _region_modality(self, region: str) -> str:
        from .recording import region_modality
        modality = region_modality(self.owner, region)
        if modality is None:
            raise ValueError(
                f'Region {region!r} does not say which input it reads. Map it with '
                "LayerSelector(name=..., modality=...) to record it during a task.")
        return modality

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
        modality = owner._pick_modality(owner.detect_modalities(stimuli))
        for region in layers:
            if owner._region_modality_map.get(region) and self._region_modality(region) != modality:
                raise ValueError(
                    f'Region {region!r} reads {self._region_modality(region)!r} input, but '
                    f'the readout presents {modality!r} input; one presentation cannot '
                    'record it.')
        all_layers = list(dict.fromkeys([readout_layer, *(l for ls in layers.values() for l in ls)]))
        extracted = owner._recorder.ensure_layer_provenance(
            owner._extract_for_modality(stimuli, modality, all_layers), all_layers,
            description='Readout extraction')
        layer_of = extracted['layer'].values
        features = extracted.isel(neuroid=layer_of == readout_layer)
        if 'presentation' in features.dims and 'neuroid' in features.dims:
            features = features.transpose('presentation', 'neuroid')
        if raw and isinstance(self.task.response, Scale):
            behavior = [(float(v), None) for v in self.readout_classifier.predict(features)]
        else:
            behavior = _implied(self.readout_classifier.predict_proba(features), 'readout')
        neural = {}
        for region, region_layers in layers.items():
            per_layer = [extracted.isel(neuroid=layer_of == layer) for layer in region_layers]
            selected = self._gather(region, per_layer)
            neural[region] = selected.assign_coords(
                region=('neuroid', [region] * selected.sizes['neuroid']))
            if not raw:
                neural[region] = neural[region].assign_attrs(
                    recording=self._stamp('readout', 'readout extraction'))
        return behavior, neural

@contextmanager
def _hook_recording(model, layers, identifier):
    """Forward hooks on ``model`` for extractors without ``record()``.

    ``take()`` returns the first pass since the previous take, each layer's
    output flattened per presentation, as a still-image extractor packages it.
    """
    from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly

    def module_at(path):
        module = model
        for part in path.split('.'):
            module = module[int(part)] if part.isdigit() else getattr(module, part, None)
            if module is None:
                raise NotImplementedError(f'Layer {path!r} is not a module of the model passed as model=; '
                                          'give the extractor a record(layers) method.')
        return module

    modules = {layer: module_at(layer) for layer in layers}   # unknown paths fail before compute
    captured = {}

    def keep(layer):
        def hook(_module, _inputs, output):
            if layer not in captured:
                output = output[0] if isinstance(output, (tuple, list)) else output
                captured[layer] = np.asarray(output.detach().float().cpu().numpy())
        return hook

    handles = [module.register_forward_hook(keep(layer)) for layer, module in modules.items()]

    class Recording:
        @staticmethod
        def take():
            missing = [layer for layer in layers if layer not in captured]
            if missing:
                raise ValueError(f'No forward pass reached {missing} since the last presentation.')
            rows = {layer: captured[layer].reshape(1, -1) for layer in layers}
            captured.clear()
            values = np.concatenate([rows[layer] for layer in layers], axis=1)
            neuroid_layers = [layer for layer in layers for _ in range(rows[layer].shape[1])]
            numbers = [i for layer in layers for i in range(rows[layer].shape[1])]
            return NeuroidAssembly(values, dims=['presentation', 'neuroid'], coords={
                'stimulus_id': ('presentation', ['?']),
                'neuroid_id': ('neuroid', [f'{identifier}.{l}.{i}' for l, i in zip(neuroid_layers, numbers)]),
                'neuroid_num': ('neuroid', numbers), 'layer': ('neuroid', neuroid_layers)})
    try:
        yield Recording()
    finally:
        for handle in handles:
            handle.remove()


def _legacy_one_hot(answers, label_set, stimuli):
    """A legacy task's answers as one-hot rows over ``label_set``.

    An answer counts only when it is exactly a label; any other answer is
    recorded as invalid (NaN row, ``answer_valid=False``) with a warning, as
    legacy behavioral benchmarks expect. The coordinates keep their order:
    ``stimulus_id``, the other stimulus columns, ``raw_answer``, ``answer_valid``.
    """
    from brainscore_core.supported_data_standards.brainio.assemblies import BehavioralAssembly
    import warnings
    label_to_idx = {label: i for i, label in enumerate(label_set)}
    proba = np.zeros((len(stimuli), len(label_set)), dtype=np.float32)
    raw_answers, valid = [], []
    for i, predicted in enumerate(answers):
        raw_answers.append(str(predicted))
        valid.append(predicted in label_to_idx)
        if valid[-1]:
            proba[i, label_to_idx[predicted]] = 1.0
        else:
            # No label is credited: a substituted label would bias accuracy toward it.
            proba[i, :] = np.nan
            warnings.warn(
                f"answer function returned {predicted!r}, not in "
                f"label_set={label_set}. The response is recorded as invalid "
                f"(NaN probabilities, answer_valid=False)."
            )
    coords = {'stimulus_id': ('presentation', list(stimuli['stimulus_id'].values))}
    for column in stimuli.columns:
        if column != 'stimulus_id':
            coords[column] = ('presentation', list(stimuli[column].values))
    coords['raw_answer'] = ('presentation', raw_answers)
    coords['answer_valid'] = ('presentation', valid)
    return BehavioralAssembly(proba, coords={**coords, 'choice': label_set},
                              dims=['presentation', 'choice'],
                              attrs={'measure': {'kind': 'reported', 'source': 'answer'}})


def _presentation(stimuli):
    """The stimulus columns as presentation coordinates.

    A column whose cells are sequences (the options of a choice whose
    ``labels`` names a column) cannot be a coordinate and is left out.
    """
    return {column: ('presentation', list(stimuli[column].values)) for column in stimuli.columns
            if not any(isinstance(value, (list, tuple, set, dict, np.ndarray))
                       for value in stimuli[column].values)}


def _one_hot(chosen, valid, stimuli, row_labels, axis, source, raw=None, errors=None):
    """One row per stimulus, 1 for the chosen label; invalid rows are NaN (rule 6.2).

    ``row_labels`` are each row's options; ``axis`` names the choice axis (the
    labels, or option positions when the options differ for each row).
    """
    from brainscore_core.supported_data_standards.brainio.assemblies import BehavioralAssembly
    values = np.zeros((len(stimuli), len(axis)), dtype=np.float32)
    for i, (label, ok) in enumerate(zip(chosen, valid)):
        if ok:
            values[i, list(row_labels[i]).index(label)] = 1.0
        else:
            values[i] = np.nan
    coords = {**_presentation(stimuli), 'choice': list(axis),
              'answer_valid': ('presentation', list(valid)),
              'raw_answer': ('presentation', [str(r) for r in (raw or chosen)])}
    if errors is not None:
        coords['parse_error'] = ('presentation', [e or '' for e in errors])
    return BehavioralAssembly(values, coords=coords, dims=['presentation', 'choice'],
                              attrs={'measure': {'kind': 'reported' if source in ('reply', 'answer')
                                                 else 'implied', 'source': source}})


def _numbers(parsed, stimuli, source, raw=None):
    """One number per stimulus (a rating or a confidence); invalid ones are NaN."""
    from brainscore_core.supported_data_standards.brainio.assemblies import BehavioralAssembly
    values = np.array([np.nan if v is None else v for v, _ in parsed], dtype=float)
    coords = {**_presentation(stimuli),
              'answer_valid': ('presentation', [v is not None for v, _ in parsed]),
              'parse_error': ('presentation', [e or '' for _, e in parsed])}
    if raw is not None:
        coords['raw_answer'] = ('presentation', [str(r) for r in raw])
    return BehavioralAssembly(values, coords=coords, dims=['presentation'],
                              attrs={'measure': {'kind': 'reported' if source == 'reply'
                                                 else 'implied', 'source': source}})


def _reply_and_meta(reply):
    """A reply's text and its extra data: a reply returns the text, or a
    ``StreamEvent`` whose payload is the text and whose meta is the extra data."""
    if isinstance(reply, StreamEvent):
        return reply.payload, dict(reply.meta)
    return reply, {}


def _trials_meta(extras):
    """The meta of an answer event from the extra data of its trials.

    One trial: its extra data as is. Several trials: each key holds a list with
    one value per trial, in order (``None`` for a trial without that key).
    """
    if len(extras) == 1:
        return dict(extras[0])
    keys = list(dict.fromkeys(key for extra in extras for key in extra))
    return {key: [extra.get(key) for extra in extras] for key in keys}


def _texts(parsed, replies, stimuli):
    from brainscore_core.supported_data_standards.brainio.assemblies import BehavioralAssembly
    values = np.array([v if v is not None else '' for v, _ in parsed], dtype=object)
    coords = {**_presentation(stimuli),
              'answer_valid': ('presentation', [v is not None for v, _ in parsed])}
    return BehavioralAssembly(values, coords=coords, dims=['presentation'],
                              attrs={'measure': {'kind': 'reported', 'source': 'reply'}})


def _reaction_time(probabilities, link, stimuli):
    """An implied reaction time per stimulus from label probabilities (larger = slower)."""
    from brainscore_core.supported_data_standards.brainio.assemblies import BehavioralAssembly
    values = np.asarray(probabilities.values, dtype=float)
    ordered = np.sort(values, axis=1)[:, ::-1]
    if link == 'evidence_margin':
        rt = 1.0 - (ordered[:, 0] - ordered[:, 1])
    else:                                         # surprisal of the chosen (most probable) label
        rt = -np.log2(np.clip(ordered[:, 0], 1e-12, 1.0))
    return BehavioralAssembly(rt, dims=['presentation'], coords=_presentation(stimuli),
                              attrs={'measure': {'kind': 'implied', 'link': link,
                                                 'source': probabilities.attrs.get('measure', {}).get('source')}})


def _score_rows(scores, row_labels):
    """Each row's likelihood scores, in the order of that row's options."""
    return np.array([[float(row[label]) for label in options]
                     for row, options in zip(scores, row_labels)], dtype=float)


def _softmax(logp):
    logp = logp - logp.max(axis=1, keepdims=True)
    p = np.exp(logp)
    return p / p.sum(axis=1, keepdims=True)


def _implied_array(values, stimuli, axis, source, scale=None):
    """Implied values per option. Raw scores (``scale='log'``) keep full precision."""
    from brainscore_core.supported_data_standards.brainio.assemblies import BehavioralAssembly
    measure = {'kind': 'implied', 'source': source}
    if scale is not None:
        measure['scale'] = scale
    return BehavioralAssembly(values if scale is not None else values.astype(np.float32),
                              coords={**_presentation(stimuli), 'choice': list(axis)},
                              dims=['presentation', 'choice'], attrs={'measure': measure})


def _in_order(assembly, labels):
    """Readout columns in the task's label order, as ``behavior:choice`` has them."""
    present = [label for label in labels if label in set(assembly['choice'].values)]
    return assembly.sel(choice=present) if len(present) == assembly.sizes['choice'] else assembly


def _implied(assembly, source):
    return assembly.assign_attrs(measure={'kind': 'implied', 'source': source})


def _trials_to_assembly(trials, stimuli, layer: str, region: str):
    """Stack one-presentation recordings into a NeuroidAssembly for ``region``."""
    import numpy as np
    import pandas as pd
    from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly

    rows = []
    # A clip keeps its time steps and a multi-image trial its parts: one extra dimension.
    extra = [dim for dim in trials[0].dims if dim not in ('presentation', 'neuroid')]
    if len(extra) > 1:
        raise ValueError(f'A recording during a task may have one extra dimension; got {extra}.')
    timed = bool(extra)
    extra = extra[0] if extra else None
    for index, trial in enumerate(trials):
        selected = trial.isel(neuroid=trial['layer'].values == layer)
        if selected.sizes['neuroid'] == 0:
            raise ValueError(
                f'No activity reached layer {layer!r} while answering stimulus '
                f'{stimuli["stimulus_id"].iloc[index]!r}; that input may not use it.')
        if timed:
            if selected.sizes[extra] != trials[0].sizes[extra]:
                raise ValueError(f'Trials gave different numbers of {extra} steps; present trials of one '
                                 'length (clips of one duration, sequences of one length) together.')
            rows.append(np.asarray(selected.transpose('presentation', extra, 'neuroid').values)[0])
        else:
            rows.append(np.asarray(selected.values).reshape(-1))
    first = trials[0].isel(neuroid=trials[0]['layer'].values == layer)
    index = first.indexes.get('neuroid')
    names = list(index.names) if isinstance(index, pd.MultiIndex) else []
    names += [name for name, coord in first.coords.items()
              if coord.dims == ('neuroid',) and name != 'neuroid' and name not in names]
    coords = {name: ('neuroid', np.asarray(first[name].values)) for name in names}
    coords['region'] = ('neuroid', [region] * first.sizes['neuroid'])
    for column in stimuli.columns:
        coords[column] = ('presentation', list(stimuli[column].values))
    if not timed:
        return NeuroidAssembly(np.stack(rows), coords=coords, dims=['presentation', 'neuroid'])
    extra_index = first.indexes.get(extra)
    extra_names = list(extra_index.names) if isinstance(extra_index, pd.MultiIndex) else []
    extra_names += [name for name, coord in first.coords.items()
                    if coord.dims == (extra,) and name != extra and name not in extra_names]
    for name in extra_names:
        coords[name] = (extra, np.asarray(first[name].values))
    return NeuroidAssembly(np.stack(rows), coords=coords,
                           dims=['presentation', extra, 'neuroid'])


