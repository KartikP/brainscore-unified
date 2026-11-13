"""Concrete Brain-Score subject implementation."""

import threading
from contextlib import contextmanager
from copy import copy
from types import MappingProxyType
from collections.abc import Mapping
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Union

from ._model.behavioral import BehavioralReadout
from brainscore_core.contract import UnifiedModel, TaskContext
from ._model.dispatch import InputDispatcher
from brainscore_core.io_catalog import MODALITY_TO_INPUT_CHANNEL, canonical_modality
from brainscore_core.events import (
    InputEvent,
    OutputEvent,
    StateChange,
)
from ._model.intervention import Interventions
from ._model.recording import Recorder
from .selection import UnitSelector, _promote_to_selector
from brainscore_core.streaming import StreamEvent
from brainscore_core.streaming_helpers import (
    drain_session_inputs as _drain_neural_stream_events,
    session_outputs,
    session_stimuli as _neural_stimuli_from_stream_session,
)


# Models that are taking part in a session now: id -> the token of the
# invocation that reserved it. A model's recording, task and change state
# belong to the one session it takes part in, so a second session on the same
# model is refused instead of sharing that state.
_active_sessions: Dict[int, object] = {}
_active_sessions_lock = threading.Lock()


@contextmanager
def _one_session_at_a_time(model):
    """Refuse a second session on ``model`` while its first one is running.

    The cleanup is in place before the reservation is made, and it removes
    only this invocation's reservation (matched by its token). Thus the model
    is released however its session ends (a result, an exception or
    KeyboardInterrupt, also one that arrives just after the reservation), and
    a refused nested call never releases the session that is running.
    """
    key, token = id(model), object()
    try:
        with _active_sessions_lock:
            if key in _active_sessions:
                raise RuntimeError(
                    f'Subject {model.identifier!r} is already taking part in a session. '
                    'A subject takes part in one session at a time, because its '
                    'recording and task state belong to that session. Wait for the '
                    'running session to finish, or build a separate model for each '
                    'session that runs at the same time.')
            _active_sessions[key] = token
        yield
    finally:
        with _active_sessions_lock:
            if _active_sessions.get(key) is token:
                del _active_sessions[key]


class BrainScoreModel(UnifiedModel):
    """The one concrete subject, assembled from parts (extractors, region map,
    answer, reply, likelihood, policy, intervention, capabilities). Evaluated
    through ``participate``; ``process`` is the internal step the session drivers
    call. A legacy vision or language benchmark reaches it through
    :class:`~brainscore_core.legacy_facade.LegacyBenchmarkFacade`, which
    ``score()`` applies."""

    MODALITY_PRIORITY: Tuple[str, ...] = ('vision', 'text', 'audio')

    def __init__(
        self,
        identifier: str,
        model: Any = None,
        region_layer_map: Optional[Dict[str, Union[str, "UnitSelector"]]] = None,
        *,
        extractor: Optional[Mapping] = None,
        behavioral_readout_layer: Optional[str] = None,
        answer: Optional[Callable] = None,
        reply: Optional[Callable] = None,
        likelihood: Optional[Callable] = None,
        policy: Optional[Callable] = None,
        policy_inputs: Iterable[str] = ('observation',),
        policy_outputs: Iterable[str] = ('motor',),
        intervention: Optional[Callable] = None,
        memory: Optional[str] = None,
        supported_modalities: Optional[Set[str]] = None,
        required_modalities: Optional[Set[str]] = None,
        visual_degrees: int = 8,
        capability_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """A model (or, in principle, a human) evaluated through the unified interface.

        Only ``identifier`` is always required; a feature-extraction model also
        needs ``region_layer_map`` and an ``extractor`` (a modality -> extractor
        mapping). Every other part is optional; each one enables another
        evaluation path. All arguments after ``region_layer_map`` are keywords.

        :param identifier: unique model name.
        :param model: the object intervention hooks and recording during a task
            run against (e.g. the torch module); may be ``None`` for a
            feature-extraction model whose extractors hold their own modules.
        :param region_layer_map: brain region -> layer path (or a ``UnitSelector``).
            With several extractors, each region names its extractor with
            ``LayerSelector(name=..., modality=...)``.
        :param extractor: a modality -> extractor mapping. Each extractor is
            called as ``extractor(stimuli, layers=...)`` and prepares its own
            inputs. Image, text, audio and video wrappers all use this slot; the
            keys declare the input modalities (a ``video`` key takes clips). A
            fused extractor goes under a tuple of input types,
            ``{('vision', 'text'): fused}``: one forward pass over all of a
            stimulus's parts. It declares each type, and takes any stimulus that
            brings several of them at once. An extractor keys its own activation
            cache (``identifier`` and ``backbone_id`` of the extractor).
        :param behavioral_readout_layer: layer on which to fit a behavioral readout.
        :param answer: ``(stimulus_row, instruction, label_set) -> label``;
            generates and parses a behavioral answer for instruction-following
            tasks.
        :param reply: ``(prompt) -> str``; the model's raw reply to one trial.
            ``prompt`` is a ``brainscore_core.responses.Prompt``: the trial's parts
            in order (instruction, stimulus parts, the response's directive).
            Parsing is the task's. Gives choices, ratings, reported confidence
            and written answers. As a policy does, a reply may return a
            ``StreamEvent`` instead, with the text as payload and extra data
            (for example ``reasoning``, ``token_usage``) as meta; the meta goes
            into the answer event. When the event holds several trials, each
            key holds a list with one value per trial.
        :param likelihood: ``(prompt, candidates) -> {candidate: log
            probability}``; scores candidate replies without generating. Gives
            choices and their implied probabilities. Run the prompt on its own
            before any candidate, so a recording during scoring is the prompt pass.
        :param policy: ``row -> {output channel: payload}``; chooses an action for
            closed-loop / embodied evaluation. A row is a mapping of channel ->
            payload for one step; the policy needs no Brain-Score import. A value
            may be a ``StreamEvent`` to attach meta (telemetry). A policy that keeps
            episode state defines ``reset()``; it runs when an episode begins.
        :param policy_inputs: the input channels the policy reads (default
            ``('observation',)``); a session sends it only these.
        :param policy_outputs: the output channels the policy answers on (default
            ``('motor',)``); a benchmark may request any of them.
        :param intervention: ``StateChange -> (PerturbationApplied, cleanup)``;
            applies an intervention and returns its undo function. It declares
            the ``lesion``, ``stimulation`` and ``pharmacological`` inputs and
            the ``perturbation`` output; send changes with ``apply_state_change``.
        :param memory: ``'trial_history'``: the model keeps earlier trials, earlier
            steps and its responses from ``begin`` until ``end``; each trial's
            prompt and reply reach ``reply`` and ``likelihood`` as
            ``prompt.history``. It declares the ``memory:trial_history`` trait.
            A stateful extractor (a recurrent model) keeps its own state the same
            way. A ``begin`` clears that history and the episode state of every
            part; it leaves interventions and the task setup. Under the
            ``none`` trial memory every trial is its own episode, so
            the history is empty at each trial; under ``history`` it grows
            until ``end`` and never carries over to the next episode, session,
            repetition or condition. ``None`` (default): the model keeps no
            history, so every trial starts empty under either trial memory.
            ``reset()`` also clears the task setup and the interventions.
        :param supported_modalities: inputs accepted without an extractor, for
            example by a closed-weight ``answer`` function. Extractor mappings
            already declare their inputs; do not repeat them here.
        :param required_modalities: input modalities the model cannot run without.
        :param visual_degrees: the visual field legacy vision benchmarks place
            stimuli in; ignored by other benchmarks.
        :param capability_config: settings for added capabilities, keyed by the
            capability identifier; a capability reads it as
            ``subject.capability_config``.
        """
        if memory not in (None, 'trial_history'):
            raise ValueError("memory must be None or 'trial_history'")
        self._identifier_str = identifier
        self._model = model
        from ._model.dispatch import normalize_extractors
        self._fused_extractors = []     # (input types, extractor) registered under a tuple
        self._extractors = normalize_extractors(self, extractor)
        # The declared input keys, raw: a ``video`` key stays ``video`` and
        # declares ``vision:video``.
        self._input_keys: Set[str] = set(self._extractors) | set(supported_modalities or ())
        self._install_region_layer_map(region_layer_map or {})
        self._visual_degrees = visual_degrees
        self._capability_config = dict(capability_config or {})
        self._behavioral_readout_layer = behavioral_readout_layer
        self._answer, self._reply, self._likelihood = answer, reply, likelihood
        self._memory = memory
        self._policy = policy
        self._policy_inputs = frozenset(policy_inputs)
        self._policy_outputs = frozenset(policy_outputs)
        self._intervention = intervention

        self._recorder = Recorder(self)
        self._behavioral = BehavioralReadout(self)
        self._interventions = Interventions(self)
        self._dispatcher = InputDispatcher(self)
        self._capability_state: Dict[str, Any] = {}
        self._capability_setup_done: Set[str] = set()

        required = ({canonical_modality(m) for m in required_modalities}
                    if required_modalities else set())
        required_channels = self._input_channels(required_modalities or ())
        declared_channels = self._input_channels(self._input_keys)
        if not required_channels <= declared_channels:
            raise ValueError(
                f"required_modalities need the input channels {sorted(required_channels)}, "
                f"but the subject declares {sorted(declared_channels)}. A subject cannot "
                f"require an input it does not declare."
            )
        self._required_modalities = required
        self._required_keys = set(required_modalities or ())   # raw: 'video' stays 'video'

    @property
    def capability_config(self):
        """Read-only registration configuration for external capability authors."""
        return MappingProxyType(self._capability_config)

    @property
    def extractor(self) -> Mapping[str, Callable]:
        """Read-only modality -> extractor mapping, as normalized at construction.

        Named like the ``extractor`` constructor argument. For observation
        tools that hook the wired modules; dispatch does not read it.
        """
        return MappingProxyType(self._extractors)

    def capability_state(self, identifier):
        """Per-model capability state, initialized by the registry before dispatch."""
        return self._capability_state[identifier]

    def _declared_capabilities(self):
        from .capabilities import capability_registry
        return [c for c in capability_registry.values() if c.enabled_for(self)]

    @property
    def in_channels(self):
        # ``control``: the begin and respond signals clear episode state and group a closed-loop step.
        # A visual input names its form: a ``vision`` key takes stills (``vision:image``), a
        # ``video`` key takes clips (``vision:video``). The raw keys give the forms.
        return self._input_channels(self._input_keys) | {'control'} | set().union(
            *(c.declared_input_channels(self) for c in self._declared_capabilities()))

    @property
    def required_channels(self):
        return self._input_channels(self._required_keys)

    @staticmethod
    def _input_channels(modalities):
        return {MODALITY_TO_INPUT_CHANNEL.get(m, m) for m in modalities}

    @property
    def traits(self):
        return {'memory:trial_history'} if self._memory == 'trial_history' else set()

    @property
    def out_channels(self):
        # Every output comes from a registered part; nothing is found by name.
        return set().union(
            *(c.declared_output_channels(self) for c in self._declared_capabilities()))

    @property
    def output_inputs(self):
        """The inputs each ``neural:<region>`` can be answered from.

        A region lists the input channels of the extractor its selector names
        (``'video'`` takes clips, ``'vision'`` stills). In a model with one
        extractor, a region without a modality lists that extractor's channels.
        Behavior, policy and perturbation outputs are not listed, so they can
        be answered from any input the model takes.
        """
        listed = {}
        for region, selector in self._region_layer_selectors.items():
            keys = list(self._extractors)
            if selector.modality is not None:
                keys = ([key for key in keys if key == selector.modality]
                        or [key for key in keys if canonical_modality(key)
                            == canonical_modality(selector.modality)])
            channels = self._input_channels(keys)
            if channels:
                listed[f'neural:{region}'] = frozenset(channels)
        return listed

    def check_session_support(self, channels):
        """The one capability that runs a session requesting ``channels``.

        Raises before any input is consumed when no capability, or more than
        one at the same priority, claims the combination.
        """
        channels = frozenset(channels)
        if not channels:
            raise ValueError('A session must request at least one output channel.')
        claims = [c for c in self._declared_capabilities()
                  if c.supports_session(self, channels)]
        if not claims:
            if 'perturbation' in channels:
                raise NotImplementedError(
                    f"Subject {self.identifier!r} has no intervention part. An intervention "
                    f"needs BrainScoreModel(intervention=...), a callable "
                    f"intervention(state_change) -> (PerturbationApplied, undo_function).")
            raise NotImplementedError(f'No session handler for {sorted(channels)}')
        first = min(c.order for c in claims)
        winners = [c for c in claims if c.order == first]
        if len(winners) > 1:
            raise ValueError(f'Ambiguous session handlers: {[c.identifier for c in winners]}')
        return winners[0]

    @property
    def identifier(self) -> str:
        return self._identifier_str

    def _install_region_layer_map(self, region_layer_map) -> None:
        """Validate and install one lossless region configuration atomically."""
        selectors = {region: _promote_to_selector(value)
                     for region, value in dict(region_layer_map).items()}
        for region, selector in selectors.items():
            modality = getattr(selector, 'modality', None)
            if (modality is not None
                    and canonical_modality(modality) not in self.supported_modalities):
                raise ValueError(f"Region {region!r} references modalities with no extractor: {modality!r}")
        if len({id(extractor) for extractor in self._extractors.values()}) > 1:
            unrouted = sorted(r for r, sel in selectors.items() if sel.modality is None)
            if unrouted:
                # Without a named extractor the region would be recorded from
                # whichever tower gets input, and pre-flight could not check it.
                raise ValueError(
                    f"Subject {self._identifier_str!r} has several extractors, so each region must "
                    f"name the one it is recorded from. Regions with no modality: {unrouted}. "
                    f"Use LayerSelector(name=..., modality=...).")
        from ._model.recording import validate_region_layers
        validate_region_layers(self, selectors)
        self._region_layer_selectors = selectors
        self._region_layer_map_dict = {r: sel.layer_path for r, sel in selectors.items()}
        self._region_modality_map = {
            r: sel.modality for r, sel in selectors.items()
            if sel.modality is not None}

    @property
    def region_layer_map(self) -> Mapping[str, Union[str, UnitSelector]]:
        """Read-only, lossless recording configuration.

        Simple entries remain layer strings. Composite units and modality routing
        remain selectors, so assigning this mapping back preserves every setting.
        """
        return MappingProxyType({
            # A selector that its own layer path promotes to is shown as that path.
            region: (selector.layer_path
                     if _promote_to_selector(selector.layer_path) == selector else selector)
            for region, selector in self._region_layer_selectors.items()})

    @region_layer_map.setter
    def region_layer_map(self, region_layer_map) -> None:
        self._install_region_layer_map(region_layer_map)

    @property
    def region_layer_selectors(self) -> Dict[str, "UnitSelector"]:
        return dict(self._region_layer_selectors)

    @property
    def supported_modalities(self) -> Set[str]:
        # a ``video`` key reports as ``vision`` (temporal vision)
        return {canonical_modality(m) for m in self._input_keys}

    @property
    def required_modalities(self) -> Set[str]:
        return set(self._required_modalities)

    def process(self, input_event: InputEvent,
                multi_modality: bool = False) -> OutputEvent:
        """Evaluate one input event and return the corresponding output event.

        The step the session drivers call; not part of the subject contract
        (``participate`` is the evaluation method). The capability registry
        routes the event to the first capability that handles it, the same
        registry ``participate`` reads: a ``StimulusSet`` / ``MultimodalStimulusSet`` -> ``NeuroidAssembly`` of
        recorded activations (call ``start_recording`` first); a row (a
        mapping of channel -> payload) -> outputs mapping (needs ``policy``).
        The intervention session driver also hands a ``StateChange`` to this
        step; send an intervention with ``apply_state_change`` or the
        ``Intervene`` tool, never through ``process``.
        Native benchmarks evaluate subjects through sessions or the typed
        helpers in ``brainscore_core.streaming_helpers``; ``process`` is not
        available on every subject.

        :param input_event: the stimuli or the row to run.
        :param multi_modality: if True, extract from every detected-and-supported
            modality and concat along the neuroid axis; default routes a single
            modality by ``MODALITY_PRIORITY``.
        """
        output = self._dispatcher.process(input_event, multi_modality)
        time_bins = self._recorder.time_bins
        if time_bins is not None and len(time_bins) > 1 and not isinstance(
                input_event, (StateChange, Mapping)):
            # Honor every requested time bin: a static response is repeated
            # across several bins, as legacy vision models do.
            from brainscore_core.legacy_facade import repeat_over_time_bins
            output = repeat_over_time_bins(output, time_bins)
        from brainscore_core.supported_data_standards.brainio.assemblies import BehavioralAssembly
        if isinstance(output, BehavioralAssembly):
            from brainscore_core.score_metadata import note_measure
            note_measure(output)
        return output

    def participate(self, session) -> None:
        """Run a session through the one registered capability that claims it.

        The handler sees only the streams this model declares in
        ``in_channels``; every other stream the experiment sends is skipped
        before it reaches any part (Contract Rules v3, rule 5.5).
        """
        with _one_session_at_a_time(self):
            handler = self.check_session_support(session_outputs(session))
            from .capabilities.registry import _setup_once
            _setup_once(self, handler)
            # A task the session configures ends with the session (rule 4.5).
            with self._behavioral.session_scope():
                handler.participate(self, session.declare_inputs(self.in_channels))

    def detect_modalities(self, stimuli) -> Set[str]:
        """Declared input modalities whose columns appear in ``stimuli``."""
        return self._dispatcher.detect_modalities(stimuli)

    def _pick_modality(self, detected: Set[str]) -> str:
        return self._dispatcher.pick_modality(detected)

    def _filter_layers_for_modality(self, layers: List[str],
                                    modality: str) -> List[str]:
        return self._recorder.filter_layers_for_modality(layers, modality)

    def _supports_layer_extraction(self, modality: str) -> bool:
        return self._dispatcher.supports_layer_extraction(modality)

    def _extract_for_modality(self, stimuli, modality: str, layers: List[str]):
        return self._dispatcher.extract_for_modality(stimuli, modality, layers)

    def _process_multi_modality(self, stimuli, detected: Set[str],
                                layers: List[str]):
        return self._dispatcher.process_multi_modality(stimuli, detected, layers)

    def _tag_neuroids_with_regions(self, assembly, modality=None):
        return self._recorder.tag_neuroids_with_regions(assembly, modality)

    def _process_composite_regions(self, stimuli, modality):
        return self._recorder.process_composite_regions(stimuli, modality)

    def start_recording(self,
                        recording_target: Union[str, List[str]],
                        time_bins: Optional[List[Tuple[int, int]]] = None,
                        recording_type: Optional[str] = None) -> None:
        """Configure neural recording for subsequent ``process`` calls.

        ``recording_target`` is a mapped region name, a list of
        mapped regions, or a raw layer path. Multiple regions are extracted in
        one call and returned with per-neuroid ``layer`` and ``region``
        provenance. See :class:`Recorder` for validation details.
        """
        self._recorder.start_recording(recording_target, time_bins,
                                       recording_type)

    def start_task(self, task, fitting_stimuli=None,
                   **kwargs) -> None:
        """Configure the behavioral task.

        Fitting stimuli reach the readout with only the input modalities this
        model declares (rule 5.5); the caller's table is never changed. The
        same TaskContext object sent again, as a facade session does for every
        measurement, keeps the fitted readout; a new task, an edited one,
        or any task after an intervention or reset, fits anew. The last
        configuration that completed is recalled for the same unchanged
        task after a session gave the caller's task back.
        """
        from dataclasses import replace
        from brainscore_core.session_view import declared_only
        if isinstance(task, TaskContext):
            if (self._behavioral.is_configured_for(task)
                    or self._behavioral.recall(task)):
                return
            fitting = task.fitting_stimuli
            projected = declared_only(fitting, self.in_channels)
            declared = (task if projected is fitting
                        else replace(task, fitting_stimuli=projected))
            try:
                self._behavioral.start_task(declared, **kwargs)
            except BaseException:
                self._behavioral.abandon()   # a half-built task is never reused
                raise
            self._behavioral.mark_configured(task)
            return
        try:
            self._behavioral.start_task(task,
                                        declared_only(fitting_stimuli, self.in_channels),
                                        **kwargs)
        except BaseException:
            self._behavioral.abandon()
            raise
        # The task built from (task type, fitting) is what a facade sends back.
        self._behavioral.mark_configured(self._behavioral.task)

    def reset(self) -> None:
        failures = []
        for component in (self._recorder, self._behavioral, self._interventions):
            try:
                component.reset()
            except Exception as error:
                failures.append(error)
        for identifier, state in self._capability_state.items():
            from .capabilities import capability_registry
            capability = capability_registry.get(identifier)
            if capability is not None:
                try:
                    capability.reset(self, state)
                    self._capability_setup_done.discard(identifier)
                except Exception as error:
                    failures.append(error)
        failures.extend(self._reset_providers())
        if failures:
            raise RuntimeError('A model provider failed to reset; the model is not clean.') from failures[0]

    def _reset_providers(self) -> list:
        """Clear the episode state of every part once; return the failures.

        Callable capabilities and extractors may own episode/history state.
        A bound-method registration resets its owner.
        """
        providers = [self._policy, self._answer, self._reply, self._likelihood,
                     *self._extractors.values()]
        seen, failures = set(), []
        for provider in providers:
            try:
                self._reset_provider(provider, seen)
            except Exception as error:
                failures.append(error)
        return failures

    def _begin_episode(self) -> None:
        """What a ``begin`` does: history ``h`` becomes empty; weights, interventions
        and the task setup stay. Clears the kept trials and the episode state of the
        parts (policy, extractors, reply and likelihood functions). The one method
        every session kind calls when an episode begins."""
        self._behavioral.history = []
        failures = self._reset_providers()
        if failures:
            raise RuntimeError('A model provider failed to clear its episode state.') from failures[0]

    def _reset_provider(self, provider, seen: Set[int]) -> None:
        """Reset one provider (a callable part or extractor) once; a bound method resets its owner."""
        provider = getattr(provider, '__self__', provider)
        if provider is None or provider is self or id(provider) in seen:
            return
        seen.add(id(provider))
        reset = getattr(provider, 'reset', None)
        if callable(reset):
            reset()

    def visual_degrees(self) -> int:
        """The visual field this subject declares (``visual_degrees=``).

        Read at the scoring boundary (``declared_visual_degrees``) and given to
        the legacy benchmark facade, which places vision stimuli by it.
        """
        return self._visual_degrees
