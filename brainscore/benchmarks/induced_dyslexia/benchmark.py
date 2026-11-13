"""Induced dyslexia — a perturbation benchmark (Honarmand et al. 2026 ICLR).

Causal test, not a correlation: localize the model's word-form-selective units
(an fMRI-localizer analogue), switch them off, and measure the reading deficit —
against a matched random-ablation control that rules out "the model just got
damaged". This converts an induced-dyslexia top-K ablation sweep into a
registered Brain-Score benchmark
built on the core ``UnitSelection`` family:

  * :class:`FunctionalSelection` — the real-vs-pseudo Cohen's d localizer,
                                   resolved per reported layer
  * :class:`CompositeSelector`   — a multi-layer word-form region (the subject  
                                   declares it in ``region_layer_map``); localized
                                   per reported layer here
  * :class:`RandomSelection`     — the same-size random lesion at the same layer(s):
                                   the specificity control
  * ``StateChange(kind='ablation', perturbation='zero')`` — the lesion, sent
                                   with ``apply_state_change``

The reading test is the ROAR lexical-decision benchmark, reused unchanged. The
benchmark runs any subject through ``participate`` only (Contract Rules 1.1,
2.1): it records ``neural:<region>`` for the localizer, sends the lesions as
``lesion`` events and reads through ``behavior``. It requests those outputs, so
a subject that cannot record the region or report a lesion (no ``perturbation``
output) is N/A in pre-flight; one that does not take ``lesion`` input is N/A at
the start of the call, before any compute. Reading is cleanest on the
generation path (instruction-following), matching the paper; a readout-path
model re-fits its linear readout each call (a valid but different measure).

Units are addressed by what the subject reports: the benchmark localizes per
value of the recording's ``layer`` coord and addresses each lesion by it. The
benchmark undoes only its own lesions, by handle; changes the caller applied
before scoring stay in place, and the whole run measures the subject as given.

Score = lesioned reading accuracy (lower => more dyslexic). attrs carry the
baseline accuracy, the random-ablation control, the *specific* deficit
(``random - lesioned``), and a ``dyslexic`` flag: True only when reading falls
below the 0.65 threshold AND specifically below the random control (so a model
that simply breaks under any ablation does NOT read as selective dyslexia).

Localizer note: Honarmand et al. localize with word-vs-non-word *images* (line
drawings / scrambled). Our ROAR corpus has only real/pseudo *words*, so the
contrast here is real-vs-pseudo — the discriminator-unit variant documented in
the project's K-sweep finding. Different localizer, same API.
"""
from typing import List, Sequence

from brainscore_core.benchmarks import BenchmarkBase
from brainscore_core.compatibility import CompatibilityError
from brainscore_core.metrics import Score
from brainscore import FunctionalSelection, RandomSelection
from brainscore_core.model_interface import StateChange, Selection, Perturbation
from brainscore_core.streaming_helpers import apply_state_change
from brainscore.benchmarks._score import benchmark_score

from ..roar_yeatman2021.benchmark import (
    Yeatman2021LexicalDecision, DYSLEXIA_THRESHOLD, BIBTEX,
)


class InducedDyslexia(BenchmarkBase):
    """Ablate word-form units → measure the reading deficit vs a random control.

    :param localizer_region: the region (key in the subject's
        ``region_layer_map``) whose units are word-form-selective. May map to a
        single layer or be a :class:`CompositeSelector` over several layers
        (localized per layer the recording reports).
    :param n_units: how many top-selective units to ablate per layer.
    :param control_seed: seed for the matched :class:`RandomSelection` control.
    :param channels: ``('vision:image',)`` (word images) or ``('text',)`` (word strings).
    """

    # One lesion experiment is one pass; the reading trials inside it are independent.
    trial_memory = 'none'
    # Suffix of the identifier for each input channel.
    _CHANNEL_SUFFIX = {'vision:image': 'vision', 'text': 'text'}

    def __init__(self, localizer_region: str = 'VWFA', n_units: int = 500,
                 control_seed: int = 0, channels: Sequence[str] = ('vision:image',),
                 reading_benchmark=None, localizer_stimuli=None):
        self.localizer_region = localizer_region
        self.n_units = n_units
        self.control_seed = control_seed
        channels = tuple(channels)
        if len(channels) != 1 or channels[0] not in self._CHANNEL_SUFFIX:
            raise ValueError(
                f"channels must be one of {[(c,) for c in self._CHANNEL_SUFFIX]}; "
                f"got {channels!r}")
        channel = channels[0]
        # Declarations decide compatibility (rule 5): the benchmark sends the
        # words, the reading question and the lesions, and needs the region's activity, the lesion
        # record and the reading answers back. A subject without any of these
        # outputs is N/A before any compute.
        self.required_input_channels = frozenset(
            {channel, 'instruction', 'lesion'})
        self.requested_output_channels = frozenset(
            {'behavior', f'neural:{localizer_region}', 'perturbation'})

        # The reading test (reused unchanged) and the localizer stimuli (its
        # train split, which carries the real/pseudo contrast label).
        # Both injectable for tests so the unit suite needn't load ROAR data.
        if reading_benchmark is not None:
            self._reading = reading_benchmark
            self._localizer_stimuli = localizer_stimuli
        else:
            self._reading = Yeatman2021LexicalDecision(channels=channels)
            self._localizer_stimuli = self._reading._train_stimuli

        super().__init__(
            identifier=f'Yeatman2021-induced_dyslexia-{self._CHANNEL_SUFFIX[channel]}',
            version=1,
            parent='perturbation',
            ceiling=Score(1.0),   # ceiling is "no deficit"; the floor is the chance reader
            bibtex=BIBTEX,
        )

    # -- localization -------------------------------------------------------
    def _localize(self, subject) -> List[Selection]:
        """Return the lesion targets: one top-``n_units`` real-vs-pseudo set per layer.

        The :class:`FunctionalSelection` localizer records the region once
        through ``participate``, localizes each layer the recording reports in
        its ``layer`` coord (Cohen's d, highest first), and addresses each
        lesion by that reported layer.
        """
        return FunctionalSelection(
            self.localizer_region, self._localizer_stimuli,
            contrast=(['real'], ['pseudo']), contrast_column='image_label',
            n_units=self.n_units).resolve_layers(subject)

    # -- scoring ------------------------------------------------------------
    def _reading_accuracy(self, subject) -> float:
        return float(self._reading(subject).attrs['raw'])

    def _ablate(self, subject, selections: List[Selection], handles: List[str]) -> None:
        """Apply one zeroing lesion per selection, appending each handle as it lands.

        A failure part-way leaves the handles of the lesions already applied in
        ``handles``, so the caller can undo exactly those.
        """
        for sel in selections:
            record = apply_state_change(subject, StateChange(
                kind='ablation', target=sel, perturbation=Perturbation(kind='zero')))
            handles.append(record.handle_id)

    @staticmethod
    def _undo(subject, handles: List[str]) -> None:
        """Undo the benchmark's own lesions, newest first, by handle (rule 8.2).

        Changes the caller applied before scoring are not touched (rule 8.3).
        Every undo is attempted even when an earlier one raises, so one failed
        cleanup never leaves the other lesions in place; the first error is
        raised after all attempts. A failed handle stays registered in core,
        so a later ``reset()`` can retry it.
        """
        errors = []
        for handle in reversed(handles):
            try:
                apply_state_change(subject, StateChange(kind='reset', handle_id=handle))
            except Exception as error:                       # noqa: BLE001
                errors.append(error)
        handles.clear()
        if errors:
            raise errors[0]

    def __call__(self, subject) -> Score:
        # The subject must declare `lesion` to receive the lesions.
        # If it does not, the result is N/A before any compute.
        if 'lesion' not in set(subject.in_channels):
            raise CompatibilityError(
                f"Subject '{subject.identifier}' does not declare the 'lesion' family, "
                f"which benchmark '{self.identifier}' needs to apply and undo lesions; in_channels: "
                f"{sorted(subject.in_channels)}.")
        return self._score(subject)

    def _score(self, subject) -> Score:
        baseline_acc = self._reading_accuracy(subject)

        # word-form lesion
        selections = self._localize(subject)
        handles: List[str] = []
        try:
            self._ablate(subject, selections, handles)
            lesioned_acc = self._reading_accuracy(subject)
        finally:
            self._undo(subject, handles)

        # matched random-ablation control
        controls = [RandomSelection(layer=s.layer, n_units=len(s.indices),
                                    axis=s.axis, expected_shape=s.expected_shape,
                                    n_total=int(s.metadata['n_recorded']),
                                    seed=self.control_seed,
                                    population=s.metadata.get('unit_population')).resolve(subject)
                    for s in selections]
        try:
            self._ablate(subject, controls, handles)
            random_acc = self._reading_accuracy(subject)
        finally:
            self._undo(subject, handles)

        specific_deficit = random_acc - lesioned_acc      # >0 ⇒ word-form-specific
        dyslexic = bool(lesioned_acc < DYSLEXIA_THRESHOLD and specific_deficit > 0)

        # The score is the lesioned reading accuracy; the ceiling (no deficit)
        # is a reference and does not divide it.
        score = benchmark_score(self, lesioned_acc, raw=lesioned_acc,
                                metric='lesioned_reading_accuracy', normalized=False)
        score.attrs['baseline_accuracy'] = baseline_acc
        score.attrs['lesioned_accuracy'] = lesioned_acc
        score.attrs['random_control_accuracy'] = random_acc
        score.attrs['deficit'] = baseline_acc - lesioned_acc
        score.attrs['specific_deficit'] = specific_deficit
        score.attrs['dyslexia_threshold'] = DYSLEXIA_THRESHOLD
        score.attrs['dyslexic'] = dyslexic
        score.attrs['n_units_per_layer'] = self.n_units
        score.attrs['n_layers_ablated'] = len(selections)
        return score
