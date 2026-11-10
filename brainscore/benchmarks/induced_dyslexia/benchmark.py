"""Induced dyslexia — a perturbation benchmark (Honarmand et al. 2026 ICLR).

Causal test, not a correlation: localize the model's word-form-selective units
(an fMRI-localizer analogue), switch them off, and measure the reading deficit —
against a matched random-ablation control that rules out "the model just got
damaged". This converts an induced-dyslexia top-K ablation sweep into a
registered Brain-Score benchmark
built on the core ``UnitSelection`` family:

  * the real-vs-pseudo Cohen's d localizer (the :class:`FunctionalSelection`
                                   statistic, computed per reported layer)
  * :class:`CompositeSelector`   — a multi-layer word-form region (the candidate
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
from typing import List

import numpy as np

from brainscore_core.benchmarks import BenchmarkBase
from brainscore_core.compatibility import CompatibilityError
from brainscore_core.io_catalog import modalities_to_input_channels
from brainscore_core.metrics import Score
from brainscore_core.selection import selection_unit_indices, selection_coordinates
from brainscore_core.model_interface import (
    StateChange, Selection, Perturbation, RandomSelection,
)
from brainscore_core.streaming_helpers import apply_state_change, neural_response

from ..roar_yeatman2021.benchmark import (
    Yeatman2021LexicalDecision, DYSLEXIA_THRESHOLD, BIBTEX,
)


class InducedDyslexia(BenchmarkBase):
    """Ablate word-form units → measure the reading deficit vs a random control.

    :param localizer_region: the region (key in the candidate's
        ``region_layer_map``) whose units are word-form-selective. May map to a
        single layer or be a :class:`CompositeSelector` over several layers
        (localized per layer the recording reports).
    :param n_units: how many top-selective units to ablate per layer.
    :param control_seed: seed for the matched :class:`RandomSelection` control.
    :param modality: ``'vision'`` (word images) or ``'text'`` (word strings).
    """

    def __init__(self, localizer_region: str = 'VWFA', n_units: int = 500,
                 control_seed: int = 0, modality: str = 'vision',
                 reading_benchmark=None, localizer_stimuli=None):
        self.localizer_region = localizer_region
        self.n_units = n_units
        self.control_seed = control_seed
        self.modality = modality
        self.required_modalities = {modality}
        # Declarations decide compatibility (rule 5): the benchmark sends the
        # words and the lesions, and needs the region's activity, the lesion
        # record and the reading answers back. A subject without any of these
        # outputs is N/A before any compute.
        self.required_input_channels = frozenset(
            modalities_to_input_channels({modality}) | {'lesion'})
        self.requested_output_channels = frozenset(
            {'behavior', f'neural:{localizer_region}', 'perturbation'})

        # The reading test (reused unchanged) and the localizer stimuli (its
        # train split, which carries the real/pseudo contrast label).
        # Both injectable for tests so the unit suite needn't load ROAR data.
        if reading_benchmark is not None:
            self._reading = reading_benchmark
            self._localizer_stimuli = localizer_stimuli
        else:
            self._reading = Yeatman2021LexicalDecision(modality=modality)
            self._localizer_stimuli = self._reading._train_stimuli

        super().__init__(
            identifier=f'Yeatman2021-induced_dyslexia-{modality}',
            version=1,
            parent='perturbation',
            ceiling=Score(1.0),   # ceiling is "no deficit"; the floor is the chance reader
            bibtex=BIBTEX,
        )

    # -- localization -------------------------------------------------------
    def _localize(self, candidate) -> List[Selection]:
        """Return the lesion targets: one top-``n_units`` real-vs-pseudo set per layer.

        Records the region once through ``participate``, localizes each layer
        the recording reports in its ``layer`` coord (Cohen's d, highest
        first), and addresses each lesion by that reported layer.
        """
        asm = neural_response(candidate, self._localizer_stimuli,
                              record=self.localizer_region)
        extra = [d for d in asm.dims if d not in ('presentation', 'neuroid')]
        if extra:
            asm = asm.mean(extra)
        asm = asm.transpose('presentation', 'neuroid')
        try:
            layer_coord = np.asarray(asm['layer'].values).ravel()
        except KeyError:
            raise ValueError(
                f"induced-dyslexia addresses each lesion by the layer the recording "
                f"reports; the '{self.localizer_region}' recording has no 'layer' coord "
                f"on its neuroids.") from None
        # Labels come from the benchmark's own stimuli, matched by stimulus_id,
        # so a subject need not echo stimulus columns back.
        label_of = dict(zip(self._localizer_stimuli['stimulus_id'].values,
                            self._localizer_stimuli['image_label'].values))
        labels = np.asarray([label_of[sid] for sid in asm['stimulus_id'].values])
        vals = asm.values
        pos = labels == 'real'
        neg = labels == 'pseudo'
        out = []
        for L in dict.fromkeys(str(layer) for layer in layer_coord):
            cols = np.where(layer_coord == L)[0]
            sub = vals[:, cols]
            mp, mn = sub[pos].mean(0), sub[neg].mean(0)
            sp, sn = sub[pos].std(0, ddof=1), sub[neg].std(0, ddof=1)
            pooled = np.sqrt((sp ** 2 + sn ** 2) / 2.0)
            d = np.divide(mp - mn, pooled, out=np.zeros_like(mp, dtype=float),
                          where=pooled > 0)
            order = np.argsort(d)[::-1][:self.n_units]
            population = selection_unit_indices(
                asm.isel(neuroid=cols), candidate, self.localizer_region)
            axis, expected_shape = selection_coordinates(asm.isel(neuroid=cols))
            out.append(Selection(layer=L, indices=sorted(int(i) for i in population[order]),
                                 axis=axis, expected_shape=expected_shape,
                                 metadata={'selector': 'functional',
                                           'n_recorded': int(sub.shape[1]),
                                           'unit_population': population.tolist()}))
        return out

    # -- scoring ------------------------------------------------------------
    def _reading_accuracy(self, candidate) -> float:
        return float(self._reading(candidate).attrs['raw'])

    def _ablate(self, candidate, selections: List[Selection], handles: List[str]) -> None:
        """Apply one zeroing lesion per selection, appending each handle as it lands.

        A failure part-way leaves the handles of the lesions already applied in
        ``handles``, so the caller can undo exactly those.
        """
        for sel in selections:
            record = apply_state_change(candidate, StateChange(
                kind='ablation', target=sel, perturbation=Perturbation(kind='zero')))
            handles.append(record.handle_id)

    @staticmethod
    def _undo(candidate, handles: List[str]) -> None:
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
                apply_state_change(candidate, StateChange(kind='reset', handle_id=handle))
            except Exception as error:                       # noqa: BLE001
                errors.append(error)
        handles.clear()
        if errors:
            raise errors[0]

    def __call__(self, candidate) -> Score:
        # Pre-flight cannot require that a subject *accepts* an input: inputs
        # need only overlap (rule 5.2) and `perturbation` does not say which
        # change families the subject takes. Check the declaration here, before
        # any compute, so a subject that cannot take a lesion is N/A.
        #
        # A subject receives an event when its in_channels hold the exact
        # channel or its bare family (core DeclaredInputSession). The lesion
        # channels sent here are `lesion:<layer>[<start>:<stop>]` and, to undo,
        # `lesion:<handle>`. Neither address is known before compute, so only
        # the bare `lesion` family accepts them all; an addressed declaration
        # such as `lesion:other` would drop the lesion or its undo.
        # ponytail: a subject that lists every possible lesion address one by
        # one would also work but is refused here; accept addressed lists only
        # if such a subject appears.
        if 'lesion' not in set(candidate.in_channels):
            raise CompatibilityError(
                f"Subject '{candidate.identifier}' does not declare the 'lesion' family, "
                f"which benchmark '{self.identifier}' needs to apply and undo lesions at "
                f"addresses fixed only after localization; in_channels: "
                f"{sorted(candidate.in_channels)}.")
        return self._score(candidate)

    def _score(self, candidate) -> Score:
        baseline_acc = self._reading_accuracy(candidate)

        # word-form lesion
        selections = self._localize(candidate)
        handles: List[str] = []
        try:
            self._ablate(candidate, selections, handles)
            lesioned_acc = self._reading_accuracy(candidate)
        finally:
            self._undo(candidate, handles)

        # matched random-ablation control
        controls = [RandomSelection(layer=s.layer, n_units=len(s.indices),
                                    axis=s.axis, expected_shape=s.expected_shape,
                                    n_total=int(s.metadata['n_recorded']),
                                    seed=self.control_seed,
                                    population=s.metadata.get('unit_population')).resolve(candidate)
                    for s in selections]
        try:
            self._ablate(candidate, controls, handles)
            random_acc = self._reading_accuracy(candidate)
        finally:
            self._undo(candidate, handles)

        specific_deficit = random_acc - lesioned_acc      # >0 ⇒ word-form-specific
        dyslexic = bool(lesioned_acc < DYSLEXIA_THRESHOLD and specific_deficit > 0)

        score = Score(lesioned_acc)
        score.attrs['raw'] = Score(lesioned_acc)
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
