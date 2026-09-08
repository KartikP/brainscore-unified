"""Induced dyslexia — a perturbation benchmark (Honarmand et al. 2026 ICLR).

Causal test, not a correlation: localize the model's word-form-selective units
(an fMRI-localizer analogue), switch them off, and measure the reading deficit —
against a matched random-ablation control that rules out "the model just got
damaged". This converts an induced-dyslexia top-K ablation sweep into a
registered Brain-Score benchmark
built on the core ``UnitSelection`` family:

  * :class:`FunctionalSelection` — the real-vs-pseudo Cohen's d localizer (one layer)
  * :class:`CompositeSelector`   — a multi-layer word-form region (the candidate
                                   declares it in ``region_layer_map``); localized
                                   per-layer here
  * :class:`RandomSelection`     — the same-size random lesion at the same layer(s):
                                   the specificity control
  * ``StateChange(kind='ablation', perturbation='zero')`` — the lesion, applied
                                   through the candidate's ``state_change_fn``

The reading test is the ROAR lexical-decision benchmark, reused unchanged. The
candidate must expose a ``state_change_fn`` (e.g.
``brainscore.perturbation.build_pytorch_ablation_fn``) and a word-form region in
its ``region_layer_map``. Reading is cleanest on the generation path
(instruction-following), matching the paper; a readout-path model re-fits its
linear readout each call (a valid but different measure — flagged in attrs).

Score = lesioned reading accuracy (lower ⇒ more dyslexic). attrs carry the
baseline accuracy, the random-ablation control, the *specific* deficit
(``random − lesioned``), and a ``dyslexic`` flag — True only when reading falls
below the 0.65 threshold AND specifically below the random control (so a model
that simply breaks under any ablation does NOT read as selective dyslexia).

Localizer note: Honarmand et al. localize with word-vs-non-word *images* (line
drawings / scrambled). Our ROAR corpus has only real/pseudo *words*, so the
contrast here is real-vs-pseudo — the discriminator-unit variant documented in
the project's K-sweep finding. Different localizer, same API.
"""
import dataclasses
import os
from typing import List, Optional

import numpy as np

from brainscore_core.benchmarks import BenchmarkBase
from brainscore_core.metrics import Score
from brainscore_core.selection import selection_unit_indices
from brainscore_core.model_interface import (
    StateChange, Selection, Perturbation, FunctionalSelection, RandomSelection,
)

from ..roar_yeatman2021.benchmark import (
    Yeatman2021LexicalDecision, DYSLEXIA_THRESHOLD, BIBTEX,
)


class InducedDyslexia(BenchmarkBase):
    """Ablate word-form units → measure the reading deficit vs a random control.

    :param localizer_region: the region (key in the candidate's
        ``region_layer_map``) whose units are word-form-selective. May map to a
        single layer (uses :class:`FunctionalSelection`) or be a
        :class:`CompositeSelector` over several layers (localized per layer).
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
            version=2,
            parent='perturbation',
            ceiling=Score(1.0),   # ceiling is "no deficit"; the floor is the chance reader
            bibtex=BIBTEX,
        )

    # -- localization -------------------------------------------------------
    def _region_layers(self, candidate) -> List[str]:
        selectors = candidate.region_layer_selectors
        if self.localizer_region not in selectors:
            raise ValueError(
                f"induced-dyslexia needs a '{self.localizer_region}' region in the "
                f"candidate's region_layer_map; have {list(selectors)}")
        sel = selectors[self.localizer_region]
        if hasattr(sel, 'layer_paths'):          # CompositeSelector
            return list(sel.layer_paths)
        return [sel.layer_path]                   # LayerSelector

    def _localize(self, candidate, layers: List[str]) -> List[Selection]:
        """Return one top-``n_units`` real-vs-pseudo Selection per layer.

        Single-layer region → idiomatic :class:`FunctionalSelection`. Composite
        region → record once and compute the same Cohen's d per layer (the core
        FunctionalSelection is single-layer only).
        """
        # Clear any behavioral task left active by a prior reading call — otherwise
        # process() takes the behavioral path and the localizer records choices,
        # not neuroids. (No perturbation is active yet at localization time.)
        candidate.reset()
        if len(layers) == 1:
            fs = FunctionalSelection(
                recording_target=self.localizer_region,
                localizer_stimuli=self._localizer_stimuli,
                contrast=(['real'], ['pseudo']),
                contrast_column='image_label',
                n_units=self.n_units, sign='positive')
            sel = fs.resolve(candidate)
            candidate.reset()
            # FunctionalSelection labels Selection.layer from the recorded 'layer'
            # coord; when the assembly carries no such coord it falls back to the
            # region name. The ablation needs the actual module path, so remap via
            # the candidate's region_layer_map.
            actual = candidate.region_layer_map.get(self.localizer_region)
            if actual and sel.layer != actual:
                sel = dataclasses.replace(sel, layer=actual)
            return [sel]
        # composite: record the whole region, split by the 'layer' coord
        candidate.start_recording(self.localizer_region)
        asm = candidate.process(self._localizer_stimuli)
        candidate.reset()
        extra = [d for d in asm.dims if d not in ('presentation', 'neuroid')]
        if extra:
            asm = asm.mean(extra)
        asm = asm.transpose('presentation', 'neuroid')
        labels = np.asarray(asm['image_label'].values)
        layer_coord = np.asarray(asm['layer'].values).ravel()
        vals = asm.values
        pos = labels == 'real'
        neg = labels == 'pseudo'
        out = []
        for L in layers:
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
            out.append(Selection(layer=L, indices=sorted(int(i) for i in population[order]),
                                 metadata={'selector': 'functional',
                                           'n_recorded': int(sub.shape[1]),
                                           'unit_population': population.tolist()}))
        return out

    # -- scoring ------------------------------------------------------------
    def _reading_accuracy(self, candidate) -> float:
        return float(self._reading(candidate).attrs['raw'])

    def _ablate(self, candidate, selections: List[Selection]) -> None:
        for sel in selections:
            candidate.process(StateChange(
                kind='ablation', target=sel, perturbation=Perturbation(kind='zero')))

    def __call__(self, candidate) -> Score:
        if getattr(candidate, '_state_change_fn', None) is None:
            raise ValueError(
                "induced-dyslexia requires a candidate with a state_change_fn "
                "(e.g. brainscore.perturbation.build_pytorch_ablation_fn).")
        # Keep live extraction for every condition, including third-party
        # candidates whose caches may not fingerprint perturbation state.
        # Built-in wrappers now fingerprint hooks, but that is not a requirement
        # of every Subject accepted by this benchmark.
        _prev_cache = os.environ.get('RESULTCACHING_DISABLE')
        os.environ['RESULTCACHING_DISABLE'] = '1'
        try:
            return self._score(candidate)
        finally:
            if _prev_cache is None:
                os.environ.pop('RESULTCACHING_DISABLE', None)
            else:
                os.environ['RESULTCACHING_DISABLE'] = _prev_cache

    def _score(self, candidate) -> Score:
        layers = self._region_layers(candidate)

        baseline_acc = self._reading_accuracy(candidate)

        # word-form lesion
        selections = self._localize(candidate, layers)
        try:
            self._ablate(candidate, selections)
            lesioned_acc = self._reading_accuracy(candidate)
        finally:
            candidate.reset()

        # matched random-ablation control
        controls = [RandomSelection(layer=s.layer, n_units=len(s.indices),
                                    n_total=int(s.metadata['n_recorded']),
                                    seed=self.control_seed,
                                    population=s.metadata.get('unit_population')).resolve(candidate)
                    for s in selections]
        try:
            self._ablate(candidate, controls)
            random_acc = self._reading_accuracy(candidate)
        finally:
            candidate.reset()

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
        score.attrs['n_layers_ablated'] = len(layers)
        return score
