"""The one Cohen's d localizer selects the same units as the two copies it replaced.

Before, ``FunctionalSelection.resolve`` recorded with ``start_recording`` +
``process`` and the induced-dyslexia benchmark had its own per-layer copy of
the statistic. Both copies are kept here, as they were at commit f6e2a59, as
the reference. The localizer now records through a session. On a small fake
model, both give the same layer, unit indices, unit population and tensor
layout, so a lesion built from either selects the same units.
"""
import numpy as np
import pytest
import torch
from torch import nn

from brainscore import BrainScoreModel, CompositeSelector, FunctionalSelection
from brainscore.benchmarks.induced_dyslexia.benchmark import InducedDyslexia
from brainscore.perturbation import build_pytorch_ablation_fn
from brainscore.selection import UnitSelector, selection_coordinates, selection_unit_indices
from brainscore_core.model_interface import Perturbation, Selection, StateChange
from brainscore_core.streaming_helpers import neural_response
from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet

LABELS = ['real', 'pseudo', 'real', 'pseudo', 'real', 'pseudo', 'real', 'pseudo']


# -- the two copies at f6e2a59 (reference only) ----------------------------

def _reference_functional_resolve(model, region, stimuli, contrast, contrast_column,
                                  n_units, threshold, sign):
    """FunctionalSelection.resolve at f6e2a59: records with start_recording + process."""
    model.start_recording(region)
    asm = model.process(stimuli)
    extra = [d for d in asm.dims if d not in ('presentation', 'neuroid')]
    if extra:
        asm = asm.mean(extra)
    labels = np.asarray(asm[contrast_column].values)
    pos_mask = np.isin(labels, list(contrast[0]))
    neg_mask = np.isin(labels, list(contrast[1]))
    vals = asm.transpose('presentation', 'neuroid').values
    pos, neg = vals[pos_mask], vals[neg_mask]
    mp, mn = pos.mean(0), neg.mean(0)
    sp, sn = pos.std(0, ddof=1), neg.std(0, ddof=1)
    pooled = np.sqrt((sp ** 2 + sn ** 2) / 2.0)
    d = np.divide(mp - mn, pooled, out=np.zeros_like(mp, dtype=float), where=pooled > 0)
    score = {'positive': d, 'negative': -d, 'abs': np.abs(d)}[sign]
    order = (np.argsort(score)[::-1][:n_units] if n_units is not None
             else np.where(score >= threshold)[0])
    layer_path = getattr(model, 'region_layer_map', {}).get(region, region)
    uniq = list(dict.fromkeys(str(x) for x in np.asarray(asm['layer'].values).ravel()))
    if len(uniq) == 1:
        layer_path = uniq[0]
    if isinstance(layer_path, UnitSelector):
        layer_path = layer_path.layer_path
    population = selection_unit_indices(asm, model, region)
    axis, expected_shape = selection_coordinates(asm)
    idx = sorted(int(i) for i in population[order])
    return Selection(layer=layer_path, indices=idx, axis=axis, expected_shape=expected_shape,
                     metadata={'n_recorded': int(vals.shape[1]),
                               'unit_population': population.tolist()})


def _reference_localize(subject, region, stimuli, n_units):
    """InducedDyslexia._localize at f6e2a59: its own per-layer copy of the statistic."""
    asm = neural_response(subject, stimuli, [f'neural:{region}'])
    extra = [d for d in asm.dims if d not in ('presentation', 'neuroid')]
    if extra:
        asm = asm.mean(extra)
    asm = asm.transpose('presentation', 'neuroid')
    layer_coord = np.asarray(asm['layer'].values).ravel()
    label_of = dict(zip(stimuli['stimulus_id'].values, stimuli['image_label'].values))
    labels = np.asarray([label_of[sid] for sid in asm['stimulus_id'].values])
    vals = asm.values
    pos, neg = labels == 'real', labels == 'pseudo'
    out = []
    for L in dict.fromkeys(str(layer) for layer in layer_coord):
        cols = np.where(layer_coord == L)[0]
        sub = vals[:, cols]
        mp, mn = sub[pos].mean(0), sub[neg].mean(0)
        sp, sn = sub[pos].std(0, ddof=1), sub[neg].std(0, ddof=1)
        pooled = np.sqrt((sp ** 2 + sn ** 2) / 2.0)
        d = np.divide(mp - mn, pooled, out=np.zeros_like(mp, dtype=float), where=pooled > 0)
        order = np.argsort(d)[::-1][:n_units]
        population = selection_unit_indices(asm.isel(neuroid=cols), subject, region)
        axis, expected_shape = selection_coordinates(asm.isel(neuroid=cols))
        out.append(Selection(layer=L, indices=sorted(int(i) for i in population[order]),
                             axis=axis, expected_shape=expected_shape,
                             metadata={'n_recorded': int(sub.shape[1]),
                                       'unit_population': population.tolist()}))
    return out


# -- a small fake model ----------------------------------------------------

WIDTH = 12


def _net():
    torch.manual_seed(0)
    net = nn.Module()
    for layer in ('L', 'M'):
        linear = nn.Linear(WIDTH, WIDTH, bias=False)
        with torch.no_grad():
            linear.weight.copy_(torch.eye(WIDTH))
        net.add_module(layer, linear)
    return net


def _inputs():
    rng = np.random.RandomState(3)
    x = rng.normal(0, 1, (len(LABELS), WIDTH))
    x[np.array(LABELS) == 'real'] += np.linspace(2.0, -2.0, WIDTH)   # graded selectivity
    return torch.tensor(x, dtype=torch.float32)


class _Extractor:
    """Runs each layer of the net on fixed inputs; echoes the stimulus columns."""
    identifier = 'parity-extractor'

    def __init__(self, net, x):
        self.net, self.x = net, x

    def __call__(self, stimuli, layers):
        rows = [int(sid[1:]) for sid in stimuli['stimulus_id']]
        blocks = [getattr(self.net, layer)(self.x[rows] * (1 + i)).detach().numpy()
                  for i, layer in enumerate(layers)]
        return NeuroidAssembly(np.concatenate(blocks, axis=1), dims=['presentation', 'neuroid'], coords={
            **{column: ('presentation', list(stimuli[column])) for column in stimuli.columns},
            'layer': ('neuroid', [layer for layer in layers for _ in range(WIDTH)]),
            'neuroid_id': ('neuroid', [f'{layer}.{i}' for layer in layers for i in range(WIDTH)])})


def _stimuli():
    ids = [f's{i}' for i in range(len(LABELS))]
    stimuli = StimulusSet({'stimulus_id': ids, 'sentence': ids, 'image_label': LABELS})
    stimuli.identifier = 'localizer-parity'
    return stimuli


def _model(region_layer_map):
    net = _net()
    return net, BrainScoreModel('parity', model=net, region_layer_map=region_layer_map,
                                preprocessors={'text': _Extractor(net, _inputs())},
                                state_change_fn=build_pytorch_ablation_fn(net))


def _same(new, old):
    assert (new.layer, new.indices, new.axis, new.expected_shape) == \
        (old.layer, old.indices, old.axis, old.expected_shape)
    assert new.metadata['n_recorded'] == old.metadata['n_recorded']
    assert new.metadata['unit_population'] == old.metadata['unit_population']


@pytest.mark.parametrize('sign', ['positive', 'negative', 'abs'])
@pytest.mark.parametrize('n_units, threshold', [(3, 2.0), (None, 1.0)])
def test_functional_selection_selects_the_units_of_the_process_copy(sign, n_units, threshold):
    _, model = _model({'VWFA': 'M'})
    settings = dict(contrast=(['real'], ['pseudo']), contrast_column='image_label',
                    n_units=n_units, threshold=threshold, sign=sign)
    old = _reference_functional_resolve(model, 'VWFA', _stimuli(), **settings)
    model.reset()
    new = FunctionalSelection('VWFA', _stimuli(), **settings).resolve(model)
    assert new.indices                                  # the case selects units
    _same(new, old)


@pytest.mark.parametrize('layers', [
    (('L', None), ('M', None)),
    (('M', (11, 0, 5, 7, 2)), ('L', (3, 9, 1, 4))),
    (('L', (6, 2, 10)), ('M', ())),
])
def test_benchmark_localizer_selects_the_units_of_its_old_copy(layers):
    net, model = _model({'VWFA': CompositeSelector(layers)})
    old = _reference_localize(model, 'VWFA', _stimuli(), n_units=2)
    bench = InducedDyslexia(n_units=2, modality='text', reading_benchmark=object(),
                            localizer_stimuli=_stimuli())
    new = bench._localize(model)
    assert [s.layer for s in new] == [s.layer for s in old]
    for new_selection, old_selection in zip(new, old):
        _same(new_selection, old_selection)

    # The lesions built from the old and the new selections zero the same units.
    x = _inputs()
    outputs = []
    for selections in (old, new):
        for selection in selections:
            model.process(StateChange('ablation', target=selection, perturbation=Perturbation('zero')))
        outputs.append({layer: getattr(net, layer)(x).detach().numpy() for layer in ('L', 'M')})
        model.reset()
    for layer in ('L', 'M'):
        np.testing.assert_array_equal(outputs[0][layer], outputs[1][layer])
