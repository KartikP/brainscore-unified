"""A region whose ``modality`` is a tuple: the recording extractor first, then the other inputs that reach its layer.

The registration states that a picture reaches the language model's layer in an answering pass (rule 2.5), as in
Qwen-VL, where the picture tokens enter the language model. ``output_inputs`` lists both inputs; every routing reader
keeps the first entry. So a picture task records the region from the answering pass, a passive picture session and
the readout are refused before compute, and a wrong statement fails at the first pass.
"""
import numpy as np
import pandas as pd
import pytest

from brainscore import BrainScoreModel, LayerSelector
from brainscore_core.compatibility import ExecutionNotFeasibleError, check_compatibility
from brainscore_core.contract import TaskContext
from brainscore_core.responses import Choice
from brainscore_core.streaming_helpers import stimulus_session
from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly

pytestmark = pytest.mark.unit

PICTURES = pd.DataFrame({'stimulus_id': ['s0', 's1'], 'image_file_name': ['w0.png', 'w1.png']})
SENTENCES = pd.DataFrame({'stimulus_id': ['t0', 't1'], 'sentence': ['real word', 'blick']})
FIT = pd.DataFrame({'stimulus_id': ['f0', 'f1'], 'image_file_name': ['f0.png', 'f1.png'],
                    'image_label': ['yes', 'no']})
TASK = TaskContext(instruction='Is it a word?', response=Choice(['yes', 'no']))


def _assembly(ids, layers):
    return NeuroidAssembly(np.ones((len(ids), len(layers))), dims=['presentation', 'neuroid'], coords={
        'stimulus_id': ('presentation', list(ids)), 'stimulus_path': ('presentation', list(ids)),
        'neuroid_id': ('neuroid', list(layers)), 'layer': ('neuroid', list(layers))})


class Tower:
    """A stand-in extractor of one tower: its layer names, a passive call and a recording during a task."""

    def __init__(self, name, layers, log):
        self.name, self._layers, self.log = name, layers, log

    def layer_names(self):
        return list(self._layers)

    def __call__(self, stimuli, layers, **_):
        self.log.append((self.name, 'call', tuple(layers)))
        return _assembly(list(stimuli['stimulus_id']), layers)

    def record(self, layers, stimuli=None):
        self.log.append((self.name, 'record', tuple(layers)))
        tower = self

        class Live:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def take(self, row):
                tower.log.append((tower.name, 'take', row['stimulus_id']))
                return {layer: _assembly([row['stimulus_id']], [layer]) for layer in layers}
        return Live()


def _subject(log, language=('text', 'vision'), **parts):
    return BrainScoreModel(
        'reads-pictures',
        extractor={'vision': Tower('vision', ['blocks.28'], log), 'text': Tower('text', ['layers.18'], log)},
        region_layer_map={'VWFA': 'blocks.28',
                          'language_system': LayerSelector(name='layers.18', modality=language)},
        **parts)


class _Task:
    """What the experiment runner declares for a picture task with an instruction (notebook 14)."""
    identifier, trial_memory = 'task', 'none'
    required_input_channels = frozenset({'vision:image', 'instruction'})
    requested_output_channels = frozenset({'behavior:choice', 'neural:VWFA', 'neural:language_system'})


def test_the_region_lists_every_stated_input_and_routes_by_the_first():
    model = _subject([], reply=lambda prompt: 'yes')
    assert model.output_inputs['neural:language_system'] == {'text', 'vision:image'}
    assert model.output_inputs['neural:VWFA'] == {'vision:image'}
    assert model._region_modality_map['language_system'] == 'text'
    assert model.region_layer_selectors['language_system'].modality == 'text'
    # Lossless: the configuration assigned back gives the same declarations.
    given = model.region_layer_map
    assert given['language_system'].modality == ('text', 'vision')
    model.region_layer_map = given
    assert model.output_inputs['neural:language_system'] == {'text', 'vision:image'}


def test_pre_flight_passes_the_picture_task_only_with_the_statement():
    check_compatibility(_subject([], reply=lambda prompt: 'yes'), _Task())
    with pytest.raises(Exception, match=r"answers 'neural:language_system' only from the inputs \{'text'\}"):
        check_compatibility(_subject([], language='text', reply=lambda prompt: 'yes'), _Task())


@pytest.mark.parametrize('parts, output', [
    (dict(reply=lambda prompt: 'yes'), 'behavior:choice'),
    (dict(answer=lambda stimulus_row, instruction, label_set: label_set[0]), 'behavior:choice'),
    (dict(likelihood=lambda prompt, labels: {label: -float(i) for i, label in enumerate(labels)},
          answer_owners={'behavior:probabilities': ('likelihood', 'softmax'),
                                                   'behavior:choice': ('likelihood', 'softmax_argmax')}),
     'behavior:probabilities'),
], ids=['reply', 'answer', 'likelihood'])
def test_a_picture_task_records_the_region_from_the_answering_pass(parts, output):
    log = []
    session = stimulus_session(PICTURES, [output, 'neural:language_system'], TASK)
    _subject(log, **parts).participate(session)
    assert session.collect('neural:language_system').sizes['presentation'] == 2
    assert log == [('text', 'record', ('layers.18',)), ('text', 'take', 's0'), ('text', 'take', 's1')]


def test_the_readout_is_refused_before_compute():
    log = []
    model = _subject(log, behavioral_readout_layer='blocks.28',
                     answer_owners={'behavior:choice': ('readout', 'argmax')})
    task = TaskContext(instruction='Is it a word?', fitting_stimuli=FIT, response=Choice(['yes', 'no']))
    with pytest.raises(ExecutionNotFeasibleError, match=r"Region 'language_system' reads 'text' input, but the "
                                                        r"readout presents 'vision' input"):
        model.participate(stimulus_session(PICTURES, ['behavior:choice', 'neural:language_system'], task))
    assert log == []


def test_a_passive_picture_session_is_refused_before_compute():
    log = []
    with pytest.raises(ExecutionNotFeasibleError,
                       match=r"Recording targets require missing input modalities: \['text'\]"):
        _subject(log, reply=lambda prompt: 'yes').participate(
            stimulus_session(PICTURES, ['neural:language_system']))
    assert log == []


@pytest.mark.parametrize('language', [('text', 'vision'), 'text'], ids=['stated', 'not stated'])
def test_a_text_session_is_unchanged(language):
    log = []
    session = stimulus_session(SENTENCES, ['neural:language_system'])
    _subject(log, language=language, reply=lambda prompt: 'yes').participate(session)
    assert log == [('text', 'call', ('layers.18',))]
    assert session.collect('neural:language_system').sizes['presentation'] == 2


def test_a_wrong_statement_fails_at_the_first_pass():
    """A picture-tower region stated as reached from text passes pre-flight; a text reply never runs that tower."""
    torch = pytest.importorskip('torch')
    from brainscore.model_helpers.extractors import LiveRecording

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.visual, self.language_model = torch.nn.Module(), torch.nn.Module()
            self.visual.block, self.language_model.layer = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)

        def forward(self, input_ids):
            return self.language_model.layer(input_ids)     # a text prompt reaches only the language model

    net, passes = Net(), []

    class HookedTower(Tower):
        def __init__(self, name, root, layers):
            super().__init__(name, layers, [])
            self.root = root

        def record(self, layers, stimuli=None):
            return LiveRecording(net, self.root, list(layers), lambda value: value.detach().numpy(),
                                 lambda activity, read, row: {layer: _assembly([row['stimulus_id']], [layer])
                                                              for layer in activity})

    def reply(prompt):
        passes.append(1)
        net(torch.ones(1, 2))
        return 'yes'

    model = BrainScoreModel(
        'wrong-statement', model=net, reply=reply,
        extractor={'vision': HookedTower('vision', net.visual, ['block']),
                   'text': HookedTower('text', net.language_model, ['layer'])},
        region_layer_map={'VWFA': LayerSelector(name='block', modality=('vision', 'text'))})
    assert model.output_inputs['neural:VWFA'] == {'vision:image', 'text'}

    class TextTask(_Task):
        required_input_channels = frozenset({'text', 'instruction'})
        requested_output_channels = frozenset({'behavior:choice', 'neural:VWFA'})
    check_compatibility(model, TextTask())                  # the statement is not checked before a pass
    with pytest.raises(ValueError, match=r"No forward pass reached \['block'\] since the last presentation\."):
        model.participate(stimulus_session(SENTENCES, ['behavior:choice', 'neural:VWFA'], TASK))
    assert passes == [1]


@pytest.mark.parametrize('modality, message', [
    (('text', 'audio'), "references modalities with no extractor: 'audio'"),
    (('text', 'text'), 'a tuple modality names distinct input families'),
    ((), 'a tuple modality names distinct input families'),
])
def test_a_malformed_statement_is_refused_when_the_model_is_built(modality, message):
    with pytest.raises(ValueError, match=message):
        _subject([], language=modality, reply=lambda prompt: 'yes')
