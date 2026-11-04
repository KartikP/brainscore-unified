"""Design a task, invite a subject, attach recorders, run once.

The subject's network counts forward passes, so these check that each
stimulus is presented once per trial for both measurements.
"""
from contextlib import contextmanager

import numpy as np
import pytest
import torch

from brainscore_core.brainscore_model import BrainScoreModel
from brainscore_core.events import Perturbation, PerturbationApplied, Selection, StateChange
from brainscore_core.model_interface import TaskContext
from brainscore_core.selection import LayerSelector
from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
from brainscore.experiments import (Experiment, Intervene, RecordBehavior, RecordNeural,
                                    TaskProtocol)
from brainscore.model_helpers.text_wrapper import TextWrapper

pytestmark = pytest.mark.unit

STIMULI = StimulusSet({'stimulus_id': ['a', 'b', 'c'], 'sentence': ['x', 'yy', 'zzz']})
STIMULI.identifier = 'task-protocol-toy'
TASK = TaskContext(task_type='label', label_set=['short', 'long'],
                   instruction='Is it long?', prefer_path='generation')


class Net:
    def __init__(self):
        self.passes, self.hooks, self.silenced = 0, [], False

    def forward(self, text):
        self.passes += 1
        early = np.zeros(3) if self.silenced else np.full(3, float(len(text)))
        for hook in self.hooks:
            hook(early)
        return early


class Extractor:
    identifier = backbone_id = 'toy'

    def __init__(self, net):
        self.net = net

    def __call__(self, stimuli, layers, **kwargs):
        raise AssertionError('a task session presents each stimulus once, through answer')

    @contextmanager
    def record(self, layers):
        seen = []

        class Recording:
            @staticmethod
            def take():
                values = seen[0]
                seen.clear()
                return NeuroidAssembly(values[None], dims=['presentation', 'neuroid'], coords={
                    'stimulus_id': ('presentation', ['?']),
                    'neuroid_id': ('neuroid', ['early.0', 'early.1', 'early.2']),
                    'layer': ('neuroid', ['early'] * 3)})
        self.net.hooks.append(seen.append)
        try:
            yield Recording()
        finally:
            self.net.hooks.remove(seen.append)


def _subject(net):
    def answer(stimulus_row, instruction, label_set):
        early = net.forward(stimulus_row['sentence'])
        return label_set[int(early[0] > 1)]

    def intervention(change):
        net.silenced = True

        def cleanup():
            net.silenced = False
        return PerturbationApplied(handle_id='h', target=change.target,
                                   perturbation=change.perturbation, applied_at=0), cleanup

    return BrainScoreModel(identifier='toy', extractor={'text': Extractor(net)},
                           region_layer_map={'A': LayerSelector('early', modality='text')},
                           answer=answer, intervention=intervention)


def test_one_presentation_gives_behavior_and_activity_per_condition(tmp_path):
    net = Net()
    silence = StateChange(kind='ablation', target=Selection(layer='early'),
                          perturbation=Perturbation(kind='zero'))
    result = Experiment(
        subject=_subject(net),
        protocol=TaskProtocol(STIMULI, TASK, conditions=['baseline', 'silenced']),
        tools=[RecordNeural(['A']), RecordBehavior(), Intervene(silence, conditions=['silenced'])],
        output_dir=tmp_path / 'run').run()
    assert net.passes == 2 * len(STIMULI)              # once per stimulus per condition
    baseline = result.neural('A', condition='baseline')
    assert list(baseline['stimulus_id'].values) == ['a', 'b', 'c']
    np.testing.assert_array_equal(baseline.values[:, 0], [1, 2, 3])
    np.testing.assert_array_equal(result.neural('A', condition='silenced').values, 0)
    assert list(result.behavior(condition='baseline')['raw_answer'].values) == ['short', 'long', 'long']
    assert list(result.behavior(condition='silenced')['raw_answer'].values) == ['short'] * 3
    assert not net.silenced                            # the change was undone


def test_recorder_for_an_undeclared_region_fails_before_running(tmp_path):
    net = Net()
    experiment = Experiment(subject=_subject(net), protocol=TaskProtocol(STIMULI, TASK),
                            tools=[RecordNeural(['IT'])], output_dir=tmp_path / 'run')
    with pytest.raises(ValueError, match='neural:IT'):
        experiment.run()
    assert net.passes == 0


class TinyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Linear(4, 4) for _ in range(2)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def test_text_wrapper_records_the_prompt_pass_not_generated_tokens():
    model = TinyLM()
    wrapper = TextWrapper(model=model, tokenizer=None, identifier='tiny')
    prompt = torch.randn(1, 5, 4, device=wrapper._device)
    with wrapper.record(['layers.1']) as recording:
        expected = model(prompt)[0, -1]
        model(torch.randn(1, 1, 4, device=wrapper._device))       # a generated token
        taken = recording.take()
        with pytest.raises(ValueError, match='No forward pass'):
            recording.take()
    np.testing.assert_allclose(taken.values[0], expected.detach().cpu().numpy(), rtol=1e-6)
    assert set(taken['layer'].values) == {'layers.1'}
    assert not model.layers[1]._forward_hooks                      # hooks removed


class TinyVision(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([torch.nn.Linear(4, 4)])

    def forward(self, patches):                  # (n_patches, hidden), Qwen-style
        return self.blocks[0](patches)


def test_vlm_vision_wrapper_records_the_mean_over_patches():
    from brainscore.model_helpers.vlm_vision_wrapper import VLMVisionWrapper
    model = TinyVision()
    wrapper = VLMVisionWrapper(model=model, processor=None, identifier='tiny-vision')
    patches = torch.randn(6, 4, device=wrapper._device)
    with wrapper.record(['blocks.0']) as recording:
        expected = model(patches).mean(dim=0)
        taken = recording.take()
    np.testing.assert_allclose(taken.values[0], expected.detach().cpu().numpy(), rtol=1e-5)
