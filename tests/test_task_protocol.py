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


def test_passive_task_records_viewing_and_refuses_behavior(tmp_path):
    net = Net()
    passive = TaskContext(task_type='passive')
    with pytest.raises(ValueError, match='passive task produces no behavior'):
        Experiment(subject=_subject(net), protocol=TaskProtocol(STIMULI, passive),
                   tools=[RecordBehavior()], output_dir=tmp_path / 'bad').validate()
    assert net.passes == 0


def test_recorders_on_a_protocol_that_will_not_request_them_fail(tmp_path):
    from brainscore.experiments import SessionProtocol

    @contextmanager
    def sessions(trial):
        raise AssertionError('not reached')
    protocol = SessionProtocol('own', sessions, output_channels=['behavior'])
    with pytest.raises(ValueError, match="Recorders need \\['neural:A'\\]"):
        Experiment(subject=_subject(Net()), protocol=protocol, tools=[RecordNeural(['A'])],
                   output_dir=tmp_path / 'x').validate()


def test_recording_is_labelled_with_how_it_was_made(tmp_path):
    result = Experiment(subject=_subject(Net()), protocol=TaskProtocol(STIMULI, TASK),
                        tools=[RecordNeural(['A'])], output_dir=tmp_path / 'run').run()
    recording = result.neural('A').attrs['recording']
    assert recording['task_type'] == 'label' and recording['forward_pass'] == 'prompt'
    with pytest.raises(FileNotFoundError, match='attach RecordBehavior'):
        result.behavior()                                  # not recorded, not requested


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


class CharTokenizer:
    """One token per character, ids 1..26 for a..z."""
    def __call__(self, text, add_special_tokens=False):
        return {'input_ids': [ord(c) - 96 if 'a' <= c <= 'z' else 0 for c in text]}

    def decode(self, ids, skip_special_tokens=True):
        return ''.join(chr(int(i) + 96) for i in ids)


class Batch(dict):
    def to(self, device):
        return self


class CharProcessor:
    tokenizer = CharTokenizer()

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kwargs):
        self.kwargs = kwargs
        return messages[0]['content'][-1]['text']

    def __call__(self, text, return_tensors, padding, images=None):
        ids = torch.tensor([self.tokenizer(text[0])['input_ids']])
        return Batch(input_ids=ids, attention_mask=torch.ones_like(ids),
                     token_type_ids=torch.zeros_like(ids))


class TinyLM2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.embed, self.head = torch.nn.Embedding(27, 8), torch.nn.Linear(8, 27)
        self.device = torch.device('cpu')

    def forward(self, input_ids, attention_mask, token_type_ids):
        assert input_ids.shape == attention_mask.shape == token_type_ids.shape
        return type('Out', (), {'logits': self.head(self.embed(input_ids))})()

    def generate(self, input_ids, max_new_tokens, do_sample, **kwargs):
        return torch.cat([input_ids, torch.tensor([[4, 15, 7]])], dim=1)   # 'dog'


def test_chat_parts_reply_and_likelihood():
    from brainscore.model_helpers.replies import chat_parts
    model, processor = TinyLM2(), CharProcessor()
    from brainscore_core.responses import Prompt
    reply, likelihood = chat_parts(model, processor, chat_template_kwargs={'enable_thinking': False})
    prompt = Prompt((('text', 'x\n\nwhich'),), stimulus_id='a')
    assert reply(prompt) == 'dog'
    assert processor.kwargs == {'enable_thinking': False}
    scores = likelihood(prompt, ['cat', 'dog'])
    # Reference: score 'cat' after the exact prompt the helper built.
    built = processor.tokenizer(processor.apply_chat_template(
        [{'role': 'user', 'content': [{'type': 'text', 'text': 'x\n\nwhich'}]}],
        tokenize=False, add_generation_prompt=True))['input_ids']
    ids = torch.tensor([built + [3, 1, 20]])
    logp = model.head(model.embed(ids))[0].log_softmax(-1)
    expected = sum(float(logp[len(built) - 1 + i, t]) for i, t in enumerate([3, 1, 20]))
    assert scores['cat'] == pytest.approx(expected, rel=1e-5)
    assert set(scores) == {'cat', 'dog'}


def test_experiment_saves_every_kind_of_answer(tmp_path):
    from brainscore_core.responses import Choice, Scale, Text

    def reply(prompt):
        sentence = [content for kind, content in prompt.parts if kind == 'text'][1]
        if 'confidence' in prompt.text():
            return 'long\n4' if len(sentence) > 1 else 'short\n5'
        return f"It reads {sentence}."

    model = BrainScoreModel(identifier='replier', supported_modalities={'text'}, reply=reply)
    choice = TaskContext(instruction='Is it long?', response=Choice(['short', 'long'], confidence=Scale(1, 5, 1)))
    result = Experiment(subject=model, protocol=TaskProtocol(STIMULI, choice),
                        tools=[RecordBehavior()], output_dir=tmp_path / 'choice').run()
    assert list(result.behavior()['raw_answer'].values) == ['short\n5', 'long\n4', 'long\n4']
    assert list(result.behavior(measure='confidence').values) == [5, 4, 4]

    written = TaskContext(instruction='Read it.', response=Text())
    result = Experiment(subject=model, protocol=TaskProtocol(STIMULI, written),
                        tools=[RecordBehavior()], output_dir=tmp_path / 'text').run()
    assert list(result.behavior(measure='text').values) == ['It reads x.', 'It reads yy.', 'It reads zzz.']


def test_a_model_that_cannot_write_fails_at_validation(tmp_path):
    from brainscore_core.responses import Text
    net = Net()
    experiment = Experiment(subject=_subject(net),       # answer= only: choices, not text
                            protocol=TaskProtocol(STIMULI, TaskContext(response=Text())),
                            tools=[RecordBehavior()], output_dir=tmp_path / 'x')
    with pytest.raises(ValueError, match='behavior:text'):
        experiment.validate()
    assert net.passes == 0


class TinyVisionGrid(torch.nn.Module):
    """Qwen-style tower: flattened patches for several images plus their grid."""
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([torch.nn.Linear(4, 4)])

    def forward(self, patches, grid_thw):
        return self.blocks[0](patches)


def test_vlm_vision_wrapper_records_each_image_of_a_trial():
    from brainscore.model_helpers.vlm_vision_wrapper import VLMVisionWrapper
    model = TinyVisionGrid()
    wrapper = VLMVisionWrapper(model=model, processor=None, identifier='tiny-vision')
    grid = torch.tensor([[1, 2, 2], [1, 2, 3], [1, 1, 2]])          # 4, 6 and 2 patches
    patches = torch.randn(12, 4, device=wrapper._device)
    with wrapper.record(['blocks.0']) as recording:
        out = model(patches, grid_thw=grid)
        several = recording.take()
        model(patches[:4], grid_thw=grid[:1])
        one = recording.take()
    expected = [out[0:4].mean(0), out[4:10].mean(0), out[10:12].mean(0)]
    assert several.dims == ('presentation', 'part', 'neuroid')
    np.testing.assert_allclose(several.values[0], torch.stack(expected).detach().cpu().numpy(), rtol=1e-5)
    assert one.dims == ('presentation', 'neuroid')
    np.testing.assert_allclose(one.values[0], out[0:4].mean(0).detach().cpu().numpy(), rtol=1e-5)
