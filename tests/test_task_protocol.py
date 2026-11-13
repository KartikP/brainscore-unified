"""Design a task, invite a subject, attach recorders, run once.

The subject's network counts forward passes, so these check that each
stimulus is presented once per trial for both measurements.
"""
from contextlib import contextmanager

import numpy as np
import pytest
import torch

from brainscore.brainscore_model import BrainScoreModel
from brainscore_core.compatibility import CompatibilityError
from brainscore_core.events import Perturbation, PerturbationApplied, Selection, StateChange
from brainscore_core.contract import TaskContext
from brainscore.selection import LayerSelector
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


def test_a_run_with_an_intervention_is_recorded_and_replayed(tmp_path):
    """Replay sends the saved inputs again; the intervention tool is attached again."""
    from brainscore.experiments import RecordInputsOutputs, compare_outputs, replay_sessions
    net = Net()
    subject = _subject(net)
    silence = StateChange(kind='ablation', target=Selection(layer='early'),
                          perturbation=Perturbation(kind='zero'))
    conditions = ['baseline', 'silenced']
    original = Experiment(
        subject=subject, protocol=TaskProtocol(STIMULI, TASK, conditions=conditions),
        tools=[RecordInputsOutputs(), RecordBehavior(), Intervene(silence, conditions=['silenced'])],
        output_dir=tmp_path / 'run').run()
    events = list(original.record.events())
    assert [e['payload'] for e in events if e['kind'] == 'intervention'] == [{'handle_id': 'h'}]
    replay = Experiment(
        subject=subject, protocol=replay_sessions(original.directory),
        tools=[RecordInputsOutputs(), Intervene(silence, conditions=['silenced'])],
        output_dir=tmp_path / 'replay').run()
    assert compare_outputs(original.directory, replay.directory)['equal']
    unchanged = Experiment(subject=subject, protocol=replay_sessions(original.directory),
                           tools=[RecordInputsOutputs()], output_dir=tmp_path / 'no-intervention').run()
    assert not compare_outputs(original.directory, unchanged.directory)['equal']
    assert not net.silenced and subject._interventions.active_interventions == {}


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
    protocol = SessionProtocol('own', sessions, requested_output_channels=['behavior'])
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
    with pytest.raises(CompatibilityError, match='neural:IT'):
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


def test_chat_parts_reply_obeys_the_task_generation_settings():
    """The Text response's settings decide the generate call; without them it is unchanged."""
    from brainscore_core.responses import Choice, Prompt, Text
    from brainscore.model_helpers.replies import chat_parts
    calls = []

    class Recorder(TinyLM2):
        def generate(self, input_ids, attention_mask, token_type_ids, **kwargs):
            calls.append((kwargs, float(torch.rand(1))))
            return torch.cat([input_ids, torch.tensor([[4, 15, 7]])], dim=1)

    reply, _ = chat_parts(Recorder(), CharProcessor())
    shown = (('text', 'which'),)
    # No settings (no response, a Choice, an empty Text): greedy, 64 tokens, as before.
    for response in (None, Choice(['cat', 'dog']), Text()):
        assert reply(Prompt(shown, response=response)) == 'dog'
    assert [kwargs for kwargs, _ in calls] == [{'max_new_tokens': 64, 'do_sample': False}] * 3
    calls.clear()
    reply(Prompt(shown, response=Text(max_tokens=512)))
    reply(Prompt(shown, response=Text(max_tokens=8, temperature=0.0)))
    reply(Prompt(shown, response=Text(max_tokens=8, temperature=0.7, top_p=0.9, seed=3)))
    reply(Prompt(shown, response=Text(max_tokens=8, temperature=0.7, top_p=0.9, seed=3)))
    assert [kwargs for kwargs, _ in calls] == [
        {'max_new_tokens': 512, 'do_sample': False},
        {'max_new_tokens': 8, 'do_sample': False},
        {'max_new_tokens': 8, 'do_sample': True, 'temperature': 0.7, 'top_p': 0.9},
        {'max_new_tokens': 8, 'do_sample': True, 'temperature': 0.7, 'top_p': 0.9},
    ]
    # The seed is set before generation, so the same seed draws the same numbers.
    assert calls[2][1] == calls[3][1]


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


def test_extra_data_of_a_reply_goes_into_the_meta_of_the_answer_event():
    """One answer event for several trials: each key holds one value per trial."""
    from brainscore_core.responses import Choice
    from brainscore_core.streaming import StreamEvent
    from brainscore_core.streaming_helpers import stimulus_session

    def reply(prompt):
        sentence = [content for kind, content in prompt.parts if kind == 'text'][0]
        if sentence == 'yy':
            return 'long'   # the text alone: no extra data for this trial
        return StreamEvent('behavior:choice', 'short', None, {'reasoning': f'{sentence} is short'})

    model = BrainScoreModel(identifier='replier', supported_modalities={'text'}, reply=reply)
    session = stimulus_session(STIMULI, ['behavior:choice'], TaskContext(response=Choice(['short', 'long'])))
    model.participate(session)
    event, = [e for e in session.emitted if e.channel == 'behavior:choice']
    assert event.meta['reasoning'] == ['x is short', None, 'zzz is short']
    assert event.meta['driver'] == 'interact'
    assert list(event.payload['raw_answer'].values) == ['short', 'long', 'short']


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


def test_chat_parts_replays_earlier_trials_as_turns():
    from brainscore_core.responses import Prompt
    from brainscore.model_helpers.replies import chat_parts
    seen = {}

    class Recorder(CharProcessor):
        def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kwargs):
            seen['messages'] = messages
            return messages[-1]['content'][-1]['text']
    reply, _ = chat_parts(TinyLM2(), Recorder())
    first = Prompt((('text', 'cat'),))
    reply(Prompt((('text', 'dog'),), history=((first, 'no'),)))
    assert [m['role'] for m in seen['messages']] == ['user', 'assistant', 'user']
    assert seen['messages'][1]['content'][0]['text'] == 'no'


def test_a_memory_task_rejects_a_subject_without_memory(tmp_path):
    from brainscore_core.responses import Choice
    forgets = BrainScoreModel(identifier='forgets', supported_modalities={'text'}, reply=lambda p: 'no')
    protocol = TaskProtocol(STIMULI, TaskContext(response=Choice(['yes', 'no'])),
                            required_traits={'memory:trial_history'})
    with pytest.raises(CompatibilityError, match='memory:trial_history'):
        Experiment(subject=forgets, protocol=protocol, tools=[RecordBehavior()],
                   output_dir=tmp_path / 'x').validate()


class TinyVLM(torch.nn.Module):
    """Token 27 stands for image tokens; the decoder is `lm.layers`."""
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.embed = torch.nn.Embedding(28, 4)
        self.lm = torch.nn.Module()
        self.lm.layers = torch.nn.ModuleList([torch.nn.Linear(4, 4)])
        self.config = type('Config', (), {'image_token_id': 27, 'video_token_id': None})()
        self.device = torch.device('cpu')

    def forward(self, input_ids, attention_mask=None, **kwargs):
        return self.lm.layers[0](self.embed(input_ids))


class ImageTokenProcessor(CharProcessor):
    def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kwargs):
        return ''.join('#' if part['type'] == 'image' else part['text']
                       for message in messages for part in message['content'])

    def __call__(self, text, return_tensors, padding, images=None):
        ids = []
        for character in text[0]:
            ids += [27, 27, 27] if character == '#' else self.tokenizer(character)['input_ids']
        ids = torch.tensor([ids])
        return Batch(input_ids=ids, attention_mask=torch.ones_like(ids))


def test_fused_chat_extractor_splits_one_pass_by_input(tmp_path):
    from PIL import Image as PILImage
    from brainscore.model_helpers.fused import fused_chat_extractor
    PILImage.new('RGB', (4, 4)).save(tmp_path / 'a.png')
    model = TinyVLM()
    fused = fused_chat_extractor(model, ImageTokenProcessor(), root=model.lm, identifier='tiny-fused')
    stimuli = StimulusSet({'stimulus_id': ['s'], 'image_path': [str(tmp_path / 'a.png')], 'sentence': ['ab']})
    recorded = fused(stimuli, layers=['layers.0'])
    assert recorded.dims == ('presentation', 'part', 'neuroid')
    assert list(recorded['part_name'].values) == ['image', 'text', 'final']
    ids = torch.tensor([[27, 27, 27, 1, 2]])                     # image first, then 'ab'
    hidden = model(ids)[0].detach().numpy()
    np.testing.assert_allclose(recorded.values[0], [hidden[:3].mean(0), hidden[3:].mean(0), hidden[-1]], rtol=1e-5)


def test_partial_inputs_match_scoring_and_preserve_responses(tmp_path):
    from types import SimpleNamespace
    from brainscore_core import check_compatibility

    net = Net()
    subject = _subject(net)
    mixed = StimulusSet(STIMULI.assign(image_path='unused-image.png'))
    mixed.identifier = 'mixed-task'
    protocol = TaskProtocol(mixed, TASK)
    experiment = Experiment(subject=subject, protocol=protocol, tools=[RecordBehavior()],
                            output_dir=tmp_path / 'partial')
    plan = experiment.validate()
    benchmark = SimpleNamespace(identifier='same-task',
                                required_input_channels=protocol.required_input_channels,
                                requested_output_channels=protocol.requested_output_channels)
    expected = check_compatibility(subject, benchmark).as_attrs()
    assert expected['match'] == 'partial'
    assert expected['consumed_inputs'] == ['text']
    assert expected['dropped_inputs'] == ['vision:image']
    assert plan['protocol']['config']['compatibility'] == expected
    assert net.passes == 0
    assert not experiment.output_dir.exists()

    partial = experiment.run()
    full = Experiment(subject=subject, protocol=TaskProtocol(STIMULI, TASK),
                      tools=[RecordBehavior()], output_dir=tmp_path / 'full').run()
    np.testing.assert_array_equal(partial.behavior().values, full.behavior().values)
    assert net.passes == 2 * len(STIMULI)
    assert full.manifest['plan']['protocol']['config']['compatibility']['match'] == 'full'


@pytest.mark.parametrize('mistake, expected', [
    ('output', "cannot produce output channels requested by benchmark 'task': {'neural:IT'}"),
    ('input', "takes in none of the inputs benchmark 'task' sends: {'vision:image'}"),
    ('required', "hard-requires input channels {'vision:image'}"),
])
def test_channel_errors_explain_declarations_before_resets(tmp_path, mistake, expected):
    calls = []
    subject = BrainScoreModel(identifier='reader', supported_modalities={'text', 'vision'}
                             if mistake == 'required' else {'text'},
                             required_modalities={'vision'} if mistake == 'required' else set(),
                             reply=lambda prompt: calls.append('inference') or 'yes')
    subject.reset = lambda: calls.append('reset')
    stimuli = (StimulusSet({'stimulus_id': ['x'], 'image_path': ['unused.png']})
               if mistake == 'input' else STIMULI)
    task = TaskContext(task_type='label', label_set=['yes', 'no'])
    experiment = Experiment(subject=subject, protocol=TaskProtocol(stimuli, task),
                            tools=[RecordNeural(['IT'])] if mistake == 'output' else [RecordBehavior()],
                            output_dir=tmp_path / 'bad')
    # The experiment uses the rule score() uses (core check_channel_compatibility).
    with pytest.raises(CompatibilityError) as caught:
        experiment.run()
    message = str(caught.value)
    assert expected in message
    assert "Subject 'reader'" in message
    assert calls == []
    assert not experiment.output_dir.exists()


@pytest.mark.parametrize('operation', ['validate', 'run'])
def test_neural_only_recording_checks_the_task_response_before_reset(tmp_path, operation):
    from brainscore_core.responses import Text

    net = Net()
    subject = _subject(net)
    resets = []
    subject.reset = lambda: resets.append('reset')
    experiment = Experiment(
        subject=subject, protocol=TaskProtocol(STIMULI, TaskContext(response=Text())),
        tools=[RecordNeural(['A'])], output_dir=tmp_path / 'bad-task')
    with pytest.raises(ValueError, match='Task response requires outputs.*behavior:text'):
        if operation == 'validate':
            experiment.validate()
        else:
            experiment.run()
    assert net.passes == 0
    assert resets == []
    assert not experiment.output_dir.exists()


def test_a_run_without_output_dir_reads_back_and_is_removed_with_its_result():
    import gc
    experiment = Experiment(subject=_subject(Net()), protocol=TaskProtocol(STIMULI, TASK),
                            tools=[RecordBehavior()])
    result = experiment.run()
    folder = result.directory
    assert result.temporary and folder.exists()
    assert result.behavior().sizes['presentation'] == len(STIMULI)    # readers work as usual
    del result, experiment
    gc.collect()
    assert not folder.exists()


def test_save_keeps_a_temporary_run_and_refuses_an_existing_folder(tmp_path):
    result = Experiment(subject=_subject(Net()), protocol=TaskProtocol(STIMULI, TASK),
                        tools=[RecordBehavior()]).run()
    kept = result.save(tmp_path / 'kept')
    assert not kept.temporary and (kept.directory / 'experiment.json').exists()
    np.testing.assert_array_equal(kept.behavior().values, result.behavior().values)
    with pytest.raises(FileExistsError):
        result.save(tmp_path / 'kept')


def test_without_output_dir_each_run_gets_its_own_folder(tmp_path):
    experiment = Experiment(subject=_subject(Net()), protocol=TaskProtocol(STIMULI, TASK),
                            tools=[RecordBehavior()])
    first, second = experiment.run(), experiment.run()
    assert first.directory != second.directory and first.directory.exists()
    explicit = Experiment(subject=_subject(Net()), protocol=TaskProtocol(STIMULI, TASK),
                          tools=[RecordBehavior()], output_dir=tmp_path / 'run')
    assert not explicit.run().temporary
    with pytest.raises(FileExistsError):
        explicit.run()                                  # an explicit folder is never overwritten


def test_task_protocol_keeps_the_task_readout_fit():
    from brainscore_core import TaskContext
    from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
    from brainscore.experiments import TaskProtocol
    stimuli = StimulusSet({'stimulus_id': ['a'], 'sentence': ['cat']})
    stimuli.identifier = 'readout-fit-test'
    for fit in ('current', 'intact'):
        protocol = TaskProtocol(stimuli, TaskContext(task_type='passive', readout_fit=fit))
        assert protocol.task.readout_fit == fit


def test_trials_gives_one_row_per_trial(tmp_path):
    net = Net()
    silence = StateChange(kind='ablation', target=Selection(layer='early'),
                          perturbation=Perturbation(kind='zero'))
    result = Experiment(
        subject=_subject(net),
        protocol=TaskProtocol(STIMULI, TASK, conditions=['baseline', 'silenced']),
        tools=[RecordBehavior(), Intervene(silence, conditions=['silenced'])]).run()
    trials = result.trials()
    assert list(trials.columns[:3]) == ['condition', 'repetition', 'stimulus_id']
    assert len(trials) == 2 * len(STIMULI)
    baseline = trials[trials['condition'] == 'baseline']
    assert list(baseline['stimulus_id']) == ['a', 'b', 'c']
    assert list(baseline['choice']) == ['short', 'long', 'long']
    assert list(trials[trials['condition'] == 'silenced']['choice']) == ['short'] * 3


def _options_stimuli():
    stimuli = StimulusSet({'stimulus_id': ['a', 'b'], 'text': ['the cat', 'the dog'],
                           'options': [('sat', 'ran'), ('kind', 'mean')]})
    stimuli.identifier = 'task-protocol-options'
    return stimuli


def test_options_that_differ_for_each_row_reach_each_part():
    """labels names a column: each row's options reach likelihood, reply and answer; the axis is positions."""
    from brainscore_core.responses import Choice
    from brainscore_core.streaming_helpers import stimulus_session
    seen = []

    def likelihood(prompt, candidates):
        seen.append(('likelihood', prompt.parts, tuple(candidates)))
        return {c: (0.5 if c in ('ran', 'kind') else -0.25) for c in candidates}

    def reply(prompt):
        seen.append(('reply', prompt.parts, prompt.response.labels))
        return prompt.response.labels[1]

    def answer(stimulus_row, instruction, label_set):
        seen.append(('answer', stimulus_row['text'], tuple(label_set)))
        return label_set[0]

    raw = TaskContext(response=Choice(labels='options', raw_scores=True, raw_text=True))
    scorer = BrainScoreModel(identifier='scorer', supported_modalities={'text'}, likelihood=likelihood)
    session = stimulus_session(_options_stimuli(), ['behavior:choice', 'behavior:probabilities'], raw)
    scorer.participate(session)
    scores = session.collect('behavior:probabilities')
    # Raw scores, as the part gave them, in the order of each row's options.
    np.testing.assert_array_equal(scores.values, [[-0.25, 0.5], [0.5, -0.25]])
    assert list(scores['choice'].values) == [0, 1]
    assert scores.attrs['measure'] == {'kind': 'implied', 'source': 'likelihood', 'scale': 'log'}
    np.testing.assert_array_equal(session.collect('behavior:choice').values, [[0, 1], [1, 0]])
    assert seen == [('likelihood', (('text', 'the cat'),), ('sat', 'ran')),
                    ('likelihood', (('text', 'the dog'),), ('kind', 'mean'))]
    seen.clear()
    phrased = TaskContext(response=Choice(labels='options'))
    for part in ({'reply': reply}, {'answer': answer}):
        model = BrainScoreModel(identifier='other', supported_modalities={'text'}, **part)
        session = stimulus_session(_options_stimuli(), ['behavior:choice'], phrased)
        model.participate(session)
        chosen = session.collect('behavior:choice')
        assert list(chosen['raw_answer'].values) == (['ran', 'mean'] if 'reply' in part else ['sat', 'kind'])
    assert seen[0] == ('reply', (('text', 'the cat'), ('text', 'Answer with exactly one of: sat, ran.')),
                       ('sat', 'ran'))
    assert seen[2] == ('answer', 'the cat', ('sat', 'ran'))


def test_raw_scores_need_a_likelihood_part():
    """A model that would have to guess or normalize is N/A, never scored."""
    from brainscore_core.compatibility import CompatibilityError
    from brainscore_core.responses import Choice
    from brainscore_core.streaming_helpers import stimulus_session
    model = BrainScoreModel(identifier='replier', supported_modalities={'text'}, reply=lambda p: 'sat')
    task = TaskContext(response=Choice(labels='options', raw_scores=True, raw_text=True))
    with pytest.raises(CompatibilityError, match='likelihood'):
        model.start_task(task)


def test_chat_parts_scores_plain_text_under_raw_text():
    """No template, no BOS: the option's tokens after the text's own tokens."""
    from brainscore_core.responses import Choice, Prompt
    from brainscore.model_helpers.replies import chat_parts
    model, processor = TinyLM2(), CharProcessor()
    _, likelihood = chat_parts(model, processor)
    seen = {}

    def forward(input_ids, attention_mask, token_type_ids=None):
        seen.setdefault('ids', []).append(input_ids)
        return TinyLM2.forward(model, input_ids, attention_mask, torch.zeros_like(input_ids))
    model.forward = forward
    prompt = Prompt((('text', 'ab'),), response=Choice(['c', 'dd'], raw_text=True))
    scores = likelihood(prompt, ['c', 'dd'])
    tok = processor.tokenizer
    for option in ('c', 'dd'):
        ids = tok(f'ab {option}')['input_ids']
        logp = model.head(model.embed(torch.tensor([ids])))[0].log_softmax(-1)
        expected = sum(float(logp[i - 1, ids[i]]) for i in range(len(tok('ab')['input_ids']), len(ids)))
        assert scores[option] == pytest.approx(expected, abs=1e-5)
    # The text alone first (the pass a recording keeps), then the options; never a template.
    assert seen['ids'][0].tolist() == [tok('ab')['input_ids']]
    assert not hasattr(processor, 'kwargs')


def test_an_answer_part_cannot_serve_a_raw_text_choice():
    """answer= takes no response settings and adds its own directive, so it is refused."""
    from brainscore_core.responses import Choice
    from brainscore_core.streaming_helpers import stimulus_session
    calls = []

    def answer(stimulus_row, instruction, label_set):
        calls.append(1)
        return label_set[0]
    model = BrainScoreModel(identifier='answerer', supported_modalities={'text'}, answer=answer)
    task = TaskContext(response=Choice(labels='options', raw_text=True))
    session = stimulus_session(_options_stimuli(), ['behavior:choice'], task)
    with pytest.raises(ValueError, match='cannot give a Choice answer'):
        model.participate(session)
    assert calls == []


@pytest.mark.parametrize('declares_control', [False, True])
def test_an_experiment_and_a_benchmark_agree_on_which_pairs_fit(declares_control):
    """A protocol is checked with the rule score() uses, including trial history and control."""
    from types import SimpleNamespace
    from brainscore_core.compatibility import check_compatibility
    from brainscore_core.contract import Subject

    class Remembers(Subject):
        identifier = 'remembers'
        in_channels = frozenset({'text', 'control'} if declares_control else {'text'})
        out_channels = frozenset({'behavior'})
        traits = frozenset({'memory:trial_history'})

        def participate(self, session):
            pass

        def reset(self):
            pass

    from brainscore.experiments import SessionProtocol
    protocol = SessionProtocol('memory-check', lambda repetition: None,
                               required_input_channels=['text'],
                               requested_output_channels=['behavior'])
    benchmark = SimpleNamespace(identifier='memory-check', required_input_channels={'text'},
                                requested_output_channels={'behavior'})
    if declares_control:
        protocol.validate(Remembers())
        assert protocol.describe()['compatibility'] == \
            check_compatibility(Remembers(), benchmark).as_attrs()
    else:
        # Trial history without control fails both ways (the old protocol rule let it pass).
        with pytest.raises(CompatibilityError, match='control'):
            check_compatibility(Remembers(), benchmark)
        with pytest.raises(CompatibilityError, match='control'):
            protocol.validate(Remembers())
