"""What a local chat model is shown: ``chat_inputs`` turns one trial's Prompt into the processor call.

The task's instruction (the text part named ``'instruction'``) is the system turn.
A screen in a stimulus column of that name (``present=('image:instruction', ...)``,
the example of ``TaskContext``) is an image of the user turn, in its place.
"""
import pytest
import torch
from PIL import Image

from brainscore_core import TaskContext
from brainscore_core.responses import Choice, trial_prompt
from brainscore.model_helpers.replies import chat_inputs

pytestmark = pytest.mark.unit


class _Processor:
    """Keeps the chat messages and the processor call."""
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        self.messages = messages
        return 'templated'

    def __call__(self, **call):
        self.call = call
        return self

    def to(self, device):
        return self


class _Model(torch.nn.Module):
    """A model on the CPU: its inputs go to the CPU."""
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(1, 1)


def _screen(folder, name):
    path = folder / f'{name}.png'
    Image.new('RGB', (8, 8), 'white').save(path)
    return str(path)


def _row(folder):
    return {'stimulus_id': 's0', 'instruction': _screen(folder, 'rule'),
            'word_1': _screen(folder, 'w1'), 'word_2': _screen(folder, 'w2')}


SCREENS = ('image:instruction', 'image:word_1', 'image:word_2')


def test_a_screen_in_a_column_named_instruction_is_shown_as_an_image(tmp_path):
    prompt = trial_prompt(TaskContext(present=SCREENS, response=Choice(['yes', 'no'])), _row(tmp_path))
    processor = _Processor()
    chat_inputs(_Model(), processor, prompt)
    assert [message['role'] for message in processor.messages] == ['user']       # no text instruction: no system turn
    kinds = [part['type'] for part in processor.messages[0]['content']]
    assert kinds == ['image', 'image', 'image', 'text']                           # three screens, then the directive
    assert len(processor.call['images']) == 3


def test_the_text_instruction_is_the_system_turn_and_the_screen_stays(tmp_path):
    task = TaskContext(instruction='Rate how sure you are.', present=SCREENS, response=Choice(['yes', 'no']))
    processor = _Processor()
    chat_inputs(_Model(), processor, trial_prompt(task, _row(tmp_path)))
    system, user = processor.messages
    assert system == {'role': 'system', 'content': [{'type': 'text', 'text': 'Rate how sure you are.'}]}
    assert [part['type'] for part in user['content']] == ['image', 'image', 'image', 'text']
    assert len(processor.call['images']) == 3
