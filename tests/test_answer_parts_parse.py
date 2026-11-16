"""The answer parts of Qwen2.5-VL and BLIP-2: their own prompt wording; the generated text,
which BrainScoreModel parses with core's Choice.parse.

No weights: a stub processor and a stub generate stand in for the model. The old parse
of the two parts is kept here as a frozen reference, to show the only replies whose label
changes: a reply with both labels, and a reply whose label is not on its first line.
"""
import re

import pytest
import torch
from PIL import Image

from brainscore_core.responses import Choice

pytestmark = pytest.mark.unit

LABELS = ['real', 'pseudo']


def _old_parse(response, label_set):
    """The parse of both answer parts before they used core's Choice parse (unified 01fd1d2), on the lower-cased reply."""
    for lbl in label_set:
        if response == lbl.lower():
            return lbl
    for lbl in label_set:
        if re.search(rf'\b{re.escape(lbl.lower())}\b', response):
            return lbl
    return response


class _Tokenizer:
    """Decodes to ``echo + reply``; ``echo`` is the prompt when the stand-in echoes it."""

    def __init__(self, reply):
        self.reply, self.echo = reply, ''

    def decode(self, ids, skip_special_tokens=True):
        return self.echo + self.reply


class _Inputs(dict):
    def to(self, device):
        return self


class _Processor:
    def __init__(self, reply, echo=False):
        self.tokenizer, self.texts, self.echoes = _Tokenizer(reply), [], echo

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        (message,) = messages
        self.texts.append(next(part['text'] for part in message['content'] if part['type'] == 'text'))
        return 'templated'

    def __call__(self, text=None, images=None, return_tensors='pt', padding=False):
        if not isinstance(text, list):          # BLIP-2 passes the prompt itself
            self.texts.append(text)
            if self.echoes:                      # its generation returns the prompt first
                self.tokenizer.echo = text
        return _Inputs(input_ids=torch.zeros((1, 3), dtype=torch.long))


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def generate(self, **kwargs):
        return torch.zeros((1, 5), dtype=torch.long)


@pytest.fixture
def row(tmp_path):
    path = tmp_path / 'w.png'
    Image.new('RGB', (4, 4)).save(path)
    return {'stimulus_id': 'w', 'image_file_name': str(path)}


def _qwen(reply):
    from brainscore.models.qwen25_vl_3b.model import _make_answer
    processor = _Processor(reply)
    return _make_answer(_Model(), processor), processor


def _blip(reply, echo=False):
    from brainscore.models.blip2_opt_2_7b.model import _make_answer
    processor = _Processor(reply, echo)
    return _make_answer(_Model(), processor), processor


def test_qwen_prompt_with_an_instruction_is_unchanged(row):
    answer, processor = _qwen('real')
    assert answer(row, 'Is this a real word?', LABELS) == 'real'
    assert processor.texts == ['Is this a real word? Answer with exactly one of: real, pseudo.']


def test_qwen_answers_without_an_instruction(row):
    answer, processor = _qwen('pseudo')
    assert answer(row, None, LABELS) == 'pseudo'
    assert processor.texts == ['Answer with exactly one of: real, pseudo.']


def test_blip2_prompt_with_an_instruction_is_unchanged(row):
    answer, processor = _blip(' real', echo=True)
    assert answer(row, 'Is this a real word?', LABELS) == ' real'      # the echo removed, nothing else
    assert processor.texts == ['Question: Is this a real word? Choose from: real, pseudo. Answer:']


def test_blip2_answers_without_an_instruction_and_shows_no_none(row):
    answer, processor = _blip('pseudo')
    assert answer(row, None, LABELS) == 'pseudo'
    assert processor.texts == ['Choose from: real, pseudo. Answer:']
    assert 'None' not in processor.texts[0]


@pytest.mark.parametrize('part', [_qwen, _blip])
@pytest.mark.parametrize('reply', ['  Pseudo, not REAL  ', 'REAL.', 'Answer: real'])
def test_the_generated_text_is_returned_exactly(row, part, reply):
    """No change of case or spaces, no label: BrainScoreModel parses the text."""
    answer, _ = part(reply)
    assert answer(row, 'Is this a real word?', LABELS) == reply


# Replies a generation part can give for ROAR and induced dyslexia (labels real, pseudo),
# with max_new_tokens 8: the label alone, with punctuation or case, in a sentence, a word
# that contains a label, empty, unrelated, both labels, a label after the first line.
REPLIES = ['real', 'pseudo', 'Real', 'PSEUDO', ' real ', 'real.', 'pseudo!', '"real"',
           'it is real', 'the word is pseudo', 'unreal', 'pseudoword', '', 'maybe', 'yes',
           'real or pseudo', 'pseudo, not real', 'not real', '\nreal', 'hmm\npseudo',
           'real\npseudo']
# 'real\npseudo' keeps 'real': the old parse took the first label found, the new one the first line.
STATED = {'real or pseudo', 'pseudo, not real', 'hmm\npseudo'}


def test_the_label_changes_only_for_two_labels_or_a_label_after_the_first_line():
    """The old parts read the lower-cased, stripped reply; now Choice.parse reads the exact text."""
    changed = set()
    for reply in REPLIES:
        old = _old_parse(reply.strip().lower(), LABELS)
        old = old if old in LABELS else None
        new, _ = Choice(LABELS).parse(reply)
        if new != old:
            changed.add(reply)
    assert changed == STATED
