"""An answer part returns the generated text exactly; BrainScoreModel parses it and stores it.

The registered closures (Qwen2.5-VL, BLIP-2, Qwen3.6-27B, the API answer part) run around
weight-free stand-ins whose generation decodes to a fixed reply; every session goes
through ``participate``. The stored raw answer is the text as generated, for a valid and
for an invalid reply; the choice is read by core's ``Choice.parse``.
"""
import numpy as np
import pytest
import torch
from PIL import Image

from brainscore import BrainScoreModel
from brainscore_core.contract import TaskContext
from brainscore_core.response_formats import response_record
from brainscore_core.responses import Choice
from brainscore_core.streaming import StreamEvent
from brainscore_core.streaming_helpers import behavioral_response, stimulus_session
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet

pytestmark = pytest.mark.unit

LABELS = ['real', 'pseudo']
TASK = TaskContext(instruction='Is this a real word or a pseudo word?', response=Choice(LABELS))
INVALID = '  Pseudo, not REAL  '
VALID = 'REAL.'


class _Inputs(dict):
    def to(self, device):
        return self


class _Tokenizer:
    def __init__(self, reply):
        self.reply, self.echo = reply, ''

    def decode(self, ids, skip_special_tokens=True):
        return self.echo + self.reply


class _Processor:
    def __init__(self, reply, echo=False):
        self.tokenizer, self.echoes = _Tokenizer(reply), echo

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
        return 'templated'

    def __call__(self, text=None, images=None, return_tensors='pt', padding=False):
        if self.echoes and not isinstance(text, list):
            self.tokenizer.echo = text             # BLIP-2's generation returns its prompt first
        return _Inputs(input_ids=torch.zeros((1, 3), dtype=torch.long))


class _Generator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))

    @property
    def device(self):
        return torch.device('cpu')

    def generate(self, **kwargs):
        return torch.zeros((1, 5), dtype=torch.long)


def _qwen25(reply):
    from brainscore.models.qwen25_vl_3b.model import _make_answer
    return _make_answer(_Generator(), _Processor(reply))


def _blip2(reply):
    from brainscore.models.blip2_opt_2_7b.model import _make_answer
    return _make_answer(_Generator(), _Processor(reply, echo=True))


def _qwen36(reply):
    from brainscore.models.qwen36_27b.model import _make_answer
    return _make_answer(_Generator(), _Processor(reply))


def _api(reply):
    from brainscore.model_helpers.api_behavioral import build_api_generation_fn

    def client(model, system, messages, settings):
        return reply
    return build_api_generation_fn(client, 'stand-in')


PARTS = {'qwen2.5-vl': _qwen25, 'blip-2': _blip2, 'qwen3.6': _qwen36, 'api': _api}


@pytest.fixture
def stimuli(tmp_path):
    path = tmp_path / 'w.png'
    Image.new('RGB', (4, 4)).save(path)
    table = StimulusSet({'stimulus_id': ['w1'], 'image_file_name': [str(path)],
                         'image_label': ['real']})
    table.identifier = 'answer-raw-text'
    table.stimulus_paths = {'w1': str(path)}
    return table


def _choice(answer, stimuli):
    model = BrainScoreModel('answer-raw', in_channels={'vision:image', 'text'}, answer=answer)
    session = stimulus_session(stimuli, ['behavior:choice'], TASK)
    model.participate(session)
    return session.collect('behavior:choice')


@pytest.mark.parametrize('part', sorted(PARTS))
def test_an_invalid_reply_is_stored_exactly_with_its_state_and_reason(part, stimuli):
    choice = _choice(PARTS[part](INVALID), stimuli)
    assert list(choice['raw_answer'].values) == [INVALID]
    assert list(choice['response_state'].values) == ['invalid']
    assert list(choice['parse_error'].values) == ["several labels in the reply: ['real', 'pseudo']"]
    assert np.isnan(choice.values).all()
    assert choice.attrs['response'] == response_record('behavior:choice', 'generation',
                                                       parsed_by='model')


@pytest.mark.parametrize('part', sorted(PARTS))
def test_a_valid_reply_is_stored_exactly_and_parsed_to_its_label(part, stimuli):
    choice = _choice(PARTS[part](VALID), stimuli)
    assert list(choice['raw_answer'].values) == [VALID]
    assert list(choice['response_state'].values) == ['valid']
    np.testing.assert_array_equal(choice.values, [[1.0, 0.0]])       # real


def test_an_answer_function_that_returns_a_label_still_works(stimuli):
    """A label is a text that parses to itself (the lessons' and guides' answer functions)."""
    choice = _choice(lambda stimulus_row, instruction, label_set: label_set[1], stimuli)
    assert list(choice['raw_answer'].values) == ['pseudo']
    np.testing.assert_array_equal(choice.values, [[0.0, 1.0]])


def test_an_answer_may_return_a_stream_event_with_extra_data(stimuli):
    model = BrainScoreModel('answer-event', in_channels={'vision:image'},
                            answer=lambda stimulus_row, instruction, label_set:
                            StreamEvent('behavior', VALID, None, {'token_usage': 3}))
    session = stimulus_session(stimuli, ['behavior:choice'], TASK)
    model.participate(session)
    (event,) = [e for e in session.emitted if e.channel == 'behavior:choice']
    assert event.meta['token_usage'] == 3
    assert list(session.collect('behavior:choice')['raw_answer'].values) == [VALID]


def test_a_legacy_task_reads_the_answer_text_with_the_same_parse(stimuli):
    """A legacy task_type task (bare behavior) parses the text with Choice.parse too."""
    answers = iter([VALID, INVALID])
    model = BrainScoreModel('answer-legacy', in_channels={'vision:image'},
                            answer=lambda stimulus_row, instruction, label_set: next(answers))
    table = StimulusSet({'stimulus_id': ['w1', 'w2'],
                         'image_file_name': list(stimuli['image_file_name']) * 2})
    table.identifier = 'answer-raw-legacy'
    table.stimulus_paths = dict(zip(table['stimulus_id'], table['image_file_name']))
    with pytest.warns(UserWarning, match='names no single label'):
        answered = behavioral_response(model, table, TaskContext(
            task_type='probabilities', label_set=LABELS, instruction='Real or pseudo?'))
    assert list(answered['raw_answer'].values) == [VALID, INVALID]
    assert list(answered['answer_valid'].values) == [True, False]
    np.testing.assert_array_equal(answered.values[0], [1.0, 0.0])
    assert np.isnan(answered.values[1]).all()
