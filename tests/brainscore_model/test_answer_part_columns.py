"""The registered answer parts read a row's picture and text by the core catalog's columns.

Since a plain Choice goes to the ``answer`` part (stage 3, S7a), the part reads the row
itself. It reads the picture (and the text) by the same rule as a trial's prompt
(``brainscore_core.responses.part_columns``): the catalog's columns, in its order. The
registered closures run around weight-free stand-ins (a stub processor and a stub
``generate``); every session goes through ``participate``.
"""
import numpy as np
import pytest
import torch
from PIL import Image

from brainscore import BrainScoreModel
from brainscore_core.contract import TaskContext
from brainscore_core.responses import Choice, part_columns
from brainscore_core.streaming_helpers import stimulus_session
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet

from tests.brainscore_model import tiny_wrappers

pytestmark = pytest.mark.unit

LABELS = ['yes', 'no']
TASK = TaskContext(instruction='Is it there?', response=Choice(LABELS))
IMAGE_COLUMNS = part_columns('image')


class _Inputs(dict):
    def to(self, device):
        return self


class _Tokenizer:
    def decode(self, ids, skip_special_tokens=True):
        return 'yes'


class _Processor:
    """Records the picture each call receives; ``on_image`` runs for each one."""

    def __init__(self, on_image=None):
        self.tokenizer, self.images, self.on_image = _Tokenizer(), [], on_image

    def _seen(self, image):
        self.images.append(np.asarray(image).tobytes())
        if self.on_image is not None:
            self.on_image(image)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
        for part in messages[0]['content']:
            if part['type'] == 'image':
                self._seen(part['image'])
        return 'templated'

    def __call__(self, text=None, images=None, return_tensors='pt', padding=False):
        if not isinstance(text, list):          # BLIP-2 passes the picture here
            self._seen(images)
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


def _qwen25(processor):
    from brainscore.models.qwen25_vl_3b.model import _make_answer
    return _make_answer(_Generator(), processor)


def _blip2(processor):
    from brainscore.models.blip2_opt_2_7b.model import _make_answer
    return _make_answer(_Generator(), processor)


def _qwen36(processor):
    from brainscore.models.qwen36_27b.model import _make_answer
    return _make_answer(_Generator(), processor)


PARTS = {'qwen2.5-vl': _qwen25, 'blip-2': _blip2, 'qwen3.6': _qwen36}


def _pictures(folder, n=3):
    rng = np.random.default_rng(0)
    paths, contents = [], {}
    for index in range(n):
        path = folder / f'p{index}.png'
        pixels = rng.integers(0, 255, (6, 6, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(path)
        paths.append(str(path))
        contents[f's{index}'] = Image.open(path).convert('RGB').tobytes()
    return paths, contents


def _table(columns, identifier):
    ids = [f's{i}' for i in range(len(next(iter(columns.values()))))]
    stimuli = StimulusSet({'stimulus_id': ids, **columns})
    stimuli.identifier = identifier
    for column, values in columns.items():
        if column in IMAGE_COLUMNS:
            stimuli.stimulus_paths = dict(zip(ids, values))
            break
    return stimuli


def _answer(part, stimuli):
    processor = _Processor()
    model = BrainScoreModel(f'answer-{part}', in_channels={'vision:image', 'text'},
                            answer=PARTS[part](processor))
    session = stimulus_session(stimuli, ['behavior:choice'], TASK)
    model.participate(session)
    return session.collect('behavior:choice'), processor


@pytest.mark.parametrize('column', IMAGE_COLUMNS)
@pytest.mark.parametrize('part', sorted(PARTS))
def test_each_catalog_picture_column_reaches_the_answer_part(tmp_path, part, column):
    paths, contents = _pictures(tmp_path)
    choice, processor = _answer(part, _table({column: paths}, f'answer-columns-{column}'))
    assert list(choice['response_state'].values) == ['valid'] * 3
    assert processor.images == [contents[f's{i}'] for i in range(3)]


@pytest.mark.parametrize('part', sorted(PARTS))
def test_two_picture_columns_take_the_catalog_s_first(tmp_path, part):
    paths, contents = _pictures(tmp_path, 6)
    first, second = IMAGE_COLUMNS[0], IMAGE_COLUMNS[1]      # image_file_name, image_path
    stimuli = _table({second: paths[3:], first: paths[:3]}, 'answer-columns-two')
    _, processor = _answer(part, stimuli)
    assert processor.images == [contents[f's{i}'] for i in range(3)]


@pytest.mark.parametrize('part', ['qwen2.5-vl', 'blip-2'])
def test_a_row_without_a_picture_column_names_the_accepted_columns(part):
    processor = _Processor()
    answer = PARTS[part](processor)
    with pytest.raises(ValueError, match=r"no image column; accepted columns, in order: "
                                         r"\['image_file_name', 'image_path'"):
        answer({'stimulus_id': 's0', 'notes': 'x'}, 'Is it there?', LABELS)
    assert processor.images == []


def test_qwen36_without_a_picture_or_text_column_names_the_accepted_text_columns():
    with pytest.raises(ValueError, match=r"no text column; accepted columns, in order: "
                                         r"\['sentence', 'text'\]"):
        _qwen36(_Processor())({'stimulus_id': 's0', 'notes': 'x'}, 'Is it there?', LABELS)


@pytest.mark.parametrize('part', sorted(PARTS))
def test_a_roar_table_gives_the_same_file_per_row(tmp_path, part):
    """ROAR's own image table (image_file_name only, ``_CHANNEL_COLUMNS``) is unchanged."""
    from brainscore.benchmarks.roar_yeatman2021.benchmark import _project_to_channel
    paths, contents = _pictures(tmp_path)
    full = _table({'image_file_name': paths, 'sentence': ['a', 'b', 'c'],
                   'image_label': ['real', 'pseudo', 'real']}, 'answer-columns-roar')
    stimuli = _project_to_channel(full, 'vision:image')
    assert 'image_file_name' in stimuli.columns
    _, processor = _answer(part, stimuli)
    assert processor.images == [contents[f's{i}'] for i in range(3)]


def test_the_cloud_step_a_plain_choice_with_a_region_on_an_image_path_table(tmp_path):
    """The failing cloud step: a plain Choice and a region recording on the Qwen form,
    with a table whose picture column is image_path."""
    wrapper, stimuli, regions, _ = tiny_wrappers.vlm_vision(tmp_path)
    assert 'image_path' in stimuli.columns and 'image_file_name' not in stimuli.columns
    layer = regions['V4'].name
    by_content = {Image.open(path).convert('RGB').tobytes(): sid
                  for sid, path in zip(stimuli['stimulus_id'], stimuli['image_path'])}

    def forward(image):       # the model's own forward pass runs the hooked tower
        tiny_wrappers.show(wrapper, stimuli, by_content[np.asarray(image).tobytes()], layer)

    processor = _Processor(on_image=forward)
    model = BrainScoreModel('answer-qwen-form', extractor={'vision': wrapper},
                            region_layer_map={'V4': regions['V4']}, answer=_qwen25(processor))
    session = stimulus_session(stimuli, ['behavior:choice', 'neural:V4'], TASK)
    model.participate(session)
    choice = session.collect('behavior:choice')
    recorded = session.collect('neural:V4')
    assert list(choice['response_state'].values) == ['valid'] * 3
    assert list(recorded['stimulus_id'].values) == ['i0', 'i1', 'i2']
    assert len(processor.images) == 3
