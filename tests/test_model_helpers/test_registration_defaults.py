"""A registration states a setting only when it differs from the default.

Scans every registration under ``brainscore/models`` for calls of the unified
extractors and ``BrainScoreModel``; a keyword whose literal value equals the
default of that argument is a repeated default.
"""
import ast
import inspect
from pathlib import Path

import pytest

import brainscore
from brainscore import BrainScoreModel
from brainscore.model_helpers.audio_wrapper import AudioWrapper
from brainscore.model_helpers.text_wrapper import TextWrapper
from brainscore.model_helpers.video_wrapper import VideoWrapper
from brainscore.model_helpers.vlm_video_wrapper import VLMVideoWrapper
from brainscore.model_helpers.vlm_vision_wrapper import VLMVisionWrapper

pytestmark = pytest.mark.unit

CALLABLES = {part.__name__: part for part in (
    BrainScoreModel, AudioWrapper, TextWrapper, VideoWrapper, VLMVideoWrapper, VLMVisionWrapper)}
MODELS = Path(brainscore.__file__).parent / 'models'


def _repeated_defaults(path, root=MODELS):
    for node in ast.walk(ast.parse(path.read_text())):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, 'id', getattr(node.func, 'attr', None))
        if name not in CALLABLES:
            continue
        parameters = inspect.signature(CALLABLES[name]).parameters
        for keyword in node.keywords:
            parameter = parameters.get(keyword.arg)
            if parameter is None or parameter.default is inspect.Parameter.empty:
                continue
            try:
                value = ast.literal_eval(keyword.value)
            except (ValueError, TypeError, SyntaxError):
                continue
            if value == parameter.default and type(value) is type(parameter.default):
                yield f'{path.relative_to(root)}:{keyword.lineno} {name}({keyword.arg}={value!r})'


def test_no_registration_repeats_a_default():
    found = [line for path in sorted(MODELS.rglob('*.py')) for line in _repeated_defaults(path)]
    assert found == []


def test_the_scan_finds_a_repeated_default(tmp_path):
    path = tmp_path / 'model.py'
    path.write_text("TextWrapper(model, tokenizer, max_length=512, batch_size=8)\n"
                    "BrainScoreModel('m', visual_degrees=8)\n")
    assert list(_repeated_defaults(path, tmp_path)) == [
        'model.py:1 TextWrapper(max_length=512)', 'model.py:2 BrainScoreModel(visual_degrees=8)']
