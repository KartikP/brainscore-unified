"""The concrete subject exposes its observation configuration."""
# -- Public observation surface: ``extractor`` + ``detect_modalities`` --------
# Core-only fakes (no framework). Observation tools read these instead of the
# private ``_extractors``; neither is a Subject member.

from types import MappingProxyType

import pytest

from brainscore import BrainScoreModel


class _Columns:
    def __init__(self, columns):
        self.columns = columns


class _FakeExtractor:
    """Extractor with the layer-aware signature and an episode reset."""

    def __init__(self, output):
        self.output, self.calls, self.resets = output, [], 0

    def __call__(self, stimuli, layers):
        self.calls.append((stimuli, list(layers)))
        return self.output

    def reset(self):
        self.resets += 1


def _model_and_extractors():
    vision, text = _FakeExtractor('v-out'), _FakeExtractor('t-out')
    expected = {'vision': vision, 'text': text}
    return BrainScoreModel('m', extractor=dict(expected)), expected


def test_extractor_is_read_only_normalized_mapping():
    model, expected = _model_and_extractors()
    mapping = model.extractor
    assert isinstance(mapping, MappingProxyType)
    assert set(mapping) == set(expected)
    assert all(mapping[k] is expected[k] for k in expected)  # same objects, no copies
    with pytest.raises(TypeError):
        mapping['audio'] = _FakeExtractor('a')
    with pytest.raises(TypeError):
        del mapping['vision']
    with pytest.raises(AttributeError):
        model.extractor = {}
    assert dict(model.extractor) == dict(mapping)  # unchanged by attempts


def test_reading_the_extractor_mapping_keeps_declarations():
    model, _ = _model_and_extractors()
    before = (model.in_channels, model.out_channels, model.supported_modalities)
    model.extractor  # reading the mapping must not change declarations
    assert (model.in_channels, model.out_channels, model.supported_modalities) == before
    assert model.supported_modalities == {'vision', 'text'}


@pytest.mark.parametrize('extractors, columns, expected', [
    (('vision', 'text'), ['image_file_name'], {'vision'}),
    (('vision', 'text'), ['image_path'], {'vision'}),
    (('vision', 'text'), ['filename'], {'vision'}),
    (('vision', 'text'), ['sentence'], {'text'}),
    (('vision', 'text'), ['text'], {'text'}),
    (('vision', 'text'), ['image_file_name', 'sentence'], {'vision', 'text'}),
    (('vision', 'text'), ['video_path'], {'vision'}),       # video folds into vision
    (('vision', 'text'), ['audio_path'], set()),            # no audio extractor
    (('vision', 'text'), ['mystery_col'], set()),
    (('vision',), ['image_file_name', 'sentence'], {'vision'}),  # no text extractor
    (('audio',), ['audio_path'], {'audio'}),
    (('video',), ['video_path'], {'vision'}),
])
def test_detect_modalities_reads_the_columns(extractors, columns, expected):
    model = BrainScoreModel('m', extractor={key: _FakeExtractor(key) for key in extractors})
    assert model.detect_modalities(_Columns(columns)) == expected


def test_extractor_mapping_survives_reset_and_reuse():
    vision = _FakeExtractor('v-out')
    model = BrainScoreModel('m', extractor={'vision': vision})
    mapping = model.extractor
    stimuli = _Columns(['image_path'])
    assert model.process(stimuli) == 'v-out'
    model.reset()
    assert vision.resets == 1
    assert model.extractor['vision'] is vision and dict(mapping) == {'vision': vision}
    assert model.process(stimuli) == 'v-out'
    assert [c[0] for c in vision.calls] == [stimuli, stimuli]


# Keep the existing imports usable as implementation files move into packages.


def test_extractor_access_imports_no_framework():
    import subprocess
    import sys

    code = (
        "import sys\n"
        "from brainscore import BrainScoreModel\n"
        "m = BrainScoreModel('m', extractor={'text': lambda stimuli, layers: None})\n"
        "dict(m.extractor); m.detect_modalities(type('S', (), {'columns': ['sentence']})())\n"
        "bad = [n for n in ('torch', 'transformers', 'tensorflow', 'jax') if n in sys.modules]\n"
        "assert not bad, bad\n"
    )
    subprocess.run([sys.executable, '-c', code], check=True)
