"""Every stimulus column keeps the channel it fed before the column tables became one.

Before 2026-11-13 four tables said which channel a stimulus column feeds; the two
frozen below were the ones still read (the session table in ``_sessions/payloads.py``
and the catalog table that ``BrainScoreModel`` copied). The catalog's
``stimulus_columns`` is now the one answer. For every column a registered unified
benchmark sends and every column an old table knew, this test checks the session
channel and the model modality before and after.
"""

import pytest

from brainscore_core.io_catalog import canonical_modality, column_channel, modalities_to_input_channels
from brainscore_core.streaming import parse_channel

pytestmark = pytest.mark.unit

# The session table before (column -> modality; the session sent
# modalities_to_input_channels({modality})).
OLD_SESSION = {
    'vision': 'vision', 'image': 'vision', 'image_file_name': 'vision', 'image_path': 'vision',
    'filename': 'vision', 'text': 'text', 'sentence': 'text', 'audio': 'audio',
    'audio_file': 'audio', 'audio_file_name': 'audio', 'audio_path': 'audio',
    'video': 'video', 'video_path': 'video',
}
# The catalog table before (column -> modality), copied into BrainScoreModel.COLUMN_TO_MODALITY.
OLD_MODEL = {
    'image_file_name': 'vision', 'image_path': 'vision', 'filename': 'vision',
    'sentence': 'text', 'text': 'text', 'video_path': 'vision',
    'audio_path': 'audio', 'audio_file_name': 'audio', 'audio_file': 'audio',
}

# The columns of the tables registered unified benchmarks send (from the benchmark
# code; no data read). Metadata columns are listed too: they must feed nothing.
BENCHMARK_COLUMNS = {
    'Algonauts2025 image': ['stimulus_id', 'frame_id', 'clip_id', 't_within_run', 'image_file_name'],
    'Algonauts2025 video': ['stimulus_id', 'frame_id', 'clip_id', 't_within_run', 'video_path'],
    'Algonauts2025 video-audio': ['stimulus_id', 'frame_id', 'clip_id', 't_within_run',
                                  'video_path', 'audio_path'],
    'Lahner2024 clips': ['stimulus_id', 'video_path'],
    'Lahner2024 frames': ['stimulus_id', 'video_path', 'clip_id', 'frame_time_ms', 'image_file_name'],
    'Lahner2024 audio': ['stimulus_id', 'audio_path'],
    'LeBel2023 TRs': ['story_id', 'tr_index', 'tr_time_sec', 'sentence', 'stimulus_id'],
    'LeBel2023 words': ['story_id', 'word_index', 'word_time_sec', 'sentence', 'stimulus_id'],
    'Yeatman2021 image': ['stimulus_id', 'image_file_name', 'image_label', 'numeric_label',
                          'word', 'realpseudo'],
    'Yeatman2021 text': ['stimulus_id', 'sentence', 'image_label', 'numeric_label',
                         'word', 'realpseudo'],
    'EWoK logprobs': ['stimulus_id', 'text', 'options'],
    'EWoK choice': ['stimulus_id', 'text'],
    'CogGym': ['stimulus_id', 'text', 'text_2', 'image_path', 'image_path_2',
               'video_path', 'video_path_2'],
}

# Columns that feed a channel now and fed none before, on one side (named in the
# change record): the session gains video_file_name; the model gains the five
# columns only the session knew.
NEW_FOR_SESSION = {'video_file_name'}
NEW_FOR_MODEL = {'image', 'vision', 'video', 'audio', 'video_file_name'}

ALL_COLUMNS = sorted({c for cols in BENCHMARK_COLUMNS.values() for c in cols}
                     | set(OLD_SESSION) | set(OLD_MODEL) | NEW_FOR_SESSION)


def _old_session_channel(column):
    modality = OLD_SESSION.get(column)
    return None if modality is None else next(iter(modalities_to_input_channels({modality})))


def _new_model_modality(column):
    channel = column_channel(column)
    return None if channel is None else canonical_modality(parse_channel(channel)[0])


@pytest.mark.parametrize('column', ALL_COLUMNS)
def test_session_channel_is_unchanged(column):
    if column in NEW_FOR_SESSION:
        assert _old_session_channel(column) is None
        assert column_channel(column) == 'vision:video'
    else:
        assert column_channel(column) == _old_session_channel(column)


@pytest.mark.parametrize('column', ALL_COLUMNS)
def test_model_modality_is_unchanged(column):
    if column in NEW_FOR_MODEL:
        assert column not in OLD_MODEL
        assert _new_model_modality(column) == canonical_modality(OLD_SESSION.get(column, 'video'))
    else:
        old = OLD_MODEL.get(column)
        assert _new_model_modality(column) == (None if old is None else canonical_modality(old))


def test_no_registered_benchmark_sends_a_column_whose_reading_changed():
    sent = {c for cols in BENCHMARK_COLUMNS.values() for c in cols}
    assert not sent & (NEW_FOR_SESSION | NEW_FOR_MODEL)


def test_brainscore_model_detects_inputs_from_the_one_table():
    from types import SimpleNamespace
    from brainscore import BrainScoreModel

    model = BrainScoreModel('columns', extractor={'vision': lambda s, layers: None,
                                                  'video': lambda s, layers: None,
                                                  'text': lambda s, layers: None,
                                                  'audio': lambda s, layers: None})
    for columns, expected in [(['image_file_name'], {'vision'}), (['video_path'], {'vision'}),
                              (['sentence', 'audio_path'], {'text', 'audio'}),
                              (['stimulus_id', 'object_name'], set())]:
        assert model._detect_modalities(SimpleNamespace(columns=columns)) == expected
