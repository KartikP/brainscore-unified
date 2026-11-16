"""An answer or reply part opens the pictures of an old stimulus set (a bare file name and a path map)."""
import os

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from brainscore.brainscore_model import BrainScoreModel
from brainscore_core.contract import TaskContext
from brainscore_core.legacy_facade import LegacyBenchmarkFacade
from brainscore_core.responses import Choice
from brainscore_core.streaming_helpers import stimulus_session
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet


def _old_stimulus_set(folder, identifier='picture-old'):
    names = ['a.png', 'b.png', 'c.png']
    for name in names:
        Image.new('RGB', (4, 4)).save(os.path.join(folder, name))
    table = StimulusSet(pd.DataFrame({'stimulus_id': ['s1', 's2', 's3'], 'image_file_name': names,
                                      'object_name': ['x', 'y', 'x']}))
    table.identifier = identifier
    table.stimulus_paths = {sid: os.path.join(folder, name) for sid, name in zip(table['stimulus_id'], names)}
    return table


def _new_style_table(folder):
    table = _old_stimulus_set(folder, 'picture-new')
    table['image_path'] = [table.stimulus_paths[sid] for sid in table['stimulus_id']]
    table = table.drop(columns=['image_file_name'])
    return StimulusSet(table)


def _model(opened, **parts):
    def answer(stimulus_row, instruction, label_set):
        opened.append(Image.open(stimulus_row['image_file_name'] if 'image_file_name' in stimulus_row
                                 else stimulus_row['image_path']).size)
        return label_set[0]
    return BrainScoreModel(identifier='pictures', in_channels={'vision:image'}, answer=answer, **parts)


def test_an_answer_part_opens_the_picture_of_an_old_stimulus_set_through_the_facade(tmp_path):
    opened = []
    table = _old_stimulus_set(str(tmp_path))
    facade = LegacyBenchmarkFacade(_model(opened))
    facade.start_task('label', ['cat', 'dog'])
    answer = facade.look_at(table)
    assert opened == [(4, 4)] * 3                                         # every picture opened
    assert list(answer['image_file_name'].values) == ['a.png', 'b.png', 'c.png']   # the table's own values stay


def test_a_reply_part_is_shown_a_path_that_opens(tmp_path):
    shown = []

    def reply(prompt):
        shown.append([content for kind, content in prompt.parts if kind == 'image'])
        return 'cat'
    model = BrainScoreModel(identifier='pictures-reply', in_channels={'vision:image'}, reply=reply)
    table = _old_stimulus_set(str(tmp_path), 'picture-reply')
    session = stimulus_session(table, ['behavior:choice'], TaskContext(response=Choice(['cat', 'dog'])))
    model.participate(session)
    assert [Image.open(paths[0]).size for paths in shown] == [(4, 4)] * 3
    stored = session.collect('behavior:choice')
    assert list(stored['image_file_name'].values) == ['a.png', 'b.png', 'c.png'] if 'image_file_name' in stored.indexes['presentation'].names else True


def test_a_table_with_existing_paths_gives_the_same_rows(tmp_path):
    from brainscore._model.recording import openable_paths
    table = _new_style_table(str(tmp_path))
    assert openable_paths(table) is table                                  # nothing to resolve: the same table
    old = _old_stimulus_set(str(tmp_path), 'picture-old-2')
    shown = openable_paths(old)
    assert list(old['image_file_name']) == ['a.png', 'b.png', 'c.png']     # the stimulus set is not changed
    assert all(os.path.exists(path) for path in shown['image_file_name'])
    assert list(shown['stimulus_id']) == list(old['stimulus_id'])
    assert list(shown['object_name']) == list(old['object_name'])


def test_a_cell_presented_as_text_keeps_its_value(tmp_path):
    from brainscore._model.recording import openable_paths
    table = _old_stimulus_set(str(tmp_path), 'picture-text')
    table['image_path'] = table['image_file_name']
    table = StimulusSet(table.drop(columns=['image_file_name']))
    table.identifier = 'picture-text'
    table.stimulus_paths = {sid: os.path.join(str(tmp_path), f'{n}.png') for sid, n in zip(table['stimulus_id'], 'abc')}
    task = TaskContext(response=Choice(['cat', 'dog']), present=('text:image_path',))
    assert list(openable_paths(table, task)['image_path']) == ['a.png', 'b.png', 'c.png']   # text: untouched
    seen = []
    model = BrainScoreModel(identifier='text-cell', in_channels={'vision:image'},
                            reply=lambda prompt: seen.append(prompt.text()) or 'cat')
    session = stimulus_session(table, ['behavior:choice'], task)
    model.participate(session)
    assert all(name in text for name, text in zip(['a.png', 'b.png', 'c.png'], seen))
    assert str(tmp_path) not in ''.join(seen)


def test_a_string_with_a_uri_scheme_is_never_replaced(tmp_path):
    from brainscore._model.recording import openable_paths
    table = _old_stimulus_set(str(tmp_path), 'picture-uri')
    table['image_file_name'] = ['https://example.invalid/a.png', 'data:image/png;base64,AAAA', 'file:///x/c.png']
    assert openable_paths(table) is table


def test_with_two_media_cells_that_need_a_path_nothing_is_resolved(tmp_path):
    from brainscore._model.recording import openable_paths
    table = _old_stimulus_set(str(tmp_path), 'picture-two')
    table['audio_path'] = ['x.wav', 'y.wav', 'z.wav']
    assert openable_paths(table) is table                       # two bare media cells: no guess
    table['audio_path'] = [os.path.join(str(tmp_path), 'a.png')] * 3   # one cell opens already: the other is resolved
    assert all(os.path.exists(p) for p in openable_paths(table)['image_file_name'])
