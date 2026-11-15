"""A plugin folder registers by its presence: the kind folders import every plugin sub-package.

A synthetic kind folder in a temporary directory holds a good plugin, a broken one, a
folder whose name starts with ``_`` and a plain helper module. Only the good plugin and
the broken one are imported; the broken one is named with its cause and hides nothing.
"""
import logging
import sys
import types

import pytest

import brainscore

pytestmark = pytest.mark.unit

KIND = 'plugindiscoveryfake'


@pytest.fixture
def kind_folder(tmp_path, monkeypatch):
    def package(name, body):
        folder = tmp_path / name
        folder.mkdir()
        (folder / '__init__.py').write_text(body)
    package('good', 'import brainscore\nbrainscore._fake_registered = True\n')
    package('broken', 'import a_module_that_does_not_exist_here\n')
    package('_private', 'raise RuntimeError("a folder starting with _ is not a plugin")\n')
    (tmp_path / 'helper.py').write_text('raise RuntimeError("a module is not a plugin")\n')
    name = f'brainscore.{KIND}'
    module = types.ModuleType(name)
    module.__path__ = [str(tmp_path)]
    monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(brainscore, '_plugin_import_failures', {})
    yield name
    for plugin in ('good', 'broken'):
        sys.modules.pop(f'{name}.{plugin}', None)
    if hasattr(brainscore, '_fake_registered'):
        del brainscore._fake_registered


def test_every_plugin_folder_is_imported_and_a_broken_one_is_named(kind_folder, caplog):
    with caplog.at_level(logging.WARNING, logger='brainscore'):
        brainscore._import_plugins(kind_folder)
    assert brainscore._fake_registered is True                    # the good plugin registered
    assert f'{kind_folder}.good' in sys.modules
    assert f'{kind_folder}._private' not in sys.modules
    assert f'{kind_folder}.helper' not in sys.modules
    failures = brainscore._plugin_import_failures
    assert list(failures) == [f'{kind_folder}.broken']
    assert 'a_module_that_does_not_exist_here' in failures[f'{kind_folder}.broken']
    assert f'plugin {kind_folder}.broken failed to import' in caplog.text
    message = str(brainscore._not_found('Thing', 'x', 'unified', KIND))
    assert f'{kind_folder}.broken (ModuleNotFoundError' in message
    assert 'failed to import' in message


def test_the_not_found_message_needs_no_parent_edit():
    message = str(brainscore._not_found('Model', 'x', 'unified', 'models'))
    assert 'with an __init__.py is imported with brainscore' in message
    assert 'add' not in message.split('registries.')[1].split('A plugin')[0]
    assert "from . import" not in message
