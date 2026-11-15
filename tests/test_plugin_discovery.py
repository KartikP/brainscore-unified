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
    package('broken', "import brainscore\n"
                      "def build():\n    return 'partial'\n"
                      "brainscore.model_registry['discovery-partial'] = build\n"
                      "brainscore.model_registry['discovery-partial-lambda'] = lambda: 'partial'\n"
                      "import a_module_that_does_not_exist_here\n")
    package('_private', 'raise RuntimeError("a folder starting with _ is not a plugin")\n')
    (tmp_path / 'helper.py').write_text('raise RuntimeError("a module is not a plugin")\n')
    name = f'brainscore.{KIND}'
    module = types.ModuleType(name)
    module.__path__ = [str(tmp_path)]
    monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(brainscore, '_plugin_import_failures', {})
    monkeypatch.setattr(brainscore, '_plugin_failed_files', {})
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


def test_import_brainscore_loads_only_the_init_of_a_dormant_plugin():
    """MIRAGE registers nothing and is on hold: the discovery imports its __init__ only."""
    import subprocess
    code = ("import sys, brainscore\n"
            "loaded = sorted(m for m in sys.modules if m.startswith('brainscore.models.mirage.'))\n"
            "assert 'brainscore.models.mirage' in sys.modules\n"
            "assert not loaded, loaded\n"
            "from brainscore.models.mirage import MirageRuntime, MirageSubject\n"
            "assert MirageSubject.__module__ == 'brainscore.models.mirage.model'\n"
            "assert MirageRuntime.__module__ == 'brainscore.models.mirage.runtime'\n")
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-2000:]


def test_a_failed_plugin_leaves_no_entry_it_registered_before_its_error(kind_folder):
    brainscore._import_plugins(kind_folder)
    assert 'discovery-partial' not in brainscore.model_registry          # a factory of the package
    assert 'discovery-partial-lambda' not in brainscore.model_registry   # a statement in its folder
    assert 'chance-baseline' in brainscore.model_registry                # other plugins stay


def test_a_failed_plugin_that_names_the_identifier_refuses_its_load(kind_folder):
    brainscore._import_plugins(kind_folder)
    with pytest.raises(brainscore.LoadError) as refused:
        brainscore._refuse_failed_owner('Model', 'discovery-partial', KIND)
    message = str(refused.value)
    assert f'{kind_folder}.broken' in message and 'a_module_that_does_not_exist_here' in message
    brainscore._refuse_failed_owner('Model', 'another-identifier', KIND)   # not named: no refusal


@pytest.fixture
def old_registries(monkeypatch):
    """Stand-ins for the old registries: vision does not know the identifier, language does."""
    import brainscore_language
    import brainscore_vision
    from brainscore_core.plugin_management.import_plugin import PluginNotFoundError

    def unknown(identifier):
        raise PluginNotFoundError(identifier)
    monkeypatch.setattr(brainscore_vision, 'load_model', unknown)
    monkeypatch.setattr(brainscore_language, 'load_model', lambda identifier: f'old {identifier}')


def test_load_model_does_not_fall_back_past_a_failed_unified_owner(tmp_path, monkeypatch, old_registries):
    init = tmp_path / '__init__.py'
    init.write_text("model_registry['overlap-id'] = build\n")
    monkeypatch.setattr(brainscore, '_plugin_import_failures',
                        {'brainscore.models.standin': 'ImportError: no module named x'})
    monkeypatch.setattr(brainscore, '_plugin_failed_files', {'brainscore.models.standin': init})
    with pytest.raises(brainscore.LoadError, match=r'brainscore\.models\.standin.*no module named x'):
        brainscore.load_model('overlap-id')
    assert brainscore.load_model('other-id') == 'old other-id'          # not named: the fallback


def test_with_no_failed_plugin_the_fallback_is_unchanged(monkeypatch, old_registries):
    monkeypatch.setattr(brainscore, '_plugin_import_failures', {})
    monkeypatch.setattr(brainscore, '_plugin_failed_files', {})
    assert brainscore.load_model('overlap-id') == 'old overlap-id'
