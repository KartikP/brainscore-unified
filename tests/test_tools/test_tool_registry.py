"""The tool registry is one table and one loader; a tool is imported when it is asked for."""
import subprocess
import sys

import pytest

from brainscore.tools import list_tools, load_tool
from tests.heavy_import import heavy_import_message

MINIMAL = {
    'ablate': {'targets': ['population']},
    'contrastive-steering': {'target': 'population'},
    'leace': {'target': 'population'},
    'linear-probe': {'target': 'population'},
    'patch-activity': {'targets': ['population'], 'reference': {}},
    'record-activity': {'targets': ['population']},
    'record-io': {},
    'record-reasoning': {},
    'scale-activity': {'targets': ['population'], 'factor': 0.5},
}


def test_the_minimal_configurations_cover_every_registered_tool():
    missing, extra = sorted(set(list_tools()) - set(MINIMAL)), sorted(set(MINIMAL) - set(list_tools()))
    assert not missing, (f'add a minimal configuration for {missing} to MINIMAL in tests/test_tools/test_tool_registry.py: '
                         'the next test loads every registered tool with it, offline')
    assert not extra, f'MINIMAL names tools that are not registered: {extra}; remove their lines'


@pytest.mark.parametrize('identifier', sorted(MINIMAL))
def test_each_registered_tool_loads_through_the_one_loader(identifier):
    if identifier == 'leace':
        pytest.importorskip('concept_erasure')       # an optional package
    tool = load_tool(identifier, **MINIMAL[identifier])
    assert tool is not load_tool(identifier, **MINIMAL[identifier])      # a fresh tool each time


def test_an_unknown_tool_raises_and_lists_the_names():
    with pytest.raises(KeyError) as raised:
        load_tool('no-such-tool')
    assert all(name in str(raised.value) for name in list_tools())


def test_importing_the_package_imports_no_tool_module_and_no_model_library():
    code = (
        "import sys, brainscore, brainscore.tools\n"
        "loaded = [m for m in ('brainscore.tools.linear_probe', 'brainscore.tools.leace',\n"
        "                      'brainscore.tools.contrastive_steering', 'brainscore.experiments',\n"
        "                      'torch', 'transformers', 'sklearn') if m in sys.modules]\n"
        "assert not loaded, loaded\n"
    )
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert result.returncode == 0, heavy_import_message('Importing brainscore.tools', result.stderr)
