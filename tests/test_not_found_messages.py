"""A loader that finds no identifier says how a plugin registers."""
import pytest

import brainscore
from brainscore.tools import load_tool

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('load,folder', [
    (brainscore.load_model, 'models'),
    (brainscore.load_benchmark, 'benchmarks'),
    (brainscore.load_metric, 'metrics'),
    (brainscore.load_dataset, 'data'),
    (brainscore.load_stimulus_set, 'data'),
])
def test_says_a_plugin_folder_registers_by_its_presence(load, folder):
    with pytest.raises(KeyError) as raised:
        load('no-such-plugin-x7')
    message = str(raised.value)
    assert "'no-such-plugin-x7' not found" in message
    assert f'a folder brainscore/{folder}/<name>/ with an __init__.py is imported with brainscore' in message
    assert 'a plugin in another package registers when you import that package' in message


def test_tool_message_names_both_ways():
    with pytest.raises(KeyError) as raised:
        load_tool('no-such-tool-x7')
    assert '_TOOLS' in str(raised.value) and 'register_tool' in str(raised.value)
