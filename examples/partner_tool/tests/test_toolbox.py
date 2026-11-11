"""Run the installed examples and check the measurements they claim to show."""

from importlib import metadata
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from brainscore.experiments import read_events
from brainscore_core import io_catalog
from brainscore_core.streaming import Session, StreamEvent
from brainscore_core.supported_data_standards.brainio.assemblies import DataAssembly
import umi_partner_tool
from umi_partner_tool.toolbox import EXAMPLES, make_example, run, saved_values


def test_examples_import_from_the_installed_distribution():
    distribution = metadata.distribution('umi-partner-tool-example')
    installed = Path(distribution.locate_file('umi_partner_tool/__init__.py')).resolve()
    assert Path(umi_partner_tool.__file__).resolve() == installed
    assert 'site-packages' in installed.parts


@pytest.mark.parametrize('name', EXAMPLES)
def test_record_intervene_restore_and_replay(tmp_path, name):
    summary = run(directory=tmp_path / 'examples', examples=(name,))[name]
    assert summary['equal'] and summary['recovery_exact']
    assert summary['intervention_changed_output'] and summary['synthetic']
    original = tmp_path / 'examples' / name
    events = read_events(original)
    outputs = [event for event in events if event['kind'] == 'output']
    assert len(outputs) == 18  # an action has a proposal and a receipt
    assert {event['condition'] for event in outputs} == {'baseline', 'silenced', 'restored'}
    for event in outputs:
        response = event['payload']
        assert not io_catalog.payload_type_notes(response.channel, response.payload)
        assert isinstance(response.payload, np.ndarray if name == 'robotics' else DataAssembly)
        provenance = response.meta['provenance']
        assert provenance['device'] == 'cpu' and provenance['dtype'] == 'float32'
        assert len(provenance['weights_sha256']) == 64
        assert provenance['packages']['umi-partner-tool-example']
    if name == 'robotics':
        assert sum(event['payload'].meta['action_status'] == 'applied' for event in outputs) == 9
    # An actual forward pass, rather than a fabricated activity array, was recorded.
    activity = [event for event in events if event['kind'] == 'activity']
    assert len(activity) == 9
    for event in activity:
        values = event['payload']['value']['array']
        if event['condition'] == 'silenced':
            np.testing.assert_array_equal(values, np.zeros_like(values))
        else:
            assert np.any(values != 0)
    assert json.loads((tmp_path / 'examples' / 'summary.json').read_text())[name] == summary


@pytest.mark.parametrize('changed_channel', ['vision', 'text'])
def test_multimodal_example_uses_both_inputs(changed_channel):
    subject, _, _ = make_example('vlm')
    outputs = []
    for index in range(2):
        image = np.full((4, 4, 3), index if changed_channel == 'vision' else 1, dtype=np.uint8)
        text = 'three different words' if changed_channel == 'text' and index else 'one'
        subject.reset()
        session = Session([
                StreamEvent(
                    channel='vision', payload=image,
                    t_ms=None, meta={'stimulus_id': 's'},
                ),
                StreamEvent(channel='text', payload=text, t_ms=None, meta={'stimulus_id': 's'}),
            ], ['behavior'])
        subject.participate(session)
        outputs.append(session.collect(channel='behavior').values)
    assert not np.array_equal(*outputs)
    assert subject.required_channels == {'vision', 'text'}


def test_missing_multimodal_input_fails_without_a_response():
    subject, _, _ = make_example('vlm')
    session = Session([StreamEvent(
            channel='text', payload='incomplete', t_ms=None, meta={'stimulus_id': 's'},
        )], ['behavior'])
    with pytest.raises(ValueError, match='all its inputs'):
        subject.participate(session)
    assert session.emitted == []


def test_recurrent_subject_carries_state_then_resets(tmp_path):
    run(directory=tmp_path / 'examples', examples=('brain_behavior',))
    responses = saved_values(tmp_path / 'examples' / 'brain_behavior')['baseline']
    # Each input yields behavior then activity. Repeated inputs accumulate memory.
    assert not np.array_equal(responses[0], responses[2])


def test_feedback_uses_the_action_and_keeps_step_counts_out_of_time(tmp_path):
    run(directory=tmp_path / 'examples', examples=('robotics',))
    events = read_events(tmp_path / 'examples' / 'robotics')
    states = {}
    for event in events:
        if event['kind'] == 'input' and event['payload'].channel == 'observation':
            row = event['payload']
            assert row.t_ms is None
            states.setdefault(event['condition'], []).append(row.payload['state'])
    assert not np.array_equal(states['baseline'][0], states['baseline'][1])
    np.testing.assert_array_equal(states['silenced'][0], states['silenced'][1])


def test_existing_output_is_preserved(tmp_path):
    directory = tmp_path / 'existing'
    directory.mkdir()
    marker = directory / 'keep.txt'
    marker.write_text('existing measurement')
    with pytest.raises(FileExistsError):
        run(directory=directory, examples=('vision',))
    assert marker.read_text() == 'existing measurement'


def test_failed_inference_leaves_a_failed_record_and_removes_tools(tmp_path, monkeypatch):
    from umi_partner_tool import toolbox

    subject, factory, layer = make_example('vision')

    def fail(inputs):
        raise RuntimeError('example inference failed')

    subject.encode = fail
    monkeypatch.setattr(toolbox, 'make_example', lambda name: (subject, factory, layer))
    with pytest.raises(RuntimeError, match='example inference failed'):
        run(directory=tmp_path / 'examples', examples=('vision',))
    output = tmp_path / 'examples' / 'vision'
    assert json.loads((output / 'experiment.json').read_text())['status'] == 'failed'
    assert json.loads((output / 'inputs_outputs' / 'manifest.json').read_text())['status'] == 'failed'
    assert not layer._forward_hooks


def test_module_runs_outside_checkout_with_isolated_python(tmp_path):
    result = subprocess.run(
        [sys.executable, '-I', '-m', 'umi_partner_tool.toolbox', '--out', str(tmp_path / 'run')],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads(result.stdout)
    assert set(summary) == set(EXAMPLES)
    assert all(case['equal'] and case['recovery_exact'] for case in summary.values())
