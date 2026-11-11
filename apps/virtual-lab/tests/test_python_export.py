"""Exported programs execute the public UMI workflow and preserve its data."""
from dataclasses import asdict
import io
import json
from pathlib import Path
import subprocess
import sys
import threading
from urllib.request import Request as HTTPRequest, urlopen
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
import pytest

from umi_lab.python_export import python_source, script_for_run, write_archive
from umi_lab.protocol import build_protocol
from umi_lab.workbench import Request, run_session, same_output


def saved(request, subject='export-fixture'):
    return {'kind': 'workbench', 'subject': subject, 'model_source': 'registered', 'request': asdict(request)}


def module(source, location):
    namespace = {'__name__': 'exported_experiment', '__file__': str(location / 'experiment.py')}
    exec(compile(source, 'experiment.py', 'exec'), namespace)
    return namespace


def configuration(**changes):
    value = {'identifier': 'written-answers', 'task': {'kind': 'text', 'instruction': 'Answer the question.'},
             'measurements': {'behavior': True}, 'conditions': [{'name': 'baseline'}], 'repetitions': 2}
    value.update(changes)
    return value


@pytest.mark.parametrize('kind,reply', [('text', 'A written answer.'), ('rating', '4'), ('choice', 'yes')])
def test_generated_task_matches_gui_outputs_without_needing_the_lab_at_runtime(tmp_path, kind, reply):
    from brainscore import BrainScoreModel
    calls = []
    subject = BrainScoreModel('export-fixture', supported_modalities={'text'},
                             reply=lambda prompt: calls.append(prompt) or reply)
    task = {'kind': kind, 'path': 'generation', 'labels': ['yes', 'no']}
    request = Request(mode='protocol', rows=[{'sentence': 'A stimulus'}], protocol=configuration(task=task))
    source = python_source(saved(request))
    assert 'umi_lab' not in source and 'plan.json' not in source
    exported = module(source, tmp_path)
    assert not calls  # Importing the exported file never loads/runs a model.
    generated = exported['build_experiment'](subject, tmp_path / 'generated', tmp_path)
    generated.validate()
    assert not calls
    actual = generated.run().value
    expected, _, _ = build_protocol(subject, request, tmp_path, tmp_path / 'gui')
    assert same_output(actual, expected.run().value)
    assert len(calls) == 4  # Two repetitions in each of the independent runs.


def test_fitting_joint_recordings_conditions_and_cleanup_match_gui(tmp_path):
    import torch
    from brainscore import BrainScoreModel
    from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
    from brainscore.perturbation import build_pytorch_ablation_fn
    net = torch.nn.Sequential(torch.nn.Identity())
    def extract(stimuli, layers):
        values = net(torch.tensor([[len(text), 1.] for text in stimuli['sentence']])).numpy()
        return NeuroidAssembly(values, dims=['presentation', 'neuroid'],
            coords={'stimulus_id': ('presentation', list(stimuli['stimulus_id'])),
                    'neuroid_id': ('neuroid', ['u0', 'u1']), 'layer': ('neuroid', ['0', '0'])})
    subject = BrainScoreModel('export-fixture', model=net, extractor={'text': extract},
        region_layer_map={'IT': '0'}, behavioral_readout_layer='0', intervention=build_pytorch_ablation_fn(net))
    request = Request(mode='protocol', rows=[{'sentence': 'cat'}, {'sentence': 'elephant'}],
        fitting_rows=[{'sentence': word, 'label': label} for word,label in
                      [('a','short'),('bb','short'),('giraffe','long'),('hippopotamus','long')]],
        protocol=configuration(task={'kind': 'choice', 'path': 'readout', 'readout_fit': 'intact',
                                     'labels': ['short', 'long'], 'probabilities': True},
            measurements={'behavior': True, 'regions': ['IT'], 'replay': True},
            conditions=[{'name': 'baseline'}, {'name': 'lesioned', 'intervention': {
                'target': '0', 'indices': [0], 'axis': -1, 'operation': 'zero'}}, {'name': 'restored'}]))
    exported = module(python_source(saved(request)), tmp_path)
    generated = exported['build_experiment'](subject, tmp_path / 'generated', tmp_path)
    gui, _, _ = build_protocol(subject, request, tmp_path, tmp_path / 'gui')
    assert list(generated.protocol.task.fitting_stimuli['stimulus_id']) == [f'fitting-{i}' for i in range(4)]
    assert same_output(generated.run().value, gui.run().value)
    np.testing.assert_array_equal(net(torch.ones((1,2))).detach().numpy(), np.ones((1,2)))


def test_media_paths_are_relative_and_literals_cannot_become_executable_code(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    inputs = tmp_path / 'inputs'; inputs.mkdir()
    (inputs / 'image.png').write_bytes(b'fixture')
    attack = "'''\n__import__('pathlib').Path('UNEXPECTED').touch()\n#"
    request = Request(mode='protocol', rows=[{'stimulus_id': 'one', 'image_path': 'image.png', 'sentence': attack}],
        protocol=configuration(task={'kind': 'text', 'instruction': attack, 'present': ['image', 'text']}))
    source = python_source(saved(request, subject=attack))
    assert str(tmp_path) not in source
    exported = module(source, tmp_path)
    table = exported['make_stimuli'](request.rows, inputs)
    assert table.iloc[0]['image_path'] == str(inputs / 'image.png')
    assert table.iloc[0]['sentence'] == attack and exported['SUBJECT'] == attack
    assert not (tmp_path / 'UNEXPECTED').exists()


def test_json_event_session_matches_gui_and_preserves_optional_event_time(tmp_path):
    from brainscore_core.contract import Subject
    from brainscore_core.streaming import StreamEvent
    from brainscore_core.supported_data_standards.brainio.assemblies import BehavioralAssembly
    class Echo(Subject):
        identifier = 'echo'; in_channels = {'text'}; out_channels = {'behavior:text'}
        def participate(self, session):
            while (event := session.next_input()) is not None:
                session.respond(StreamEvent('behavior:text', BehavioralAssembly([event.payload],
                    dims=['presentation']), event.t_ms))
    request = Request(mode='session', output='behavior:text', events=[{'channel': 'text', 'payload': 'hello'}], repeats=2)
    subject = Echo()
    generated = module(python_source(saved(request)), tmp_path)['build_experiment'](subject, tmp_path / 'generated')
    actual = generated.run().value
    run_session(subject, request, tmp_path / 'gui', tmp_path, lambda event: None, lambda: False)
    from brainscore.experiments import ExperimentResult
    assert same_output(actual, ExperimentResult.open(tmp_path / 'gui').value)


def test_quick_behavior_session_and_registered_stimulus_reference(tmp_path):
    from brainscore import load_model
    from brainscore.experiments import ExperimentResult
    request = Request(rows=[{'sentence': 'a question'}], labels=['yes','no'], repeats=2)
    subject = load_model('chance-baseline')
    actual = module(python_source(saved(request)), tmp_path)['build_experiment'](subject, tmp_path / 'generated', tmp_path).run()
    run_session(subject, request, tmp_path / 'gui', tmp_path, lambda event: None, lambda: False)
    assert same_output(actual.value, ExperimentResult.open(tmp_path / 'gui').value)
    request.stimulus_set = 'my-registered-stimuli'
    assert "load_stimulus_set('my-registered-stimuli')" in python_source(saved(request))


def test_help_needs_no_umi_install_and_benchmark_is_an_explicit_score_call(tmp_path):
    source = python_source(saved(Request(mode='benchmark', benchmark='fixture-benchmark')))
    assert 'return score(candidate, benchmark)' in source
    file = tmp_path / 'experiment.py'; file.write_text(source)
    result = subprocess.run([sys.executable, '-I', str(file), '--help'], capture_output=True, text=True)
    assert result.returncode == 0 and '--validate-only' in result.stdout


def test_validate_only_does_not_run_inference_or_create_a_run(monkeypatch, tmp_path, capsys):
    import brainscore
    from brainscore import BrainScoreModel
    calls = []
    subject = BrainScoreModel('export-fixture', supported_modalities={'text'}, reply=lambda prompt: calls.append(prompt) or 'answer')
    request = Request(mode='protocol', rows=[{'sentence': 'question'}], protocol=configuration())
    namespace = module(python_source(saved(request)), tmp_path)
    monkeypatch.setattr(brainscore, 'load_model', lambda identifier: subject)
    monkeypatch.setattr(sys, 'argv', ['experiment.py', '--validate-only', '--output', str(tmp_path / 'unused')])
    namespace['main']()
    assert calls == [] and not (tmp_path / 'unused').exists()
    assert 'written-answers' in capsys.readouterr().out


def test_invalid_fitting_is_rejected_before_recording(tmp_path):
    request = Request(mode='protocol', rows=[{'sentence': 'question'}], fitting_rows=[{'sentence': 'fit', 'label': ''}],
        protocol=configuration(task={'kind': 'choice', 'path': 'readout', 'labels': ['yes','no']}))
    namespace = module(python_source(saved(request)), tmp_path)
    with pytest.raises(ValueError, match='Fill the label'):
        namespace['build_experiment'](None, tmp_path / 'unused', tmp_path)


def test_profile_factories_are_explicit_and_old_archives_are_augmented_without_changes(tmp_path):
    from umi_lab.demo import create_profile
    from umi_lab.experiment import Plan, build_experiment
    directory = tmp_path / 'old-run'; directory.mkdir()
    plan = Plan(activity=True, ablation=True)
    (directory / 'plan.json').write_text(json.dumps(asdict(plan)))
    (directory / 'profile.json').write_text(json.dumps({'factory': 'umi_lab.demo:create_profile'}))
    (directory / 'summary.json').write_text('{"phase":"complete"}')
    (directory / 'inputs').mkdir(); (directory / 'inputs/example.png').write_bytes(b'input')
    original = {p.relative_to(directory): p.read_bytes() for p in directory.rglob('*') if p.is_file()}
    source = script_for_run(directory)
    assert "PROFILE_FACTORY = 'umi_lab.demo:create_profile'" in source
    profile = create_profile()
    generated = module(source, tmp_path)['build_experiment'](profile.subject, tmp_path / 'generated', profile=profile)
    expected = build_experiment(profile, plan, tmp_path / 'gui')
    assert same_output(generated.run().value, expected.run().value)
    archive = tmp_path / 'export.zip'; write_archive(directory, archive)
    with ZipFile(archive) as bundle:
        assert bundle.read('experiment.py').decode() == source
        assert bundle.read('inputs/example.png') == b'input'
        assert 'machine-readable configuration' in bundle.read('README.txt').decode()
    assert original == {p.relative_to(directory): p.read_bytes() for p in directory.rglob('*') if p.is_file()}


def test_http_exports_drafts_and_old_runs_without_loading_a_model(tmp_path):
    from umi_lab.server import make_server
    directory = tmp_path / 'run'; directory.mkdir()
    plan = saved(Request(mode='protocol', rows=[{'sentence': 'question'}], protocol=configuration()))
    (directory / 'plan.json').write_text(json.dumps(plan))
    controller = SimpleNamespace(runs={'old': directory}, token='fixture', factory='umi_lab.demo:create_profile',
                                 snapshot=lambda: {'run': None})
    server = make_server(controller, tmp_path, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    url = f'http://127.0.0.1:{server.server_port}'
    try:
        with urlopen(url + '/api/runs/old/experiment.py') as response:
            source = response.read().decode()
            compile(source, 'experiment.py', 'exec')
        with urlopen(HTTPRequest(url + '/api/export-python', json.dumps(plan).encode(),
                {'Content-Type': 'application/json', 'X-Lab-Token': 'fixture'})) as response:
            assert json.load(response)['source'] == source
        with urlopen(url + '/api/runs/old/artifacts.zip') as response:
            with ZipFile(io.BytesIO(response.read())) as bundle:
                assert bundle.read('experiment.py').decode() == source
        assert not (directory / 'experiment.py').exists()
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=2)
