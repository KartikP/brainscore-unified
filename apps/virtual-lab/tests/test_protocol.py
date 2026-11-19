"""GUI plans execute the same public protocol as the lessons."""
from copy import deepcopy
import json

import numpy as np
import pytest

from umi_lab.protocol import ProtocolSpec, build_protocol, run_protocol, validate_protocol
from umi_lab.workbench import Request
from umi_lab.observations import observation_page
from umi_lab.activation_data import activation_page


def spec(**changes):
    value = {'identifier': 'animal-or-vehicle',
             'task': {'kind': 'choice', 'labels': ['short', 'long'], 'probabilities': True,
                      'path': 'readout', 'readout_fit': 'current'},
             'measurements': {'behavior': True, 'regions': ['IT'], 'replay': False},
             'conditions': [{'name': 'baseline'}], 'repetitions': 1}
    value.update(changes)
    return value


def request(configuration=None):
    return Request(mode='protocol', protocol=configuration or spec(),
                   rows=[{'stimulus_id': 'test-a', 'sentence': 'cat'},
                         {'stimulus_id': 'test-b', 'sentence': 'elephant'}],
                   fitting_rows=[{'stimulus_id': f'fit-{index}', 'sentence': sentence, 'label': label}
                                 for index, (sentence, label) in enumerate([('a', 'short'), ('bb', 'short'),
                                                                          ('giraffes', 'long'), ('hippopotamus', 'long')])])


@pytest.fixture
def subject():
    from brainscore import BrainScoreModel
    from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
    calls = []
    def extract(stimuli, layers):
        calls.append(list(stimuli['stimulus_id']))
        return NeuroidAssembly(np.array([[len(text), 1.] for text in stimuli['sentence']]),
            dims=['presentation', 'neuroid'],
            coords={'stimulus_id': ('presentation', list(stimuli['stimulus_id'])),
                    'sentence': ('presentation', list(stimuli['sentence'])),
                    'neuroid_id': ('neuroid', ['u0', 'u1']), 'layer': ('neuroid', ['units', 'units'])})
    return BrainScoreModel('protocol-fixture', extractor={'text': extract},
                           region_layer_map={'IT': 'units'}, behavioral_readout_layer='units'), calls


def test_preflight_builds_taskprotocol_and_recorders_without_running_subject(subject, tmp_path):
    from brainscore.experiments import TaskProtocol, RecordBehavior, RecordNeural
    model, calls = subject
    experiment, _, _ = build_protocol(model, request(), tmp_path)
    assert isinstance(experiment.protocol, TaskProtocol)
    assert any(isinstance(tool, RecordBehavior) for tool in experiment.tools)
    assert any(isinstance(tool, RecordNeural) for tool in experiment.tools)
    summary = validate_protocol(model, request(), tmp_path)
    assert set(summary['channels']) == {'behavior:choice', 'behavior:probabilities', 'neural:IT'}
    assert summary['presentations'] == 2
    assert calls == []


def test_passive_export_matches_gui_without_response_labels(subject, tmp_path):
    from dataclasses import asdict
    from umi_lab.python_export import python_source
    from umi_lab.workbench import same_output
    model, calls = subject
    configuration = spec(task={'kind': 'passive', 'labels': ['yes', 'no']},
                         measurements={'behavior': False, 'regions': ['IT']})
    passive_request = request(configuration)
    passive_request.fitting_rows = []
    source = python_source({'kind': 'workbench', 'subject': model.identifier,
                            'model_source': 'registered', 'request': asdict(passive_request)})
    namespace = {'__name__': 'exported_experiment', '__file__': str(tmp_path / 'experiment.py')}
    exec(compile(source, 'experiment.py', 'exec'), namespace)
    generated = namespace['build_experiment'](model, tmp_path / 'generated', tmp_path)
    gui, _, _ = build_protocol(model, passive_request, tmp_path, tmp_path / 'gui')
    assert generated.protocol.task.label_set is None
    assert gui.protocol.task.label_set is None
    assert 'label_set=' not in source
    assert 'umi-lab-protocol-builder-v1' not in source
    assert calls == []
    assert same_output(generated.run().value, gui.run().value)


def test_protocol_matches_manual_session_and_collects_neural_and_behavior_once(subject, tmp_path):
    from brainscore_core.streaming_helpers import stimulus_session
    from brainscore.experiments import ExperimentResult
    model, calls = subject
    experiment, _, _ = build_protocol(model, request(), tmp_path)
    session = stimulus_session(experiment.protocol.stimuli, ['behavior:choice', 'behavior:probabilities', 'neural:IT'],
                               experiment.protocol.task)
    model.participate(session)
    expected_behavior, expected_neural = session.collect('behavior:choice'), session.collect('neural:IT')
    model.reset()
    calls.clear()
    result = run_protocol(model, request(), tmp_path / 'experiment', tmp_path, lambda update: None, lambda: False)
    assert result['report']['kind'] == 'protocol'
    assert calls.count(['test-a', 'test-b']) == 1
    saved = ExperimentResult.open(tmp_path / 'experiment')
    np.testing.assert_array_equal(saved.behavior(condition='baseline', repetition=0, measure='choice'), expected_behavior)
    np.testing.assert_array_equal(saved.neural('IT', condition='baseline', repetition=0), expected_neural)
    assert len(saved.trials()) == 2
    outputs = observation_page(tmp_path / 'experiment', index=0)
    assert {output['channel'] for output in outputs} == set(result['report']['channels'])
    last = activation_page(tmp_path / 'experiment', channel='neural:IT', offset=1, limit=1)
    assert last['values'] == [1.]


def test_conditions_and_repetitions_keep_separate_saved_results(subject, tmp_path):
    from brainscore.experiments import ExperimentResult
    model, calls = subject
    configuration = spec(conditions=[{'name': 'first'}, {'name': 'second'}], repetitions=2)
    run_protocol(model, request(configuration), tmp_path / 'experiment', tmp_path, lambda update: None, lambda: False)
    assert calls.count(['test-a', 'test-b']) == 4
    saved = ExperimentResult.open(tmp_path / 'experiment')
    trials = saved.trials()
    assert set(trials['condition']) == {'first', 'second'}
    assert set(trials['repetition']) == {0, 1}
    assert len(trials) == 8
    assert observation_page(tmp_path / 'experiment', index=3, channel='neural:IT')[0]['result_index'] == 3


def test_missing_region_trait_and_response_fail_without_inference(subject, tmp_path):
    model, calls = subject
    invalid = [spec(measurements={'behavior': True, 'regions': ['missing']}),
               spec(required_traits=['memory:trial_history']),
               spec(task={'kind': 'text'}, measurements={'behavior': False, 'regions': ['IT']})]
    for configuration in invalid:
        with pytest.raises(ValueError):
            validate_protocol(model, request(configuration), tmp_path)
    assert calls == []


def test_history_memory_needs_the_trait_and_the_record_shows_traits_and_memory(subject, tmp_path):
    model, calls = subject
    history = spec(task={**spec()['task'], 'trial_memory': 'history'})
    with pytest.raises(ValueError, match='memory:trial_history'):
        validate_protocol(model, request(history), tmp_path)
    assert calls == []
    with pytest.raises(ValueError):
        ProtocolSpec.parse(spec(task={**spec()['task'], 'trial_memory': 'session'}))
    result = run_protocol(model, request(), tmp_path / 'experiment', tmp_path, lambda update: None, lambda: False)
    assert result['report']['trial_memory'] == 'none'
    assert result['report']['subject_traits'] == []
    assert validate_protocol(model, request(), tmp_path)['trial_memory'] == 'none'


def test_multimodal_trial_order_is_checked_before_any_forward(subject, tmp_path):
    model, calls = subject
    (tmp_path / 'image.png').write_bytes(b'not read during structural validation')
    proposed = request()
    proposed.rows[0]['image_path'] = 'image.png'
    with pytest.raises(ValueError, match='order'):
        validate_protocol(model, proposed, tmp_path)
    assert calls == []


@pytest.mark.parametrize('configuration', [
    spec(task={'kind': 'passive'}, measurements={'behavior': True}),
    spec(task={'kind': 'rating', 'scale': {'low': 7, 'high': 1}}),
    spec(task={'kind': 'choice', 'labels': ['yes', 'YES']}),
    spec(conditions=[{'name': 'same'}, {'name': 'same'}]),
    spec(conditions=[{'name': 'changed', 'intervention': {'target': '', 'operation': 'zero'}}]),
    spec(repetitions=0), spec(unknown='not executable code'),
])
def test_invalid_protocols_fail_as_data_before_execution(configuration):
    with pytest.raises(ValueError):
        ProtocolSpec.parse(configuration)


def test_protocol_validation_receipt_is_bound_to_exact_draft_and_model(tmp_path):
    from umi_lab.server import LabController
    from umi_lab.benchmark import wait
    controller = LabController(tmp_path / 'runs')
    try:
        wait(controller, ready=True)
        data = {'mode': 'protocol', 'rows': [{'sentence': '7 3'}, {'sentence': '1 6'}],
                'protocol': {'identifier': 'comparison', 'task': {'kind': 'legacy', 'labels': ['greater', 'less'], 'instruction': 'Compare numbers'},
                             'measurements': {'behavior': True}, 'conditions': [{'name': 'baseline'}]}}
        with pytest.raises(ValueError, match='Validate'):
            controller.submit_request(data)
        controller.validate_request(data)
        state = wait(controller, ready=True)
        assert state['validation']['phase'] == 'valid'
        assert state['run'] is None
        changed = deepcopy(data); changed['protocol']['identifier'] = 'changed'
        with pytest.raises(ValueError, match='Validate'):
            controller.submit_request(changed)
        controller.submit_request(data)
        result = wait(controller)
        assert result['phase'] == 'complete'
        controller.load({'identifier': '__calibration__'})
        assert wait(controller, ready=True).get('validation') is None
    finally:
        controller.close()


@pytest.mark.parametrize('kind, reply, expected', [('text', 'A written answer.', 'A written answer.'), ('rating', '4', 4.)])
def test_reported_text_and_rating_follow_public_responses_and_saved_views(tmp_path, kind, reply, expected):
    from brainscore import BrainScoreModel
    seen = []
    def respond(prompt):
        seen.append(prompt)
        return reply
    model = BrainScoreModel('reply-fixture', supported_modalities={'text'}, reply=respond)
    proposed = Request(mode='protocol', rows=[{'sentence': 'A stimulus'}], protocol=spec(
        task={'kind': kind, 'path': 'generation'}, measurements={'behavior': True}))
    validate_protocol(model, proposed, tmp_path)
    assert not seen
    run_protocol(model, proposed, tmp_path / 'experiment', tmp_path, lambda event: None, lambda: False)
    assert len(seen) == 1
    output = observation_page(tmp_path / 'experiment')[0]
    assert output['channel'] == f'behavior:{kind}'
    assert output['response']['measure']['kind'] == 'reported'
    assert output['response']['measure']['source'] == 'reply'
    if kind == 'text':
        assert output['response']['preview'][0]['raw_answer'] == expected
    else:
        assert output['response']['scalar_rows'][0]['value'] == expected


def test_intervention_is_condition_specific_and_restored_with_joint_recording(tmp_path):
    import torch
    from brainscore import BrainScoreModel
    from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
    from brainscore.perturbation import build_pytorch_ablation_fn
    from brainscore.experiments import ExperimentResult
    net = torch.nn.Sequential(torch.nn.Identity())
    def extract(stimuli, layers):
        values = net(torch.tensor([[len(text), 1.] for text in stimuli['sentence']])).numpy()
        return NeuroidAssembly(values, dims=['presentation', 'neuroid'],
            coords={'stimulus_id': ('presentation', list(stimuli['stimulus_id'])),
                    'neuroid_id': ('neuroid', ['u0', 'u1']), 'layer': ('neuroid', ['0', '0'])})
    model = BrainScoreModel('perturbed-protocol-fixture', model=net, extractor={'text': extract},
        region_layer_map={'IT': '0'}, behavioral_readout_layer='0', intervention=build_pytorch_ablation_fn(net))
    proposed = request(spec(conditions=[{'name': 'baseline'}, {'name': 'changed', 'intervention': {
        'target': '0', 'indices': [0], 'axis': -1, 'operation': 'zero'}}, {'name': 'restored'}]))
    run_protocol(model, proposed, tmp_path / 'experiment', tmp_path, lambda event: None, lambda: False)
    saved = ExperimentResult.open(tmp_path / 'experiment')
    before = saved.neural('IT', condition='baseline', repetition=0)
    changed = saved.neural('IT', condition='changed', repetition=0)
    after = saved.neural('IT', condition='restored', repetition=0)
    assert np.all(changed.values[:, 0] == 0)
    np.testing.assert_array_equal(before.values, after.values)
