"""A readout's probabilities are the same table in every process (Contract Rules 10.1).

The old vision extractor gives its presentation coordinates in an order that depends on the
Python hash seed. The same session in two processes with different ``PYTHONHASHSEED`` values
must give identical collected probabilities, and a run recorded in one process must replay
and compare equal in the other. The table shapes are those of lessons 06 and 10.
"""
import os
import pickle
import subprocess
import sys
import textwrap

import pytest

pytestmark = pytest.mark.integration

SCRIPT = textwrap.dedent('''
    import pickle, sys
    from contextlib import contextmanager
    from pathlib import Path
    import numpy as np, torch
    from brainscore import BrainScoreModel
    from brainscore_core.contract import TaskContext
    from brainscore_core.responses import Choice
    from brainscore_core.streaming_helpers import stimulus_session
    from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
    from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
    from brainscore.experiments import Experiment, RecordInputsOutputs, SessionProtocol, compare_outputs, replay_sessions

    root, stage, shape = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    CHANNELS = ['behavior:choice', 'behavior:probabilities', 'behavior:rt']

    def table(prefix, fitting=False):
        columns = {'stimulus_id': [prefix + str(i) for i in range(4)],
                   'image_path': ['/standin/' + prefix + str(i) for i in range(4)]}
        if shape == '06':
            columns['category'] = ['cat', 'ship', 'dog', 'airplane']
        if shape == '06' or fitting:
            columns['label'] = ['animal', 'vehicle'] * 2
        stimuli = StimulusSet(columns)
        stimuli.identifier = prefix + shape
        folder = root / 'inputs'
        folder.mkdir(exist_ok=True)
        for i in range(4):
            (folder / (prefix + str(i))).write_text('2 0' if i % 2 == 0 else '0 2')
        stimuli.stimulus_paths = {i: folder / i for i in stimuli.stimulus_id}
        return stimuli

    def model():
        torch.manual_seed(0)
        net = torch.nn.Sequential(torch.nn.Linear(2, 2, bias=False))
        extractor = PytorchWrapper(model=net, identifier='order-' + shape, batch_size=4, device='cpu',
                                   preprocessing=lambda paths: np.array([np.loadtxt(p) for p in paths], dtype=np.float32))
        return BrainScoreModel('order-' + shape, model=net, extractor={'vision': extractor}, behavioral_readout_layer='0',
                               capability_config={'behavior-sessions': {'behavior:choice': ('readout', 'argmax')}})

    task = TaskContext(response=Choice(['animal', 'vehicle'], reaction_time='evidence_margin'),
                       fitting_stimuli=table('f', fitting=True))
    session = stimulus_session(table('s'), CHANNELS, task)
    model().participate(session)
    (root / (stage + '.pkl')).write_bytes(pickle.dumps(session.collect('behavior:probabilities')))

    @contextmanager
    def factory(repetition):
        yield stimulus_session(table('s'), CHANNELS, task)
    protocol = SessionProtocol('order', factory, required_input_channels=['vision:image'], requested_output_channels=CHANNELS)
    if stage == 'replay':
        protocol = replay_sessions(root / 'record')
    run = Experiment(subject=model(), protocol=protocol, tools=[RecordInputsOutputs()], output_dir=root / stage).run()
    if stage == 'replay':
        (root / 'compare.pkl').write_bytes(pickle.dumps(compare_outputs(root / 'record', run.directory)))
''')


def _run(tmp_path, stage, seed, shape):
    script = tmp_path / 'session.py'
    script.write_text(SCRIPT)
    env = {**os.environ, 'PYTHONHASHSEED': str(seed), 'RESULTCACHING_DISABLE': '1', 'BS_INSTALL_DEPENDENCIES': 'no'}
    done = subprocess.run([sys.executable, str(script), str(tmp_path), stage, shape], env=env,
                          capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr[-2000:]


@pytest.mark.parametrize('shape', ['06', '10'])
def test_a_readout_session_gives_the_same_probabilities_and_replays_equal_in_another_process(tmp_path, shape):
    _run(tmp_path, 'record', 0, shape)
    _run(tmp_path, 'replay', 1, shape)
    recorded, replayed = (pickle.loads((tmp_path / f'{stage}.pkl').read_bytes()) for stage in ('record', 'replay'))
    assert recorded.identical(replayed)
    assert pickle.loads((tmp_path / 'compare.pkl').read_bytes())['equal']
