"""A complete local robotics/tool integration, using synthetic observations.

Run: python examples/partner_integration.py --out /tmp/my-umi-demo
The output directory must not already exist. No API calls or model downloads.
"""
import argparse
from pathlib import Path
import json

import numpy as np
import torch

from brainscore.activation_window import ActivationWindow
from brainscore.instrumentation import observe, intervene
from brainscore.perturbation import build_pytorch_ablation_fn
from brainscore.harnesses.robotics import ActionSpec, droid_steps, evaluate_droid_episode
from brainscore.run_record import RunRecord, RunRecorder
from brainscore_core.events import EnvironmentResponse, StateChange, Selection, Perturbation
from brainscore_core.model_interface import BrainScoreModel


def demo(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(0)
    network = torch.nn.Sequential(torch.nn.Linear(7, 12), torch.nn.Tanh(),
                                  torch.nn.Linear(12, 7), torch.nn.Tanh()).eval()

    def policy(step):
        with torch.no_grad():
            state = torch.tensor(step.observation['proprioception'].joint_position, dtype=torch.float32)
            return EnvironmentResponse(action=network(state).numpy())

    model = BrainScoreModel('synthetic-robot-policy', model=network, action_fn=policy,
                            state_change_fn=build_pytorch_ablation_fn(network))
    spec = ActionSpec(tuple(f'component_{i}' for i in range(7)), ('normalized',)*7,
                      'synthetic_fixture', 50, (-1.,)*7, (1.,)*7)
    rows = []
    for i in range(4):
        obs = {key: np.full((8, 12, 3), i, dtype=np.uint8) for key in (
            'wrist_image_left', 'exterior_image_1_left', 'exterior_image_2_left')}
        obs.update(joint_position=np.full(7, i/10), cartesian_position=np.zeros(6),
                   gripper_position=np.zeros(1))
        rows.append(dict(observation=obs, action=np.zeros(7),
                         language_instruction=b'place the cup', is_first=i==0, is_last=i==3))

    class LatencyTool:
        def __init__(self):
            self.durations = []
        def on_result(self, call, result):
            self.durations.append(call.duration_s)
    tool = LatencyTool()
    with RunRecorder(out/'run', metadata={
            'model': model.identifier, 'seed': 0, 'data': 'synthetic DROID-shaped fixture',
            'protocol': 'recorded_trajectory', 'action_units': 'normalized fixture units'}) as record:
        with ActivationWindow(network, layers=['0']) as activations, observe(model, tool):
            result = evaluate_droid_episode(model, {'steps': rows}, action_spec=spec,
                                             action_source=lambda row: row['action'], recorder=record)
        for capture in activations.captures:
            record.record('output', {'tool': 'activation_window', 'layer': capture.layer,
                'call': capture.call, 'values': capture.tensor.numpy()})

    # A paired intervention on one fixed observation, outside episode reset.
    step, _ = next(droid_steps({'steps': rows}, action_spec=spec,
                               action_source=lambda row: row['action']))
    with RunRecorder(out/'intervention', metadata={
            'model': model.identifier, 'data': 'synthetic fixture', 'seed': 0,
            'protocol': 'fixed-observation baseline/ablation/recovery'}) as record:
        with observe(model, record):
            baseline = model.process(step).action
            with intervene(model, StateChange('ablation', Selection('0'), Perturbation('zero'))):
                ablated = model.process(step).action
            recovered = model.process(step).action
    np.testing.assert_array_equal(recovered, baseline)
    assert not np.array_equal(ablated, baseline)

    # All measurement analysis below runs from the record without model calls.
    replay = list(RunRecord(out/'run').outputs())
    motors = [item for item in replay if hasattr(item, 'channel') and item.channel == 'motor']
    summary = {'fixture': True, 'steps': len(motors), 'activation_captures': len(activations.captures),
        'observed_calls': len(tool.durations),
        'intervention_changed_action': bool(not np.array_equal(ablated, baseline)),
        'intervention_recovery_exact': bool(np.array_equal(recovered, baseline)),
        'action_mse': float(np.mean([(event.payload.action-event.meta['target'])**2 for event in motors])),
        'time_ms': [event.t_ms for event in motors],
        'claim': 'Recorded observation/action and activation integration; not robot task success'}
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    print(json.dumps(demo(parser.parse_args().out), indent=2))
