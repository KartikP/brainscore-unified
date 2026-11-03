"""Five synthetic CPU examples using the same experiment and tool setup.

Run after installing this package: python -m umi_partner_tool.toolbox --out NEW_DIR
These fixtures demonstrate integration, not trained-model performance.
"""

import argparse
from contextlib import contextmanager
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import torch

from brainscore.experiments import (
    Ablate,
    Experiment,
    RecordActivity,
    RecordInputsOutputs,
    SessionProtocol,
    TorchInstrumentation,
    compare_outputs,
    read_events,
    replay_sessions,
)
from brainscore_core.contract import Subject
from brainscore_core.environment import ArraySpec, MappingSpec
from brainscore_core.events import EnvironmentResponse, EnvironmentStep
from brainscore_core.model_interface import BrainScoreModel
from brainscore_core.streaming import BufferedSession, Session, StreamEvent
from brainscore_core.streaming_helpers import EnvironmentSession
from brainscore_core.supported_data_standards.brainio.assemblies import (
    BehavioralAssembly,
    NeuroidAssembly,
)

EXAMPLES = ('vision', 'language', 'vlm', 'robotics', 'brain_behavior')
CONDITIONS = ('baseline', 'silenced', 'restored')
WEIGHTS = np.array([[1.0, 0.5], [0.5, 1.0]], dtype=np.float32)
PROVENANCE = {
    'checkpoint': 'fixed-two-unit-fixture-v1',
    'weights_sha256': hashlib.sha256(WEIGHTS.tobytes()).hexdigest(),
    'dtype': 'float32',
    'device': 'cpu',
    'packages': {name: version(name) for name in ('umi-partner-tool-example', 'torch', 'numpy')},
}


def population() -> torch.nn.Linear:
    """Use fixed weights so a replay can be compared exactly."""
    layer = torch.nn.Linear(
        in_features=2,
        out_features=2,
        bias=False,
        dtype=torch.float32,
        device='cpu',
    )
    with torch.no_grad():
        layer.weight.copy_(torch.from_numpy(WEIGHTS))
    return layer.eval()


class DemonstrationSubject(Subject):
    """Show how a user-owned subject records activity and behavior together."""

    out_channels = {'neural:population', 'behavior'}

    def __init__(
        self,
        *,
        identifier: str,
        input_channels: set[str],
        encode: Callable[[dict], list[float]],
        recurrent: bool = False,
    ) -> None:
        self.name = identifier
        self.input_channels = frozenset(input_channels)
        self.encode = encode
        self.recurrent = recurrent
        self.population = population()
        self.reset()

    @property
    def identifier(self) -> str:
        return self.name

    @property
    def in_channels(self) -> frozenset[str]:
        return self.input_channels

    @property
    def required_channels(self) -> frozenset[str]:
        return self.input_channels

    def reset(self) -> None:
        self.memory = torch.zeros(2, dtype=torch.float32, device='cpu')

    def participate(self, session: Session) -> None:
        inputs = {}
        stimulus_id = None
        while (event := session.next_input()) is not None:
            if event.channel not in self.in_channels:
                continue
            # The multimodal case waits for both inputs of the same stimulus.
            current_id = event.meta['stimulus_id']
            if inputs and current_id != stimulus_id:
                raise ValueError('A stimulus ended before all its inputs arrived')
            stimulus_id = current_id
            inputs[event.channel] = event.payload
            if set(inputs) != self.in_channels:
                continue
            features = torch.tensor(self.encode(inputs), dtype=torch.float32, device='cpu')
            with torch.no_grad():
                activity = self.population(features + self.memory)
                self.memory = activity * 0.1 if self.recurrent else torch.zeros_like(activity)

            # Assemblies name the stimuli and units that each value belongs to.
            coordinates = {
                'stimulus_id': ('presentation', [stimulus_id]),
                'presentation': ('presentation', [stimulus_id]),
            }
            responses = {
                'neural:population': NeuroidAssembly(
                    activity.numpy()[None, :].copy(),
                    coords={
                        **coordinates,
                        'neuroid_id': ('neuroid', ['population.0', 'population.1']),
                        'layer': ('neuroid', ['population', 'population']),
                    },
                    dims=['presentation', 'neuroid'],
                ),
                'behavior': BehavioralAssembly(
                    [float(activity.sum())],
                    coords=coordinates,
                    dims=['presentation'],
                ),
            }
            for channel in session.requested_output_channels:
                session.respond(StreamEvent(
                    channel=channel,
                    payload=responses[channel],
                    t_ms=event.t_ms,
                    meta={'provenance': {**PROVENANCE, 'model': self.identifier}},
                ))
            inputs.clear()
        if inputs:
            raise ValueError('A stimulus ended before all its inputs arrived')


class FeedbackEnvironment:
    """Apply three actions to a toy position; no simulator or robot is involved."""

    def observation_spec(self) -> MappingSpec:
        return MappingSpec({'state': ArraySpec(shape=(2,), dtype='float32')})

    def action_spec(self) -> ArraySpec:
        return ArraySpec(shape=(1,), dtype='float32', lower=-1.0, upper=1.0)

    def observation(self) -> EnvironmentStep:
        return EnvironmentStep(
            observation={
                'state': np.array([1.0 - self.position, self.position], dtype=np.float32),
            },
            step_num=self.steps,
            is_first=self.steps == 0,
            is_last=self.steps == 3,
        )

    def reset(self, *, seed: int | None = None) -> EnvironmentStep:
        self.position = 0.0
        self.steps = 0
        return self.observation()

    def step(self, action: np.ndarray) -> EnvironmentStep:
        self.position += float(action[0]) * 0.1
        self.steps += 1
        return self.observation()


def make_example(name: str) -> tuple[Subject, Callable, torch.nn.Module]:
    """Keep fixture construction separate from the shared experiment setup."""
    if name == 'robotics':
        layer = population()

        def policy(step: EnvironmentStep) -> EnvironmentResponse:
            with torch.no_grad():
                activity = layer(torch.from_numpy(step.observation['state']))
            return EnvironmentResponse(
                action=torch.tanh(activity[:1]).numpy(),
                metadata={'provenance': {**PROVENANCE, 'model': 'synthetic-policy'}},
            )

        subject = BrainScoreModel(identifier='synthetic-policy', policy=policy)

        @contextmanager
        def session_factory(trial) -> Iterator[EnvironmentSession]:
            with EnvironmentSession(
                environment=FeedbackEnvironment(),
                seed=trial.identifier,
                require_specs=True,
            ) as session:
                yield session

        return subject, session_factory, layer

    image = np.ones((4, 4, 3), dtype=np.uint8)
    configurations = {
        'vision': ({'vision': image}, lambda x: [float(x['vision'].mean()), 0.5]),
        'language': ({'text': 'a short sentence'}, lambda x: [len(x['text'].split()) / 10, 0.5]),
        'vlm': (
            {'vision': image, 'text': 'describe this'},
            lambda x: [float(x['vision'].mean()), len(x['text'].split()) / 10],
        ),
        'brain_behavior': ({'vision': image}, lambda x: [float(x['vision'].mean()), 0.4]),
    }
    payloads, encode = configurations[name]
    subject = DemonstrationSubject(
        identifier=f'synthetic-{name}',
        input_channels=set(payloads),
        encode=encode,
        recurrent=name == 'brain_behavior',
    )

    @contextmanager
    def session_factory(trial) -> Iterator[BufferedSession]:
        # Static stimuli have no physical clock. IDs group image/text pairs.
        yield BufferedSession(
            inputs=[
                StreamEvent(
                    channel=channel,
                    payload=payload,
                    t_ms=None,
                    meta={'stimulus_id': f'stimulus-{index}'},
                )
                for index in range(3)
                for channel, payload in payloads.items()
            ],
            requested_output_channels=sorted(subject.out_channels),
        )

    return subject, session_factory, subject.population


def saved_values(directory: Path) -> dict[str, list[np.ndarray]]:
    """Analyze the saved responses without calling the subject."""
    values = {condition: [] for condition in CONDITIONS}
    for event in read_events(directory):
        if event['kind'] == 'output':
            output = event['payload']
            # A live action has both a proposal and a receipt. Compare it once.
            if output.channel == 'motor' and output.meta['action_status'] != 'proposed':
                continue
            response = output.payload
            value = response.action if output.channel == 'motor' else response.values
            values[event['condition']].append(np.asarray(value))
    return values


def run_example(name: str, directory: Path) -> dict:
    """Record, intervene, recover and replay one installed example."""
    subject, session_factory, layer = make_example(name)
    protocol = SessionProtocol(
        identifier=f'synthetic-{name}',
        session_factory=session_factory,
        conditions=CONDITIONS,
        input_channels=subject.in_channels,
        output_channels=subject.out_channels,
    )
    # This tool list is identical for every input type and subject.
    result = Experiment(
        subject=subject,
        protocol=protocol,
        tools=[
            RecordInputsOutputs(),
            Ablate(targets=['population'], conditions=['silenced']),
            RecordActivity(targets=['population']),
        ],
        instrumentation=TorchInstrumentation(model_or_targets={'population': layer}),
        output_dir=directory / name,
        metadata={'trained': False, 'data': 'synthetic fixture', 'provenance': PROVENANCE},
    ).run()
    # Replay reruns saved inputs. It does not restart the feedback environment.
    replay = Experiment(
        subject=subject,
        protocol=replay_sessions(directory=result.directory),
        tools=[
            RecordInputsOutputs(),
            Ablate(targets=['population'], conditions=['silenced']),
        ],
        instrumentation=TorchInstrumentation(model_or_targets={'population': layer}),
        output_dir=directory / f'{name}-replay',
        metadata={'trained': False, 'data': 'saved synthetic inputs', 'provenance': PROVENANCE},
    ).run()
    values = saved_values(result.directory)
    baseline, silenced, restored = (values[condition] for condition in CONDITIONS)
    def same(left: list[np.ndarray], right: list[np.ndarray]) -> bool:
        return len(left) == len(right) and all(
            np.array_equal(a, b) for a, b in zip(left, right)
        )
    if name == 'robotics':
        # Inference durations and execution receipts differ during open-loop replay.
        # Keep that telemetry in the records; compare the actual action arrays.
        replayed = saved_values(replay.directory)
        comparison = {
            'equal': all(same(values[condition], replayed[condition]) for condition in CONDITIONS),
            'scope': 'exact action arrays on saved observations; excludes timing and execution receipts',
        }
    else:
        comparison = compare_outputs(reference=result.directory, candidate=replay.directory)
    summary = {
        **comparison,
        'intervention_changed_output': not same(baseline, silenced),
        'recovery_exact': same(baseline, restored),
        'input_channels': sorted(subject.in_channels),
        'output_channels': sorted(subject.out_channels),
        'synthetic': True,
    }
    if not all(summary[key] for key in ('equal', 'intervention_changed_output', 'recovery_exact')):
        raise RuntimeError(f'Intervention, recovery or replay failed for {name}')
    return summary


def run(directory: Path, *, examples: tuple[str, ...] = EXAMPLES) -> dict:
    """Run selected examples and save their pass/fail comparisons."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    summary = {name: run_example(name=name, directory=directory) for name in examples}
    (directory / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--example', choices=('all', *EXAMPLES), default='all')
    args = parser.parse_args()
    selected = EXAMPLES if args.example == 'all' else (args.example,)
    print(json.dumps(run(directory=args.out, examples=selected), indent=2))


if __name__ == '__main__':
    main()
