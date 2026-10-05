"""Five small CPU demonstrations. No trained-model or scientific validity claim."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import numpy as np
import torch
from brainscore_core.contract import Subject
from brainscore_core.streaming import InMemorySession, StreamEvent
from brainscore.experiments import (Experiment, SessionProtocol, RecordInputsOutputs,
    RecordActivity, Ablate, TorchInstrumentation, replay_sessions, compare_outputs)


class DemonstrationSubject(Subject):
    in_channels = {'observation'}
    out_channels = {'behavior', 'neural'}

    def __init__(self, identifier, encode, *, recurrent=False):
        self.identifier_value, self.encode, self.recurrent = identifier, encode, recurrent
        self.population = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.population.weight.copy_(torch.tensor([[1., .5], [.5, 1.]]))
        self.reset()

    @property
    def identifier(self):
        return self.identifier_value

    def reset(self):
        self.memory = torch.zeros(2)

    def interact(self, session):
        while (event := session.next_input()) is not None:
            features = torch.tensor(self.encode(event.payload), dtype=torch.float32)
            with torch.no_grad():
                activity = self.population(features + self.memory)
                self.memory = activity * .1 if self.recurrent else torch.zeros(2)
            session.emit(StreamEvent('neural', activity.numpy().copy(), event.t_ms))
            session.emit(StreamEvent('behavior', float(activity.sum()), event.t_ms))


class FeedbackSession(InMemorySession):
    """Toy controlled environment; not LIBERO, MuJoCo or a robotics qualification."""
    def __init__(self):
        super().__init__()
        self.position, self.step = 0., 0
    def next_input(self):
        if self.step == 3:
            return None
        return StreamEvent('observation', {'position': self.position, 'goal': 1.}, self.step * 50)
    def emit(self, event):
        super().emit(event)
        if event.channel == 'behavior':
            self.position += event.payload * .1
            self.step += 1


def run(directory):
    configurations = {
        'vision': (np.ones((4, 4, 3)), lambda x: [float(x.mean()), float(x.std())]),
        'language': ('a short sentence', lambda x: [len(x.split()) / 10, len(x) / 100]),
        'vlm': ({'image': np.ones((4, 4, 3)), 'text': 'describe this'},
                lambda x: [float(x['image'].mean()), len(x['text'].split()) / 10]),
        'robotics': (None, lambda x: [x['goal'] - x['position'], x['position']]),
        'brain_behavior': (np.array([.2, .4]), lambda x: x),
    }
    summary = {}
    for name, (payload, encode) in configurations.items():
        subject = DemonstrationSubject(name, encode, recurrent=name == 'brain_behavior')
        @contextmanager
        def factory(trial):
            if name == 'robotics':
                yield FeedbackSession()
            else:
                yield InMemorySession([StreamEvent('observation', payload, t) for t in (0, 20, 40)])
        protocol = SessionProtocol(name, factory, trials=['baseline', 'ablation'],
            input_channels=['observation'], output_channels=['behavior', 'neural'],
            metadata={'evidence': 'synthetic demonstration'})
        result = Experiment(subject=subject, protocol=protocol,
            tools=[RecordInputsOutputs(), Ablate(['population'], trials=['ablation']),
                   RecordActivity(['population'])],
            instrumentation=TorchInstrumentation({'population': subject.population}),
            output_dir=directory/name, metadata={'trained': False}).run()
        replay = Experiment(subject=subject, protocol=replay_sessions(result.directory),
            tools=[RecordInputsOutputs(), Ablate(['population'], trials=['ablation'])],
            instrumentation=TorchInstrumentation({'population': subject.population}),
            output_dir=directory/(name+'-replay'), metadata={'trained': False}).run()
        summary[name] = compare_outputs(result.directory, replay.directory)
        if not summary[name]['equal']:
            raise AssertionError(f'Replay differed for {name}')
    (directory/'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    print(json.dumps(run(args.out), indent=2))
