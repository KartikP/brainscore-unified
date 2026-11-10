"""Fit, save, reload and attach two probes; CPU fixtures, no downloads."""
import argparse
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from brainscore.experiments import (
    Experiment, SessionProtocol, TorchInstrumentation,
)
from brainscore.tools import attach_tools, load_tool
from brainscore_core.contract import Subject
from brainscore_core.streaming import Session, StreamEvent
from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet


class TextFixture(Subject):
    """A fixed synthetic encoder demonstrates integration, not learned behavior."""
    identifier = 'linear-probe-text-fixture'
    in_channels, out_channels = {'text'}, {'neural:population'}

    def __init__(self) -> None:
        self.population = torch.nn.Linear(2, 4, bias=False)
        with torch.no_grad():
            self.population.weight.copy_(torch.tensor([[1., 0.], [0., 1.], [1., 1.], [1., -1.]]))

    def participate(self, session: Session) -> None:
        while (event := session.next_input()) is not None:
            stimuli = event.payload
            vectors = [[1. if 'red' in text else -1., 1. if 'square' in text else -1.]
                       for text in stimuli['sentence']]
            activity = self.population(torch.tensor(vectors)).detach().numpy()
            session.respond(StreamEvent('neural:population', NeuroidAssembly(
                activity, dims=('presentation', 'neuroid'), coords={
                    'stimulus_id': ('presentation', stimuli['stimulus_id'].tolist()),
                    'sentence': ('presentation', stimuli['sentence'].tolist()),
                    'unit_index': ('neuroid', list(range(4))),
                    'layer': ('neuroid', ['population'] * 4),
                }), event.t_ms))


def make_session(texts: list[str]) -> Session:
    stimuli = StimulusSet({'stimulus_id': [f'item-{i}' for i in range(len(texts))], 'sentence': texts})
    return Session([StreamEvent('text', stimuli, None)], ['neural:population'])


def run(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=False)
    subject = TextFixture()
    texts = ['red square', 'blue circle', 'red circle', 'blue square']
    session = make_session(texts)
    subject.participate(session)
    discovery = session.collect()
    probe = load_tool('linear-probe', target='population').fit(
        discovery, ['red', 'blue', 'red', 'blue'],
        provenance={'checkpoint': 'fixed-synthetic-weights-v1', 'dataset': 'four-discovery-phrases'})
    artifact = directory / 'color-probe.json'
    probe.save(artifact)
    before = load_tool('linear-probe', artifact=artifact, name='color_before', when='before')
    after = load_tool('linear-probe', artifact=artifact, name='color_after')
    provider = TorchInstrumentation({'population': subject.population})

    @contextmanager
    def factory(trial):
        yield make_session(['a red square', 'a blue circle'])

    result = Experiment(
        subject=subject,
        protocol=SessionProtocol('color-probe-demo', factory, conditions=['baseline', 'lesioned'],
                                 input_channels=['text'], output_channels=['neural:population']),
        tools=[load_tool('record-io'),
               load_tool('record-activity', targets=['population'], name='activity_before', when='before'),
               before,
               load_tool('ablate', targets=['population'], conditions=['lesioned']),
               load_tool('record-activity', targets=['population'], name='activity_after'),
               after],
        instrumentation=provider, output_dir=directory / 'experiment',
    ).run()
    for event in result.record.events():
        if event['kind'] == 'probe':
            print(event['source'], event['condition'], event['payload'].values.round(3).tolist())

    with attach_tools(subject, tools=[load_tool('record-io'), after], instrumentation=provider,
                      output_dir=directory / 'direct'):
        subject.participate(make_session(['a red circle']))

    restored = make_session(texts)
    subject.participate(restored)
    np.testing.assert_array_equal(restored.collect().values, discovery.values)
    print('Restored baseline exactly; protocol and direct scope completed.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    run(parser.parse_args().out)
