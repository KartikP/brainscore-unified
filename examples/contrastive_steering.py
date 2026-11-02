"""Fit one contrastive direction and reuse signed doses with probes on a CPU fixture."""
import argparse
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from brainscore.experiments import Experiment, SessionProtocol, TorchInstrumentation
from brainscore.tools import load_tool
from linear_probe import TextFixture, make_session


def run(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=False)
    subject = TextFixture()
    session = make_session(['red square', 'blue square', 'red circle', 'blue circle'])
    subject.interact(session)
    features = session.collect()
    labels = ['red', 'blue', 'red', 'blue']
    provenance = {'checkpoint': 'fixed-synthetic-weights-v1', 'dataset': 'four-discovery-phrases'}
    direction = load_tool('contrastive-steering', target='population').fit(
        features, labels, positive='red', negative='blue', provenance=provenance)
    direction.save(directory / 'direction.json')
    probe = load_tool('linear-probe', target='population').fit(features, labels, provenance=provenance)
    probe.save(directory / 'probe.json')

    @contextmanager
    def factory(trial):
        yield make_session(['a red square', 'a blue circle'])

    result = Experiment(subject=subject,
        protocol=SessionProtocol('steering-demo', factory,
            conditions=['baseline', 'toward_red', 'toward_blue', 'restored'],
            input_channels=['text'], output_channels=['neural:population']),
        tools=[load_tool('record-io'),
               load_tool('linear-probe', artifact=directory / 'probe.json', name='before', when='before'),
               load_tool('contrastive-steering', artifact=directory / 'direction.json', strength=0.5,
                         name='toward_red', conditions=['toward_red']),
               load_tool('contrastive-steering', artifact=directory / 'direction.json', strength=-0.5,
                         name='toward_blue', conditions=['toward_blue']),
               load_tool('linear-probe', artifact=directory / 'probe.json', name='after')],
        instrumentation=TorchInstrumentation({'population': subject.population}),
        output_dir=directory / 'experiment').run()
    for event in result.record.events():
        if event['kind'] == 'probe':
            print(event['source'], event['condition'], event['payload'].values.round(3).tolist())
    baseline, positive, negative, restored = [value.values for value in result.value]
    np.testing.assert_array_equal(baseline, restored)
    assert not np.array_equal(baseline, positive)
    np.testing.assert_allclose(positive - baseline, baseline - negative, atol=1e-6)
    print('Direction reused at opposite strengths; baseline restored exactly.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    run(parser.parse_args().out)
