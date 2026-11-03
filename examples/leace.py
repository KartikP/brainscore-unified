"""Fit, save, reload and apply LEACE alongside probes; CPU fixture, no downloads."""
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
    subject.participate(session)
    features = session.collect()
    labels = ['red', 'blue', 'red', 'blue']
    provenance = {'checkpoint': 'fixed-synthetic-weights-v1', 'dataset': 'four-discovery-phrases'}
    eraser = load_tool('leace', target='population', shrinkage=False,
                       constrain_cov_trace=False, svd_tol=1e-10).fit(features, labels, provenance=provenance)
    eraser.save(directory / 'eraser.json')
    probe = load_tool('linear-probe', target='population').fit(features, labels, provenance=provenance)
    probe.save(directory / 'probe.json')

    @contextmanager
    def factory(trial):
        yield make_session(['a red square', 'a blue circle'])

    result = Experiment(subject=subject,
        protocol=SessionProtocol('leace-demo', factory, conditions=['baseline', 'erased', 'restored'],
                                 input_channels=['text'], output_channels=['neural:population']),
        tools=[load_tool('record-io'),
               load_tool('linear-probe', artifact=directory / 'probe.json', name='before', when='before'),
               load_tool('leace', artifact=directory / 'eraser.json', conditions=['erased']),
               load_tool('linear-probe', artifact=directory / 'probe.json', name='after')],
        instrumentation=TorchInstrumentation({'population': subject.population}),
        output_dir=directory / 'experiment').run()
    for event in result.record.events():
        if event['kind'] == 'probe':
            print(event['source'], event['condition'], event['payload'].values.round(3).tolist())
    np.testing.assert_array_equal(result.value[0].values, result.value[2].values)
    assert not np.array_equal(result.value[0].values, result.value[1].values)
    print('LEACE artifact reused; erased activity changed; baseline restored exactly.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    run(parser.parse_args().out)
