"""The one committed table of reference scores (``fixtures/reference-scores.json``).

Each row is a registered subject and benchmark with a value and the environment it
was made in. Two kinds of rows:

  * ``reference``: a full-digit score from the cloud runs named in ``runs``, where
    every named run gave the same value. Tolerance 0: in the same environment the
    score must be equal bit for bit (rules 10.1, 10.2). Across environments there is
    no bit-for-bit claim. An environment whose fingerprint is not stated yet cannot be
    recognised, so its rows are checked offline only until scores carry a fingerprint.
    A row also states the ``math_threads`` record of its measurement; a row without one
    is compared like a row of another environment, within a stated tolerance class.
  * ``production``: a legacy pair against its Brain-Score leaderboard value
    (``production_baseline``, rounded), with the tolerance and the evidence for it.

The offline tests check the table. The slow test scores each production row that is
expected to match (needs downloads and private data; run it on the cloud machine).
"""
import json
import math
from pathlib import Path

import pytest

TABLE = Path(__file__).parent / 'fixtures' / 'reference-scores.json'
KINDS = ('reference', 'production')


def _table():
    # The versioned table is required: missing release evidence must fail.
    return json.loads(TABLE.read_text())


def _rows(kind):
    return [row for row in _table()['scores'] if row['kind'] == kind]


def _matching_production_rows():
    """Production rows expected to match their leaderboard value.

    Historical published-score mismatches, including HMAX, stay out (``match`` is not
    True). Passing this subset does not reproduce the excluded historical scores.
    """
    return [row for row in _rows('production')
            if row['production_baseline'] is not None and row['match'] is True]


def _id(row):
    return f"{row['subject']}:{row['benchmark']}"


@pytest.mark.unit
def test_every_row_is_well_formed():
    table = _table()
    environments, runs = table['environments'], table['runs']
    for row in table['scores']:
        assert {'subject', 'benchmark', 'value', 'kind', 'environment'} <= set(row), row
        assert row['kind'] in KINDS, row
        assert row['environment'] in environments, row
        assert isinstance(row['value'], (int, float)) and math.isfinite(row['value']), row
        if row['kind'] == 'reference':
            assert row['tolerance'] == 0.0, _id(row)          # a non-zero tolerance needs evidence
            assert len(row['runs']) >= 2 and set(row['runs']) <= set(runs), _id(row)
    for environment in environments.values():
        assert 'description' in environment and 'fingerprint' in environment


def _bit_for_bit_row(row, environments):
    """A row is compared bit for bit only when its environment has a fingerprint and the
    row states its math threads (``not stated`` never counts)."""
    threads = row.get('math_threads')
    return (environments[row['environment']]['fingerprint'] is not None
            and isinstance(threads, str) and 'not stated' not in threads)


@pytest.mark.unit
def test_a_row_is_bit_for_bit_only_with_a_fingerprint_and_a_thread_record():
    environments = {'a': {'fingerprint': None}, 'b': {'fingerprint': 'f' * 64}}
    assert not _bit_for_bit_row({'environment': 'b'}, environments)
    assert not _bit_for_bit_row({'environment': 'b', 'math_threads': 'pools=not stated,torch=4'}, environments)
    assert not _bit_for_bit_row({'environment': 'a', 'math_threads': 'torch=4'}, environments)
    assert _bit_for_bit_row({'environment': 'b', 'math_threads': 'torch=4'}, environments)


@pytest.mark.unit
def test_a_stated_thread_record_in_the_table_is_a_string():
    for row in _table()['scores']:
        if 'math_threads' in row:
            assert isinstance(row['math_threads'], str) and row['math_threads'], _id(row)


@pytest.mark.unit
def test_one_row_per_pair_and_kind():
    keys = [(row['kind'], row['subject'], row['benchmark']) for row in _table()['scores']]
    assert len(keys) == len(set(keys))


@pytest.mark.unit
def test_a_stated_tolerance_names_its_evidence():
    for row in _table()['scores']:
        if row.get('tolerance'):
            assert row.get('evidence'), _id(row)


@pytest.mark.unit
def test_the_reference_environment_without_a_fingerprint_names_its_package_list():
    reference = _table()['environments']['reference-2026-11-15']
    assert reference['fingerprint'] is None
    assert reference['description'] == 'reference machine, package list of 2026-11-15'


@pytest.mark.unit
def test_every_row_names_a_registered_subject_and_benchmark():
    import brainscore
    models = set(brainscore.list_models()['identifier'])
    benchmarks = set(brainscore.list_benchmarks()['identifier'])
    for row in _table()['scores']:
        assert row['subject'] in models, _id(row)
        assert row['benchmark'] in benchmarks, _id(row)


@pytest.mark.unit
def test_the_production_rows_keep_the_anchors():
    assert len(_matching_production_rows()) >= 3, 'expected several anchored production rows'
    assert any(row['subject'] == 'alexnet' and 'MajajHong' in row['benchmark']
               for row in _matching_production_rows())


@pytest.mark.slow
@pytest.mark.private_access
@pytest.mark.parametrize('row', _matching_production_rows(), ids=[_id(r) for r in _matching_production_rows()])
def test_a_production_row_reproduces_its_leaderboard_value(row):
    """Score the pair; it lands within the row's tolerance of the leaderboard value."""
    import brainscore
    score = brainscore.score(row['subject'], row['benchmark'])
    value, baseline, tolerance = float(score), row['production_baseline'], row['tolerance']
    assert abs(value - baseline) < tolerance, (
        f"{_id(row)}: scored {value:.6f}, baseline {baseline:.4f}, "
        f"|delta|={abs(value - baseline):.2e} > {tolerance}")
    assert 'runtime_sec' in score.attrs and score.attrs['runtime_sec'] >= 0.0


class _Requests:
    """A participate-only subject that keeps the channel each session requests and answers it."""
    identifier = 'format-check'
    in_channels = {'vision:image', 'text'}
    out_channels = {'behavior', 'behavior:choice', 'neural:IT', 'neural:V4', 'neural:language_system'}
    required_channels = set()

    def __init__(self):
        self.requested = []

    def participate(self, session):
        import numpy as np
        from brainscore_core.streaming import StreamEvent
        from brainscore_core.supported_data_standards.brainio.assemblies import (
            BehavioralAssembly, NeuroidAssembly)
        while session.next_input() is not None:
            pass
        self.requested += list(session.outputs)
        ids = list(session.stimuli['stimulus_id'])
        for channel in session.outputs:
            cls = NeuroidAssembly if channel.startswith('neural:') else BehavioralAssembly
            session.respond(StreamEvent(channel, cls(
                np.zeros((len(ids), 2)), coords={'stimulus_id': ('presentation', ids)},
                dims=['presentation', 'neuroid' if channel.startswith('neural:') else 'choice']), None))

    def reset(self):
        pass


@pytest.mark.unit
def test_every_production_row_names_the_format_the_facade_requests():
    """T5: each production row names the channel its benchmark's old call is asked on
    (stage 5: the probabilities task stays on bare behavior; a label list is behavior:choice)."""
    import pandas as pd
    from brainscore_core.legacy_facade import LegacyBenchmarkFacade
    from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
    stimuli = StimulusSet(pd.DataFrame({'stimulus_id': ['a', 'b'], 'image_label': ['x', 'y'],
                                        'sentence': ['one', 'two']}))
    stimuli.identifier = 'regression-format-check'
    for row in _rows('production'):
        subject = _Requests()
        facade = LegacyBenchmarkFacade(subject, visual_degrees=8)
        fmt = row['format']
        if 'old_task' in row:
            facade.start_task(row['old_task'], stimuli)
            facade.look_at(stimuli)
        elif row['domain'] == 'language':
            facade.start_neural_recording(fmt.split(':', 1)[1])
            facade.digest_text(['one two'])
        else:
            facade.start_recording(fmt.split(':', 1)[1], time_bins=[(70, 170)])
            facade.look_at(stimuli)
        assert subject.requested == [fmt], _id(row)
