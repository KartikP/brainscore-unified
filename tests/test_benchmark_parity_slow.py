"""Explicit opt-in only: real data/weights, excluded from the laptop fast tier."""
import json
import os
from pathlib import Path

import pytest

from brainscore.validation.benchmark_parity import (
    CASES, PEREIRA_CASES, assert_pereira_score_drift, local_candidate, validate_case,
)

pytestmark = pytest.mark.slow


def _require_checkpoint(domain):
    if os.environ.get('RUN_UMI_PARITY') != '1':
        pytest.skip('Set RUN_UMI_PARITY=1 on a machine with staged benchmark data and weights')
    key = 'UMI_PARITY_RESNET18' if domain == 'vision' else 'UMI_PARITY_GPT2'
    if not os.environ.get(key) or not Path(os.environ[key]).exists():
        pytest.skip(f'Missing local checkpoint: {key}')


@pytest.mark.parametrize('case', [case for case in CASES if case.domain == 'vision'],
                         ids=lambda case: case.unified)
def test_registered_benchmark_both_routes(case, tmp_path):
    _require_checkpoint(case.domain)
    report = validate_case(case, local_candidate)
    path = tmp_path / (case.unified + '.json')
    path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, sort_keys=True))


def test_pereira_243_and_384_score_parity(tmp_path):
    _require_checkpoint('language')
    reports = []
    for case in PEREIRA_CASES:
        report = validate_case(case, local_candidate)
        (tmp_path / (case.unified + '.json')).write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, sort_keys=True))
        reports.append(report)
    # Keep the two cases in one test: independent parametrized tests cannot
    # reject same-sign bias or compare the frozen per-benchmark drift envelopes.
    assert_pereira_score_drift(reports)
