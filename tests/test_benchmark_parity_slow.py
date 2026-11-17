"""Explicit opt-in only: real data/weights, excluded from the laptop fast tier."""
import json
import os
from pathlib import Path

import pytest

from brainscore.validation.benchmark_parity import CASES, local_subject, validate_case

pytestmark = pytest.mark.slow


def _require_checkpoint(domain):
    if os.environ.get('RUN_UMI_PARITY') != '1':
        pytest.skip('Set RUN_UMI_PARITY=1 on a machine with staged benchmark data and weights')
    key = 'UMI_PARITY_RESNET18' if domain == 'vision' else 'UMI_PARITY_GPT2'
    if not os.environ.get(key) or not Path(os.environ[key]).exists():
        pytest.skip(f'Missing local checkpoint: {key}')


@pytest.mark.parametrize('case', CASES,
                         ids=lambda case: case.legacy)
def test_registered_benchmark_both_routes(case, tmp_path):
    _require_checkpoint(case.domain)
    report = validate_case(case, local_subject)
    path = tmp_path / (case.legacy + '.json')
    path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, sort_keys=True))

