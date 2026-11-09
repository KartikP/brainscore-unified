"""Regression baselines as pytest — Tier-B.

Wraps the manual `scripts/m3_regression_test.py` + `baselines/baselines.json`
into pytest so the "does our pipeline still reproduce the leaderboard scores"
check runs on demand. Two tiers:

  * test_baseline_manifest_wellformed  (@unit, fast, CI)  — validates the manifest
    structure without scoring anything.
  * test_score_matches_baseline  (@slow @private_access)  — actually scores each
    deterministic pair via brainscore.score and asserts it lands within tolerance
    of its recorded baseline (and that runtime_sec is attached). Needs AWS creds +
    model downloads; NOT run in normal CI — invoke with `-m slow`.
"""
import json
from pathlib import Path

import pytest

BASELINES = Path(__file__).parent / 'fixtures' / 'regression-baselines.json'
# Contract Rule 10.2 tolerances, per pair.
#
# Measured pairs: MEASURED_TOL = 6e-4. `production_baseline` is the
# leaderboard value rounded to 3 decimals, so rounding alone allows |delta| up
# to 5e-4. The other 1e-4 is about 3x the largest observed delta between a
# fresh score and the recorded score_value (3.3e-5).
# Evidence (2026-11-09, EC2 g5.4xlarge, AMD EPYC 7R32 + A10G, cold caches):
# UMI (core d16e90f, vision 1c2191b, language 06c4066) equals current
# upstream (vision master c560866, core main 2e1cb4e == PyPI brainscore-core
# 2.5.0) bit-for-bit in the same environment, on GPU and on CPU, for alexnet
# on MajajHong2015.IT-pls, MajajHong2015.V4-pls and Rajalingham2018-i2n.
#   GPU: IT 0.41746951668719495, V4 0.4919152765977861, i2n 0.3702114698148112
#   CPU: IT 0.4174799858892488,  V4 0.4919518163116004, i2n 0.3701657913463906
# The cause of the delta from the stored baselines is not identified: the
# device moves scores by up to 4.6e-5, but CPU reproduces only the IT
# baseline at every stored digit. Environment: Python 3.11.15, numpy 1.26.4,
# scipy 1.17.1, scikit-learn 1.7.2, torch 2.6.0+cu124, torchvision
# 0.21.0+cu124, xarray 2022.3.0, pandas 2.3.3.
#
# Unmeasured pairs: UNMEASURED_TOL = 0.01, the historical bound. No same-env
# UMI-vs-upstream measurement exists for them, so the tight bound does not apply.
MEASURED_TOL = 6e-4
UNMEASURED_TOL = 0.01
MEASURED_PAIRS = frozenset({
    'vision:alexnet:MajajHong2015.IT-pls',
    'vision:alexnet:MajajHong2015.V4-pls',
    'vision:alexnet:Rajalingham2018-i2n',
})


def _tolerance(key):
    return MEASURED_TOL if key in MEASURED_PAIRS else UNMEASURED_TOL


def _load_all():
    # This versioned fixture is required: missing release evidence must fail.
    return json.loads(BASELINES.read_text())


def _require_baselines():
    data = _load_all()
    assert data, 'Baseline manifest is empty'
    return data


def _deterministic_pairs():
    """Pairs with a production baseline that are expected to match.

    Historical published-score mismatches, including HMAX, stay excluded.
    Current-upstream comparisons are recorded separately from these references.
    Passing this subset does not reproduce the excluded historical scores.
    """
    return [(k, v) for k, v in _load_all().items()
            if v.get('production_baseline') is not None and v.get('match') is True]


@pytest.mark.unit
def test_baseline_manifest_wellformed():
    data = _require_baselines()
    for key, entry in data.items():
        assert {'score_value', 'domain', 'model_id', 'benchmark_id'} <= set(entry), key
        assert entry['domain'] in ('vision', 'language'), key
        assert isinstance(entry['score_value'], (int, float)), key
    # at least the anchored AlexNet/MajajHong regression pairs are present
    assert any('alexnet' in k and 'MajajHong' in k for k in data), "missing AlexNet anchor"


@pytest.mark.unit
def test_at_least_one_deterministic_pair():
    _require_baselines()
    assert len(_deterministic_pairs()) >= 3, "expected several anchored regression pairs"


@pytest.mark.slow
@pytest.mark.private_access
@pytest.mark.parametrize('key,entry', _deterministic_pairs(),
                         ids=[k for k, _ in _deterministic_pairs()])
def test_score_matches_baseline(key, entry):
    """Score the pair and assert it reproduces the recorded baseline within its tolerance.
    Slow + needs S3/AWS — run on demand, not in CI."""
    import brainscore
    score = brainscore.score(entry['model_id'], entry['benchmark_id'])
    val = float(score)
    base = entry['production_baseline']
    tol = _tolerance(key)
    assert abs(val - base) < tol, (
        f"{key}: scored {val:.6f}, baseline {base:.4f}, |Δ|={abs(val-base):.2e} > {tol}")
    assert 'runtime_sec' in score.attrs and score.attrs['runtime_sec'] >= 0.0
