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
# Contract Rule 10.2 tolerance. `production_baseline` is the leaderboard value
# rounded to 3 decimals, so rounding alone allows |delta| up to 5e-4. The rest
# (1e-4) covers the measured environment drift, with about 3x margin.
# Evidence (2026-11-09, g5.4xlarge A10G, cold caches): UMI (core d16e90f,
# vision 1c2191b, language 06c4066) and clean upstream (vision master c560866,
# core main 2e1cb4e == PyPI brainscore-core 2.5.0) give bit-for-bit identical
# scores for alexnet on MajajHong2015.IT-pls (0.41746951668719495),
# MajajHong2015.V4-pls (0.4919152765977861) and Rajalingham2018-i2n
# (0.3702114698148112). Both move together from the recorded score_value by
# at most 3.1e-5 (V4). Environment: Python 3.11.15, numpy 1.26.4, scipy 1.17.1,
# scikit-learn 1.7.2, torch 2.6.0+cu124, torchvision 0.21.0+cu124, xarray
# 2022.3.0, pandas 2.3.3. Scope: measured for these three AlexNet pairs only;
# the other anchored pairs inherit the bound unmeasured.
TOL = 6e-4


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
    """Score the pair and assert it reproduces the recorded baseline within TOL.
    Slow + needs S3/AWS — run on demand, not in CI."""
    import brainscore
    score = brainscore.score(entry['model_id'], entry['benchmark_id'])
    val = float(score)
    base = entry['production_baseline']
    assert abs(val - base) < TOL, (
        f"{key}: scored {val:.4f}, baseline {base:.4f}, |Δ|={abs(val-base):.4f} > {TOL}")
    assert 'runtime_sec' in score.attrs and score.attrs['runtime_sec'] >= 0.0
