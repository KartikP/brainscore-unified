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

BASELINES = Path(__file__).resolve().parents[2] / 'baselines' / 'baselines.json'
TOL = 0.01   # allows PLS/scikit-learn non-determinism; tight enough to catch real regressions


def _load_all():
    if not BASELINES.exists():
        return {}
    return json.load(open(BASELINES))


def _require_baselines():
    """Skip when the baseline manifest is absent.

    `baselines/baselines.json` records scores measured on our infrastructure and lives
    at the workspace root, outside any of the four repositories — so a fresh clone of
    this package does not have it. Failing there would greet every new developer with
    two red tests about data they were never given.
    """
    data = _load_all()
    if not data:
        pytest.skip(f'no baseline manifest at {BASELINES}; it is recorded per-workspace '
                    f'and is not part of this package')
    return data


def _deterministic_pairs():
    """Pairs with a production baseline that are expected to match (excludes the
    documented non-deterministic ones, e.g. hmax layer-search jitter)."""
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
