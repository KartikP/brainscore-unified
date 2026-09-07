"""Tests for ``python -m brainscore.doctor``.

The command exists to be trusted when it says "ok", so what matters is that a
bad environment is actually reported as bad rather than quietly passing.
"""

import pytest

from brainscore import doctor

pytestmark = pytest.mark.unit


class TestReport:
    def test_names_the_interpreter_that_answered(self):
        """Which env replied is usually the whole question."""
        import sys
        text, _ = doctor.report()
        assert sys.executable in text

    def test_reports_passing_dependencies_too(self, monkeypatch):
        monkeypatch.setattr(doctor, '_dependency_rows',
                            lambda: [('numpy', '1.26.4', '>=1.21,<2', True)])
        monkeypatch.setattr(doctor, '_asset_rows', lambda: [])
        text, problems = doctor.report()
        assert problems == 0
        assert 'numpy' in text and '1.26.4' in text

    def test_out_of_bounds_dependency_is_a_problem(self, monkeypatch):
        monkeypatch.setattr(doctor, '_dependency_rows',
                            lambda: [('scikit-learn', '1.7.2', '>=1.5,<1.6', False)])
        monkeypatch.setattr(doctor, '_asset_rows', lambda: [])
        text, problems = doctor.report()
        assert problems == 1
        assert 'outside' in text
        assert 'wrong rather than merely broken' in text

    def test_absent_dependency_is_a_problem(self, monkeypatch):
        monkeypatch.setattr(doctor, '_dependency_rows',
                            lambda: [('transformers', None, '>=4.57,<6', None)])
        monkeypatch.setattr(doctor, '_asset_rows', lambda: [])
        _, problems = doctor.report()
        assert problems == 1

    def test_missing_data_asset_is_not_a_dependency_problem(self, monkeypatch):
        """Absent data is expected; only the benchmarks needing it care."""
        monkeypatch.setattr(doctor, '_dependency_rows', lambda: [])
        monkeypatch.setattr(doctor, '_asset_rows',
                            lambda: [{'name': 'a', 'kind': 'stimuli',
                                      'present': False, 'path': '/nowhere'}])
        text, problems = doctor.report()
        assert problems == 0
        assert '/nowhere' in text

    def test_exit_code_follows_the_verdict(self, monkeypatch):
        monkeypatch.setattr(doctor, '_dependency_rows',
                            lambda: [('numpy', '3.0', '>=1.21,<2', False)])
        monkeypatch.setattr(doctor, '_asset_rows', lambda: [])
        assert doctor.main([]) == 1


class TestAgainstTheRealEnvironment:
    def test_this_environment_is_within_bounds(self):
        """The suite is meaningless if run under drifted dependencies."""
        _, problems = doctor.report()
        assert problems == 0, 'this environment is outside the verified bounds'

    def test_describe_env_covers_every_bound(self):
        from brainscore_core._env_check import _BOUNDS, describe_env
        assert len(describe_env()) == len(_BOUNDS)
