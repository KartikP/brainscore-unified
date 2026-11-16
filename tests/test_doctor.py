"""Tests for ``python -m brainscore.doctor``.

The command exists to be trusted when it says "ok", so what matters is that a
bad environment is actually reported as bad rather than quietly passing.
"""

import pytest

from brainscore import doctor

pytestmark = pytest.mark.unit


class TestReport:
    @pytest.fixture(autouse=True)
    def stand_in_environment(self, monkeypatch):
        """Each test counts only the rows it sets up. The real lock section, imports and
        result cache add rows of the machine (a Linux machine with the full lock and other
        installed packages adds one per difference); a test that needs one sets it itself."""
        monkeypatch.setattr(doctor, '_lock_file', lambda: None)
        monkeypatch.setattr(doctor, 'IMPORT_CHECKS', ())
        monkeypatch.setenv('RESULTCACHING_DISABLE', '1')

    def test_names_the_interpreter_that_answered(self):
        """Which env replied is usually the whole question."""
        import sys
        text, _ = doctor.report()
        assert sys.executable in text

    def test_prints_the_fingerprint_every_score_carries(self):
        from brainscore_core.environment_check import environment_fingerprint
        text, _ = doctor.report()
        assert f'environment fingerprint  {environment_fingerprint()}' in text

    def test_prints_the_math_threads_next_to_the_fingerprint(self):
        from brainscore_core.environment_check import math_threads
        lines = doctor.report()[0].splitlines()
        index = next(i for i, l in enumerate(lines) if l.startswith('environment fingerprint'))
        assert lines[index + 1] == f'math threads             {math_threads()}'

    def test_another_process_prints_the_same_fingerprint_as_a_score_here(self):
        """Two processes of one environment give one value, whatever each imported first."""
        import os, subprocess, sys
        from brainscore_core.environment_check import _score_environment
        code = ('import sys; sys.path.append("/nonexistent/vendor"); from brainscore import doctor; '
                'doctor._lock_file = lambda: None; doctor.IMPORT_CHECKS = (); print(doctor.report()[0])')
        out = subprocess.run([sys.executable, '-W', 'ignore', '-c', code], capture_output=True,
                             text=True, env=dict(os.environ), check=True).stdout
        line = next(l for l in out.splitlines() if l.startswith('environment fingerprint'))
        assert line.split()[-1] == _score_environment()['software']['environment_fingerprint']

    def test_an_unavailable_fingerprint_is_printed_and_the_report_finishes(self, monkeypatch):
        from brainscore_core import environment_check
        monkeypatch.setattr(environment_check, 'environment_fingerprint',
                            lambda: 'unavailable: a distribution has no name (at /x)')
        text, _ = doctor.report()
        assert 'environment fingerprint  unavailable: a distribution has no name (at /x)' in text

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

    def test_dependency_rows_come_from_the_public_environment_check(self, monkeypatch):
        from brainscore_core import environment_check
        monkeypatch.setattr(environment_check, 'describe_env',
                            lambda: [('numpy', '3.0', '>=1.21,<2', False)])
        monkeypatch.setattr(doctor, '_asset_rows', lambda: [])
        text, problems = doctor.report()
        assert problems == 1
        assert '3.0' in text and 'outside' in text

    def test_package_that_fails_to_import_is_a_problem(self, monkeypatch):
        """pip check passes a pyarrow built for NumPy 2; importing it does not."""
        monkeypatch.setattr(doctor, '_dependency_rows', lambda: [])
        monkeypatch.setattr(doctor, '_asset_rows', lambda: [])
        monkeypatch.setattr(doctor, 'IMPORT_CHECKS', ('json', 'no_such_package_xyz'))
        text, problems = doctor.report()
        assert problems == 1
        assert 'XX no_such_package_xyz' in text and 'ok json' in text

    def test_exit_code_follows_the_verdict(self, monkeypatch):
        monkeypatch.setattr(doctor, '_dependency_rows',
                            lambda: [('numpy', '3.0', '>=1.21,<2', False)])
        monkeypatch.setattr(doctor, '_asset_rows', lambda: [])
        assert doctor.main([]) == 1


class TestAgainstTheRealEnvironment:
    def test_this_environment_is_within_bounds(self):
        """The suite is meaningless if run under drifted dependencies.

        An environment check, not a check of the code: in the GitHub jobs (which install
        the pinned versions and also run `python -m brainscore.doctor`) a drift fails
        here. On a developer machine the drift is a condition of that machine, so the
        test is skipped and names it.
        """
        import os
        text, problems = doctor.report()
        if problems and os.environ.get('GITHUB_ACTIONS') != 'true':
            drifted = [f'{package} {installed} (needs {expected})'
                       for package, installed, expected, ok in doctor._dependency_rows() if not ok]
            pytest.skip('this environment is outside the verified bounds: '
                        + ('; '.join(drifted) or 'see python -m brainscore.doctor')
                        + '. Install with -c unified/install/v2-constraints.txt.')
        assert problems == 0, text

    def test_describe_env_covers_every_bound(self):
        from brainscore_core.environment_check import _BOUNDS, describe_env
        assert len(describe_env()) == len(_BOUNDS)


def test_the_lock_section_reads_the_fingerprint_inventory(tmp_path, monkeypatch):
    """A folder added to sys.path later, holding a second numpy, does not change the version
    the lock section reads: the first startup entry of a name counts, as for the fingerprint."""
    import sys
    from brainscore_core.environment_check import environment_distributions
    before = doctor._installed_distributions()
    info = tmp_path / 'numpy-0.0.0.dist-info'
    info.mkdir()
    (info / 'METADATA').write_text('Metadata-Version: 2.1\nName: numpy\nVersion: 0.0.0\n')
    monkeypatch.setattr(sys, 'path', sys.path + [str(tmp_path)])
    assert doctor._installed_distributions() == before
    assert before['numpy'] != '0.0.0'
    assert before == {name: version for name, (version, _) in environment_distributions().items()}
