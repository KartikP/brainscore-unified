"""Loaded observers retain explicit call boundaries and restore method wrappers."""
from types import SimpleNamespace

import pytest

from brainscore.experiments import Experiment, ObserveCalls, compare_outputs
from brainscore.tools import attach_tools, load_tool
from tests.test_ablation_tool_loader import FixtureSubject, protocol

pytestmark = pytest.mark.unit


class Observer:
    def __init__(self):
        self.starts, self.results, self.errors = [], [], []

    def on_start(self, call):
        self.starts.append(call.method)

    def on_result(self, call, result):
        self.results.append(call.method)

    def on_error(self, call, error):
        self.errors.append((call.method, type(error)))


def test_loaded_observers_preserve_native_outputs_and_independent_counts(tmp_path):
    subject = FixtureSubject()
    first, second = Observer(), Observer()
    loaded = load_tool('observe-calls', observer=first, methods=['participate'], name='first')
    assert type(loaded) is ObserveCalls
    assert loaded.describe() == ObserveCalls(first, methods=['participate'], name='first').describe()
    Experiment(subject=subject, protocol=protocol(), tools=[load_tool('record-io')],
               output_dir=tmp_path / 'baseline').run()
    Experiment(subject=subject, protocol=protocol(), tools=[load_tool('record-io'), loaded,
               load_tool('observe-calls', observer=second, methods=['participate'], name='second')],
               output_dir=tmp_path / 'observed').run()
    assert compare_outputs(tmp_path / 'baseline', tmp_path / 'observed')['equal']
    assert first.starts == first.results == second.starts == second.results == ['participate'] * 4
    assert not first.errors and not second.errors
    assert 'participate' not in vars(subject)


@pytest.mark.parametrize('error', [None, RuntimeError, KeyboardInterrupt])
def test_legacy_nested_calls_count_once_and_restore_exact_attributes(tmp_path, error):
    subject = SimpleNamespace(identifier='legacy-observer-fixture', process=lambda value: value * 2)

    def look_at(value):
        result = subject.process(value)
        if error is not None:
            raise error('stopped')
        return result

    subject.look_at = look_at
    original_process = subject.process
    observer = Observer()

    def run():
        with attach_tools(subject, tools=[load_tool('observe-calls', observer=observer,
                          methods=['look_at', 'process'])], output_dir=tmp_path / 'run'):
            assert subject.look_at(3) == 6

    if error is None:
        run()
        assert observer.results == ['look_at']
    else:
        with pytest.raises(error, match='stopped'):
            run()
        assert observer.errors == [('look_at', error)]
    assert observer.starts == ['look_at']
    assert subject.process is original_process and subject.look_at is look_at


def test_observer_callback_failure_is_not_swallowed_and_wrappers_restore(tmp_path):
    class Broken(Observer):
        def on_result(self, call, result):
            raise RuntimeError('observer failed')

    subject = SimpleNamespace(identifier='fixture', process=lambda value: value)
    original = subject.process
    with pytest.raises(RuntimeError, match='observer failed'):
        with attach_tools(subject, tools=[load_tool('observe-calls', observer=Broken())],
                          output_dir=tmp_path / 'run'):
            subject.process(1)
    assert subject.process is original


def test_missing_selected_method_fails_before_wrapping_or_writing(tmp_path):
    subject = FixtureSubject()
    with pytest.raises(ValueError, match='does not expose process'):
        with attach_tools(subject, tools=[load_tool('observe-calls', observer=Observer())],
                          output_dir=tmp_path / 'run'):
            pytest.fail('unsupported method entered scope')
    assert 'participate' not in vars(subject) and not (tmp_path / 'run').exists()


@pytest.mark.parametrize('name', ['', ' ', 7])
def test_loaded_observer_rejects_invalid_names(name):
    with pytest.raises(ValueError, match='name'):
        load_tool('observe-calls', observer=Observer(), name=name)
