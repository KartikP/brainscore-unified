"""Checks that run once for every identifier a model plugin registers, after a real load.

Core's plugin runner (``brainscore_core.plugin_management``) finds this file at
``<package>/model_helpers/generic_plugin_tests.py`` and runs it with
``--plugin_directory <plugin folder>``; ``pytest_generate_tests`` then gives each
test the identifiers registered in that folder's ``__init__.py``. A model author
writes no extra file. To run it by hand:

    pytest brainscore/model_helpers/generic_plugin_tests.py --plugin_directory brainscore/models/<plugin>

(``conftest.py`` in this folder adds the ``--plugin_directory`` option.)

The checks hold for every subject and use only the subject contract: the
identifier and the declaration. The subject is loaded once per identifier.

Limit: no declared output is exercised with input; add that when a production
job for unified exists.
"""
from types import SimpleNamespace

from brainscore_core.compatibility import check_compatibility
# the following import is needed to configure pytest
# noinspection PyUnresolvedReferences
from brainscore_core.plugin_management.generic_plugin_tests_helper import pytest_generate_tests  # noqa: F401
from brainscore import load_model


def test_identifier_and_declaration(identifier: str):
    model = load_model(identifier)
    assert model.identifier == identifier
    in_channels, out_channels = set(model.in_channels), set(model.out_channels)
    assert in_channels, 'the model declares no input channel'
    assert out_channels, 'the model declares no output channel'
    # Core's own pre-flight validates the whole declaration (catalog names, forms,
    # directions, addresses, required_channels, output_inputs, traits). The experiment
    # presents the subject's own inputs and requests its own outputs, so only a
    # defect of the declaration can fail.
    check_compatibility(model, SimpleNamespace(
        identifier='declaration-check',
        required_input_channels=in_channels - {'control'},
        requested_output_channels=out_channels))


# --- The session duties of the Subject base: input filter and one session at a time ---
# Behavior checks with no compute: the sessions carry no input the model declares.

from functools import lru_cache

from brainscore_core.contract import TaskContext
from brainscore_core.responses import Choice
from brainscore_core.streaming import Session, StreamEvent, channel_declared

_loaded = lru_cache(maxsize=1)(load_model)
# Input channels with a plain payload; each check uses the first one the model does not declare.
_UNDECLARED = (('reward', 1.0), ('text', 'word'), ('audio', [0.0]), ('instruction', 'answer'))


def _sessions(model, events):
    """One session for each declared output, with the first plain task that output accepts."""
    sessions = []
    for output in sorted(model.out_channels):
        for task in (TaskContext(response=Choice(['a', 'b'])), TaskContext(task_type='passive'), None):
            try:
                sessions.append(Session(list(events), outputs=(output,), task=task))
                break
            except (TypeError, ValueError):
                continue
    return sessions


def test_an_undeclared_input_never_reaches_the_subject(identifier: str):
    model = _loaded(identifier)
    channel, payload = next((channel, payload) for channel, payload in _UNDECLARED
                            if not channel_declared(channel, model.in_channels))
    for session in _sessions(model, [StreamEvent(channel, payload, None, {})]):
        delivered, read = [], session.next_input
        session.next_input = lambda: delivered.append(read()) or delivered[-1]
        try:
            model.participate(session)
        except Exception:   # a refusal of this plain session is not what this check is about
            pass
        assert channel not in {event.channel for event in delivered if event is not None}, (
            f'{identifier} received an input on {channel!r}, which it does not declare')


def test_a_second_session_on_the_busy_subject_is_refused(identifier: str):
    model = _loaded(identifier)
    first, second = _sessions(model, [])[0], _sessions(model, [])[0]
    outcome = []

    def start_the_second():
        if not outcome:
            try:
                model.participate(second)
                outcome.append('ran')
            except RuntimeError as refusal:
                outcome.append(refusal)

    # The second session starts while the first one runs: at the first one's first
    # declaration or read.
    for name in ('declare_inputs', 'next_input'):
        method = getattr(first, name)
        setattr(first, name, lambda *args, method=method: (start_the_second(), method(*args))[1])
    try:
        model.participate(first)
    except Exception:       # only the second session's outcome is checked
        pass
    assert outcome and isinstance(outcome[0], RuntimeError), (
        f'{identifier} took part in a second session while one ran: {outcome}')
    assert 'one session at a time' in str(outcome[0])
