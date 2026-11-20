"""Checks that run once for every identifier a model plugin registers, after a real load.

Core's plugin runner (``brainscore_core.plugin_management``) finds this file at
``<package>/model_helpers/generic_plugin_tests.py`` and runs it with
``--plugin_directory <plugin folder>``; ``pytest_generate_tests`` then gives each
test the identifiers registered in that folder's ``__init__.py``. A model author
writes no extra file. To run it by hand:

    pytest brainscore/model_helpers/generic_plugin_tests.py --plugin_directory brainscore/models/<plugin>

(``conftest.py`` in this folder adds the ``--plugin_directory`` option.)

The checks hold for every subject and use only the subject contract: the
identifier and the declaration. The subject is loaded once per identifier. A
first check reads the plugin's ``__init__.py``: core's plugin runner reads the
identifier on one registration line, and the file imports no ``model.py`` at
module level (the factory imports it when the model is loaded).

Limit: no declared output is exercised with input; add that when a production
job for unified exists.
"""
import ast
from pathlib import Path
from types import SimpleNamespace

from brainscore_core.compatibility import check_compatibility
# the following import is needed to configure pytest
# noinspection PyUnresolvedReferences
from brainscore_core.plugin_management.generic_plugin_tests_helper import pytest_generate_tests  # noqa: F401
from brainscore_core.plugin_management.parse_plugin_changes import get_plugin_ids
from brainscore import load_model


def test_the_init_states_the_identifier_once_and_loads_nothing(identifier: str, request):
    folder = Path(request.config.option.plugin_directory).resolve()
    init = folder / '__init__.py'
    keys = get_plugin_ids('models', [folder.name], str(folder.parents[1]))  # the runner's reader
    assert keys.count(identifier) == 1, (
        f"write one line model_registry['{identifier}'] = <factory> in {init}")
    statements = ast.parse(init.read_text()).body
    eager = [statement.lineno for statement in statements
             if isinstance(statement, ast.ImportFrom) and statement.level == 1
             and (statement.module == 'model'
                  or (statement.module is None and any(a.name == 'model' for a in statement.names)))]
    assert not eager, (f'{init} imports model.py at module level (line {eager}); '
                       'import it inside the factory')


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

import numpy as np
import pytest

from brainscore_core.contract import TaskContext
from brainscore_core.responses import Choice
from brainscore_core.streaming import Session, StreamEvent, channel_declared

_loaded = lru_cache(maxsize=1)(load_model)
# Input channels with a valid payload; the input check uses the first one the model does not declare.
_UNDECLARED = (('reward', 1.0), ('text', 'word'), ('audio', np.zeros(16)),
               ('instruction', 'answer'), ('vision:image', 'probe.png'), ('vision:video', 'probe.mp4'))


def _sessions(model, events):
    """One session for each declared output, with the first plain task that output accepts."""
    sessions, refusals = [], []
    for output in sorted(model.out_channels):
        for task in (TaskContext(response=Choice(['a', 'b'])), TaskContext(task_type='passive'), None):
            try:
                sessions.append(Session(list(events), outputs=(output,), task=task))
                break
            except (TypeError, ValueError) as refusal:
                refusals.append(f'{output}: {refusal}')
    assert sessions, f'no session could be built for {sorted(model.out_channels)}: {refusals}'
    return sessions


def test_an_undeclared_input_never_reaches_the_subject(identifier: str):
    model = _loaded(identifier)
    probe = next(((channel, payload) for channel, payload in _UNDECLARED
                  if not channel_declared(channel, model.in_channels)), None)
    if probe is None:
        pytest.skip(f'{identifier} declares every probe input; no undeclared input to send')
    channel = probe[0]
    for session in _sessions(model, [StreamEvent(channel, probe[1], None, {})]):
        # Every read of the session is recorded: the body's reads, and one read
        # right after the subject's declaration (the session as the body sees it).
        delivered, read, declare = [], session.next_input, session.declare_inputs
        session.next_input = lambda: delivered.append(read()) or delivered[-1]

        def declare_then_read(channels):
            declare(channels)
            session.next_input()
            return session
        session.declare_inputs = declare_then_read
        failure = None
        try:
            model.participate(session)
        except Exception as error:   # a refusal of a session with no declared input is allowed
            failure = error
        assert delivered, (f'{identifier}: no read of the session was observed (no declaration, '
                           f'no read); participate ended with {failure!r}')
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
                outcome.append('the second session ran to its end')
            except Exception as error:
                outcome.append(error)

    # The second session starts while the first one runs: at the first one's first
    # declaration or read.
    for name in ('declare_inputs', 'next_input'):
        method = getattr(first, name)
        setattr(first, name, lambda *args, method=method: (start_the_second(), method(*args))[1])
    try:
        model.participate(first)
    except Exception:       # only the second session's outcome is checked
        pass
    assert outcome, (f'{identifier}: the second session never started: the first one made no '
                     'declaration and no read')
    assert isinstance(outcome[0], RuntimeError) and 'one session at a time' in str(outcome[0]), (
        f'{identifier} took part in a second session while one ran: {outcome[0]!r}')
