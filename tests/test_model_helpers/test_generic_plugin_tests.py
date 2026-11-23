"""The generic model plugin suite, run on small stand-in subjects (``load_model`` is patched)."""
import pytest

from brainscore_core.compatibility import CompatibilityError
from brainscore_core.contract import Subject
from brainscore_core.streaming import StreamEvent
# Do not import the ``test_*`` functions by name: pytest would collect them here.
from brainscore.model_helpers import generic_plugin_tests

pytestmark = pytest.mark.unit


class _Declared(Subject):
    def __init__(self, **declarations):
        self.declarations = declarations

    identifier = property(lambda self: self.declarations['identifier'])
    in_channels = property(lambda self: self.declarations['in_channels'])
    out_channels = property(lambda self: self.declarations['out_channels'])
    required_channels = property(lambda self: self.declarations['required_channels'])
    output_inputs = property(lambda self: self.declarations['output_inputs'])
    traits = property(lambda self: self.declarations['traits'])

    def participate(self, session):
        while (event := session.next_input()) is not None:
            session.respond(StreamEvent(next(iter(self.out_channels)), event.payload, None, {}))


class _SkipsTheBase(_Declared):
    """Reads its methods past the Subject base, as a subject that skips both duties."""
    def __getattribute__(self, name):
        return object.__getattribute__(self, name)


def _declared(identifier='dummy', in_channels=('text',), out_channels=('behavior:choice',),
              required_channels=(), output_inputs=None, traits=()):
    return _Declared(identifier=identifier, in_channels=set(in_channels),
                     out_channels=set(out_channels), required_channels=set(required_channels),
                     output_inputs=output_inputs or {}, traits=set(traits))


def test_a_valid_declaration_passes_with_one_load(monkeypatch):
    subject, loads = _declared(in_channels=('text', 'vision:image'), out_channels=('neural:IT', 'behavior:choice'),
                               required_channels=('text',)), []
    monkeypatch.setattr(generic_plugin_tests, 'load_model', lambda identifier: loads.append(identifier) or subject)
    generic_plugin_tests.test_identifier_and_declaration('dummy')
    assert loads == ['dummy']


@pytest.mark.parametrize('declared', [
    dict(identifier=None),
    dict(identifier='other'),
    dict(in_channels=()),
    dict(out_channels=()),
    dict(in_channels=('no-such-channel',)),
    dict(in_channels=('vision',)),                  # a visual input names its form
    dict(in_channels=('behavior:choice',)),         # reversed direction: an output as input
    dict(in_channels=('text:bogus',)),              # illegal address: text takes none
    dict(required_channels=('audio',)),             # not an input channel
    dict(output_inputs={'behavior:choice': {'audio'}}),   # answers from an input it does not take
    dict(traits=('memory:trial_history',)),         # keeps trials but cannot see where one starts
])
def test_a_bad_identifier_or_declaration_fails(monkeypatch, declared):
    monkeypatch.setattr(generic_plugin_tests, 'load_model', lambda identifier: _declared(**declared))
    with pytest.raises((AssertionError, ValueError, CompatibilityError)):
        generic_plugin_tests.test_identifier_and_declaration('dummy')


def test_a_subject_whose_participate_skips_the_base_fails_both_session_checks(monkeypatch):
    class Skips(_SkipsTheBase):
        def participate(self, session):
            while session.next_input() is not None:
                pass

    _stand_in(monkeypatch, Skips(
        identifier='late', in_channels={'text'}, out_channels={'behavior:choice'},
        required_channels=set(), output_inputs={}, traits=set()))
    with pytest.raises(AssertionError):
        generic_plugin_tests.test_an_undeclared_input_never_reaches_the_subject('late')
    with pytest.raises(AssertionError):
        generic_plugin_tests.test_a_second_session_on_the_busy_subject_is_refused('late')
    _stand_in(monkeypatch, _declared())
    generic_plugin_tests.test_an_undeclared_input_never_reaches_the_subject('dummy')
    generic_plugin_tests.test_a_second_session_on_the_busy_subject_is_refused('dummy')


def _stand_in(monkeypatch, subject):
    monkeypatch.setattr(generic_plugin_tests, '_loaded', lambda *args, **kwargs: subject)
    return subject.identifier


def test_a_valid_subject_that_declares_four_inputs_passes_the_input_check(monkeypatch):
    identifier = _stand_in(monkeypatch, _declared(in_channels=('reward', 'text', 'audio', 'instruction')))
    generic_plugin_tests.test_an_undeclared_input_never_reaches_the_subject(identifier)


def test_a_subject_that_declares_every_probe_input_is_skipped_with_the_reason(monkeypatch):
    channels = [channel for channel, _ in generic_plugin_tests._UNDECLARED]
    identifier = _stand_in(monkeypatch, _declared(in_channels=channels))
    with pytest.raises(pytest.skip.Exception, match='every'):
        generic_plugin_tests.test_an_undeclared_input_never_reaches_the_subject(identifier)


def test_the_input_check_fails_when_no_session_can_be_built(monkeypatch):
    # An invalid probe payload: every session is refused at construction.
    monkeypatch.setattr(generic_plugin_tests, '_UNDECLARED', (('audio', [0.0]),))
    identifier = _stand_in(monkeypatch, _declared())
    with pytest.raises(AssertionError, match='no session'):
        generic_plugin_tests.test_an_undeclared_input_never_reaches_the_subject(identifier)


def test_the_session_checks_fail_when_they_observe_nothing(monkeypatch):
    class ReadsNothing(_Declared):
        def participate(self, session):
            pass

    class Skips(ReadsNothing, _SkipsTheBase):     # never declares and never reads
        pass

    subject = ReadsNothing(identifier='quiet', in_channels={'text'}, out_channels={'behavior:choice'},
                           required_channels=set(), output_inputs={}, traits=set())
    generic_plugin_tests.test_an_undeclared_input_never_reaches_the_subject(_stand_in(monkeypatch, subject))
    _stand_in(monkeypatch, Skips(identifier='quiet', in_channels={'text'},
                                out_channels={'behavior:choice'}, required_channels=set(),
                                output_inputs={}, traits=set()))
    with pytest.raises(AssertionError, match='observed'):
        generic_plugin_tests.test_an_undeclared_input_never_reaches_the_subject('quiet')
    with pytest.raises(AssertionError, match='never started'):
        generic_plugin_tests.test_a_second_session_on_the_busy_subject_is_refused('quiet')


def test_the_guard_check_names_what_the_second_session_did(monkeypatch):
    class Skips(_SkipsTheBase):
        def participate(self, session):     # an empty session fails
            if session.next_input() is None:
                raise ValueError('no input')

    identifier = _stand_in(monkeypatch, Skips(identifier='late', in_channels={'text'},
                                        out_channels={'behavior:choice'}, required_channels=set(),
                                        output_inputs={}, traits=set()))
    with pytest.raises(AssertionError, match='no input'):
        generic_plugin_tests.test_a_second_session_on_the_busy_subject_is_refused(identifier)
