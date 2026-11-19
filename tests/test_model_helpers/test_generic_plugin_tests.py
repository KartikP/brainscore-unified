"""The generic model plugin suite, run on small stand-in subjects (``load_model`` is patched)."""
import pytest

from brainscore_core.compatibility import CompatibilityError
from brainscore_core.contract import Subject
from brainscore_core.streaming import StreamEvent
# Do not import the ``test_*`` functions by name: pytest would collect them here.
from brainscore.model_helpers import generic_plugin_tests

pytestmark = pytest.mark.unit

LOAD = 'brainscore.model_helpers.generic_plugin_tests.load_model'


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


def _declared(identifier='dummy', in_channels=('text',), out_channels=('behavior:choice',),
              required_channels=(), output_inputs=None, traits=()):
    return _Declared(identifier=identifier, in_channels=set(in_channels),
                     out_channels=set(out_channels), required_channels=set(required_channels),
                     output_inputs=output_inputs or {}, traits=set(traits))


def test_a_valid_declaration_passes_with_one_load(mocker):
    load = mocker.patch(LOAD, return_value=_declared(in_channels=('text', 'vision:image'),
                                                     out_channels=('neural:IT', 'behavior:choice'),
                                                     required_channels=('text',)))
    generic_plugin_tests.test_identifier_and_declaration('dummy')
    load.assert_called_once_with('dummy')


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
def test_a_bad_identifier_or_declaration_fails(mocker, declared):
    mocker.patch(LOAD, return_value=_declared(**declared))
    with pytest.raises((AssertionError, ValueError, CompatibilityError)):
        generic_plugin_tests.test_identifier_and_declaration('dummy')


def test_a_subject_whose_participate_skips_the_base_fails_both_session_checks(mocker):
    class Late(_Declared):
        pass

    def unwrapped(self, session):   # assigned after the class statement: not wrapped
        while session.next_input() is not None:
            pass
    Late.participate = unwrapped
    mocker.patch.object(generic_plugin_tests, '_loaded', return_value=Late(
        identifier='late', in_channels={'text'}, out_channels={'behavior:choice'},
        required_channels=set(), output_inputs={}, traits=set()))
    with pytest.raises(AssertionError):
        generic_plugin_tests.test_an_undeclared_input_never_reaches_the_subject('late')
    with pytest.raises(AssertionError):
        generic_plugin_tests.test_a_second_session_on_the_busy_subject_is_refused('late')
    mocker.patch.object(generic_plugin_tests, '_loaded', return_value=_declared())
    generic_plugin_tests.test_an_undeclared_input_never_reaches_the_subject('dummy')
    generic_plugin_tests.test_a_second_session_on_the_busy_subject_is_refused('dummy')
