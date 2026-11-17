"""The declared answer formats come from the owner statements only (rules 2.4, 4.11).

A format is declared when exactly one part gives it natively or a statement
names its owner; ``behavior:rt`` with ``behavior:probabilities``. A derived
format with no statement is not declared. Bare ``behavior`` needs ``answer=`` or
a readout with an extractor.

The registration rows build the ``BrainScoreModel(...)`` calls of the library named in the tables
(found by the scan of ``test_answer_ownership``) with stand-in parts, its real
statement, the same two extractors (text and vision), a ``VWFA`` region and an
intervention, so that the pre-flight matrix shows the behavior part only.
"""
from functools import lru_cache
from types import SimpleNamespace

import pytest

import brainscore
from brainscore import BrainScoreModel
from brainscore_core.compatibility import (CompatibilityError, ExecutionNotFeasibleError,
                                           benchmark_required_input_channels, check_channel_compatibility)
from brainscore_core.contract import TaskContext
from brainscore_core.responses import Choice, Scale, Text
from brainscore_core.streaming_helpers import stimulus_session
from tests.brainscore_model.test_answer_ownership import _registrations
from tests.brainscore_model.test_behavior_paths import FITTING, STIMULI, _features
from tests.brainscore_model.stage2_answers_baseline import RATINGS
from tests.test_causal_lm import subject as causal_lm  # noqa: F401  (fixture)
from tests.test_ewok import native, prepared  # noqa: F401  (fixtures)
from tests.test_induced_dyslexia import synthetic_roar  # noqa: F401  (fixture)

pytestmark = pytest.mark.unit

PARTS = {'answer': lambda stimulus_row, instruction, label_set: label_set[0], 'reply': lambda prompt: 'cat',
         'likelihood': lambda prompt, labels: {label: -float(i) for i, label in enumerate(labels)},
         'behavioral_readout_layer': 'late'}
B, CH, P, RA, RT, TE = ('behavior', 'behavior:choice', 'behavior:probabilities',
                        'behavior:rating', 'behavior:rt', 'behavior:text')
LP = 'behavior:log_probabilities'
NEURAL_ONLY = ('benchmarks/grid_game/benchmark.py', 'models/multimodal_av_blip2_wav2vec2/model.py',
               'models/multimodal_av_clip_wav2vec2/model.py', 'models/multimodal_av_qwen_wav2vec2/model.py',
               'models/multimodal_av_vjepa_wav2vec2/model.py', 'models/random_wav2vec2/model.py',
               'models/videomae_base/model.py', 'models/vjepa/model.py', 'models/vjepa_v1/model.py',
               'tools/auto_register.py', 'validation/benchmark_parity.py')
# Declared behavior formats per library call, before the owner statements and now (a
# likelihood part now also gives behavior:log_probabilities natively).
# The Qwen2.5-VL builder states the probabilities part of each registration (not a literal
# the scan reads; tests/test_qwen_probability_statements.py checks it); qwen3.6-27b has its parts.
DECLARED = {
    'model_helpers/causal_lm.py': ({B, CH, P, RA, RT, TE}, {CH, P, LP, RA, RT, TE}),
    'models/api_closed/model.py': ({B, CH, RA, TE}, {B, CH, RA, TE}),
    'models/blip2_opt_2_7b/model.py': ({B, CH, P, RA, RT}, {B, CH, P, RA, RT}),
    'models/clip_vit_b_32/model.py': ({B, CH, P, RA, RT}, {B, CH, P, RA, RT}),
    'models/gpt2_small/model.py': ({B, CH, P, RA, RT}, {B, CH, P, RA, RT}),
    'models/qwen36_27b/model.py': ({B, CH, P, RA, RT, TE}, {B, CH, P, LP, RA, RT, TE}),
    'models/random_vit_b_32/model.py': ({B, CH, P, RA, RT}, {B, CH, P, RA, RT}),
    'tools/rajalingham2018_2afc/score_2afc.py': ({B, CH}, {B, CH}),
    **{where: (set(), set()) for where in NEURAL_ONLY},
}
# The requested outputs and inputs of the benchmarks (their declarations; the
# test below checks ROAR and EWoK against the real classes).
BENCHMARKS = {
    'ROAR image': ({'vision:image', 'instruction'}, {CH}),
    'ROAR text': ({'text', 'instruction'}, {CH}),
    'dyslexia image': ({'vision:image', 'instruction', 'lesion'}, {CH, 'neural:VWFA', 'perturbation'}),
    'dyslexia text': ({'text', 'instruction', 'lesion'}, {CH, 'neural:VWFA', 'perturbation'}),
    'EWoK choice': ({'text'}, {CH}),
    'EWoK logprobs': ({'text'}, {LP}),
    'CogGym': ({'instruction', 'text'}, {TE}),
    'facade': ({'text'}, {B}),
}
ALL = set(BENCHMARKS)
# The benchmarks each library call is compatible with (else N/A), before and after.
COMPATIBLE = {
    'model_helpers/causal_lm.py': (ALL, ALL - {'facade'}),
    'models/api_closed/model.py': (ALL - {'EWoK logprobs'}, ALL - {'EWoK logprobs'}),
    'models/blip2_opt_2_7b/model.py': (ALL - {'CogGym'}, ALL - {'CogGym', 'EWoK logprobs'}),
    'models/clip_vit_b_32/model.py': (ALL - {'CogGym'}, ALL - {'CogGym', 'EWoK logprobs'}),
    'models/gpt2_small/model.py': (ALL - {'CogGym'}, ALL - {'CogGym', 'EWoK logprobs'}),
    'models/qwen36_27b/model.py': (ALL, ALL),
    'models/random_vit_b_32/model.py': (ALL - {'CogGym'}, ALL - {'CogGym', 'EWoK logprobs'}),
    'tools/rajalingham2018_2afc/score_2afc.py': (ALL - {'EWoK logprobs', 'CogGym'},
                                                 ALL - {'EWoK logprobs', 'CogGym'}),
    **{where: (set(), set()) for where in NEURAL_ONLY},
}


def _stand_in(wired, stated, calls=None):
    calls = [] if calls is None else calls

    def extract(stimuli, layers, **kwargs):
        calls.append('extract')
        return _features(stimuli, layers, **kwargs)
    parts = {name: (lambda f, name: lambda *a, **k: calls.append(name) or f(*a, **k))(PARTS[name], name)
             if callable(PARTS[name]) else PARTS[name] for name in wired}
    return BrainScoreModel('stand-in', extractor={'text': extract, 'vision': extract},
                           region_layer_map={'VWFA': 'late'}, intervention=lambda *a, **k: None,
                           capability_config={'behavior-sessions': stated}, **parts)


def _behavior(model):
    return {c for c in model.out_channels if c == B or c.startswith('behavior:')}


def _compatible(model):
    found = set()
    for name, (inputs, outputs) in BENCHMARKS.items():
        benchmark = SimpleNamespace(identifier=name, required_input_channels=frozenset(inputs),
                                    requested_output_channels=frozenset(outputs))
        try:
            check_channel_compatibility(model, benchmark)
            found.add(name)
        except CompatibilityError:
            pass
    return found


@lru_cache(maxsize=None)
def _scan():
    # The source scan runs inside a test, never at import: one plugin the scan cannot
    # read must not stop the collection of the tier.
    return _registrations()[0]


def _calls(where):
    """The library calls in the file ``where``, as (wired parts, stated owners)."""
    calls = [(wired, stated) for path, wired, stated in _scan() if path == where]
    assert calls, (f'the table names {where!r}, but the library has no readable BrainScoreModel call '
                   'there: remove its row from DECLARED and COMPATIBLE, or rename it')
    return calls


def test_the_two_tables_name_the_same_library_calls():
    """A library call that is in neither table passes: the rule-based tests below
    (session support, every declared format answered) cover every call the scan reads."""
    assert set(DECLARED) == set(COMPATIBLE)


@pytest.mark.parametrize('where', sorted(DECLARED))
def test_each_library_call_declares_the_formats_of_the_table(where):
    for wired, stated in _calls(where):
        assert _behavior(_stand_in(wired, stated)) == DECLARED[where][1]


@pytest.mark.parametrize('where', sorted(COMPATIBLE))
def test_each_library_call_is_compatible_where_the_matrix_says(where):
    for wired, stated in _calls(where):
        assert _compatible(_stand_in(wired, stated)) == COMPATIBLE[where][1]


def test_the_matrix_changes_only_where_the_declarations_change():
    """The changed cells: causal_lm loses bare behavior, so
    it is N/A on the facade; ROAR and dyslexia now request
    behavior:choice, which causal_lm gives through its reply, so those cells are
    applicable again (the stand-in maps a VWFA region; the real registration does not). The readout-only
    registrations state ('readout', 'argmax') and keep behavior:choice. EWoK logprobs now
    requests behavior:log_probabilities, which only a likelihood
    gives, so the readout-only registrations are N/A there (before, the run raised:
    raw scores came only from a likelihood)."""
    changed = {(where, name) for where, (before, after) in COMPATIBLE.items() for name in before ^ after}
    assert changed == {('model_helpers/causal_lm.py', 'facade')} | {
        (where, 'EWoK logprobs') for where in ('models/blip2_opt_2_7b/model.py', 'models/clip_vit_b_32/model.py',
                                               'models/gpt2_small/model.py', 'models/random_vit_b_32/model.py')}
    assert all(after <= before for before, after in COMPATIBLE.values())


def test_no_pair_that_answers_today_becomes_not_applicable():
    """Each newly N/A pair was an error of the run before: no part could answer its task.
    A readout without a choice statement (no registration) could not answer EWoK choice."""
    ewok_choice = TaskContext(response=Choice(labels=('1', '2'), verbatim=True), trial_memory='none')
    with pytest.raises(ExecutionNotFeasibleError, match='Paths tried'):
        _stand_in(['behavioral_readout_layer'], {})._behavioral._choose_path(ewok_choice)


def test_the_benchmark_declarations_match_the_matrix(synthetic_roar, prepared):
    from brainscore.benchmarks.roar_yeatman2021.benchmark import Yeatman2021LexicalDecision
    for word, name in (('vision:image', 'ROAR image'), ('text', 'ROAR text')):
        roar = Yeatman2021LexicalDecision(channels=(word,), identifier='Yeatman2021-test')
        assert (benchmark_required_input_channels(roar), set(roar.requested_output_channels)) == BENCHMARKS[name]
    for mode in ('choice', 'logprobs'):
        ewok = brainscore.load_benchmark(f'EWoK-core-1.0-{mode}', root=prepared)
        assert (set(ewok.required_input_channels), set(ewok.requested_output_channels)) == \
            BENCHMARKS[f'EWoK {mode}']


# --- the plan's acceptance lines ---------------------------------------------

def test_a_stated_readout_choice_is_declared_and_derived_by_argmax():
    model = _stand_in(['behavioral_readout_layer'], {CH: ('readout', 'argmax')})
    assert CH in model.out_channels
    session = stimulus_session(STIMULI, [CH], TaskContext(instruction='Which?', response=Choice(['cat', 'dog']),
                                                         fitting_stimuli=FITTING))
    model.participate(session)
    record = session.collect(CH).attrs['response']
    assert (record['origin'], record['conversion'], record['parameters']['option_order']) == \
        ('derived', 'argmax', 'readout')


def test_causal_lm_answers_roar_text_through_its_reply(causal_lm, synthetic_roar):
    """ROAR requests behavior:choice (S7): a reply gives it, parsed by the model."""
    from brainscore.benchmarks.roar_yeatman2021.benchmark import Yeatman2021LexicalDecision
    model, net, _ = causal_lm
    assert B not in model.out_channels and CH in model.out_channels
    sessions = []
    participate = model.participate
    model.participate = lambda session: sessions.append(session) or participate(session)
    score = brainscore.score(model, Yeatman2021LexicalDecision(channels=('text',), identifier='Yeatman2021-lexical_decision-text'), check_mem=False)
    assert score.attrs['n_trials'] == 100
    assert sessions[0].collect(CH).attrs['response']['parsed_by'] == 'model'


def test_a_readout_only_model_on_ewok_choice_is_not_applicable(prepared, tmp_path):
    calls = []
    model = _stand_in(['behavioral_readout_layer'], {}, calls=calls)
    benchmark = brainscore.load_benchmark('EWoK-core-1.0-choice', root=prepared, output_dir=tmp_path / 'run')
    score = brainscore.score(model, benchmark, check_mem=False)
    assert score.attrs['status'] == 'N/A' and calls == []


def test_a_readout_layer_without_an_extractor_declares_no_behavior():
    model = BrainScoreModel('m', in_channels={'text'}, behavioral_readout_layer='late')
    assert _behavior(model) == set()


# --- reviewer questions --------------------------------------------------------

def test_session_support_uses_the_declared_set():
    """For every library call the scan reads, a new one included."""
    for where, wired, stated in _scan():
        model = _stand_in(wired, stated)
        behavior = next(c for c in model._declared_capabilities() if c.identifier == 'behavior-sessions') \
            if _behavior(model) else None
        for channel in (B, CH, P, LP, RA, RT, TE):
            supported = behavior is not None and behavior.supports_session(model, [channel])
            assert supported == (channel in model.out_channels), (where, channel)


# One session per format that requests that format alone.
SESSIONS = {
    B: TaskContext(instruction='Which?', response=Choice(['cat', 'dog']), fitting_stimuli=FITTING),
    CH: TaskContext(instruction='Which?', response=Choice(['cat', 'dog']), fitting_stimuli=FITTING),
    P: TaskContext(instruction='Which?', response=Choice(['cat', 'dog']),
                   fitting_stimuli=FITTING),
    RA: TaskContext(instruction='How much?', response=Scale(1, 7), fitting_stimuli=RATINGS),
    TE: TaskContext(instruction='Say it.', response=Text()),
    LP: TaskContext(response=Choice(['cat', 'dog'], verbatim=True)),
    RT: TaskContext(instruction='Which?', response=Choice(['cat', 'dog'], reaction_time='surprisal'),
                    fitting_stimuli=FITTING),
}


def test_every_declared_format_is_answered():
    """For every library call with an answer part that the scan reads, a new one included."""
    for where, wired, stated in _scan():
        model = _stand_in(wired, stated)
        for channel in sorted(_behavior(model)):
            session = stimulus_session(STIMULI, [channel], SESSIONS[channel])
            model.participate(session)
            assert session.collect(channel).sizes['presentation'] == len(STIMULI), (where, channel)


def test_a_behavior_request_without_a_task_shows_the_line_to_write_before_any_answer():
    from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
    calls = []
    model = BrainScoreModel('answers', in_channels={'text'},
                            answer=lambda stimulus_row, instruction, label_set: calls.append(1) or 'a')
    table = StimulusSet({'stimulus_id': ['s'], 'sentence': ['x']})
    table.identifier = 'no-task'
    with pytest.raises(ExecutionNotFeasibleError, match=r'needs a task.*TaskContext\(response=Choice\(\[\.\.\.\]\)\)'):
        model.participate(stimulus_session(table, ['behavior:choice']))
    assert calls == []
