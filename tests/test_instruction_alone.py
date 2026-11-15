"""Rule 5.2: an instruction alone does not make a pair applicable.

A subject must take in at least one stimulus input of the benchmark. On ROAR image the
word is an image; a subject that takes in only text and the instruction would answer
from the instruction alone, with the word dropped. It is N/A before any call. The
three text-only API models are N/A there (a stated change) and stay applicable on ROAR
text, the form they can take. No weights, no network: stand-ins and fake providers.
"""
import math

import pytest

import brainscore
from brainscore import BrainScoreModel
from brainscore.benchmarks.roar_yeatman2021.benchmark import Yeatman2021LexicalDecision
from brainscore_core.compatibility import check_channel_compatibility
from tests.test_causal_lm import subject as causal_lm  # noqa: F401  (fixture)
from tests.test_induced_dyslexia import synthetic_roar  # noqa: F401  (fixture)

pytestmark = pytest.mark.unit

TEXT_ONLY_API_MODELS = ('deepseek-chat-behavioral', 'deepseek-r1-behavioral', 'llama-3.3-70b-behavioral')
ONLY_INSTRUCTION = ("takes in only the instruction of benchmark 'Yeatman2021-lexical_decision-image', "
                    "none of its stimulus inputs {'vision:image'}; an instruction alone does not make "
                    "a pair applicable")


def _score(subject, benchmark):
    return brainscore.score(subject, benchmark, check_mem=False)


def _counting(calls):
    return BrainScoreModel('text-reader', in_channels={'text'},
                           reply=lambda prompt: calls.append('reply') or 'real',
                           answer=lambda **kwargs: calls.append('answer') or 'real')


def test_a_text_only_reader_is_not_applicable_on_roar_image_before_any_call(synthetic_roar):
    calls = []
    score = _score(_counting(calls), Yeatman2021LexicalDecision(channels=('vision:image',), identifier='Yeatman2021-lexical_decision-image'))
    assert score.attrs['status'] == 'N/A' and math.isnan(float(score))
    assert ONLY_INSTRUCTION in score.attrs['reason']
    assert calls == []


def test_the_causal_lm_is_not_applicable_on_roar_image_before_any_forward_pass(causal_lm, synthetic_roar):
    """The pair that became applicable through the instruction alone at S7 (plan point s)."""
    model, net, _ = causal_lm
    calls = []
    hook = net.register_forward_pre_hook(lambda module, args: calls.append(1))
    try:
        score = _score(model, Yeatman2021LexicalDecision(channels=('vision:image',), identifier='Yeatman2021-lexical_decision-image'))
    finally:
        hook.remove()
    assert score.attrs['status'] == 'N/A' and ONLY_INSTRUCTION in score.attrs['reason']
    assert calls == []


def test_a_picture_and_text_reader_is_applicable_on_roar_image(synthetic_roar):
    subject = BrainScoreModel('picture-reader', in_channels={'vision:image', 'text'},
                              reply=lambda prompt: 'real')
    match = check_channel_compatibility(subject, Yeatman2021LexicalDecision(channels=('vision:image',), identifier='Yeatman2021-lexical_decision-image'))
    assert match.consumed_inputs == {'vision:image', 'instruction'}


def test_a_text_only_reader_is_applicable_on_roar_text(synthetic_roar):
    calls = []
    score = _score(_counting(calls), Yeatman2021LexicalDecision(channels=('text',), identifier='Yeatman2021-lexical_decision-text'))
    assert 'status' not in score.attrs and score.attrs['n_trials'] == 100
    assert calls


def test_a_subject_with_only_the_instruction_is_not_applicable_on_a_custom_benchmark():
    from types import SimpleNamespace
    from brainscore_core import Subject

    class InstructionOnly(Subject):
        identifier = 'instruction-only'
        in_channels = {'instruction'}
        out_channels = {'behavior:choice'}

        def participate(self, session):
            raise AssertionError('never called')

    benchmark = SimpleNamespace(identifier='custom-reading', trial_memory='none',
                                required_input_channels={'instruction', 'text'},
                                requested_output_channels={'behavior:choice'})
    from brainscore_core.compatibility import CompatibilityError
    with pytest.raises(CompatibilityError, match="takes in only the instruction of benchmark 'custom-reading', "
                                                 r"none of its stimulus inputs \{'text'\}"):
        check_channel_compatibility(InstructionOnly(), benchmark)


@pytest.mark.parametrize('identifier', TEXT_ONLY_API_MODELS)
def test_a_text_only_api_model_is_na_on_roar_image_and_applicable_on_roar_text(identifier, tmp_path,
                                                                             monkeypatch, synthetic_roar):
    """A stated change of stage 3: before, these pairs were applicable through the
    instruction alone and scored the instruction without the word."""
    from brainscore.model_helpers import api_behavioral
    from brainscore.models.api_closed import model as api_closed
    calls = []

    def provider(model, system, messages, settings):
        calls.append(model)
        return 'real'
    for name in ('deepseek', 'openrouter'):
        monkeypatch.setitem(api_behavioral.PROVIDERS, name, provider)
    monkeypatch.setattr(api_closed, '_CACHE_ROOT', str(tmp_path))
    subject = api_closed.get_model(identifier)
    assert subject.in_channels >= {'text', 'instruction'} and 'vision:image' not in subject.in_channels
    image = _score(subject, Yeatman2021LexicalDecision(channels=('vision:image',), identifier='Yeatman2021-lexical_decision-image'))
    assert image.attrs['status'] == 'N/A' and ONLY_INSTRUCTION in image.attrs['reason']
    assert calls == []
    match = check_channel_compatibility(subject, Yeatman2021LexicalDecision(channels=('text',), identifier='Yeatman2021-lexical_decision-text'))
    assert match.consumed_inputs == {'text', 'instruction'}
