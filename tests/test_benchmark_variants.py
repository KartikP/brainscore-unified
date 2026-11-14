"""Pin the registered benchmarks and models of the unified package, and what each benchmark builds.

A variant of a benchmark is a registry line (`lambda` or `partial` of a class with
arguments). This file pins every identifier, and for each benchmark that builds
without local data: the identifier it builds, its version, its trial memory, and
the channels it sends and requests. The Yeatman2021 benchmarks load hosted data
when they are built; here a synthetic ROAR set stands behind their loaders, so
the test needs no network. A change to a registry line that moves one of
these fails here (Contract Rules 10: identifiers are stable and scores do not move).
"""
import warnings

import pytest

import brainscore
import brainscore.benchmarks  # noqa: F401  (registers the unified benchmarks)
from tests.test_induced_dyslexia import synthetic_roar  # noqa: F401  (fixture)

FAMILIES = ('Algonauts2025-', 'CogGym.', 'EWoK-', 'GridGame-', 'Lahner2024-',
            'LeBel2023-', 'Yeatman2021-')

REGISTERED = [
    'Algonauts2025-friends-sub01',
    'Algonauts2025-friends-sub01-video',
    'Algonauts2025-friends-sub01-video-audio',
    'Algonauts2025-friends-sub02',
    'Algonauts2025-friends-sub03',
    'Algonauts2025-friends-sub05',
    'CogGym.Aboody2025Inferring.exp1',
    'CogGym.Aboody2025Inferring.exp2',
    'CogGym.Bass2022Partial.exp1',
    'CogGym.Chandra2024Cooperative.exp1',
    'CogGym.Fu2025Hierarchical.exp1',
    'CogGym.Fu2025Hierarchical.exp2',
    'CogGym.Hu2023Fine.exp1',
    'CogGym.JaraEttinger2021Quantitative.exp1',
    'CogGym.JaraEttinger2021Quantitative.exp2',
    'CogGym.JaraEttinger2021Quantitative.exp3',
    'CogGym.Levine2020Logic.exp1',
    'CogGym.Levine2020Logic.exp2',
    'CogGym.Levine2020Logic.exp3',
    'CogGym.Levine2020Logic.exp4',
    'CogGym.Levine2020Logic.exp5',
    'CogGym.Levine2020Logic.exp6',
    'CogGym.Tsvilodub2025Nonliteral.exp1',
    'CogGym.Tsvilodub2025Nonliteral.exp3b',
    'CogGym.Yoon2020Polite.exp1',
    'CogGym.Yoon2020Polite.exp2',
    'CogGym.sosa2021Moral.exp1',
    'CogGym.sosa2021Moral.exp2',
    'CogGym.sosa2021Moral.exp3',
    'EWoK-core-1.0-choice',
    'EWoK-core-1.0-logprobs',
    'GridGame-reach-5x5',
    'Lahner2024-fMRI-naturalistic',
    'Lahner2024-fMRI-naturalistic-multimodal',
    'Lahner2024-fMRI-naturalistic-multimodal-auditoryROI',
    'Lahner2024-fMRI-naturalistic-multimodal-visualROI',
    'Lahner2024-fMRI-naturalistic-timeresolved',
    'Lahner2024-fMRI-naturalistic-timeresolved-improved',
    'Lahner2024-fMRI-naturalistic-timeresolved-improved-visualROI',
    'Lahner2024-fMRI-naturalistic-timeresolved-multimodal',
    'Lahner2024-fMRI-naturalistic-timeresolved-multimodal-auditoryROI',
    'Lahner2024-fMRI-naturalistic-timeresolved-multimodal-visualROI',
    'Lahner2024-fMRI-naturalistic-timeresolved-visualROI',
    'Lahner2024-fMRI-naturalistic-visualROI',
    'LeBel2023-UTS03-encoding',
    'LeBel2023-UTS03-encoding-contextwindow',
    'LeBel2023-UTS03-encoding-languagemask',
    'LeBel2023-UTS03-encoding-smoke',
    'Yeatman2021-induced_dyslexia-image',
    'Yeatman2021-induced_dyslexia-text',
    'Yeatman2021-lexical_decision',
    'Yeatman2021-lexical_decision-image',
    'Yeatman2021-lexical_decision-text',
]

# Registry identifier -> (built identifier, version, trial memory, inputs, outputs). A benchmark that
# needs local data to build is checked by its identifier alone.
BUILT = {
    'Algonauts2025-friends-sub01': (
        'Algonauts2025-friends-sub01', 2, 'none',
        ['vision:image'],
        ['neural:IT', 'neural:V1', 'neural:V4']),
    'Algonauts2025-friends-sub01-video': (
        'Algonauts2025-friends-sub01-video', 2, 'none',
        ['vision:video'],
        ['neural:IT', 'neural:V1', 'neural:V4']),
    'Algonauts2025-friends-sub01-video-audio': (
        'Algonauts2025-friends-sub01-video-audio', 2, 'none',
        ['audio', 'vision:video'],
        ['neural:A1', 'neural:IT', 'neural:V1', 'neural:V4']),
    'Algonauts2025-friends-sub02': (
        'Algonauts2025-friends-sub02', 2, 'none',
        ['vision:image'],
        ['neural:IT', 'neural:V1', 'neural:V4']),
    'Algonauts2025-friends-sub03': (
        'Algonauts2025-friends-sub03', 2, 'none',
        ['vision:image'],
        ['neural:IT', 'neural:V1', 'neural:V4']),
    'Algonauts2025-friends-sub05': (
        'Algonauts2025-friends-sub05', 2, 'none',
        ['vision:image'],
        ['neural:IT', 'neural:V1', 'neural:V4']),
    'GridGame-reach-5x5': (
        'GridGame-reach-5x5', 1, 'none',
        ['instruction', 'observation', 'reward'],
        ['motor']),
    'Lahner2024-fMRI-naturalistic': (
        'Lahner2024-fMRI-naturalistic', 2, 'none',
        ['vision:image', 'vision:video'],
        ['neural:IT']),
    'Lahner2024-fMRI-naturalistic-multimodal': (
        'Lahner2024-fMRI-naturalistic-multimodal', 2, 'none',
        ['audio', 'vision:image', 'vision:video'],
        ['neural:A1', 'neural:IT']),
    'Lahner2024-fMRI-naturalistic-multimodal-auditoryROI': (
        'Lahner2024-fMRI-naturalistic-multimodal-auditoryROI', 2, 'none',
        ['audio', 'vision:image', 'vision:video'],
        ['neural:A1', 'neural:IT']),
    'Lahner2024-fMRI-naturalistic-multimodal-visualROI': (
        'Lahner2024-fMRI-naturalistic-multimodal-visualROI', 2, 'none',
        ['audio', 'vision:image', 'vision:video'],
        ['neural:A1', 'neural:IT']),
    'Lahner2024-fMRI-naturalistic-timeresolved': (
        'Lahner2024-fMRI-naturalistic-timeresolved', 2, 'none',
        ['vision:image', 'vision:video'],
        ['neural:IT']),
    'Lahner2024-fMRI-naturalistic-timeresolved-improved': (
        'Lahner2024-fMRI-naturalistic-timeresolved-improved', 2, 'none',
        ['vision:image', 'vision:video'],
        ['neural:IT']),
    'Lahner2024-fMRI-naturalistic-timeresolved-improved-visualROI': (
        'Lahner2024-fMRI-naturalistic-timeresolved-improved-visualROI', 2, 'none',
        ['vision:image', 'vision:video'],
        ['neural:IT']),
    'Lahner2024-fMRI-naturalistic-timeresolved-multimodal': (
        'Lahner2024-fMRI-naturalistic-timeresolved-multimodal', 2, 'none',
        ['audio', 'vision:image', 'vision:video'],
        ['neural:A1', 'neural:IT']),
    'Lahner2024-fMRI-naturalistic-timeresolved-multimodal-auditoryROI': (
        'Lahner2024-fMRI-naturalistic-timeresolved-multimodal-auditoryROI', 2, 'none',
        ['audio', 'vision:image', 'vision:video'],
        ['neural:A1', 'neural:IT']),
    'Lahner2024-fMRI-naturalistic-timeresolved-multimodal-visualROI': (
        'Lahner2024-fMRI-naturalistic-timeresolved-multimodal-visualROI', 2, 'none',
        ['audio', 'vision:image', 'vision:video'],
        ['neural:A1', 'neural:IT']),
    'Lahner2024-fMRI-naturalistic-timeresolved-visualROI': (
        'Lahner2024-fMRI-naturalistic-timeresolved-visualROI', 2, 'none',
        ['vision:image', 'vision:video'],
        ['neural:IT']),
    'Lahner2024-fMRI-naturalistic-visualROI': (
        'Lahner2024-fMRI-naturalistic-visualROI', 2, 'none',
        ['vision:image', 'vision:video'],
        ['neural:IT']),
    'LeBel2023-UTS03-encoding': (
        'LeBel2023-UTS03-encoding', 1, 'none',
        ['text'],
        ['neural:language_system']),
    'LeBel2023-UTS03-encoding-contextwindow': (
        'LeBel2023-UTS03-encoding-contextwindow', 1, 'none',
        ['text'],
        ['neural:language_system']),
    'LeBel2023-UTS03-encoding-languagemask': (
        'LeBel2023-UTS03-encoding-languagemask', 1, 'none',
        ['text'],
        ['neural:language_system']),
    'LeBel2023-UTS03-encoding-smoke': (
        'LeBel2023-UTS03-encoding-smoke', 1, 'none',
        ['text'],
        ['neural:language_system']),
    'Yeatman2021-induced_dyslexia-image': (
        'Yeatman2021-induced_dyslexia-image', 1, 'none',
        ['instruction', 'lesion', 'vision:image'],
        ['behavior', 'neural:VWFA', 'perturbation']),
    'Yeatman2021-induced_dyslexia-text': (
        'Yeatman2021-induced_dyslexia-text', 1, 'none',
        ['instruction', 'lesion', 'text'],
        ['behavior', 'neural:VWFA', 'perturbation']),
    'Yeatman2021-lexical_decision': (
        'Yeatman2021-lexical_decision-image', 1, 'none',
        ['instruction', 'vision:image'],
        ['behavior']),
    'Yeatman2021-lexical_decision-image': (
        'Yeatman2021-lexical_decision-image', 1, 'none',
        ['instruction', 'vision:image'],
        ['behavior']),
    'Yeatman2021-lexical_decision-text': (
        'Yeatman2021-lexical_decision-text', 1, 'none',
        ['instruction', 'text'],
        ['behavior']),
}

def test_the_registered_identifiers_are_the_pinned_list():
    registered = sorted(i for i in brainscore.benchmark_registry if i.startswith(FAMILIES))
    assert registered == sorted(REGISTERED)


@pytest.mark.parametrize('identifier', sorted(BUILT))
def test_a_registry_line_builds_the_pinned_benchmark(identifier, request):
    if identifier.startswith('Yeatman2021-'):
        request.getfixturevalue('synthetic_roar')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            benchmark = brainscore.benchmark_registry[identifier]()
    except FileNotFoundError:  # includes LocalDataMissing
        pytest.skip(f'{identifier} needs local data to build')
    built, version, trial_memory, inputs, outputs = BUILT[identifier]
    assert benchmark.identifier == built
    assert benchmark.version == version
    assert benchmark.trial_memory == trial_memory
    assert sorted(benchmark.required_input_channels) == inputs
    assert sorted(benchmark.requested_output_channels) == outputs


MODELS = [
    'blip2-opt-2.7b', 'blip2-wav2vec2', 'chance-baseline', 'claude-haiku-behavioral',
    'claude-opus-behavioral', 'clip-random-wav2vec2', 'clip-vit-b-32', 'clip-wav2vec2',
    'deepseek-chat-behavioral', 'deepseek-r1-behavioral', 'gpt-4o-behavioral', 'gpt2',
    'llama-3.3-70b-behavioral', 'or-claude-sonnet-behavioral', 'qwen2.5-0.5b-instruct',
    'qwen2.5-vl-3b', 'qwen2.5-vl-3b-bf16', 'qwen2.5-vl-3b-vwfa', 'qwen2.5-vl-7b',
    'qwen2.5-vl-7b-vwfa', 'qwen2.5-vl-wav2vec2', 'qwen3.6-27b', 'random-clip-random-wav2vec2',
    'random-clip-wav2vec2', 'random-vit-b-32', 'random-vjepa1-random-wav2vec2',
    'random-vjepa1-wav2vec2', 'random-wav2vec2-base', 'videomae-base', 'vjepa1-random-wav2vec2',
    'vjepa1-vitl', 'vjepa1-wav2vec2', 'vjepa2-vitl',
]


def test_the_registered_model_identifiers_are_the_pinned_list():
    """A model identifier is stable: adding one adds a line here, removing one is a decision."""
    assert sorted(brainscore.model_registry) == sorted(MODELS)
