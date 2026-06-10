"""Closed-weight API models, evaluated behaviorally through the unified interface.

A closed-weight model (Claude, GPT-4) exposes only its output, so it registers as
a ``BrainScoreModel`` with a ``generation_fn`` (the API closure) and NO
``activations_model`` — it can run behavioral / instruction-following benchmarks
(object matching, lexical decision, the game) but NOT activation-based neural
encoding. Scoring requires the provider's API key in the environment
(``ANTHROPIC_API_KEY`` / ``OPENAI_API_KEY``); registration + load do not.
"""
from brainscore_core.model_interface import BrainScoreModel
from brainscore.model_helpers.api_behavioral import build_api_generation_fn

# identifier -> (provider, model id, declared modalities). Pin dated snapshots
# where the provider offers them (OpenAI) so behavioral scores reproduce; Claude
# uses the canonical alias + the response cache for reproducibility.
REGISTRY = {
    'claude-opus-behavioral': dict(
        provider='anthropic', model='claude-opus-4-8',
        modalities=('vision', 'text')),
    'claude-haiku-behavioral': dict(
        provider='anthropic', model='claude-haiku-4-5',
        modalities=('vision', 'text')),
    'gpt-4o-behavioral': dict(
        provider='openai', model='gpt-4o-2024-08-06',
        modalities=('vision', 'text')),
    # DeepSeek's API is text-only (no vision) — register text modality only, so
    # the pre-flight check routes only text behavioral benchmarks here.
    # deepseek-chat = V3, deepseek-reasoner = R1. Needs DEEPSEEK_API_KEY.
    'deepseek-chat-behavioral': dict(
        provider='deepseek', model='deepseek-chat',
        modalities=('text',)),
    'deepseek-r1-behavioral': dict(
        provider='deepseek', model='deepseek-reasoner',
        modalities=('text',)),
    # OpenRouter gateway — any model in https://openrouter.ai/models registers
    # by adding a line here with its namespaced id. Declare modalities to match
    # the underlying model (vision only if it's multimodal). Needs OPENROUTER_API_KEY.
    'llama-3.3-70b-behavioral': dict(
        provider='openrouter', model='meta-llama/llama-3.3-70b-instruct',
        modalities=('text',)),
    'or-claude-sonnet-behavioral': dict(
        provider='openrouter', model='anthropic/claude-3.5-sonnet',
        modalities=('vision', 'text')),
}

# Cache lives on disk so re-scoring a benchmark doesn't re-bill the API and the
# scored result reproduces even though the live model may change behind the API.
_CACHE_ROOT = '~/.brainscore/api_behavioral_cache'


def _behavioral_only_preprocessor(modality: str):
    """Nominal preprocessor: declares modality support for the pre-flight
    compatibility check but is never called on the behavioral generation path.
    If a NEURAL benchmark tries to extract activations from a closed API model,
    this fires with a clear, honest error rather than a confusing crash."""
    def _raise(*_args, **_kwargs):
        raise NotImplementedError(
            f"This is a closed-weight API model: it answers behavioral tasks "
            f"via generation, but exposes no internal activations, so it cannot "
            f"run activation-based neural-encoding benchmarks (modality "
            f"'{modality}'). Use it on behavioral / instruction-following "
            f"benchmarks instead.")
    return _raise


def get_model(identifier: str) -> BrainScoreModel:
    if identifier not in REGISTRY:
        raise ValueError(f"unknown closed-API model '{identifier}'; "
                         f"known: {sorted(REGISTRY)}")
    cfg = REGISTRY[identifier]
    generation_fn = build_api_generation_fn(
        provider=cfg['provider'],
        model=cfg['model'],
        max_tokens=16,
        cache_dir=f"{_CACHE_ROOT}/{identifier}",
    )
    return BrainScoreModel(
        identifier=identifier,
        model=None,                 # no local weights — output-only
        region_layer_map={},        # no neural recording targets
        preprocessors={m: _behavioral_only_preprocessor(m)
                       for m in cfg['modalities']},
        activations_model=None,
        generation_fn=generation_fn,
    )
