"""Closed-weight API models, evaluated behaviorally through the unified interface.

A closed-weight model (Claude, GPT-4) exposes only its output, so it registers as
a ``BrainScoreModel`` with an ``answer`` part (the API closure) and NO
``extractor`` — it can run behavioral / instruction-following benchmarks
(object matching, lexical decision, the game) but NOT activation-based neural
encoding. Scoring requires the provider's API key in the environment
(``ANTHROPIC_API_KEY`` / ``OPENAI_API_KEY``), or, for the ``bedrock`` provider, the
user's AWS credentials with access to the model; registration + load do not. A model
that takes pictures also has the NodeKit act policy (``brainscore.harnesses.nodekit.policy``),
so it takes part in a task on a screen with the keyboard or the pointer.
"""
from brainscore import BrainScoreModel
from brainscore.model_helpers.api_behavioral import api_answer, build_api_reply

# identifier -> (provider, model id, declared input channels). Pin dated snapshots
# where the provider offers them (OpenAI) so behavioral scores reproduce. Claude
# uses the canonical alias; its replies are samples (no temperature, no seed),
# so the response store keeps none of them.
REGISTRY = {
    'claude-opus-behavioral': dict(
        provider='anthropic', model='claude-opus-4-8',
        in_channels=('vision:image', 'text')),
    # Claude Sonnet 5.5: the same model through Amazon Bedrock (the user's AWS credentials) or through
    # the Anthropic API (ANTHROPIC_API_KEY); the registration picks the way.
    'claude-sonnet-5-5-bedrock': dict(
        provider='bedrock', model='us.anthropic.claude-sonnet-5-5',
        in_channels=('vision:image', 'text')),
    'claude-sonnet-5-5-behavioral': dict(
        provider='anthropic', model='claude-sonnet-5-5',
        in_channels=('vision:image', 'text')),
    # Claude Opus 5.5, both ways (Sonnet 5.5 answered NodeKit's briefing with empty tool markup on 2026-11-25).
    'claude-opus-5-5-bedrock': dict(
        provider='bedrock', model='us.anthropic.claude-opus-5-5',
        in_channels=('vision:image', 'text')),
    'claude-opus-5-5-behavioral': dict(
        provider='anthropic', model='claude-opus-5-5',
        in_channels=('vision:image', 'text')),
    'claude-haiku-behavioral': dict(
        provider='anthropic', model='claude-haiku-4-5',
        in_channels=('vision:image', 'text')),
    'gpt-4o-behavioral': dict(
        provider='openai', model='gpt-4o-2024-08-06',
        in_channels=('vision:image', 'text')),
    # DeepSeek's API is text-only (no vision) — declare the text channel only, so
    # the pre-flight check routes only text behavioral benchmarks here.
    # deepseek-chat = V3, deepseek-reasoner = R1. Needs DEEPSEEK_API_KEY.
    'deepseek-chat-behavioral': dict(
        provider='deepseek', model='deepseek-chat',
        in_channels=('text',)),
    'deepseek-r1-behavioral': dict(
        provider='deepseek', model='deepseek-reasoner',
        in_channels=('text',)),
    # OpenRouter gateway — any model in https://openrouter.ai/models registers
    # by adding a line here with its namespaced id. Declare input channels to match
    # the underlying model (images only if it's multimodal). Needs OPENROUTER_API_KEY.
    'llama-3.3-70b-behavioral': dict(
        provider='openrouter', model='meta-llama/llama-3.3-70b-instruct',
        in_channels=('text',)),
    'or-claude-sonnet-behavioral': dict(
        provider='openrouter', model='anthropic/claude-3.5-sonnet',
        in_channels=('vision:image', 'text')),
}

# A task on a screen (NodeKit): the act policy reads these and acts on these forms.
SCREEN_INPUTS = ('vision:image', 'vision:video', 'instruction', 'observation')
SCREEN_FORMS = ('motor:key', 'motor:pointer')
# The behavior outputs the answer and reply parts give (their inputs are stated for a model with a screen policy).
ANSWERED = ('behavior', 'behavior:choice', 'behavior:rating', 'behavior:text')

# Cache lives on disk so re-scoring a benchmark doesn't re-bill the API and the
# scored result reproduces even though the live model may change behind the API.
# The two Anthropic registrations store no reply: each is a sample with no seed.
_CACHE_ROOT = '~/.brainscore/api_behavioral_cache'


def get_model(identifier: str) -> BrainScoreModel:
    """An API model with two parts from one client: ``answer`` for a plain Choice, and
    ``reply`` for every other response task (choices with settings, ratings, written
    answers such as CogGym's), which obeys the task's settings."""
    if identifier not in REGISTRY:
        raise ValueError(f"unknown closed-API model '{identifier}'; "
                         f"known: {sorted(REGISTRY)}")
    cfg = REGISTRY[identifier]
    answer = api_answer(
        provider=cfg['provider'],
        model=cfg['model'],
        max_tokens=16,
        cache_dir=f"{_CACHE_ROOT}/{identifier}",
    )
    reply = build_api_reply(cfg['provider'], cfg['model'], cache_dir=f"{_CACHE_ROOT}/{identifier}")
    screen = 'vision:image' in cfg['in_channels']
    if screen:
        from brainscore.harnesses.nodekit.policy import act_policy
    return BrainScoreModel(
        identifier=identifier,
        in_channels=set(cfg['in_channels']),
        answer=answer,
        reply=reply,
        policy=act_policy(reply) if screen else None,
        policy_in_channels=SCREEN_INPUTS if screen else ('observation',),
        policy_out_channels=SCREEN_FORMS if screen else ('motor',),
        # The frames of a screen task (vision:video) reach the act policy only: the answer and reply parts
        # answer from the model's own inputs (the reply refuses a video part).
        output_inputs={output: set(cfg['in_channels']) for output in ANSWERED} if screen else None,
    )
