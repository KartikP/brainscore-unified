"""Behavioral ``answer`` part for closed-weight API models (Claude, GPT-4, Gemini).

A closed-weight model exposes only its OUTPUT — not internal activations — so it
plugs straight into ``BrainScoreModel``'s behavioral generation path with NO core
changes: it's just a ``BrainScoreModel(answer=<closure built here>)``. This module builds that closure for the common API
providers.

What it handles (the "graceful" parts):
  - one closure shape for every provider, matching the
    ``answer(stimulus_row, instruction, label_set) -> str`` contract;
  - lazy SDK import + env-var API keys (no hard dependency, no key in code);
  - per-stimulus on-disk response cache keyed by (provider, model, instruction,
    label_set, stimulus) — re-runs don't re-bill AND the scored result reproduces
    even though the live model may change behind the API; an Anthropic reply is
    a sample with no seed and is never stored;
  - the directive of core's ``Choice`` (one text); the reply text is returned
    exactly and ``BrainScoreModel`` parses it with ``Choice.parse``;
  - multimodal (base64 image) and text dispatch from the stimulus row.

``build_api_reply`` builds a ``reply`` part from the same clients (one table,
``PROVIDERS``, one client per provider): it answers
any task (a choice, a rating, a written answer such as CogGym's) from the
trial's prompt and obeys the task's generation settings.

What it intentionally does NOT do: neural-encoding paths. Closed APIs don't expose
layer activations, so an API model has no ``extractor``; it declares its inputs with
``in_channels``, and a neural benchmark gets a clear error.
"""
import base64
import functools
import hashlib
import json
import os
import re
import warnings
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple, Union

from brainscore_core.compatibility import ExecutionNotFeasibleError
from brainscore_core.responses import Choice

_IMG_MIME = {'.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
             '.gif': 'image/gif', '.webp': 'image/webp'}


def _find_column(row, kind: str) -> Optional[str]:
    """The first column of ``row`` that the core catalog binds to the part ``kind``
    (``'image'``, ``'text'``, ...), in the catalog's order, with a usable value.

    It is the rule a trial's prompt uses (``brainscore_core.responses.part_columns``),
    so an ``answer`` part reads the same column as a ``reply`` part.
    """
    from brainscore_core.responses import part_columns
    for column in part_columns(kind):
        try:
            value = row[column]
        except (KeyError, TypeError, IndexError):
            continue
        # pandas Series rows: a missing/NaN cell is not a usable path/string
        if value is not None and not (isinstance(value, float) and value != value):
            return column
    return None


def row_parts(row, part: str) -> Tuple[Optional[str], Optional[str]]:
    """The picture column and the text column of ``row`` (``_find_column``), either may be
    ``None``; a row with neither raises a ValueError that names the accepted columns.

    An answer part shows what the row has: its picture, its text, or both.
    """
    from brainscore_core.responses import part_columns
    image, text = _find_column(row, 'image'), _find_column(row, 'text')
    if image is None and text is None:
        raise ValueError(f"{part}: the stimulus row has no image column and no text column; "
                         f"accepted columns, in order: image {list(part_columns('image'))}, "
                         f"text {list(part_columns('text'))}.")
    return image, text


def _encode_image(path: Union[str, Path]) -> Tuple[str, str]:
    """Return (media_type, base64-data) for an image file."""
    path = Path(path)
    media = _IMG_MIME.get(path.suffix.lower(), 'image/png')
    with open(path, 'rb') as f:
        b64 = base64.standard_b64encode(f.read()).decode('utf-8')
    return media, b64


def answer_text(instruction: Optional[str], label_set: Sequence[str]) -> str:
    """The instruction, then the choice directive of core's ``Choice`` (the one text)."""
    return '\n\n'.join(part for part in ((instruction or '').strip(),
                                          Choice(list(label_set)).directive()) if part)


# ── provider clients ───────────────────────────────────────────────────
# One client per provider, used by every part built here (answer, reply,
# policy): (model, system, messages, settings) -> text, or a StreamEvent with
# the text as payload and extra data (reasoning, token counts) as meta.
# messages are [{'role': 'user'|'assistant', 'content': [part, ...]}] with parts
# {'type': 'text', 'text': ...} or {'type': 'image', 'media_type': ..., 'data': <base64>};
# settings hold max_tokens and any of temperature, top_p, seed. SDKs are
# imported lazily so `unified` has no hard dependency on anthropic / openai; a
# missing SDK or key raises a clear error only when a call is made.

# The Anthropic Messages API has no seed parameter and refuses temperature and
# top_p on current Claude models. A temperature of 1.0 is the API's documented
# default, so the client leaves it out; any other value fails before a request.
_ANTHROPIC_DEFAULT_TEMPERATURE = 1.0


def _anthropic_settings(settings) -> dict:
    refused = [name for name in ('top_p', 'seed') if settings.get(name) is not None]
    temperature = settings.get('temperature')
    if temperature is not None and float(temperature) != _ANTHROPIC_DEFAULT_TEMPERATURE:
        refused.append('temperature')
    if refused:
        raise ExecutionNotFeasibleError(
            f'The Anthropic API refuses the task settings {refused}: it has no seed, and '
            'current Claude models reject temperature and top_p. Only a temperature of 1.0, '
            "the API's default, can run, and it is not sent.")
    return {'max_tokens': settings['max_tokens']}


def _call_anthropic(model, system, messages, settings):
    request = _anthropic_settings(settings)   # checked before the SDK is loaded
    try:
        import anthropic
    except ImportError as e:
        raise RuntimeError(
            "anthropic SDK not installed — `pip install anthropic` to score "
            "Claude models behaviorally.") from e
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    content = lambda parts: [
        {'type': 'text', 'text': p['text']} if p['type'] == 'text' else
        {'type': 'image', 'source': {'type': 'base64', 'media_type': p['media_type'], 'data': p['data']}}
        for p in parts]
    kwargs = dict(model=model, messages=[{'role': m['role'], 'content': content(m['content'])}
                                         for m in messages], **request)
    if system:
        kwargs['system'] = system
    resp = client.messages.create(**kwargs)
    text = ''.join(b.text for b in resp.content if b.type == 'text')   # exactly as given
    reasoning = ''.join(b.thinking for b in resp.content if b.type == 'thinking') or None
    usage = getattr(resp, 'usage', None)
    return _with_meta(text, reasoning, None if usage is None else {
        'input_tokens': usage.input_tokens, 'output_tokens': usage.output_tokens})


def _with_meta(text, reasoning, token_usage):
    """A reply with its extra data: a ``StreamEvent`` with the text as payload and,
    in meta, ``reasoning`` and ``token_usage`` when the provider gives them.
    The text alone when it gives neither."""
    from brainscore_core.streaming import StreamEvent
    meta = {name: value for name, value in (('reasoning', reasoning), ('token_usage', token_usage))
            if value}
    return StreamEvent('behavior:text', text, None, meta) if meta else text


def _make_openai_compatible(base_url=None, api_key_env='OPENAI_API_KEY', label='openai',
                            base_url_env=None) -> Callable:
    """Client for any OpenAI-compatible chat-completions endpoint. DeepSeek,
    Together, Fireworks, vLLM, etc. all speak this protocol — only the base_url
    and key env var differ. ``base_url=None`` is OpenAI itself. It takes every setting.
    ``base_url_env=(variable, default)``: the address is read from that variable at each call (a server
    on the user's own machine), and the key may be absent (such a server needs none)."""
    def call(model, system, messages, settings):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError(
                f"openai SDK not installed — `pip install openai` to score "
                f"{label} models behaviorally.") from e
        url = os.getenv(*base_url_env) if base_url_env else base_url
        key = os.getenv(api_key_env) or ('none' if base_url_env else None)
        client = OpenAI(api_key=key, base_url=url)
        content = lambda parts: [
            {'type': 'text', 'text': p['text']} if p['type'] == 'text' else
            {'type': 'image_url', 'image_url': {'url': f"data:{p['media_type']};base64,{p['data']}"}}
            for p in parts]
        chat = [{'role': 'system', 'content': system}] if system else []
        chat += [{'role': m['role'], 'content': content(m['content'])} for m in messages]
        request = {name: settings[name] for name in ('max_tokens', 'temperature', 'top_p', 'seed')
                   if settings.get(name) is not None}
        try:
            resp = client.chat.completions.create(model=model, messages=chat, **request)
        except Exception as e:
            # A text-only model (e.g. DeepSeek-V3, Llama-3.3) sent an image gets
            # a cryptic provider 404 ("no endpoints found that support image
            # input"). Re-raise with the actual fix instead.
            image = any(p['type'] == 'image' for m in messages for p in m['content'])
            if image and 'image' in str(e).lower():
                raise RuntimeError(
                    f"{label} model '{model}' has no image-input endpoint — it is "
                    f"text-only, so it can't take an image. Register it with text "
                    f"inputs only, show it a text observation (obs_mode='ascii' on "
                    f"GridGameEnv; MiniGrid is frame-only), or use a vision-capable "
                    f"model (e.g. anthropic/claude-3.5-sonnet, openai/gpt-4o, "
                    f"google/gemini-2.0-flash-001).") from e
            raise
        message = resp.choices[0].message
        # DeepSeek names the reasoning reasoning_content; OpenRouter names it reasoning.
        reasoning = getattr(message, 'reasoning_content', None) or getattr(message, 'reasoning', None)
        usage = getattr(resp, 'usage', None)
        return _with_meta(message.content or '', reasoning,   # the content exactly as given
                          None if usage is None else {'input_tokens': usage.prompt_tokens,
                                                      'output_tokens': usage.completion_tokens})
    call.__name__ = f'_call_{label}'
    return call


def _call_bedrock(model, system, messages, settings):
    """Client for Amazon Bedrock's Converse API with the user's own AWS credentials (the AWS CLI's
    configuration, environment or role, read by boto3; the region is the configured one or ``AWS_REGION``).
    ``model`` is a Bedrock model or inference profile id, such as ``us.anthropic.claude-sonnet-5-5``. A Claude
    model takes the settings the Anthropic API takes (no seed; temperature only at its default)."""
    request = (_anthropic_settings(settings) if _samples_only(_call_bedrock, model)
               else {name: settings[name] for name in ('max_tokens', 'temperature', 'top_p')
                     if settings.get(name) is not None})
    if settings.get('seed') is not None:
        raise ExecutionNotFeasibleError('Bedrock\'s Converse API has no seed; leave the seed out of the task.')
    try:
        import boto3
        import botocore.exceptions
    except ImportError as e:
        raise RuntimeError("boto3 not installed: `pip install boto3` to score models on Amazon Bedrock.") from e
    names = {'max_tokens': 'maxTokens', 'temperature': 'temperature', 'top_p': 'topP'}
    content = lambda parts: [
        {'text': p['text']} if p['type'] == 'text' else
        {'image': {'format': p['media_type'].split('/')[1], 'source': {'bytes': base64.b64decode(p['data'])}}}
        for p in parts]
    kwargs = dict(modelId=model, messages=[{'role': m['role'], 'content': content(m['content'])} for m in messages],
                  inferenceConfig={names[name]: value for name, value in request.items()})
    if system:
        # A Claude model keeps the system text in Bedrock's prompt cache: a task's turns repeat it (Kimi K3
        # refuses the cache point).
        kwargs['system'] = [{'text': system}] + ([{'cachePoint': {'type': 'default'}}]
                                                 if 'anthropic.' in str(model) else [])
    try:
        resp = _bedrock_client(os.getenv('AWS_REGION')).converse(**kwargs)
    except botocore.exceptions.NoCredentialsError as e:
        raise RuntimeError('Amazon Bedrock needs AWS credentials: run `aws configure` (or set a profile or role) '
                           'with access to the model, or use an API-key provider.') from e
    blocks = resp['output']['message']['content']
    text = ''.join(b.get('text', '') for b in blocks)        # exactly as given
    reasoning = ''.join(b['reasoningContent'].get('reasoningText', {}).get('text', '')
                        for b in blocks if 'reasoningContent' in b) or None
    usage = resp.get('usage', {})
    return _with_meta(text, reasoning, {'input_tokens': usage.get('inputTokens', 0),
                                        'output_tokens': usage.get('outputTokens', 0),
                                        'cache_read_tokens': usage.get('cacheReadInputTokens', 0),
                                        'cache_write_tokens': usage.get('cacheWriteInputTokens', 0)}
                      if usage else None)


@functools.lru_cache(maxsize=None)
def _bedrock_client(region):
    """One Bedrock runtime client for each region (None: the configured one)."""
    import boto3
    return boto3.client('bedrock-runtime', region_name=region)


def _bedrock_region() -> str:
    """The AWS region a Bedrock request goes to: ``AWS_REGION`` or the configured one."""
    return _bedrock_client(os.getenv('AWS_REGION')).meta.region_name


# Bedrock models that refuse a temperature (they sample at their own default and take no seed): Claude, and
# Moonshot's Kimi K3 and K3.1 ('This model doesn't support the temperature field', 2026-11-25).
_BEDROCK_SAMPLES_ONLY = ('anthropic.', 'moonshotai.kimi-k3')


def _samples_only(call, model) -> bool:
    """Whether the client serves a model that samples at its default temperature and has no seed."""
    return call is _call_anthropic or (call is _call_bedrock and any(name in str(model)
                                                                    for name in _BEDROCK_SAMPLES_ONLY))


PROVIDERS = {
    'anthropic': _call_anthropic,
    # Amazon Bedrock with the user's AWS credentials (no API key): Claude and the other Bedrock models.
    'bedrock': _call_bedrock,
    'openai': _make_openai_compatible(),
    # DeepSeek's API is OpenAI-compatible. Text-only (no vision): register
    # DeepSeek models with modalities=('text',). deepseek-chat = V3,
    # deepseek-reasoner = R1.
    'deepseek': _make_openai_compatible(
        base_url='https://api.deepseek.com',
        api_key_env='DEEPSEEK_API_KEY', label='deepseek'),
    # OpenRouter: one OpenAI-compatible gateway to hundreds of models (OpenAI,
    # Anthropic, Google, Llama, Mistral, DeepSeek, ...). Model ids are
    # namespaced, e.g. 'meta-llama/llama-3.3-70b-instruct',
    # 'anthropic/claude-3.5-sonnet'. Vision passes through for multimodal
    # models. Needs OPENROUTER_API_KEY. Catalog: https://openrouter.ai/models
    'openrouter': _make_openai_compatible(
        base_url='https://openrouter.ai/api/v1',
        api_key_env='OPENROUTER_API_KEY', label='openrouter'),
    # A model served by vLLM's OpenAI-compatible server on the user's own machine: the address is
    # BRAINSCORE_VLLM_URL (default http://localhost:8000/v1); a key only if the server was started with one.
    'vllm': _make_openai_compatible(
        base_url_env=('BRAINSCORE_VLLM_URL', 'http://localhost:8000/v1'),
        api_key_env='BRAINSCORE_VLLM_KEY', label='vllm'),
}


def _one_turn(call, model, system, text, image, max_tokens) -> str:
    """One user turn of text and an optional image; the reply's text.

    The ``answer`` and policy parts send this request. It is the request
    they sent before the clients were merged, so their answers do not
    change: Anthropic gets the image before the text and no temperature;
    an OpenAI-compatible endpoint (or a custom client) gets the text before
    the image and a temperature of 0.
    """
    from brainscore_core.streaming import StreamEvent
    text_part = [{'type': 'text', 'text': text}]
    image_part = ([] if image is None
                  else [{'type': 'image', 'media_type': image[0], 'data': image[1]}])
    settings = _one_turn_settings(call, model, max_tokens)
    content = image_part + text_part if _samples_only(call, model) else text_part + image_part
    reply = call(model, system, [{'role': 'user', 'content': content}], settings)
    return reply.payload if isinstance(reply, StreamEvent) else reply


def _one_turn_settings(call, model, max_tokens) -> dict:
    """The generation settings of ``_one_turn``: no temperature for a Claude model, else 0."""
    return ({'max_tokens': max_tokens} if _samples_only(call, model)
            else {'max_tokens': max_tokens, 'temperature': 0})


def _state_settings(part, provider_name, model, settings) -> None:
    """State the provider, the model and the generation settings of ``part`` in the running
    session's part log, as a caching wrapper states its settings (the score's provenance)."""
    from brainscore._model import parts
    log = parts.active()
    if log is not None:
        log.settings.setdefault(id(part), set()).add(
            json.dumps({'provider': provider_name, 'model': model, **settings}, sort_keys=True))


# The form of a cached text, part of every cache key. Before it, the clients stored
# the provider's text stripped of outer spaces; those entries stay on disk and are
# never read, so each key misses once and then stores the exact text.
_CACHE_FORM = 'exact-text-v1'


class _ResponseCache:
    """Tiny disk cache: sha256 key -> response string, one JSON file per key.
    Pins outputs so re-runs are free and bit-reproducible despite a live model."""

    def __init__(self, cache_dir: Union[str, Path]):
        self.dir = Path(os.path.expanduser(str(cache_dir)))
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.dir / f'{key}.json'

    def get(self, key: str) -> Optional[str]:
        p = self._path(key)
        if p.exists():
            return json.loads(p.read_text())['response']
        return None

    def set(self, key: str, response: str) -> None:
        self._path(key).write_text(json.dumps({'response': response}))


def api_answer(
    provider: Union[str, Callable],
    model: str,
    *,
    system_prompt: Optional[str] = None,
    max_tokens: int = 16,
    cache_dir: Optional[Union[str, Path]] = None,
) -> Callable:
    """Build an ``answer(stimulus_row, instruction, label_set) -> str``
    that answers a forced-choice behavioral task via a closed-weight API model.

    :param provider: ``'anthropic'``, ``'openai'``, ``'deepseek'``,
        ``'openrouter'``, or a callable client ``(model, system, messages,
        settings)`` (used for tests and custom providers; see ``PROVIDERS``).
    :param model: provider model id (e.g. ``'claude-opus-4-8'``,
        ``'gpt-4o-2024-08-06'``). Pin a dated snapshot where the provider
        offers one so scores are reproducible.
    :param cache_dir: if set, responses are cached here per stimulus so re-runs
        don't re-bill and the scored result reproduces. Strongly recommended.
        An Anthropic reply is a sample with no seed and is never stored.
    """
    call = provider if callable(provider) else PROVIDERS[provider]
    provider_name = getattr(provider, '__name__', str(provider))
    cache = _ResponseCache(cache_dir) if cache_dir is not None else None

    def _key(prompt, label_set, stim_key, payload_hash) -> str:
        raw = json.dumps([provider_name, model, prompt, list(map(str, label_set)),
                          stim_key, payload_hash, system_prompt, max_tokens, _CACHE_FORM])
        return hashlib.sha256(raw.encode('utf-8')).hexdigest()

    def generate(stimulus_row, instruction: str, label_set) -> str:
        label_set = list(label_set)
        prompt = answer_text(instruction, label_set)

        img_col, txt_col = row_parts(stimulus_row, 'api_behavioral answer')

        # stimulus identity for the cache key
        try:
            stim_key = str(stimulus_row['stimulus_id'])
        except (KeyError, TypeError):
            stim_key = ''

        # The row's picture, then its text before the prompt; a row with both shows both.
        image, user_text, payload = None, prompt, ''
        if img_col is not None:
            media, b64 = _encode_image(stimulus_row[img_col])
            image, payload = (media, b64), b64
        if txt_col is not None:
            text = str(stimulus_row[txt_col])
            user_text, payload = f"{text}\n\n{prompt}", payload + text
        payload_hash = hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]

        key = _key(prompt, label_set, stim_key, payload_hash)
        _state_settings(generate, provider_name, model, _one_turn_settings(call, model, max_tokens))
        # Anthropic gets no temperature and no seed: its reply is a sample, never stored.
        use_cache = cache is not None and not _samples_only(call, model)
        response = cache.get(key) if use_cache else None
        if response is None:
            response = _one_turn(call, model, system_prompt, user_text, image, max_tokens)
            if use_cache:
                cache.set(key, response)

        return response   # the provider's text, exactly: BrainScoreModel parses it

    return generate


# ── written answers: a reply part ───────────────────────────────────────
# A reply part answers any response a task asks for (a choice, a rating, a
# written answer) from the trial's Prompt, and obeys the task's generation
# settings. CogGym needs one. It uses the same clients as the answer part.


def _reply_messages(prompt) -> Tuple[Optional[str], list]:
    """The system text and the user and assistant turns of one trial.

    The task's instruction becomes the system text, as CogGym's own client
    sends it; the other parts form the user turn in order. Earlier trials
    (``prompt.history``) come first as user and assistant turns.
    """
    def parts(shown, names):
        content, system = [], None
        for (kind, value), name in zip(shown, names or [None] * len(shown)):
            if kind == 'text' and name == 'instruction':
                system = value
            elif kind == 'text':
                content.append({'type': 'text', 'text': value})
            elif kind == 'image':
                media, data = _encode_image(value)
                content.append({'type': 'image', 'media_type': media, 'data': data})
            else:
                raise ExecutionNotFeasibleError(f'hosted models here cannot be shown a {kind!r} part')
        return system, content

    messages = []
    for earlier, answered in prompt.history:
        messages.append({'role': 'user', 'content': parts(earlier.parts, earlier.names)[1]})
        messages.append({'role': 'assistant', 'content': [{'type': 'text', 'text': answered}]})
    system, content = parts(prompt.parts, prompt.names)
    messages.append({'role': 'user', 'content': content})
    return system, messages


def build_api_reply(
    provider: Union[str, Callable],
    model: str,
    *,
    max_tokens: int = 512,
    cache_dir: Optional[Union[str, Path]] = None,
) -> Callable:
    """Build a ``reply(prompt)`` part for a hosted model API.

    The reply returns the text with its extra data: a ``StreamEvent`` whose
    payload is the text and whose meta holds ``reasoning`` and ``token_usage``
    when the provider gives them (the text alone when it gives neither). The
    meta reaches the answer event.

    The reply shows the trial's parts in order (text and images), with the
    task's instruction as the system text, and obeys the generation settings
    of the task's ``Text`` response: ``max_tokens`` (``max_tokens`` here when
    the task gives none), ``temperature`` (0 when the task gives none; the
    Anthropic API's default 1.0 for Claude, also on Bedrock), ``top_p`` and ``seed``.
    A provider that refuses a setting the task gives fails before any request is sent:
    the Anthropic API has no seed and refuses temperature (other than its
    default 1.0) and top_p; Bedrock has no seed.

    Limit: video and audio parts are refused at the first trial that shows one; register the
    model without those inputs so such a task is N/A.

    :param provider: ``'anthropic'``, ``'openai'``, ``'deepseek'``,
        ``'openrouter'`` (an API key in the environment), ``'bedrock'`` (the
        user's AWS credentials), or a callable client
        ``(model, system, messages, settings)`` that returns the text, or a
        ``StreamEvent`` with the text as payload and the extra data as meta.
    :param cache_dir: if set, replies are cached per trial, keyed on the
        provider, the model, the system text, the turns and the settings. A
        sampled call (temperature above 0) with no seed is never cached.
    """
    call = provider if callable(provider) else PROVIDERS[provider]
    provider_name = getattr(provider, '__name__', str(provider))
    cache = _ResponseCache(cache_dir) if cache_dir is not None else None

    def reply(prompt):
        from brainscore_core.responses import Text
        from brainscore_core.streaming import StreamEvent
        response = prompt.response
        settings = {'max_tokens': max_tokens}
        if isinstance(response, Text):
            given = {'max_tokens': response.max_tokens, 'temperature': response.temperature,
                     'top_p': response.top_p, 'seed': response.seed}
            settings.update({name: value for name, value in given.items() if value is not None})
        # The temperature is stated: the Anthropic client sends none and samples at its
        # default; any other client gets 0 when the task gives none.
        settings.setdefault('temperature', _ANTHROPIC_DEFAULT_TEMPERATURE
                            if _samples_only(call, model) else 0)
        if _samples_only(call, model):
            _anthropic_settings(settings)   # fail before the cache or a request
        _state_settings(reply, provider_name, model,
                        {**settings, 'region': _bedrock_region()} if call is _call_bedrock else settings)
        system, messages = _reply_messages(prompt)
        key = hashlib.sha256(json.dumps([provider_name, model, system, messages, settings,
                                         _CACHE_FORM], sort_keys=True).encode('utf-8')).hexdigest()
        # A sampled call without a seed differs on each repetition: do not cache it.
        use_cache = cache is not None and not (
            settings['temperature'] > 0 and settings.get('seed') is None)
        answer = cache.get(key) if use_cache else None
        if answer is None:
            answer = call(model, system, messages, settings)
            if use_cache:
                cache.set(key, {'text': answer.payload, 'meta': answer.meta}
                          if isinstance(answer, StreamEvent) else answer)
        elif isinstance(answer, dict):
            answer = StreamEvent('behavior:text', answer['text'], None, answer['meta'])
        return answer

    return reply


# ── closed-loop game: policy ───────────────────────────────
# The video game is agentic, not forced-choice: it drives the model through
# process(row) -> {'motor': action} one tick at a time (see
# brainscore.harnesses.gymnasium_harness.play_gym_episode). This builds an
# policy closure that shows the rendered frame + the legal-action menu to an
# API model and parses back an action index. Same providers/cache as above.

def _encode_frame(frame, resize: int = 384) -> Tuple[str, str]:
    """A numpy RGB frame -> (media_type, base64 PNG)."""
    import io
    import numpy as np
    from PIL import Image
    img = Image.fromarray(np.asarray(frame).astype('uint8'))
    if resize:
        img = img.resize((resize, resize), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return 'image/png', base64.standard_b64encode(buf.getvalue()).decode('utf-8')


def _parse_action_index(response: str, n_actions: int, legal=None) -> Optional[int]:
    """Parse an action index from the response. A reply that is exactly the name of one
    legal action gives that action. Else the ``Action: <n>`` line decides (a number
    outside the range is no move); with no such line, the last in-range integer; else
    None (the reply cannot be read: the caller marks the move invalid). A sign is part
    of a number: ``-1`` is not ``1``."""
    if not response:
        return None
    names = [str(name).strip().lower() for name in (legal or {}).values()]
    word = response.strip().strip('.').lower()
    if names.count(word) == 1:
        return list(legal)[names.index(word)]
    m = re.search(r'action\s*[:\-]\s*(-?\d+)', response, re.IGNORECASE)
    if m:
        return int(m.group(1)) if 0 <= int(m.group(1)) < n_actions else None
    for tok in reversed(re.findall(r'-?\d+', response)):
        if 0 <= int(tok) < n_actions:
            return int(tok)
    return None


def api_policy(
    provider: Union[str, Callable],
    model: str,
    *,
    obs_mode: str = 'vision',
    max_tokens: int = 512,
    system_prompt: Optional[str] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    frame_resize: int = 384,
    history_window: int = 0,
) -> Callable:
    """Build a ``policy(row) -> {'motor': StreamEvent}`` that plays a grid
    episode (GridGameEnv or Gymnasium/MiniGrid) via a closed-weight API model.
    It reads ``observation`` and ``instruction``: build the model with
    ``policy_in_channels=('observation', 'instruction')``. It keeps its own
    history (``.trace``) and clears it in ``reset()``.

    :param obs_mode: ``'vision'`` (default) sends the rendered RGB frame as an
        image; ``'ascii'`` sends the text board (``observation['ascii']``) — text
        models can play, and even vision models reason better on the ASCII grid
        (DeepSeek-R1 reading ASCII beat the VLMs in the original game). Either way
        the env still renders frames you can animate to watch.
    :param history_window: if > 0, the prompt includes the model's last
        ``history_window`` actions plus two derived signals: a no-op flag when the
        most recent action left the observation unchanged (blocked), and a
        repeated-action warning when the same action was chosen 3+ times running.
        Default 0 keeps each tick memoryless. **Strongly recommended (e.g. 8) for
        multi-step planning envs (MiniGrid):** without it a model re-plans from
        scratch every tick, executes only the first step, and loops forever ("move
        forward, then I'll turn" — but it never turns). The history is what lets it
        notice it's stuck and break the loop.

    Reuses the same provider adapters and response cache as the behavioral
    generation path; an Anthropic reply is a sample with no seed and is never stored. Each ``motor`` event carries the catalog's response meta
    (``RESPONSE_META_KEYS``): the raw reply (``answer``), ``valid``, ``parse_error``
    and ``provenance`` (provider and model). A reply with no readable action gives
    no move: the payload is None and ``valid`` is False (Contract Rules 6.2); the
    benchmark's policy for an invalid trial decides what follows. No GPU / weights
    — laptop-runnable with the provider key in env.

    The returned closure carries a ``.trace`` list: one dict per tick
    ``{'step', 'response', 'action', 'valid', 'payload'}`` where ``response``
    is the model's full raw reply (the reasoning text it wrote before the
    ``Action:`` line) and ``payload`` is a hash of the observation it saw (used
    for no-op detection). Read it to watch *why* the model moved;
    ``.trace.clear()`` to reset between rollouts.
    """
    if obs_mode not in ('vision', 'ascii'):
        raise ValueError(f"obs_mode must be 'vision' or 'ascii'; got {obs_mode!r}")
    from brainscore_core.streaming import StreamEvent

    call = provider if callable(provider) else PROVIDERS[provider]
    provider_name = getattr(provider, '__name__', str(provider))
    cache = _ResponseCache(cache_dir) if cache_dir is not None else None
    trace: list = []

    def _history_block(legal, payload) -> str:
        """Short-term memory: the last ``history_window`` actions + derived
        'you are blocked' / 'you are repeating yourself' signals. Empty unless
        history_window > 0 and at least one tick has happened."""
        if not history_window or not trace:
            return ''
        recent = trace[-history_window:]
        lines = [f"  step {h['step']}: "
                 + ('no readable action' if h['action'] is None
                    else legal.get(h['action'], f"action {h['action']}"))
                 for h in recent]
        warn = ''
        # no-op: did the most recent action leave the observation unchanged?
        if recent[-1].get('payload') == payload:
            warn += ("\nNOTE: your last action did NOT change the view — you are "
                     "blocked or it had no effect. Choose a DIFFERENT action.")
        # consecutive run of the same action with no escape
        run_act, run = trace[-1]['action'], 0
        for h in reversed(trace):
            if h['action'] == run_act:
                run += 1
            else:
                break
        if run >= 3:
            lbl = legal.get(run_act, f'action {run_act}')
            warn += (f"\nNOTE: you have chosen '{lbl}' {run} times in a row with no "
                     f"progress. Break the loop — try turning or another action.")
        return "\n\nYour recent moves (in order):\n" + '\n'.join(lines) + warn

    def reset():
        trace.clear()

    def act(row):
        # The episode's first call finds an empty trace: BrainScoreModel calls
        # act.reset() when an episode begins.
        obs = row.get('observation') or {}
        instruction = obs.get('instruction') or row.get('instruction') or ''
        legal = obs.get('legal_actions') or {}
        n = len(legal) or 7
        menu = ('\n'.join(f"  {i}: {d}" for i, d in sorted(legal.items()))
                if legal else '\n'.join(f"  {i}: action {i}" for i in range(n)))
        tail = ("Reason briefly, then end your reply with a line exactly:\n"
                "Action: <number>")

        if obs_mode == 'ascii':
            board = obs.get('ascii')
            if board is None:
                raise ValueError("api_policy(obs_mode='ascii'): "
                                 "the row's observation has no 'ascii'.")
            image = None
            payload = hashlib.sha256(str(board).encode('utf-8')).hexdigest()[:16]
            prompt = (
                "You control an agent on a grid (text view below).\n"
                f"Mission: {instruction}\n\n{board}\n\n"
                f"Available actions (reply with the NUMBER):\n{menu}"
                f"{_history_block(legal, payload)}\n\n{tail}")
        else:  # vision
            frame = obs.get('frame')
            if frame is None:
                raise ValueError("api_policy(obs_mode='vision'): "
                                 "the row's observation has no 'frame'.")
            media, b64 = _encode_frame(frame, frame_resize)
            image = (media, b64)
            payload = hashlib.sha256(b64.encode('utf-8')).hexdigest()[:16]
            prompt = (
                "You control an agent in a grid world. Look at the image and "
                "choose the single best next action.\n"
                f"Mission: {instruction}\n\n"
                f"Available actions (reply with the NUMBER):\n{menu}"
                f"{_history_block(legal, payload)}\n\n{tail}")

        _state_settings(act, provider_name, model, _one_turn_settings(call, model, max_tokens))
        response = None
        key = None
        use_cache = cache is not None and not _samples_only(call, model)
        if use_cache:
            # Include the policy's own step count (reset with its episode state)
            # so a stuck state (unchanged observation) at different ticks does
            # NOT collapse to one cached action; that would freeze the agent in
            # a no-op loop. Re-running the same episode still replays from the
            # cache. Caching a closed loop is inherently fraught; prefer
            # cache_dir=None for fresh per-tick calls.
            key = hashlib.sha256(
                '|'.join([provider_name, model, prompt, payload,
                          str(len(trace)), str(system_prompt), str(max_tokens),
                          _CACHE_FORM]).encode()
            ).hexdigest()
            response = cache.get(key)
        if response is None:
            response = _one_turn(call, model, system_prompt, prompt, image, max_tokens)
            if use_cache:
                cache.set(key, response)

        parsed = _parse_action_index(response, n, legal)
        valid = parsed is not None
        trace.append({'step': len(trace),
                      'response': response, 'action': parsed, 'valid': valid,
                      'payload': payload})
        return {'motor': StreamEvent('motor', parsed, None, {
            'answer': response, 'valid': valid,
            'parse_error': None if valid else f'no action number from 0 to {n - 1} and no action name in the reply',
            'provenance': {'provider': provider_name, 'model': model},
        })}

    act.reset = reset
    act.trace = trace
    return act
