"""Behavioral ``generation_fn`` for closed-weight API models (Claude, GPT-4, Gemini).

A closed-weight model exposes only its OUTPUT — not internal activations — so it
plugs straight into ``BrainScoreModel``'s behavioral generation path with NO core
changes: it's just a ``BrainScoreModel(generation_fn=<closure built here>,
activations_model=None)``. This module builds that closure for the common API
providers.

What it handles (the "graceful" parts):
  - one closure shape for every provider, matching the
    ``generation_fn(stimulus_row, instruction, label_set) -> str`` contract;
  - lazy SDK import + env-var API keys (no hard dependency, no key in code);
  - per-stimulus on-disk response cache keyed by (provider, model, instruction,
    label_set, stimulus) — re-runs don't re-bill AND the scored result reproduces
    even though the live model may change behind the API;
  - robust label parsing (exact -> word-boundary -> defer to caller's fallback);
  - multimodal (base64 image) and text dispatch from the stimulus row.

What it intentionally does NOT do: neural-encoding paths. Closed APIs don't expose
layer activations, so an API model has ``activations_model=None`` and its nominal
preprocessors raise a clear error if a neural benchmark tries to extract from it.
"""
import base64
import hashlib
import json
import os
import re
import warnings
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple, Union

# Stimulus-set columns we know how to read, in priority order. Mirrors the
# column conventions the wrappers use elsewhere; vision wins over text.
IMAGE_COLUMNS = ('image_file_name', 'image_path', 'image_file', 'filepath',
                 'stimulus_path')
TEXT_COLUMNS = ('sentence', 'text', 'word', 'transcript', 'stimulus')

_IMG_MIME = {'.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
             '.gif': 'image/gif', '.webp': 'image/webp'}


def _find_column(row, candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        try:
            val = row[c]
        except (KeyError, TypeError):
            continue
        # pandas Series rows: a missing/NaN cell is not a usable path/string
        if val is not None and not (isinstance(val, float) and val != val):
            return c
    return None


def _encode_image(path: Union[str, Path]) -> Tuple[str, str]:
    """Return (media_type, base64-data) for an image file."""
    path = Path(path)
    media = _IMG_MIME.get(path.suffix.lower(), 'image/png')
    with open(path, 'rb') as f:
        b64 = base64.standard_b64encode(f.read()).decode('utf-8')
    return media, b64


def _build_instruction(instruction: str, label_set: Sequence[str]) -> str:
    """Append a forced-choice directive so the response is parseable."""
    opts = ', '.join(str(l) for l in label_set)
    base = instruction.strip() if instruction else ''
    return (f"{base}\n\nAnswer with exactly one of: {opts}. "
            f"Reply with only that word, nothing else.").strip()


def _parse_label(response: str, label_set: Sequence[str]) -> Optional[str]:
    """Map a free-text response to a label. Exact (case-insensitive) match
    first, then a word-boundary substring match, else None so the caller's
    own fallback (BrainScoreModel defaults to label_set[0] + warns) fires."""
    if response is None:
        return None
    resp = response.strip().lower()
    for lbl in label_set:
        if resp == str(lbl).strip().lower():
            return lbl
    for lbl in label_set:
        if re.search(rf'\b{re.escape(str(lbl).strip().lower())}\b', resp):
            return lbl
    return None


# ── provider adapters ────────────────────────────────────────────────────
# Each adapter: (model, system, user_text, image|None, max_tokens) -> str
# image is (media_type, b64) or None. SDKs are imported lazily so `unified`
# carries no hard dependency on anthropic / openai; a missing SDK or key
# raises a clear, actionable error only when a call is actually made.

def _call_anthropic(model, system, user_text, image, max_tokens) -> str:
    try:
        import anthropic
    except ImportError as e:
        raise RuntimeError(
            "anthropic SDK not installed — `pip install anthropic` to score "
            "Claude models behaviorally.") from e
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    content = []
    if image is not None:
        media, b64 = image
        content.append({'type': 'image', 'source': {
            'type': 'base64', 'media_type': media, 'data': b64}})
    content.append({'type': 'text', 'text': user_text})
    kwargs = dict(model=model, max_tokens=max_tokens,
                  messages=[{'role': 'user', 'content': content}])
    if system:
        kwargs['system'] = system
    # NOTE: no temperature — the newest Claude models (Fable 5 / Opus 4.8 / 4.7)
    # reject temperature with a 400. Reproducibility comes from the response cache.
    resp = client.messages.create(**kwargs)
    return ''.join(b.text for b in resp.content if b.type == 'text').strip()


def _make_openai_compatible(base_url=None, api_key_env='OPENAI_API_KEY',
                            label='openai') -> Callable:
    """Adapter for any OpenAI-compatible chat-completions endpoint. DeepSeek,
    Together, Fireworks, vLLM, etc. all speak this protocol — only the base_url
    and key env var differ. ``base_url=None`` is OpenAI itself."""
    def call(model, system, user_text, image, max_tokens) -> str:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError(
                f"openai SDK not installed — `pip install openai` to score "
                f"{label} models behaviorally.") from e
        client = OpenAI(api_key=os.getenv(api_key_env), base_url=base_url)
        content = [{'type': 'text', 'text': user_text}]
        if image is not None:
            media, b64 = image
            content.append({'type': 'image_url', 'image_url': {
                'url': f'data:{media};base64,{b64}'}})
        messages = []
        if system:
            messages.append({'role': 'system', 'content': system})
        messages.append({'role': 'user', 'content': content})
        try:
            resp = client.chat.completions.create(
                model=model, max_tokens=max_tokens, temperature=0, messages=messages)
        except Exception as e:
            # A text-only model (e.g. DeepSeek-V3, Llama-3.3) sent an image gets
            # a cryptic provider 404 ("no endpoints found that support image
            # input"). Re-raise with the actual fix instead.
            if image is not None and 'image' in str(e).lower():
                raise RuntimeError(
                    f"{label} model '{model}' has no image-input endpoint — it is "
                    f"text-only, so it can't take the rendered frame. Use a text "
                    f"observation instead: obs_mode='ascii' on GridGameEnv (which "
                    f"has a text board), or switch to a vision-capable model "
                    f"(e.g. anthropic/claude-3.5-sonnet, openai/gpt-4o, "
                    f"google/gemini-2.0-flash-001). Note: MiniGrid is frame-only "
                    f"(no ASCII board) so it requires a vision model.") from e
            raise
        return (resp.choices[0].message.content or '').strip()
    call.__name__ = f'_call_{label}'
    return call


PROVIDERS = {
    'anthropic': _call_anthropic,
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
}


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


def build_api_generation_fn(
    provider: Union[str, Callable],
    model: str,
    *,
    system_prompt: Optional[str] = None,
    max_tokens: int = 16,
    cache_dir: Optional[Union[str, Path]] = None,
    image_columns: Sequence[str] = IMAGE_COLUMNS,
    text_columns: Sequence[str] = TEXT_COLUMNS,
) -> Callable:
    """Build a ``generation_fn(stimulus_row, instruction, label_set) -> str``
    that answers a forced-choice behavioral task via a closed-weight API model.

    :param provider: ``'anthropic'`` / ``'openai'``, or a callable adapter
        ``(model, system, user_text, image, max_tokens) -> str`` (used for
        tests and custom providers).
    :param model: provider model id (e.g. ``'claude-opus-4-8'``,
        ``'gpt-4o-2024-08-06'``). Pin a dated snapshot where the provider
        offers one so scores are reproducible.
    :param cache_dir: if set, responses are cached here per stimulus so re-runs
        don't re-bill and the scored result reproduces. Strongly recommended.
    """
    call = provider if callable(provider) else PROVIDERS[provider]
    provider_name = getattr(provider, '__name__', str(provider))
    cache = _ResponseCache(cache_dir) if cache_dir is not None else None

    def _key(prompt, label_set, stim_key, payload_hash) -> str:
        raw = '|'.join([provider_name, model, prompt,
                        ','.join(map(str, label_set)), stim_key, payload_hash])
        return hashlib.sha256(raw.encode('utf-8')).hexdigest()

    def generate(stimulus_row, instruction: str, label_set) -> str:
        label_set = list(label_set)
        prompt = _build_instruction(instruction, label_set)

        img_col = _find_column(stimulus_row, image_columns)
        txt_col = None if img_col else _find_column(stimulus_row, text_columns)
        if img_col is None and txt_col is None:
            raise ValueError(
                f"api_behavioral: stimulus row has no readable image column "
                f"({image_columns}) or text column ({text_columns}).")

        # stimulus identity for the cache key
        try:
            stim_key = str(stimulus_row['stimulus_id'])
        except (KeyError, TypeError):
            stim_key = ''

        if img_col is not None:
            media, b64 = _encode_image(stimulus_row[img_col])
            image = (media, b64)
            user_text = prompt
            payload_hash = hashlib.sha256(b64.encode('utf-8')).hexdigest()[:16]
        else:
            image = None
            text = str(stimulus_row[txt_col])
            user_text = f"{text}\n\n{prompt}"
            payload_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]

        key = _key(prompt, label_set, stim_key, payload_hash)
        response = cache.get(key) if cache is not None else None
        if response is None:
            response = call(model, system_prompt, user_text, image, max_tokens)
            if cache is not None:
                cache.set(key, response)

        parsed = _parse_label(response, label_set)
        # Returning the raw response on a parse miss lets BrainScoreModel emit
        # its standard warning and fall back to label_set[0].
        return parsed if parsed is not None else response

    return generate


# ── embodied / closed-loop game: action_fn ───────────────────────────────
# The video game is agentic, not forced-choice: it drives the model through
# process(EnvironmentStep) -> EnvironmentResponse one tick at a time (see
# brainscore.harnesses.gymnasium_harness.play_gym_episode). This builds an
# action_fn closure that shows the rendered frame + the legal-action menu to an
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


def _parse_action_index(response: str, n_actions: int) -> Optional[int]:
    """Parse an action index from the response. Prefers a trailing
    ``Action: <n>`` line; falls back to the last in-range integer; else None
    (caller picks a random action, the embodied null behavior)."""
    if not response:
        return None
    m = re.search(r'action\s*[:\-]\s*(\d+)', response, re.IGNORECASE)
    if m and 0 <= int(m.group(1)) < n_actions:
        return int(m.group(1))
    for tok in reversed(re.findall(r'\d+', response)):
        if 0 <= int(tok) < n_actions:
            return int(tok)
    return None


def build_api_action_fn(
    provider: Union[str, Callable],
    model: str,
    *,
    obs_mode: str = 'vision',
    max_tokens: int = 512,
    system_prompt: Optional[str] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    frame_resize: int = 384,
    fallback_seed: int = 0,
) -> Callable:
    """Build an ``action_fn(env_step) -> EnvironmentResponse`` that plays a grid
    episode (GridGameEnv or Gymnasium/MiniGrid) via a closed-weight API model.

    :param obs_mode: ``'vision'`` (default) sends the rendered RGB frame as an
        image; ``'ascii'`` sends the text board (``observation['ascii']``) — text
        models can play, and even vision models reason better on the ASCII grid
        (DeepSeek-R1 reading ASCII beat the VLMs in the original game). Either way
        the env still renders frames you can animate to watch.

    Reuses the same provider adapters and response cache as the behavioral
    generation path. On an unparseable reply the action collapses to a random
    legal move (the embodied null), deterministically seeded so cached re-runs
    reproduce. No GPU / weights — laptop-runnable with the provider key in env.

    The returned closure carries a ``.trace`` list: one dict per tick
    ``{'step', 'response', 'action', 'fallback'}`` where ``response`` is the
    model's full raw reply (the reasoning text it wrote before the ``Action:``
    line). Read it to watch *why* the model moved; ``.trace.clear()`` to reset
    between rollouts.
    """
    if obs_mode not in ('vision', 'ascii'):
        raise ValueError(f"obs_mode must be 'vision' or 'ascii'; got {obs_mode!r}")
    from brainscore_core.model_interface import EnvironmentResponse
    import numpy as np

    call = provider if callable(provider) else PROVIDERS[provider]
    provider_name = getattr(provider, '__name__', str(provider))
    cache = _ResponseCache(cache_dir) if cache_dir is not None else None
    rng = np.random.RandomState(fallback_seed)
    trace: list = []

    def act(env_step) -> 'EnvironmentResponse':
        obs = env_step.observation or {}
        instruction = obs.get('instruction') or getattr(env_step, 'instruction', '') or ''
        legal = obs.get('legal_actions') or {}
        n = len(legal) or 7
        menu = ('\n'.join(f"  {i}: {d}" for i, d in sorted(legal.items()))
                if legal else '\n'.join(f"  {i}: action {i}" for i in range(n)))
        tail = ("Reason briefly, then end your reply with a line exactly:\n"
                "Action: <number>")

        if obs_mode == 'ascii':
            board = obs.get('ascii')
            if board is None:
                raise ValueError("build_api_action_fn(obs_mode='ascii'): "
                                 "EnvironmentStep.observation has no 'ascii'.")
            image = None
            prompt = (
                "You control an agent on a grid (text view below).\n"
                f"Mission: {instruction}\n\n{board}\n\n"
                f"Available actions (reply with the NUMBER):\n{menu}\n\n{tail}")
            payload = hashlib.sha256(str(board).encode('utf-8')).hexdigest()[:16]
        else:  # vision
            frame = obs.get('frame')
            if frame is None:
                raise ValueError("build_api_action_fn(obs_mode='vision'): "
                                 "EnvironmentStep.observation has no 'frame'.")
            media, b64 = _encode_frame(frame, frame_resize)
            image = (media, b64)
            prompt = (
                "You control an agent in a grid world. Look at the image and "
                "choose the single best next action.\n"
                f"Mission: {instruction}\n\n"
                f"Available actions (reply with the NUMBER):\n{menu}\n\n{tail}")
            payload = hashlib.sha256(b64.encode('utf-8')).hexdigest()[:16]

        response = None
        key = None
        if cache is not None:
            # Include step_num so a stuck state (unchanged observation) at
            # different ticks does NOT collapse to one cached action — that
            # would freeze the agent in a no-op loop. Same (seed-deterministic)
            # trajectory still reproduces on re-run. Caching a closed loop is
            # inherently fraught; prefer cache_dir=None for fresh per-tick calls.
            step_num = getattr(env_step, 'step_num', '')
            key = hashlib.sha256(
                '|'.join([provider_name, model, prompt, payload,
                          str(step_num)]).encode()
            ).hexdigest()
            response = cache.get(key)
        if response is None:
            response = call(model, system_prompt, prompt, image, max_tokens)
            if cache is not None:
                cache.set(key, response)

        parsed = _parse_action_index(response, n)
        fallback = parsed is None
        idx = int(rng.randint(0, n)) if fallback else parsed
        trace.append({'step': getattr(env_step, 'step_num', len(trace)),
                      'response': response, 'action': idx, 'fallback': fallback})
        return EnvironmentResponse(action=idx)

    act.trace = trace
    return act
