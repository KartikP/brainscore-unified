"""Vision-language pointer policies for the nodekit browser harness.

``build_click_policy(generate)`` turns any ``generate(image, prompt) -> str``
into a pointer policy ``policy(observation, history) -> action`` for
``harnesses.nodekit_browser``: the model sees the board screenshot, names a
pixel, and the policy clicks there. The click is a jump with no path, the same
as computer-use models act; its timing is whatever ``dt_ms`` says.

Two ``generate`` back ends: :func:`api_generate` (any provider in
``api_behavioral.PROVIDERS``, with the response cache) and
:func:`local_vlm_generate` (HuggingFace weights). As in ``local_vlm_policy``,
the back ends need keys or weights and are not unit-tested; the parser and the
policy are.
"""
import hashlib
import io
import json
import re
from typing import Callable, Optional, Tuple

import numpy as np

from brainscore.harnesses.nodekit_browser import BOARD_SIZE, click

CLICK_PROMPT = (
    "{instruction}\n\nThe image is the task screen as it is now, {size} x {size} "
    "pixels, with (0, 0) at the top-left corner. Where should the mouse click on "
    "this screen? Reply with a final line exactly like: Click: (x, y)")


def parse_click(text: str, size: int = BOARD_SIZE) -> Optional[Tuple[float, float]]:
    """Pixel (x, y) from the last ``Click: (x, y)``, else the last in-range number pair.

    A pair opened with ``(`` but never closed, as in a reply cut off at the
    token limit ("... at (512, 51"), is ignored rather than read as a click.
    """
    if not text:
        return None
    num = r'(-?\d+(?:\.\d+)?)'
    pair = rf'(\(?)\s*{num}\s*,\s*{num}\s*(\)?)'
    pairs = (re.findall(rf'click\s*:\s*{pair}', text, re.I)
             or re.findall(pair, text))[::-1]
    for opened, x, y, closed in pairs:
        x, y = float(x), float(y)
        if (not opened or closed) and 0 <= x <= size and 0 <= y <= size:
            return x, y
    return None


def build_click_policy(generate: Callable, *, dt_ms: float = 0.0, seed: int = 0,
                       prompt: str = CLICK_PROMPT) -> Callable:
    """Pointer policy from a ``generate(image, prompt) -> str`` function.

    An unparseable reply becomes a random click (the null), seeded, and is
    counted in ``policy.stats['parse_miss']``. ``policy.log`` keeps each reply.
    """
    rng = np.random.RandomState(seed)
    stats = {'calls': 0, 'parse_miss': 0}
    log = []

    def policy(observation, history):
        image = observation.get('image')
        if image is None:  # terminal step: nothing on screen
            return click(0, 0)
        size = image.shape[0]
        stats['calls'] += 1
        reply = generate(image, prompt.format(instruction=observation.get('instruction', ''),
                                              size=size))
        point = parse_click(reply, size)
        if point is None:
            stats['parse_miss'] += 1
            point = tuple(rng.uniform(0, size, 2))
        log.append({'reply': reply, 'click_px': point})
        scale = BOARD_SIZE / size
        return click(point[0] * scale - BOARD_SIZE / 2, BOARD_SIZE / 2 - point[1] * scale, dt_ms=dt_ms)

    policy.stats, policy.log = stats, log
    return policy


def _png_b64(image: np.ndarray, resize: Optional[int]) -> str:
    import base64
    from PIL import Image
    img = Image.fromarray(np.asarray(image, dtype=np.uint8))
    if resize:
        img = img.resize((resize, resize), Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.standard_b64encode(buf.getvalue()).decode('ascii')


def _rescaled(reply: str, resize: int, size: int) -> str:
    """Append the reply's click mapped from ``resize`` pixels back to ``size``."""
    point = parse_click(reply, resize)
    if point is None:
        return reply
    s = size / resize
    return f'{reply}\nClick: ({point[0] * s:.1f}, {point[1] * s:.1f})'


def api_generate(provider, model: str, *, cache_dir=None, max_tokens: int = 300,
                 resize: Optional[int] = None) -> Callable:
    """``generate`` backed by an API provider (see ``api_behavioral.PROVIDERS``).

    ``resize`` sends a smaller square image; the prompt states the size sent,
    so returned pixels stay consistent. Replies are cached by image and prompt.
    """
    from brainscore.model_helpers.api_behavioral import PROVIDERS, _ResponseCache
    call = provider if callable(provider) else PROVIDERS[provider]
    cache = _ResponseCache(cache_dir) if cache_dir is not None else None

    def generate(image, prompt):
        b64 = _png_b64(image, resize)
        if resize:
            prompt = prompt.replace(f'{image.shape[0]} x {image.shape[0]}', f'{resize} x {resize}')
        key = hashlib.sha256(json.dumps([str(provider), model, prompt, b64, max_tokens]).encode()).hexdigest()
        reply = cache.get(key) if cache else None
        if reply is None:
            reply = call(model, None, prompt, ('image/png', b64), max_tokens)
            if cache:
                cache.set(key, reply)
        return _rescaled(reply, resize, image.shape[0]) if resize else reply

    return generate


def local_vlm_generate(model_id: str, *, max_new_tokens: int = 256,
                       resize: Optional[int] = 512) -> Callable:
    """``generate`` backed by a local HuggingFace image-text-to-text model."""
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    processor = AutoProcessor.from_pretrained(model_id)
    dtype = torch.bfloat16 if device == 'cuda' else torch.float32
    model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=dtype).to(device).eval()

    def generate(image, prompt):
        img = Image.fromarray(np.asarray(image, dtype=np.uint8))
        size = image.shape[0]
        if resize:
            img = img.resize((resize, resize), Image.BILINEAR)
            prompt = prompt.replace(f'{size} x {size}', f'{resize} x {resize}')
        messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': prompt}]}]
        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=text, images=[img], return_tensors='pt').to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        reply = processor.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return _rescaled(reply, resize, size) if resize else reply

    return generate
