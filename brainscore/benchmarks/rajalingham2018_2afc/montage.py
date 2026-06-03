"""Compose one 2-AFC trial into a single image — exactly what the model sees.

Layout mirrors the human match-to-sample screen: the briefly-shown SAMPLE image
on top, then the two object choice tokens side by side (LEFT / RIGHT). The model
is asked which option matches the sample. We compose into one image (rather than
passing three separate images) so the trial works for any image-in model — VLM
or otherwise — and so "what the model saw" is a single inspectable artifact.
"""
from PIL import Image, ImageDraw, ImageFont

_CELL = 224           # each object/sample image is rendered at CELL x CELL
_PAD = 18
_LABEL_H = 28
_BG = (255, 255, 255)
_INK = (20, 25, 40)
_LINE = (205, 212, 225)


def _font(size):
    for path in ("/System/Library/Fonts/Helvetica.ttc",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                 "/System/Library/Fonts/Supplemental/Arial.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _load_square(path):
    im = Image.open(path).convert("RGB")
    return im.resize((_CELL, _CELL), Image.LANCZOS)


def _centered(draw, text, cx, y, font, fill=_INK):
    l, t, r, b = draw.textbbox((0, 0), text, font=font)
    draw.text((cx - (r - l) / 2, y), text, font=font, fill=fill)


def compose_montage(sample_path, left_path, right_path,
                    sample_label="SAMPLE", show_choice_labels=True):
    """Return a PIL.Image: sample on top, LEFT/RIGHT tokens below.

    :param sample_path: path to the briefly-shown test image (object on background).
    :param left_path / right_path: paths to the two object choice tokens.
    :param show_choice_labels: draw the LEFT/RIGHT captions under the tokens.
    """
    w = 2 * _CELL + 3 * _PAD
    top_block = _LABEL_H + _CELL
    h = _PAD + top_block + _PAD + _LABEL_H + _CELL + (_LABEL_H if show_choice_labels else 0) + _PAD
    img = Image.new("RGB", (w, h), _BG)
    d = ImageDraw.Draw(img)
    lab = _font(18)
    head = _font(20)

    # --- sample (centered, top) ---
    _centered(d, sample_label, w / 2, _PAD, head)
    sx = (w - _CELL) // 2
    sy = _PAD + _LABEL_H
    img.paste(_load_square(sample_path), (sx, sy))
    d.rectangle([sx, sy, sx + _CELL, sy + _CELL], outline=_LINE, width=2)

    # --- divider + prompt ---
    div_y = sy + _CELL + _PAD // 2
    d.line([(_PAD, div_y), (w - _PAD, div_y)], fill=_LINE, width=1)
    _centered(d, "which option matches the sample?", w / 2, div_y + 4, lab, fill=(110, 120, 140))

    # --- choices (left, right) ---
    cy = div_y + _LABEL_H
    lx, rx = _PAD, _PAD + _CELL + _PAD
    img.paste(_load_square(left_path), (lx, cy))
    img.paste(_load_square(right_path), (rx, cy))
    d.rectangle([lx, cy, lx + _CELL, cy + _CELL], outline=_LINE, width=2)
    d.rectangle([rx, cy, rx + _CELL, cy + _CELL], outline=_LINE, width=2)
    if show_choice_labels:
        _centered(d, "LEFT", lx + _CELL / 2, cy + _CELL + 4, lab)
        _centered(d, "RIGHT", rx + _CELL / 2, cy + _CELL + 4, lab)
    return img
