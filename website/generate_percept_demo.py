"""Generate the PerceptWindow website demo assets.

Shows, for a few aspect-ratio-varied source images, three panels:
  presented  — the source image as a benchmark would hand it over
  tensor     — the raw normalized tensor clipped to [0,1] (false color): what's
               literally in the array, NOT viewable as-is
  percept    — PerceptWindow's reconstruction (de-normalized): exactly what a
               CLIP-preprocessed model ingests, incl. the resize squish + the
               center-crop that discards the edges

The percept is determined entirely by the preprocessing pipeline, not the model
weights — so feeding a CLIP-preprocessed tensor through the real PerceptWindow
forward-pre-hook (over a trivial identity module) captures the exact tensor a
real CLIP would ingest. No model weights, no GPU.
"""
import os

import numpy as np
import torch
from torch import nn
from PIL import Image, ImageDraw

from brainscore.percept_window import PerceptWindow

# CLIP ViT-B/32 preprocessing constants
MEAN = [0.48145466, 0.4578275, 0.40821073]
STD = [0.26862954, 0.26130258, 0.27577711]
SIZE = 224

OUT = os.path.join(os.path.dirname(__file__), 'assets', 'percept')


def make_source(kind, w, h):
    """A legible synthetic image: gridded background, colored edge bands with
    EDGE labels (so the center-crop loss is obvious), and a center subject."""
    img = Image.new('RGB', (w, h), (28, 32, 44))
    d = ImageDraw.Draw(img)
    # grid (makes the resize squish visible)
    for x in range(0, w, 28):
        d.line([(x, 0), (x, h)], fill=(44, 50, 66), width=1)
    for y in range(0, h, 28):
        d.line([(0, y), (w, y)], fill=(44, 50, 66), width=1)
    band = 26
    # colored edge bands + labels at the four edges
    d.rectangle([0, 0, w, band], fill=(214, 95, 70)); d.text((w // 2 - 18, 7), 'TOP', fill='white')
    d.rectangle([0, h - band, w, h], fill=(70, 130, 200)); d.text((w // 2 - 28, h - 20), 'BOTTOM', fill='white')
    d.rectangle([0, 0, band, h], fill=(95, 180, 110))
    d.rectangle([w - band, 0, w, h], fill=(190, 150, 60))
    d.text((4, h // 2 - 6), 'L', fill='white'); d.text((w - 16, h // 2 - 6), 'R', fill='white')
    # center subject: a bright ring + label
    cx, cy, r = w // 2, h // 2, min(w, h) // 5
    d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(245, 245, 245), width=5)
    d.text((cx - 26, cy - 6), kind, fill=(245, 245, 245))
    return img


def clip_preprocess(img):
    """Resize shortest side to 224 (bicubic), center-crop 224, normalize. The
    standard CLIP/ViT path. Returns a (3, 224, 224) float tensor."""
    w, h = img.size
    scale = SIZE / min(w, h)
    img2 = img.resize((round(w * scale), round(h * scale)), Image.BICUBIC)
    nw, nh = img2.size
    left, top = (nw - SIZE) // 2, (nh - SIZE) // 2
    img2 = img2.crop((left, top, left + SIZE, top + SIZE))
    arr = np.asarray(img2).astype(np.float32) / 255.0          # (H, W, 3)
    arr = (arr - np.array(MEAN)) / np.array(STD)               # normalize
    return torch.tensor(np.transpose(arr, (2, 0, 1)), dtype=torch.float32)


class Identity(nn.Module):
    def forward(self, x):
        return x


def reconstruct(src, prefix, model):
    """Save <prefix>_tensor.png (raw normalized, clipped — false color) and
    <prefix>_percept.png (PerceptWindow reconstruction) for a source PIL image."""
    tensor = clip_preprocess(src)
    with PerceptWindow(model, modality='vision', denorm=(MEAN, STD)) as eye:
        model(tensor[None])                                     # REAL hook capture
    percept = eye.reconstruct()[0]['data'][0]                   # (224, 224, 3) uint8
    Image.fromarray(percept).save(os.path.join(OUT, f'{prefix}_percept.png'))
    raw = np.clip(tensor.numpy(), 0, 1)
    raw = (np.transpose(raw, (1, 2, 0)) * 255).astype(np.uint8)
    Image.fromarray(raw).save(os.path.join(OUT, f'{prefix}_tensor.png'))


def main():
    os.makedirs(OUT, exist_ok=True)
    model = Identity()

    # Tab 1: synthetic aspect-ratio demo (crop axis depends on aspect)
    specs = [('WIDE', 640, 360), ('TALL', 360, 640), ('SQUARE', 360, 360)]
    for i, (kind, w, h) in enumerate(specs):
        src = make_source(kind, w, h)
        src.save(os.path.join(OUT, f'{i}_presented.png'))
        reconstruct(src, str(i), model)
        print(f'  {kind}: presented {w}x{h} -> ingested {SIZE}x{SIZE} '
              f'(shortest-side resize + center-crop)', flush=True)

    # Tab 2: the real Rajalingham 2-AFC montages, through CLIP's front-end
    assets = os.path.join(os.path.dirname(__file__), 'assets')
    for j, name in enumerate(['raj2afc_montage_0.png', 'raj2afc_montage_2.png']):
        path = os.path.join(assets, name)
        if not os.path.exists(path):
            print(f'  (skip {name}: not found)', flush=True)
            continue
        src = Image.open(path).convert('RGB')
        reconstruct(src, f'raj{j}', model)
        print(f'  {name}: presented {src.size[0]}x{src.size[1]} -> ingested '
              f'{SIZE}x{SIZE} (CLIP front-end: resize + center-crop)', flush=True)

    # Tab 3: multimodal — a dual-tower (CLIP) model ingests an IMAGE + a CAPTION.
    # PerceptWindow hooks BOTH towers: pixel tensor -> reconstructed image, and
    # token ids -> detokenized string. One mechanism, two modalities.
    multimodal(assets, model)
    print(f'wrote demo triplets to {OUT}', flush=True)


def multimodal(assets, vis_model):
    caption = 'a photo of a wrench on a mountainside'
    # vision tower: the LEFT objectome token from montage 0 IS a wrench on a
    # mountainside, so the caption genuinely describes the image.
    montage = os.path.join(assets, 'raj2afc_montage_0.png')
    if os.path.exists(montage):
        crop = Image.open(montage).convert('RGB').crop((12, 310, 248, 528))
        crop.save(os.path.join(OUT, 'mm_vision_presented.png'))
        reconstruct(crop, 'mm_vision', vis_model)
        print(f'  multimodal vision: {crop.size} -> {SIZE}x{SIZE}', flush=True)
    # text tower: tokenize the caption with CLIP's real tokenizer, then capture
    # the token-id tensor through the SAME PerceptWindow hook and detokenize.
    try:
        from transformers import CLIPTokenizer
        tok = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')
        ids = tok(caption, padding='max_length', max_length=77,
                  truncation=True)['input_ids']

        class IdEmbed(nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = nn.Embedding(49408, 8)

            def forward(self, x):
                return self.emb(x)

        text_model = IdEmbed()
        with PerceptWindow(text_model, modality='text') as eye:
            text_model(torch.tensor(ids)[None])
        decoded = eye.reconstruct(tokenizer=tok)[0]['data'][0]
        nonpad = [i for i in ids if i != 49407]
        print('  multimodal text:', flush=True)
        print(f'    caption : {caption!r}', flush=True)
        print(f'    ids[:12]: {ids[:12]}  (77-token context)', flush=True)
        print(f'    decoded : {decoded!r}', flush=True)
        print(f'    note    : non-pad ids = {nonpad}', flush=True)
    except Exception as e:
        print(f'  (multimodal text skipped: {type(e).__name__}: {e})', flush=True)


if __name__ == '__main__':
    main()
