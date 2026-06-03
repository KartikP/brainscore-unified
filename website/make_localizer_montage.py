"""Build a montage of the localizer stimuli that identify the VWF-selective
population: word images (the 'word-form' category) vs non-word images (scrambled
words + line-drawing objects). Saved into ./assets/localizer_stimuli.png so the
'which neurons, across which layers' panel can show what defined the selection.

Run on EC2 (has the ROAR images + the replication helpers). CPU only.
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.expanduser(
    '~/brain-score-unified/unified/scripts/honarmand_replication'))
from replicate_v2 import scramble_image, object_image  # noqa

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, 'assets')
ROAR = os.environ.get('ROAR_DATA_DIR', '/home/ubuntu/data/roar_yeatman2021')


def main():
    import pandas as pd
    df = pd.read_csv(os.path.join(ROAR, 'stimulus_Yeatman2021.csv'))
    real = df[df['label'] == 1]['filename'].tolist()
    rng = np.random.RandomState(7)
    picks = [Image.open(os.path.join(ROAR, 'stimuli', f)).convert('RGB')
             for f in rng.choice(real, 3, replace=False)]

    cell = (260, 110)
    words = [im.resize(cell) for im in picks]
    nonwords = ([scramble_image(picks[0], seed=1).resize(cell),
                 scramble_image(picks[1], seed=2).resize(cell),
                 object_image(3).resize(cell)])

    pad, lab = 12, 26
    cols = 3
    cw, ch = cell[0] + pad, cell[1] + pad
    W = cols * cw + pad
    H = 2 * (ch + lab) + pad
    canvas = Image.new('RGB', (W, H), (245, 245, 248))
    d = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    def row(imgs, y0, title, color):
        d.text((pad, y0), title, fill=color, font=font)
        for i, im in enumerate(imgs):
            x = pad + i * cw
            canvas.paste(im, (x, y0 + lab))
            d.rectangle([x, y0 + lab, x + cell[0], y0 + lab + cell[1]],
                        outline=(180, 180, 190), width=1)

    row(words, pad, "WORD  (word-form category)", (40, 110, 40))
    row(nonwords, pad + ch + lab, "NON-WORD  (scrambled + objects)", (180, 60, 50))

    out = os.path.join(ASSETS, 'localizer_stimuli.png')
    canvas.save(out)
    print('wrote', out, canvas.size)


if __name__ == '__main__':
    main()
