"""Face localizers over all 27 blocks of the vision part of Kimi K2.5 (offline, one GPU), stated before any result.

W1 (contrast over the whole vision part): for every block-output channel (27 x 1152, the mean over a picture's
tokens) Cohen's d of fLoc faces against the other categories on the odd pictures (select) and on the even ones
(held-out); one ranking over all channels of all blocks by the odd d: top ``N_UNITS`` and top ``N_LARGE``, with their
held-out d and their depth profile (count and mean d per block). Low-level control: each face picture against its
own phase-scrambled twin (``phase_scrambled``, seed ``SEED`` + picture index): d of faces against twins per half.
Set A: top by odd d. Set B: top by odd d among channels with a twin d above ``TWIN_RULE`` on the odd half (the rule
'also above its scrambled twin'), tested on the even half. Random control: ``N_RANDOM`` draws matched to the
selected set's count per block.

W2 (causal over depth): per block, the held-out drop of the final-output face probe (the localizers' probe) when
that block's top ``N_UNITS`` channels by odd d are zeroed, when one random set of as many channels of the block is
zeroed, and when the block's mean-difference direction (faces minus others, odd pictures) is projected out. Also the
whole-vision set A (top ``N_UNITS``, several blocks at once) against the ``N_RANDOM`` block-matched random draws.

    python -m brainscore.models.kimi_k25_local.whole_vision_localizer <localizer out folder>
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .localizer import N_BLOCKS, SEED, Vision, cohen_d, load_vision, phase_scrambled, project_out, zero

N_UNITS, N_LARGE, N_RANDOM = 50, 200, 20
TWIN_RULE = 0.5


@torch.no_grad()
def block_means(vision, image, lesions=None):
    """The mean over tokens of every block's output (27 x 1152) and the final features; ``lesions`` maps a block
    to a function that changes that block's output."""
    pixels, grid, rope, cu = vision.inputs(image)
    h = vision.tower.patch_embed(pixels, grid)
    means = []
    for index, block in enumerate(vision.tower.encoder.blocks):
        h = block(h, cu, int(cu[-1]), rope_freqs_cis=rope)
        if lesions and index in lesions:
            h = lesions[index](h)
        means.append(h.float().mean(0))
    return torch.stack(means).cpu().numpy(), vision.tower.encoder.final_layernorm(h).float().mean(0).cpu().numpy()


def main(folder: Path):
    from PIL import Image
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from brainscore import load_stimulus_set
    start = time.time()
    saved = np.load(folder / 'kimi-k25-face-localizer.npz', allow_pickle=True)
    vision = Vision(*load_vision(folder / 'weights'))
    floc = load_stimulus_set('fLoc')
    assert list(floc['stimulus_id'].values) == [str(s) for s in saved['floc_ids']]
    paths, number = list(floc['image_path'].values), floc['number'].values
    faces = floc['category'].values == 'faces'
    odd, test_set = number % 2 == 1, number % 4 == 0
    features = saved['features']
    scaler = StandardScaler().fit(features[odd])
    probe = LogisticRegression(C=1.0, max_iter=5000).fit(scaler.transform(features[odd]), faces[odd])
    score = lambda f: probe.decision_function(scaler.transform(np.atleast_2d(f)))

    means = np.zeros((len(paths), N_BLOCKS, 1152), np.float32)
    for i, path in enumerate(paths):
        means[i] = block_means(vision, Image.open(path))[0]
    face_index = np.flatnonzero(faces)
    twins = np.zeros((len(face_index), N_BLOCKS, 1152), np.float32)
    for k, i in enumerate(face_index):
        twins[k] = block_means(vision, phase_scrambled(Image.open(paths[i]), SEED + i))[0]
    print('passes', round(time.time() - start), 's', flush=True)

    flat = lambda a: a.reshape(len(a), -1)
    d_odd, d_even = cohen_d(flat(means[odd]), faces[odd]), cohen_d(flat(means[~odd]), faces[~odd])
    face_odd = odd[face_index]

    def twin_d(half):
        both = np.concatenate([flat(means[face_index[half]]), flat(twins[half])])
        return cohen_d(both, np.r_[np.ones(half.sum(), bool), np.zeros(half.sum(), bool)])
    twin_odd, twin_even = twin_d(face_odd), twin_d(~face_odd)
    block_of = np.repeat(np.arange(N_BLOCKS), 1152)
    rng = np.random.default_rng(SEED)

    def describe(chosen):
        counts = np.bincount(block_of[chosen], minlength=N_BLOCKS)
        randoms = []
        for _ in range(N_RANDOM):
            draw = np.concatenate([rng.choice(np.flatnonzero(block_of == b), c, replace=False)
                                   for b, c in enumerate(counts) if c])
            randoms.append(draw)
        return {'units': [[int(block_of[c]), int(c % 1152)] for c in chosen],
                'd_odd_mean': float(d_odd[chosen].mean()), 'd_even_mean': float(d_even[chosen].mean()),
                'twin_d_even_mean': float(twin_even[chosen].mean()),
                'share_above_twin_rule_even': float((twin_even[chosen] > TWIN_RULE).mean()),
                'd_even_random_mean': float(np.mean([d_even[r].mean() for r in randoms])),
                'per_block_count': counts.tolist(),
                'per_block_mean_d_even': [float(d_even[chosen][block_of[chosen] == b].mean()) if c else None
                                          for b, c in enumerate(counts)]}, randoms
    order = np.argsort(-d_odd)
    passing = np.flatnonzero(twin_odd > TWIN_RULE)
    order_b = passing[np.argsort(-d_odd[passing])]
    w1 = {'twin_rule': f'twin d > {TWIN_RULE} on the odd half (selection), reported on the even half',
          'per_block_d_even_max': [float(d_even[block_of == b].max()) for b in range(N_BLOCKS)],
          'per_block_d_even_top50_mean': [float(np.sort(d_even[block_of == b])[-50:].mean()) for b in range(N_BLOCKS)],
          'per_block_twin_d_even_top50_mean': [float(twin_even[block_of == b][np.argsort(-d_odd[block_of == b])[:50]].mean())
                                               for b in range(N_BLOCKS)]}
    random_a50 = None
    for label, ranked in (('A', order), ('B', order_b)):
        for n in (N_UNITS, N_LARGE):
            w1[f'{label}{n}'], randoms = describe(ranked[:n])
            if label == 'A' and n == N_UNITS:
                random_a50 = randoms
    print('W1', round(time.time() - start), 's', flush=True)

    # W2: the held-out drop of the probe separation, block by block.
    test = np.flatnonzero(test_set)
    f = faces[test]

    def separation(lesions):
        s = np.array([score(block_means(vision, Image.open(paths[i]), lesions)[1])[0] for i in test])
        return s

    base = separation(None)
    scale = np.sqrt((base[f].var(ddof=1) + base[~f].var(ddof=1)) / 2)
    gap = lambda s: (s[f].mean() - s[~f].mean()) / scale
    drop = lambda lesions: float(gap(base) - gap(separation(lesions)))
    w2 = {'separation_test': float(gap(base)), 'top50_zero': [], 'random50_zero': [], 'direction_projected': []}
    for b in range(N_BLOCKS):
        local = np.argsort(-d_odd[block_of == b])[:N_UNITS]
        direction = means[odd & faces, b].mean(0) - means[odd & ~faces, b].mean(0)
        w2['top50_zero'].append(drop({b: lambda h, u=local: zero(h, u)}))
        w2['random50_zero'].append(drop({b: lambda h, u=rng.choice(1152, N_UNITS, replace=False): zero(h, u)}))
        w2['direction_projected'].append(drop({b: lambda h, u=direction: project_out(h, u)}))
        print('W2 block', b, w2['top50_zero'][-1], w2['random50_zero'][-1], w2['direction_projected'][-1], flush=True)

    def multi(chosen):
        by_block = {}
        for c in chosen:
            by_block.setdefault(int(block_of[c]), []).append(int(c % 1152))
        return {b: (lambda h, u=u: zero(h, u)) for b, u in by_block.items()}
    w2['wholeA50_zero'] = drop(multi(order[:N_UNITS]))
    w2['wholeA50_random_zero'] = [drop(multi(r)) for r in random_a50]
    result = {'n_blocks': N_BLOCKS, 'channels': 1152, 'n_twins': int(len(face_index)), 'W1': w1, 'W2': w2,
              'random_draws_A50': [[[int(block_of[c]), int(c % 1152)] for c in r] for r in random_a50],
              'wall_s': time.time() - start}
    json.dump(result, open(folder / 'kimi-k25-whole-vision-localizer.json', 'w'), indent=1)
    np.savez(folder / 'kimi-k25-whole-vision-localizer.npz', d_odd=d_odd.reshape(N_BLOCKS, 1152),
             d_even=d_even.reshape(N_BLOCKS, 1152), twin_d_odd=twin_odd.reshape(N_BLOCKS, 1152),
             twin_d_even=twin_even.reshape(N_BLOCKS, 1152))
    print('DONE', round(time.time() - start), 's', flush=True)


if __name__ == '__main__':
    main(Path(sys.argv[1]))
