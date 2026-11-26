"""A neuron-level face selection inside the vision part of Kimi K2.5, selected by what a unit writes (offline, one GPU).

The units are the MLP hidden units (4304 per block) of the blocks ``BLOCKS`` (the localizers' block 23 and the two
before it). A unit adds its activation times one fixed vector (its column of ``fc1``) to the block's output. Its
score is (its mean activation on fLoc faces minus on the other categories, odd pictures, mean over tokens) times
(that output vector projected on the face probe's direction at the vision output). The projection skips the later
blocks and the final norm: a direct-path approximation, stated. The probe is the localizers' probe (refitted on the
same odd-picture features with the same settings: deterministic).

Held-out test, as for L1 to L4: the selected units (top ``N_UNITS`` and top ``N_LARGE`` by score) are zeroed in their
block for every token of the test pictures (number = 0 mod 4), and the drop of the probe separation is compared with
``N_RANDOM`` matched random draws of as many units from the same blocks. Overlap with L1 and L2: the share of the
selected units' summed squared output weights that falls on the L1 and L2 channels (chance: 50 / 1152).

    python -m brainscore.models.kimi_k25_local.neuron_localizer <localizer out folder>
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .localizer import BLOCK, SEED, Vision, cohen_d, load_vision

BLOCKS = (BLOCK - 2, BLOCK - 1, BLOCK)
N_UNITS, N_LARGE, N_RANDOM = 50, 200, 20


class Hidden:
    """Forward pre-hooks on ``fc1`` of the chosen blocks: record the token mean of the hidden units, or zero units."""

    def __init__(self, tower):
        self.mlps = {b: tower.encoder.blocks[b].mlp for b in BLOCKS}
        self.record, self.zero = {}, {}
        self.handles = [mlp.fc1.register_forward_pre_hook(self._hook(b)) for b, mlp in self.mlps.items()]

    def _hook(self, block):
        def hook(module, args):
            hidden = args[0]
            if self.zero.get(block):
                hidden = hidden.clone()
                hidden[:, self.zero[block]] = 0
            self.record[block] = hidden.float().mean(0).cpu()
            return (hidden,)
        return hook


@torch.no_grad()
def from_block(vision, start, hidden_state, state):
    rope, cu = state
    h = hidden_state
    for block in vision.tower.encoder.blocks[start:]:
        h = block(h, cu, int(cu[-1]), rope_freqs_cis=rope)
    return vision.tower.encoder.final_layernorm(h).float().mean(0)


def main(folder: Path):
    from PIL import Image
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from brainscore import load_stimulus_set
    start = time.time()
    saved = np.load(folder / 'kimi-k25-face-localizer.npz', allow_pickle=True)
    summary_l = json.load(open(folder / 'kimi-k25-face-localizer.json'))
    vision = Vision(*load_vision(folder / 'weights'))
    hidden = Hidden(vision.tower)
    floc = load_stimulus_set('fLoc')
    assert list(floc['stimulus_id'].values) == [str(s) for s in saved['floc_ids']]
    number, faces = floc['number'].values, floc['category'].values == 'faces'
    odd, test_set = number % 2 == 1, number % 4 == 0
    features = saved['features']
    scaler = StandardScaler().fit(features[odd])
    probe = LogisticRegression(C=1.0, max_iter=5000).fit(scaler.transform(features[odd]), faces[odd])
    direction = probe.coef_[0] / scaler.scale_                    # the probe's direction on the raw final features
    score = lambda f: probe.decision_function(scaler.transform(np.atleast_2d(f)))[0]

    # Activations on the odd pictures; block BLOCKS[0]'s input kept for the test pictures.
    activity = {b: np.zeros((len(floc), 4304), np.float32) for b in BLOCKS}
    kept, states = {}, {}
    first = BLOCKS[0]
    for i, path in enumerate(floc['image_path'].values):
        if not (odd[i] or test_set[i]):
            continue
        pixels, grid, rope, cu = vision.inputs(Image.open(path))
        with torch.no_grad():
            h = vision.tower.patch_embed(pixels, grid)
            for index, block in enumerate(vision.tower.encoder.blocks):
                if index == first and test_set[i]:
                    kept[i], states[i] = h.cpu(), (rope, cu)
                h = block(h, cu, int(cu[-1]), rope_freqs_cis=rope)
        for b in BLOCKS:
            activity[b][i] = hidden.record[b].numpy()
    print('activations', round(time.time() - start), 's', flush=True)

    units = [(b, j) for b in BLOCKS for j in range(4304)]
    contrast = np.concatenate([activity[b][odd & faces].mean(0) - activity[b][odd & ~faces].mean(0) for b in BLOCKS])
    writes = np.concatenate([(hidden.mlps[b].fc1.weight.float().cpu().numpy().T @ direction) for b in BLOCKS])
    unit_score = contrast * writes
    order = np.argsort(-unit_score)

    test = np.flatnonzero(test_set)

    def separation(zeroed):
        hidden.zero = {}
        for k in zeroed:
            b, j = units[k]
            hidden.zero.setdefault(b, []).append(j)
        s = np.array([score(from_block(vision, first, kept[i].to(vision.device), states[i]).cpu().numpy()) for i in test])
        hidden.zero = {}
        f = faces[test]
        return s, (s[f].mean() - s[~f].mean())

    base_scores, base = separation([])
    f = faces[test]
    scale = np.sqrt((base_scores[f].var(ddof=1) + base_scores[~f].var(ddof=1)) / 2)
    drop = lambda zeroed: float((base - separation(zeroed)[1]) / scale)
    rng = np.random.default_rng(SEED)
    result = {'blocks': list(BLOCKS), 'units_per_block': 4304, 'separation_test': float(base / scale),
              'score': 'mean activation faces minus others (odd fLoc) x output vector . probe direction (direct path)'}
    l1, l2 = set(summary_l['L1']['units']), set(summary_l['L2']['units'])
    for n in (N_UNITS, N_LARGE):
        chosen = order[:n]
        random_drops = [drop(rng.choice(len(units), n, replace=False)) for _ in range(N_RANDOM)]
        weights = np.stack([hidden.mlps[units[k][0]].fc1.weight[:, units[k][1]].float().cpu().numpy() for k in chosen])
        mass = (weights ** 2).sum(0)
        result[f'top{n}'] = {
            'units': [[int(units[k][0]), int(units[k][1])] for k in chosen],
            'per_block': {int(b): int(sum(units[k][0] == b for k in chosen)) for b in BLOCKS},
            'drop_test': drop(chosen), 'drop_test_random_mean': float(np.mean(random_drops)),
            'drop_test_random_max': float(np.max(random_drops)), 'drop_test_random': random_drops,
            'write_share_on_L1_channels': float(mass[list(l1)].sum() / mass.sum()),
            'write_share_on_L2_channels': float(mass[list(l2)].sum() / mass.sum()), 'write_share_chance': 50 / 1152,
            'd_contrast_mean': float(cohen_d(np.concatenate([activity[b] for b in BLOCKS], 1)[odd][:, chosen],
                                             faces[odd]).mean())}
        print(n, json.dumps({k: v for k, v in result[f'top{n}'].items() if k not in ('units', 'drop_test_random')}), flush=True)
    result['wall_s'] = time.time() - start
    json.dump(result, open(folder / 'kimi-k25-neuron-localizer.json', 'w'), indent=1)
    print('DONE', round(time.time() - start), 's', flush=True)


if __name__ == '__main__':
    main(Path(sys.argv[1]))
