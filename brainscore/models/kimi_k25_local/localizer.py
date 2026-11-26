"""Face localizers in the vision part of Kimi K2.5 (MoonViT), run in our own process on one GPU.

The vision part is built from the authors' remote code and the released weights (``moonshotai/Kimi-K2.5`` at
``REVISION``; the vision weights are in BF16, shard 64). Each picture goes through the authors' image processor,
one picture per call. A unit is one channel of a block's output, averaged over the picture's tokens.

Four selections, each set before any lesion run of the task, each with a held-out test and matched random
controls (``N_RANDOM`` draws):

- L1 (contrast): the top ``N_UNITS`` channels of ``BLOCK`` by Cohen's d, faces against the other fLoc
  categories, on the pictures with an odd number; held-out: d on the even ones.
- L2 (causal, ablation): a readout fixed first: a linear faces-against-others probe (logistic regression on
  standardized features) on the vision part's final output (the mean over tokens of the input of the
  projector), fitted on the odd pictures. Each channel of ``BLOCK`` is zeroed in turn on the ranking pictures
  (number = 2 mod 4) and the drop of the probe's face-against-other separation is measured; top ``N_UNITS``.
- L3 (causal, patching): for each face picture of the ranking set and its phase-scrambled version (made here:
  amplitude spectrum kept, phase random, ``SEED``), one channel of ``BLOCK`` is copied from the face run into
  the scrambled run; the share of the probe's face response that comes back is measured; top ``N_UNITS``.
- L4 (a direction, not units): faces minus others, the mean of the picture means at ``BLOCK`` on the odd
  pictures; its lesion projects it out of every token.

Held-out test of every selection (pictures with number = 0 mod 4): the drop of the probe separation when the
selection is inactivated (zeroed channels, or the direction projected out), against ``N_RANDOM`` matched random
draws (random channels of the same count; random directions); and, for channel selections, the mean d there.
The separation is the probe score's mean difference, faces minus others, in units of the intact pooled sd.

L2 to L4 are chosen by an effect on a face readout of the vision part (our linear probe, not the model's own
answer) and are tested on a different thing (the arrangement in the task), so that test is not circular.

    python -m brainscore.models.kimi_k25_local.localizer <out folder>
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPOSITORY = 'moonshotai/Kimi-K2.5'
REVISION = '94b44e57fb4797fde1c1872bed70aef3c93a07f6'
CODE_FILES = ('config.json', 'configuration_kimi_k25.py', 'modeling_kimi_k25.py', 'configuration_deepseek.py',
              'modeling_deepseek.py', 'kimi_k25_vision_processing.py', 'kimi_k25_processor.py', 'media_utils.py',
              'preprocessor_config.json')
VISION_SHARD = 'model-00064-of-000064.safetensors'   # every vision_tower weight (the index at REVISION)
N_BLOCKS = 27
BLOCK = 23            # the study's depth rule: relative depth 0.90 of 27 blocks, rounded half up -> 23
N_UNITS = 50
N_RANDOM = 20
SEED = 0


def sdpa_packed(q, k, v, q_cu_seqlens=None, k_cu_seqlens=None, **kwargs):
    """Attention within each picture of a packed batch (tokens x heads x dim), by PyTorch's SDPA."""
    out, bounds = [], q_cu_seqlens.tolist()
    for start, end in zip(bounds[:-1], bounds[1:]):
        heads = [x[start:end].transpose(0, 1)[None] for x in (q, k, v)]
        out.append(torch.nn.functional.scaled_dot_product_attention(*heads)[0].transpose(0, 1))
    return torch.cat(out).flatten(start_dim=-2)


def load_vision(folder: Path, device: str = 'cuda'):
    """The vision part (BF16, eval) and the image processor from the authors' code and weights at REVISION."""
    import sys as _sys
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoImageProcessor
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    for name in CODE_FILES + (VISION_SHARD,):
        hf_hub_download(REPOSITORY, name, revision=REVISION, local_dir=folder)
    config = AutoConfig.from_pretrained(folder, trust_remote_code=True)
    tower_class = get_class_from_dynamic_module('modeling_kimi_k25.MoonViT3dPretrainedModel', str(folder))
    config_class = get_class_from_dynamic_module('modeling_kimi_k25.VisionTowerConfig', str(folder))
    module = _sys.modules[tower_class.__module__]
    module.VL_VISION_ATTENTION_FUNCTIONS['sdpa'] = sdpa_packed
    vision_config = config.vision_config
    vision_config._attn_implementation = 'sdpa'
    tower = tower_class(config_class(vision_config))
    weights = {name[len('vision_tower.'):]: value for name, value in load_file(folder / VISION_SHARD).items()
               if name.startswith('vision_tower.')}
    tower.load_state_dict(weights, strict=True)
    for block in tower.encoder.blocks:
        block.attn_implementation = 'sdpa'
    assert len(tower.encoder.blocks) == N_BLOCKS
    processor = AutoImageProcessor.from_pretrained(folder, trust_remote_code=True)
    return tower.to(device=device, dtype=torch.bfloat16).eval(), processor


class Vision:
    """The vision part split at ``BLOCK``: ``to_block`` gives that block's output, ``rest`` the probe features
    (the mean over tokens of the final output) from a (possibly changed) block output."""

    def __init__(self, tower, processor, device='cuda'):
        self.tower, self.processor, self.device = tower, processor, device

    def inputs(self, image):
        batch = self.processor.preprocess([{'type': 'image', 'image': image}], return_tensors='pt')
        grid = batch['grid_thws'].to(self.device)
        encoder = self.tower.encoder
        rope = encoder.rope_2d.get_freqs_cis(grid_thws=grid, device=self.device)
        lengths = torch.cat((torch.zeros(1, dtype=grid.dtype, device=self.device), grid.prod(-1)))
        return batch['pixel_values'].to(self.device, torch.bfloat16), grid, rope, lengths.cumsum(0, dtype=torch.int32)

    @torch.no_grad()
    def to_block(self, image):
        """(block output tokens x channels, final features, state for ``rest``)."""
        pixels, grid, rope, cu = self.inputs(image)
        hidden = self.tower.patch_embed(pixels, grid)
        for index, block in enumerate(self.tower.encoder.blocks):
            hidden = block(hidden, cu, int(cu[-1]), rope_freqs_cis=rope)
            if index == BLOCK:
                at_block = hidden
        state = (rope, cu)
        return at_block, self.tower.encoder.final_layernorm(hidden).float().mean(0), state

    @torch.no_grad()
    def rest(self, at_block, state):
        rope, cu = state
        hidden = at_block
        for block in self.tower.encoder.blocks[BLOCK + 1:]:
            hidden = block(hidden, cu, int(cu[-1]), rope_freqs_cis=rope)
        return self.tower.encoder.final_layernorm(hidden).float().mean(0)


def cohen_d(values, positive):
    pos, neg = values[positive], values[~positive]
    pooled = np.sqrt((pos.std(0, ddof=1) ** 2 + neg.std(0, ddof=1) ** 2) / 2)
    return np.divide(pos.mean(0) - neg.mean(0), pooled, out=np.zeros(values.shape[1]), where=pooled > 0)


def zero(at_block, channels):
    changed = at_block.clone()
    changed[:, torch.as_tensor(channels, device=at_block.device)] = 0
    return changed


def project_out(at_block, direction):
    u = torch.as_tensor(direction, dtype=torch.float32, device=at_block.device)
    u = u / u.norm()
    values = at_block.float()
    return (values - (values @ u)[:, None] * u).to(at_block.dtype)


def phase_scrambled(image, seed):
    """The picture with its amplitude spectrum and a random phase (per channel the same phase)."""
    from PIL import Image
    pixels = np.asarray(image.convert('RGB'), dtype=np.float64)
    noise = np.fft.fft2(np.random.default_rng(seed).random(pixels.shape[:2]))
    phase = np.angle(noise)
    out = np.stack([np.real(np.fft.ifft2(np.abs(np.fft.fft2(pixels[..., c])) * np.exp(1j * phase)))
                    for c in range(3)], -1)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def main(out: Path):
    from PIL import Image
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from brainscore import load_stimulus_set
    start = time.time()
    out.mkdir(parents=True, exist_ok=True)
    vision = Vision(*load_vision(out / 'weights'))
    floc = load_stimulus_set('fLoc')
    number, faces = floc['number'].values, (floc['category'].values == 'faces')
    paths = list(floc['image_path'].values)
    odd, rank_set, test_set = number % 2 == 1, number % 4 == 2, number % 4 == 0
    keep = rank_set | test_set                  # pictures whose block output is kept for the causal parts
    means, features, kept, states = np.zeros((len(paths), 1152)), np.zeros((len(paths), 1152)), {}, {}
    for i, path in enumerate(paths):
        at_block, feature, state = vision.to_block(Image.open(path))
        means[i], features[i] = at_block.float().mean(0).cpu().numpy(), feature.cpu().numpy()
        if keep[i]:
            kept[i], states[i] = at_block.cpu(), state
    tokens = int(next(iter(kept.values())).shape[0])
    print('fLoc pass', round(time.time() - start), 's; tokens per picture', tokens, flush=True)

    # The probe (readout), fitted once on the odd pictures.
    scaler = StandardScaler().fit(features[odd])
    probe = LogisticRegression(C=1.0, max_iter=5000).fit(scaler.transform(features[odd]), faces[odd])
    score = lambda f: probe.decision_function(scaler.transform(np.atleast_2d(f)))

    def separation(indices, change=None, scale=None):
        """Probe separation on ``indices``; ``change`` maps a block output to a changed one."""
        s = np.array([score((vision.rest(kept[i].to(vision.device) if change is None else
                                         change(kept[i].to(vision.device)), states[i])).cpu().numpy())[0]
                      for i in indices])
        f = faces[indices]
        pooled = scale if scale is not None else np.sqrt((s[f].var(ddof=1) + s[~f].var(ddof=1)) / 2)
        return (s[f].mean() - s[~f].mean()) / pooled, pooled

    # Ranking pictures for L2: all faces of the rank set and 9 of each other subcategory (fixed by number).
    sub = floc['subcategory'].values
    ranking = np.array([i for i in np.flatnonzero(rank_set)
                        if faces[i] or number[i] in sorted(number[rank_set & (sub == sub[i])])[:9]])
    test = np.flatnonzero(test_set)
    base_rank, scale_rank = separation(ranking)
    base_test, scale_test = separation(test)
    probe_test_accuracy = float((score(features[test]) > 0).astype(bool).__eq__(faces[test]).mean())
    print('probe: separation rank', round(base_rank, 3), 'test', round(base_test, 3), 'accuracy test',
          round(probe_test_accuracy, 3), flush=True)

    # L1
    d_select, d_test = cohen_d(means[odd], faces[odd]), cohen_d(means[~odd], faces[~odd])
    l1 = np.argsort(-d_select)[:N_UNITS]
    # L2
    drop = np.array([base_rank - separation(ranking, lambda h, c=c: zero(h, [c]), scale_rank)[0]
                     for c in range(1152)])
    l2 = np.argsort(-drop)[:N_UNITS]
    print('L2 ranked', round(time.time() - start), 's', flush=True)
    # L3
    face_rank = [i for i in ranking if faces[i]]
    scrambled = {}
    for i in face_rank:
        at_block, _, state = vision.to_block(phase_scrambled(Image.open(paths[i]), SEED + i))
        scrambled[i] = at_block
    s_face = np.array([score(vision.rest(kept[i].to(vision.device), states[i]).cpu().numpy())[0] for i in face_rank])
    s_scr = np.array([score(vision.rest(scrambled[i], states[i]).cpu().numpy())[0] for i in face_rank])
    restored = np.zeros(1152)
    for c in range(1152):
        patched = []
        for k, i in enumerate(face_rank):
            h = scrambled[i].clone()
            h[:, c] = kept[i][:, c].to(vision.device)
            patched.append(score(vision.rest(h, states[i]).cpu().numpy())[0])
        restored[c] = (np.mean(patched) - s_scr.mean()) / (s_face.mean() - s_scr.mean())
    l3 = np.argsort(-restored)[:N_UNITS]
    print('L3 ranked', round(time.time() - start), 's', flush=True)
    # L4
    direction = means[odd & faces].mean(0) - means[odd & ~faces].mean(0)

    # Held-out tests, with matched random controls.
    rng = np.random.default_rng(SEED)
    random_sets = [rng.choice(1152, N_UNITS, replace=False) for _ in range(N_RANDOM)]
    random_directions = [rng.standard_normal(1152) for _ in range(N_RANDOM)]
    test_drop = lambda change: float(base_test - separation(test, change, scale_test)[0])
    random_channel_drops = [test_drop(lambda h, r=r: zero(h, r)) for r in random_sets]
    summary = {'repository': REPOSITORY, 'revision': REVISION, 'block': BLOCK, 'n_blocks': N_BLOCKS,
               'channels': 1152, 'tokens_per_floc_picture': tokens, 'n_units': N_UNITS, 'n_random': N_RANDOM,
               'seed': SEED, 'probe': {'fit': 'odd', 'test_accuracy': probe_test_accuracy,
                                       'separation_rank': float(base_rank), 'separation_test': float(base_test)},
               'splits': {'select_L1_L4': 'odd', 'rank_L2_L3': '2 mod 4 (%d pictures)' % len(ranking),
                          'test_L2_L3_L4': '0 mod 4 (%d pictures)' % len(test), 'test_L1_d': 'even'},
               'random_channel_drop_test': random_channel_drops}
    for name, units in (('L1', l1), ('L2', l2), ('L3', l3)):
        summary[name] = {'units': units.tolist(), 'd_select_mean': float(d_select[units].mean()),
                         'd_even_mean': float(d_test[units].mean()),
                         'd_even_random_mean': float(np.mean([d_test[r].mean() for r in random_sets])),
                         'drop_test': test_drop(lambda h, u=units: zero(h, u)),
                         'drop_test_random_mean': float(np.mean(random_channel_drops)),
                         'drop_test_random_max': float(np.max(random_channel_drops))}
    random_direction_drops = [test_drop(lambda h, u=u: project_out(h, u)) for u in random_directions]
    summary['L4'] = {'norm': float(np.linalg.norm(direction)), 'drop_test': test_drop(lambda h: project_out(h, direction)),
                     'drop_test_random': random_direction_drops}
    summary['overlap'] = {f'{a}-{b}': len(set(summary[a]['units']) & set(summary[b]['units']))
                          for a, b in (('L1', 'L2'), ('L1', 'L3'), ('L2', 'L3'))}
    summary['overlap']['all'] = len(set(l1) & set(l2) & set(l3))

    # hvm cross-check of L1 (its faces are renders of synthetic heads).
    hvm = load_stimulus_set('hvm-public')
    hvm_means = np.array([vision.to_block(Image.open(hvm.get_stimulus(s)))[0].float().mean(0).cpu().numpy()
                          for s in hvm['stimulus_id']])
    d_hvm = cohen_d(hvm_means, hvm['category_name'].values == 'Faces')
    summary['L1']['d_hvm_mean'] = float(d_hvm[l1].mean())
    summary['L1']['d_hvm_random_mean'] = float(np.mean([d_hvm[r].mean() for r in random_sets]))
    summary['wall_s'] = time.time() - start
    np.savez(out / 'kimi-k25-face-localizer.npz', means=means, features=features, d_select=d_select,
             d_test=d_test, drop=drop, restored=restored, direction=direction, d_hvm=d_hvm,
             l1=l1, l2=l2, l3=l3, random_sets=np.array(random_sets), floc_ids=np.array(floc['stimulus_id'].values))
    json.dump(summary, open(out / 'kimi-k25-face-localizer.json', 'w'), indent=1)
    print('DONE', json.dumps({k: v for k, v in summary.items() if k != 'random_channel_drop_test'}), flush=True)


if __name__ == '__main__':
    main(Path(sys.argv[1]))
