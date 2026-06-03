"""Reconstruct the Rajalingham2018 2-AFC trial grid + select per-object tokens.

The packaged behavioral assembly (``Rajalingham2018.public``) stores, for every
human trial, ``(image_id, sample_obj, dist_obj, choice)`` with ``choice`` always
in ``{sample_obj, dist_obj}`` — i.e. genuine binary 2-AFC responses. What it does
NOT store is the canonical "choice token" image shown for each object on the
response screen. This script:

  1. Aggregates the human trials into the i2 condition grid
     ``(image_id, sample_obj, dist_obj) -> human n, P(choose sample)``.
  2. Subsamples ``N_PER_OBJECT`` well-sampled test images per object (the classic
     ~240-image i2 subset), held out from the token pool.
  3. Picks one canonical token image per object from the *held-out* pool (the
     most-recognizable exemplar = highest mean human target-accuracy), so a token
     is never also a test image (no leakage).
  4. Writes ``trials.csv`` (one row per (test image x distractor) condition) and
     ``tokens.csv`` (object -> token image_id). Both reference image_ids only;
     the scoring driver resolves pixels via ``load_stimulus_set('objectome.public')``.

Run locally (no GPU): ``python build_trials.py``. Outputs land next to this file.
"""
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from brainscore_vision import load_dataset, load_stimulus_set

HERE = os.path.dirname(os.path.abspath(__file__))
N_PER_OBJECT = 10          # test images per object -> ~240 image i2 subset
MIN_HUMAN_TRIALS = 4       # per (image, distractor) condition, for a stable human estimate


def main():
    asm = load_dataset('Rajalingham2018.public')
    ss = load_stimulus_set('objectome.public')

    df = pd.DataFrame({
        'image_id': asm['image_id'].values,
        'sample_obj': asm['sample_obj'].values,
        'dist_obj': asm['dist_obj'].values,
        'choice': asm.values,
    })
    df['chose_sample'] = (df['choice'] == df['sample_obj']).astype(float)

    # ---- condition grid: (image, sample, dist) -> human n, P(choose sample) ----
    cond = (df.groupby(['image_id', 'sample_obj', 'dist_obj'])
              .agg(human_n=('chose_sample', 'size'),
                   human_p_sample=('chose_sample', 'mean'))
              .reset_index())
    cond = cond[cond['human_n'] >= MIN_HUMAN_TRIALS]

    # ---- per-image difficulty (mean target-accuracy across its conditions) ----
    img_stats = (cond.groupby(['image_id', 'sample_obj'])
                     .agg(img_n=('human_n', 'sum'),
                          img_acc=('human_p_sample', 'mean'),
                          n_dist=('dist_obj', 'nunique'))
                     .reset_index())

    objects = sorted(df['sample_obj'].unique())

    # ---- token per object: held-out, most-recognizable exemplar ----
    # rank each object's images by accuracy; reserve the top one as the token,
    # then pick the test subset from the REMAINING images.
    tokens = {}
    test_images = {}
    for obj in objects:
        obj_imgs = img_stats[img_stats['sample_obj'] == obj].copy()
        # need enough coverage to be a usable test image
        obj_imgs = obj_imgs[obj_imgs['n_dist'] >= 5]
        obj_imgs = obj_imgs.sort_values(['img_acc', 'img_n'], ascending=False)
        if len(obj_imgs) < N_PER_OBJECT + 1:
            raise RuntimeError(f"object {obj} has too few usable images ({len(obj_imgs)})")
        token_id = obj_imgs.iloc[0]['image_id']          # clearest exemplar = token
        tokens[obj] = token_id
        # test subset: spread across the difficulty range of the remaining images
        remaining = obj_imgs.iloc[1:].reset_index(drop=True)
        idx = np.linspace(0, len(remaining) - 1, N_PER_OBJECT).round().astype(int)
        test_images[obj] = list(remaining.loc[idx, 'image_id'])

    selected = set(img for imgs in test_images.values() for img in imgs)

    # ---- trials.csv: conditions for the selected test images ----
    trials = cond[cond['image_id'].isin(selected)].copy()
    trials['token_sample_id'] = trials['sample_obj'].map(tokens)
    trials['token_dist_id'] = trials['dist_obj'].map(tokens)
    # sanity: a token must never equal the test image
    assert not (trials['token_sample_id'] == trials['image_id']).any()

    trials = trials[['image_id', 'sample_obj', 'dist_obj',
                     'token_sample_id', 'token_dist_id',
                     'human_n', 'human_p_sample']]
    trials.to_csv(os.path.join(HERE, 'trials.csv'), index=False)

    tok_df = pd.DataFrame({'object': objects,
                           'token_image_id': [tokens[o] for o in objects]})
    tok_df.to_csv(os.path.join(HERE, 'tokens.csv'), index=False)

    # ---- verify every referenced image resolves to a real file ----
    all_ids = set(trials['image_id']) | set(tokens.values())
    missing = [i for i in all_ids if i not in ss.stimulus_paths]
    assert not missing, f"{len(missing)} image_ids not in stimulus set"

    print(f"objects                : {len(objects)}")
    print(f"test images selected    : {len(selected)} ({N_PER_OBJECT}/object)")
    print(f"tokens (held out)       : {len(tokens)}")
    print(f"2-AFC conditions (trials): {len(trials)}")
    print(f"distractors per image    : {trials.groupby('image_id').size().median():.0f} (median)")
    print(f"human reps per condition : {trials['human_n'].median():.0f} (median)")
    print(f"wrote {os.path.join(HERE, 'trials.csv')}")
    print(f"wrote {os.path.join(HERE, 'tokens.csv')}")


if __name__ == '__main__':
    main()
