"""Faithful 2-AFC Rajalingham2018 — score any per-trial choice stream against humans.

The original ``Rajalingham2018-i2n`` reconstructs the binary task post-hoc from a
global 24-way classifier softmax. This variant scores models that make the *actual*
2-AFC choice (see the sample, pick between two object tokens). The scoring is
deliberately symmetric: model choices and human choices BOTH flow through the
metric's ``build_response_matrix_from_responses -> dprimes -> correlate`` path, and
the result is normalized by the human split-half ceiling — the identical computation
the stock benchmark uses for its noise ceiling. So a model is scored exactly like a
human subject; the only thing that varies across paths is *how the choice is made*:

  - generation  : a VLM answers LEFT/RIGHT on the composed montage
  - similarity  : nearest object token to the sample in feature space
  - readout     : argmax of a fitted classifier's {P(sample), P(dist)}
  - random      : coin flip (the chance null)

This module is model-agnostic: it turns a choices DataFrame into a score. The paths
themselves live in ``score_2afc.py`` (they need model weights / a GPU).
"""
import os
import numpy as np
import pandas as pd
import xarray as xr

from brainscore_core.supported_data_standards.brainio.assemblies import BehavioralAssembly
from brainscore_vision import load_dataset, load_metric

HERE = os.path.dirname(os.path.abspath(__file__))
TRIALS_CSV = os.path.join(HERE, 'trials.csv')
TOKENS_CSV = os.path.join(HERE, 'tokens.csv')


def load_trials():
    return pd.read_csv(TRIALS_CSV)


def load_tokens():
    df = pd.read_csv(TOKENS_CSV)
    return dict(zip(df['object'], df['token_image_id']))


def choices_to_assembly(df):
    """DataFrame[image_id, sample_obj, dist_obj, choice] -> BehavioralAssembly.

    Matches the coord structure the i1i2 metric consumes: a 1-D presentation
    array whose *values* are the chosen object, with stimulus_id / sample_obj /
    dist_obj / truth coords (truth == sample_obj).
    """
    df = df.reset_index(drop=True)
    coords = {
        'stimulus_id': ('presentation', df['image_id'].values),
        'sample_obj': ('presentation', df['sample_obj'].values),
        'dist_obj': ('presentation', df['dist_obj'].values),
        'truth': ('presentation', df['sample_obj'].values),
    }
    return BehavioralAssembly(df['choice'].values, coords=coords, dims=['presentation'])


def human_trials_subset():
    """Raw human trials restricted to exactly the (image, distractor) conditions
    in trials.csv, as a DataFrame of one row per human response."""
    trials = load_trials()
    conditions = set(zip(trials['image_id'], trials['dist_obj']))
    asm = load_dataset('Rajalingham2018.public')
    hdf = pd.DataFrame({
        'image_id': asm['image_id'].values,
        'sample_obj': asm['sample_obj'].values,
        'dist_obj': asm['dist_obj'].values,
        'choice': asm.values,
    })
    mask = [(i, d) in conditions for i, d in zip(hdf['image_id'], hdf['dist_obj'])]
    return hdf[np.array(mask)].reset_index(drop=True)


def _align(source_matrix, target_matrix, collapse):
    """Restrict both response matrices to shared stimulus_ids (and choices)."""
    s_ids = set(source_matrix['stimulus_id'].values.tolist())
    t_ids = set(target_matrix['stimulus_id'].values.tolist())
    shared_ids = sorted(s_ids & t_ids)
    source_matrix = source_matrix.sel(presentation=np.isin(source_matrix['stimulus_id'].values, shared_ids))
    target_matrix = target_matrix.sel(presentation=np.isin(target_matrix['stimulus_id'].values, shared_ids))
    if not collapse:
        s_ch = set(source_matrix['choice'].values.tolist())
        t_ch = set(target_matrix['choice'].values.tolist())
        shared_ch = sorted(s_ch & t_ch)
        source_matrix = source_matrix.sel(choice=np.isin(source_matrix['choice'].values, shared_ch))
        target_matrix = target_matrix.sel(choice=np.isin(target_matrix['choice'].values, shared_ch))
    return source_matrix, target_matrix


def score_choices(model_df, human_df=None, metric_name='i2n', repetitions=20):
    """Score a model's per-trial choices against the human pool.

    Returns dict(raw, ceiling, ceiled). ``raw`` is the Pearson correlation of the
    model's d-prime response matrix with a human half; ``ceiling`` is the human
    split-half consistency; ``ceiled = raw / sqrt(ceiling)``.
    """
    metric = load_metric(metric_name)
    metric._repetitions = repetitions
    collapse = metric._collapse_distractors
    if human_df is None:
        human_df = human_trials_subset()
    human_asm = choices_to_assembly(human_df)
    model_asm = choices_to_assembly(model_df)

    # model response matrix (built once; the model has one fixed choice set)
    src = metric.dprimes(metric.build_response_matrix_from_responses(model_asm))

    rs = metric._initialize_random_state()
    raws = []
    for _ in range(repetitions):
        half = metric.generate_halves(human_asm, random_state=rs)[0]
        tgt = metric.dprimes(metric.build_response_matrix_from_responses(half))
        s, t = (metric.collapse_distractors(src), metric.collapse_distractors(tgt)) if collapse else (src, tgt)
        s, t = _align(s, t, collapse)
        raws.append(metric.correlate(s, t, skipna=True, collapse_distractors=collapse))
    raw = float(np.nanmean(raws))

    ceiling = float(metric.ceiling(human_asm, skipna=True).sel().values)
    ceiled = raw / np.sqrt(ceiling) if ceiling > 0 else float('nan')
    return {'metric': metric_name, 'raw': raw, 'ceiling': ceiling, 'ceiled': float(ceiled),
            'n_model_trials': int(len(model_df)), 'n_human_trials': int(len(human_df))}


def score_all(model_df, human_df=None, metrics=('i1', 'i2n')):
    if human_df is None:
        human_df = human_trials_subset()
    return {m: score_choices(model_df, human_df=human_df, metric_name=m) for m in metrics}
