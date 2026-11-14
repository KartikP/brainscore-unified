"""How a BrainScoreModel's answers become assemblies.

``BehavioralReadout.respond`` gets the answers (replies, likelihood scores,
readout values); the functions here package them: one-hot choices, ratings,
texts, implied probabilities and reaction times, and a legacy task's one-hot
rows over ``label_set``. An answer that cannot be read is kept with
``answer_valid=False`` (a NaN row for a choice or a number, rule 6.2).
"""

import numpy as np

from brainscore_core.streaming import StreamEvent


def legacy_one_hot(answers, label_set, stimuli):
    """A legacy task's answers as one-hot rows over ``label_set``.

    An answer counts only when it is exactly a label; any other answer is
    recorded as invalid (NaN row, ``answer_valid=False``) with a warning, as
    legacy behavioral benchmarks expect. The coordinates keep their order:
    ``stimulus_id``, the other stimulus columns, ``raw_answer``, ``answer_valid``.
    """
    from brainscore_core.supported_data_standards.brainio.assemblies import BehavioralAssembly
    import warnings
    label_to_idx = {label: i for i, label in enumerate(label_set)}
    proba = np.zeros((len(stimuli), len(label_set)), dtype=np.float32)
    raw_answers, valid = [], []
    for i, predicted in enumerate(answers):
        raw_answers.append(str(predicted))
        valid.append(predicted in label_to_idx)
        if valid[-1]:
            proba[i, label_to_idx[predicted]] = 1.0
        else:
            # No label is credited: a substituted label would bias accuracy toward it.
            proba[i, :] = np.nan
            warnings.warn(
                f"answer function returned {predicted!r}, not in "
                f"label_set={label_set}. The response is recorded as invalid "
                f"(NaN probabilities, answer_valid=False)."
            )
    coords = {'stimulus_id': ('presentation', list(stimuli['stimulus_id'].values))}
    for column in stimuli.columns:
        if column != 'stimulus_id':
            coords[column] = ('presentation', list(stimuli[column].values))
    coords['raw_answer'] = ('presentation', raw_answers)
    coords['answer_valid'] = ('presentation', valid)
    return BehavioralAssembly(proba, coords={**coords, 'choice': label_set},
                              dims=['presentation', 'choice'],
                              attrs={'measure': {'kind': 'reported', 'source': 'answer'}})


def _presentation(stimuli):
    """The stimulus columns as presentation coordinates.

    A column whose cells are sequences (the options of a choice whose
    ``labels`` names a column) cannot be a coordinate and is left out.
    """
    return {column: ('presentation', list(stimuli[column].values)) for column in stimuli.columns
            if not any(isinstance(value, (list, tuple, set, dict, np.ndarray))
                       for value in stimuli[column].values)}


def one_hot(chosen, valid, stimuli, row_labels, axis, source, raw=None, errors=None):
    """One row per stimulus, 1 for the chosen label; invalid rows are NaN (rule 6.2).

    ``row_labels`` are each row's options; ``axis`` names the choice axis (the
    labels, or option positions when the options differ for each row).
    """
    from brainscore_core.supported_data_standards.brainio.assemblies import BehavioralAssembly
    values = np.zeros((len(stimuli), len(axis)), dtype=np.float32)
    for i, (label, ok) in enumerate(zip(chosen, valid)):
        if ok:
            values[i, list(row_labels[i]).index(label)] = 1.0
        else:
            values[i] = np.nan
    coords = {**_presentation(stimuli), 'choice': list(axis),
              'answer_valid': ('presentation', list(valid)),
              'raw_answer': ('presentation', [str(r) for r in (raw or chosen)])}
    if errors is not None:
        coords['parse_error'] = ('presentation', [e or '' for e in errors])
    return BehavioralAssembly(values, coords=coords, dims=['presentation', 'choice'],
                              attrs={'measure': {'kind': 'reported' if source in ('reply', 'answer')
                                                 else 'implied', 'source': source}})


def numbers(parsed, stimuli, source, raw=None):
    """One number per stimulus (a rating or a confidence); invalid ones are NaN."""
    from brainscore_core.supported_data_standards.brainio.assemblies import BehavioralAssembly
    values = np.array([np.nan if v is None else v for v, _ in parsed], dtype=float)
    coords = {**_presentation(stimuli),
              'answer_valid': ('presentation', [v is not None for v, _ in parsed]),
              'parse_error': ('presentation', [e or '' for _, e in parsed])}
    if raw is not None:
        coords['raw_answer'] = ('presentation', [str(r) for r in raw])
    return BehavioralAssembly(values, coords=coords, dims=['presentation'],
                              attrs={'measure': {'kind': 'reported' if source == 'reply'
                                                 else 'implied', 'source': source}})


def reply_and_meta(reply):
    """A reply's text and its extra data: a reply returns the text, or a
    ``StreamEvent`` whose payload is the text and whose meta is the extra data."""
    if isinstance(reply, StreamEvent):
        return reply.payload, dict(reply.meta)
    return reply, {}


def trials_meta(extras):
    """The meta of an answer event from the extra data of its trials.

    One trial: its extra data as is. Several trials: each key holds a list with
    one value per trial, in order (``None`` for a trial without that key).
    """
    if len(extras) == 1:
        return dict(extras[0])
    keys = list(dict.fromkeys(key for extra in extras for key in extra))
    return {key: [extra.get(key) for extra in extras] for key in keys}


def texts(parsed, replies, stimuli):
    from brainscore_core.supported_data_standards.brainio.assemblies import BehavioralAssembly
    values = np.array([v if v is not None else '' for v, _ in parsed], dtype=object)
    coords = {**_presentation(stimuli),
              'answer_valid': ('presentation', [v is not None for v, _ in parsed])}
    return BehavioralAssembly(values, coords=coords, dims=['presentation'],
                              attrs={'measure': {'kind': 'reported', 'source': 'reply'}})


def reaction_time(probabilities, link, stimuli):
    """An implied reaction time per stimulus from label probabilities (larger = slower)."""
    from brainscore_core.supported_data_standards.brainio.assemblies import BehavioralAssembly
    values = np.asarray(probabilities.values, dtype=float)
    ordered = np.sort(values, axis=1)[:, ::-1]
    if link == 'evidence_margin':
        rt = 1.0 - (ordered[:, 0] - ordered[:, 1])
    else:                                         # surprisal of the chosen (most probable) label
        rt = -np.log2(np.clip(ordered[:, 0], 1e-12, 1.0))
    return BehavioralAssembly(rt, dims=['presentation'], coords=_presentation(stimuli),
                              attrs={'measure': {'kind': 'implied', 'link': link,
                                                 'source': probabilities.attrs.get('measure', {}).get('source')}})


def score_rows(scores, row_labels):
    """Each row's likelihood scores, in the order of that row's options."""
    return np.array([[float(row[label]) for label in options]
                     for row, options in zip(scores, row_labels)], dtype=float)


def softmax(logp):
    logp = logp - logp.max(axis=1, keepdims=True)
    p = np.exp(logp)
    return p / p.sum(axis=1, keepdims=True)


def implied_array(values, stimuli, axis, source, scale=None):
    """Implied values per option. Raw scores (``scale='log'``) keep full precision."""
    from brainscore_core.supported_data_standards.brainio.assemblies import BehavioralAssembly
    measure = {'kind': 'implied', 'source': source}
    if scale is not None:
        measure['scale'] = scale
    return BehavioralAssembly(values if scale is not None else values.astype(np.float32),
                              coords={**_presentation(stimuli), 'choice': list(axis)},
                              dims=['presentation', 'choice'], attrs={'measure': measure})


def in_order(assembly, labels):
    """Readout columns in the task's label order, as ``behavior:choice`` has them."""
    present = [label for label in labels if label in set(assembly['choice'].values)]
    return assembly.sel(choice=present) if len(present) == assembly.sizes['choice'] else assembly


def implied(assembly, source):
    return assembly.assign_attrs(measure={'kind': 'implied', 'source': source})
