"""How a BrainScoreModel's answers become assemblies.

``BehavioralReadout.respond`` gets the answers (replies, likelihood scores,
readout values); the functions here package them: one-hot choices, ratings,
texts, implied probabilities and reaction times, and ratings put on the steps of a
scale. An answer that cannot be read is kept with
``answer_valid=False`` (a NaN row for a choice or a number, rule 6.2); a trial
with no reply keeps ``raw_answer`` None. Every answer carries its response
record in ``attrs['response']`` (``brainscore_core.response_formats``); the
session then gives each trial its state.
"""

import math
from typing import Tuple

import numpy as np

from brainscore_core.response_formats import response_record
from brainscore_core.streaming import StreamEvent


def _raw(values):
    """Raw responses as text; None stays None (no reply: a missing trial)."""
    return [None if value is None else str(value) for value in values]


def _presentation(stimuli):
    """The stimulus columns as presentation coordinates.

    A column whose cells are sequences (the options of a choice whose
    ``labels`` names a column) cannot be a coordinate and is left out.
    """
    return {column: ('presentation', list(stimuli[column].values)) for column in stimuli.columns
            if not any(isinstance(value, (list, tuple, set, dict, np.ndarray))
                       for value in stimuli[column].values)}


def one_hot(chosen, valid, stimuli, row_labels, axis, record, *, raw, errors=None):
    """One row per stimulus, 1 for the chosen label; invalid rows are NaN (rule 6.2).

    ``row_labels`` are each row's options; ``axis`` names the choice axis (the
    labels, or option positions when the options differ for each row).
    ``record`` is the answer's response record. ``raw`` is each trial's raw
    response: the reply, or the source row of a derived choice.
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
              'raw_answer': ('presentation', _raw(raw))}
    if errors is not None:
        coords['parse_error'] = ('presentation', [e or '' for e in errors])
    return BehavioralAssembly(values, coords=coords, dims=['presentation', 'choice'],
                              attrs={'response': record})


def numbers(parsed, stimuli, record, raw=None):
    """One number per stimulus (a rating); invalid ones are NaN."""
    from brainscore_core.supported_data_standards.brainio.assemblies import BehavioralAssembly
    values = np.array([np.nan if v is None else v for v, _ in parsed], dtype=float)
    coords = {**_presentation(stimuli),
              'answer_valid': ('presentation', [v is not None for v, _ in parsed]),
              'parse_error': ('presentation', [e or '' for _, e in parsed])}
    if raw is not None:
        coords['raw_answer'] = ('presentation', _raw(raw))
    return BehavioralAssembly(values, coords=coords, dims=['presentation'],
                              attrs={'response': record})


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
    # A reply of None is no answer (missing): its value stays None.
    values = np.array([v if v is not None else (None if r is None else '')
                       for (v, _), r in zip(parsed, replies)], dtype=object)
    coords = {**_presentation(stimuli),
              'answer_valid': ('presentation', [v is not None for v, _ in parsed]),
              'raw_answer': ('presentation', _raw(replies))}
    return BehavioralAssembly(values, coords=coords, dims=['presentation'],
                              attrs={'response': response_record('behavior:text', 'generation')})


def reaction_time(probabilities, valid, source, link, stimuli):
    """An implied reaction time per stimulus from label probabilities (larger = slower).

    ``probabilities`` are the float64 rows the link reads, ``valid`` their validity
    as probability rows (a reaction time from an invalid row is invalid), ``source``
    their response record. The rows are kept as the source evidence.
    """
    from brainscore_core.supported_data_standards.brainio.assemblies import BehavioralAssembly
    values = np.asarray(probabilities, dtype=float)
    ordered = np.sort(values, axis=1)[:, ::-1]
    if link == 'evidence_margin':
        rt = 1.0 - (ordered[:, 0] - ordered[:, 1])
    else:                                         # surprisal of the chosen (most probable) label
        rt = -np.log2(np.clip(ordered[:, 0], 1e-12, 1.0))
    coords = {**_presentation(stimuli),
              'answer_valid': ('presentation', [bool(v) for v in valid]),
              'parse_error': ('presentation', conversion_errors(valid)),
              'raw_answer': ('presentation', source_rows(values))}
    # A link gives a reading-time predictor, not a time in ms: units 'monotone'.
    return BehavioralAssembly(rt, dims=['presentation'], coords=coords,
                              attrs={'response': response_record(
                                  'behavior:rt', 'link', source_format=source['format'],
                                  parameters={'link': link, 'source_method': source['method'],
                                              'units': 'monotone'})})


def score_rows(scores, row_labels):
    """Each row's likelihood scores, in the order of that row's options."""
    return np.array([[float(row[label]) for label in options]
                     for row, options in zip(scores, row_labels)], dtype=float)


def implied_array(values, valid, stimuli, axis, record, raw=None):
    """A likelihood's implied probabilities per option, float32 (the copy its reaction-time
    link reads; the format keeps float64 when a subject gives it). ``valid`` is the validity
    of each row the conversion returned (or None); an invalid row is marked so. ``raw`` is the source evidence of a derived answer (each source row as text).
    """
    from brainscore_core.supported_data_standards.brainio.assemblies import BehavioralAssembly
    coords = {**_presentation(stimuli), 'choice': list(axis)}
    if valid is not None:
        coords['answer_valid'] = ('presentation', [bool(v) for v in valid])
        coords['parse_error'] = ('presentation', conversion_errors(valid))
    if raw is not None:
        coords['raw_answer'] = ('presentation', list(raw))
    return BehavioralAssembly(values.astype(np.float32), coords=coords,
                              dims=['presentation', 'choice'], attrs={'response': record})


def log_probabilities(values, stimuli, axis, record):
    """The log probability of each option, float64, as the likelihood gave it; core checks them."""
    from brainscore_core.supported_data_standards.brainio.assemblies import BehavioralAssembly
    coords = {**_presentation(stimuli), 'choice': list(axis)}
    return BehavioralAssembly(np.asarray(values, dtype=np.float64), coords=coords,
                              dims=['presentation', 'choice'], attrs={'response': record})


def source_rows(values):
    """Each source row of a derived answer as text: the raw evidence of the answer."""
    return [str([float(v) for v in row]) for row in np.asarray(values)]


def conversion_errors(valid):
    """The reason of each row a conversion could not use ('' for a valid row)."""
    return ['' if ok else 'the source row is not a valid row of its format' for ok in valid]


def readout_probabilities(assembly, stimuli):
    """A readout's probabilities, one row per stimulus, for ``behavior:probabilities``: the
    readout's own values and float type (float64), its own presentation coordinates (the
    extractor's) and the readout's record. The coordinates have one order in every process:
    first the columns of the stimulus table in the table's order, then every other
    coordinate sorted by name (an extractor can give its coordinates in a hash-dependent
    order). The legacy facade adds the benchmark's columns the answer lacks, in the
    benchmark's order, as it did on the bare channel."""
    from brainscore_core.supported_data_standards.brainio.assemblies import BehavioralAssembly
    index = assembly.indexes.get('presentation')
    present = {name for name in (getattr(index, 'names', None) or ()) if name and name != 'presentation'}
    present |= {name for name, coord in assembly.coords.items()
                if coord.dims == ('presentation',) and name != 'presentation'}
    table = [name for name in stimuli.columns if name in present]
    names = table + sorted(present - set(table))
    coords = {name: ('presentation', list(assembly[name].values)) for name in names}
    if 'stimulus_id' not in coords:   # the format matches rows by stimulus id
        coords = {'stimulus_id': ('presentation', list(stimuli['stimulus_id'])), **coords}
    coords['choice'] = list(assembly['choice'].values)
    return BehavioralAssembly(np.asarray(assembly.values), coords=coords, dims=['presentation', 'choice'],
                              attrs={'response': response_record('behavior:probabilities', 'readout')})


def in_order(assembly, labels):
    """Readout columns in the task's label order, as ``behavior:choice`` has them."""
    present = [label for label in labels if label in set(assembly['choice'].values)]
    return assembly.sel(choice=present) if len(present) == assembly.sizes['choice'] else assembly


def with_record(assembly, record):
    """``assembly`` with ``record`` as its response record."""
    return assembly.assign_attrs(response=record)


RATING_ROUNDING = 'nearest step, ties to the higher step'
# The schema of the log probabilities a likelihood part gives under a verbatim Choice:
# f"{text} {option}" as plain text, no special tokens, no template, summed natural logs.
VERBATIM_LOG_PROBABILITIES = {
    'layout': 'continuation', 'context': 'shared', 'aggregation': 'sum', 'units': 'nats',
    'tokenization': {'join': 'space', 'special_tokens': 'none', 'template': 'none',
                     'truncation': 'none'}}
# A verbatim trial text reaches a reply part as given; the part may wrap it in its own chat
# template. The record says only that: a reply part is any callable.
VERBATIM_IN_TEMPLATE = {'template': "the reply part's own, if it has one"}


def _on_scale_steps(values, scale) -> Tuple[list, dict]:
    """Each value on the nearest step of ``scale``, and the record parameters that say so.

    The grid of a scale is decimal: the points Decimal(repr(low)) + n * Decimal(repr(step)).
    The step index is read from the raw value in decimal; an index within 1e-9 of a half is a
    tie and goes to the higher step. The stored value is the float of that grid point, the
    value core's step rule accepts, kept inside the bounds: the index never passes the last
    point inside them, and a point one float rounding beyond ``high`` (or ``low``) is
    stored as that bound. A raw value outside the bounds, or not finite,
    is not rounded and not clipped (core marks it invalid). A scale with no step leaves every
    value unchanged and records no rounding.
    """
    if scale.step is None:
        return list(values), {}
    return [_grid_point(v, scale) for v in values], {'rounding': RATING_ROUNDING}


def _grid_point(value, scale):
    from decimal import Decimal, ROUND_FLOOR, localcontext
    if not (math.isfinite(value) and scale.low <= value <= scale.high):
        return value
    with localcontext() as context:
        context.prec = 50
        low, step = Decimal(repr(float(scale.low))), Decimal(repr(float(scale.step)))
        last = int(((Decimal(repr(float(scale.high))) - low) / step + Decimal('1e-9'))
                   .to_integral_value(ROUND_FLOOR))
        index = (Decimal(repr(float(value))) - low) / step
        below = int(index.to_integral_value(ROUND_FLOOR))
        n = below + 1 if index - below > Decimal('0.5') - Decimal('1e-9') else below
        point = float(low + min(n, last) * step)
    # A high (or low) one float rounding off a grid point: the stored value stays inside the
    # bounds; it moves by at most 1e-9 of a step, which core's step rule accepts.
    return min(max(point, float(scale.low)), float(scale.high))
