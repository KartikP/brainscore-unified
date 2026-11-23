"""Which part of a BrainScoreModel owns which answer format (rule 4.11).

The formats each part gives, the formats a model declares, and the check of the
``answer_owners`` statement at construction, with the line to write when a statement
is missing.
"""

from typing import Dict, Optional


# A likelihood part gives each option's summed log probability: the continuation
# log-probability format (Response Format Standard, F6; no channel yet).
LOG_PROBABILITIES = 'behavior:log_probabilities'


# The answer formats of each part kind (Response Format Standard 4.2; rule
# 4.11): format -> None when the part gives it natively, else the conversion
# that derives it. ``answer`` and ``reply`` are the two kinds of the part
# ``generation``. The likelihood gives the log probabilities natively. ``'link'``:
# ``behavior:rt`` follows the owner of ``behavior:probabilities`` and is
# never stated. Bare ``behavior`` is no part's format: it carries the answer of a response
# task under its first format, given by that format's owner (``_declared_formats``).
_PART_FORMATS = {
    'answer': {'behavior:choice': None},
    'reply': {'behavior:choice': None, 'behavior:rating': None, 'behavior:text': None},
    'likelihood': {LOG_PROBABILITIES: None, 'behavior:probabilities': 'softmax',
                   'behavior:choice': 'softmax_argmax', 'behavior:rt': 'link'},
    'readout': {'behavior:probabilities': None, 'behavior:rating': None,
                'behavior:choice': 'argmax', 'behavior:rt': 'link'},
}


def _formats_by_part(subject) -> Dict[str, Dict[str, Optional[str]]]:
    """The answer formats of each wired part, read from ``_PART_FORMATS``.

    Per part, format -> ``None`` (native) or the conversion that derives it.
    """
    wired = {'answer': subject._answer, 'reply': subject._reply,
             'likelihood': subject._likelihood, 'readout': subject._behavioral_readout_layer}
    parts: Dict[str, Dict[str, Optional[str]]] = {}
    for kind, formats in _PART_FORMATS.items():
        if wired[kind] is not None:
            part = 'generation' if kind in ('answer', 'reply') else kind
            parts.setdefault(part, {}).update(formats)
    return parts


def _declared_formats(subject) -> set:
    """The answer formats the subject declares (rules 2.4, 4.11): each format with an owner.

    A format has an owner when a statement names it or exactly one part gives it
    natively (``answer_owners``); a derived format with no statement has none, so
    it is not declared. Bare ``behavior`` is declared when ``answer=`` or a readout
    with an extractor is wired. A ``'link'`` format (``behavior:rt``) is declared with
    ``behavior:probabilities``, whose owner gives it.
    """
    declared = set(subject._behavioral.owners)
    if subject._answer is not None or (subject._behavioral_readout_layer is not None
                                       and subject._extractors):
        declared.add('behavior')
    if 'behavior:probabilities' in declared:
        declared |= {f for formats in _PART_FORMATS.values() for f, how in formats.items()
                     if how == 'link'}
    return declared


def statement_hint(subject, missing) -> str:
    """The statement that would declare each missing format a wired part only derives.

    A refusal names the exact line to write, for example
    ``answer_owners={'behavior:choice': ('readout', 'argmax')}``.
    With two candidate parts it names each line and asks for one. Empty when no wired
    part derives a missing format.
    """
    parts = _formats_by_part(subject)
    if not subject._extractors:
        parts.pop('readout', None)
    lines = []
    for channel in sorted(missing):
        candidates = [(part, formats[channel]) for part, formats in sorted(parts.items())
                      if formats.get(channel) not in (None, 'link')]
        written = [f"answer_owners={{{channel!r}: ({part!r}, {how!r})}}"
                   for part, how in candidates]
        if len(written) == 1:
            lines.append(f' {channel!r} is derived by a wired part; state it: {written[0]}.')
        elif written:
            lines.append(f' {channel!r} is derived by {len(written)} wired parts; state one: '
                         + ' or '.join(written) + '.')
    return ''.join(lines)


# The methods a part states (the record's ``method``) and the argument that wires each.
# ``similarity``: an answer chosen by the similarity of an input to each option (rule
# 4.11); the ``answer=`` part states it in place of ``generation``, and reads no instruction.
_ARGUMENTS = {'generation': 'answer= or reply=', 'likelihood': 'likelihood=',
              'readout': 'behavioral_readout_layer= (with an extractor)',
              'similarity': 'answer='}


def answer_owners(subject, stated) -> Dict[str, str]:
    """Check the owner statements at construction; return the owner part of each format (rule 4.11).

    ``stated`` is the ``answer_owners`` argument of the model: format ->
    a part (``'generation'``, ``'likelihood'``, ``'readout'``, or ``'similarity'`` for an
    ``answer=`` part that chooses by similarity) for a format the part gives natively, or ``(part, conversion)`` for a format it derives
    (``_PART_FORMATS``). A format with exactly one native part is owned by that
    part without a statement; a format with two or more native parts needs a
    statement. A wrong statement fails with a message that says what to write.
    """
    who = f"Subject '{subject.identifier}': "
    prefix = who + "answer_owners "
    parts = _formats_by_part(subject)
    if not subject._extractors:
        parts.pop('readout', None)   # a readout layer without an extractor never answers
    followers = {f for formats in _PART_FORMATS.values() for f, how in formats.items()
                 if how == 'link'}
    formats = sorted({f for kinds in _PART_FORMATS.values() for f in kinds} - followers)
    stated_parts = dict(parts)
    if subject._answer is not None:
        stated_parts['similarity'] = dict(_PART_FORMATS['answer'])
    owners: Dict[str, str] = {}
    for key, value in (stated or {}).items():
        if key == 'behavior':
            raise ValueError(f"{prefix}names 'behavior'. Delete this line: bare 'behavior' "
                             f"carries the answer of a response task under its first format, "
                             f"given by the owner of that format; it takes no statement.")
        if key in followers:
            raise ValueError(f"{prefix}names {key!r}: {key!r} follows the owner of "
                             f"'behavior:probabilities'; state that owner instead.")
        if key not in formats:
            raise ValueError(f"{prefix}names {key!r}, which is not a response format. "
                             f"Formats: {formats}.")
        if not (isinstance(value, str) or (type(value) is tuple and len(value) == 2
                                           and all(isinstance(v, str) for v in value))):
            raise ValueError(f"{prefix}gives {value!r} for {key!r}; a statement is a part "
                             f"('readout') or a tuple of a part and a conversion "
                             f"(('readout', 'argmax')).")
        part, conversion = (value, None) if isinstance(value, str) else value
        if part not in _ARGUMENTS:
            raise ValueError(f"{prefix}names the part {part!r} for {key!r}; parts are "
                             + ', '.join(map(repr, _ARGUMENTS)) + '.')
        if conversion is not None and conversion not in ('argmax', 'softmax', 'softmax_argmax'):
            raise ValueError(f"{prefix}names the conversion {conversion!r} for {key!r}; "
                             f"conversions are 'argmax', 'softmax', 'softmax_argmax'.")
        if part not in stated_parts:
            raise ValueError(f"{prefix}names {part!r} for {key!r}, but no {_ARGUMENTS[part]} "
                             f"part is wired.")
        if key not in stated_parts[part]:
            givers = sorted(p for p, given in stated_parts.items() if key in given)
            raise ValueError(f"{prefix}names {part!r} for {key!r}, but {part!r} does not give "
                             f"{key!r}. Parts of this subject that give it: {givers}.")
        given = stated_parts[part][key]
        if conversion is None and given is not None:
            raise ValueError(f"{prefix}{part!r} gives {key!r} only by a conversion. "
                             f"Write {key!r}: ({part!r}, {given!r}).")
        if conversion is not None and given is None:
            raise ValueError(f"{prefix}{part!r} gives {key!r} natively. Write {key!r}: {part!r}.")
        if conversion != given:
            raise ValueError(f"{prefix}{part!r} gives {key!r} by {given!r}, not by "
                             f"{conversion!r}. Write ({part!r}, {given!r}).")
        owners[key] = part
    for output in formats:
        native = sorted(p for p, given in parts.items()
                        if output in given and given[output] is None)
        if output in owners or not native:
            continue
        if len(native) > 1:
            raise ValueError(
                f"{who}the parts {native} give {output!r} natively. State the owner: "
                f"answer_owners={{{output!r}: {native[0]!r}}} (or "
                + ' or '.join(repr(p) for p in native[1:]) + ').')
        owners[output] = native[0]
    return owners
