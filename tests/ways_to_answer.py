"""The table of registered models by ways to answer, made from the rows each plugin states.

Each plugin folder that registers models states one row for each of its identifiers in
``ways_to_answer.json``: the inputs, the regions, the ways to answer it has (writing,
likelihood, readout) and, for each kind of answer, the way that gives it. The plugin's own
``test.py`` checks its rows with ``check_statement`` on a model it builds offline (a small
model of its real class, or a stand-in of its stated layout). This file builds nothing.

    python -m tests.ways_to_answer docs/reference/ways_to_answer.md     # the reference page
"""
import json
import sys
from pathlib import Path

STATEMENT = 'ways_to_answer.json'
REGENERATE = 'python -m tests.ways_to_answer docs/reference/ways_to_answer.md'
WAYS = {'generation': 'writing', 'likelihood': 'likelihood', 'readout': 'readout'}


def statement_file(identifier, registry) -> Path:
    """The statement file of the folder whose module registers ``identifier``."""
    return Path(sys.modules[registry[identifier].__module__].__file__).parent / STATEMENT


def _missing(identifier, path) -> str:
    folder = path.parent.name
    return (f"'{identifier}' (folder {folder}) has no row in {folder}/{STATEMENT}. Add one line "
            f'"{identifier}": {{"inputs": [...], "regions": [...], "ways": [...], "answers": {{...}}}} '
            f"with what its subject declares, and check it in the folder's test.py with "
            f'tests.ways_to_answer.check_statement(subject, parts) (parts: the BrainScoreModel whose '
            f'parts answer; None for a subject that answers with its own code).')


def stated_rows(registry) -> dict:
    """Each registered identifier with the row its folder states; raises naming every one without."""
    rows, missing = {}, []
    for identifier in sorted(registry):
        path = statement_file(identifier, registry)
        stated = json.loads(path.read_text()) if path.exists() else {}
        if identifier in stated:
            rows[identifier] = stated[identifier]
        else:
            missing.append(_missing(identifier, path))
    if missing:
        raise ValueError('\n'.join(missing))
    return rows


def declared_row(subject, parts=None) -> dict:
    """The row of a built ``subject``: inputs, regions and kinds of answers from its declarations;
    the ways and the way for each kind from ``parts``, the ``BrainScoreModel`` whose wired parts
    answer (the subject itself, or the one it forwards to). ``parts`` None: the subject answers
    with its own code."""
    outputs = sorted(subject.out_channels)
    kinds = [channel.split(':', 1)[1] for channel in outputs if channel.startswith('behavior:')]
    row = {'inputs': sorted(subject.in_channels),
           'regions': [channel.split(':', 1)[1] for channel in outputs if channel.startswith('neural:')]}
    if parts is None:
        return {**row, 'ways': [], 'answers': {kind: 'own code' for kind in kinds}}
    from brainscore._model.behavioral import _formats_by_part
    wired = _formats_by_part(parts)
    if not parts.extractor:
        wired.pop('readout', None)         # a readout layer without an extractor never answers
    owners = parts._behavioral.owners      # the part that gives each kind (rule 4.11)
    # rt comes with probabilities, from the same part.
    answers = {kind: WAYS[owners.get(f'behavior:{kind}') or owners.get('behavior:probabilities')]
               for kind in kinds}
    return {**row, 'ways': [WAYS[part] for part in WAYS if part in wired], 'answers': answers}


def check_statement(subject, parts=None, registry=None) -> None:
    """Fail when the row its folder states for ``subject`` differs from what it declares."""
    if registry is None:
        from brainscore import model_registry as registry
    path = statement_file(subject.identifier, registry)
    stated = json.loads(path.read_text()).get(subject.identifier) if path.exists() else None
    if stated is None:
        raise AssertionError(_missing(subject.identifier, path))
    declared = declared_row(subject, parts)
    if stated != declared:
        raise AssertionError(f"{path.parent.name}/{STATEMENT} states for '{subject.identifier}' {stated}; "
                             f'its subject declares {declared}. State what it declares, then run '
                             f'{REGENERATE}.')


def _cell(values):
    return ', '.join(values) if values else '-'


def cells(row) -> str:
    """The table cells of one row after its models cell."""
    answers = _cell([f'{kind} ({way})' for kind, way in row['answers'].items()])
    return f'{_cell(row["inputs"])} | {_cell(row["regions"])} | {_cell(row["ways"])} | {answers} |'


def markdown(rows):
    """The page: two sentences, the table (identifiers with equal rows share a line), the command."""
    groups = {}
    for identifier, row in rows.items():
        groups.setdefault(json.dumps(row, sort_keys=True), []).append(identifier)
    lines = [
        '# Ways to answer',
        '',
        'Each registered model answers with the parts its registration wires: writing (a reply it '
        'generates), likelihood (the log probability of each option) or a readout (a classifier fitted '
        'on one of its layers). The table lists what each registration declares, and for each kind '
        'of answer the one part that gives it.',
        '',
        '| Models | Inputs | Regions | Ways | Kinds of answers (the way that gives each) |',
        '| --- | --- | --- | --- | --- |',
    ]
    for key, identifiers in sorted(groups.items(), key=lambda item: sorted(item[1])[0]):
        models = ', '.join(f'`{identifier}`' for identifier in sorted(identifiers))
        lines.append(f'| {models} | {cells(json.loads(key))}')
    lines += ['',
              '"own code": the subject is not a `BrainScoreModel`; its own `participate` gives the '
              'answer.',
              '',
              "A model that answers through a provider's API builds without a key and without a "
              "network; it needs the provider's key only to answer.",
              '',
              f'Each plugin states its rows in `{STATEMENT}` in its folder and checks them in its own '
              'test. Regenerate the table from the folder `unified`:',
              '',
              '```bash',
              REGENERATE,
              '```',
              '']
    return '\n'.join(lines)


def main(argv):
    from brainscore import model_registry
    page = markdown(stated_rows(model_registry))
    if argv:
        with open(argv[0], 'w') as file:
            file.write(page)
    else:
        print(page)


if __name__ == '__main__':
    main(sys.argv[1:])
