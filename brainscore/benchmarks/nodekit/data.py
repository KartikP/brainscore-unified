"""The participants' trials of a NodeKit task, read from the task repository's pinned clone (nothing is
copied into Brain-Score).

Each participant's run is an nkl trace (``prolific/<folder>/submissions/<file>.json``: a JSON array of
lines ``[index, time_us, kind, payload]``); the payload of its ``ProgramCompleted`` line is the program's
result, a list of trial records. Only that line is read. A run with no such line did not complete: it is
counted and left out (``trials(...).attrs['not_completed']``). Participants are named ``p1``, ``p2``, ...
in the order of their file names; the files' own identifiers never leave the clone.

``trials(...).attrs['longest_run_ms']`` is the time of the longest completed run (its trace's last line).

``rows`` turns a program's result (a participant's or a subject's, the same value JSON) into one row per
trial: the fields of each record, with a nested record's fields beside them.
"""
from __future__ import annotations

import json

import pandas as pd

from brainscore_core.supported_data_standards.brainio.assemblies import BehavioralAssembly

from brainscore.harnesses.nodekit import VIEWER_FOLDERS as FOLDERS      # each bundle's participants' folder


def rows(result: list) -> pd.DataFrame:
    """One row per trial record, in order (``trial`` counts from 0); a nested record's fields become
    columns of their own (a list stays one cell)."""
    flat = []
    for number, record in enumerate(result):
        row = {'trial': number}
        for key, value in record.items():
            if isinstance(value, dict):
                row.update(value)
            else:
                row[key] = value
        flat.append(row)
    return pd.DataFrame(flat)


def completed_result(trace: list):
    """The payload of the trace's ``ProgramCompleted`` line, or None."""
    return next((line[3] for line in trace if len(line) > 3 and line[2] == 'ProgramCompleted'), None)


def trials(task: str) -> BehavioralAssembly:
    """Every participant's trials of ``task`` (a bundle name of ``FOLDERS``) as one assembly over
    ``presentation``; every field is a coordinate (a list as JSON text), the values are the trial numbers."""
    from brainscore.harnesses.nodekit import source
    files = sorted((source() / 'prolific' / FOLDERS[task] / 'submissions').glob('*.json'))
    tables, not_completed, run_ms = [], 0, []
    for number, path in enumerate(files, 1):
        trace = json.loads(path.read_text())
        result = completed_result(trace)
        if result is None:
            not_completed += 1
            continue
        run_ms.append(trace[-1][1] / 1000)          # the run's last line, in ms
        table = rows(result)
        table.insert(0, 'participant', f'p{number}')
        tables.append(table)
    table = pd.concat(tables, ignore_index=True)
    coords = {column: ('presentation', [value if not isinstance(value, (list, dict)) else json.dumps(value)
                                        for value in table[column]])
              for column in table.columns}
    assembly = BehavioralAssembly(table['trial'].values, coords=coords, dims=['presentation'])
    assembly.attrs.update(task=task, not_completed=not_completed, participants=len(tables),
                          longest_run_ms=max(run_ms) if run_ms else None)
    return assembly


def table(assembly: BehavioralAssembly) -> pd.DataFrame:
    """The assembly's trials as a table (one column per coordinate; lists as JSON text)."""
    return assembly['presentation'].to_index().to_frame(index=False)
