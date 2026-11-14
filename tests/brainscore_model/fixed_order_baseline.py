"""Capture what the fixed answer order chose, from the code of the base commit.

Run from the unified root to regenerate ``tests/fixtures/fixed_order_baseline.json``::

    python -m tests.brainscore_model.fixed_order_baseline

It takes ``BehavioralReadout._choose_path`` and ``_PATHS`` from
``git show 280f49c:brainscore/_model/behavioral.py`` (the last commit with the
fixed order generation, likelihood, readout), compiles them in memory and calls
them with fake parts for every part combination and task configuration below.
The test (``test_answer_ownership.py``) rebuilds the same tasks with
``task_from`` and compares the current selection with the captured outcomes.
"""
import ast
import itertools
import json
import subprocess
import warnings
from pathlib import Path
from types import SimpleNamespace

BASE = '280f49c'
FIXTURE = Path(__file__).resolve().parents[1] / 'fixtures' / 'fixed_order_baseline.json'
ARGUMENTS = ('answer', 'reply', 'likelihood', 'behavioral_readout_layer')
PART_SETS = [list(c) for size in range(1, 5) for c in itertools.combinations(ARGUMENTS, size)]


def configurations():
    """Every task form the selection reads: legacy labels and instruction, fitting, response settings."""
    found = []
    for instruction, labels, fitting in itertools.product((False, True), (None, [], ['cat', 'dog']),
                                                          (False, True)):
        found.append({'kind': 'legacy', 'instruction': instruction, 'label_set': labels,
                      'fitting': fitting})
    for labels, probabilities, rt, raw_scores, raw_text, confidence, fitting in itertools.product(
            ('shared', 'per_row'), *[(False, True)] * 2, (False, True), (False, True), (False, True),
            (False, True)):
        description = {'kind': 'choice', 'labels': labels, 'probabilities': probabilities,
                       'reaction_time': 'surprisal' if rt else None, 'raw_scores': raw_scores,
                       'raw_text': raw_text, 'confidence': confidence, 'fitting': fitting}
        try:
            task_from(description, object())
        except ValueError:
            continue   # a combination Choice itself refuses
        found.append(description)
    for kind, fitting in itertools.product(('scale', 'text'), (False, True)):
        found.append({'kind': kind, 'fitting': fitting})
    return found


def task_from(description, fitting_stimuli):
    from brainscore_core.contract import TaskContext
    from brainscore_core.responses import Choice, Scale, Text
    fitting = fitting_stimuli if description['fitting'] else None
    kind = description['kind']
    if kind == 'legacy':
        return TaskContext(task_type='probabilities', label_set=description['label_set'],
                           instruction='Which?' if description['instruction'] else None,
                           fitting_stimuli=fitting)
    if kind == 'scale':
        return TaskContext(instruction='How much?', response=Scale(1, 7), fitting_stimuli=fitting)
    if kind == 'text':
        return TaskContext(instruction='Say it.', response=Text(), fitting_stimuli=fitting)
    choice = Choice(['cat', 'dog'] if description['labels'] == 'shared' else 'options',
                    probabilities=description['probabilities'],
                    reaction_time=description['reaction_time'], raw_scores=description['raw_scores'],
                    raw_text=description['raw_text'],
                    confidence=Scale(1, 5) if description['confidence'] else None)
    return TaskContext(instruction=None if description['raw_text'] else 'Which?', response=choice,
                       fitting_stimuli=fitting)


def _base_selection():
    source = subprocess.run(['git', 'show', f'{BASE}:brainscore/_model/behavioral.py'],
                            capture_output=True, text=True, check=True,
                            cwd=Path(__file__).resolve().parents[2]).stdout
    cls = next(node for node in ast.parse(source).body
               if isinstance(node, ast.ClassDef) and node.name == 'BehavioralReadout')
    wanted = [node for node in cls.body if (isinstance(node, ast.FunctionDef) and node.name == '_choose_path')
              or (isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == '_PATHS')]
    namespace = {}
    exec('from typing import Optional, Tuple', namespace)
    exec(compile(ast.Module(body=wanted, type_ignores=[]), f'{BASE}:behavioral.py', 'exec'), namespace)
    return namespace['_choose_path'], namespace['_PATHS']


def capture():
    from brainscore_core.compatibility import ExecutionNotFeasibleError
    choose, paths = _base_selection()
    cases = []
    for parts in PART_SETS:
        wired = SimpleNamespace(identifier='fake', _extractors={'text': object()},
                                **{f'_{name}': object() if name in parts else None
                                   for name in ('answer', 'reply', 'likelihood')},
                                _behavioral_readout_layer='late' if 'behavioral_readout_layer' in parts else None)
        for description in configurations():
            readout = SimpleNamespace(owner=wired, _PATHS=paths)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    outcome = list(choose(readout, task_from(description, object())))
            except ExecutionNotFeasibleError:
                outcome = 'refused'
            cases.append({'parts': parts, 'task': description, 'outcome': outcome})
    return {'base': BASE, 'cases': cases}


if __name__ == '__main__':
    captured = capture()
    lines = ',\n'.join(json.dumps(case, sort_keys=True) for case in captured['cases'])
    FIXTURE.write_text(f'{{"base": "{BASE}", "cases": [\n{lines}\n]}}\n', encoding='utf-8')
    print(f'{FIXTURE}: {len(captured["cases"])} cases')
