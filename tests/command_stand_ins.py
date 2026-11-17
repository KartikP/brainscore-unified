"""Stand-ins for the external commands that make a library's folder (CogGym's by default): git, and
the Python that makes the environment and runs pip in it.

Each stand-in is an executable script. It records its call (program, arguments, environment,
process id) as one JSON line, then does the least the folder needs: ``git clone`` copies a
stand-in checkout, ``git rev-parse`` prints the pinned commit, ``-m venv`` writes an environment
whose Python runs this interpreter, and ``-m pip`` does nothing. Anything else runs this
interpreter, so the worker and the supervisor run for real. Nothing is cloned or installed.

``behavior`` changes one step: ``fail`` (a word of the failing command, for example
``'clone'`` or ``'install'``), ``head`` (the commit ``rev-parse`` prints), ``status`` (what
``git status`` prints: tracked edits), ``sleep`` (a word of a
command that sleeps for a minute after it writes its process id), ``make`` (a folder a pip
command creates, as another program would).
"""
import json
import os
import sys
from pathlib import Path

import pytest

from .coggym_stand_in import write_checkout

PINNED_COMMIT = 'a1cd9df1118fec80eba7463de237d1497d77e041'

_COMMON = '''#!{real}
import json, os, shutil, sys, time
LOG, BEHAVIOR, TEMPLATE, REAL, FAKE_ENV_PYTHON = {log!r}, {behavior!r}, {template!r}, {real!r}, {env_python!r}
behavior = json.load(open(BEHAVIOR)) if os.path.exists(BEHAVIOR) else {{}}
args = sys.argv[1:]


def record(program):
    with open(LOG, 'a') as log:
        log.write(json.dumps({{'program': program, 'executable': sys.argv[0], 'args': args, 'env': dict(os.environ),
                              'pid': os.getpid()}}) + '\\n')


def step(word):
    if behavior.get('sleep') == word:
        with open(LOG + '.pid', 'w') as handle:
            handle.write(str(os.getpid()))
        time.sleep(60)
    if behavior.get('fail') == word:
        sys.stderr.write(word + ' failed in the stand-in\\n')
        sys.exit(1)
'''

_GIT = '''
record('git')
if 'clone' in args:
    step('clone')
    shutil.copytree(TEMPLATE, args[-1], dirs_exist_ok=True)
elif 'checkout' in args:
    step('checkout')
elif 'rev-parse' in args:
    print(behavior.get('head', {commit!r}))
elif 'status' in args:
    print(behavior.get('status', ''))
'''

_PYTHON = '''
if args[:2] == ['-m', 'venv']:
    record('venv')
    step('venv')
    python = os.path.join(args[-1], 'bin', 'python')
    os.makedirs(os.path.dirname(python))
    shutil.copy(FAKE_ENV_PYTHON, python)
    os.chmod(python, 0o755)
elif args[:2] == ['-m', 'pip']:
    record('pip')
    step(next(word for word in args[2:] if not word.startswith('-')))
    if behavior.get('make') and 'freeze' in args:
        os.makedirs(behavior['make'])
    if 'freeze' in args:
        print('google-genai==3.2.0')
else:
    os.execv(REAL, [REAL] + args)
'''


class Commands:
    """The stand-ins' records and behavior."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.log, self.behavior_file = self.root / 'calls.jsonl', self.root / 'behavior.json'

    @property
    def calls(self) -> list:
        if not self.log.exists():
            return []
        return [json.loads(line) for line in self.log.read_text().splitlines()]

    def of(self, program: str) -> list:
        return [call for call in self.calls if call['program'] == program]

    def behave(self, **behavior) -> None:
        self.behavior_file.write_text(json.dumps(behavior))


def stand_ins(root: Path, monkeypatch, python: bool = True, template: Path | None = None,
              commit: str = PINNED_COMMIT) -> Commands:
    """The stand-ins in ``root``: ``git clone`` copies ``template`` (default: a CogGym checkout)
    and ``rev-parse`` prints ``commit``."""
    bin_folder = root / 'bin'
    bin_folder.mkdir(parents=True)
    template = template or write_checkout(root / 'template')
    names = dict(real=sys.executable, log=str(root / 'calls.jsonl'), behavior=str(root / 'behavior.json'),
                 template=str(template), env_python=str(bin_folder / 'env-python'))
    programs = [('git', _GIT.format(commit=commit))]
    if python:
        programs += [('python', _PYTHON), ('env-python', _PYTHON)]
    for name, body in programs:
        script = bin_folder / name
        script.write_text(_COMMON.format(**names) + body)
        script.chmod(0o755)
    monkeypatch.setenv('PATH', f"{bin_folder}{os.pathsep}{os.environ['PATH']}")
    if python:
        monkeypatch.setattr(sys, 'executable', str(bin_folder / 'python'))
    return Commands(root)


@pytest.fixture
def commands(tmp_path, monkeypatch):
    """git on ``PATH`` and this process's Python replaced by recording stand-ins."""
    return stand_ins(tmp_path / 'commands', monkeypatch)


def git_stand_in(root: Path, monkeypatch) -> Commands:
    """git on ``PATH`` replaced by a recording stand-in: a checkout reads as clean, at the pinned commit."""
    return stand_ins(root, monkeypatch, python=False)


@pytest.fixture
def pinned_git(tmp_path, monkeypatch):
    """``git_stand_in`` as a fixture."""
    return git_stand_in(tmp_path / 'git', monkeypatch)
