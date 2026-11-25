"""Any outside library runs in its own environment through the general path, shown with a
made-up library (Lanternfield) whose code is the code blocks of ``brainscore/harnesses/README.md``.

Each python block of the README starts with the comment of its file; the blocks of one file
are joined in order and written to a temporary folder. git and the Python that makes the
environment and runs pip are recording stand-ins: nothing is cloned or installed.
"""
import dataclasses
import importlib.util
import json
import os
import signal
import time
import re
import sys
from pathlib import Path

import pytest

import brainscore.harnesses
from brainscore import BrainScoreModel, LoadError
from brainscore.experiments import Experiment, RecordInputsOutputs

from tests.command_stand_ins import stand_ins

pytestmark = pytest.mark.unit

README = Path(brainscore.harnesses.__file__).with_name('README.md')
COMMIT = '4c1d6e0a9b3f27d85e6c0b1a2f4d9e7c3b5a8f60'
# The library itself, in the checkout the clone stand-in copies.
LIBRARY = '''
def run_trials(questions, ask):
    answers = [ask(question) for question in questions]
    return {'answers': answers, 'score': sum(answer == 'yes' for answer in answers) / len(answers)}
'''


def _readme_files(folder: Path) -> dict:
    """The README's python blocks as files in ``folder``, by the file named in their first line."""
    files = {}
    for block in re.findall(r'^```python\n(.*?)^```', README.read_text(), flags=re.S | re.M):
        first, _, body = block.partition('\n')
        name = Path(first.removeprefix('# ')).name
        files[name] = files.get(name, '') + body + '\n'
    for name, text in files.items():
        (folder / name).write_text(text)
    return files


@pytest.fixture
def lanternfield(tmp_path, monkeypatch):
    """The README's module of Lanternfield, its stand-in commands and an empty place."""
    code = tmp_path / 'code'
    code.mkdir()
    assert {'lanternfield.py', '_lanternfield_worker.py'} <= set(_readme_files(code))
    (code / 'lanternfield_requirements.txt').write_text('lanternfield-stand-in==1.0\n')
    template = tmp_path / 'template'
    (template / 'lanternfield').mkdir(parents=True)
    (template / 'lanternfield' / '__init__.py').write_text(LIBRARY)
    commands = stand_ins(tmp_path / 'commands', monkeypatch, template=template, commit=COMMIT)
    monkeypatch.setenv('BS_INSTALL_DEPENDENCIES', 'yes')
    monkeypatch.setenv('BRAINSCORE_ENVIRONMENTS_HOME', str(tmp_path / 'environments'))
    spec = importlib.util.spec_from_file_location('lanternfield_readme', code / 'lanternfield.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, commands


def test_a_made_up_library_runs_a_command_in_its_own_environment(lanternfield, tmp_path):
    module, commands = lanternfield
    protocol = module.protocol(['Is the lantern lit?', 'Is the field dark?'])
    # The folder is under the library's own name, made from its repository and commit.
    (folder,) = [path for path in (tmp_path / 'environments' / 'Lanternfield').iterdir()
                 if not path.name.startswith('.')]
    assert folder.name.startswith(COMMIT)
    assert any('https://example.org/lanternfield.git' in call['args'] for call in commands.of('git'))
    assert json.loads((folder / 'ready.json').read_text())['revision'] == COMMIT
    # One command through the child: each call of the library is one session of the subject.
    subject = BrainScoreModel('fixture', in_channels={'text'}, reply=lambda prompt: 'yes')
    result = Experiment(subject=subject, protocol=protocol, tools=[RecordInputsOutputs()],
                        output_dir=tmp_path / 'run').run()
    assert result.value == {'answers': ['yes', 'yes'], 'score': 1.0}


def _python_reporting(path: Path, version: str, log: Path) -> str:
    """A stand-in Python program that reports ``version``, and whose ``-m venv`` makes an
    environment whose Python reports it too (pip there is the recording stand-in)."""
    env_python = Path(sys.executable).with_name('env-python')        # the commands' stand-in
    path.write_text(
        f'#!/bin/sh\nif [ "$1" = "-c" ]; then echo {version}; exit; fi\necho "$@" >> "{log}"\n'
        f'if [ "$1" = "-m" ] && [ "$2" = "venv" ]; then mkdir -p "$3/bin"\n'
        f'  printf \'#!/bin/sh\\nif [ "$1" = "-c" ]; then echo {version}; exit; fi\\nexec "{env_python}" "$@"\\n\' > "$3/bin/python"\n'
        f'  chmod +x "$3/bin/python"; exit; fi\nexec "{sys.executable}" "$@"\n')
    path.chmod(0o755)
    return str(path)


def test_another_python_gives_another_folder_with_its_version(lanternfield, tmp_path):
    module, commands = lanternfield
    usual = module.ready_folder(module.LANTERNFIELD, 'Lanternfield.demo')
    log = tmp_path / 'other-python.log'
    other = _python_reporting(tmp_path / 'other-python', '3.99.0', log)
    folder = module.ready_folder(dataclasses.replace(module.LANTERNFIELD, python=other), 'Lanternfield.demo')
    assert folder != usual and folder.parent == usual.parent
    assert json.loads((folder / 'ready.json').read_text())['python'] == '3.99.0'
    assert log.read_text().startswith('-m venv')        # the named Python made the environment


def test_an_empty_python_name_makes_the_folder_with_this_python(lanternfield):
    module, _ = lanternfield
    usual = module.ready_folder(module.LANTERNFIELD, 'Lanternfield.demo')
    empty = module.ready_folder(dataclasses.replace(module.LANTERNFIELD, python=''), 'Lanternfield.demo')
    assert empty == usual


def test_a_requirement_line_names_a_file_of_the_checkout_only_when_the_library_asks(lanternfield, tmp_path):
    module, commands = lanternfield
    requirements = tmp_path / 'checkout_requirements.txt'
    requirements.write_text('./vendor/lantern-1.0-py3-none-any.whl\nplain==1.0\n')
    library = dataclasses.replace(module.LANTERNFIELD, name='Wheelfield', requirements=requirements,
                                  checkout_requirements=True)
    folder = module.ready_folder(library, 'Wheelfield.demo')
    (install,) = [call['args'] for call in commands.of('pip') if 'install' in call['args']
                  and 'Wheelfield' in call['args'][-1]]
    assert install[-1].endswith('requirements.txt')
    lines = (folder / 'requirements.txt').read_text().splitlines()
    assert Path(lines[0]).is_absolute() and lines[0].endswith('checkout/vendor/lantern-1.0-py3-none-any.whl')
    assert lines[1] == 'plain==1.0'
    # The default: the library's own file, every line as it is.
    module.ready_folder(module.LANTERNFIELD, 'Lanternfield.demo')
    assert any(call['args'][-1] == str(module.LANTERNFIELD.requirements) for call in commands.of('pip'))
    assert folder.exists()


def test_a_folder_made_for_another_python_or_library_is_refused_also_when_named(lanternfield, tmp_path):
    module, _ = lanternfield
    log = tmp_path / 'python.log'
    older = dataclasses.replace(module.LANTERNFIELD,
                                python=_python_reporting(tmp_path / 'older', '3.98.0', log))
    folder = module.ready_folder(older, 'Lanternfield.demo')
    newer = dataclasses.replace(module.LANTERNFIELD,
                                python=_python_reporting(tmp_path / 'newer', '3.99.0', log))
    other_library = dataclasses.replace(older, name='Otherfield', repository='https://example.org/other.git')
    for library, recorded, needed in [(newer, '3.98.0', '3.99.0'),
                                      (other_library, 'Lanternfield', 'Otherfield')]:
        with pytest.raises(LoadError) as refused:
            module.ready_folder(library, 'Lanternfield.demo', folder)
        assert recorded in str(refused.value) and needed in str(refused.value)
    assert json.loads((folder / 'ready.json').read_text())['python'] == '3.98.0'      # unchanged


def test_two_names_that_differ_only_by_case_get_two_folders(lanternfield):
    module, _ = lanternfield
    upper = module.ready_folder(module.LANTERNFIELD, 'Lanternfield.demo')
    lower = module.ready_folder(dataclasses.replace(module.LANTERNFIELD, name='lanternfield'), 'Lanternfield.demo')
    assert not upper.samefile(lower)
    assert {json.loads((folder / 'ready.json').read_text())['library'] for folder in (upper, lower)} == {
        'Lanternfield', 'lanternfield'}


@pytest.mark.parametrize('program', ['missing', 'python2'])
def test_a_python_that_is_absent_or_not_python_3_is_refused_before_anything_is_made(
        lanternfield, tmp_path, program):
    module, commands = lanternfield
    python = tmp_path / program
    if program == 'python2':
        python.write_text('#!/bin/sh\necho 2.7.18\n')
        python.chmod(0o755)
    with pytest.raises(LoadError) as refused:
        module.ready_folder(dataclasses.replace(module.LANTERNFIELD, python=str(python)), 'Lanternfield.demo')
    assert 'Lanternfield.demo' in str(refused.value) and 'Python 3' in str(refused.value)
    assert commands.calls == [] and not (tmp_path / 'environments').exists()


def test_a_helper_that_the_python_query_starts_does_not_outlive_the_query(lanternfield, tmp_path, monkeypatch):
    """A launcher starts a helper, prints a Python 3 version and exits: the helper is stopped."""
    module, _ = lanternfield
    monkeypatch.setenv('BS_INSTALL_DEPENDENCIES', 'no')
    launcher, pid_file = tmp_path / 'launcher', tmp_path / 'helper.pid'
    launcher.write_text(f'#!/bin/sh\nsleep 60 >/dev/null 2>&1 &\necho $! > "{pid_file}"\necho 3.11.0\n')
    launcher.chmod(0o755)
    with pytest.raises(LoadError):
        module.ready_folder(dataclasses.replace(module.LANTERNFIELD, python=str(launcher)), 'Lanternfield.demo')
    helper = int(pid_file.read_text())
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            os.kill(helper, 0)
        except ProcessLookupError:
            return
        time.sleep(0.05)
    os.kill(helper, signal.SIGKILL)
    pytest.fail('the helper outlived the Python query')
