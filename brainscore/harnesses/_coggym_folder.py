"""The folder Brain-Score makes for CogGym: the pinned checkout and CogGym's own environment.

``<folder>/checkout`` is the authors' repository at the pinned commit; ``<folder>/env`` is a
virtual environment with CogGym's pinned packages only (``coggym_requirements.txt``,
installed with ``--no-deps`` and checked with ``pip check``); ``<folder>/ready.json`` says the
folder is complete (the commit, the requirements' hash and the environment's ``pip freeze``).
Nothing is installed into the scoring environment.

The default place is ``~/.brain-score/coggym/<commit>-<requirements hash>`` (the root can be
moved with ``BRAINSCORE_COGGYM_FOLDER``), so a changed pin gets a new folder. The folder is
made at the first use when ``BS_INSTALL_DEPENDENCIES`` allows installs (``yes``, the default, or
``newenv``: the folder's environment is a new one either way): in a temporary
sibling folder, checked, then published by a rename that never replaces (an existing folder
is never changed). Creation is serialized by a lock file next to the folder. A failed step
removes its own temporary folder. Each git and pip command runs under a supervisor with an
allowlisted environment; if the creating process is killed, the command is stopped and the
temporary folder stays until the next creation removes it. Linux and macOS only.
"""
from __future__ import annotations

import contextlib
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile

REFERENCE_REVISION = 'a1cd9df1118fec80eba7463de237d1497d77e041'
REPOSITORY = 'https://github.com/lance-ying/coggym.git'
REQUIREMENTS = Path(__file__).with_name('coggym_requirements.txt')
# The files of the checkout that the worker reads.
_SUPERVISOR = Path(__file__).with_name('_supervise.py')
# The record inside a temporary folder that this code made it.
_CREATION_RECORD = '.brainscore-creation.json'
_REQUIRED = ('evaluation/__init__.py', 'evaluation/providers.py', 'evaluation/cli.py',
             'evaluation/analyze.py', 'evaluation/public_manifest.json',
             'evaluation/trial_selection_map.json', 'EML')
# Seconds one git or pip command may take (a stated bound, not a measured budget).
_COMMAND_TIMEOUT_S = 3600
# The only variables of this process that a git or pip command sees: the user, the locale,
# the temporary folder, certificates, proxies and the package index. Nothing that can move
# pip's destination (PIP_TARGET, PIP_PREFIX, PIP_ROOT, PIP_USER, a pip configuration file) or
# git's repository (GIT_DIR, GIT_WORK_TREE) passes.
_PASSED_VARIABLES = (
    'PATH', 'HOME', 'USER', 'LOGNAME', 'LANG', 'LC_ALL', 'LC_CTYPE', 'TMPDIR',
    'SSL_CERT_FILE', 'SSL_CERT_DIR', 'REQUESTS_CA_BUNDLE', 'CURL_CA_BUNDLE',
    'HTTP_PROXY', 'HTTPS_PROXY', 'NO_PROXY', 'http_proxy', 'https_proxy', 'no_proxy',
    'PIP_INDEX_URL', 'PIP_EXTRA_INDEX_URL', 'PIP_TRUSTED_HOST', 'PIP_CERT', 'PIP_CLIENT_CERT',
    'PIP_PROXY', 'PIP_TIMEOUT', 'PIP_RETRIES')


def environment_python(folder: Path) -> Path:
    """The Python of the folder's CogGym environment."""
    return Path(folder) / 'env' / 'bin' / 'python'


def default_folder() -> Path:
    """``<root>/<commit>-<requirements hash>``; the root is ``BRAINSCORE_COGGYM_FOLDER`` or
    ``~/.brain-score/coggym``."""
    root = os.environ.get('BRAINSCORE_COGGYM_FOLDER') or str(Path.home() / '.brain-score' / 'coggym')
    return Path(root).expanduser() / f'{REFERENCE_REVISION}-{_requirements_hash()[:12]}'


def ready_folder(identifier: str, folder: Path | None = None) -> Path:
    """The ready CogGym folder for ``identifier`` (a benchmark identifier, for the messages):
    ``folder``, or the default one; made first when it is absent and installs are allowed."""
    from brainscore import LoadError
    from brainscore_core.plugin_management.import_plugin import installation_preference
    if not REQUIREMENTS.is_file():
        raise LoadError(f'{identifier}: this Brain-Score installation lacks {REQUIREMENTS}, the pinned '
                        f"packages of CogGym's environment (a packaging fault). Next step: install "
                        f'Brain-Score from a source that ships it.')
    folder = Path(folder) if folder is not None else default_folder()
    if not folder.exists():
        if installation_preference() == 'no':
            raise LoadError(
                f"{identifier} needs CogGym's folder {folder} (CogGym at commit "
                f"{REFERENCE_REVISION} and CogGym's own environment), and it is not there. "
                f"BS_INSTALL_DEPENDENCIES=no forbids making it. Next step: load the benchmark "
                f"once with BS_INSTALL_DEPENDENCIES unset (or 'yes'); Brain-Score then makes the "
                f"folder (a clone and an install into that folder only; the scoring environment "
                f"is not changed).")
        try:
            with _creation_lock(folder):
                if not folder.exists():
                    _make(identifier, folder)
        except OSError as error:
            # The lock, the folder's parent or the temporary folder could not be made: nothing of
            # this attempt exists to remove.
            raise LoadError(
                f"{identifier}: Brain-Score could not make CogGym's folder {folder}: "
                f"{type(error).__name__}: {error}. Next step: make {folder.parent} writable, or "
                f"put CogGym's folders elsewhere (BRAINSCORE_COGGYM_FOLDER, or folder=).") from error
    _check_ready(identifier, folder)
    return folder


def _check_ready(identifier, folder):
    from brainscore import LoadError
    try:
        marker = json.loads((folder / 'ready.json').read_text())
    except (OSError, ValueError):
        marker = None
    expected = {'revision': REFERENCE_REVISION, 'requirements_sha256': _requirements_hash()}
    complete = (isinstance(marker, dict) and {key: marker.get(key) for key in expected} == expected
                and isinstance(marker.get('freeze'), list)
                and all(isinstance(line, str) for line in marker['freeze']))
    if not complete or not environment_python(folder).exists():
        raise LoadError(
            f'{identifier}: {folder} exists but is not a ready CogGym folder for commit '
            f'{REFERENCE_REVISION} and the pinned requirements (no matching ready.json, or no '
            f'environment). Brain-Score does not change a folder it did not finish. Next step: '
            f'remove it yourself, or use another place (BRAINSCORE_COGGYM_FOLDER, or folder=).')


def _make(identifier, folder):
    from brainscore import LoadError
    folder.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(tempfile.mkdtemp(prefix=f'.{folder.name}.partial-', dir=folder.parent))
    try:
        # The record that this code made this temporary folder: only such a folder is ever
        # removed by a later creation (a name alone proves nothing).
        (partial / _CREATION_RECORD).write_text(json.dumps({
            'folder': str(folder), 'made_by': 'brainscore CogGym folder creation', 'pid': os.getpid()}))
        checkout, python = partial / 'checkout', environment_python(partial)
        # No checkout of the default branch, and no local hooks: only the pinned commit is checked out.
        _run(['git', 'clone', '--quiet', '--no-checkout', REPOSITORY, str(checkout)])
        _run(['git', '-c', 'core.hooksPath=/dev/null', '-C', str(checkout), 'checkout', '--quiet',
              REFERENCE_REVISION])
        head = _run(['git', '-C', str(checkout), 'rev-parse', 'HEAD']).strip()
        if head != REFERENCE_REVISION:
            raise ValueError(f'the clone is at {head}, not at {REFERENCE_REVISION}')
        missing = [name for name in _REQUIRED if not (checkout / name).exists()]
        if missing:
            raise ValueError(f'the checkout lacks {missing}')
        _run([sys.executable, '-m', 'venv', str(partial / 'env')])
        # Every package at its pinned version, nothing resolved beyond the list; then a check.
        # Into this environment only (pip refuses to run outside a virtual environment).
        _run([str(python), '-m', 'pip', 'install', '--require-virtualenv', '--no-deps', '--no-input',
              '--disable-pip-version-check', '-r', str(REQUIREMENTS)])
        _run([str(python), '-m', 'pip', 'check', '--disable-pip-version-check'])
        freeze = _run([str(python), '-m', 'pip', 'freeze', '--all', '--disable-pip-version-check'])
        (partial / 'ready.json').write_text(json.dumps({
            'revision': REFERENCE_REVISION, 'requirements_sha256': _requirements_hash(),
            'freeze': sorted(line for line in freeze.splitlines() if line.strip())}, indent=2) + '\n')
        # One step that never replaces: a folder that appeared meanwhile belongs to someone else.
        _rename_no_replace(partial, folder)
        (folder / _CREATION_RECORD).unlink()          # published: the record has done its job
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        shutil.rmtree(partial, ignore_errors=True)
        detail = getattr(error, 'stderr', None) or ''
        if isinstance(error, OSError) and error.errno in (errno.EEXIST, errno.ENOTEMPTY):
            detail = (f'{folder} appeared while Brain-Score made the folder; it belongs to another '
                      f'program or person and is not changed.')
        raise LoadError(
            f"{identifier}: Brain-Score could not make CogGym's folder {folder}: "
            f"{type(error).__name__}: {error}" + (f'\n{detail.strip()[-2000:]}' if detail else '')
            + f"\nNothing was left at {folder}. Next step: check the network, the disk and the "
            f"permissions of {folder.parent}, and load the benchmark again.") from error


def _command_environment() -> dict:
    """The environment of a git or pip command: the passed variables only; no pip
    configuration file is read, no user site is used, and git asks nothing."""
    environment = {name: os.environ[name] for name in _PASSED_VARIABLES if name in os.environ}
    environment.update(PIP_CONFIG_FILE=os.devnull, PIP_NO_INPUT='1', PIP_DISABLE_PIP_VERSION_CHECK='1',
                       PYTHONNOUSERSITE='1', GIT_TERMINAL_PROMPT='0', GIT_CONFIG_NOSYSTEM='1')
    return environment


def _rename_no_replace(source: Path, destination: Path) -> None:
    """Rename ``source`` to ``destination`` in one step that fails if ``destination`` exists
    (Linux ``renameat2`` with ``RENAME_NOREPLACE``, macOS ``renamex_np`` with ``RENAME_EXCL``)."""
    import ctypes
    libc = ctypes.CDLL(None, use_errno=True)
    names = os.fsencode(str(source)), os.fsencode(str(destination))
    if sys.platform == 'darwin':
        result = libc.renamex_np(*names, ctypes.c_uint(0x4))                 # RENAME_EXCL
    elif sys.platform.startswith('linux') and hasattr(libc, 'renameat2'):
        result = libc.renameat2(-100, names[0], -100, names[1], ctypes.c_uint(1))   # AT_FDCWD, NOREPLACE
    else:
        raise OSError(errno.ENOTSUP, 'no rename that refuses to replace on this platform', str(destination))
    if result != 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), str(destination))


def _run(command) -> str:
    """Run one git or pip command under the supervisor and return its output.

    The command and its descendants are killed when it takes longer than the time limit, when
    this process stops waiting, and when this process dies (the supervisor sees its pipe close).
    """
    from brainscore.harnesses._child_process import ChildProcess
    with tempfile.TemporaryDirectory(prefix='coggym-command-') as logs:
        with ChildProcess([sys.executable, '-I', str(_SUPERVISOR), *command],
                          env=_command_environment(), log_directory=Path(logs)) as child:
            try:
                child.wait(_COMMAND_TIMEOUT_S)
            except subprocess.CalledProcessError as error:
                raise subprocess.CalledProcessError(error.returncode, command,
                                                    stderr=child.log_tail()) from None
        return (Path(logs) / 'stdout.log').read_text()


@contextlib.contextmanager
def _creation_lock(folder):
    """One creation of ``folder`` at a time, across threads and processes: a lock file next to it
    (``.<name>.lock``, kept), so every creator shares it whatever its temporary folder. While the
    lock is held no other creation runs, so a temporary folder that an earlier attempt of this
    code made (``_created_here``) belongs to a creator that died, and it is removed. Any other
    sibling, whatever its name, is left alone."""
    import fcntl
    folder.parent.mkdir(parents=True, exist_ok=True)
    with open(folder.parent / f'.{folder.name}.lock', 'w') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            for abandoned in folder.parent.glob(f'.{folder.name}.partial-*'):
                if _created_here(abandoned, folder):
                    shutil.rmtree(abandoned, ignore_errors=True)
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _created_here(candidate: Path, folder: Path) -> bool:
    """Whether ``candidate`` is a temporary folder this code made for ``folder``: a folder (not a
    link) that holds a creation record (a regular file, not a link) naming ``folder``."""
    try:
        record = candidate / _CREATION_RECORD
        if not (stat.S_ISDIR(os.lstat(candidate).st_mode) and stat.S_ISREG(os.lstat(record).st_mode)):
            return False
        return json.loads(record.read_text()).get('folder') == str(folder)
    except (OSError, ValueError, AttributeError):
        return False


def _requirements_hash() -> str:
    return hashlib.sha256(REQUIREMENTS.read_bytes()).hexdigest()
