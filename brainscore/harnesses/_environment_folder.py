"""The folder Brain-Score makes for an outside experiment library: the library's checkout at a
pinned commit and the library's own environment. One ``Library`` describes the library.

``<folder>/checkout`` is the library's repository at the pinned commit; ``<folder>/env`` is a
virtual environment with the library's pinned packages only (its requirements file,
installed with ``--no-deps`` and checked with ``pip check``); ``<folder>/ready.json`` says the
folder is complete (the commit, the requirements' hash, the Python version and the
environment's ``pip freeze``).
Nothing is installed into the scoring environment.

The default place is ``~/.brain-score/environments/<name>/<commit>-<hash>`` (the root of every
library's folders can be moved with ``BRAINSCORE_ENVIRONMENTS_HOME``); the hash covers the
requirements and the Python version, so a changed pin or Python gets a new folder. The folder is
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
import dataclasses
import errno
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import tempfile

_SUPERVISOR = Path(__file__).with_name('_supervise.py')
# The record inside a temporary folder that this code made it.
_CREATION_RECORD = '.brainscore-creation.json'
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


@dataclasses.dataclass(frozen=True)
class Library:
    """An outside experiment library: what Brain-Score needs to make its folder."""
    name: str                       # one plain word: the folder's parent and the messages
    repository: str                 # what ``git clone`` takes
    commit: str                     # the full commit (40 hex characters)
    requirements: Path              # the pinned packages, one ``name==version`` per line
    required_files: tuple = ()      # paths in the checkout that must exist
    python: str | None = None       # the Python 3 that makes the environment; None: this one

    def __post_init__(self) -> None:
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', self.name):
            raise ValueError(f'a library name is one plain word, not {self.name!r}')
        if not re.fullmatch(r'[0-9a-f]{40}', self.commit):
            raise ValueError(f'{self.name}: the commit must be a full commit (40 hex characters), '
                             f'not {self.commit!r}')


def environment_python(folder: Path) -> Path:
    """The Python of the folder's own environment."""
    return Path(folder) / 'env' / 'bin' / 'python'


def default_folder(library: Library, version: str | None = None) -> Path:
    """``<root>/<name>/<commit>-<hash>``; the root is ``BRAINSCORE_ENVIRONMENTS_HOME`` or
    ``~/.brain-score/environments``. A root that is empty or not absolute is refused. The hash
    covers the folder's whole identity (``_identity``), with ``version`` the Python version
    (queried when not given), so two names that differ only by case get two folders."""
    value = os.environ.get('BRAINSCORE_ENVIRONMENTS_HOME', '~/.brain-score/environments')
    root = Path(value).expanduser()
    if not value or not root.is_absolute():
        raise ValueError(f'BRAINSCORE_ENVIRONMENTS_HOME must name an absolute folder, not {value!r}. '
                         f'Next step: set it to an absolute path, or unset it.')
    identity = _identity(library, version or python_version(library))
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return root / library.name / f'{library.commit}-{digest[:12]}'


def _identity(library: Library, version: str) -> dict:
    """What a folder is for: the library, its repository and commit, the hash of its
    requirements and the Python version of its environment. ``ready.json`` records it, and
    every load checks it."""
    return {'library': library.name, 'repository': library.repository, 'revision': library.commit,
            'requirements_sha256': _requirements_hash(library), 'python': version}


def python_version(library: Library) -> str:
    """The version (``3.x.y``) of the Python that makes the library's environment; a program
    that is not found or is not Python 3 is refused (``ValueError``). Nothing is made here.
    This Python (no ``python`` named) answers in this process: a build starts no program."""
    if library.python is None:
        return '.'.join(map(str, sys.version_info[:3]))
    return _reported_version(library.python, library.name)


def _reported_version(python, name: str) -> str:
    try:
        # Under the supervisor, as every git and pip command: a helper the program starts is
        # stopped with its process group.
        version = _run([python, '-c', 'import sys; print(*sys.version_info[:3], sep=".")'])[:100].strip()
    except subprocess.CalledProcessError:
        version = ''
    except subprocess.SubprocessError as error:
        version = f'{type(error).__name__}: {error}'
    if not re.fullmatch(r'3\.\d+\.\d+', version):
        raise ValueError(
            f"{name}'s environment is made with the Python program {python}, and it is not "
            f"found or is not Python 3 ({version or 'no version'}). Next step: install Python 3 "
            f"and name it as the python of the library's description; Brain-Score installs no "
            f"Python.")
    return version


def ready_folder(library: Library, identifier: str, folder: Path | None = None) -> Path:
    """The ready folder of ``library`` for ``identifier`` (a benchmark identifier, for the
    messages): ``folder``, or the default one; made first when it is absent and installs are
    allowed."""
    from brainscore import LoadError
    from brainscore_core.plugin_management.import_plugin import installation_preference
    name = library.name
    if not library.requirements.is_file():
        raise LoadError(f'{identifier}: this Brain-Score installation lacks {library.requirements}, '
                        f"the pinned packages of {name}'s environment (a packaging fault). Next step: "
                        f'install Brain-Score from a source that ships it.')
    try:
        version = python_version(library)        # once for each load, also for folder=
        folder = Path(folder) if folder is not None else default_folder(library, version)
    except ValueError as error:
        raise LoadError(f'{identifier}: {error}') from None
    if not folder.exists():
        if installation_preference() == 'no':
            raise LoadError(
                f"{identifier} needs {name}'s folder {folder} ({name} at commit "
                f"{library.commit} and {name}'s own environment), and it is not there. "
                f"BS_INSTALL_DEPENDENCIES=no forbids making it. Next step: load the benchmark "
                f"once with BS_INSTALL_DEPENDENCIES unset (or 'yes'); Brain-Score then makes the "
                f"folder (a clone and an install into that folder only; the scoring environment "
                f"is not changed).")
        try:
            with _creation_lock(folder):
                if not folder.exists():
                    _make(library, identifier, folder, version)
        except OSError as error:
            # The lock, the folder's parent or the temporary folder could not be made: nothing of
            # this attempt exists to remove.
            raise LoadError(
                f"{identifier}: Brain-Score could not make {name}'s folder {folder}: "
                f"{type(error).__name__}: {error}. Next step: make {folder.parent} writable, or "
                f"put the folders elsewhere (BRAINSCORE_ENVIRONMENTS_HOME, or folder=).") from error
    _check_ready(library, identifier, folder, version)
    return folder


def _check_ready(library, identifier, folder, version):
    from brainscore import LoadError
    try:
        marker = json.loads((folder / 'ready.json').read_text())
    except (OSError, ValueError):
        marker = None
    expected = _identity(library, version)
    recorded = {key: marker.get(key) for key in expected} if isinstance(marker, dict) else None
    if (recorded is not None and recorded != expected and isinstance(marker.get('freeze'), list)
            and environment_python(folder).exists()):
        raise LoadError(
            f'{identifier}: {folder} is a folder made for {recorded}, and this load needs '
            f'{expected}. Brain-Score does not use or change a folder made for another library or '
            f'Python. Next step: use the default place (no folder=), or another folder.')
    complete = (recorded == expected and isinstance(marker.get('freeze'), list)
                and all(isinstance(line, str) for line in marker['freeze']))
    if not complete or not environment_python(folder).exists():
        raise LoadError(
            f'{identifier}: {folder} exists but is not a ready {library.name} folder for commit '
            f'{library.commit} and the pinned requirements (no matching ready.json, or no '
            f'environment). Brain-Score does not change a folder it did not finish. Next step: '
            f'remove it yourself, or use another place (BRAINSCORE_ENVIRONMENTS_HOME, or folder=).')


def _make(library, identifier, folder, version):
    from brainscore import LoadError
    folder.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(tempfile.mkdtemp(prefix=f'.{folder.name}.partial-', dir=folder.parent))
    try:
        # The record that this code made this temporary folder: only such a folder is ever
        # removed by a later creation (a name alone proves nothing).
        (partial / _CREATION_RECORD).write_text(json.dumps({
            'folder': str(folder), 'made_by': f'brainscore {library.name} folder creation',
            'pid': os.getpid()}))
        checkout, python = partial / 'checkout', environment_python(partial)
        # No checkout of the default branch, and no local hooks: only the pinned commit is checked out.
        _run(['git', 'clone', '--quiet', '--no-checkout', library.repository, str(checkout)])
        _run(['git', '-c', 'core.hooksPath=/dev/null', '-C', str(checkout), 'checkout', '--quiet',
              library.commit])
        head = _run(['git', '-C', str(checkout), 'rev-parse', 'HEAD']).strip()
        if head != library.commit:
            raise ValueError(f'the clone is at {head}, not at {library.commit}')
        missing = [name for name in library.required_files if not (checkout / name).exists()]
        if missing:
            raise ValueError(f'the checkout lacks {missing}')
        _run([library.python or sys.executable, '-m', 'venv', str(partial / 'env')])
        # The environment must run the Python that was asked for, before anything is installed.
        made = _reported_version(str(python), library.name)
        if made != version:
            raise ValueError(f'the environment runs Python {made}, not {version}')
        # Every package at its pinned version, nothing resolved beyond the list; then a check.
        # Into this environment only (pip refuses to run outside a virtual environment).
        _run([str(python), '-m', 'pip', 'install', '--require-virtualenv', '--no-deps', '--no-input',
              '--disable-pip-version-check', '-r', str(library.requirements)])
        _run([str(python), '-m', 'pip', 'check', '--disable-pip-version-check'])
        freeze = _run([str(python), '-m', 'pip', 'freeze', '--all', '--disable-pip-version-check'])
        (partial / 'ready.json').write_text(json.dumps({
            **_identity(library, version),
            'freeze': sorted(line for line in freeze.splitlines() if line.strip())}, indent=2) + '\n')
        # One step that never replaces: a folder that appeared meanwhile belongs to someone else.
        _rename_no_replace(partial, folder)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        shutil.rmtree(partial, ignore_errors=True)
        detail = getattr(error, 'stderr', None) or ''
        if isinstance(error, OSError) and error.errno in (errno.EEXIST, errno.ENOTEMPTY):
            detail = (f'{folder} appeared while Brain-Score made the folder; it belongs to another '
                      f'program or person and is not changed.')
        raise LoadError(
            f"{identifier}: Brain-Score could not make {library.name}'s folder {folder}: "
            f"{type(error).__name__}: {error}" + (f'\n{detail.strip()[-2000:]}' if detail else '')
            + f"\nNothing was left at {folder}. Next step: check the network, the disk and the "
            f"permissions of {folder.parent}, and load the benchmark again.") from error
    # The rename above is the success boundary: the folder is complete and published. Removing
    # the creation record is best effort. A record left behind never leads to a deletion: a folder
    # with ready.json (a complete folder, also one renamed to a temporary name) is never reclaimed.
    with contextlib.suppress(OSError):
        (folder / _CREATION_RECORD).unlink()


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
    with tempfile.TemporaryDirectory(prefix='brainscore-command-') as logs:
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
    lock = folder.parent / f'.{folder.name}.lock'
    # Opened without truncation and without following a link; an existing regular file of this
    # name keeps its content (a lock never writes), and anything else is refused.
    handle = os.open(lock, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(handle).st_mode):
            raise OSError(errno.EEXIST, 'the lock is not a regular file', str(lock))
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            for abandoned in folder.parent.glob(f'.{folder.name}.partial-*'):
                if _created_here(abandoned, folder):
                    shutil.rmtree(abandoned, ignore_errors=True)
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
    finally:
        os.close(handle)


def _created_here(candidate: Path, folder: Path) -> bool:
    """Whether ``candidate`` is an unfinished temporary folder this code made for ``folder``: a
    folder (not a link) that holds a creation record (a regular file, not a link) naming
    ``folder`` and no ``ready.json``. A complete folder is never reclaimed, whatever its name."""
    try:
        record = candidate / _CREATION_RECORD
        if os.path.lexists(candidate / 'ready.json'):
            return False
        if not (stat.S_ISDIR(os.lstat(candidate).st_mode) and stat.S_ISREG(os.lstat(record).st_mode)):
            return False
        return json.loads(record.read_text()).get('folder') == str(folder)
    except (OSError, ValueError, AttributeError):
        return False


def _requirements_hash(library: Library) -> str:
    return hashlib.sha256(library.requirements.read_bytes()).hexdigest()
