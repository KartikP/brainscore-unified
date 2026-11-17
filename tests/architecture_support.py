"""Shared helpers for the architecture enforcement checks (E3, E4, E5).

Each check reads source with ``ast`` and never imports the code it reads. A
check collects a set of findings and compares it with an allow-list of
today's known violations: a new finding fails, and an allow-list entry that
is no longer found fails too, so the allow-list can only shrink.
"""
import ast
from importlib.util import find_spec
from pathlib import Path


def package_root(package):
    """Folder of a top-level package, found without importing it."""
    spec = find_spec(package)
    assert spec is not None and spec.submodule_search_locations, package
    return Path(list(spec.submodule_search_locations)[0])


def python_files(root, folders=('',)):
    """(path relative to root, source) for every .py file below the folders."""
    root = Path(root)
    for folder in folders:
        target = root / folder
        files = [target] if target.is_file() else sorted(target.rglob('*.py'))
        for path in files:
            yield path.relative_to(root).as_posix(), path.read_text(encoding='utf-8')


def imports(source, module_name):
    """(module, imported names) for each import; relative imports resolved.

    ``module_name`` is the dotted name of the file that holds ``source``
    (``brainscore.tools.leace``, ``brainscore.tools.__init__``), as given by
    ``module_name``.
    """
    tree = ast.parse(source) if isinstance(source, str) else source
    parts = module_name.split('.')
    package = parts[:-1]
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend((alias.name, []) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package[:len(package) - node.level + 1]
                module = '.'.join(base + ([node.module] if node.module else []))
            else:
                module = node.module or ''
            found.append((module, [alias.name for alias in node.names]))
    return found


def module_name(package, relative_path):
    """Dotted module name for a file path relative to a package folder."""
    name = relative_path[:-len('.py')].replace('/', '.')
    return f'{package}.{name}'


def is_private(name):
    return name.startswith('_') and not name.startswith('__')


def compare(found, allowed, what):
    """Fail on a finding that is not allowed and on an allowed entry not found."""
    new = sorted(set(found) - set(allowed))
    stale = sorted(set(allowed) - set(found))
    assert not new, f'new {what}: {new}'
    assert not stale, f'allow-list entries that no longer violate; remove them: {stale}'
