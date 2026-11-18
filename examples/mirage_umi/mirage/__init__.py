"""Explicitly configured MIRAGE integration; no weights load on import.

Importing this package loads nothing else: ``MirageSubject`` (``model``) and
``MirageRuntime`` (``runtime``) are loaded at their first use, so the plugin
discovery of ``import brainscore`` imports only this file.
"""
from importlib import import_module

_MODULE_OF = {'MirageSubject': '.model', 'MirageRuntime': '.runtime'}

__all__ = ['MirageRuntime', 'MirageSubject']


def __getattr__(name):
    if name not in _MODULE_OF:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    return getattr(import_module(_MODULE_OF[name], __name__), name)
