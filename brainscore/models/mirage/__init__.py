"""Explicitly configured MIRAGE integration; no weights load on import."""

from .model import MirageSubject
from .runtime import MirageRuntime

__all__ = ['MirageRuntime', 'MirageSubject']
