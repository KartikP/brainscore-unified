"""Unified-owned data plugins.

Every plugin folder here (a sub-package whose name does not start with ``_``) is
imported when the package loads, and only registers lightweight factories or local
assets; ``local``, ``preparation`` and ``__main__`` are modules, not plugins. Actual data I/O
stays deferred until ``brainscore.load_dataset`` or
``brainscore.load_stimulus_set`` is called.
"""

from brainscore import _import_plugins

_import_plugins(__name__)    # every plugin folder here registers by its presence
