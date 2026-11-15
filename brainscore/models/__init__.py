"""Unified Brain-Score model registrations.

Each plugin folder here (a sub-package whose name does not start with ``_``) adds
its entries to ``brainscore.model_registry``. Every one is imported when the package
loads; ``_downloads`` and ``_shared`` are helper modules, not plugins.
"""

from brainscore import _import_plugins

_import_plugins(__name__)    # every plugin folder here registers by its presence
