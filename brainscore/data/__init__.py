"""Unified-owned data plugins.

Every plugin folder here (a sub-package whose name does not start with ``_``) is
imported when the package loads, and only registers lightweight factories, local
assets and its ``DataBuilder``; ``local``, ``packaging`` and ``__main__`` are modules,
not plugins. Actual data I/O stays deferred until ``brainscore.load_dataset``,
``brainscore.load_stimulus_set`` or ``package_data`` is called.
"""

from brainscore import _import_plugins
from brainscore.data.packaging import DataBuilder, package_data

_import_plugins(__name__)    # every plugin folder here registers by its presence
