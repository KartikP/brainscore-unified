"""LeBel 2023 story-listening fMRI (subject UTS03), whole cortex.

Registered lazily: the factories below do no I/O until called, because the
source pickle takes minutes to read and holds several GB.
"""

from brainscore import data_registry, stimulus_set_registry

from .data import IDENTIFIER, load

data_registry[IDENTIFIER] = lambda: load()[1]
stimulus_set_registry[IDENTIFIER] = lambda: load()[0]
