"""Registers your stimuli and measurements so benchmarks can load them by identifier.

To install: copy this folder to ``brainscore/data/your_name/``, then add
``from . import your_name`` to ``brainscore/data/__init__.py``.

Then anywhere (typically inside a benchmark):

    from brainscore import load_stimulus_set, load_dataset
    stimuli  = load_stimulus_set('your-stimuli')
    measured = load_dataset('your-measurements')
"""
from brainscore import data_registry, stimulus_set_registry

from .data import (ASSEMBLY_ID, STIMULUS_SET_ID, load_assembly,
                   load_stimulus_set)

# The values are factories, not the data itself — nothing is read until something asks
# for it, so importing the package stays cheap even with a large dataset behind it.
stimulus_set_registry[STIMULUS_SET_ID] = load_stimulus_set
data_registry[ASSEMBLY_ID] = load_assembly
