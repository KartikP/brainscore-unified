from functools import partial

from brainscore import benchmark_registry


def _load(**configuration):
    from .benchmark import Paugam2025Mario
    return Paugam2025Mario(**configuration)


# The five public participants pooled; the participant's own play on vision:video (with its middle frame on
# vision:image and its sound on audio), neural:cortex requested. See benchmark.py.
benchmark_registry['Paugam2025-fMRI-mario-cortex'] = partial(_load, identifier='Paugam2025-fMRI-mario-cortex')
