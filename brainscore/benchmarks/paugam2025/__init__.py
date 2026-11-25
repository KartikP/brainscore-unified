from functools import partial

from brainscore import benchmark_registry


def _load(**configuration):
    from .benchmark import Paugam2025Mario
    return Paugam2025Mario(**configuration)


def _load_buttons(**configuration):
    from .benchmark import Paugam2025MarioButtons
    return Paugam2025MarioButtons(**configuration)


# The five public participants pooled; the participant's own play on vision:video (with its middle frame on
# vision:image and its sound on audio), neural:cortex requested. See benchmark.py.
benchmark_registry['Paugam2025-fMRI-mario-cortex'] = partial(_load, identifier='Paugam2025-fMRI-mario-cortex')
# The behavior form: the buttons the participant held in each TR, on the same rows and the same test runs.
benchmark_registry['Paugam2025-behavior-mario-buttons'] = partial(
    _load_buttons, identifier='Paugam2025-behavior-mario-buttons')
