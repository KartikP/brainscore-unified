"""Shared device selection for the activations-model wrappers."""


def select_device(device=None):
    """``device`` if given (``'cpu'``, ``'cuda'``, ``'mps'``), else the best available:
    CUDA, else MPS (Apple Silicon), else CPU.

    Imports torch lazily so importing this module stays light.
    """
    import torch
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
