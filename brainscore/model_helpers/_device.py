"""Shared device selection for the activations-model wrappers."""


def select_device():
    """Best available torch device: CUDA, else MPS (Apple Silicon), else CPU.

    Imports torch lazily so importing this module stays light.
    """
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
