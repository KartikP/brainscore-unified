"""Shared device selection for the extractor wrappers."""


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


def place_model(model, device=None):
    """The model as placed, and the device for its inputs.

    A model that something else already placed is left where it is: one with an
    ``hf_device_map``, or whose parameters are on ``meta`` or on more than one
    device (a submodule of a device-mapped model carries no map of its own).
    Moving it would fail ("Cannot copy out of meta tensor") or undo the plan.
    Its inputs go to the device of its token embedding (``embed_tokens``) when it
    has one, else to the device of its first parameter. Any other model is moved
    to ``select_device(device)``, which is also the device for its inputs.
    """
    devices = set()
    for parameter in model.parameters():
        devices.add(parameter.device)
        if parameter.device.type == 'meta' or len(devices) > 1:
            break
    placed = (bool(getattr(model, 'hf_device_map', None)) or len(devices) > 1
              or any(d.type == 'meta' for d in devices))
    if not placed:
        target = select_device(device)
        return model.to(target), target
    embedding = getattr(model, 'embed_tokens', None)
    return model, next((embedding if embedding is not None else model).parameters()).device
