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

    A model that something else already placed is left where it is: one with a
    nonempty ``hf_device_map``, or whose parameters are on ``meta`` or on more
    than one device (a submodule of a device-mapped model carries no map of its
    own). Moving it would fail ("Cannot copy out of meta tensor") or undo the
    plan. Its inputs go to ``input_device(model, device)``: a ``device`` given
    here is used for the inputs and does not move the model.

    Any other model is moved to ``select_device(device)``, which is also the
    device for its inputs, exactly as before this rule (so its cache key is
    unchanged). A stated ``meta`` device is refused (``stated_device``).
    """
    if device is not None:
        stated_device(device)
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
    return model, input_device(model, device)


def input_device(model, device=None):
    """Where the inputs of ``model`` go, without moving it, in one form (``canonical_device``).

    ``device`` when given. Else the one device of the parameters of the module
    that takes the input: the input embeddings that the model names
    (``get_input_embeddings()``, the loading library's own statement), else the
    whole model. Parameters on ``meta`` (offloaded) are not a place for inputs.
    Refuses when no single device remains: only ``meta`` parameters, or
    parameters on several devices with no input module named.
    """
    import torch
    if device is not None:
        return canonical_device(stated_device(device))
    try:
        entry = model.get_input_embeddings()
    except (AttributeError, NotImplementedError):
        entry = None
    owner = entry if isinstance(entry, torch.nn.Module) else model
    devices = {p.device for p in owner.parameters() if p.device.type != 'meta'}
    if len(devices) != 1:
        raise ValueError(
            f'Cannot tell where the inputs of {type(model).__name__} go: its input parameters are on '
            f'{sorted(map(str, devices)) or "meta only"}. State the input device: an extractor wrapper '
            f'takes device=... (the model is not moved); a reply, likelihood or clip pass takes the '
            f'device of the module that the model names with get_input_embeddings(), so name one '
            f'with parameters on one device.')
    return canonical_device(devices.pop())


def stated_device(device):
    """``device`` as a torch device; refuses ``meta``, which holds no values."""
    import torch
    device = torch.device(device)
    if device.type == 'meta':
        raise ValueError("The device 'meta' holds no values; state the device where the inputs go.")
    return device


def canonical_device(device):
    """``device`` in the form torch gives a tensor's device: ``cpu`` without an index,
    any other type with one (``cuda`` is the current CUDA device, ``mps`` is ``mps:0``)."""
    import torch
    device = torch.device(device)
    if device.type == 'cpu':
        return torch.device('cpu')
    if device.index is None:
        return torch.device(device.type, torch.cuda.current_device() if device.type == 'cuda' else 0)
    return device
