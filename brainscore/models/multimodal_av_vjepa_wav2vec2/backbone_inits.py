"""Helper for randomizing already-loaded module weights in-place.

The pretrained V-JEPA v1 backbone has a custom architecture that we
load via ``_load_vjepa_v1_vitl`` (vendored backbone code). Constructing
from-config-only doesn't work cleanly there, so the cleanest null
control is to load the architecture from the trained checkpoint and
then re-initialize all parameters with the same dtype/shape but
random values from a deterministic seed.
"""
import torch


def randomize_module_in_place(module: torch.nn.Module, seed: int = 0) -> None:
    """Re-initialize every parameter of ``module`` in-place with random
    normal values (matching each tensor's existing shape / dtype /
    device). Deterministic via ``torch.manual_seed(seed)``.

    Approximates the distribution PyTorch uses by default for module
    construction without requiring per-layer init schemes — for null-
    control purposes the only requirement is "weights are statistically
    indistinguishable from random," which N(0, 0.02) satisfies for any
    transformer backbone.
    """
    torch.manual_seed(seed)
    with torch.no_grad():
        for p in module.parameters():
            p.data.normal_(mean=0.0, std=0.02)
        # Buffers (e.g. layernorm running stats) left as-is; they don't
        # encode learned content.
