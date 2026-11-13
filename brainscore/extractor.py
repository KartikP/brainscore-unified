"""The extractor protocol: what records activity for one input modality.

An extractor owns input preparation, the forward pass, hooks, caching and
assembly packaging. ``BrainScoreModel(extractor={modality: extractor})``
takes one per modality; the keys declare the modalities the subject takes in.
"Wrapper" is the name of the extractors Brain-Score ships (``TextWrapper``,
``PytorchWrapper``, ...). Nothing needs to subclass this protocol.
"""

from typing import Any, Protocol, Sequence


class Extractor(Protocol):
    """Callable ``(stimuli, layers) -> NeuroidAssembly``.

    ``identifier`` and ``backbone_id`` key the activation cache (registrations
    that share weights share ``backbone_id``). An extractor that keeps state
    between calls also defines ``reset()``.

    An extractor may also define ``layer_names()``, returning every path it
    can record, relative to the module it wraps. ``BrainScoreModel`` then
    rejects an unknown ``region_layer_map`` path or raw recording target
    before any forward pass; a ``':input'`` site is checked by its module
    path. An extractor without ``layer_names()`` skips this early check.

    ``record(layers, stimuli)`` records while another part (``reply``,
    ``likelihood``, ``answer``, ``policy``) runs the network on ``stimuli`` (None
    for a policy step): its ``take(row)`` gives ``{layer: assembly}`` for the
    first forward pass since the previous take, packaged on its own per layer by
    the functions of ``__call__``. ``row`` is the stimulus row of that pass (None
    for a policy step), for what the pass does not carry (a clip's duration).
    The recording equals what ``__call__`` gives for ``stimuli``, or is refused:
    before any pass when ``stimuli`` decide it, else before the pass. Every extractor in
    ``brainscore.model_helpers`` has it, so a region of such a model is
    recorded during any task. A callable without it (a plain function, or the
    legacy ``PytorchWrapper``) still records plain viewing; during a task it
    records through hooks on ``BrainScoreModel(model=...)`` when every layer of
    the region is a module of that network, and otherwise the pair is N/A
    before any compute (``brainscore._model.recording.records_while_answering``).
    """

    identifier: str

    def __call__(self, stimuli: Any, layers: Sequence[str]) -> Any:
        ...

    def record(self, layers: Sequence[str], stimuli: Any = None) -> Any:
        ...
