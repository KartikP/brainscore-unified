"""Explicit, lazy factories for experiment and interpretation tools."""
from typing import Any, Callable


def _linear_probe(**configuration: Any) -> Any:
    from .linear_probe import LinearProbe
    if 'artifact' in configuration:
        artifact = configuration.pop('artifact')
        return LinearProbe.load(artifact, **configuration)
    return LinearProbe(**configuration)


def _ablate(**configuration: Any) -> Any:
    from brainscore.experiments import Ablate
    return Ablate(**configuration)


def _record_io(**configuration: Any) -> Any:
    from brainscore.experiments import RecordInputsOutputs
    return RecordInputsOutputs(**configuration)


def _record_activity(**configuration: Any) -> Any:
    from brainscore.experiments import RecordActivity
    return RecordActivity(**configuration)


def _scale_activity(**configuration: Any) -> Any:
    from brainscore.experiments import ScaleActivity
    return ScaleActivity(**configuration)


def _patch_activity(**configuration: Any) -> Any:
    from brainscore.experiments import PatchActivity
    return PatchActivity(**configuration)


_FACTORIES: dict[str, Callable[..., Any]] = {
    'ablate': _ablate,
    'linear-probe': _linear_probe,
    'patch-activity': _patch_activity,
    'record-activity': _record_activity,
    'record-io': _record_io,
    'scale-activity': _scale_activity,
}


def list_tools() -> tuple[str, ...]:
    """List registered tool identifiers without importing their dependencies."""
    return tuple(sorted(_FACTORIES))


def register_tool(identifier: str, factory: Callable[..., Any], *, replace: bool = False) -> None:
    """Register a factory; replacement must be explicitly requested."""
    if not isinstance(identifier, str) or not identifier.strip() or not callable(factory):
        raise ValueError('A tool needs a nonempty identifier and a callable factory')
    if identifier in _FACTORIES and not replace:
        raise ValueError(f'Tool {identifier!r} is already registered')
    _FACTORIES[identifier] = factory


def load_tool(identifier: str, **configuration: Any) -> Any:
    """Construct a fresh tool, or load its fitted artifact when supported."""
    try:
        factory = _FACTORIES[identifier]
    except KeyError:
        raise KeyError(f'Unknown tool {identifier!r}; available: {list_tools()}') from None
    return factory(**configuration)
