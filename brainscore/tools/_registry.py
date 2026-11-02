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


_FACTORIES: dict[str, Callable[..., Any]] = {
    'ablate': _ablate,
    'linear-probe': _linear_probe,
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
