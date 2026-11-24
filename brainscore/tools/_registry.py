"""One table of lazy import paths and one loader for experiment and interpretation tools.

A tool that ships in Brain-Score is one line in ``_TOOLS``: module, class, and whether
``artifact=`` loads a saved tool. The module is imported when ``load_tool`` asks for it.
A tool written outside Brain-Score is added with ``register_tool(identifier, factory)``;
``load_tool`` calls the factory. The two cases are the two branches of ``load_tool``.
"""
from importlib import import_module
from typing import Any, Callable


# One line per built-in tool: (module, class, whether `artifact=` loads a fitted tool).
# The module is imported when the tool is loaded, not before.
_TOOLS: dict[str, Any] = {      # a path tuple here; a callable for a registered tool
    'ablate': ('brainscore.experiments', 'Ablate', False),
    'contrastive-steering': ('brainscore.tools.contrastive_steering', 'ContrastiveSteering', True),
    'linear-probe': ('brainscore.tools.linear_probe', 'LinearProbe', True),
    'leace': ('brainscore.tools.leace', 'LEACE', True),
    'patch-activity': ('brainscore.experiments', 'PatchActivity', False),
    'record-activity': ('brainscore.experiments', 'RecordActivity', False),
    'record-percept': ('brainscore.tools.percept_window', 'PerceptWindow', False),
    'record-io': ('brainscore.experiments', 'RecordInputsOutputs', False),
    'record-reasoning': ('brainscore.experiments', 'RecordReasoning', False),
    'scale-activity': ('brainscore.experiments', 'ScaleActivity', False),
}


def list_tools() -> tuple[str, ...]:
    """List registered tool identifiers without importing their dependencies."""
    return tuple(sorted(_TOOLS))


def register_tool(identifier: str, factory: Callable[..., Any], *, replace: bool = False) -> None:
    """Register a factory; replacement must be explicitly requested."""
    if not isinstance(identifier, str) or not identifier.strip() or not callable(factory):
        raise ValueError('A tool needs a nonempty identifier and a callable factory')
    if identifier in _TOOLS and not replace:
        raise ValueError(f'Tool {identifier!r} is already registered')
    _TOOLS[identifier] = factory     # a callable: an external tool, built by calling it


def load_tool(identifier: str, **configuration: Any) -> Any:
    """Construct a fresh tool, or load its fitted artifact when supported."""
    try:
        entry = _TOOLS[identifier]
    except KeyError:
        raise KeyError(f'Unknown tool {identifier!r}; available: {list_tools()}. A tool that ships '
                       f'in Brain-Score is one line in _TOOLS of brainscore/tools/_registry.py; a '
                       f'tool outside Brain-Score is added with register_tool(identifier, factory) '
                       f'before load_tool.') from None
    if callable(entry):
        return entry(**configuration)
    module, name, fitted = entry
    tool = getattr(import_module(module), name)
    if fitted and 'artifact' in configuration:
        return tool.load(configuration.pop('artifact'), **configuration)
    return tool(**configuration)
