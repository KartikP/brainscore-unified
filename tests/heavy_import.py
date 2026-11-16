"""The failure message of the tests that keep ``import brainscore`` and the loaders light."""


def heavy_import_message(what: str, output: str) -> str:
    return (f'{what} imported a heavy library (the output below names it). A plugin package '
            'imports its heavy code inside its loader, not at import, as '
            'templates/new_model/__init__.py does (def _load(): from .model import get_model).\n'
            + output)
