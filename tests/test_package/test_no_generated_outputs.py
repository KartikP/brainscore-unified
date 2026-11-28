"""The package holds code and the data its plugins read, not the outputs of a check (figures, documents)."""
from pathlib import Path

import pytest

import brainscore

pytestmark = pytest.mark.unit

OUTPUTS = {'.png', '.pdf', '.jpg', '.jpeg', '.svg', '.gif', '.mp4', '.avi'}


def test_no_figure_or_document_is_in_the_package():
    folder = Path(brainscore.__file__).parent
    found = sorted(str(path.relative_to(folder)) for path in folder.rglob('*') if path.suffix.lower() in OUTPUTS)
    assert not found, (f'move these outputs out of the package (a check script writes them to a folder it is '
                       f'given): {found}')
